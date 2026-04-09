"""
pipeline/moshi_runner.py
========================
Wraps Moshi inference with a safe **subclass-only** token-capture approach.
No modification to the original moshi-inference source code.

Key design
----------
* ``TokenCapturingInferenceState`` subclasses ``InferenceState`` and overrides
  ``run()`` to intercept the LM output tokens before Mimi decoding.
* Acoustic tokens shape: (T_out, 8) int64  — dep_q=8 codebooks @ 12.5 Hz
* These are exactly the tokens the bridge module was trained on.
* Moshi output audio (24 kHz WAV) is saved to a temp file and its path returned.

Usage
-----
    runner = MoshiTokenRunner(hf_repo="kyutai/moshiko-pytorch-bf16", device="cuda")
    audio_path, tokens = runner.run("input.wav", batch_index=0)
    # audio_path  : str  path to Moshi-generated WAV
    # tokens      : torch.Tensor (T, 8) int64 — acoustic codebook tokens
"""

import sys
import time
import tempfile
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import sphn

# ---------------------------------------------------------------------------
# Import from local moshi-inference package (installed with pip install -e)
# ---------------------------------------------------------------------------
import sys, os
_MOSHI_PKG = os.path.join(os.path.dirname(__file__), "..", "moshi-inference")
if _MOSHI_PKG not in sys.path:
    sys.path.insert(0, _MOSHI_PKG)

from moshi.models import LMGen, LMModel, MimiModel, loaders
from moshi.run_inference import InferenceState, get_condition_tensors, seed_all


# ---------------------------------------------------------------------------
# Subclass: captures raw LM acoustic tokens during generation
# ---------------------------------------------------------------------------

class TokenCapturingInferenceState(InferenceState):
    """
    Extends InferenceState to also collect the raw acoustic tokens
    produced by the Moshi LM *before* Mimi decoding.

    After ``run()`` completes, ``self.captured_acoustic_tokens`` contains a
    list of tensors, one per generated step:
        each tensor: (batch_size, dep_q)  — int64, 8 codebooks
    Concatenate along dim=0 to get (T_out, 8) for batch_index=0.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_acoustic_tokens: list[torch.Tensor] = []

    def run(self, in_pcms: torch.Tensor):
        """
        Identical to InferenceState.run() but also stores raw tokens[:, 1:]
        (the 8 acoustic codebooks) at every generation step.
        """
        # Reset capture buffer
        self.captured_acoustic_tokens = []

        out_pcms_per_item = [[] for _ in range(self.batch_size)]
        out_text_tokens_per_item = [[] for _ in range(self.batch_size)]
        eos_reached = [False] * self.batch_size
        need_eos_input = True

        self.printer.log(
            "info",
            "starting inference (TokenCapturing), "
            f"sampling: {self.lm_gen.use_sampling}, "
            f"audio temp: {self.lm_gen.temp}, "
            f"text temp: {self.lm_gen.temp_text}",
        )

        device = self.lm_gen.lm_model.device
        start_time = time.time()
        ntokens = 0
        first_frame = True

        if self.model_type == "stt":
            stt_config = self.checkpoint_info.stt_config
            pad_right = stt_config.get("audio_delay_seconds", 0.0)
            pad_left = stt_config.get("audio_silence_prefix_seconds", 0.0)
            pad_left = int(pad_left * 24000)
            pad_right = int((pad_right + 1.0) * 24000)
            in_pcms = torch.nn.functional.pad(
                in_pcms, (pad_left, pad_right), mode="constant"
            )

        chunks = deque(
            [
                chunk
                for chunk in in_pcms.split(self.frame_size, dim=2)
                if chunk.shape[-1] == self.frame_size
            ]
        )

        self.printer.print_header()

        while not all(eos_reached):
            if chunks:
                chunk = chunks.popleft()
                codes = self.mimi.encode(chunk)
            else:
                if self.model_type == "hibiki":
                    if need_eos_input:
                        need_eos_input = False
                        eos_value = self.mimi.cardinality
                        codes = torch.full(
                            (self.batch_size, self.mimi.num_codebooks, 1),
                            eos_value,
                            device=device,
                            dtype=torch.long,
                        )
                    else:
                        silence = torch.zeros(
                            (self.batch_size, self.mimi.channels, self.frame_size),
                            device=device,
                        )
                        codes = self.mimi.encode(silence)
                else:
                    break

            if first_frame:
                tokens = self.lm_gen.step(codes)
                if max(self.lm_gen.lm_model.delays) > 0:
                    assert tokens is None
                first_frame = False

            tokens = self.lm_gen.step(codes)
            if tokens is None:
                continue

            assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1

            # ── Capture acoustic tokens ────────────────────────────────────
            # tokens shape: (batch_size, dep_q+1, 1)
            # tokens[:, 0, 0]  → text token
            # tokens[:, 1:, 0] → acoustic tokens (dep_q=8 codebooks)
            if self.lm_gen.lm_model.dep_q > 0:
                # (batch_size, dep_q)  int64
                acoustic = tokens[:, 1:, 0].cpu()
                self.captured_acoustic_tokens.append(acoustic)

            if self.lm_gen.lm_model.dep_q > 0:
                out_pcm = self.mimi.decode(tokens[:, 1:]).cpu()
                for b, (one_text, one_pcm) in enumerate(
                    zip(tokens[:, 0].cpu(), out_pcm)
                ):
                    if eos_reached[b]:
                        continue
                    elif one_text.item() == self.text_tokenizer.eos_id():
                        if need_eos_input:
                            self.printer.log("warning", "EOS sampled too early.")
                        else:
                            eos_reached[b] = True
                    out_text_tokens_per_item[b].append(one_text)
                    out_pcms_per_item[b].append(one_pcm)
                    if b == 0:
                        if one_text.item() not in [0, 3]:
                            text = self.text_tokenizer.id_to_piece(one_text.item())
                            text = text.replace("▁", " ")
                            self.printer.print_token(text)
            else:
                one_text = tokens[0, 0].cpu()
                if one_text.item() not in [0, 3]:
                    text = self.text_tokenizer.id_to_piece(one_text.item())
                    text = text.replace("▁", " ")
                    self.printer.print_token(text)

            ntokens += 1

        dt = time.time() - start_time
        self.printer.log(
            "info",
            f"processed {ntokens} steps in {dt:.0f}s, {1000*dt/ntokens:.2f}ms/step",
        )

        if self.lm_gen.lm_model.dep_q > 0:
            out = [
                (torch.cat(one_texts, dim=0), torch.cat(one_pcms, dim=1))
                for one_texts, one_pcms in zip(
                    out_text_tokens_per_item, out_pcms_per_item
                )
            ]
            return out
        else:
            return []


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

class MoshiTokenRunner:
    """
    Loads Moshi + Mimi models and runs inference on a single audio file.

    Parameters
    ----------
    hf_repo        : HuggingFace repo (default: kyutai/moshiko-pytorch-bf16)
    moshi_weight   : optional local path to Moshi .safetensors
    mimi_weight    : optional local path to Mimi .safetensors
    tokenizer      : optional local path to text tokenizer
    device         : "cuda" or "cpu"
    dtype          : torch.bfloat16 (default) or torch.float16
    batch_size     : Moshi default is 8 parallel streams
    cfg_coef       : CFG coefficient (default 1.0)
    seed           : random seed for reproducibility
    """

    def __init__(
        self,
        hf_repo: str = "kyutai/moshiko-pytorch-bf16",
        moshi_weight: Optional[str] = None,
        mimi_weight: Optional[str] = None,
        tokenizer: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        batch_size: int = 8,
        cfg_coef: float = 1.0,
        seed: int = 4242,
    ):
        seed_all(seed)
        self.device = device
        self.batch_size = batch_size

        print("[MoshiTokenRunner] Loading checkpoint info...")
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
            hf_repo, moshi_weight, mimi_weight, tokenizer
        )

        print("[MoshiTokenRunner] Loading Mimi encoder...")
        self.mimi = checkpoint_info.get_mimi(device=device)

        print("[MoshiTokenRunner] Loading Moshi LM...")
        lm = checkpoint_info.get_moshi(device=device, dtype=dtype)
        if lm.dep_q == 0:
            self.batch_size = 1

        self.text_tokenizer = checkpoint_info.get_text_tokenizer()

        print("[MoshiTokenRunner] Initializing inference state...")
        self.state = TokenCapturingInferenceState(
            checkpoint_info=checkpoint_info,
            mimi=self.mimi,
            text_tokenizer=self.text_tokenizer,
            lm=lm,
            batch_size=self.batch_size,
            cfg_coef=cfg_coef,
            device=device,
            **checkpoint_info.lm_gen_config,
        )
        print("[MoshiTokenRunner] Ready.")

    @torch.no_grad()
    def run(
        self,
        audio_input_path: str,
        batch_index: int = 0,
        output_audio_path: Optional[str] = None,
    ) -> tuple[str, torch.Tensor]:
        """
        Run full Moshi inference on an audio file.

        Parameters
        ----------
        audio_input_path  : path to input WAV (any sample rate; auto-resampled to 24kHz)
        batch_index       : which batch item to return (0-indexed, default 0)
        output_audio_path : where to save Moshi's generated audio;
                            if None, a temp file is created automatically

        Returns
        -------
        (moshi_audio_path, acoustic_tokens)
            moshi_audio_path : str   — path to the saved 24kHz Moshi output WAV
            acoustic_tokens  : torch.Tensor (T, 8) int64 — raw LM acoustic tokens
                               These feed directly into the Bridge module.
        """
        print(f"[MoshiTokenRunner] Reading audio: {audio_input_path}")
        in_pcms, _ = sphn.read(audio_input_path, sample_rate=self.mimi.sample_rate)
        in_pcms = torch.from_numpy(in_pcms).to(device=self.device)
        in_pcms = in_pcms[None, 0:1].expand(self.batch_size, -1, -1)

        print("[MoshiTokenRunner] Running inference...")
        out_items = self.state.run(in_pcms)

        # ── Save Moshi output audio ────────────────────────────────────────
        if not out_items:
            raise RuntimeError(
                "Moshi produced no output. Check the model type (dep_q must be > 0)."
            )

        if batch_index >= len(out_items):
            raise IndexError(
                f"batch_index={batch_index} out of range (got {len(out_items)} items)."
            )

        _, out_pcm = out_items[batch_index]   # (1, T_samples)

        if output_audio_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_audio_path = tmp.name
            tmp.close()

        duration = out_pcm.shape[1] / self.mimi.sample_rate
        print(f"[MoshiTokenRunner] Writing audio ({duration:.1f}s) → {output_audio_path}")
        sphn.write_wav(
            output_audio_path,
            out_pcm[0].float().numpy(),
            sample_rate=self.mimi.sample_rate,
        )

        # ── Extract acoustic tokens for batch_index ────────────────────────
        # captured_acoustic_tokens: list of (batch_size, 8) tensors, one per step
        if not self.state.captured_acoustic_tokens:
            raise RuntimeError("No acoustic tokens were captured. Was dep_q > 0?")

        # Stack: (T_steps, batch_size, 8)
        all_tokens = torch.stack(self.state.captured_acoustic_tokens, dim=0)
        # Select batch item → (T_steps, 8)
        acoustic_tokens = all_tokens[:, batch_index, :]

        print(
            f"[MoshiTokenRunner] Captured acoustic tokens: {acoustic_tokens.shape} "
            f"(dtype={acoustic_tokens.dtype})"
        )

        return output_audio_path, acoustic_tokens
