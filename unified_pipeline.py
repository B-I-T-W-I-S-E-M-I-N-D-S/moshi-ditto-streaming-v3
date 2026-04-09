"""
unified_pipeline.py
===================
Top-level orchestrator for the Moshi + Bridge + Ditto unified pipeline.

Full pipeline:
    User Audio  ──► Moshi LM ──► Moshi audio WAV  (saved to disk)
                         │
                         ▼  acoustic tokens (T, 8) int64 @ 12.5 Hz
                    Bridge Module
                         │
                         ▼  HuBERT-like features (N, 1024) float32 @ 25 Hz
    User Image  ──────────────────► Ditto Diffusion ──► silent .mp4
                                                             │
                          Moshi WAV + silent .mp4 ──► FFmpeg ──► final .mp4

Usage (script)
--------------
    python unified_pipeline.py \\
        --audio_input     input.wav \\
        --image_input     portrait.jpg \\
        --output_path     final_output.mp4 \\
        --bridge_ckpt     checkpoints/bridge_best.pt \\
        --ditto_data_root ditto-inference/checkpoints/ditto_trt_Ampere_Plus \\
        --ditto_cfg_pkl   ditto-inference/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl

Usage (Python API)
------------------
    from unified_pipeline import run_pipeline

    run_pipeline(
        audio_input     = "input.wav",
        image_input     = "portrait.jpg",
        output_path     = "final_output.mp4",
        bridge_ckpt     = "checkpoints/bridge_best.pt",
        ditto_data_root = "ditto-inference/checkpoints/ditto_trt_Ampere_Plus",
        ditto_cfg_pkl   = "ditto-inference/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl",
    )
"""

import argparse
import os
import sys
import tempfile
import time
import torch

# ── Make sure the pipeline package is importable ──────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline.moshi_runner import MoshiTokenRunner
from pipeline.bridge_runner import BridgeRunner
from pipeline.ditto_runner import DittoRunner
from pipeline.merge_audio_video import merge_audio_into_video


# ─────────────────────────────────────────────────────────────────────────────
# Default path constants (all relative to the project root)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_BRIDGE_CKPT     = os.path.join(_PROJECT_ROOT, "checkpoints", "bridge_best.pt")
DEFAULT_BRIDGE_CONFIG   = os.path.join(_PROJECT_ROOT, "bridge_module", "config.yaml")
DEFAULT_DITTO_DATA_ROOT = os.path.join(_PROJECT_ROOT, "ditto-inference",
                                       "checkpoints", "ditto_trt_Ampere_Plus")
DEFAULT_DITTO_CFG_PKL   = os.path.join(_PROJECT_ROOT, "ditto-inference",
                                       "checkpoints", "ditto_cfg",
                                       "v0.4_hubert_cfg_trt.pkl")
DEFAULT_MOSHI_REPO      = "kyutai/moshiko-pytorch-bf16"
DEFAULT_BATCH_SIZE      = 8   # Moshi default; matches the trained model


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline function
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    audio_input: str,
    image_input: str,
    output_path: str,
    # --- Bridge ---
    bridge_ckpt: str         = DEFAULT_BRIDGE_CKPT,
    bridge_config: str       = DEFAULT_BRIDGE_CONFIG,
    # --- Ditto ---
    ditto_data_root: str     = DEFAULT_DITTO_DATA_ROOT,
    ditto_cfg_pkl: str       = DEFAULT_DITTO_CFG_PKL,
    # --- Moshi ---
    moshi_hf_repo: str       = DEFAULT_MOSHI_REPO,
    moshi_weight: str | None = None,
    mimi_weight: str | None  = None,
    tokenizer: str | None    = None,
    batch_size: int          = DEFAULT_BATCH_SIZE,
    batch_index: int         = 0,
    device: str              = "cuda",
    dtype: torch.dtype       = torch.bfloat16,
    # --- Optional intermediate file saving ---
    save_moshi_audio: str | None  = None,   # if set, copy Moshi WAV here
    save_bridge_features: str | None = None, # if set, save .npy here
    save_silent_video: str | None  = None,   # if set, keep silent .mp4 here
) -> str:
    """
    Run the full Moshi → Bridge → Ditto pipeline.

    Parameters
    ----------
    audio_input      : Path to the user's input audio (any format; auto-resampled
                       internally to 24 kHz for Moshi / 16 kHz for Ditto).
    image_input      : Path to the portrait image for Ditto (.jpg / .png).
    output_path      : Destination for the final .mp4 with Moshi audio.
    bridge_ckpt      : Path to the trained bridge .pt checkpoint.
    bridge_config    : Path to bridge_module/config.yaml.
    ditto_data_root  : Path to Ditto TRT model directory.
    ditto_cfg_pkl    : Path to Ditto .pkl config file.
    moshi_hf_repo    : HuggingFace repo for Moshi (default: moshiko-pytorch-bf16).
    moshi_weight     : Optional local path to Moshi .safetensors.
    mimi_weight      : Optional local path to Mimi .safetensors.
    tokenizer        : Optional local path to text tokenizer.
    batch_size       : Moshi batch size (default 8, matching trained model).
    batch_index      : Which batch item to use for downstream pipeline (0–7).
    device           : "cuda" or "cpu".
    dtype            : Moshi LM dtype (torch.bfloat16 recommended).
    save_moshi_audio : If provided, copy Moshi's generated WAV to this path.
    save_bridge_features : If provided, save bridge output as .npy to this path.
    save_silent_video    : If provided, keep the silent Ditto video at this path.

    Returns
    -------
    str : path to the final output .mp4 with Moshi-generated audio.
    """

    t_total_start = time.time()
    print("\n" + "═" * 60)
    print("  Moshi + Bridge + Ditto — Unified Talking-Head Pipeline")
    print("═" * 60)
    print(f"  Audio input  : {audio_input}")
    print(f"  Image input  : {image_input}")
    print(f"  Output       : {output_path}")
    print(f"  Device       : {device}")
    print("═" * 60 + "\n")

    # ── Resolve device ────────────────────────────────────────────────────────
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU.")
        device = "cpu"

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Moshi Inference
    # Input audio → Moshi LM → (Moshi audio WAV) + (acoustic tokens T×8)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[Step 1/4] 🎙️  Running Moshi inference...")
    t0 = time.time()

    moshi_runner = MoshiTokenRunner(
        hf_repo    = moshi_hf_repo,
        moshi_weight = moshi_weight,
        mimi_weight  = mimi_weight,
        tokenizer    = tokenizer,
        device       = device,
        dtype        = dtype,
        batch_size   = batch_size,
    )

    # Use a temp file for Moshi audio unless user wants to keep it
    _moshi_audio_tmp = None
    if save_moshi_audio:
        moshi_audio_dest = os.path.abspath(save_moshi_audio)
    else:
        _moshi_audio_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        moshi_audio_dest = _moshi_audio_tmp.name
        _moshi_audio_tmp.close()

    moshi_audio_path, acoustic_tokens = moshi_runner.run(
        audio_input_path  = audio_input,
        batch_index       = batch_index,
        output_audio_path = moshi_audio_dest,
    )

    print(f"[Step 1/4] ✅  Done ({time.time()-t0:.1f}s) — "
          f"tokens: {tuple(acoustic_tokens.shape)}, "
          f"audio: {moshi_audio_path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Bridge Module
    # Acoustic tokens (T, 8) → HuBERT-like features (N, 1024) float32
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[Step 2/4] 🔗  Running Bridge module (tokens → features)...")
    t0 = time.time()

    bridge_runner = BridgeRunner(
        checkpoint_path = bridge_ckpt,
        config_path     = bridge_config,
        device          = device,
    )
    audio_features = bridge_runner.run(acoustic_tokens)   # (N, 1024) float32 numpy

    if save_bridge_features:
        import numpy as np
        feat_path = os.path.abspath(save_bridge_features)
        np.save(feat_path, audio_features)
        print(f"[Step 2/4]    Bridge features saved → {feat_path}")

    print(f"[Step 2/4] ✅  Done ({time.time()-t0:.1f}s) — "
          f"features: {audio_features.shape}")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3 — Ditto Video Generation
    # Image + features (N, 1024) → silent talking-head .mp4
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[Step 3/4] 🎬  Running Ditto video generation...")
    t0 = time.time()

    ditto_runner = DittoRunner(
        data_root = ditto_data_root,
        cfg_pkl   = ditto_cfg_pkl,
    )

    # Determine silent video path
    if save_silent_video:
        silent_video_path = os.path.abspath(save_silent_video)
    else:
        _silent_tmp = tempfile.NamedTemporaryFile(suffix="_silent.mp4", delete=False)
        silent_video_path = _silent_tmp.name
        _silent_tmp.close()
        import os as _os
        _os.unlink(silent_video_path)   # remove empty file so Ditto can write it

    silent_video_path = ditto_runner.run(
        image_path     = image_input,
        audio_features = audio_features,
        output_path    = silent_video_path,
    )

    print(f"[Step 3/4] ✅  Done ({time.time()-t0:.1f}s) — "
          f"silent video: {silent_video_path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 4 — Merge Moshi Audio + Ditto Video
    # silent .mp4 + moshi .wav → final .mp4 with Moshi voice
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[Step 4/4] 🔊  Merging Moshi audio into video...")
    t0 = time.time()

    final_path = merge_audio_into_video(
        video_path  = silent_video_path,
        audio_path  = moshi_audio_path,
        output_path = output_path,
        overwrite   = True,
    )

    print(f"[Step 4/4] ✅  Done ({time.time()-t0:.1f}s)")

    # ── Cleanup temp files ────────────────────────────────────────────────────
    if _moshi_audio_tmp and os.path.isfile(moshi_audio_path) and not save_moshi_audio:
        try:
            os.unlink(moshi_audio_path)
        except OSError:
            pass

    if not save_silent_video and os.path.isfile(silent_video_path):
        try:
            os.unlink(silent_video_path)
        except OSError:
            pass

    # Cleanup any leftover .tmp.mp4 from Ditto SDK
    ditto_tmp = silent_video_path + ".tmp.mp4"
    if os.path.isfile(ditto_tmp):
        try:
            os.unlink(ditto_tmp)
        except OSError:
            pass

    t_total = time.time() - t_total_start
    print("\n" + "═" * 60)
    print(f"  ✅  Pipeline complete in {t_total:.1f}s")
    print(f"  📹  Output → {final_path}")
    print("═" * 60 + "\n")

    return final_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Moshi + Bridge + Ditto — Unified Talking-Head Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    p.add_argument("--audio_input",  required=True,
                   help="Path to input audio file (any format, e.g. .wav .mp3 .flac)")
    p.add_argument("--image_input",  required=True,
                   help="Path to the portrait image (.jpg / .png)")
    p.add_argument("--output_path",  required=True,
                   help="Path for the final .mp4 output with Moshi-generated audio")

    # ── Bridge ────────────────────────────────────────────────────────────────
    p.add_argument("--bridge_ckpt",
                   default=DEFAULT_BRIDGE_CKPT,
                   help=f"Path to trained bridge .pt checkpoint\n"
                        f"[default: {DEFAULT_BRIDGE_CKPT}]")
    p.add_argument("--bridge_config",
                   default=DEFAULT_BRIDGE_CONFIG,
                   help=f"Path to bridge config.yaml\n"
                        f"[default: {DEFAULT_BRIDGE_CONFIG}]")

    # ── Ditto ─────────────────────────────────────────────────────────────────
    p.add_argument("--ditto_data_root",
                   default=DEFAULT_DITTO_DATA_ROOT,
                   help=f"Path to Ditto TRT model directory\n"
                        f"[default: {DEFAULT_DITTO_DATA_ROOT}]")
    p.add_argument("--ditto_cfg_pkl",
                   default=DEFAULT_DITTO_CFG_PKL,
                   help=f"Path to Ditto .pkl config file\n"
                        f"[default: {DEFAULT_DITTO_CFG_PKL}]")

    # ── Moshi ─────────────────────────────────────────────────────────────────
    p.add_argument("--moshi_hf_repo",  default=DEFAULT_MOSHI_REPO,
                   help="HuggingFace repo for Moshi models")
    p.add_argument("--moshi_weight",   default=None,
                   help="(optional) Local path to Moshi .safetensors")
    p.add_argument("--mimi_weight",    default=None,
                   help="(optional) Local path to Mimi .safetensors")
    p.add_argument("--tokenizer",      default=None,
                   help="(optional) Local path to text tokenizer .model file")
    p.add_argument("--batch_size",     type=int, default=DEFAULT_BATCH_SIZE,
                   help=f"Moshi batch size [default: {DEFAULT_BATCH_SIZE}]")
    p.add_argument("--batch_index",    type=int, default=0,
                   help="Which Moshi batch item to use for the video [default: 0]")
    p.add_argument("--device",         default="cuda",
                   help="Compute device: cuda | cpu [default: cuda]")
    p.add_argument("--half",           action="store_const",
                   const=torch.float16, default=torch.bfloat16, dest="dtype",
                   help="Use float16 instead of bfloat16 (for older GPUs)")

    # ── Optional saves ────────────────────────────────────────────────────────
    p.add_argument("--save_moshi_audio",     default=None,
                   help="(optional) Save Moshi-generated WAV to this path")
    p.add_argument("--save_bridge_features", default=None,
                   help="(optional) Save bridge features as .npy to this path")
    p.add_argument("--save_silent_video",    default=None,
                   help="(optional) Keep the silent Ditto video at this path")

    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    run_pipeline(
        audio_input            = args.audio_input,
        image_input            = args.image_input,
        output_path            = args.output_path,
        bridge_ckpt            = args.bridge_ckpt,
        bridge_config          = args.bridge_config,
        ditto_data_root        = args.ditto_data_root,
        ditto_cfg_pkl          = args.ditto_cfg_pkl,
        moshi_hf_repo          = args.moshi_hf_repo,
        moshi_weight           = args.moshi_weight,
        mimi_weight            = args.mimi_weight,
        tokenizer              = args.tokenizer,
        batch_size             = args.batch_size,
        batch_index            = args.batch_index,
        device                 = args.device,
        dtype                  = args.dtype,
        save_moshi_audio       = args.save_moshi_audio,
        save_bridge_features   = args.save_bridge_features,
        save_silent_video      = args.save_silent_video,
    )
