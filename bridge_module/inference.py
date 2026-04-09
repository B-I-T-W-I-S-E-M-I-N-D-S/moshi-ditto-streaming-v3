"""
inference.py — Streaming & Batch Inference
==========================================
Provides:
  - BridgeInference: batch mode  (.pt  OR  .onnx)
  - StreamingBridgeInference: causal chunk-by-chunk streaming (.pt only)
  - OnnxBridgeInference: fast ONNX batch inference (.onnx)
  - CLI entry point  (including --compare mode)

Both modes produce HuBERT-compatible features (25 Hz, output_dim-dim) from
Mimi tokens. output_dim is read from config.yaml (default: 1024 for HuBERT-large).

ONNX Inference
--------------
When the checkpoint path ends with .onnx the CLI and BridgeInference
automatically route through OnnxBridgeInference for maximum speed.

  python inference.py \\
      --checkpoint checkpoints/bridge_best.onnx \\
      --config config.yaml \\
      --audio input.wav \\
      --output features.pt

--compare mode (shortcut to compare_inference.py)
-------------------------------------------------
  python inference.py \\
      --audio  path/to/audio.wav \\
      --checkpoint checkpoints/best.pt      # or best.onnx \\
      --config config.yaml \\
      --compare

This extracts:
  1. Real HuBERT features via the ONNX model (ground-truth)  → hubert_gt_features.npy
  2. Bridge-model prediction via Mimi tokens                 → bridge_pred_features.npy
Then prints MSE / MAE / RMSE / cosine-similarity / SNR error metrics.

Override default .npy output paths with:
  --save-gt-npy   my_hubert.npy
  --save-pred-npy my_bridge.npy

Disable auto .npy saving:
  --no-auto-save-npy
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from model import MimiHuBERTBridge

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Shared checkpoint loader  (.pt)
# ──────────────────────────────────────────────────────────────────────────────

def _load_checkpoint(path: str, model: MimiHuBERTBridge, device: torch.device):
    """
    Load bridge weights from a .pt checkpoint file.
    Supports both full trainer checkpoints ({"bridge": state_dict, ...})
    and bare state_dicts saved directly.
    """
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    sd = ckpt.get("bridge", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning(f"Missing keys in checkpoint: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys in checkpoint: {unexpected}")


def _is_onnx(path: str) -> bool:
    return str(path).lower().endswith(".onnx")


# ──────────────────────────────────────────────────────────────────────────────
# ONNX Batch Inference  (fastest path)
# ──────────────────────────────────────────────────────────────────────────────

class OnnxBridgeInference:
    """
    ONNX-Runtime batch inference wrapper.
    Loads a bridge_best.onnx model and converts Mimi tokens → HuBERT-like features.

    The ONNX model was exported with:
      input  : "tokens"   — int64  (B, T, num_codebooks)
      output : "features" — float32 (B, upsample_factor*T, output_dim)

    This class has the same public API as BridgeInference so callers can
    switch between .pt and .onnx transparently.
    """

    def __init__(self, onnx_path: str, config_path: str, device: Optional[str] = None):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX inference.\n"
                "Install with:  pip install onnxruntime-gpu   # GPU\n"
                "           or: pip install onnxruntime        # CPU-only"
            )

        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.output_dim    = self.cfg["model"]["output_dim"]
        self.num_codebooks = self.cfg["model"]["num_codebooks"]

        # Resolve device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device_str = device

        # Build ONNX-Runtime session with the best available provider
        providers = self._build_providers(device)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(onnx_path, sess_opts, providers=providers)

        self._input_name  = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        logger.info(
            f"Loaded ONNX bridge from {onnx_path}  "
            f"[providers={providers}]  output_dim={self.output_dim}"
        )

    @staticmethod
    def _build_providers(device: str) -> list:
        """Return the best ONNX-Runtime execution providers for the device."""
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
        except Exception:
            available = []

        if "cuda" in device.lower():
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "TensorrtExecutionProvider" in available:
            return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def __call__(
        self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        tokens : (B, T, num_codebooks) or (T, num_codebooks)  int64
        returns: (B, upsample*T, output_dim)  float32  — always on CPU
        """
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)

        tokens_np = tokens.cpu().numpy().astype(np.int64)

        outputs = self._session.run(
            [self._output_name],
            {self._input_name: tokens_np},
        )
        features = torch.from_numpy(outputs[0])   # (B, uT, output_dim) float32

        if mask is not None:
            upsample = self.cfg["model"]["upsample_factor"]
            mask_up  = mask.repeat_interleave(upsample, dim=-1)   # (B, uT)
            features = features * mask_up.unsqueeze(-1)

        return features   # CPU float32

    def from_audio(self, audio_path: str) -> torch.Tensor:
        """End-to-end: audio file → HuBERT-like features  (T_h, output_dim)."""
        from dataset import MimiExtractor
        import torchaudio

        waveform, native_sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)

        extractor = MimiExtractor(self.cfg["paths"]["mimi_model"])
        tokens    = extractor.extract(waveform, native_sr)   # (T, num_codebooks)

        features  = self(tokens)                              # (1, uT, output_dim)
        return features.squeeze(0)                            # (uT, output_dim)


# ──────────────────────────────────────────────────────────────────────────────
# Batch Inference  (.pt  — original PyTorch path)
# ──────────────────────────────────────────────────────────────────────────────

class BridgeInference:
    """
    Simple batch inference wrapper for .pt checkpoints.
    For .onnx checkpoints use OnnxBridgeInference (or pass the path to
    create_inference() which auto-routes).
    """

    def __init__(self, checkpoint_path: str, config_path: str, device: Optional[str] = None):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device(self.cfg["inference"].get("device", "cuda"))
        else:
            self.device = torch.device("cpu")

        self.output_dim    = self.cfg["model"]["output_dim"]
        self.num_codebooks = self.cfg["model"]["num_codebooks"]

        self.model = MimiHuBERTBridge(self.cfg).to(self.device)
        _load_checkpoint(checkpoint_path, self.model, self.device)
        self.model.eval()
        logger.info(
            f"Loaded bridge (.pt) from {checkpoint_path} on {self.device} "
            f"(output_dim={self.output_dim})"
        )

    @torch.no_grad()
    def __call__(
        self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        tokens : (B, T, num_codebooks) or (T, num_codebooks)  int64
        returns: (B, uT, output_dim)  float32  — always on CPU
        """
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)

        tokens   = tokens.to(self.device)
        features, _ = self.model(tokens)       # (B, uT, output_dim)
        features = features.float()

        if mask is not None:
            upsample = self.cfg["model"]["upsample_factor"]
            mask_up  = mask.repeat_interleave(upsample, dim=-1).to(self.device)
            features = features * mask_up.unsqueeze(-1)

        return features.cpu()

    @torch.no_grad()
    def from_audio(self, audio_path: str) -> torch.Tensor:
        """End-to-end: audio file → HuBERT-like features  (T_h, output_dim)."""
        from dataset import MimiExtractor
        import torchaudio

        waveform, native_sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)

        extractor = MimiExtractor(self.cfg["paths"]["mimi_model"])
        tokens    = extractor.extract(waveform, native_sr)

        features  = self(tokens)
        return features.squeeze(0)


# ──────────────────────────────────────────────────────────────────────────────
# Factory: auto-select .pt or .onnx backend
# ──────────────────────────────────────────────────────────────────────────────

def create_inference(
    checkpoint_path: str,
    config_path: str,
    device: Optional[str] = None,
):
    """
    Return the right inference object based on the checkpoint extension.

      .onnx  →  OnnxBridgeInference   (fastest — uses ONNX Runtime)
      .pt    →  BridgeInference        (standard PyTorch)

    Both objects expose the same  __call__(tokens)  and  from_audio(path)  API.
    """
    if _is_onnx(checkpoint_path):
        return OnnxBridgeInference(checkpoint_path, config_path, device=device)
    return BridgeInference(checkpoint_path, config_path, device=device)


# ──────────────────────────────────────────────────────────────────────────────
# Streaming Inference  (.pt only — ONNX does not support KV-cache streaming)
# ──────────────────────────────────────────────────────────────────────────────

class StreamingBridgeInference:
    """
    Causal streaming inference with KV-cache.  Requires a .pt checkpoint.

    Usage:
        stream = StreamingBridgeInference(checkpoint, config)
        stream.reset()
        for mimi_chunk in token_stream:           # (chunk_size, num_codebooks)
            feat_chunk = stream.step(mimi_chunk)  # (upsample*chunk_size, output_dim)
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        chunk_size: Optional[int] = None,
        device: Optional[str] = None,
    ):
        if _is_onnx(checkpoint_path):
            raise ValueError(
                "StreamingBridgeInference requires a .pt checkpoint. "
                "ONNX models do not support KV-cache streaming. "
                "Use OnnxBridgeInference for batch ONNX inference."
            )

        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.chunk_size    = chunk_size or self.cfg["inference"].get("chunk_size", 50)
        self.output_dim    = self.cfg["model"]["output_dim"]
        self.num_codebooks = self.cfg["model"]["num_codebooks"]

        self.model = MimiHuBERTBridge(self.cfg).to(self.device)
        _load_checkpoint(checkpoint_path, self.model, self.device)
        self.model.eval()

        self._past_kvs: Optional[list] = None
        self._step_count = 0

    def reset(self):
        """Reset streaming state (call at the start of each new utterance)."""
        self._past_kvs   = None
        self._step_count = 0

    @torch.no_grad()
    def step(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        Process one chunk of Mimi tokens.

        chunk   : (C, num_codebooks)  or  (1, C, num_codebooks)  int64
        returns : (upsample*C, output_dim)  float32  CPU tensor
        """
        if chunk.dim() == 2:
            chunk = chunk.unsqueeze(0)          # (1, C, num_codebooks)

        chunk = chunk.to(self.device)
        features, present_kvs = self.model(
            chunk, use_cache=True, past_kvs=self._past_kvs
        )
        self._past_kvs   = present_kvs
        self._step_count += chunk.shape[1]

        return features.squeeze(0).float().cpu()   # (uC, output_dim)

    def stream_tokens(self, tokens: torch.Tensor) -> Iterator[torch.Tensor]:
        """
        Yield feature chunks from a full token sequence.
        tokens : (T, num_codebooks)
        yields : (upsample*chunk_size, output_dim) tensors
        """
        self.reset()
        T = tokens.shape[0]
        for start in range(0, T, self.chunk_size):
            chunk = tokens[start : start + self.chunk_size]
            yield self.step(chunk)


# ──────────────────────────────────────────────────────────────────────────────
# Latency Benchmark Utility
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_streaming(
    checkpoint: str,
    config: str,
    num_chunks: int = 100,
    chunk_size: int = 50,
    warmup: int = 5,
):
    """Quick per-chunk latency benchmark (streaming .pt or batch .onnx)."""
    with open(config) as f:
        cfg = yaml.safe_load(f)

    num_codebooks = cfg["model"]["num_codebooks"]
    output_dim    = cfg["model"]["output_dim"]
    audio_secs    = chunk_size / cfg["data"]["mimi_rate"]

    dummy_tokens = torch.randint(0, cfg["model"]["vocab_size"], (chunk_size, num_codebooks))

    if _is_onnx(checkpoint):
        print("\n=== ONNX Batch Latency Benchmark ===")
        infer = OnnxBridgeInference(checkpoint, config)

        # Warmup
        for _ in range(warmup):
            infer(dummy_tokens)

        times = []
        for _ in range(num_chunks):
            t0 = time.perf_counter()
            infer(dummy_tokens)
            times.append((time.perf_counter() - t0) * 1000)
    else:
        print("\n=== Streaming (.pt) Latency Benchmark ===")
        stream = StreamingBridgeInference(checkpoint, config, chunk_size=chunk_size)
        for _ in range(warmup):
            stream.reset()
            stream.step(dummy_tokens)

        times = []
        stream.reset()
        for _ in range(num_chunks):
            t0 = time.perf_counter()
            stream.step(dummy_tokens)
            times.append((time.perf_counter() - t0) * 1000)

    import statistics
    print(f"Chunk size  : {chunk_size} Mimi frames  |  output_dim={output_dim}")
    print(f"Audio/chunk : {audio_secs:.2f}s")
    print(f"Median : {statistics.median(times):.2f} ms")
    print(f"P95    : {sorted(times)[int(0.95 * len(times))]:.2f} ms")
    print(f"Max    : {max(times):.2f} ms")
    print(
        f"Throughput: {1000 / statistics.median(times):.1f} chunks/s  "
        f"({1000 * audio_secs / statistics.median(times):.1f}× realtime)"
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Mimi-to-HuBERT Bridge Inference  (.pt or .onnx)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pt or .onnx checkpoint")
    parser.add_argument("--config",     required=True,
                        help="Path to config.yaml")
    parser.add_argument("--audio",      default=None,
                        help="Input audio file (.wav / .flac)")
    parser.add_argument("--tokens",     default=None,
                        help="Pre-extracted .pt token file (T, num_codebooks)")
    parser.add_argument("--output",     default="features.pt",
                        help="Output .pt file path (non-compare mode)")
    parser.add_argument("--streaming",  action="store_true",
                        help="Use causal streaming mode with KV-cache (.pt only)")
    parser.add_argument("--chunk-size", type=int, default=50,
                        help="Chunk size in Mimi frames (streaming only)")
    parser.add_argument("--benchmark",  action="store_true",
                        help="Run latency benchmark instead of inference")
    parser.add_argument("--device",     default=None,
                        help="Force device (cuda / cpu)")

    # ── Compare mode ──────────────────────────────────────────────────────────
    parser.add_argument("--compare",    action="store_true",
                        help=(
                            "Compare real HuBERT output (ONNX) vs bridge model prediction "
                            "for the given --audio file. Supports both .pt and .onnx bridge. "
                            "Prints MSE/MAE/RMSE/cosine/SNR.  Requires --audio."
                        ))
    parser.add_argument("--hubert-model", default=None,
                        help="(compare mode) Override path to HuBERT ONNX file")
    parser.add_argument("--mimi-model",   default=None,
                        help="(compare mode) Override Mimi HF repo or local path")
    parser.add_argument("--save-gt",      default=None,
                        help="(compare mode) Also save ground-truth features as .pt")
    parser.add_argument("--save-pred",    default=None,
                        help="(compare mode) Also save bridge prediction features as .pt")
    parser.add_argument("--save-gt-npy",   default="hubert_gt_features.npy",
                        help="(compare mode) Path for HuBERT GT .npy output")
    parser.add_argument("--save-pred-npy", default="bridge_pred_features.npy",
                        help="(compare mode) Path for Bridge pred .npy output")
    parser.add_argument("--no-auto-save-npy", action="store_true",
                        help="(compare mode) Disable automatic .npy saving")
    parser.add_argument("--plot",         action="store_true",
                        help="(compare mode) Show matplotlib heatmap plots")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.benchmark:
        benchmark_streaming(args.checkpoint, args.config, chunk_size=args.chunk_size)
        return

    # ── Compare mode ──────────────────────────────────────────────────────────
    if args.compare:
        if args.audio is None:
            parser.error("--compare requires --audio")
        from compare_inference import compare as run_compare
        run_compare(
            audio_path            = args.audio,
            checkpoint_path       = args.checkpoint,
            config_path           = args.config,
            device                = args.device,
            hubert_model_override = args.hubert_model,
            mimi_model_override   = args.mimi_model,
            save_gt               = args.save_gt,
            save_pred             = args.save_pred,
            save_gt_npy           = args.save_gt_npy,
            save_pred_npy         = args.save_pred_npy,
            auto_save_npy         = not args.no_auto_save_npy,
            plot                  = args.plot,
        )
        return

    if args.audio is None and args.tokens is None:
        parser.error("Provide at least one of --audio or --tokens")

    # ── Load tokens ───────────────────────────────────────────────────────────
    def get_tokens(cfg: dict) -> torch.Tensor:
        if args.tokens:
            try:
                return torch.load(args.tokens, weights_only=True)
            except TypeError:
                return torch.load(args.tokens)
        from dataset import MimiExtractor
        import torchaudio
        wav, native_sr = torchaudio.load(args.audio)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        return MimiExtractor(cfg["paths"]["mimi_model"]).extract(wav, native_sr)

    # ── ONNX path (batch only — ONNX does not support streaming KV-cache) ─────
    if _is_onnx(args.checkpoint):
        if args.streaming:
            logger.warning(
                "ONNX models do not support streaming/KV-cache mode. "
                "Falling back to batch inference."
            )
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        tokens = get_tokens(cfg) if not args.audio else None

        infer = OnnxBridgeInference(args.checkpoint, args.config, device=args.device)
        if args.audio:
            features = infer.from_audio(args.audio)
        else:
            features = infer(tokens).squeeze(0)

        logger.info(f"ONNX batch output: {features.shape}")
        torch.save(features, args.output)
        logger.info(f"Saved features → {args.output}")
        return

    # ── .pt path ──────────────────────────────────────────────────────────────
    if args.streaming:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        tokens = get_tokens(cfg)

        stream   = StreamingBridgeInference(
            args.checkpoint, args.config,
            chunk_size=args.chunk_size, device=args.device,
        )
        chunks   = list(stream.stream_tokens(tokens))
        features = torch.cat(chunks, dim=0)
        logger.info(f"Streaming output: {features.shape}")

    else:
        infer = BridgeInference(args.checkpoint, args.config, device=args.device)
        if args.audio:
            features = infer.from_audio(args.audio)
        else:
            with open(args.config) as f:
                cfg = yaml.safe_load(f)
            tokens   = get_tokens(cfg)
            features = infer(tokens).squeeze(0)
        logger.info(f"Batch output: {features.shape}")

    torch.save(features, args.output)
    logger.info(f"Saved features → {args.output}")


if __name__ == "__main__":
    main()
