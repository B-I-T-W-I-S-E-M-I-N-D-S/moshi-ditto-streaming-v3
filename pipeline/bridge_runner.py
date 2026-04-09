"""
pipeline/bridge_runner.py
=========================
Wraps BridgeInference (bridge_module/inference.py) to convert Mimi acoustic
tokens → HuBERT-like continuous features compatible with Ditto's diffusion model.

Input  : (T, 8) int64 tensor — raw Moshi LM acoustic tokens @ 12.5 Hz
Output : (N, 1024) numpy float32 array @ 25 Hz  (N ≈ 2T due to ×2 upsampling)

The output shape matches exactly what Ditto's wav2feat HuBERT extractor produces,
so it can be fed directly into `StreamSDK.audio2motion_queue`.

Usage
-----
    runner = BridgeRunner(
        checkpoint_path="checkpoints/bridge_best.pt",
        config_path="bridge_module/config.yaml",
        device="cuda",
    )
    features = runner.run(acoustic_tokens)   # numpy (N, 1024) float32
"""

import os
import sys
import numpy as np
import torch
from typing import Optional

# ---------------------------------------------------------------------------
# Ensure bridge_module is importable
# ---------------------------------------------------------------------------
_BRIDGE_DIR = os.path.join(os.path.dirname(__file__), "..", "bridge_module")
if _BRIDGE_DIR not in sys.path:
    sys.path.insert(0, _BRIDGE_DIR)

from inference import BridgeInference  # bridge_module/inference.py


class BridgeRunner:
    """
    Loads the trained MimiHuBERTBridge model and converts acoustic tokens to
    HuBERT-compatible feature vectors for Ditto.

    Parameters
    ----------
    checkpoint_path : path to the trained bridge .pt file
                      (default: checkpoints/bridge_best.pt relative to project root)
    config_path     : path to bridge config.yaml
                      (default: bridge_module/config.yaml)
    device          : "cuda" or "cpu"
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/bridge_best.pt",
        config_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "bridge_module", "config.yaml"
            )

        # Resolve relative paths to absolute so they work from any cwd
        checkpoint_path = os.path.abspath(checkpoint_path)
        config_path = os.path.abspath(config_path)

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Bridge checkpoint not found: {checkpoint_path}\n"
                "Please place your trained bridge model at that path."
            )
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Bridge config not found: {config_path}")

        print(f"[BridgeRunner] Loading bridge from: {checkpoint_path}")
        self.bridge = BridgeInference(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=device,
        )
        print(
            f"[BridgeRunner] Ready "
            f"(output_dim={self.bridge.output_dim}, device={self.bridge.device})"
        )

    def run(self, acoustic_tokens: torch.Tensor) -> np.ndarray:
        """
        Convert Moshi LM acoustic tokens → HuBERT-like features.

        Parameters
        ----------
        acoustic_tokens : torch.Tensor (T, 8) or (B, T, 8) int64
            Raw Moshi LM acoustic codebook tokens.
            When a 2-D tensor is passed (T, 8), a batch dimension is added automatically.

        Returns
        -------
        numpy.ndarray (N, 1024) float32
            HuBERT-compatible features at 25 Hz (N ≈ 2T).
            This is exactly the shape Ditto's audio2motion pipeline expects.
        """
        if acoustic_tokens.dim() == 2:
            # (T, 8) → (1, T, 8)
            tokens_batch = acoustic_tokens.unsqueeze(0)
        else:
            tokens_batch = acoustic_tokens

        print(
            f"[BridgeRunner] Running bridge inference on tokens "
            f"{tuple(acoustic_tokens.shape)} ..."
        )

        # BridgeInference.__call__ returns (B, 2T, 1024) float32 on CPU
        features = self.bridge(tokens_batch)  # (1, 2T, 1024)
        features = features.squeeze(0)         # (2T, 1024)
        features_np = features.numpy().astype(np.float32)

        print(
            f"[BridgeRunner] Features produced: {features_np.shape} "
            f"(dtype={features_np.dtype})"
        )

        # Sanity check: Ditto requires exactly 1024-dim features
        assert features_np.ndim == 2 and features_np.shape[1] == 1024, (
            f"Expected (N, 1024) features, got {features_np.shape}"
        )

        return features_np
