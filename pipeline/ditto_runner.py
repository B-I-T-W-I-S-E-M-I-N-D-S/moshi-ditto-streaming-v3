"""
pipeline/ditto_runner.py
========================
Wraps Ditto's StreamSDK to generate a silent talking-head video from
a portrait image and pre-computed HuBERT-like audio features.

The TRT models are assumed to be pre-loaded on disk (no runtime downloads).
No HuBERT extraction is performed here; features come directly from the
Bridge module.

Usage
-----
    runner = DittoRunner(
        data_root="ditto-inference/checkpoints/ditto_trt_Ampere_Plus",
        cfg_pkl="ditto-inference/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl",
    )
    silent_video_path = runner.run(
        image_path="portrait.jpg",
        audio_features=features_np,   # (N, 1024) float32
        output_path="output_silent.mp4",
    )
"""

import os
import sys
import numpy as np
import math
from typing import Optional

# ---------------------------------------------------------------------------
# Ensure ditto-inference is importable
# ---------------------------------------------------------------------------
_DITTO_DIR = os.path.join(os.path.dirname(__file__), "..", "ditto-inference")
if _DITTO_DIR not in sys.path:
    sys.path.insert(0, _DITTO_DIR)

from stream_pipeline_offline import StreamSDK  # ditto-inference/stream_pipeline_offline.py


class DittoRunner:
    """
    Pre-loads Ditto TRT models once; generates silent talking-head video on demand.

    Parameters
    ----------
    data_root : path to the Ditto TRT model directory
                (e.g. "ditto-inference/checkpoints/ditto_trt_Ampere_Plus")
    cfg_pkl   : path to the Ditto config .pkl file
                (e.g. "ditto-inference/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl")
    """

    def __init__(self, data_root: str, cfg_pkl: str):
        data_root = os.path.abspath(data_root)
        cfg_pkl   = os.path.abspath(cfg_pkl)

        if not os.path.isdir(data_root):
            raise FileNotFoundError(
                f"Ditto TRT model directory not found: {data_root}"
            )
        if not os.path.isfile(cfg_pkl):
            raise FileNotFoundError(
                f"Ditto config .pkl not found: {cfg_pkl}"
            )

        print(f"[DittoRunner] Loading Ditto SDK from: {data_root}")
        self.sdk = StreamSDK(cfg_pkl, data_root)
        print("[DittoRunner] Ditto SDK ready.")

    def run(
        self,
        image_path: str,
        audio_features: np.ndarray,
        output_path: str,
        fade_in: int = -1,
        fade_out: int = -1,
    ) -> str:
        """
        Generate a **silent** talking-head video from a portrait image and
        pre-computed HuBERT-like audio features.

        Parameters
        ----------
        image_path     : path to the portrait image (.jpg / .png)
        audio_features : numpy array (N, 1024) float32 — bridge module output
                         N frames at 25 Hz (defines video duration)
        output_path    : path for the silent output .mp4
        fade_in        : number of frames to fade in  (-1 = disabled)
        fade_out       : number of frames to fade out (-1 = disabled)

        Returns
        -------
        str : ``output_path`` (the silent .mp4 that was written)
        """
        image_path = os.path.abspath(image_path)
        output_path = os.path.abspath(output_path)

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Portrait image not found: {image_path}")

        audio_features = audio_features.astype(np.float32)
        if audio_features.ndim != 2 or audio_features.shape[1] != 1024:
            raise ValueError(
                f"Expected audio_features shape (N, 1024), got {audio_features.shape}"
            )

        num_frames = len(audio_features)
        duration_sec = num_frames / 25.0
        print(
            f"[DittoRunner] Generating video: {num_frames} frames "
            f"({duration_sec:.1f}s) from image '{os.path.basename(image_path)}'"
        )

        # Setup the SDK for this source image and output file
        self.sdk.setup(image_path, output_path)

        # Setup frame count + optional fade transition
        self.sdk.setup_Nd(N_d=num_frames, fade_in=fade_in, fade_out=fade_out)

        # Push features directly into the audio→motion queue (offline mode)
        # This bypasses HuBERT extraction entirely — exactly what we want.
        assert not self.sdk.online_mode, (
            "online_mode is not supported when using pre-computed features"
        )
        self.sdk.audio2motion_queue.put(audio_features)

        # Wait for all worker threads to finish rendering
        self.sdk.close()

        # The silent video is written to output_path + ".tmp.mp4" by the SDK,
        # then we rename it to the requested output_path.
        tmp_path = output_path + ".tmp.mp4"
        if os.path.isfile(tmp_path) and not os.path.isfile(output_path):
            import shutil
            shutil.move(tmp_path, output_path)
            print(f"[DittoRunner] Silent video written → {output_path}")
        elif os.path.isfile(output_path):
            # Ditto already wrote it to output_path directly
            print(f"[DittoRunner] Silent video written → {output_path}")
        else:
            raise RuntimeError(
                f"Ditto did not produce expected output at {output_path} "
                f"or {tmp_path}."
            )

        return output_path
