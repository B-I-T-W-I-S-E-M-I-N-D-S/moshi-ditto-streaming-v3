"""
pipeline — Unified Moshi + Bridge + Ditto integration layer.

Sub-modules
-----------
moshi_runner         : Offline batch inference; captures acoustic tokens + output audio.
bridge_runner        : Converts Mimi tokens → HuBERT-like features (batch mode).
ditto_runner         : Drives Ditto StreamSDK with pre-computed features (offline).
merge_audio_video    : FFmpeg utility to mux audio into a silent video.
streaming_moshi      : Real-time Moshi streaming state (per-frame token + audio capture).
ditto_stream_adapter : Ditto online mode with per-frame JPEG output instead of file write.
sync_types           : Shared seq-tagged dataclasses for inter-component communication.
"""

from .moshi_runner import MoshiTokenRunner
from .bridge_runner import BridgeRunner
from .ditto_runner import DittoRunner
from .merge_audio_video import merge_audio_into_video
from .streaming_moshi import StreamingMoshiState
from .ditto_stream_adapter import DittoStreamAdapter
from .sync_types import TaggedToken, TaggedFeatures, TaggedFrame, seq_pack, seq_unpack

__all__ = [
    # Offline pipeline
    "MoshiTokenRunner",
    "BridgeRunner",
    "DittoRunner",
    "merge_audio_into_video",
    # Streaming pipeline
    "StreamingMoshiState",
    "DittoStreamAdapter",
    # Sync utilities
    "TaggedToken",
    "TaggedFeatures",
    "TaggedFrame",
    "seq_pack",
    "seq_unpack",
]
