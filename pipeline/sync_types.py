"""
pipeline/sync_types.py
======================
Shared typed dataclasses for inter-component communication in the
Moshi → Bridge → Ditto pipeline.

Every item that crosses a queue boundary carries a ``seq`` (sequence
number) that is stamped by the Moshi layer and propagated unchanged
through Bridge and Ditto all the way to the WebSocket sender.  The
browser uses this sequence number to align audio packets with video
frames, enabling soft audio-video synchronisation without a shared
wall-clock.

Sequence numbers are uint32 (0–4 294 967 295) and wrap around safely.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Sequence number helpers
# ---------------------------------------------------------------------------

SEQ_MAX: int = 0xFFFF_FFFF  # uint32 max


def seq_pack(seq: int) -> bytes:
    """Pack a uint32 sequence number into 4 big-endian bytes."""
    return struct.pack(">I", seq & SEQ_MAX)


def seq_unpack(data: bytes, offset: int = 0) -> int:
    """Unpack a uint32 sequence number from 4 big-endian bytes."""
    return struct.unpack_from(">I", data, offset)[0]


# ---------------------------------------------------------------------------
# Tagged data containers
# ---------------------------------------------------------------------------

@dataclass
class TaggedToken:
    """
    One set of Moshi LM acoustic tokens, tagged with the session sequence
    number of the LM step that produced them.

    Attributes
    ----------
    seq    : monotonically increasing LM step counter (wraps at 2^32)
    tensor : (1, dep_q) int64 CPU tensor — dep_q acoustic codebook tokens
    """
    seq: int
    tensor: torch.Tensor


@dataclass
class TaggedFeatures:
    """
    Bridge output for one (or more) Moshi LM steps, tagged with the
    sequence number of the *first* step in the batch.

    Attributes
    ----------
    seq      : sequence number of the first Moshi step covered by this chunk
    features : (N, 1024) float32 numpy array — HuBERT-compatible features
    """
    seq: int
    features: np.ndarray


@dataclass
class TaggedFrame:
    """
    One JPEG video frame from Ditto, tagged with the Moshi step sequence
    number whose features drove this frame.

    Attributes
    ----------
    seq  : sequence number matching the Moshi audio step this frame aligns to
    jpeg : JPEG-encoded bytes of the rendered video frame
    """
    seq: int
    jpeg: bytes
