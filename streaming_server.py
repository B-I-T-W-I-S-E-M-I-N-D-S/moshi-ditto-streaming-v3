"""
streaming_server.py
===================
FastAPI WebSocket server for the real-time Moshi + Bridge + Ditto pipeline.

Endpoints
---------
  GET  /                    → browser UI (static/index.html)
  POST /upload_image        → multipart upload; returns {"session_id": "..."}
  GET  /session/{sid}/status→ {"ready": true/false}
  WS   /ws/{session_id}     → bidirectional audio/video stream

WebSocket message protocol
--------------------------
  Browser → Server:
    0x01 <opus_bytes>      user's mic audio (Opus encoded)

  Server → Browser:
    0x00                   handshake / ready signal
    0x01 <seq:4> <pcm>     Moshi response audio — raw float32 LE + 4-byte seq
    0x02 <seq:4> <jpeg>    animated talking-head video frame + 4-byte seq
    0x03 <utf8_text>       text token / transcript piece
    0xFF <utf8_error>      error message

The 4-byte sequence number (big-endian uint32) in audio and video messages
allows the browser to align lip-sync: audio packet seq=N was played at a
known AudioContext time, and video frame seq=N should be displayed at that
same time.

Startup
-------
  python streaming_server.py [--host HOST] [--port PORT] [options]

RunPod
------
  The server listens on 0.0.0.0 so RunPod's public port proxy can forward
  traffic.  Set port to match your RunPod "HTTP Port" setting (default 7860).

Key fixes in this version
-------------------------
1. Thread-safe asyncio.Queue crossing: frame_reader_task uses
   asyncio.run_coroutine_threadsafe() instead of put_nowait() from a thread.
2. Adaptive bridge batching: flushes after BRIDGE_FLUSH_TIMEOUT_MS even if
   chunk is not full — eliminates 320ms+ stalls during pauses.
3. Shared error_event: any task failure signals all others to abort cleanly.
4. Session rejection: second WebSocket immediately gets an error instead of
   silently queuing behind the Moshi lock.
5. Dedicated CUDA stream for Bridge inference to reduce implicit CUDA syncs.
6. Sequence numbers propagated Audio→seq and Video→seq for A/V alignment.
7. Cleanup timeout: asyncio.gather in finally has a 10-second deadline.
"""

import argparse
import asyncio
import logging
import os
import sys
import time
import struct
import uuid
from pathlib import Path
from typing import Optional

import torch
import numpy as np

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ── Project root on sys.path ─────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline.streaming_moshi import StreamingMoshiState
from pipeline.ditto_stream_adapter import DittoStreamAdapter
from pipeline.sync_types import TaggedToken, seq_pack

# Bridge imports
_BRIDGE_DIR = os.path.join(_ROOT, "bridge_module")
if _BRIDGE_DIR not in sys.path:
    sys.path.insert(0, _BRIDGE_DIR)
from inference import StreamingBridgeInference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("streaming_server")

# ─────────────────────────────────────────────────────────────────────────────
# Global configuration  (overridden by CLI args or environment variables)
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    # Moshi
    MOSHI_HF_REPO:     str = os.environ.get("MOSHI_HF_REPO", "kyutai/moshiko-pytorch-bf16")
    MOSHI_WEIGHT:      Optional[str] = os.environ.get("MOSHI_WEIGHT")
    MIMI_WEIGHT:       Optional[str] = os.environ.get("MIMI_WEIGHT")
    TOKENIZER:         Optional[str] = os.environ.get("MOSHI_TOKENIZER")

    # Bridge
    BRIDGE_CKPT:   str = os.environ.get("BRIDGE_CKPT",
                         os.path.join(_ROOT, "checkpoints", "bridge_best.pt"))
    BRIDGE_CONFIG: str = os.environ.get("BRIDGE_CONFIG",
                         os.path.join(_ROOT, "bridge_module", "config.yaml"))
    BRIDGE_CHUNK:  int = int(os.environ.get("BRIDGE_CHUNK", "4"))  # Mimi frames per chunk

    # Adaptive bridge flush: send partial batch to Ditto after this many ms
    # even if the chunk is not full.  Prevents stalls during pauses.
    BRIDGE_FLUSH_TIMEOUT_MS: int = int(os.environ.get("BRIDGE_FLUSH_TIMEOUT_MS", "100"))

    # Ditto
    DITTO_DATA_ROOT: str = os.environ.get("DITTO_DATA_ROOT",
                           os.path.join(_ROOT, "ditto-inference", "checkpoints",
                                        "ditto_trt_Ampere_Plus"))
    DITTO_CFG_PKL:   str = os.environ.get("DITTO_CFG_PKL",
                           os.path.join(_ROOT, "ditto-inference", "checkpoints",
                                        "ditto_cfg", "v0.4_hubert_cfg_trt.pkl"))
    DITTO_EMO:               int = int(os.environ.get("DITTO_EMO", "4"))
    DITTO_SAMPLING_STEPS:    int = int(os.environ.get("DITTO_SAMPLING_STEPS", "50"))
    DITTO_JPEG_QUALITY:      int = int(os.environ.get("DITTO_JPEG_QUALITY", "80"))

    # Runtime
    DEVICE:     str = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE:      torch.dtype = torch.bfloat16
    UPLOAD_DIR: str = os.path.join(_ROOT, "_uploads")
    STATIC_DIR: str = os.path.join(_ROOT, "static")


cfg = Config()

# ─────────────────────────────────────────────────────────────────────────────
# Lazy-loaded model singletons
# ─────────────────────────────────────────────────────────────────────────────

_moshi_state:    Optional[StreamingMoshiState]     = None
_bridge_stream:  Optional[StreamingBridgeInference] = None

# Dedicated CUDA stream for Bridge inference to avoid implicit CUDA syncs
# with Moshi on the default stream.
_bridge_cuda_stream: Optional[torch.cuda.Stream] = None

# Per-session state: session_id → {"image_path": str, "ready": bool}
_sessions: dict = {}


def get_moshi() -> StreamingMoshiState:
    global _moshi_state
    if _moshi_state is None:
        raise RuntimeError("Moshi model not loaded. Call /startup or wait for server init.")
    return _moshi_state


def get_bridge() -> StreamingBridgeInference:
    global _bridge_stream
    if _bridge_stream is None:
        raise RuntimeError("Bridge model not loaded.")
    return _bridge_stream


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Moshi-Bridge-Ditto Streaming API", version="2.0.0")

# Serve static files (browser UI)
os.makedirs(cfg.STATIC_DIR, exist_ok=True)
os.makedirs(cfg.UPLOAD_DIR, exist_ok=True)

# Mount static dir
app.mount("/static", StaticFiles(directory=cfg.STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the browser UI."""
    index_path = os.path.join(cfg.STATIC_DIR, "index.html")
    if not os.path.isfile(index_path):
        return HTMLResponse("<h2>UI not found — place index.html in ./static/</h2>", status_code=404)
    with open(index_path, encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ─────────────────────────────────────────────────────────────────────────────
# Image upload
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload a portrait image.
    Returns {"session_id": "<uuid>", "filename": "<saved name>"}.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(status_code=400, detail=f"Unsupported image format: {ext}")

    session_id = str(uuid.uuid4())[:8]
    dest = os.path.join(cfg.UPLOAD_DIR, f"{session_id}{ext}")

    contents = await file.read()
    with open(dest, "wb") as f:
        f.write(contents)

    _sessions[session_id] = {"image_path": dest, "ready": True}
    logger.info(f"[upload_image] Session {session_id} → {dest}")
    return JSONResponse({"session_id": session_id, "filename": os.path.basename(dest)})


@app.get("/session/{session_id}/status")
async def session_status(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return _sessions[session_id]


# ─────────────────────────────────────────────────────────────────────────────
# Health / info
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    moshi = _moshi_state
    return {
        "status": "ok",
        "device": cfg.DEVICE,
        "moshi_loaded":   moshi is not None,
        "moshi_busy":     moshi is not None and moshi._lock.locked(),
        "bridge_loaded":  _bridge_stream is not None,
        "bridge_chunk":   cfg.BRIDGE_CHUNK,
        "bridge_flush_ms": cfg.BRIDGE_FLUSH_TIMEOUT_MS,
        "ditto_per_session": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Bridge inference helper (thread target)
# ─────────────────────────────────────────────────────────────────────────────

def _run_bridge_step(bridge: StreamingBridgeInference, chunk: torch.Tensor) -> np.ndarray:
    """
    Run one bridge inference step, optionally inside a dedicated CUDA stream.
    Returns (N, 1024) float32 numpy array.

    Runs in asyncio.to_thread() so the event loop stays responsive.
    """
    global _bridge_cuda_stream
    if cfg.DEVICE == "cuda" and _bridge_cuda_stream is not None:
        with torch.cuda.stream(_bridge_cuda_stream):
            features = bridge.step(chunk)
        # Ensure current stream sees the result before we return
        torch.cuda.current_stream().wait_stream(_bridge_cuda_stream)
    else:
        features = bridge.step(chunk)
    return features.numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"[WS] Client connected — session {session_id}")

    # ── Validate session ──────────────────────────────────────────────────────
    if session_id not in _sessions:
        await websocket.send_bytes(b"\xff" + b"Unknown session_id")
        await websocket.close()
        return

    session    = _sessions[session_id]
    image_path = session["image_path"]
    moshi      = get_moshi()
    bridge     = get_bridge()

    # ── Reject if Moshi is already busy (FIX: was silently queueing) ─────────
    if moshi._lock.locked():
        logger.warning(f"[WS] Rejecting session {session_id} — Moshi is busy")
        await websocket.send_bytes(
            b"\xff" + b"Server busy: another session is active. Try again shortly."
        )
        await websocket.close()
        return

    # ── Per-session Ditto adapter ─────────────────────────────────────────────
    ditto = DittoStreamAdapter(
        cfg_pkl      = cfg.DITTO_CFG_PKL,
        data_root    = cfg.DITTO_DATA_ROOT,
        jpeg_quality = cfg.DITTO_JPEG_QUALITY,
    )
    ditto.setup(
        image_path         = image_path,
        emo                = cfg.DITTO_EMO,
        sampling_timesteps = cfg.DITTO_SAMPLING_STEPS,
    )

    # ── Queues ────────────────────────────────────────────────────────────────
    # token_queue: Moshi → Bridge  (TaggedToken items or None sentinel)
    token_queue: asyncio.Queue = asyncio.Queue(maxsize=500)

    # frame_queue: Ditto thread → frame_forwarder (JPEG bytes or None sentinel)
    # Populated via loop.call_soon_threadsafe so it is always touched from the
    # event-loop thread — no run_coroutine_threadsafe futures needed.
    frame_queue: asyncio.Queue = asyncio.Queue(maxsize=300)

    # send_queue: the ONE queue that the ws_sender_task drains.
    # ALL outbound WebSocket messages go here (audio, video, text, handshake).
    # This guarantees only one coroutine ever calls websocket.send_bytes(),
    # eliminating the concurrent-drain AssertionError crash.
    send_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

    # Shared error signal: any task can set this to trigger coordinated shutdown
    error_event: asyncio.Event = asyncio.Event()

    # ── async receive wrapper ─────────────────────────────────────────────────
    async def receive_fn():
        try:
            data = await websocket.receive_bytes()
            return data
        except (WebSocketDisconnect, Exception):
            return None

    # ══════════════════════════════════════════════════════════════════════════
    # ws_sender_task — THE ONLY TASK THAT CALLS websocket.send_bytes()
    # ══════════════════════════════════════════════════════════════════════════
    async def ws_sender_task():
        """
        Single serialised WebSocket writer.

        All other tasks put bytes into send_queue; this task drains it.
        This eliminates the concurrent-drain AssertionError caused by
        audio, video, and text coroutines all calling send_bytes() at once.

        Termination: a None sentinel in send_queue stops this task.
        """
        try:
            while True:
                msg = await send_queue.get()
                if msg is None:
                    break
                try:
                    await websocket.send_bytes(msg)
                except (WebSocketDisconnect, Exception) as exc:
                    logger.debug(f"[ws_sender] Send failed ({type(exc).__name__}) — stopping")
                    error_event.set()
                    break
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception(f"[ws_sender] Unexpected error: {exc}")
            error_event.set()
        logger.debug("[ws_sender] Exited.")

    # ── Bridge task: token_queue → features → Ditto ──────────────────────────
    async def bridge_task():
        """
        Pull TaggedToken items from token_queue, batch them, run bridge
        inference, and push features into Ditto.

        Adaptive batching: flush to Ditto when EITHER:
          (a) token_buffer reaches BRIDGE_CHUNK tokens, OR
          (b) BRIDGE_FLUSH_TIMEOUT_MS has elapsed since the first token arrived.
        This eliminates the 320ms+ stall during pauses.
        """
        bridge.reset()
        token_buffer: list = []
        chunk_size    = cfg.BRIDGE_CHUNK
        flush_timeout = cfg.BRIDGE_FLUSH_TIMEOUT_MS / 1000.0
        first_token_time: Optional[float] = None

        async def _flush():
            nonlocal token_buffer, first_token_time
            if not token_buffer:
                return
            chunk_tensor = torch.cat([t.tensor for t in token_buffer], dim=0)
            batch_seq    = token_buffer[0].seq
            token_buffer = []
            first_token_time = None
            try:
                features_np = await asyncio.to_thread(
                    _run_bridge_step, bridge, chunk_tensor
                )
                ditto.push_features(features_np, seq=batch_seq)
                logger.debug(
                    f"[bridge_task] Flushed {chunk_tensor.shape[0]} tokens "
                    f"seq={batch_seq} → {features_np.shape}"
                )
            except Exception as exc:
                logger.error(f"[bridge_task] Bridge step error: {exc}")
                error_event.set()

        try:
            while True:
                if first_token_time is not None:
                    elapsed = time.monotonic() - first_token_time
                    wait_time = max(0.0, flush_timeout - elapsed)
                else:
                    wait_time = flush_timeout

                try:
                    item = await asyncio.wait_for(token_queue.get(), timeout=wait_time)
                except asyncio.TimeoutError:
                    if token_buffer:
                        await _flush()
                    continue

                if item is None:
                    await _flush()
                    break

                if error_event.is_set():
                    break

                token_buffer.append(item)
                if first_token_time is None:
                    first_token_time = time.monotonic()

                if len(token_buffer) >= chunk_size:
                    await _flush()

        except Exception as exc:
            logger.exception(f"[bridge_task] Unhandled error: {exc}")
            error_event.set()
        finally:
            ditto.close()
            logger.info("[bridge_task] Done.")

    # ── Frame reader task: Ditto thread → frame_queue ────────────────────────
    async def frame_reader_task():
        """
        Reads JPEG bytes from Ditto's blocking iter_frames() inside
        asyncio.to_thread(), then pushes them into frame_queue using
        loop.call_soon_threadsafe() + put_nowait() — no Future overhead,
        no blocking the reader thread waiting for the event loop.

        Frames are DROPPED (not blocked) when frame_queue is full.
        """
        loop = asyncio.get_running_loop()

        def _safe_enqueue(item):
            """Called on the event-loop thread via call_soon_threadsafe."""
            try:
                frame_queue.put_nowait(item)
            except asyncio.QueueFull:
                if item is not None:
                    logger.warning("[frame_reader_task] frame_queue full — frame dropped")
                else:
                    # Must deliver sentinel even if queue is full; retry once
                    # by clearing one slot.
                    try:
                        frame_queue.get_nowait()   # drop oldest frame
                    except asyncio.QueueEmpty:
                        pass
                    frame_queue.put_nowait(None)

        def _blocking_iter():
            for jpeg in ditto.iter_frames():
                if error_event.is_set():
                    break
                loop.call_soon_threadsafe(_safe_enqueue, jpeg)
            # Always deliver sentinel so frame_forwarder_task can exit
            loop.call_soon_threadsafe(_safe_enqueue, None)

        try:
            await asyncio.to_thread(_blocking_iter)
        except Exception as exc:
            logger.exception(f"[frame_reader_task] Error: {exc}")
            error_event.set()
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(_safe_enqueue, None)

    # ── Frame forwarder task: frame_queue → send_queue ───────────────────────
    async def frame_forwarder_task():
        """
        Drains frame_queue and puts JPEG frames into send_queue for the
        ws_sender_task to transmit.  Separate from the WebSocket write so
        frame buffering never blocks the sender.

        Wire format: 0x02 | <frame_seq: 4 bytes BE uint32> | <jpeg bytes>
        """
        frame_count = 0
        seq_counter = 0
        try:
            while True:
                jpeg = await frame_queue.get()
                if jpeg is None or error_event.is_set():
                    break
                frame_count += 1
                hdr = b"\x02" + struct.pack(">I", seq_counter & 0xFFFF_FFFF)
                seq_counter += 1
                # put_nowait: if send_queue is full drop frame rather than block
                try:
                    send_queue.put_nowait(hdr + jpeg)
                except asyncio.QueueFull:
                    logger.warning("[frame_forwarder] send_queue full — frame dropped")
        except Exception as exc:
            logger.exception(f"[frame_forwarder_task] Error: {exc}")
            error_event.set()
        logger.info(f"[frame_forwarder_task] Done. Forwarded {frame_count} frames.")

    # ── Start background tasks ────────────────────────────────────────────────
    t_ws_sender    = asyncio.create_task(ws_sender_task(),       name="ws_sender")
    t_bridge       = asyncio.create_task(bridge_task(),          name="bridge_task")
    t_frame_reader = asyncio.create_task(frame_reader_task(),    name="frame_reader_task")
    t_frame_fwd    = asyncio.create_task(frame_forwarder_task(), name="frame_forwarder_task")

    # ── Moshi main loop (drives audio I/O + token capture) ───────────────────
    try:
        async for kind, payload in moshi.handle_connection(receive_fn, token_queue):
            if error_event.is_set():
                logger.warning("[WS] Error event set — stopping Moshi loop")
                break

            if kind == "handshake":
                await send_queue.put(b"\x00")

            elif kind == "audio":
                # payload = (seq, raw_float32_pcm_bytes)
                moshi_seq, pcm_bytes = payload
                hdr = b"\x01" + seq_pack(moshi_seq)
                try:
                    send_queue.put_nowait(hdr + pcm_bytes)
                except asyncio.QueueFull:
                    pass  # drop audio frame rather than block Moshi loop

            elif kind == "text":
                try:
                    send_queue.put_nowait(b"\x03" + payload.encode("utf-8"))
                except asyncio.QueueFull:
                    pass

    except WebSocketDisconnect:
        logger.info(f"[WS] Client disconnected — session {session_id}")
    except RuntimeError as exc:
        logger.warning(f"[WS] {exc}")
        try:
            await send_queue.put(b"\xff" + str(exc).encode())
        except Exception:
            pass
    except Exception as exc:
        logger.exception(f"[WS] Unexpected error in session {session_id}: {exc}")
        try:
            await send_queue.put(b"\xff" + str(exc).encode())
        except Exception:
            pass
    finally:
        error_event.set()

        # Stop the ws_sender cleanly by putting the sentinel
        try:
            send_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

        # Wait for all tasks to finish (10s hard deadline)
        all_tasks = (t_ws_sender, t_bridge, t_frame_reader, t_frame_fwd)
        try:
            await asyncio.wait_for(
                asyncio.gather(*all_tasks, return_exceptions=True),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            logger.warning(f"[WS] Session {session_id} cleanup timed out — cancelling tasks")
            for t in all_tasks:
                t.cancel()
            await asyncio.gather(*all_tasks, return_exceptions=True)

        try:
            await websocket.close()
        except Exception:
            pass
        logger.info(f"[WS] Session {session_id} fully closed.")


# ─────────────────────────────────────────────────────────────────────────────
# Startup event: load all models once
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def load_models():
    global _moshi_state, _bridge_stream, _bridge_cuda_stream

    logger.info("=" * 60)
    logger.info("  Moshi + Bridge + Ditto — Streaming Server Starting  v2")
    logger.info("=" * 60)
    logger.info(f"  Device : {cfg.DEVICE}")

    # ── Moshi ────────────────────────────────────────────────────────────────
    logger.info("\n[1/2] Loading Moshi …")
    t0 = time.time()
    _moshi_state = StreamingMoshiState(
        hf_repo      = cfg.MOSHI_HF_REPO,
        moshi_weight = cfg.MOSHI_WEIGHT,
        mimi_weight  = cfg.MIMI_WEIGHT,
        tokenizer    = cfg.TOKENIZER,
        device       = cfg.DEVICE,
        dtype        = cfg.DTYPE,
    )
    _moshi_state.warmup()
    logger.info(f"[1/2] ✅ Moshi ready ({time.time()-t0:.1f}s)")

    # ── Bridge ───────────────────────────────────────────────────────────────
    logger.info("\n[2/2] Loading Bridge …")
    t0 = time.time()
    _bridge_stream = StreamingBridgeInference(
        checkpoint_path = cfg.BRIDGE_CKPT,
        config_path     = cfg.BRIDGE_CONFIG,
        chunk_size      = cfg.BRIDGE_CHUNK,
        device          = cfg.DEVICE,
    )
    if cfg.DEVICE == "cuda":
        _bridge_cuda_stream = torch.cuda.Stream()
        logger.info("[2/2]   CUDA stream created for Bridge isolation")
    logger.info(f"[2/2] ✅ Bridge ready ({time.time()-t0:.1f}s)")

    # ── Ditto ────────────────────────────────────────────────────────────────
    logger.info("\n[3/3] Ditto: adapter created per-session (image-specific).")
    logger.info(f"       TRT models: {cfg.DITTO_DATA_ROOT}")
    logger.info(f"       Adaptive bridge flush: {cfg.BRIDGE_FLUSH_TIMEOUT_MS}ms")

    logger.info("\n" + "=" * 60)
    logger.info("  ✅  All models loaded. Server ready for connections.")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Moshi + Bridge + Ditto — Real-Time Streaming Server v2"
    )
    p.add_argument("--host",      default="0.0.0.0")
    p.add_argument("--port",      type=int, default=7860)
    p.add_argument("--log-level", default="info",
                   choices=["debug", "info", "warning", "error"])

    # Model path overrides (all have env-var equivalents, see Config above)
    p.add_argument("--hf-repo",         default=None)
    p.add_argument("--moshi-weight",    default=None)
    p.add_argument("--mimi-weight",     default=None)
    p.add_argument("--tokenizer",       default=None)
    p.add_argument("--bridge-ckpt",     default=None)
    p.add_argument("--bridge-config",   default=None)
    p.add_argument("--bridge-chunk",    type=int, default=None)
    p.add_argument("--bridge-flush-ms", type=int, default=None,
                   help="Adaptive bridge flush timeout in milliseconds (default 100)")
    p.add_argument("--ditto-data-root", default=None)
    p.add_argument("--ditto-cfg-pkl",   default=None)
    p.add_argument("--ditto-emo",            type=int, default=None)
    p.add_argument("--ditto-sampling-steps", type=int, default=None)
    p.add_argument("--jpeg-quality",    type=int, default=None)
    p.add_argument("--half", action="store_const",
                   const=torch.float16, default=None, dest="dtype",
                   help="Use float16 instead of bfloat16")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Apply CLI overrides to Config
    if args.hf_repo:         cfg.MOSHI_HF_REPO    = args.hf_repo
    if args.moshi_weight:    cfg.MOSHI_WEIGHT      = args.moshi_weight
    if args.mimi_weight:     cfg.MIMI_WEIGHT       = args.mimi_weight
    if args.tokenizer:       cfg.TOKENIZER         = args.tokenizer
    if args.bridge_ckpt:     cfg.BRIDGE_CKPT       = args.bridge_ckpt
    if args.bridge_config:   cfg.BRIDGE_CONFIG     = args.bridge_config
    if args.bridge_chunk:    cfg.BRIDGE_CHUNK      = args.bridge_chunk
    if args.bridge_flush_ms: cfg.BRIDGE_FLUSH_TIMEOUT_MS = args.bridge_flush_ms
    if args.ditto_data_root: cfg.DITTO_DATA_ROOT   = args.ditto_data_root
    if args.ditto_cfg_pkl:   cfg.DITTO_CFG_PKL     = args.ditto_cfg_pkl
    if args.ditto_emo:            cfg.DITTO_EMO            = args.ditto_emo
    if args.ditto_sampling_steps: cfg.DITTO_SAMPLING_STEPS = args.ditto_sampling_steps
    if args.jpeg_quality:    cfg.DITTO_JPEG_QUALITY = args.jpeg_quality
    if args.dtype:           cfg.DTYPE             = args.dtype

    uvicorn.run(
        "streaming_server:app",
        host      = args.host,
        port      = args.port,
        log_level = args.log_level,
        loop      = "asyncio",
    )
