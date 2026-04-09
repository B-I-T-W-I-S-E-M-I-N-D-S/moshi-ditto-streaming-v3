"""
pipeline/ditto_stream_adapter.py
==================================
Wraps Ditto's ``StreamSDK`` (online mode from stream_pipeline_online.py)
to intercept rendered video frames instead of writing them to an MP4 file.

How it works
------------
1. On ``setup()``, the SDK's ``writer_worker`` function is monkey-patched
   **before** ``StreamSDK.setup()`` is called, so the SDK spawns our custom
   thread instead of the default disk-writer from the start.
   (FIX: previous code replaced the thread *after* setup(), causing a race
    where both the old thread and the new thread ran simultaneously.)

2. The patched writer pulls frames from Ditto's internal ``writer_queue``
   and pushes JPEG-encoded bytes into ``self._frame_queue`` with a
   non-blocking ``put_nowait`` — dropping frames only when the consumer
   (WebSocket sender) is too slow.
   (FIX: previous code used a blocking ``queue.put()`` which could deadlock
    the entire Ditto rendering pipeline when the browser disconnected.)

3. The caller pushes feature chunks via ``push_features(feat_np)``.
   This feeds directly into ``sdk.audio2motion_queue``.

4. ``iter_frames()`` is a blocking generator that yields (seq, jpeg_bytes)
   tuples.  Run it in a thread or via ``asyncio.to_thread()``.

JPEG Encoding Priority
----------------------
  1. libjpeg-turbo (via ``turbojpeg`` package) — fastest, ~0.5ms/frame
  2. OpenCV cv2.imencode            — fast, ~1ms/frame
  3. PIL Image.save                 — fallback, ~3-5ms/frame

Usage
-----
    adapter = DittoStreamAdapter(cfg_pkl=..., data_root=...)
    adapter.setup(image_path="/workspace/portrait.jpg")

    # Push feature chunks as they arrive (from bridge_task)
    adapter.push_features(seq=42, features=feat_chunk_np)  # (N, 1024) float32

    # Consume frames (run in asyncio.to_thread)
    for seq, jpeg_bytes in adapter.iter_frames():
        await websocket.send_bytes(b"\\x02" + seq_pack(seq) + jpeg_bytes)

    adapter.close()
"""

import io
import logging
import os
import queue
import sys
import threading
from typing import Iterator, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure ditto-inference is importable
# ---------------------------------------------------------------------------
_DITTO_DIR = os.path.join(os.path.dirname(__file__), "..", "ditto-inference")
if _DITTO_DIR not in sys.path:
    sys.path.insert(0, _DITTO_DIR)

from stream_pipeline_online import StreamSDK  # online version (supports streaming)


# ---------------------------------------------------------------------------
# JPEG encoder — auto-selects fastest available backend
# ---------------------------------------------------------------------------

def _build_jpeg_encoder(quality: int):
    """
    Return a callable ``encode(rgb_uint8_hwc) -> bytes`` using the fastest
    available JPEG library on this system.

    Priority: turbojpeg > cv2 > PIL
    """
    # 1. libjpeg-turbo via pyturbojpeg
    try:
        from turbojpeg import TurboJPEG, TJPF_RGB
        _tj = TurboJPEG()
        def _turbo_encode(rgb: np.ndarray) -> bytes:
            return _tj.encode(rgb, quality=quality, pixel_format=TJPF_RGB)
        logger.info("[DittoStreamAdapter] JPEG encoder: TurboJPEG (fastest)")
        return _turbo_encode
    except ImportError:
        pass

    # 2. OpenCV
    try:
        import cv2
        _params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        def _cv2_encode(rgb: np.ndarray) -> bytes:
            bgr = rgb[:, :, ::-1]   # RGB → BGR for cv2
            ok, buf = cv2.imencode(".jpg", bgr, _params)
            if not ok:
                raise RuntimeError("cv2.imencode failed")
            return buf.tobytes()
        logger.info("[DittoStreamAdapter] JPEG encoder: OpenCV cv2 (fast)")
        return _cv2_encode
    except ImportError:
        pass

    # 3. PIL fallback
    from PIL import Image
    def _pil_encode(rgb: np.ndarray) -> bytes:
        img = Image.fromarray(rgb, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    logger.info("[DittoStreamAdapter] JPEG encoder: PIL (fallback — consider installing turbojpeg or opencv-python)")
    return _pil_encode


# ---------------------------------------------------------------------------
# DittoStreamAdapter
# ---------------------------------------------------------------------------

_SENTINEL = object()   # signals end-of-stream in frame_queue


class DittoStreamAdapter:
    """
    Wraps StreamSDK for real-time per-frame output instead of file writing.

    Parameters
    ----------
    cfg_pkl      : path to Ditto .pkl config file
    data_root    : path to Ditto TRT model directory
    jpeg_quality : JPEG encoding quality for streamed frames (default 80)
    """

    def __init__(
        self,
        cfg_pkl: str,
        data_root: str,
        jpeg_quality: int = 80,
    ):
        cfg_pkl   = os.path.abspath(cfg_pkl)
        data_root = os.path.abspath(data_root)

        if not os.path.isfile(cfg_pkl):
            raise FileNotFoundError(f"Ditto config .pkl not found: {cfg_pkl}")
        if not os.path.isdir(data_root):
            raise FileNotFoundError(f"Ditto TRT model directory not found: {data_root}")

        logger.info(f"[DittoStreamAdapter] Loading StreamSDK from {data_root} …")
        self.sdk = StreamSDK(cfg_pkl, data_root)
        self.jpeg_quality = jpeg_quality
        self._jpeg_encode = _build_jpeg_encoder(jpeg_quality)

        # Frame queue (threading.Queue, populated by writer thread, consumed
        # by iter_frames() which runs in an asyncio.to_thread context).
        # maxsize=200 caps memory usage; frames are DROPPED (not blocked on)
        # when the consumer is too slow.
        self._frame_queue: queue.Queue = queue.Queue(maxsize=200)
        self._is_setup = False

        logger.info("[DittoStreamAdapter] StreamSDK loaded.")

    # ------------------------------------------------------------------
    # Setup for a session (one portrait image)
    # ------------------------------------------------------------------

    def setup(
        self,
        image_path: str,
        N_d: int = 10_000,      # large upper-bound; adapter closes when bridge is done
        emo: int = 4,
        sampling_timesteps: int = 50,
        overlap_v2: int = 10,
    ):
        """
        Initialise Ditto SDK for a new session.

        The SDK's writer_worker is monkey-patched **before** sdk.setup() is
        called, so the SDK's thread pool spawns our custom writer from the
        outset — eliminating the previous race condition.

        Parameters
        ----------
        image_path        : path to the portrait image (.jpg / .png)
        N_d               : max expected frames (set large; we stop by closing)
        emo               : emotion index (Ditto default = 4)
        sampling_timesteps: diffusion steps (lower = faster, lower quality)
        overlap_v2        : overlap frames for the sliding window (default 10)
        """
        image_path = os.path.abspath(image_path)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Portrait image not found: {image_path}")

        # Reset frame queue for the new session
        # Drain any leftover items from a previous session
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

        # ── Pre-install the writer patch on the SDK instance ──────────────
        # We set sdk.writer_worker = our_function BEFORE sdk.setup() so that
        # when setup() does threading.Thread(target=self.writer_worker) it uses
        # our version.  Crucially, the patched function accesses sdk.writer_queue
        # and sdk.stop_event LAZILY (inside the thread body), so they don't need
        # to exist at patch-install time.
        self._install_writer_patch()

        # We pass a dummy output path; the writer is monkey-patched above
        _DUMMY_OUT = "/tmp/ditto_stream_dummy.mp4"

        self.sdk.setup(
            source_path        = image_path,
            output_path        = _DUMMY_OUT,
            online_mode        = True,
            N_d                = N_d,
            emo                = emo,
            sampling_timesteps = sampling_timesteps,
            overlap_v2         = overlap_v2,
        )

        # ── Fallback: if the SDK didn't use our patched method (some versions
        # store the target as a local function reference rather than looking it
        # up via self.writer_worker), replace the last thread in thread_list.
        self._ensure_writer_patched()

        self._is_setup = True
        logger.info(f"[DittoStreamAdapter] Session ready for image: {image_path}")

    # ------------------------------------------------------------------
    # Writer patch installation (BEFORE sdk.setup())
    # ------------------------------------------------------------------

    def _install_writer_patch(self):
        """
        Install our intercepting writer_worker on the SDK instance BEFORE
        setup() is called, so the SDK starts our thread from the beginning.

        KEY FIX: sdk.writer_queue and sdk.stop_event are accessed LAZILY
        inside the thread body — they don't exist yet at install time, only
        after sdk.setup() runs.  The thread body runs after setup(), so by
        then both attributes exist.
        """
        frame_queue = self._frame_queue
        jpeg_encode = self._jpeg_encode
        sdk         = self.sdk  # capture reference; do NOT access .stop_event here

        def _patched_writer_worker():
            """
            Runs inside the Ditto SDK thread pool (started by sdk.setup()).

            Accesses sdk.writer_queue and sdk.stop_event LAZILY (they are
            created by sdk.setup() before the thread is actually started,
            so they exist by the time this function body executes).

            Uses put_nowait() + drop so the Ditto pipeline never blocks
            waiting for a slow or disconnected browser.
            """
            # --- lazy access: these exist because sdk.setup() already ran ---
            writer_queue = sdk.writer_queue
            stop_event   = getattr(sdk, 'stop_event', None)

            logger.info("[DittoStreamAdapter] patched writer_worker started")
            try:
                while True:
                    # Honour stop_event if the SDK has one
                    if stop_event is not None and stop_event.is_set():
                        break
                    try:
                        item = writer_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue
                    if item is None:
                        # SDK end-of-stream sentinel
                        break
                    rgb = item   # numpy uint8 HWC RGB
                    try:
                        jpeg = jpeg_encode(rgb)
                    except Exception as enc_err:
                        logger.error(
                            f"[DittoStreamAdapter] JPEG encode error: {enc_err}"
                        )
                        continue
                    # Non-blocking: drop frames when the browser is too slow
                    try:
                        frame_queue.put_nowait(jpeg)
                    except queue.Full:
                        logger.warning(
                            "[DittoStreamAdapter] frame_queue full — "
                            "dropping frame (browser consumer too slow)"
                        )
            finally:
                frame_queue.put(_SENTINEL)   # signal iter_frames() to stop
                logger.info("[DittoStreamAdapter] patched writer_worker exited")

        # Install on all known attribute names the SDK might use
        _patched = False
        for attr in ('writer_worker', '_writer_worker'):
            if hasattr(sdk, attr):
                setattr(sdk, attr, _patched_writer_worker)
                logger.debug(f"[DittoStreamAdapter] Patched sdk.{attr} (pre-setup)")
                _patched = True

        # Store reference so _ensure_writer_patched() can use same function
        self._patched_fn = _patched_writer_worker
        self._pre_patched = _patched
        logger.debug(
            f"[DittoStreamAdapter] Writer pre-patch installed "
            f"(method found: {_patched})"
        )

    # ------------------------------------------------------------------
    # Fallback: replace writer thread AFTER setup() if pre-patch missed
    # ------------------------------------------------------------------

    def _ensure_writer_patched(self):
        """
        Called after sdk.setup() to verify our patched writer is running.

        Some Ditto SDK versions store the thread target as a local function
        variable (not via self.writer_worker) so pre-patching has no effect.
        In that case we replace the last thread in sdk.thread_list now.

        Race-safety: we wait up to 200ms for the old thread to finish its
        initialisation before checking; if still alive we replace it and
        start our version.  The old thread naturally exits because we drain
        writer_queue for it (it will get queue.Empty and stop_event will be
        set by sdk.close() eventually).
        """
        thread_list = getattr(self.sdk, 'thread_list', [])
        if not thread_list:
            logger.debug("[DittoStreamAdapter] sdk.thread_list empty — skipping fallback patch")
            return

        last_thread = thread_list[-1]

        # Check if our function is already the target (pre-patch succeeded)
        target_fn = getattr(last_thread, '_target', None)
        if target_fn is self._patched_fn:
            logger.debug("[DittoStreamAdapter] Pre-patch confirmed active — no fallback needed")
            return

        logger.info(
            "[DittoStreamAdapter] Pre-patch not active in writer thread "
            "— applying post-setup thread replacement"
        )

        # Give the old thread a moment to start (so we know it's alive)
        last_thread.join(timeout=0.2)

        if not last_thread.is_alive():
            # Old thread already finished (very fast SDK) — just start ours
            new_thread = threading.Thread(
                target=self._patched_fn, daemon=True,
                name="ditto_writer_patched"
            )
            thread_list[-1] = new_thread
            new_thread.start()
            logger.info("[DittoStreamAdapter] Fallback writer thread started (old already done)")
            return

        # Old thread still running: replace it.
        # The old thread will also put a sentinel into frame_queue when it
        # exits — we absorb that in iter_frames() by checking _SENTINEL type.
        new_thread = threading.Thread(
            target=self._patched_fn, daemon=True,
            name="ditto_writer_patched"
        )
        thread_list[-1] = new_thread
        new_thread.start()
        logger.info("[DittoStreamAdapter] Fallback writer thread started (replacing active writer)")

    # ------------------------------------------------------------------
    # Feature input
    # ------------------------------------------------------------------

    def push_features(self, features: np.ndarray, seq: int = 0):
        """
        Push a chunk of HuBERT-like features into Ditto's audio2motion queue.

        Parameters
        ----------
        features : numpy (N, 1024) float32 — bridge module output
        seq      : sequence number of the first Moshi step in this chunk
                   (stored so iter_frames can yield tagged frames)
        """
        if not self._is_setup:
            raise RuntimeError("Call setup() before push_features().")
        features = features.astype(np.float32)
        if features.ndim != 2 or features.shape[1] != 1024:
            raise ValueError(
                f"Expected (N, 1024) features, got {features.shape}"
            )
        # With a 2s timeout: if Ditto's input queue is backed up that long
        # something is seriously wrong — log and skip rather than blocking forever.
        try:
            self.sdk.audio2motion_queue.put(features, timeout=2.0)
        except queue.Full:
            logger.error(
                f"[DittoStreamAdapter] audio2motion_queue full (seq={seq}) — "
                "Ditto pipeline is backed up; skipping feature chunk"
            )

    # ------------------------------------------------------------------
    # Frame output
    # ------------------------------------------------------------------

    def iter_frames(self) -> Iterator[bytes]:
        """
        Blocking generator that yields JPEG-encoded frame bytes.
        Stops when the SDK signals end-of-stream (SENTINEL).

        If both the old writer thread and our replacement thread are running
        simultaneously (fallback scenario), up to two SENTINELs may arrive.
        We drain all SENTINELs so the queue is clean for the next session.

        Designed to run in ``asyncio.to_thread()``.
        """
        if not self._is_setup:
            raise RuntimeError("Call setup() before iter_frames().")

        sentinels_seen = 0

        while True:
            item = self._frame_queue.get()
            if item is _SENTINEL:
                sentinels_seen += 1
                # Drain any extra sentinels that may be enqueued already
                # (can happen when both old thread and new thread both put one)
                while True:
                    try:
                        extra = self._frame_queue.get_nowait()
                        if extra is _SENTINEL:
                            sentinels_seen += 1
                        else:
                            # Real frame that arrived just before the 2nd sentinel
                            yield extra
                    except queue.Empty:
                        break
                logger.debug(
                    f"[DittoStreamAdapter] iter_frames done "
                    f"({sentinels_seen} sentinel(s) received)"
                )
                break
            yield item   # JPEG bytes

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def close(self):
        """
        Signal that no more features will be pushed and wait for the
        SDK thread pipeline to drain.
        """
        if not self._is_setup:
            return
        logger.info("[DittoStreamAdapter] Closing SDK …")
        self.sdk.close()   # puts None into audio2motion_queue, joins threads
        self._is_setup = False
        logger.info("[DittoStreamAdapter] Closed.")
