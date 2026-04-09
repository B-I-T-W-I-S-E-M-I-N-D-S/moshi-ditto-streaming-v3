import librosa
import math
import os
import numpy as np
import random
import torch
import pickle

from stream_pipeline_offline import StreamSDK


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pkl(pkl):
    with open(pkl, "rb") as f:
        return pickle.load(f)


def run(
    SDK: StreamSDK,
    source_path: str,
    output_path: str,
    audio_path: str | None = None,
    audio_feat_path: str | None = None,
    more_kwargs: str | dict = {},
):
    """
    Args:
        source_path     : path to input image or video
        output_path     : path to output .mp4
        audio_path      : path to input .wav  (used for HuBERT extraction AND
                          for muxing audio into the final video)
        audio_feat_path : path to pre-computed .npy features  shape (N, 1024)
                          When provided, HuBERT extraction is skipped entirely.
                          audio_path is still used for the final audio mux if given.
    """
    if audio_path is None and audio_feat_path is None:
        raise ValueError("Provide at least one of --audio_path or --audio_feat_path")

    if isinstance(more_kwargs, str):
        more_kwargs = load_pkl(more_kwargs)
    setup_kwargs = more_kwargs.get("setup_kwargs", {})
    run_kwargs = more_kwargs.get("run_kwargs", {})

    SDK.setup(source_path, output_path, **setup_kwargs)

    # ── Decide where features come from ──────────────────────────────────────
    if audio_feat_path is not None:
        # ── Path A: load pre-computed .npy (skip HuBERT) ─────────────────────
        print(f"[run] Loading pre-computed audio features: {audio_feat_path}")
        aud_feat = np.load(audio_feat_path).astype(np.float32)   # (N, 1024)
        assert aud_feat.ndim == 2 and aud_feat.shape[1] == 1024, (
            f"Expected shape (N, 1024), got {aud_feat.shape}"
        )
        num_f = len(aud_feat)
        print(f"[run] Features loaded — shape: {aud_feat.shape}  ({num_f} frames @ 25 FPS)")
    else:
        # ── Path B: load wav and run HuBERT (original behaviour) ─────────────
        print(f"[run] Extracting HuBERT features from: {audio_path}")
        audio, sr = librosa.core.load(audio_path, sr=16000)
        num_f = math.ceil(len(audio) / 16000 * 25)

    fade_in = run_kwargs.get("fade_in", -1)
    fade_out = run_kwargs.get("fade_out", -1)
    ctrl_info = run_kwargs.get("ctrl_info", {})
    SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)

    online_mode = SDK.online_mode

    if audio_feat_path is not None:
        # Path A: features already ready — put directly in queue (offline only)
        assert not online_mode, "online_mode is not supported when using --audio_feat_path"
        SDK.audio2motion_queue.put(aud_feat)
    elif online_mode:
        # Path B online
        chunksize = run_kwargs.get("chunksize", (3, 5, 2))
        audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80  # 6480
        for i in range(0, len(audio), chunksize[1] * 640):
            audio_chunk = audio[i:i + split_len]
            if len(audio_chunk) < split_len:
                audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
            SDK.run_chunk(audio_chunk, chunksize)
    else:
        # Path B offline
        aud_feat = SDK.wav2feat.wav2feat(audio)
        SDK.audio2motion_queue.put(aud_feat)

    SDK.close()

    # ── Mux audio into video ──────────────────────────────────────────────────
    if audio_path is not None:
        cmd = (
            f'ffmpeg -loglevel error -y'
            f' -i "{SDK.tmp_output_path}"'
            f' -i "{audio_path}"'
            f' -map 0:v -map 1:a -c:v copy -c:a aac'
            f' "{output_path}"'
        )
        print(cmd)
        os.system(cmd)
    else:
        # No audio source: just rename the silent tmp video
        import shutil
        shutil.move(SDK.tmp_output_path, output_path)
        print(f"[run] No audio_path given — output is video-only: {output_path}")

    print(output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Ditto talking-head inference",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--data_root", type=str, default="./checkpoints/ditto_trt_Ampere_Plus",
                        help="path to model data_root")
    parser.add_argument("--cfg_pkl",   type=str, default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl",
                        help="path to cfg_pkl")

    parser.add_argument("--source_path",     type=str, required=True,
                        help="path to input image or video")
    parser.add_argument("--output_path",     type=str, required=True,
                        help="path to output .mp4")

    # ── Audio input: provide ONE of these two ─────────────────────────────────
    audio_group = parser.add_mutually_exclusive_group()
    audio_group.add_argument("--audio_path",      type=str, default=None,
                             help="path to input .wav  (HuBERT extraction runs as normal)")
    audio_group.add_argument("--audio_feat_path", type=str, default=None,
                             help="path to pre-computed .npy features  shape (N, 1024)\n"
                                  "Skips HuBERT entirely — features go straight into the diffusion model.\n"
                                  "Tip: also pass --audio_path_for_mux if you want audio in the output video.")

    # Optional: mux a wav into the video even when --audio_feat_path is used
    parser.add_argument("--audio_path_for_mux", type=str, default=None,
                        help="(optional) .wav to mux into the output video when --audio_feat_path is used")

    args = parser.parse_args()

    if args.audio_path is None and args.audio_feat_path is None:
        parser.error("Provide either --audio_path or --audio_feat_path")

    # ── Init SDK ──────────────────────────────────────────────────────────────
    SDK = StreamSDK(args.cfg_pkl, args.data_root)

    # ── Resolve audio_path for muxing ─────────────────────────────────────────
    # When using --audio_feat_path the user may optionally pass --audio_path_for_mux
    # to still get the audio track in the final video.
    mux_audio = args.audio_path or args.audio_path_for_mux

    # run
    # seed_everything(1024)
    run(
        SDK,
        source_path=args.source_path,
        output_path=args.output_path,
        audio_path=mux_audio,
        audio_feat_path=args.audio_feat_path,
    )
