"""
pipeline/merge_audio_video.py
==============================
FFmpeg utility: mux a WAV audio track into a silent MP4 video to produce
the final talking-head video with Moshi-generated audio.

Usage
-----
    from pipeline.merge_audio_video import merge_audio_into_video

    output = merge_audio_into_video(
        video_path  = "output_silent.mp4",   # Ditto's output (no audio)
        audio_path  = "moshi_audio.wav",     # Moshi's output WAV (24kHz)
        output_path = "final_output.mp4",    # merged result
    )
"""

import os
import subprocess
import shutil


def merge_audio_into_video(
    video_path: str,
    audio_path: str,
    output_path: str,
    overwrite: bool = True,
) -> str:
    """
    Mux ``audio_path`` into ``video_path`` to produce ``output_path``.

    The video stream is copied without re-encoding; the audio is encoded as AAC.
    If the Moshi audio is longer than the video, the audio is trimmed to match.
    If the video is longer, the audio stream just ends (video continues silent).

    Parameters
    ----------
    video_path  : path to the silent .mp4 from Ditto
    audio_path  : path to the .wav from Moshi (24kHz mono)
    output_path : destination .mp4 with audio
    overwrite   : if True, overwrite output_path if it exists

    Returns
    -------
    str : ``output_path``

    Raises
    ------
    RuntimeError  : if FFmpeg is not found or returns a non-zero exit code
    FileNotFoundError : if inputs do not exist
    """

    video_path  = os.path.abspath(video_path)
    audio_path  = os.path.abspath(audio_path)
    output_path = os.path.abspath(output_path)

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Silent video not found: {video_path}")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Moshi audio not found: {audio_path}")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found in PATH. "
            "Install it with: apt-get install ffmpeg  (or conda install ffmpeg)"
        )

    overwrite_flag = "-y" if overwrite else "-n"
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        overwrite_flag,
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v",        # video stream from silent video
        "-map", "1:a",        # audio stream from Moshi WAV
        "-c:v", "copy",       # copy video codec (no re-encode)
        "-c:a", "aac",        # encode audio as AAC
        "-shortest",          # trim to shortest stream
        output_path,
    ]

    print(f"[merge] FFmpeg: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed (code {result.returncode}):\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    print(f"[merge] Final video with audio → {output_path}")
    return output_path
