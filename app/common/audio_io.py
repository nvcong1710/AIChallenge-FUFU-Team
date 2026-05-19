"""Audio I/O utilities — load audio file (kể cả video) thành PCM mono 16kHz numpy."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def load_audio_mono_16k(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Load audio (từ bất kỳ file audio hoặc video) thành float32 PCM mono.

    Dùng ffmpeg subprocess để robust hơn librosa (không cần libsndfile cho mọi codec).
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vn",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def audio_duration(path: str | Path) -> float:
    """Trả về duration (giây) của file audio/video qua ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        return float(out) if out else 0.0
    except Exception:
        return 0.0
