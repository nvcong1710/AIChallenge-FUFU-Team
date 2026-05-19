"""Trích keyframe từ video — fixed count hoặc adaptive theo độ dài shot."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def get_video_duration(video_path: str | Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return float(frames / fps) if fps else 0.0


def _keyframe_count_for_shot(
    duration: float,
    density_per_sec: float,
    min_n: int,
    max_n: int,
) -> int:
    if duration <= 0:
        return 0
    n = int(math.ceil(duration * density_per_sec))
    return max(min_n, min(max_n, n))


def _timestamps_inside_shot(start: float, end: float, n: int) -> List[float]:
    if n <= 0:
        return []
    if n == 1:
        return [(start + end) / 2.0]
    return list(np.linspace(start, end, n + 2)[1:-1])


def extract_keyframes_adaptive(
    video_path: str | Path,
    shot_ranges: List[Tuple[float, float]],
    density_per_sec: float = 1.0,
    min_per_shot: int = 1,
    max_per_shot: int = 12,
) -> List[Tuple[float, np.ndarray, int]]:
    """Per shot: số keyframe = ceil(duration × density), clamp [min, max].

    Returns: list (timestamp, RGB frame, shot_idx).
    """
    cap = cv2.VideoCapture(str(video_path))
    results: List[Tuple[float, np.ndarray, int]] = []

    if not shot_ranges:
        # Fallback: uniform 1fps trên toàn video
        duration = get_video_duration(video_path)
        if duration <= 0:
            cap.release()
            return []
        n = max(1, int(duration))
        for ts in np.linspace(0, duration, n, endpoint=False):
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
            ok, frame = cap.read()
            if ok:
                results.append((float(ts), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 0))
        cap.release()
        return results

    for shot_idx, (start, end) in enumerate(shot_ranges):
        duration = max(0.0, end - start)
        n = _keyframe_count_for_shot(duration, density_per_sec, min_per_shot, max_per_shot)
        for ts in _timestamps_inside_shot(start, end, n):
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
            ok, frame = cap.read()
            if ok:
                results.append((float(ts), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), shot_idx))
    cap.release()
    return results
