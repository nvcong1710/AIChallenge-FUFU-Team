"""Định nghĩa segments — 2 strategy: shots-as-segments hoặc sliding window."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple


def shots_to_segments(
    shots: List[Tuple[float, float]],
    max_segment_len: float = 15.0,
) -> List[Tuple[int, float, float]]:
    """Mỗi shot → 1 segment. Shot quá dài (> max_segment_len) chia nhỏ đều.

    Trả về list (seg_idx, start_sec, end_sec). seg_idx liên tục từ 0.
    """
    out: List[Tuple[int, float, float]] = []
    seg_idx = 0
    for start, end in shots:
        duration = max(0.0, end - start)
        if duration <= 0:
            continue
        if duration <= max_segment_len:
            out.append((seg_idx, start, end))
            seg_idx += 1
            continue
        n = int(math.ceil(duration / max_segment_len))
        step = duration / n
        for i in range(n):
            s = start + i * step
            e = end if i == n - 1 else s + step
            out.append((seg_idx, s, e))
            seg_idx += 1
    return out


def build_sliding_segments(
    video_duration: float,
    segment_len: float = 5.0,
    stride: float = 2.5,
) -> List[Tuple[int, float, float]]:
    """Sliding window có overlap (fallback nếu không dùng shots)."""
    if video_duration <= 0:
        return []
    out: List[Tuple[int, float, float]] = []
    seg_idx = 0
    t = 0.0
    while t < video_duration:
        out.append((seg_idx, t, min(t + segment_len, video_duration)))
        seg_idx += 1
        t += stride
    return out


def assign_frames_to_segments(
    frame_timestamps: List[float],
    segments: List[Tuple[int, float, float]],
) -> Dict[int, List[int]]:
    """Map frame index → list of seg_idx mà frame thuộc về (inclusive cả 2 đầu)."""
    out: Dict[int, List[int]] = {}
    for i, ts in enumerate(frame_timestamps):
        belongs = [sid for sid, s, e in segments if s <= ts <= e]
        if belongs:
            out[i] = belongs
    return out
