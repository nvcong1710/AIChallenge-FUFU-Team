"""Audio segmentation — ưu tiên dùng ASR chunks làm 'cảnh tự nhiên'."""

from __future__ import annotations

import math
from typing import List, Tuple

from ...common.types import ASRSegment


def merge_close_chunks(
    asr_segments: List[ASRSegment],
    max_gap_sec: float = 0.5,
) -> List[ASRSegment]:
    """Gộp các ASR chunk liền kề có gap < max_gap_sec để tránh phân mảnh.

    Khi gộp, text nối bằng dấu cách; thời gian lấy min start / max end.
    """
    if not asr_segments:
        return []
    merged: List[ASRSegment] = []
    cur = asr_segments[0]
    for nxt in asr_segments[1:]:
        gap = nxt.start - cur.end
        if gap <= max_gap_sec:
            cur = ASRSegment(
                start=min(cur.start, nxt.start),
                end=max(cur.end, nxt.end),
                text=f"{cur.text} {nxt.text}".strip(),
            )
        else:
            merged.append(cur)
            cur = nxt
    merged.append(cur)
    return merged


def asr_chunks_to_segments(
    asr_segments: List[ASRSegment],
    max_segment_len: float = 15.0,
) -> List[Tuple[int, float, float]]:
    """Mỗi ASR chunk → 1 segment. Chunk quá dài (> max) chia đều."""
    out: List[Tuple[int, float, float]] = []
    seg_idx = 0
    for asr in asr_segments:
        duration = max(0.0, asr.end - asr.start)
        if duration <= 0:
            continue
        if duration <= max_segment_len:
            out.append((seg_idx, asr.start, asr.end))
            seg_idx += 1
            continue
        n = int(math.ceil(duration / max_segment_len))
        step = duration / n
        for i in range(n):
            s = asr.start + i * step
            e = asr.end if i == n - 1 else s + step
            out.append((seg_idx, s, e))
            seg_idx += 1
    return out


def build_sliding_segments(
    duration: float,
    segment_len: float = 10.0,
    stride: float = 5.0,
) -> List[Tuple[int, float, float]]:
    """Fallback sliding window cho audio không có speech (nhạc / ambient)."""
    if duration <= 0:
        return []
    out: List[Tuple[int, float, float]] = []
    seg_idx = 0
    t = 0.0
    while t < duration:
        out.append((seg_idx, t, min(t + segment_len, duration)))
        seg_idx += 1
        t += stride
    return out
