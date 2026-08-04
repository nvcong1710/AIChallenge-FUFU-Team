"""Chấm điểm theo HCM AI Challenge 2025 — thuần stdlib (không numpy, test nhanh).

Metric chính thức (đã verify 2 nguồn: Codabench 10187 + arXiv 2603.02888):

    "Mean of Top-k R-Scores", trung bình qua k ∈ {1, 5, 20, 50, 100}.

Với mỗi truy vấn, hệ trả về **danh sách submission đã xếp hạng**. Với mỗi ngưỡng k,
lấy R-Score CAO NHẤT trong k submission đầu (prefix-max), rồi trung bình 5 ngưỡng.

R-Score của 1 submission tuỳ task:
  • KIS   : 1.0 nếu video khớp tên GT **và** frame nằm trong khoảng [s,e]; else 0.
  • VQA   : như KIS, **và** đáp án text khớp chính xác (sau chuẩn hoá); else 0.
  • TRAKE : nếu video khớp → (1/N)·số event có frame_i ∈ [s_i, e_i]; else 0.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Sequence

# Ngưỡng k chính thức của cuộc thi — KHÔNG đổi (xem PHU-LUC §1.1).
K_VALUES: tuple[int, ...] = (1, 5, 20, 50, 100)


# ----------------------------------------------------------------------------
# Core: Mean of Top-k R-Scores
# ----------------------------------------------------------------------------

def mean_top_k_r_score(rscores: Sequence[float], ks: Sequence[int] = K_VALUES) -> float:
    """Trung bình qua các ngưỡng k của (R-score cao nhất trong top-k submission).

    rscores: R-score của từng submission, **theo thứ tự hạng** (tốt nhất trước).
    Trả về 0.0 nếu rỗng.
    """
    if not rscores:
        return 0.0
    total = 0.0
    for k in ks:
        prefix = rscores[:k]
        total += max(prefix) if prefix else 0.0
    return total / len(ks)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _video_match(predicted: str, gt: str) -> bool:
    """Khớp nếu tên video GT là substring của predicted (path/tên), không phân biệt hoa-thường.

    Đủ bền cho cả 'path/L01_V001.mp4' lẫn 'L01_V001'. (Đề thi: 'predicted video khớp tên GT'.)
    """
    if not predicted or not gt:
        return False
    return gt.strip().lower() in predicted.strip().lower()


def _in_range(value: float, lo: float, hi: float) -> bool:
    """Bao gồm 2 biên [lo, hi]."""
    return lo <= value <= hi


def _norm_answer(text: str) -> str:
    """Chuẩn hoá đáp án VQA: bỏ khoảng trắng thừa + hạ thường (khớp 'chính xác' nhẹ tay)."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


# ----------------------------------------------------------------------------
# R-Score per task
# ----------------------------------------------------------------------------

def r_score_kis(result_video: str, result_frame: float, gt_video: str,
                gt_range: tuple[float, float]) -> float:
    if not _video_match(result_video, gt_video):
        return 0.0
    lo, hi = gt_range
    return 1.0 if _in_range(result_frame, lo, hi) else 0.0


def r_score_vqa(result_video: str, result_frame: float, result_answer: str,
                gt_video: str, gt_range: tuple[float, float], gt_answer: str) -> float:
    if r_score_kis(result_video, result_frame, gt_video, gt_range) == 0.0:
        return 0.0
    return 1.0 if _norm_answer(result_answer) == _norm_answer(gt_answer) else 0.0


def r_score_trake(result_video: str, result_frames: Sequence[float], gt_video: str,
                  gt_events: Sequence[tuple[float, float]]) -> float:
    if not _video_match(result_video, gt_video):
        return 0.0
    n = len(gt_events)
    if n == 0:
        return 0.0
    hit = 0
    for i, (lo, hi) in enumerate(gt_events):
        if i < len(result_frames) and _in_range(result_frames[i], lo, hi):
            hit += 1
    return hit / n


# ----------------------------------------------------------------------------
# Aggregate over queries
# ----------------------------------------------------------------------------

def aggregate(per_query: Sequence[dict]) -> dict:
    """per_query: [{'task': str, 'score': float}, ...] (score = mean_top_k_r_score của câu đó).

    Trả về {'overall': float, 'by_task': {task: mean}, 'n': int, 'n_by_task': {task: int}}.
    """
    n = len(per_query)
    if n == 0:
        return {"overall": 0.0, "by_task": {}, "n": 0, "n_by_task": {}}
    overall = sum(q["score"] for q in per_query) / n
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for q in per_query:
        sums[q["task"]] += q["score"]
        counts[q["task"]] += 1
    by_task = {t: sums[t] / counts[t] for t in sums}
    return {"overall": overall, "by_task": by_task, "n": n, "n_by_task": dict(counts)}
