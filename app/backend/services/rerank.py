"""Hybrid score fusion + aggregate frame-level → segment-level (item-level cho image/audio)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


def _minmax(pairs: List[Tuple[int, float]]) -> Dict[int, float]:
    """Min-max normalize cho dense (cosine bounded [-1,1])."""
    if not pairs:
        return {}
    scores = [s for _, s in pairs]
    lo, hi = min(scores), max(scores)
    if hi <= lo:
        return {k: 1.0 for k, _ in pairs}
    return {k: (s - lo) / (hi - lo) for k, s in pairs}


# BM25 raw score scaling: BM25 unbounded above; raw 4-5 = strong phrase match,
# raw 8-12 = strong multi-token match. Scale linearly cap at 1.0 — giữ được
# độ mạnh tuyệt đối (vs min-max equalize tất cả về 1.0 nếu chỉ 1 hit).
BM25_SCALE = 8.0


def _raw_scaled_bm25(pairs: List[Tuple[int, float]]) -> Dict[int, float]:
    return {k: min(s / BM25_SCALE, 1.0) for k, s in pairs}


@dataclass
class Hit:
    """Một kết quả gộp tại cấp (item_id, segment_id_or_None)."""

    item_id: int
    segment_id: Optional[int]  # None nếu item là image (1 seg ảo) hoặc audio aggregate item-level
    score: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    best_frame: Optional[dict] = None    # meta của frame đại diện (chỉ video/image)
    best_asr: Optional[dict] = None      # snippet ASR (audio + video có lời)


def fuse_and_aggregate(
    dense_hits: List[Tuple[int, float]],
    bm25_visual_hits: List[Tuple[int, float]],
    bm25_asr_hits: List[Tuple[int, float]],
    frame_meta_by_faiss: Dict[int, dict],
    frame_meta_by_db: Dict[int, dict],
    asr_meta_by_id: Dict[int, dict],
    weights: dict,
) -> List[Hit]:
    """Trả về list Hit sorted desc theo final score.

    dense_hits: (faiss_id, raw_cosine) — visual SigLIP
    bm25_visual_hits: (frame_id, bm25_neg_score đã đảo dấu)
    bm25_asr_hits: (asr_id, ditto)
    """

    w_dense = float(weights.get("dense", 0.6))
    w_bm25v = float(weights.get("bm25_visual", 0.25))
    w_bm25a = float(weights.get("bm25_asr", 0.15))

    # Normalize từng kênh: dense min-max (bounded), BM25 raw-scaled (unbounded)
    dense_norm = _minmax(dense_hits)
    bm25v_norm = _raw_scaled_bm25(bm25_visual_hits)
    bm25a_norm = _raw_scaled_bm25(bm25_asr_hits)

    # Build dict aggregate theo (item_id, segment_id)
    # Khi frame có nhiều segment_ids → broadcast tất cả
    accum: Dict[Tuple[int, Optional[int]], Hit] = {}

    def touch(item_id: int, seg_id: Optional[int]) -> Hit:
        key = (item_id, seg_id)
        if key not in accum:
            accum[key] = Hit(item_id=item_id, segment_id=seg_id)
        return accum[key]

    # Kênh dense (visual frame)
    for fid, raw in dense_hits:
        meta = frame_meta_by_faiss.get(fid)
        if meta is None:
            continue
        s = dense_norm.get(fid, 0.0)
        item_id = meta["item_id"]
        seg_ids = meta["segment_ids"] or [None]
        for sid in seg_ids:
            hit = touch(item_id, sid)
            prev = hit.score_breakdown.get("dense", 0.0)
            if s > prev:
                hit.score_breakdown["dense"] = s
                # Lưu best_frame tốt nhất theo kênh dense
                if hit.best_frame is None or raw > (hit.best_frame.get("_raw") or -1):
                    hit.best_frame = {**meta, "_raw": float(raw)}

    # Kênh BM25 visual (frame_text)
    for fid, raw in bm25_visual_hits:
        meta = frame_meta_by_db.get(fid)
        if meta is None:
            continue
        s = bm25v_norm.get(fid, 0.0)
        item_id = meta["item_id"]
        seg_ids = meta["segment_ids"] or [None]
        for sid in seg_ids:
            hit = touch(item_id, sid)
            prev = hit.score_breakdown.get("bm25_visual", 0.0)
            if s > prev:
                hit.score_breakdown["bm25_visual"] = s
                if hit.best_frame is None:
                    hit.best_frame = {**meta, "_raw": float(raw)}

    # Kênh BM25 ASR
    for aid, raw in bm25_asr_hits:
        meta = asr_meta_by_id.get(aid)
        if meta is None:
            continue
        s = bm25a_norm.get(aid, 0.0)
        item_id = meta["item_id"]
        sid = meta.get("segment_id")
        hit = touch(item_id, sid)
        prev = hit.score_breakdown.get("bm25_asr", 0.0)
        if s > prev:
            hit.score_breakdown["bm25_asr"] = s
            # Lưu ASR snippet
            if hit.best_asr is None or raw > (hit.best_asr.get("_raw") or -1):
                hit.best_asr = {**meta, "_raw": float(raw)}

    # Final hybrid score — raw weighted sum, KHÔNG renormalize.
    # Item multi-channel match → score cao hơn item single-channel (đúng intent).
    # Weights tune trong settings.yaml để w_asr ≥ w_dense → audio item với
    # ASR match thắng visual item dense match khi mỗi item single-channel.
    for hit in accum.values():
        hit.score = (
            w_dense * hit.score_breakdown.get("dense", 0.0)
            + w_bm25v * hit.score_breakdown.get("bm25_visual", 0.0)
            + w_bm25a * hit.score_breakdown.get("bm25_asr", 0.0)
        )

    results = list(accum.values())
    results.sort(key=lambda h: h.score, reverse=True)
    return results
