"""Gom các shot kề nhau có ngữ nghĩa visual tương tự thành 'scene'.

Heuristic đơn giản: cosine(last_frame_of_shot_i, first_frame_of_shot_i+1) >= threshold
→ cùng scene (camera đổi góc trong cùng bối cảnh / cùng event).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def cluster_shots_into_scenes(
    shot_segments: List[Tuple[int, float, float]],
    frame_indices_per_shot: Dict[int, List[int]],
    vectors: np.ndarray,
    threshold: float = 0.85,
) -> List[Tuple[int, List[int], float, float]]:
    """Phân cụm shot liền kề thành scene dựa trên cosine giữa frame biên.

    Args:
        shot_segments: list (shot_seg_idx, start, end) — sản phẩm của shots_to_segments
        frame_indices_per_shot: dict {shot_seg_idx: [vector_idx, ...]} — frame nào thuộc shot nào
        vectors: ndarray (N, D) — SigLIP vectors đã L2-normalize
        threshold: cosine ngưỡng để gom (0.85 mặc định)

    Returns:
        list of (scene_idx, [shot_seg_idx,...], scene_start, scene_end), sắp xếp theo thời gian.
    """
    if not shot_segments:
        return []

    # Map shot_seg_idx → vị trí trong shot_segments (cho lookup nhanh)
    shot_segments_sorted = sorted(shot_segments, key=lambda s: s[1])

    scenes: List[Tuple[int, List[int], float, float]] = []
    cur_shot_ids: List[int] = []
    cur_start: float = 0.0

    def _close_scene():
        if not cur_shot_ids:
            return
        last_seg_idx = cur_shot_ids[-1]
        last_end = next(e for sid, s, e in shot_segments_sorted if sid == last_seg_idx)
        scenes.append((len(scenes), list(cur_shot_ids), cur_start, last_end))

    for i, (seg_idx, start, end) in enumerate(shot_segments_sorted):
        if not cur_shot_ids:
            cur_shot_ids = [seg_idx]
            cur_start = start
            continue

        prev_seg_idx = cur_shot_ids[-1]
        prev_frames = frame_indices_per_shot.get(prev_seg_idx, [])
        curr_frames = frame_indices_per_shot.get(seg_idx, [])
        if not prev_frames or not curr_frames:
            # Không đủ data → break scene để an toàn
            _close_scene()
            cur_shot_ids = [seg_idx]
            cur_start = start
            continue

        prev_last = vectors[prev_frames[-1]]
        curr_first = vectors[curr_frames[0]]
        sim = float(np.dot(prev_last, curr_first))

        if sim >= threshold:
            cur_shot_ids.append(seg_idx)
        else:
            _close_scene()
            cur_shot_ids = [seg_idx]
            cur_start = start

    _close_scene()
    return scenes
