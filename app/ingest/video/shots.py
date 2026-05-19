from pathlib import Path
from typing import List, Tuple

from scenedetect import ContentDetector, detect


def detect_shots(video_path: str | Path, threshold: float = 27.0) -> List[Tuple[float, float]]:
    """Phát hiện shot boundaries. Trả về list (start_sec, end_sec)."""
    scene_list = detect(str(video_path), ContentDetector(threshold=threshold))
    return [(s.get_seconds(), e.get_seconds()) for s, e in scene_list]
