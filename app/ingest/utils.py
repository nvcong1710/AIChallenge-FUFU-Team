"""Helper chung cho ingest pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

from ..common.types import (
    AUDIO_EXTS,
    IMAGE_EXTS,
    VIDEO_EXTS,
    MediaType,
    detect_media_type,
)


def collect_files(paths: List[str | Path], allow_ext: tuple[str, ...] | None = None) -> List[Path]:
    """Recursively collect files. Nếu allow_ext None, dùng tổng hợp video+audio+image."""
    if allow_ext is None:
        allow_ext = VIDEO_EXTS + AUDIO_EXTS + IMAGE_EXTS
    allow_ext = tuple(e.lower() for e in allow_ext)
    out: List[Path] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"⚠ không tìm thấy: {p}")
            continue
        if path.is_dir():
            for f in sorted(path.rglob("*")):
                if f.is_file() and f.suffix.lower() in allow_ext:
                    out.append(f)
        elif path.suffix.lower() in allow_ext:
            out.append(path)
    return out


def save_thumbnail(image_rgb: np.ndarray, out_path: Path, quality: int = 85) -> None:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def resize_keep_aspect(image_rgb: np.ndarray, max_size: int) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_size:
        return image_rgb
    scale = max_size / longest
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def group_by_type(files: List[Path]) -> dict[MediaType, List[Path]]:
    groups: dict[MediaType, List[Path]] = {MediaType.VIDEO: [], MediaType.AUDIO: [], MediaType.IMAGE: []}
    for f in files:
        mt = detect_media_type(str(f))
        if mt is not None:
            groups[mt].append(f)
    return groups
