"""Kiểu dữ liệu chung dùng xuyên hệ thống."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class MediaType(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"


VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv", ".wmv")
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif")


def detect_media_type(path: str) -> Optional[MediaType]:
    p = path.lower()
    if any(p.endswith(e) for e in VIDEO_EXTS):
        return MediaType.VIDEO
    if any(p.endswith(e) for e in AUDIO_EXTS):
        return MediaType.AUDIO
    if any(p.endswith(e) for e in IMAGE_EXTS):
        return MediaType.IMAGE
    return None


@dataclass
class DetectionBox:
    label: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] absolute pixel coords

    def to_dict(self) -> dict:
        return {"label": self.label, "conf": self.confidence, "bbox": self.bbox}


@dataclass
class FrameAnnotation:
    """Tất cả output extractor đính kèm 1 frame visual."""
    ocr_text: str = ""
    ocr_lines: List[dict] = field(default_factory=list)
    caption: str = ""
    objects: List[DetectionBox] = field(default_factory=list)

    @property
    def labels_joined(self) -> str:
        return " ".join(sorted({o.label for o in self.objects}))


@dataclass
class ASRSegment:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "text": self.text}
