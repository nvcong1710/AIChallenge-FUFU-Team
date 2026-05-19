"""Cross-modal feature extractors — load lazy 1 lần và share giữa ingest + backend."""

from __future__ import annotations

from typing import Any, Dict

_singletons: Dict[str, Any] = {}


def get_ocr(cfg: dict):
    if "ocr" not in _singletons:
        from .ocr import OCRExtractor
        _singletons["ocr"] = OCRExtractor(cfg)
    return _singletons["ocr"]


def get_caption(cfg: dict):
    if "caption" not in _singletons:
        from .caption import CaptionExtractor
        _singletons["caption"] = CaptionExtractor(cfg)
    return _singletons["caption"]


def get_detection(cfg: dict):
    if "detection" not in _singletons:
        from .detection import DetectionExtractor
        _singletons["detection"] = DetectionExtractor(cfg)
    return _singletons["detection"]


def get_asr(cfg: dict):
    if "asr" not in _singletons:
        from .asr import ASRExtractor
        _singletons["asr"] = ASRExtractor(cfg)
    return _singletons["asr"]


def reset():
    _singletons.clear()
