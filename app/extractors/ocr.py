"""PaddleOCR-VN extractor — đọc chữ trên ảnh / keyframe video."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..common.types import FrameAnnotation


class OCRExtractor:
    def __init__(self, cfg: dict):
        self.enabled = False
        self.ocr = None
        ex_cfg = cfg.get("extractors", {})
        if not ex_cfg.get("enable_ocr", True):
            print("[ocr] disabled by config.")
            return
        self.min_conf = float(ex_cfg.get("ocr_min_confidence", 0.5))
        lang = ex_cfg.get("ocr_lang", "vi")
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            print("[ocr] paddleocr không có sẵn; bỏ qua OCR.")
            return
        try:
            self.ocr = PaddleOCR(lang=lang, use_angle_cls=True, show_log=False)
        except Exception as e:
            print(f"[ocr] init lang={lang!r} fail ({e}); fallback 'en'.")
            try:
                self.ocr = PaddleOCR(lang="en", use_angle_cls=True, show_log=False)
            except Exception as e2:
                print(f"[ocr] fallback 'en' cũng fail ({e2}); disabled.")
                return
        self.enabled = True

    def extract(self, image_rgb: np.ndarray) -> Tuple[str, List[dict]]:
        """Trả về (joined_text, [{text, conf, bbox}, ...])."""
        if not self.enabled or self.ocr is None:
            return "", []
        try:
            result = self.ocr.ocr(image_rgb, cls=True)
        except Exception:
            return "", []
        if not result or not result[0]:
            return "", []
        lines: List[dict] = []
        for r in result[0]:
            if not r or len(r) < 2:
                continue
            bbox = r[0]
            text, conf = r[1]
            conf = float(conf)
            if conf < self.min_conf:
                continue
            lines.append({"text": text, "conf": conf, "bbox": bbox})
        joined = " ".join(l["text"] for l in lines)
        return joined, lines

    def annotate(self, image_rgb: np.ndarray, annotation: FrameAnnotation) -> None:
        """Inject OCR vào FrameAnnotation."""
        text, lines = self.extract(image_rgb)
        annotation.ocr_text = text
        annotation.ocr_lines = lines
