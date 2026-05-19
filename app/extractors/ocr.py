"""OCR extractor — EasyOCR (Python 3.12 compatible) thay PaddleOCR.

EasyOCR có hỗ trợ tiếng Việt native (lang code 'vi'), runs trên py3.12, GPU optional.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..common.types import FrameAnnotation


class OCRExtractor:
    def __init__(self, cfg: dict):
        self.enabled = False
        self.reader = None
        ex_cfg = cfg.get("extractors", {})
        if not ex_cfg.get("enable_ocr", True):
            print("[ocr] disabled by config.")
            return

        self.min_conf = float(ex_cfg.get("ocr_min_confidence", 0.5))
        lang = ex_cfg.get("ocr_lang", "vi")
        device = cfg.get("models", {}).get("device", "cuda")
        use_gpu = device == "cuda"

        try:
            import easyocr
        except ImportError:
            print("[ocr] easyocr không có sẵn; bỏ qua OCR.")
            return

        # EasyOCR tự tải models lần đầu (~80MB cho vi+en)
        try:
            self.reader = easyocr.Reader(
                [lang, "en"] if lang != "en" else ["en"],
                gpu=use_gpu,
                verbose=False,
            )
            self.enabled = True
        except Exception as e:
            print(f"[ocr] init lang={lang!r} fail: {e}")
            try:
                self.reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
                self.enabled = True
                print("[ocr] fallback 'en' only")
            except Exception as e2:
                print(f"[ocr] fallback also fail: {e2}; disabled.")

    def extract(self, image_rgb: np.ndarray) -> Tuple[str, List[dict]]:
        if not self.enabled or self.reader is None:
            return "", []
        try:
            # readtext returns list of (bbox, text, confidence)
            results = self.reader.readtext(image_rgb, detail=1, paragraph=False)
        except Exception as e:
            print(f"[ocr] inference fail: {e}")
            return "", []
        lines: List[dict] = []
        for r in results:
            if len(r) < 3:
                continue
            bbox, text, conf = r[0], r[1], float(r[2])
            if conf < self.min_conf or not text.strip():
                continue
            # bbox là 4 điểm corner — convert sang x1y1x2y2
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            xyxy = [min(xs), min(ys), max(xs), max(ys)]
            lines.append({"text": text.strip(), "conf": conf, "bbox": xyxy})
        joined = " ".join(l["text"] for l in lines)
        return joined, lines

    def annotate(self, image_rgb: np.ndarray, annotation: FrameAnnotation) -> None:
        text, lines = self.extract(image_rgb)
        annotation.ocr_text = text
        annotation.ocr_lines = lines
