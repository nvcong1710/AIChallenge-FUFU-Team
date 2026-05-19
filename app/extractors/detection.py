"""YOLO-World v2 open-vocabulary object detection.

Dùng ultralytics YOLOWorld. Class list mặc định cover các đối tượng phổ biến cho
ngữ cảnh VBS; user có thể override qua config.detection_classes.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..common.types import DetectionBox, FrameAnnotation


DEFAULT_CLASSES = [
    # người + bộ phận
    "person", "face", "hand", "child", "elderly person",
    # giao thông
    "car", "motorcycle", "bicycle", "bus", "truck", "boat", "airplane", "traffic light", "traffic sign", "license plate",
    # động vật
    "cat", "dog", "bird", "horse", "cow", "fish",
    # nội thất / vật dụng
    "table", "chair", "bed", "sofa", "tv", "laptop", "computer", "phone", "book", "clock",
    # ăn uống
    "food", "rice bowl", "noodle bowl", "cup", "bottle", "fruit",
    # kiến trúc / cảnh
    "building", "house", "tree", "mountain", "river", "beach", "road", "bridge",
    # văn hoá / sự kiện
    "flag", "logo", "sign", "text", "screen", "microphone", "stage", "crowd",
    # trò chơi / giải trí
    "chess board", "chess piece", "ball", "guitar", "piano",
    # văn phòng / công nghiệp
    "document", "money", "machine", "tool",
]


class DetectionExtractor:
    def __init__(self, cfg: dict):
        self.enabled = False
        self.model = None

        ex_cfg = cfg.get("extractors", {})
        if not ex_cfg.get("enable_detection", True):
            print("[detect] disabled by config.")
            return

        classes = ex_cfg.get("detection_classes")
        self.classes: List[str] = classes if classes else DEFAULT_CLASSES
        self.min_conf = float(ex_cfg.get("detection_min_confidence", 0.25))
        model_path = ex_cfg.get("detection_model", "yolov8l-world.pt")

        device = cfg.get("models", {}).get("device", "cuda")
        self.device = device if device == "cuda" else "cpu"

        try:
            from ultralytics import YOLOWorld

            self.model = YOLOWorld(model_path)
            self.model.set_classes(self.classes)
            self.enabled = True
        except Exception as e:
            print(f"[detect] init fail: {e}; disabled.")

    def extract(self, image_rgb: np.ndarray) -> List[DetectionBox]:
        if not self.enabled or self.model is None:
            return []
        try:
            results = self.model.predict(
                image_rgb,
                conf=self.min_conf,
                verbose=False,
                device=self.device,
            )
        except Exception as e:
            print(f"[detect] inference fail: {e}")
            return []

        out: List[DetectionBox] = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().tolist()
            confs = r.boxes.conf.cpu().tolist()
            clss = r.boxes.cls.cpu().tolist()
            for box, conf, cls_idx in zip(xyxy, confs, clss):
                label = self.classes[int(cls_idx)] if 0 <= int(cls_idx) < len(self.classes) else str(int(cls_idx))
                out.append(DetectionBox(label=label, confidence=float(conf), bbox=box))
        return out

    def annotate(self, image_rgb: np.ndarray, annotation: FrameAnnotation) -> None:
        annotation.objects = self.extract(image_rgb)
