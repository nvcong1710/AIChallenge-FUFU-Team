"""Ingest 1 ảnh tĩnh: 1 item → 1 segment → 1 frame với mọi annotation visual."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from ... import extractors
from ...common.encoder import SiglipEncoder
from ...common.types import FrameAnnotation, MediaType
from ..storage import IndexWriter
from ..utils import resize_keep_aspect, save_thumbnail


def _read_image_rgb(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def ingest_image(
    path: Path,
    encoder: SiglipEncoder,
    writer: IndexWriter,
    cfg: dict,
) -> None:
    print(f"\n[image] {path.name}")
    t0 = time.time()

    img = _read_image_rgb(path)
    if img is None:
        print(f"  ⚠ không đọc được ảnh.")
        return

    max_size = int(cfg["ingest"]["image"].get("max_size", 1024))
    img = resize_keep_aspect(img, max_size)

    item_id = writer.add_or_get_item(str(path), MediaType.IMAGE, duration=0.0)
    if writer.item_already_ingested(item_id):
        print("  đã ingest trước, bỏ qua.")
        return

    # Ảnh tĩnh: 1 segment trải [0, 0]
    seg_id_map = writer.add_segments(item_id, [(0, 0.0, 0.0)])

    # Extractors
    annotation = FrameAnnotation()
    extractors.get_ocr(cfg).annotate(img, annotation)
    extractors.get_caption(cfg).annotate(img, annotation)
    extractors.get_detection(cfg).annotate(img, annotation)
    print(
        f"  OCR={'✓' if annotation.ocr_text else '∅'}  "
        f"Caption={'✓' if annotation.caption else '∅'}  "
        f"Objects={len(annotation.objects)}"
    )

    # SigLIP embed
    vec = encoder.encode_images([img])

    # Thumbnail
    thumb_dir = Path(cfg["storage"]["thumbnail_dir"]) / "images"
    thumb_path = thumb_dir / f"img_{item_id:08d}.jpg"
    save_thumbnail(img, thumb_path)

    frame_records = [
        {
            "timestamp": 0.0,
            "thumbnail": str(thumb_path),
            "annotation": annotation,
        }
    ]
    writer.add_frames(item_id, frame_records, vec, {0: [0]}, seg_id_map)
    writer.persist()
    print(f"  ✓ {time.time() - t0:.1f}s")
