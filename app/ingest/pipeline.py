"""Ingest router — dispatch theo media_type."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..common.config import ensure_storage_dirs, get_config
from ..common.encoder import SiglipEncoder
from ..common.types import MediaType, detect_media_type
from .audio.ingest import ingest_audio
from .image.ingest import ingest_image
from .storage import IndexWriter
from .video.ingest import ingest_video


def run_ingest(paths: Iterable[Path], cfg: dict | None = None) -> None:
    cfg = cfg or get_config()
    ensure_storage_dirs(cfg)

    encoder = SiglipEncoder(cfg["models"]["siglip"], device=cfg["models"]["device"])
    writer = IndexWriter(
        cfg["storage"]["index_path"],
        cfg["storage"]["db_path"],
        dim=encoder.dim,
        hnsw_m=int(cfg["retrieval"]["hnsw_m"]),
        ef_construct=int(cfg["retrieval"]["hnsw_ef_construct"]),
    )

    try:
        for p in paths:
            p = Path(p)
            mt = detect_media_type(str(p))
            if mt is None:
                print(f"⚠ {p.name}: không nhận diện được media type, bỏ qua.")
                continue
            try:
                if mt == MediaType.VIDEO:
                    ingest_video(p, encoder, writer, cfg)
                elif mt == MediaType.AUDIO:
                    ingest_audio(p, writer, cfg)
                elif mt == MediaType.IMAGE:
                    ingest_image(p, encoder, writer, cfg)
            except Exception as e:
                print(f"  ✗ lỗi với {p}: {e}")
                import traceback
                traceback.print_exc()
    finally:
        writer.persist()
        stats = writer.stats()
        writer.close()

    print("\n=== Stats sau ingest ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
