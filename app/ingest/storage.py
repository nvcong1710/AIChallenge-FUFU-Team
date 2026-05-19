"""Persistent storage: FAISS HNSW + SQLite + 2 FTS5 (visual annotations & ASR)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np

from ..common.types import ASRSegment, FrameAnnotation, MediaType


SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    media_type TEXT NOT NULL CHECK(media_type IN ('video','audio','image')),
    duration_sec REAL DEFAULT 0,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_items_type ON items(media_type);

CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL,
    seg_idx INTEGER NOT NULL,
    start_sec REAL NOT NULL,
    end_sec REAL NOT NULL,
    scene_id INTEGER,
    UNIQUE(item_id, seg_idx),
    FOREIGN KEY(item_id) REFERENCES items(id),
    FOREIGN KEY(scene_id) REFERENCES scenes(id)
);
CREATE INDEX IF NOT EXISTS idx_segments_item ON segments(item_id);
CREATE INDEX IF NOT EXISTS idx_segments_scene ON segments(scene_id);

CREATE TABLE IF NOT EXISTS scenes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL,
    scene_idx INTEGER NOT NULL,
    start_sec REAL NOT NULL,
    end_sec REAL NOT NULL,
    n_shots INTEGER DEFAULT 0,
    UNIQUE(item_id, scene_idx),
    FOREIGN KEY(item_id) REFERENCES items(id)
);
CREATE INDEX IF NOT EXISTS idx_scenes_item ON scenes(item_id);

CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    thumbnail_path TEXT,
    faiss_id INTEGER UNIQUE,
    caption TEXT,
    objects_json TEXT,
    FOREIGN KEY(item_id) REFERENCES items(id)
);
CREATE INDEX IF NOT EXISTS idx_frames_item ON frames(item_id);
CREATE INDEX IF NOT EXISTS idx_frames_faiss ON frames(faiss_id);

CREATE TABLE IF NOT EXISTS frame_segments (
    frame_id INTEGER NOT NULL,
    segment_id INTEGER NOT NULL,
    PRIMARY KEY (frame_id, segment_id),
    FOREIGN KEY(frame_id) REFERENCES frames(id),
    FOREIGN KEY(segment_id) REFERENCES segments(id)
);
CREATE INDEX IF NOT EXISTS idx_fs_seg ON frame_segments(segment_id);

CREATE TABLE IF NOT EXISTS asr_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL,
    start_sec REAL NOT NULL,
    end_sec REAL NOT NULL,
    text TEXT NOT NULL,
    segment_id INTEGER,
    FOREIGN KEY(item_id) REFERENCES items(id),
    FOREIGN KEY(segment_id) REFERENCES segments(id)
);
CREATE INDEX IF NOT EXISTS idx_asr_item ON asr_segments(item_id);
CREATE INDEX IF NOT EXISTS idx_asr_segment ON asr_segments(segment_id);

CREATE VIRTUAL TABLE IF NOT EXISTS frame_text USING fts5(
    ocr_text, caption, labels,
    tokenize='unicode61 remove_diacritics 0'
);

CREATE VIRTUAL TABLE IF NOT EXISTS asr_text USING fts5(
    transcript,
    tokenize='unicode61 remove_diacritics 0'
);
"""


class IndexWriter:
    """Quản lý FAISS + SQLite + FTS5 indices. Visual-only assets (image/video frames) đẩy vector vào FAISS;
    audio-only items không có vector visual, chỉ index transcript qua FTS5."""

    def __init__(
        self,
        index_path: str | Path,
        db_path: str | Path,
        dim: int,
        hnsw_m: int = 32,
        ef_construct: int = 200,
    ):
        self.index_path = Path(index_path)
        self.db_path = Path(db_path)
        self.dim = dim

        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = ef_construct

        self.conn = sqlite3.connect(self.db_path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---- items ----

    def add_or_get_item(self, path: str, media_type: MediaType, duration: float = 0.0) -> int:
        self.conn.execute(
            "INSERT OR IGNORE INTO items (path, media_type, duration_sec) VALUES (?, ?, ?)",
            (path, media_type.value, duration),
        )
        self.conn.commit()
        row = self.conn.execute("SELECT id FROM items WHERE path = ?", (path,)).fetchone()
        return int(row[0])

    def item_already_ingested(self, item_id: int) -> bool:
        n_frames = self.conn.execute(
            "SELECT COUNT(*) FROM frames WHERE item_id = ?", (item_id,)
        ).fetchone()[0]
        n_asr = self.conn.execute(
            "SELECT COUNT(*) FROM asr_segments WHERE item_id = ?", (item_id,)
        ).fetchone()[0]
        return (n_frames + n_asr) > 0

    # ---- segments ----

    def add_segments(self, item_id: int, segments: List[Tuple[int, float, float]]) -> Dict[int, int]:
        for seg_idx, s, e in segments:
            self.conn.execute(
                "INSERT OR IGNORE INTO segments (item_id, seg_idx, start_sec, end_sec) VALUES (?, ?, ?, ?)",
                (item_id, seg_idx, s, e),
            )
        self.conn.commit()
        rows = self.conn.execute(
            "SELECT seg_idx, id FROM segments WHERE item_id = ?", (item_id,)
        ).fetchall()
        return {int(idx): int(db_id) for idx, db_id in rows}

    def add_scenes_and_link(
        self,
        item_id: int,
        scenes: List[Tuple[int, List[int], float, float]],
        seg_id_map: Dict[int, int],
    ) -> Dict[int, int]:
        """scenes: list (scene_idx, [shot_seg_idx,...], start, end).
        seg_id_map: {shot_seg_idx: segments.id}.
        Trả về dict {scene_idx: scenes.id}, đồng thời update segments.scene_id.
        """
        scene_id_map: Dict[int, int] = {}
        for scene_idx, shot_seg_idxs, start, end in scenes:
            self.conn.execute(
                "INSERT OR IGNORE INTO scenes (item_id, scene_idx, start_sec, end_sec, n_shots) "
                "VALUES (?, ?, ?, ?, ?)",
                (item_id, scene_idx, start, end, len(shot_seg_idxs)),
            )
        self.conn.commit()
        rows = self.conn.execute(
            "SELECT scene_idx, id FROM scenes WHERE item_id = ?", (item_id,)
        ).fetchall()
        scene_id_map = {int(idx): int(db_id) for idx, db_id in rows}

        # Link shot-segments → scene
        for scene_idx, shot_seg_idxs, _, _ in scenes:
            scene_db_id = scene_id_map.get(scene_idx)
            if scene_db_id is None:
                continue
            for shot_seg_idx in shot_seg_idxs:
                seg_db_id = seg_id_map.get(shot_seg_idx)
                if seg_db_id is None:
                    continue
                self.conn.execute(
                    "UPDATE segments SET scene_id = ? WHERE id = ?",
                    (scene_db_id, seg_db_id),
                )
        self.conn.commit()
        return scene_id_map

    # ---- frames + visual annotations ----

    def add_frames(
        self,
        item_id: int,
        frame_records: List[dict],
        vectors: np.ndarray,
        frame_to_segs: Dict[int, List[int]],
        seg_id_map: Dict[int, int],
    ) -> None:
        """
        frame_records: list of {timestamp, thumbnail, annotation: FrameAnnotation}
        vectors: (N, D) float32 đã L2-normalize
        frame_to_segs: dict {frame_idx: [seg_idx]}
        """
        if len(frame_records) == 0:
            return
        assert vectors.shape[0] == len(frame_records), "vectors / frame_records mismatch"

        start_faiss = self.index.ntotal
        self.index.add(vectors.astype(np.float32))

        for i, rec in enumerate(frame_records):
            ann: FrameAnnotation = rec["annotation"]
            objects_json = json.dumps(
                [o.to_dict() for o in ann.objects], ensure_ascii=False
            )
            faiss_id = start_faiss + i
            cur = self.conn.execute(
                "INSERT INTO frames (item_id, timestamp, thumbnail_path, faiss_id, caption, objects_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    item_id,
                    float(rec["timestamp"]),
                    rec.get("thumbnail"),
                    faiss_id,
                    ann.caption,
                    objects_json,
                ),
            )
            frame_id = cur.lastrowid
            for seg_idx in frame_to_segs.get(i, []):
                seg_db = seg_id_map.get(seg_idx)
                if seg_db is not None:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO frame_segments (frame_id, segment_id) VALUES (?, ?)",
                        (frame_id, seg_db),
                    )
            # FTS5 frame_text — rowid = frame_id để JOIN dễ
            self.conn.execute(
                "INSERT INTO frame_text (rowid, ocr_text, caption, labels) VALUES (?, ?, ?, ?)",
                (frame_id, ann.ocr_text, ann.caption, ann.labels_joined),
            )
        self.conn.commit()

    # ---- ASR ----

    def add_asr_segments(
        self,
        item_id: int,
        asr_segments: List[ASRSegment],
        seg_id_map: Dict[int, int] | None = None,
        item_segments: List[Tuple[int, float, float]] | None = None,
    ) -> None:
        """seg_id_map + item_segments dùng để gán asr về segment chứa nó (overlap > 50%)."""
        if not asr_segments:
            return

        for asr in asr_segments:
            seg_db_id = None
            if seg_id_map and item_segments:
                best_overlap = 0.0
                best_seg_idx = None
                for seg_idx, s, e in item_segments:
                    overlap = max(0.0, min(e, asr.end) - max(s, asr.start))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_seg_idx = seg_idx
                if best_seg_idx is not None:
                    seg_db_id = seg_id_map.get(best_seg_idx)

            cur = self.conn.execute(
                "INSERT INTO asr_segments (item_id, start_sec, end_sec, text, segment_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (item_id, asr.start, asr.end, asr.text, seg_db_id),
            )
            asr_id = cur.lastrowid
            self.conn.execute(
                "INSERT INTO asr_text (rowid, transcript) VALUES (?, ?)",
                (asr_id, asr.text),
            )
        self.conn.commit()

    # ---- persist ----

    def persist(self) -> None:
        self.conn.commit()
        faiss.write_index(self.index, str(self.index_path))

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def stats(self) -> dict:
        c = self.conn.execute
        return {
            "items": int(c("SELECT COUNT(*) FROM items").fetchone()[0]),
            "items_video": int(c("SELECT COUNT(*) FROM items WHERE media_type='video'").fetchone()[0]),
            "items_audio": int(c("SELECT COUNT(*) FROM items WHERE media_type='audio'").fetchone()[0]),
            "items_image": int(c("SELECT COUNT(*) FROM items WHERE media_type='image'").fetchone()[0]),
            "frames": int(c("SELECT COUNT(*) FROM frames").fetchone()[0]),
            "segments": int(c("SELECT COUNT(*) FROM segments").fetchone()[0]),
            "scenes": int(c("SELECT COUNT(*) FROM scenes").fetchone()[0]),
            "asr_segments": int(c("SELECT COUNT(*) FROM asr_segments").fetchone()[0]),
            "faiss_total": int(self.index.ntotal),
        }
