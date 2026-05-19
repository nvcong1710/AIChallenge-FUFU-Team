"""Retrieval: FAISS dense + FTS5 BM25 (visual annotations) + FTS5 BM25 (ASR transcripts)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np


class Retriever:
    def __init__(self, index_path: str | Path, db_path: str | Path, ef_search: int = 128):
        self.db_path = Path(db_path)
        self.index = faiss.read_index(str(index_path))
        try:
            self.index.hnsw.efSearch = ef_search
        except Exception:
            pass

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # ---- search channels ----

    def faiss_search(self, query_vec: np.ndarray, top_k: int = 500) -> List[Tuple[int, float]]:
        q = np.atleast_2d(query_vec).astype(np.float32)
        scores, ids = self.index.search(q, top_k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    @staticmethod
    def _build_fts_or_query(queries) -> str:
        """Build FTS5 query: tách thành tokens, OR tất cả. BM25 ranking sẽ ưu tiên
        row khớp nhiều token hơn. Phrase match cũ quá strict — đoạn lời ASR khó
        match 4-5 từ liên tiếp với query gốc.
        """
        if isinstance(queries, str):
            queries = [queries]
        tokens: set[str] = set()
        for q in queries:
            q = (q or "").strip().lower()
            if not q:
                continue
            # Split bằng whitespace, bỏ token quá ngắn / có ký tự đặc biệt
            for tok in q.split():
                # Lọc ký tự đặc biệt FTS5: " ( ) : -
                cleaned = "".join(c for c in tok if c.isalnum() or c in "ăâđêôơưàáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ")
                if len(cleaned) >= 2:
                    tokens.add(cleaned)
        if not tokens:
            return ""
        # Wrap mỗi token trong quote (an toàn FTS5)
        return " OR ".join(f'"{t}"' for t in tokens)

    # Filter raw BM25 score threshold — score < này bị coi là noise (1-token match)
    MIN_BM25_RAW = 3.0

    def bm25_visual(self, queries, top_k: int = 200) -> List[Tuple[int, float]]:
        """BM25 trên frame_text. Returns [(frame_id, score)] đã filter weak matches."""
        fts_query = self._build_fts_or_query(queries)
        if not fts_query:
            return []
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT rowid, bm25(frame_text) FROM frame_text "
                    "WHERE frame_text MATCH ? ORDER BY bm25(frame_text) LIMIT ?",
                    (fts_query, top_k),
                ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(int(rid), -float(s)) for rid, s in rows if -float(s) >= self.MIN_BM25_RAW]

    def bm25_asr(self, queries, top_k: int = 200) -> List[Tuple[int, float]]:
        """BM25 trên asr_text. Filter raw score < MIN_BM25_RAW để bỏ single-token noise."""
        fts_query = self._build_fts_or_query(queries)
        if not fts_query:
            return []
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT rowid, bm25(asr_text) FROM asr_text "
                    "WHERE asr_text MATCH ? ORDER BY bm25(asr_text) LIMIT ?",
                    (fts_query, top_k),
                ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(int(rid), -float(s)) for rid, s in rows if -float(s) >= self.MIN_BM25_RAW]

    # ---- enrichment ----

    def frames_by_faiss_ids(self, faiss_ids: List[int]) -> Dict[int, dict]:
        if not faiss_ids:
            return {}
        ph = ",".join("?" * len(faiss_ids))
        sql = f"""
            SELECT f.faiss_id, f.id, f.item_id, f.timestamp, f.thumbnail_path,
                   f.caption, f.objects_json,
                   GROUP_CONCAT(fs.segment_id)
            FROM frames f
            LEFT JOIN frame_segments fs ON fs.frame_id = f.id
            WHERE f.faiss_id IN ({ph})
            GROUP BY f.id
        """
        with self._conn() as conn:
            rows = conn.execute(sql, faiss_ids).fetchall()
        out: Dict[int, dict] = {}
        for fid, rid, item_id, ts, thumb, caption, objects_json, segs in rows:
            out[int(fid)] = {
                "frame_id": int(rid),
                "item_id": int(item_id),
                "timestamp": float(ts),
                "thumbnail": thumb,
                "caption": caption or "",
                "objects": json.loads(objects_json) if objects_json else [],
                "segment_ids": [int(x) for x in (segs or "").split(",") if x],
            }
        return out

    def frames_by_db_ids(self, frame_ids: List[int]) -> Dict[int, dict]:
        if not frame_ids:
            return {}
        ph = ",".join("?" * len(frame_ids))
        sql = f"""
            SELECT f.id, f.faiss_id, f.item_id, f.timestamp, f.thumbnail_path,
                   f.caption, f.objects_json,
                   GROUP_CONCAT(fs.segment_id)
            FROM frames f
            LEFT JOIN frame_segments fs ON fs.frame_id = f.id
            WHERE f.id IN ({ph})
            GROUP BY f.id
        """
        with self._conn() as conn:
            rows = conn.execute(sql, frame_ids).fetchall()
        out: Dict[int, dict] = {}
        for rid, fid, item_id, ts, thumb, caption, objects_json, segs in rows:
            out[int(rid)] = {
                "frame_id": int(rid),
                "faiss_id": int(fid) if fid is not None else None,
                "item_id": int(item_id),
                "timestamp": float(ts),
                "thumbnail": thumb,
                "caption": caption or "",
                "objects": json.loads(objects_json) if objects_json else [],
                "segment_ids": [int(x) for x in (segs or "").split(",") if x],
            }
        return out

    def asr_segments_by_ids(self, asr_ids: List[int]) -> Dict[int, dict]:
        if not asr_ids:
            return {}
        ph = ",".join("?" * len(asr_ids))
        sql = f"""
            SELECT id, item_id, start_sec, end_sec, text, segment_id
            FROM asr_segments WHERE id IN ({ph})
        """
        with self._conn() as conn:
            rows = conn.execute(sql, asr_ids).fetchall()
        return {
            int(aid): {
                "asr_id": int(aid),
                "item_id": int(item_id),
                "start": float(s),
                "end": float(e),
                "text": text,
                "segment_id": int(seg_id) if seg_id is not None else None,
            }
            for aid, item_id, s, e, text, seg_id in rows
        }

    def segments_meta(self, segment_ids: List[int]) -> Dict[int, dict]:
        if not segment_ids:
            return {}
        ph = ",".join("?" * len(segment_ids))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT id, item_id, seg_idx, start_sec, end_sec, scene_id FROM segments WHERE id IN ({ph})",
                segment_ids,
            ).fetchall()
        return {
            int(sid): {
                "item_id": int(item_id),
                "seg_idx": int(idx),
                "start": float(s),
                "end": float(e),
                "scene_id": int(scene_id) if scene_id is not None else None,
            }
            for sid, item_id, idx, s, e, scene_id in rows
        }

    def scenes_meta(self, scene_ids: List[int]) -> Dict[int, dict]:
        ids = [int(s) for s in scene_ids if s is not None]
        if not ids:
            return {}
        ph = ",".join("?" * len(ids))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT id, item_id, scene_idx, start_sec, end_sec, n_shots FROM scenes WHERE id IN ({ph})",
                ids,
            ).fetchall()
        return {
            int(sid): {
                "item_id": int(item_id),
                "scene_idx": int(idx),
                "start": float(s),
                "end": float(e),
                "n_shots": int(n),
            }
            for sid, item_id, idx, s, e, n in rows
        }

    def items_meta(self, item_ids) -> Dict[int, dict]:
        ids = list({int(i) for i in item_ids})
        if not ids:
            return {}
        ph = ",".join("?" * len(ids))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT id, path, media_type, duration_sec FROM items WHERE id IN ({ph})",
                ids,
            ).fetchall()
        return {
            int(i): {"path": p, "media_type": mt, "duration_sec": float(d)}
            for i, p, mt, d in rows
        }

    def stats(self) -> dict:
        c = lambda q: self._conn().execute(q).fetchone()[0]
        return {
            "items": int(c("SELECT COUNT(*) FROM items")),
            "items_video": int(c("SELECT COUNT(*) FROM items WHERE media_type='video'")),
            "items_audio": int(c("SELECT COUNT(*) FROM items WHERE media_type='audio'")),
            "items_image": int(c("SELECT COUNT(*) FROM items WHERE media_type='image'")),
            "frames": int(c("SELECT COUNT(*) FROM frames")),
            "segments": int(c("SELECT COUNT(*) FROM segments")),
            "scenes": int(c("SELECT COUNT(*) FROM scenes")),
            "asr_segments": int(c("SELECT COUNT(*) FROM asr_segments")),
            "faiss_total": int(self.index.ntotal),
        }
