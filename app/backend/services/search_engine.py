"""Search engine top-level — query expansion → dense + 2 BM25 → hybrid fuse → enrich → JSON."""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from ...common.encoder import SiglipEncoder
from .rerank import fuse_and_aggregate
from .retrieval import Retriever
from .translator import Translator


class SearchEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        device = cfg["models"]["device"]
        self.encoder = SiglipEncoder(cfg["models"]["siglip"], device=device)

        self.translator = None
        self.paraphraser = None
        if cfg["query_expansion"].get("enable_translation", True):
            try:
                self.translator = Translator(cfg["models"]["translator"], device=device)
            except Exception as e:
                print(f"[search] translator init thất bại: {e}")
        if cfg["query_expansion"].get("enable_paraphrase", True):
            try:
                from .paraphraser import Paraphraser
                self.paraphraser = Paraphraser(cfg["models"]["paraphraser"], device=device)
            except Exception as e:
                print(f"[search] paraphraser init thất bại: {e}")

        self.retriever = Retriever(
            cfg["storage"]["index_path"],
            cfg["storage"]["db_path"],
            ef_search=int(cfg["retrieval"]["hnsw_ef_search"]),
        )

    # ---- query expansion ----

    def expand_query(self, query: str) -> dict:
        """Sinh các biến thể query, tách ra 'all' (dùng cho dense FAISS) và 'bm25' (chỉ original + translation, bỏ paraphrase để không phá phrase match)."""
        original = query.strip()
        translated: str | None = None
        paraphrases: List[str] = []

        if self.translator:
            try:
                en = self.translator.translate(original, src_lang="vie_Latn", tgt_lang="eng_Latn")
                if en and en.lower() != original.lower():
                    translated = en.strip()
            except Exception as e:
                print(f"[search] translate fail: {e}")

        if self.paraphraser:
            try:
                paras = self.paraphraser.paraphrase(
                    original,
                    n=int(self.cfg["query_expansion"]["num_paraphrases"]),
                    max_new_tokens=int(self.cfg["query_expansion"]["paraphrase_max_tokens"]),
                )
                paraphrases = [p.strip() for p in paras if p and p.strip()]
            except Exception as e:
                print(f"[search] paraphrase fail: {e}")

        # 'all' cho dense channel — original + translated + paraphrases
        all_variants = [original]
        if translated:
            all_variants.append(translated)
        all_variants.extend(paraphrases)
        seen = set()
        all_dedup = []
        for v in all_variants:
            k = v.lower()
            if k and k not in seen:
                seen.add(k)
                all_dedup.append(v)

        # 'bm25' — chỉ original + translated (bỏ paraphrase: phrase match với paraphrase
        # dài thường không khớp OCR/ASR ngắn, gây nhiễu)
        bm25 = [original]
        if translated:
            bm25.append(translated)

        return {
            "original": original,
            "translated": translated,
            "paraphrases": paraphrases,
            "all": all_dedup,
            "bm25": bm25,
        }

    # ---- main search ----

    def search(self, query: str, top_k: int = 20) -> dict:
        timing: Dict[str, float] = {}
        cfg_r = self.cfg["retrieval"]

        t = time.time()
        qe = self.expand_query(query)
        timing["expand_ms"] = (time.time() - t) * 1000

        # Encode query text — DENSE dùng tất cả variants (VI + EN + paraphrases)
        t = time.time()
        text_vecs = self.encoder.encode_text(qe["all"])
        if text_vecs.shape[0] == 0:
            return {"query": query, "expanded_queries": qe["all"], "results": [], "timing_ms": timing}
        q_vec = text_vecs.mean(axis=0)
        nrm = np.linalg.norm(q_vec)
        if nrm > 0:
            q_vec = q_vec / nrm
        timing["encode_ms"] = (time.time() - t) * 1000

        # 3 channels
        t = time.time()
        dense_hits = self.retriever.faiss_search(q_vec, top_k=int(cfg_r["top_k_dense"]))
        timing["faiss_ms"] = (time.time() - t) * 1000

        # BM25 channels — dùng original + translated (cross-lingual hybrid query)
        t = time.time()
        bm25v = self.retriever.bm25_visual(qe["bm25"], top_k=int(cfg_r["top_k_bm25_visual"]))
        timing["bm25_visual_ms"] = (time.time() - t) * 1000

        t = time.time()
        bm25a = self.retriever.bm25_asr(qe["bm25"], top_k=int(cfg_r["top_k_bm25_asr"]))
        timing["bm25_asr_ms"] = (time.time() - t) * 1000

        # Enrich metadata
        t = time.time()
        frame_meta_faiss = self.retriever.frames_by_faiss_ids([fid for fid, _ in dense_hits])
        frame_meta_db = self.retriever.frames_by_db_ids([fid for fid, _ in bm25v])
        asr_meta = self.retriever.asr_segments_by_ids([aid for aid, _ in bm25a])
        timing["fetch_meta_ms"] = (time.time() - t) * 1000

        # Hybrid fuse + aggregate
        t = time.time()
        hits = fuse_and_aggregate(
            dense_hits,
            bm25v,
            bm25a,
            frame_meta_faiss,
            frame_meta_db,
            asr_meta,
            weights=cfg_r["weights"],
        )
        timing["rerank_ms"] = (time.time() - t) * 1000

        top_hits = hits[:top_k]

        # Enrich segments + items + scenes
        seg_ids = [h.segment_id for h in top_hits if h.segment_id is not None]
        seg_meta = self.retriever.segments_meta(seg_ids)
        item_ids = [h.item_id for h in top_hits]
        item_meta = self.retriever.items_meta(item_ids)
        scene_ids_collected = [(seg.get("scene_id") if seg else None) for seg in seg_meta.values()]
        scene_meta = self.retriever.scenes_meta(scene_ids_collected)

        results = []
        for h in top_hits:
            item = item_meta.get(h.item_id, {})
            seg = seg_meta.get(h.segment_id) if h.segment_id is not None else None
            scene = scene_meta.get(seg["scene_id"]) if (seg and seg.get("scene_id")) else None

            bf = None
            if h.best_frame:
                bf = {
                    "frame_id": h.best_frame.get("frame_id"),
                    "timestamp": h.best_frame.get("timestamp", 0.0),
                    "thumbnail": h.best_frame.get("thumbnail"),
                    "caption": h.best_frame.get("caption", ""),
                    "objects": h.best_frame.get("objects", []),
                    "raw_cosine": h.best_frame.get("_raw"),
                }

            ba = None
            if h.best_asr:
                ba = {
                    "asr_id": h.best_asr.get("asr_id"),
                    "start": h.best_asr.get("start"),
                    "end": h.best_asr.get("end"),
                    "text": h.best_asr.get("text"),
                }

            results.append(
                {
                    "item_id": h.item_id,
                    "media_type": item.get("media_type"),
                    "item_path": item.get("path"),
                    "segment_id": h.segment_id,
                    "segment_start": seg.get("start") if seg else None,
                    "segment_end": seg.get("end") if seg else None,
                    "scene_id": seg.get("scene_id") if seg else None,
                    "scene_start": scene.get("start") if scene else None,
                    "scene_end": scene.get("end") if scene else None,
                    "scene_n_shots": scene.get("n_shots") if scene else None,
                    "score": float(h.score),
                    "score_breakdown": h.score_breakdown,
                    "best_frame": bf,
                    "best_asr": ba,
                }
            )

        return {
            "query": query,
            "expanded_queries": qe["all"],
            "bm25_queries": qe["bm25"],
            "translated": qe["translated"],
            "num_dense": len(dense_hits),
            "num_bm25_visual": len(bm25v),
            "num_bm25_asr": len(bm25a),
            "results": results,
            "timing_ms": {k: round(v, 1) for k, v in timing.items()},
        }
