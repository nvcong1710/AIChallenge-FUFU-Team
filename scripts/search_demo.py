"""CLI demo: python scripts/search_demo.py "câu query" [top_k]."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backend.services.search_engine import SearchEngine
from app.common.config import get_config


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/search_demo.py "câu query" [top_k]')
        sys.exit(1)
    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    engine = SearchEngine(get_config())
    res = engine.search(query, top_k=top_k)

    print(f"\n=== QUERY: {res['query']} ===")
    print(f"All variants (dense): {res['expanded_queries']}")
    print(f"BM25 queries (OR):    {res.get('bm25_queries', [])}")
    if res.get("translated"):
        print(f"Translated EN: {res['translated']}")
    print(
        f"Channels: dense={res['num_dense']} bm25_v={res['num_bm25_visual']} bm25_a={res['num_bm25_asr']}"
    )
    print(f"Timing: {json.dumps(res['timing_ms'])}\n")

    if not res["results"]:
        print("Không có kết quả.")
        return

    for i, r in enumerate(res["results"], 1):
        icon = {"video": "🎥", "audio": "🎵", "image": "🖼"}.get(r["media_type"], "?")
        print(f"[{i:>2}] {icon} score={r['score']:.4f}  item={r['item_id']}  type={r['media_type']}")
        print(f"     file: {Path(r['item_path']).name}")
        if r["segment_start"] is not None:
            print(f"     seg:  {r['segment_start']:.1f}s — {r['segment_end']:.1f}s")
        if r["best_frame"]:
            bf = r["best_frame"]
            extra = []
            if bf.get("raw_cosine") is not None:
                extra.append(f"raw_cos={bf['raw_cosine']:.4f}")
            if bf.get("timestamp") is not None:
                extra.append(f"ts={bf['timestamp']:.2f}s")
            print(f"     frame: {' '.join(extra)}")
            if bf.get("caption"):
                print(f"     caption: {bf['caption']}")
            if bf.get("objects"):
                labels = ", ".join(sorted({o['label'] for o in bf['objects']}))
                print(f"     objects: {labels}")
        if r["best_asr"]:
            print(f"     asr: [{r['best_asr']['start']:.1f}-{r['best_asr']['end']:.1f}s] {r['best_asr']['text']}")
        b = r["score_breakdown"]
        print(f"     breakdown: dense={b.get('dense',0):.2f} bm25_v={b.get('bm25_visual',0):.2f} bm25_a={b.get('bm25_asr',0):.2f}")
        print()


if __name__ == "__main__":
    main()
