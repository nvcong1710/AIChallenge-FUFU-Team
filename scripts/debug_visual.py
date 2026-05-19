"""Debug visual queries failing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backend.services.search_engine import SearchEngine
from app.common.config import get_config

cfg = get_config()
cfg["query_expansion"]["enable_paraphrase"] = False
cfg["models"]["device"] = "cpu"

engine = SearchEngine(cfg)

for q, expect in [
    ("con thỏ trắng", "big_buck_bunny"),
    ("chơi cờ vua", "chess_set"),
    ("con mèo lông trắng", "cat"),
    ("hoàng hôn Hà Nội", "Sunset"),
]:
    res = engine.search(q, top_k=5)
    print(f"\n--- '{q}' (expect: {expect}) ---")
    for i, r in enumerate(res["results"][:5], 1):
        p = Path(r['item_path']).name[:40]
        bd = r['score_breakdown']
        mark = " ✓" if expect.lower() in p.lower() else ""
        print(f"  [{i}] {r['media_type']:<5} {p:<40} sc={r['score']:.3f}  d={bd.get('dense',0):.2f} a={bd.get('bm25_asr',0):.2f}{mark}")
