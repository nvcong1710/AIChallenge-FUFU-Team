"""Debug what FTS query gets generated and why audio item ranks low."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backend.services.search_engine import SearchEngine
from app.backend.services.retrieval import Retriever
from app.common.config import get_config

cfg = get_config()
cfg["query_expansion"]["enable_paraphrase"] = False
cfg["query_expansion"]["enable_translation"] = False   # tắt cả để test
cfg["models"]["device"] = "cpu"

engine = SearchEngine(cfg)
qe = engine.expand_query("thị trường chứng khoán")
print(f"Expanded: {qe}")

# Test FTS direct
fts_q = engine.retriever._build_fts_or_query(qe["bm25"])
print(f"FTS query: {fts_q!r}")

bm25 = engine.retriever.bm25_asr(qe["bm25"], top_k=20)
print(f"\nbm25_asr results: {len(bm25)}")
for fid, score in bm25[:5]:
    print(f"  rowid={fid}  score={score:.3f}")

# Full search
print("\n=== Full search ===")
res = engine.search("thị trường chứng khoán", top_k=5)
for i, r in enumerate(res["results"], 1):
    p = Path(r['item_path']).name[:40]
    bd = r['score_breakdown']
    print(f"  [{i}] {r['media_type']:<5} {p:<40} score={r['score']:.3f}  d={bd.get('dense',0):.2f} v={bd.get('bm25_visual',0):.2f} a={bd.get('bm25_asr',0):.2f}")
