"""Debug specific failing ASR queries."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backend.services.search_engine import SearchEngine
from app.common.config import get_config

cfg = get_config()
cfg["query_expansion"]["enable_paraphrase"] = False
cfg["query_expansion"]["enable_translation"] = False
cfg["models"]["device"] = "cpu"

engine = SearchEngine(cfg)

queries = [
    ("Thủ tướng Phạm Minh Chính", "tin_chinh_tri"),
    ("vịnh Hạ Long du lịch", "tin_du_lich"),
    ("Tết Nguyên Đán bánh chưng", "ke_chuyen_van_hoa"),
    ("bệnh viện Bạch Mai ung thư", "tin_y_te"),
    ("Sơn Tùng MTP album", "tin_giai_tri"),
]

for q, expect in queries:
    print(f"\n--- '{q}' (expect: {expect}) ---")
    qe = engine.expand_query(q)
    bm25_a = engine.retriever.bm25_asr(qe["bm25"], top_k=10)
    print(f"  bm25_asr rows: {len(bm25_a)}")
    for fid, sc in bm25_a[:3]:
        with engine.retriever._conn() as conn:
            r = conn.execute("SELECT i.path, substr(a.text, 1, 70) FROM asr_segments a JOIN items i ON a.item_id=i.id WHERE a.id=?", (fid,)).fetchone()
        if r:
            print(f"    asr_id={fid} score={sc:.2f}  {Path(r[0]).name[:30]}  text='{r[1]}'")

    res = engine.search(q, top_k=10)
    print(f"  Top 5:")
    for i, r in enumerate(res["results"][:5], 1):
        p = Path(r['item_path']).name[:35]
        bd = r['score_breakdown']
        mark = " ✓" if expect.lower() in p.lower() else ""
        print(f"    [{i}] {r['media_type']:<5} {p:<35} sc={r['score']:.3f}  d={bd.get('dense',0):.2f} a={bd.get('bm25_asr',0):.2f}{mark}")
