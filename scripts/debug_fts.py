"""Debug FTS5 BM25 queries directly."""
import sqlite3
from pathlib import Path

db = sqlite3.connect(str(Path(__file__).resolve().parents[1] / "data" / "meta.sqlite"))

queries = [
    'thị trường chứng khoán',
    'chứng khoán',
    'phở bò',
    'phạm minh chính',
    'vịnh hạ long',
    'bạch mai',
    'sơn tùng',
]

for q in queries:
    safe = q.replace('"', '""')
    fts = f'"{safe}"'
    try:
        rows = db.execute(
            "SELECT rowid, bm25(asr_text), substr(transcript, 1, 80) FROM asr_text "
            "WHERE asr_text MATCH ? ORDER BY bm25(asr_text) LIMIT 3",
            (fts,)
        ).fetchall()
        print(f"\nQ: '{q}' (as FTS phrase {fts!r})")
        for r in rows:
            print(f"  rowid={r[0]}  bm25={r[1]:.2f}  text={r[2]}")
        if not rows:
            print("  (no match)")
    except sqlite3.OperationalError as e:
        print(f"\nQ: '{q}' → ERROR {e}")

    # Try without phrase quoting (token AND)
    try:
        rows = db.execute(
            "SELECT rowid, bm25(asr_text), substr(transcript, 1, 80) FROM asr_text "
            "WHERE asr_text MATCH ? ORDER BY bm25(asr_text) LIMIT 3",
            (q,)
        ).fetchall()
        print(f"  (token AND): {len(rows)} rows")
        for r in rows[:2]:
            print(f"    rowid={r[0]}  bm25={r[1]:.2f}  text={r[2]}")
    except sqlite3.OperationalError as e:
        print(f"  token AND ERROR: {e}")
