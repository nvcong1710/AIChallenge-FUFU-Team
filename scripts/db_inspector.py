import sqlite3
import sys
import argparse
from pathlib import Path

def check_state(db_path: Path):
    db = sqlite3.connect(str(db_path))
    print(f"=== DATABASE STATE: {db_path} ===")
    print('Items by type:')
    for r in db.execute("SELECT media_type, COUNT(*) FROM items GROUP BY media_type"):
        print(f'  {r[0]:>5}: {r[1]}')
    print(f"Frames: {db.execute('SELECT COUNT(*) FROM frames').fetchone()[0]}")
    print(f"Segments: {db.execute('SELECT COUNT(*) FROM segments').fetchone()[0]}")
    print(f"Scenes: {db.execute('SELECT COUNT(*) FROM scenes').fetchone()[0]}")
    print(f"ASR segs: {db.execute('SELECT COUNT(*) FROM asr_segments').fetchone()[0]}")
    print(f"asr_text FTS rows: {db.execute('SELECT COUNT(*) FROM asr_text').fetchone()[0]}")
    print(f"frame_text FTS rows: {db.execute('SELECT COUNT(*) FROM frame_text').fetchone()[0]}")
    
    print('\nSample ASR text (first 5):')
    for r in db.execute("SELECT i.path, a.text FROM asr_segments a JOIN items i ON a.item_id=i.id LIMIT 5"):
        print(f"  [{Path(r[0]).name[:40]}] {(r[1] or '')[:80]}")

def check_asr(db_path: Path):
    db = sqlite3.connect(str(db_path))
    print('=== ALL ASR TRANSCRIPTS ===')
    for r in db.execute("""
        SELECT i.path, a.text, a.start_sec, a.end_sec
        FROM asr_segments a
        JOIN items i ON a.item_id = i.id
        ORDER BY i.path, a.start_sec
    """):
        name = Path(r[0]).name[:38]
        print(f"  [{name:<38}] {(r[1] or '')[:100]}")

def debug_fts(db_path: Path, queries: list[str]):
    db = sqlite3.connect(str(db_path))
    print("=== DEBUG FTS ===")
    if not queries:
        queries = [
            'thị trường chứng khoán', 'phở bò', 'phạm minh chính', 'vịnh hạ long'
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

def main():
    parser = argparse.ArgumentParser(description="Inspect FUFU SQLite Database")
    parser.add_argument("--db", type=str, help="Path to meta.sqlite")
    parser.add_argument("action", choices=["state", "asr", "fts"], help="Action to perform")
    parser.add_argument("--queries", nargs="*", help="Queries for fts debug")
    args = parser.parse_args()
    
    db_path = Path(args.db) if args.db else Path(__file__).resolve().parents[1] / "data" / "meta.sqlite"
    
    if args.action == "state":
        check_state(db_path)
    elif args.action == "asr":
        check_asr(db_path)
    elif args.action == "fts":
        debug_fts(db_path, args.queries)

if __name__ == "__main__":
    main()