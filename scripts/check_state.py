"""Quick query DB state."""
import sqlite3
import sys
from pathlib import Path

db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data" / "meta.sqlite"
db = sqlite3.connect(str(db_path))
print(f"DB: {db_path}\n")
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
