"""Dump all ASR transcripts."""
import sqlite3
import sys
from pathlib import Path

db = sqlite3.connect(str(Path(__file__).resolve().parents[1] / "data" / "meta.sqlite"))
print('All ASR transcripts:\n')
for r in db.execute("""
    SELECT i.path, a.text, a.start_sec, a.end_sec
    FROM asr_segments a
    JOIN items i ON a.item_id = i.id
    ORDER BY i.path, a.start_sec
"""):
    name = Path(r[0]).name[:38]
    print(f"  [{name:<38}] {(r[1] or '')[:100]}")
