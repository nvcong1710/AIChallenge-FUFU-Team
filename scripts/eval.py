"""Eval Recall@K + MRR trên dev set.

dev_set.json format:
[
    {"query": "...", "video_id": 1, "timestamp": 12.5},
    {"query": "...", "segment_id": 42},
    ...
]

Một item match nếu (segment_id khớp) HOẶC (cùng video_id & timestamp nằm trong segment).

Usage: python scripts/eval.py dev_set.json [--top-k 20]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backend.services.search_engine import SearchEngine
from app.common.config import get_config


def is_match(query_item: dict, result: dict) -> bool:
    if "segment_id" in query_item and result["segment_id"] == query_item["segment_id"]:
        return True
    if "video_id" in query_item and "timestamp" in query_item:
        if result["video_id"] != query_item["video_id"]:
            return False
        s, e = result.get("segment_start"), result.get("segment_end")
        if s is None or e is None:
            return False
        return s <= query_item["timestamp"] <= e
    return False


def run(dev_set_path: Path, k_values=(1, 5, 10, 20)) -> None:
    cfg = get_config()
    engine = SearchEngine(cfg)

    queries = json.loads(dev_set_path.read_text(encoding="utf-8"))
    n = len(queries)
    if not n:
        print("Dev set rỗng.")
        return

    hits = {k: 0 for k in k_values}
    rr_sum = 0.0
    top_k = max(k_values)

    for i, q in enumerate(queries, 1):
        res = engine.search(q["query"], top_k=top_k)
        rank = None
        for idx, r in enumerate(res["results"], start=1):
            if is_match(q, r):
                rank = idx
                break
        if rank:
            rr_sum += 1.0 / rank
            for k in k_values:
                if rank <= k:
                    hits[k] += 1
        print(f"[{i}/{n}] rank={rank or '∅'}  query={q['query'][:60]}")

    print("\n=== Kết quả ===")
    for k in k_values:
        print(f"Recall@{k}: {hits[k] / n:.4f}  ({hits[k]}/{n})")
    print(f"MRR:        {rr_sum / n:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dev_set", type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    run(args.dev_set, k_values=(1, 5, 10, args.top_k))


if __name__ == "__main__":
    main()
