"""Đánh giá Recall@K + MRR trên test_cases.json.

Match logic: result.item_path chứa substring `expect` (case-insensitive).
Có phân loại theo channel (visual / asr / visual_en) để báo cáo riêng từng nhóm.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backend.services.search_engine import SearchEngine
from app.common.config import get_config


def is_match(r: dict, expect: str) -> bool:
    """Match nếu item_path chứa substring expect (lower, ignore unicode normalization)."""
    path = (r.get("item_path") or "").lower()
    asr_text = ""
    if r.get("best_asr"):
        asr_text = (r["best_asr"].get("text") or "").lower()
    return expect.lower() in path or expect.lower() in asr_text


def eval_one(engine: SearchEngine, query: str, expect: str, top_k: int = 20):
    t0 = time.time()
    res = engine.search(query, top_k=top_k)
    elapsed_ms = (time.time() - t0) * 1000
    rank = None
    top_paths = []
    for i, r in enumerate(res["results"], 1):
        item_basename = Path(r.get("item_path") or "").name
        top_paths.append(f"{r['media_type'][0]}:{item_basename[:30]}")
        if is_match(r, expect):
            rank = i
            break
    return rank, top_paths[:3], elapsed_ms, res


def main():
    cfg = get_config()
    # Override settings to disable paraphrase (slow on CPU). Translation cũng off
    # cho fast run; nếu muốn so sánh, đổi enable_translation thành True.
    cfg["query_expansion"]["enable_paraphrase"] = False
    enable_translation = "--translate" in sys.argv
    cfg["query_expansion"]["enable_translation"] = enable_translation
    cfg["models"]["device"] = "cpu"
    print(f"  Translation: {'ON' if enable_translation else 'OFF'}")

    # Default: test_cases.json. Override với --cases <path> để dùng MSR-VTT etc.
    cases_path = Path(__file__).resolve().parents[1] / "scripts" / "test_cases.json"
    for i, arg in enumerate(sys.argv):
        if arg == "--cases" and i + 1 < len(sys.argv):
            cases_path = Path(sys.argv[i + 1])
    print(f"  Test cases: {cases_path.name}")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    n = len(cases)
    print(f"Loading SearchEngine (CPU)...")
    engine = SearchEngine(cfg)
    print(f"\nRunning {n} test cases...\n")

    K_VALUES = (1, 5, 10, 20)
    hits_overall = {k: 0 for k in K_VALUES}
    hits_by_channel: dict[str, dict[int, int]] = defaultdict(lambda: {k: 0 for k in K_VALUES})
    total_by_channel: dict[str, int] = defaultdict(int)
    rr_overall = 0.0
    rr_by_channel: dict[str, float] = defaultdict(float)
    timing_total = 0.0
    failures = []

    for i, case in enumerate(cases, 1):
        q = case["q"]
        expect = case["expect"]
        channel = case.get("channel", "?")
        total_by_channel[channel] += 1

        rank, top3, ms, res = eval_one(engine, q, expect)
        timing_total += ms

        if rank:
            rr_overall += 1.0 / rank
            rr_by_channel[channel] += 1.0 / rank
            for k in K_VALUES:
                if rank <= k:
                    hits_overall[k] += 1
                    hits_by_channel[channel][k] += 1
            mark = f"@{rank:>2}"
        else:
            mark = "∅  "
            failures.append((q, expect, channel, top3))

        print(f"  [{i:>2}/{n}] {mark} ({channel:<10}) {q[:50]:<50}  → {top3[0] if top3 else '?'}")

    # Summary
    print(f"\n{'='*70}")
    print(f"OVERALL ({n} queries)")
    print('='*70)
    for k in K_VALUES:
        pct = 100 * hits_overall[k] / n
        print(f"  Recall@{k:<2}: {hits_overall[k]:>2}/{n}  ({pct:>5.1f}%)")
    print(f"  MRR:       {rr_overall / n:.4f}")
    print(f"  Avg ms:    {timing_total / n:.0f} ms / query")

    print(f"\nBy channel:")
    for channel in sorted(total_by_channel):
        cn = total_by_channel[channel]
        print(f"\n  {channel} ({cn} queries):")
        for k in K_VALUES:
            ch = hits_by_channel[channel][k]
            pct = 100 * ch / cn
            print(f"    R@{k:<2}: {ch:>2}/{cn} ({pct:>5.1f}%)")
        print(f"    MRR:  {rr_by_channel[channel] / cn:.4f}")

    if failures:
        print(f"\n{'='*70}\nFailures ({len(failures)}):")
        for q, expect, channel, top3 in failures:
            print(f"  [{channel}] '{q}' expected '{expect}'")
            for t in top3:
                print(f"      top: {t}")


if __name__ == "__main__":
    main()
