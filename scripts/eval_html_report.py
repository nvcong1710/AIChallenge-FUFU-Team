"""Sinh HTML report đầy đủ thumbnails để xem result từng test case."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backend.services.search_engine import SearchEngine
from app.common.config import get_config


ROOT = Path(__file__).resolve().parents[1]
OUT_HTML = ROOT / "eval_report.html"


def is_match(r: dict, expect: str) -> bool:
    path = (r.get("item_path") or "").lower()
    asr_text = ""
    if r.get("best_asr"):
        asr_text = (r["best_asr"].get("text") or "").lower()
    return expect.lower() in path or expect.lower() in asr_text


def thumb_file_url(thumb_path: str | None) -> str:
    """Return relative URL (data/thumbnails/...) URL-encoded.

    Chrome chặn file:// → file:// cross-origin nên file:/// URLs không load được.
    Solution: dùng path tương đối, người dùng chạy `python -m http.server` từ project root.
    """
    if not thumb_path:
        return ""
    import urllib.parse
    p = str(thumb_path).replace("\\", "/")
    idx = p.find("/thumbnails/")
    if idx == -1:
        return ""
    rel = p[idx + len("/thumbnails/"):]
    # Verify file tồn tại local
    local = ROOT / "data" / "thumbnails" / rel
    if not local.exists():
        return ""
    # URL-encode từng path segment (giữ slash)
    encoded = "/".join(urllib.parse.quote(seg) for seg in rel.split("/"))
    return f"data/thumbnails/{encoded}"


def render_card(r: dict, rank: int, expected: str) -> str:
    media_type = r.get("media_type", "?")
    icon = {"video": "🎥", "audio": "🎵", "image": "🖼"}.get(media_type, "?")
    item_path = r.get("item_path") or ""
    item_name = Path(item_path).name
    hit = is_match(r, expected)
    cls = "card hit" if hit else "card"

    thumb_html = ""
    if media_type in ("video", "image"):
        tu = thumb_file_url(r.get("best_frame", {}).get("thumbnail"))
        if tu:
            thumb_html = f'<img src="{html.escape(tu)}" alt="">'
        else:
            thumb_html = '<div class="no-thumb">(no thumb)</div>'
    else:
        thumb_html = '<div class="audio-icon">🎵</div>'

    bd = r.get("score_breakdown", {})
    bf = r.get("best_frame") or {}
    ba = r.get("best_asr") or {}

    body_lines = []
    body_lines.append(f'<div class="row"><span class="rank">{icon} #{rank}</span><span class="score">{r["score"]:.3f}</span></div>')
    body_lines.append(f'<div class="name" title="{html.escape(item_path)}">{html.escape(item_name)}</div>')
    if r.get("segment_start") is not None:
        body_lines.append(f'<div class="ts">{r["segment_start"]:.1f}s – {r["segment_end"]:.1f}s</div>')
    if bf.get("caption"):
        body_lines.append(f'<div class="caption">{html.escape(bf["caption"][:120])}</div>')
    if ba.get("text"):
        body_lines.append(f'<div class="asr">🗣 {html.escape(ba["text"][:150])}</div>')
    if bf.get("objects"):
        labels = ", ".join(sorted({o["label"] for o in bf["objects"]}))
        body_lines.append(f'<div class="objs">📦 {html.escape(labels[:80])}</div>')
    body_lines.append(
        f'<div class="bd">d:{bd.get("dense",0):.2f} v:{bd.get("bm25_visual",0):.2f} a:{bd.get("bm25_asr",0):.2f}</div>'
    )

    return f'<div class="{cls}">{thumb_html}<div class="info">{"".join(body_lines)}</div></div>'


HTML_TEMPLATE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<title>FUFU Eval Report</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; background:#0b1220; color:#e2e8f0;
         margin:0; padding:20px; }}
  h1 {{ font-size:22px; margin:0 0 6px; }}
  .stats {{ background:#1e293b; padding:14px 18px; border-radius:8px; margin-bottom:24px; }}
  .stats table {{ border-collapse:collapse; }}
  .stats td {{ padding:4px 16px 4px 0; font-family:ui-monospace,monospace; font-size:13px; }}
  .stats td.label {{ color:#94a3b8; }}
  .case {{ background:#111827; border-radius:8px; padding:14px; margin-bottom:18px; }}
  .case h3 {{ font-size:14px; margin:0 0 4px; }}
  .case .meta {{ font-size:12px; color:#94a3b8; margin-bottom:10px;
                font-family:ui-monospace,monospace; }}
  .case .meta .rank-ok {{ color:#34d399; font-weight:bold; }}
  .case .meta .rank-bad {{ color:#f87171; }}
  .case .meta .channel {{ display:inline-block; padding:1px 7px; border-radius:999px;
                          background:#312e81; color:#c7d2fe; margin-right:8px; }}
  .grid {{ display:grid; grid-template-columns:repeat(5, 1fr); gap:8px; }}
  .card {{ background:#1e293b; border-radius:6px; overflow:hidden;
           border:2px solid transparent; }}
  .card.hit {{ border-color:#10b981; box-shadow:0 0 12px rgba(16,185,129,.4); }}
  .card img {{ width:100%; aspect-ratio:16/9; object-fit:cover; display:block;
               background:#0b1220; }}
  .card .no-thumb {{ aspect-ratio:16/9; background:#0b1220; display:flex;
                     align-items:center; justify-content:center; color:#475569; font-size:11px;}}
  .card .audio-icon {{ aspect-ratio:16/9; background:#312e81; display:flex;
                       align-items:center; justify-content:center; font-size:40px; }}
  .card .info {{ padding:6px 8px; font-size:11px; line-height:1.4; }}
  .card .row {{ display:flex; justify-content:space-between; }}
  .card .rank {{ color:#fbbf24; font-weight:bold; }}
  .card .score {{ color:#94a3b8; font-family:ui-monospace,monospace; }}
  .card .name {{ margin-top:3px; font-size:10px; color:#cbd5e1;
                 overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .card .ts {{ color:#94a3b8; font-size:10px; margin-top:3px;
               font-family:ui-monospace,monospace; }}
  .card .caption {{ margin-top:4px; color:#e2e8f0; font-size:10px; }}
  .card .asr {{ margin-top:4px; color:#a5f3fc; font-size:10px; font-style:italic;
                max-height:48px; overflow:hidden; }}
  .card .objs {{ margin-top:3px; color:#fcd34d; font-size:10px; }}
  .card .bd {{ margin-top:4px; color:#64748b; font-size:9px;
               font-family:ui-monospace,monospace; }}
</style></head><body>
<h1>FUFU Eval Report</h1>
<div class="stats">
  <table>
    <tr><td class="label">Test cases:</td><td>{n}</td></tr>
    <tr><td class="label">Recall@1:</td><td>{r1}/{n} ({r1p:.1f}%)</td></tr>
    <tr><td class="label">Recall@5:</td><td>{r5}/{n} ({r5p:.1f}%)</td></tr>
    <tr><td class="label">Recall@10:</td><td>{r10}/{n} ({r10p:.1f}%)</td></tr>
    <tr><td class="label">MRR:</td><td>{mrr:.4f}</td></tr>
    <tr><td class="label">Latency:</td><td>{ms:.0f} ms / query (CPU)</td></tr>
  </table>
</div>
{cases}
</body></html>"""


def main():
    cfg = get_config()
    cfg["query_expansion"]["enable_paraphrase"] = False
    cfg["query_expansion"]["enable_translation"] = False
    cfg["models"]["device"] = "cpu"

    # Default: test_cases.json. Override với --cases <path>
    cases_path = ROOT / "scripts" / "test_cases.json"
    for i, arg in enumerate(sys.argv):
        if arg == "--cases" and i + 1 < len(sys.argv):
            cases_path = Path(sys.argv[i + 1])
    print(f"Test cases: {cases_path.name}")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    n = len(cases)
    print(f"Loading SearchEngine (CPU)...")
    engine = SearchEngine(cfg)
    print(f"Running {n} queries + generating HTML...\n")

    hits = {1: 0, 5: 0, 10: 0}
    rr_sum = 0.0
    total_ms = 0.0
    cards_html = []

    for i, c in enumerate(cases, 1):
        q = c["q"]
        expect = c["expect"]
        channel = c.get("channel", "?")

        import time
        t0 = time.time()
        res = engine.search(q, top_k=10)
        ms = (time.time() - t0) * 1000
        total_ms += ms

        rank = None
        for r_idx, r in enumerate(res["results"], 1):
            if is_match(r, expect):
                rank = r_idx
                break
        if rank:
            rr_sum += 1.0 / rank
            for k in (1, 5, 10):
                if rank <= k:
                    hits[k] += 1

        rank_html = (
            f'<span class="rank-ok">✓ rank #{rank}</span>' if rank
            else '<span class="rank-bad">✗ no hit in top-10</span>'
        )
        result_cards = "".join(render_card(r, i_r + 1, expect) for i_r, r in enumerate(res["results"][:5]))

        cards_html.append(f"""
<div class="case">
  <h3>[{i}/{n}] {html.escape(q)}</h3>
  <div class="meta">
    <span class="channel">{channel}</span>
    expect:<b>{html.escape(expect)}</b> &middot; {rank_html} &middot; {ms:.0f}ms
  </div>
  <div class="grid">{result_cards}</div>
</div>""")

        mark = f"@{rank:>2}" if rank else "∅  "
        print(f"  [{i:>2}/{n}] {mark} ({channel:<10}) {q[:50]}")

    final_html = HTML_TEMPLATE.format(
        n=n,
        r1=hits[1], r1p=100*hits[1]/n,
        r5=hits[5], r5p=100*hits[5]/n,
        r10=hits[10], r10p=100*hits[10]/n,
        mrr=rr_sum / n,
        ms=total_ms / n,
        cases="\n".join(cards_html),
    )

    OUT_HTML.write_text(final_html, encoding="utf-8")
    print(f"\n✓ Report: {OUT_HTML}")
    print(f"\nĐể xem ảnh, chạy http server:")
    print(f"  cd {ROOT}")
    print(f"  py -3.10 -m http.server 8765")
    print(f"  → mở http://localhost:8765/eval_report.html")


if __name__ == "__main__":
    main()
