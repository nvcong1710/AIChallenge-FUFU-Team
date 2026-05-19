"""Dịch test_cases_msrvtt.json (English captions) → tiếng Việt qua NLLB.

Output: test_cases_msrvtt_vn.json — same structure nhưng query dịch sang VN.
Mục đích: test khả năng query VN của hệ thống trên dataset chuẩn quốc tế.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "scripts" / "test_cases_msrvtt.json"
OUT_PATH = ROOT / "scripts" / "test_cases_msrvtt_vn.json"


def main():
    if not IN_PATH.exists():
        print(f"⚠ {IN_PATH} không tồn tại. Chạy download_msrvtt.py trước.")
        sys.exit(1)

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Load NLLB-200 ({device})...")
    tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-600M",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    cases = json.loads(IN_PATH.read_text(encoding="utf-8"))
    print(f"Dịch {len(cases)} captions EN → VN...")

    tok.src_lang = "eng_Latn"
    out_cases = []
    t0 = time.time()
    for i, c in enumerate(cases):
        en = c["q"]
        inputs = tok(en, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                forced_bos_token_id=tok.convert_tokens_to_ids("vie_Latn"),
                max_length=128,
                num_beams=2,
            )
        vn = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
        out_cases.append({**c, "q": vn, "q_original_en": en, "channel": c.get("channel", "") + "_vn"})
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(cases)}] '{en[:50]}' → '{vn[:50]}'")

    OUT_PATH.write_text(json.dumps(out_cases, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ {len(out_cases)} VN cases → {OUT_PATH} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
