"""Ingest 1 audio file.

Chiến lược: ASR chunks làm segments (mỗi đoạn lời = 1 'cảnh') — analogous với
shots-as-segments của video. Fallback sliding window nếu audio không có speech.
"""

from __future__ import annotations

import time
from pathlib import Path

from ... import extractors
from ...common.audio_io import audio_duration, load_audio_mono_16k
from ...common.types import MediaType
from ..storage import IndexWriter
from .segments import asr_chunks_to_segments, build_sliding_segments, merge_close_chunks


def ingest_audio(path: Path, writer: IndexWriter, cfg: dict) -> None:
    print(f"\n[audio] {path.name}")
    t0 = time.time()

    sr = int(cfg["ingest"]["audio"].get("target_sample_rate", 16000))
    duration = audio_duration(path)
    if duration <= 0:
        print("  ⚠ không đọc được duration.")
        return

    item_id = writer.add_or_get_item(str(path), MediaType.AUDIO, duration=duration)
    if writer.item_already_ingested(item_id):
        print("  đã ingest trước, bỏ qua.")
        return
    print(f"  duration: {duration:.1f}s")

    # 1. Load waveform
    try:
        waveform = load_audio_mono_16k(path, sample_rate=sr)
    except Exception as e:
        print(f"  ⚠ load audio fail: {e}")
        return

    # 2. ASR
    asr = extractors.get_asr(cfg)
    if not asr.enabled:
        print("  ASR disabled — audio này không index được. Bỏ qua.")
        return

    t1 = time.time()
    asr_segments = asr.extract(waveform, sample_rate=sr)
    print(f"  ASR: {time.time() - t1:.1f}s | {len(asr_segments)} đoạn lời thô")

    # 3. Merge close chunks (tránh phân mảnh do Whisper cắt vụn)
    acfg = cfg["ingest"]["audio"]
    if asr_segments and acfg.get("merge_close_chunks_sec", 0) > 0:
        merged = merge_close_chunks(asr_segments, max_gap_sec=float(acfg["merge_close_chunks_sec"]))
        if len(merged) < len(asr_segments):
            print(f"    merged {len(asr_segments)} → {len(merged)} (gap ≤ {acfg['merge_close_chunks_sec']}s)")
        asr_segments = merged

    # 4. Segments = ASR chunks (hoặc sliding window fallback)
    if asr_segments and acfg.get("use_asr_as_segments", True):
        item_segs = asr_chunks_to_segments(
            asr_segments,
            max_segment_len=float(acfg.get("max_segment_len_sec", 15.0)),
        )
        seg_strategy = "asr_chunks"
        if asr_segments:
            durs = [a.end - a.start for a in asr_segments]
            print(
                f"    speech stats: min={min(durs):.1f}s avg={sum(durs)/len(durs):.1f}s "
                f"max={max(durs):.1f}s | total speech: {sum(durs):.1f}s / {duration:.1f}s "
                f"({100*sum(durs)/duration:.0f}%)"
            )
    else:
        item_segs = build_sliding_segments(
            duration,
            segment_len=float(acfg["segment_length_sec"]),
            stride=float(acfg["segment_stride_sec"]),
        )
        seg_strategy = "sliding"
        if not asr_segments:
            print("    ⚠ không phát hiện speech — dùng sliding window (audio sẽ không retrievable qua text query).")

    seg_id_map = writer.add_segments(item_id, item_segs)
    print(f"  segments ({seg_strategy}): {len(item_segs)}")

    # 5. Index ASR
    writer.add_asr_segments(item_id, asr_segments, seg_id_map=seg_id_map, item_segments=item_segs)
    writer.persist()
    print(f"  ✓ {time.time() - t0:.1f}s")
