"""Ingest video — CHUNKED: persist mỗi N frame để không mất work khi kill midway.

Trade-off: thêm overhead ~5% (FAISS write_index() per chunk) nhưng resilient:
kill ở frame 170/555 → chỉ mất 16 frame cuối (chunk in-progress), 160 đã trên disk.

Order operations per chunk:
  annotate (caption + OCR + detection) → encode (SigLIP) → thumbnails → add_frames → persist
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ... import extractors
from ...common.audio_io import load_audio_mono_16k
from ...common.encoder import SiglipEncoder
from ...common.types import FrameAnnotation, MediaType
from ..storage import IndexWriter
from ..utils import save_thumbnail
from .keyframes import extract_keyframes_adaptive, get_video_duration
from .scenes import cluster_shots_into_scenes
from .segments import (
    assign_frames_to_segments,
    build_sliding_segments,
    shots_to_segments,
)
from .shots import detect_shots


def ingest_video(
    video_path: Path,
    encoder: SiglipEncoder,
    writer: IndexWriter,
    cfg: dict,
) -> None:
    print(f"\n[video] {video_path.name}")
    t0 = time.time()

    duration = get_video_duration(video_path)
    if duration <= 0:
        print("  ⚠ video không đọc được.")
        return

    item_id = writer.add_or_get_item(str(video_path), MediaType.VIDEO, duration=duration)
    if writer.item_already_ingested(item_id):
        print("  đã ingest trước, bỏ qua.")
        return

    vcfg = cfg["ingest"]["video"]

    # 1. Shots
    shots = detect_shots(video_path, threshold=float(vcfg["shot_detect_threshold"]))
    print(f"  shots: {len(shots)} | duration: {duration:.1f}s")
    if shots:
        shot_durs = [e - s for s, e in shots]
        print(
            f"    shot stats: min={min(shot_durs):.1f}s avg={sum(shot_durs)/len(shot_durs):.1f}s max={max(shot_durs):.1f}s"
        )

    # 2. Keyframes (load ALL upfront — bottleneck nếu video lớn, nhưng cần để encode)
    keyframes = extract_keyframes_adaptive(
        video_path,
        shots,
        density_per_sec=float(vcfg["keyframe_density_per_sec"]),
        min_per_shot=int(vcfg["min_keyframes_per_shot"]),
        max_per_shot=int(vcfg["max_keyframes_per_shot"]),
    )
    if not keyframes:
        print("  ⚠ không trích được keyframe nào.")
        return
    print(f"  keyframes: {len(keyframes)}")

    timestamps = [ts for ts, _, _ in keyframes]
    images = [img for _, img, _ in keyframes]
    shot_indices = [si for _, _, si in keyframes]

    # 3. Segments
    if vcfg.get("use_shots_as_segments", True) and shots:
        segments = shots_to_segments(shots, max_segment_len=float(vcfg["max_segment_len_sec"]))
        seg_strategy = "shots"
    else:
        segments = build_sliding_segments(
            duration,
            segment_len=float(vcfg["segment_length_sec"]),
            stride=float(vcfg["segment_stride_sec"]),
        )
        seg_strategy = "sliding"
    frame_to_segs = assign_frames_to_segments(timestamps, segments)
    seg_id_map = writer.add_segments(item_id, segments)
    print(
        f"  segments ({seg_strategy}): {len(segments)} | "
        f"frames có segment: {len(frame_to_segs)}/{len(timestamps)}"
    )

    # 4. CHUNKED annotate + encode + persist
    ocr = extractors.get_ocr(cfg)
    cap = extractors.get_caption(cfg)
    det = extractors.get_detection(cfg)

    chunk_size = int(vcfg.get("chunk_size_frames", 16))
    n_total = len(images)
    n_chunks = (n_total + chunk_size - 1) // chunk_size

    thumb_dir = Path(cfg["storage"]["thumbnail_dir"]) / video_path.stem
    thumb_dir.mkdir(parents=True, exist_ok=True)

    all_vectors_list: list[np.ndarray] = []
    n_ocr_total = 0
    n_cap_total = 0
    n_obj_total = 0
    t_anno_total = 0.0
    t_enc_total = 0.0

    for ci in range(n_chunks):
        s = ci * chunk_size
        e = min(s + chunk_size, n_total)
        chunk_imgs = images[s:e]
        chunk_ts = timestamps[s:e]
        chunk_shot_idx = shot_indices[s:e]

        # Annotate per frame
        t1 = time.time()
        chunk_anns: list[FrameAnnotation] = []
        for img in chunk_imgs:
            a = FrameAnnotation()
            ocr.annotate(img, a)
            cap.annotate(img, a)
            det.annotate(img, a)
            chunk_anns.append(a)
        t_anno_total += time.time() - t1
        n_ocr_total += sum(1 for a in chunk_anns if a.ocr_text)
        n_cap_total += sum(1 for a in chunk_anns if a.caption)
        n_obj_total += sum(len(a.objects) for a in chunk_anns)

        # SigLIP encode batch
        t1 = time.time()
        chunk_vecs = encoder.encode_images(chunk_imgs)
        t_enc_total += time.time() - t1
        all_vectors_list.append(chunk_vecs)

        # Save thumbnails
        chunk_records = []
        for i_local, img in enumerate(chunk_imgs):
            global_i = s + i_local
            p = thumb_dir / f"s{chunk_shot_idx[i_local]:04d}_f{global_i:06d}_t{chunk_ts[i_local]:.2f}.jpg"
            save_thumbnail(img, p)
            chunk_records.append(
                {
                    "timestamp": chunk_ts[i_local],
                    "thumbnail": str(p),
                    "annotation": chunk_anns[i_local],
                }
            )

        # frame_to_segs cho chunk (local index → global → seg_idx list)
        chunk_f2s = {}
        for i_local in range(len(chunk_imgs)):
            global_i = s + i_local
            if global_i in frame_to_segs:
                chunk_f2s[i_local] = frame_to_segs[global_i]

        # PERSIST CHUNK — kill sau bước này = chunk này safe
        writer.add_frames(item_id, chunk_records, chunk_vecs, chunk_f2s, seg_id_map)
        writer.persist()

        progress = f"chunk {ci+1}/{n_chunks} ({e}/{n_total})"
        print(f"  ✓ {progress}  anno={t_anno_total:.0f}s  enc={t_enc_total:.1f}s")

    print(
        f"  Annotations total: {t_anno_total:.1f}s | "
        f"OCR={n_ocr_total}/{n_total} Caption={n_cap_total}/{n_total} Objects={n_obj_total}"
    )
    print(f"  SigLIP encode total: {t_enc_total:.1f}s")

    # 5. Scene clustering (sau khi mọi frame đã có vector)
    all_vectors = np.vstack(all_vectors_list) if all_vectors_list else np.empty((0, encoder.dim))
    frames_per_shot_seg: dict[int, list[int]] = {}
    for fi, ts in enumerate(timestamps):
        for sid, ss, ee in segments:
            if ss <= ts <= ee:
                frames_per_shot_seg.setdefault(sid, []).append(fi)
                break
    scenes_clustered = cluster_shots_into_scenes(
        segments, frames_per_shot_seg, all_vectors, threshold=0.85
    )
    print(
        f"  scenes: {len(scenes_clustered)} (gom từ {len(segments)} shots, "
        f"avg {len(segments)/max(1,len(scenes_clustered)):.1f} shot/scene)"
    )
    if scenes_clustered:
        writer.add_scenes_and_link(item_id, scenes_clustered, seg_id_map)
        writer.persist()

    # 6. ASR audio track
    if vcfg.get("extract_audio_for_asr", True):
        asr = extractors.get_asr(cfg)
        if asr.enabled:
            t1 = time.time()
            try:
                waveform = load_audio_mono_16k(video_path, sample_rate=16000)
                asr_segments = asr.extract(waveform, sample_rate=16000)
                print(f"  ASR: {time.time() - t1:.1f}s | {len(asr_segments)} đoạn lời")
                writer.add_asr_segments(
                    item_id,
                    asr_segments,
                    seg_id_map=seg_id_map,
                    item_segments=segments,
                )
                writer.persist()
            except Exception as e:
                print(f"  ⚠ ASR fail: {e}")

    print(f"  ✓ tổng {time.time() - t0:.1f}s")
