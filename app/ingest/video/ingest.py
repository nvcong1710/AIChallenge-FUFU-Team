"""Ingest video: shots → adaptive keyframes → SigLIP + OCR + detection (+ caption nếu bật) → ASR → FAISS + SQL."""

from __future__ import annotations

import time
from pathlib import Path

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

    # 1. Shots — bắt đầu / kết thúc cảnh
    shots = detect_shots(video_path, threshold=float(vcfg["shot_detect_threshold"]))
    print(f"  shots: {len(shots)} | duration: {duration:.1f}s")
    if shots:
        shot_durs = [e - s for s, e in shots]
        print(
            f"    shot stats: min={min(shot_durs):.1f}s "
            f"avg={sum(shot_durs)/len(shot_durs):.1f}s "
            f"max={max(shot_durs):.1f}s"
        )

    # 2. Keyframes — adaptive density
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

    # 3. Segments — shots hoặc sliding window
    if vcfg.get("use_shots_as_segments", True) and shots:
        segments = shots_to_segments(
            shots, max_segment_len=float(vcfg["max_segment_len_sec"])
        )
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

    # 4. Visual extractors per frame
    ocr = extractors.get_ocr(cfg)
    cap = extractors.get_caption(cfg)
    det = extractors.get_detection(cfg)

    annotations = []
    t1 = time.time()
    for i, img in enumerate(images):
        a = FrameAnnotation()
        ocr.annotate(img, a)
        cap.annotate(img, a)   # no-op nếu disable_caption
        det.annotate(img, a)
        annotations.append(a)
        if (i + 1) % 10 == 0 or i + 1 == len(images):
            print(f"    annotated {i+1}/{len(images)}", end="\r", flush=True)
    print()
    n_ocr = sum(1 for a in annotations if a.ocr_text)
    n_cap = sum(1 for a in annotations if a.caption)
    n_obj = sum(len(a.objects) for a in annotations)
    print(
        f"  Annotations: {time.time() - t1:.1f}s | "
        f"OCR={n_ocr}/{len(images)} Caption={n_cap}/{len(images)} Objects={n_obj}"
    )

    # 5. SigLIP encode
    t1 = time.time()
    vectors = encoder.encode_images(images)
    print(f"  SigLIP encode: {time.time() - t1:.1f}s  shape={vectors.shape}")

    # 5b. Cluster shots → scenes (cosine giữa frame biên của shots liền kề)
    # Map shot_seg_idx → list[vector_idx]
    frames_per_shot_seg: dict[int, list[int]] = {}
    for fi, ts in enumerate(timestamps):
        for sid, s, e in segments:
            if s <= ts <= e:
                frames_per_shot_seg.setdefault(sid, []).append(fi)
                break
    scenes_clustered = cluster_shots_into_scenes(
        segments, frames_per_shot_seg, vectors, threshold=0.85
    )
    print(
        f"  scenes: {len(scenes_clustered)} (gom từ {len(segments)} shots, "
        f"avg {len(segments)/max(1,len(scenes_clustered)):.1f} shot/scene)"
    )

    # 6. Thumbnails
    thumb_dir = Path(cfg["storage"]["thumbnail_dir"]) / video_path.stem
    frame_records = []
    for i, img in enumerate(images):
        p = thumb_dir / f"s{shot_indices[i]:04d}_f{i:06d}_t{timestamps[i]:.2f}.jpg"
        save_thumbnail(img, p)
        frame_records.append(
            {"timestamp": timestamps[i], "thumbnail": str(p), "annotation": annotations[i]}
        )

    writer.add_frames(item_id, frame_records, vectors, frame_to_segs, seg_id_map)

    # 6b. Persist scenes + link shots → scene_id
    if scenes_clustered:
        writer.add_scenes_and_link(item_id, scenes_clustered, seg_id_map)

    # 7. ASR audio track
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
            except Exception as e:
                print(f"  ⚠ ASR fail: {e}")

    writer.persist()
    print(f"  ✓ tổng {time.time() - t0:.1f}s")
