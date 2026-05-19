"""Tải MSR-VTT subset từ Kaggle + sinh ground-truth test cases.

MSR-VTT = de-facto standard cho text→video retrieval:
  - 10K video clips, mỗi clip ~20s
  - 200K natural language captions (~20 per video)
  - Test split: 1000 videos đã có benchmark R@1/R@5/R@10

Workflow:
  1) Download MSR-VTT từ Kaggle (cần kaggle.json API key)
  2) Chọn subset 200-500 videos
  3) Sinh test_cases_msrvtt.json từ captions (5 query / video)
  4) Eval: Recall@K so sánh được với literature

Yêu cầu: pip install kaggle + kaggle.json setup
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = ROOT / "test-data"
VIDEO_DIR = TEST_DATA / "videos" / "msrvtt"
OUT_TEST_CASES = ROOT / "scripts" / "test_cases_msrvtt.json"

KAGGLE_DATASET = "vtorosyan/msrvtt-dataset"
SUBSET_N_VIDEOS = 200    # cap để tiết kiệm disk (~200MB)
QUERIES_PER_VIDEO = 3    # 3 caption khác nhau / video làm query


def kaggle_available() -> bool:
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def download_msrvtt() -> Path:
    """Tải MSR-VTT về test-data/_msrvtt_raw/."""
    raw_dir = TEST_DATA / "_msrvtt_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not kaggle_available():
        print("⚠ Kaggle CLI chưa cài. Setup:")
        print("  1) pip install kaggle")
        print("  2) Tải kaggle.json từ https://www.kaggle.com/settings")
        print("     Đặt: ~/.kaggle/kaggle.json (Linux) hoặc C:/Users/<user>/.kaggle/kaggle.json")
        sys.exit(1)

    print(f"\n[1/3] Download MSR-VTT từ Kaggle: {KAGGLE_DATASET}")
    print("  (lần đầu ~7GB; sẽ extract subset 200 videos)")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(raw_dir), "--unzip"],
        check=True,
    )
    return raw_dir


def find_videos_and_captions(raw_dir: Path):
    """MSR-VTT structure thường có: TrainValVideo/videos/*.mp4 + annotations JSON."""
    # Tìm file annotation (videodatainfo / captions json)
    ann_files = list(raw_dir.rglob("*.json"))
    print(f"\n[2/3] Tìm thấy {len(ann_files)} JSON file(s).")
    captions_map = {}  # video_id → list[caption]

    for af in ann_files:
        try:
            data = json.loads(af.read_text(encoding="utf-8"))
        except Exception:
            continue
        # MSR-VTT JSON format: {"sentences": [{"video_id": "video0", "caption": "..."}]}
        if isinstance(data, dict) and "sentences" in data:
            for s in data["sentences"]:
                vid = s.get("video_id")
                cap = (s.get("caption") or "").strip()
                if vid and cap:
                    captions_map.setdefault(vid, []).append(cap)
            print(f"  ✓ {af.name}: {len(data['sentences'])} captions")
            break

    if not captions_map:
        print("  ⚠ Không parse được captions. Kiểm tra file structure manually.")
        sys.exit(1)

    # Tìm video files
    video_files = list(raw_dir.rglob("*.mp4"))
    if not video_files:
        print("  ⚠ Không tìm thấy video .mp4 nào.")
        sys.exit(1)
    print(f"  ✓ {len(video_files)} video files trong raw_dir")

    return captions_map, video_files


def copy_subset_and_build_test_cases(captions_map, video_files, n_videos: int):
    """Copy n_videos vào VIDEO_DIR + sinh test_cases JSON."""
    print(f"\n[3/3] Tạo subset {n_videos} videos + test cases")
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    # Match video file với caption map: video filename là 'videoN.mp4' (N = số)
    video_by_id = {}
    for vf in video_files:
        vid_id = vf.stem  # 'video123'
        if vid_id in captions_map:
            video_by_id[vid_id] = vf

    if not video_by_id:
        print("  ⚠ Không có video nào match với captions. Check file naming.")
        sys.exit(1)

    available_ids = sorted(video_by_id.keys())
    random.seed(42)
    random.shuffle(available_ids)
    chosen = available_ids[:n_videos]
    print(f"  chọn {len(chosen)}/{len(available_ids)} videos available")

    test_cases = []
    for vid_id in chosen:
        src = video_by_id[vid_id]
        dst = VIDEO_DIR / f"msrvtt_{vid_id}.mp4"
        if not dst.exists():
            shutil.copy2(src, dst)
        caps = captions_map[vid_id]
        random.shuffle(caps)
        # Lấy 3 caption đa dạng làm query, expect = filename substring
        for cap in caps[:QUERIES_PER_VIDEO]:
            test_cases.append({
                "q": cap,
                "expect": f"msrvtt_{vid_id}",
                "channel": "visual_msrvtt",
            })

    OUT_TEST_CASES.write_text(
        json.dumps(test_cases, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    total_size = sum(p.stat().st_size for p in VIDEO_DIR.glob("*.mp4")) / 1024 / 1024
    print(f"  ✓ Copy {len(chosen)} video ({total_size:.0f} MB)")
    print(f"  ✓ {len(test_cases)} test cases → {OUT_TEST_CASES}")


def main():
    raw_dir = download_msrvtt()
    captions_map, video_files = find_videos_and_captions(raw_dir)
    copy_subset_and_build_test_cases(captions_map, video_files, SUBSET_N_VIDEOS)
    print(f"\n=== Done ===")
    print(f"Videos:     {VIDEO_DIR}")
    print(f"Test cases: {OUT_TEST_CASES}")
    print(f"\nDùng eval:")
    print(f"  py -3.10 scripts/eval_accuracy.py  # đọc test_cases.json mặc định")
    print(f"  (hoặc sửa eval script trỏ về test_cases_msrvtt.json)")


if __name__ == "__main__":
    main()
