"""Tải MSR-VTT từ Kaggle + sinh ground-truth test cases.

Cần 2 datasets:
  - vishnutheepb/msrvtt          (videos, ~7GB full)
  - vishnutheepb/msrvttdatainfo  (captions JSON, ~10MB)

Kaggle API chỉ tải full zip — không partial. Sau khi tải xong sẽ extract
SUBSET_N_VIDEOS rồi xoá zip để tiết kiệm disk.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = ROOT / "test-data"
VIDEO_DIR = TEST_DATA / "videos" / "msrvtt"
OUT_TEST_CASES = ROOT / "scripts" / "test_cases_msrvtt.json"

DATASET_VIDEOS = "vishnutheepb/msrvtt"
DATASET_CAPTIONS = "vishnutheepb/msrvttdatainfo"
SUBSET_N_VIDEOS = 200          # ~50-100MB extracted
QUERIES_PER_VIDEO = 3


def kaggle_available() -> bool:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        return True
    except Exception:
        return False


def download_dataset(ref: str, dest_dir: Path) -> Path:
    """Tải Kaggle dataset, KHÔNG auto-unzip để control extraction."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ {ref} → {dest_dir} (có thể mất vài phút)")
    api.dataset_download_files(ref, path=str(dest_dir), unzip=False, quiet=False)
    zips = list(dest_dir.glob("*.zip"))
    if not zips:
        raise RuntimeError(f"No zip after download of {ref}")
    return zips[0]


def extract_captions(zip_path: Path, dest_dir: Path) -> dict:
    """Extract caption JSON + parse → {video_id: [captions]}."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        json_names = [n for n in names if n.lower().endswith(".json")]
        if not json_names:
            raise RuntimeError("No JSON in captions zip")
        print(f"  extract {len(json_names)} JSON files")
        zf.extractall(dest_dir)

    captions_map: dict[str, list[str]] = {}
    for jf in dest_dir.rglob("*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and "sentences" in data:
            for s in data["sentences"]:
                vid = s.get("video_id")
                cap = (s.get("caption") or "").strip()
                if vid and cap:
                    captions_map.setdefault(vid, []).append(cap)
            print(f"    ✓ {jf.name}: {len(data['sentences'])} captions")
            break
    return captions_map


def extract_video_subset(zip_path: Path, captions_map: dict, n: int) -> int:
    """Extract chỉ N videos có sẵn captions từ zip MSR-VTT."""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        # list members có video_id trong captions
        all_videos = [m for m in zf.namelist() if m.endswith(".mp4")]
        # extract video_id từ path: TrainValVideo/videoN.mp4 → videoN
        video_to_member: dict[str, str] = {}
        for m in all_videos:
            stem = Path(m).stem  # videoN
            if stem in captions_map:
                video_to_member[stem] = m

        if not video_to_member:
            raise RuntimeError("No videos match captions. Check naming.")
        print(f"  total: {len(all_videos)} videos, {len(video_to_member)} matched với captions")

        random.seed(42)
        chosen_ids = random.sample(sorted(video_to_member), min(n, len(video_to_member)))
        n_extracted = 0
        for vid_id in chosen_ids:
            member = video_to_member[vid_id]
            dst = VIDEO_DIR / f"msrvtt_{vid_id}.mp4"
            if dst.exists() and dst.stat().st_size > 0:
                n_extracted += 1
                continue
            with zf.open(member) as src, open(dst, "wb") as f:
                shutil.copyfileobj(src, f)
            n_extracted += 1
            if n_extracted % 25 == 0:
                size = sum(p.stat().st_size for p in VIDEO_DIR.glob("*.mp4")) / 1024 / 1024
                print(f"    [{n_extracted}/{n}] extracted, ~{size:.0f}MB")
        return n_extracted


def build_test_cases(captions_map: dict) -> int:
    test_cases = []
    for vf in sorted(VIDEO_DIR.glob("msrvtt_*.mp4")):
        vid_id = vf.stem.replace("msrvtt_", "")
        caps = captions_map.get(vid_id, [])
        if not caps:
            continue
        random.seed(hash(vid_id) % 10000)
        random.shuffle(caps)
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
    return len(test_cases)


def main():
    if not kaggle_available():
        print("⚠ pip install kaggle + setup ~/.kaggle/kaggle.json trước.")
        sys.exit(1)

    raw_root = TEST_DATA / "_msrvtt_raw"

    print(f"\n[1/4] Download captions: {DATASET_CAPTIONS}")
    cap_zip = download_dataset(DATASET_CAPTIONS, raw_root / "captions")
    captions_map = extract_captions(cap_zip, raw_root / "captions_extracted")
    print(f"  → {len(captions_map)} videos có captions")

    print(f"\n[2/4] Download videos: {DATASET_VIDEOS}")
    print(f"  (Full ~7GB. Sẽ extract chỉ {SUBSET_N_VIDEOS} videos rồi xoá zip)")
    vid_zip = download_dataset(DATASET_VIDEOS, raw_root / "videos")

    print(f"\n[3/4] Extract subset {SUBSET_N_VIDEOS} videos")
    n_extracted = extract_video_subset(vid_zip, captions_map, SUBSET_N_VIDEOS)

    print(f"\n[4/4] Build test cases")
    n_cases = build_test_cases(captions_map)

    # Cleanup zip để giải phóng disk
    print(f"\n[cleanup] xoá zip files để tiết kiệm disk")
    for z in raw_root.rglob("*.zip"):
        try:
            z.unlink()
        except Exception:
            pass

    total_mb = sum(p.stat().st_size for p in VIDEO_DIR.glob("*.mp4")) / 1024 / 1024
    print(f"\n=== Done ===")
    print(f"  Videos extracted: {n_extracted} → {VIDEO_DIR} ({total_mb:.0f}MB)")
    print(f"  Test cases:       {n_cases} → {OUT_TEST_CASES}")
    print(f"\nEval:")
    print(f"  py -3.10 scripts/eval_accuracy.py --cases scripts/test_cases_msrvtt.json")
    print(f"  py -3.10 scripts/eval_html_report.py --cases scripts/test_cases_msrvtt.json")


if __name__ == "__main__":
    main()
