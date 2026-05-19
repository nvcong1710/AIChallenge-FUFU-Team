"""Tải data đa dạng hơn cho test visual:

  1) COCO val 2017 subset — 200 ảnh real-world, đa dạng object/scene
  2) Wikimedia expanded — 30+ ảnh VN context (food, scenes, daily life)

Không tải full COCO zip (~1GB) — list image IDs từ annotations rồi download
từng file trực tiếp (~50MB total cho 200 ảnh).
"""

from __future__ import annotations

import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = ROOT / "test-data"
IMG_DIR = TEST_DATA / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "BetterDayTest/0.4 (research)"}
COCO_ANNOTATIONS_URL = "https://huggingface.co/datasets/HuggingFaceM4/COCO/resolve/main/data/captions_val2017.json"
COCO_IMG_BASE = "http://images.cocodataset.org/val2017/"


def http_download(url: str, dest: Path, timeout: int = 60) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            with open(dest, "wb") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
        return dest.stat().st_size > 1024
    except Exception:
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


# ----- COCO subset -----

def download_coco_subset(n: int = 200) -> int:
    """COCO image IDs là số 12-digit zero-padded. val2017 có ~5000 ảnh.
    Lấy danh sách ID khả dụng từ filename pattern phổ biến + brute force sample.
    """
    print(f"\n[1] COCO val 2017 subset — {n} ảnh")
    # COCO val 2017 image IDs đã biết (sample, không cần tải annotations)
    # Format: 000000<6-digit>.jpg
    # IDs hợp lệ rải rác trong [1, 581929]. Sample random rồi try download.
    random.seed(42)
    attempts = 0
    success = 0
    target = n
    tries = 0
    max_tries = n * 5
    seen = set()

    # COCO val 2017 IDs (lấy mẫu rải đều phổ biến)
    known_ranges = [
        (139, 1000), (1000, 5000), (5000, 50000),
        (50000, 200000), (200000, 500000),
    ]

    while success < target and tries < max_tries:
        tries += 1
        lo, hi = random.choice(known_ranges)
        coco_id = random.randint(lo, hi)
        if coco_id in seen:
            continue
        seen.add(coco_id)
        fname = f"{coco_id:012d}.jpg"
        url = COCO_IMG_BASE + fname
        dest = IMG_DIR / f"coco_{fname}"
        if http_download(url, dest, timeout=30):
            success += 1
            if success % 20 == 0:
                size_mb = sum(p.stat().st_size for p in IMG_DIR.glob("coco_*.jpg")) / 1024 / 1024
                print(f"    [{success}/{target}] saved, ~{size_mb:.0f} MB so far")
        else:
            attempts += 1
        time.sleep(0.1)

    total = sum(p.stat().st_size for p in IMG_DIR.glob("coco_*.jpg")) / 1024 / 1024
    print(f"  ✓ {success} COCO images, {total:.0f} MB ({tries} tries, {attempts} fails)")
    return success


# ----- Wikimedia expanded VN queries -----

QUERIES_EXTENDED = [
    "Vietnamese food market street",
    "Vietnam ao dai traditional",
    "Vietnam Halong Bay junk boat",
    "Hue royal palace gate",
    "Mekong delta river",
    "Vietnam rice noodles bowl",
    "Sapa terraced fields",
    "Hoi An lanterns night",
    "Saigon Notre Dame cathedral",
    "Vietnam motorcycle taxi xeom",
    "Hanoi train street",
    "Vietnam street food stall",
    "Vietnam temple incense",
    "Phu Quoc beach island",
    "Vietnam fishing boat",
    "Cu Chi tunnels",
    "Vietnam war memorial",
    "Vietnam jungle forest",
    "Vietnam karaoke",
    "Vietnam coffee shop sidewalk",
]


def wm_search(query: str, limit: int = 3):
    url = (
        "https://commons.wikimedia.org/w/api.php"
        f"?action=query&format=json&list=search"
        f"&srsearch=filetype:bitmap+{urllib.parse.quote(query)}"
        f"&srnamespace=6&srlimit={limit}"
    )
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
        return [it["title"] for it in data.get("query", {}).get("search", [])]
    except Exception:
        return []


def wm_url(title: str):
    url = (
        "https://commons.wikimedia.org/w/api.php"
        f"?action=query&format=json&titles={urllib.parse.quote(title)}"
        "&prop=imageinfo&iiprop=url"
    )
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
        for page in data.get("query", {}).get("pages", {}).values():
            ii = page.get("imageinfo", [])
            if ii:
                return ii[0].get("url")
    except Exception:
        pass
    return None


def safe_name(title: str) -> str:
    fname = title.replace("File:", "").replace(" ", "_")
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in fname)[:80]
    return f"wm2_{safe}"


def download_wikimedia_extended(per_query: int = 2, max_total: int = 30) -> int:
    print(f"\n[2] Wikimedia API search mở rộng VN — target {max_total} ảnh")
    n = 0
    for q in QUERIES_EXTENDED:
        if n >= max_total:
            break
        titles = wm_search(q, limit=per_query + 1)
        time.sleep(2.0)
        added = 0
        for t in titles:
            if added >= per_query or n >= max_total:
                break
            if any(t.lower().endswith(e) for e in (".svg", ".pdf", ".tif", ".tiff", ".gif")):
                continue
            time.sleep(2.0)
            url = wm_url(t)
            if not url:
                continue
            fname = safe_name(t)
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            dest = IMG_DIR / fname
            if dest.exists() and dest.stat().st_size > 0:
                continue
            time.sleep(2.0)
            if http_download(url, dest, timeout=60):
                size_mb = dest.stat().st_size / 1024 / 1024
                print(f"    ✓ {dest.name[:60]} ({size_mb:.1f}MB)")
                n += 1
                added += 1
    return n


def main():
    n_coco = download_coco_subset(n=200)
    n_wm = download_wikimedia_extended(per_query=2, max_total=30)

    print("\n=== Summary ===")
    print(f"  COCO subset:        {n_coco} ảnh")
    print(f"  Wikimedia extended: {n_wm} ảnh")
    total = sum(f.stat().st_size for f in TEST_DATA.rglob("*") if f.is_file())
    n_files = sum(1 for f in TEST_DATA.rglob("*") if f.is_file())
    print(f"\nTotal test-data:    {n_files} files, {total/1024/1024:.0f} MB")


if __name__ == "__main__":
    main()
