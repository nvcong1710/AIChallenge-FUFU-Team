"""Download / synthesize multimedia test data — cover 3 modality × nhiều extension.

Cấu trúc output:
    test-data/
    ├── images/  jpg, png (5 real Wikimedia + 1 synth VI text)
    ├── videos/  mp4 (synth từ images)
    └── audio/   mp3 (extract), ogg (download), wav (synth sine)
"""

from __future__ import annotations

import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = ROOT / "test-data"

# Wikimedia Commons direct URLs — stable, public domain hoặc CC-BY-SA
IMAGES = [
    # URL gốc Wikimedia (không qua /thumb/ — tránh HTTP 400 size whitelist).
    # Có alternative URL nếu primary fail.
    ("chess_set.jpg",  ["https://upload.wikimedia.org/wikipedia/commons/6/6f/ChessSet.jpg"]),
    ("cat.jpg",        ["https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"]),
    ("pho_bo.jpg",     [
        "https://upload.wikimedia.org/wikipedia/commons/9/9e/Pho-Beef-Noodles-2008.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/2/2a/Pho-Beef-Noodles-2008.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/5/55/PHO_TAI.jpg",
    ]),
    ("sunset.jpg",     [
        "https://upload.wikimedia.org/wikipedia/commons/1/13/Sunset_2007-1.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/8/82/Sunset_at_Mt._Pinatubo_Crater_Lake.jpg",
    ]),
    ("traffic.jpg",    [
        "https://upload.wikimedia.org/wikipedia/commons/c/ce/Traffic_jam_-_Strait_Crossing_Bridge.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/4/4b/Traffic_jam_Sao_Paulo_09_2006_30.JPG",
    ]),
    ("dog.jpg",        ["https://upload.wikimedia.org/wikipedia/commons/8/87/Lieserheide_dog.jpg",
                        "https://upload.wikimedia.org/wikipedia/commons/c/c9/Dog_-_നായ.JPG"]),
    ("flowers.jpg",    ["https://upload.wikimedia.org/wikipedia/commons/4/41/Sunflower_from_Silesia2.jpg",
                        "https://upload.wikimedia.org/wikipedia/commons/9/90/Sunflower_sky_backdrop.jpg"]),
]

# Real video — Big Buck Bunny (CC-BY 3.0 từ Blender Foundation, có shot cuts thật)
REAL_VIDEOS = [
    ("big_buck_bunny_480p.mp4",
     "https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_480p_h264.mov"),
]

HEADERS = {"User-Agent": "BetterDayTest/0.1 (test data downloader)"}


def http_download(url: str, dest: Path, timeout: int = 60) -> bool:
    """Download nội dung từ URL về dest. Trả True/False."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ✓ {dest.name} (đã có)")
        return True
    print(f"  ↓ {dest.name}", end=" ... ", flush=True)
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            # Stream to file để không hold RAM cho file lớn
            with open(dest, "wb") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
        size = dest.stat().st_size
        unit = "MB" if size >= 1 << 20 else "KB"
        val = size / (1 << 20) if unit == "MB" else size / 1024
        print(f"{val:.1f} {unit}")
        return True
    except Exception as e:
        if dest.exists():
            dest.unlink()
        print(f"FAIL: {e}")
        return False


def try_download(urls: list[str], dest: Path, sleep_between: float = 1.0) -> bool:
    """Thử nhiều URL theo thứ tự, sleep giữa các request để né rate limit."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ✓ {dest.name} (đã có)")
        return True
    for i, url in enumerate(urls):
        if i > 0:
            time.sleep(sleep_between)
        if http_download(url, dest):
            return True
    return False


def make_synth_video(image: Path, out_path: Path, duration: int = 10) -> bool:
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  ✓ {out_path.name} (đã có)")
        return True
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-i", str(image),
        "-t", str(duration), "-r", "10",
        "-vf", "scale=640:-2,format=yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
        str(out_path),
    ]
    print(f"  ⚙ {out_path.name}", end=" ... ", flush=True)
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"{out_path.stat().st_size // 1024} KB")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAIL: {e.stderr.decode()[:100]}")
        return False


def make_vietnamese_text_image(out_path: Path) -> bool:
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  ✓ {out_path.name} (đã có)")
        return True
    print(f"  ⚙ {out_path.name}", end=" ... ", flush=True)
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGB", (800, 600), color=(255, 250, 235))
        draw = ImageDraw.Draw(img)

        # Tìm font có hỗ trợ Vietnamese diacritics
        font_lg = None
        font_md = None
        for font_path in ("arial.ttf", "C:/Windows/Fonts/arial.ttf",
                          "C:/Windows/Fonts/segoeui.ttf",
                          "/System/Library/Fonts/Helvetica.ttc",
                          "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
            try:
                font_lg = ImageFont.truetype(font_path, 70)
                font_md = ImageFont.truetype(font_path, 32)
                break
            except (OSError, IOError):
                continue
        if font_lg is None:
            font_lg = font_md = ImageFont.load_default()

        draw.text((60, 120), "PHỞ BÒ HÀ NỘI", fill=(40, 40, 40), font=font_lg)
        draw.text((60, 260), "Đặc sản truyền thống Việt Nam", fill=(80, 80, 80), font=font_md)
        draw.text((60, 340), "Bún chả · Bánh mì · Cà phê sữa đá", fill=(60, 90, 60), font=font_md)
        draw.text((60, 440), "Giờ mở cửa: 06:00 - 22:00", fill=(120, 60, 40), font=font_md)
        draw.text((60, 500), "ĐC: 49 Bát Đàn, Hoàn Kiếm", fill=(120, 60, 40), font=font_md)
        img.save(out_path)
        print(f"{out_path.stat().st_size // 1024} KB")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False


def extract_audio_from_video(video: Path, out_path: Path) -> bool:
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  ✓ {out_path.name} (đã có)")
        return True
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video),
        "-vn", "-acodec", "libmp3lame", "-q:a", "5",
        str(out_path),
    ]
    print(f"  ⚙ {out_path.name} từ {video.name}", end=" ... ", flush=True)
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"{out_path.stat().st_size // 1024} KB")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAIL: {e.stderr.decode()[:100]}")
        return False


def make_synth_sine(out_path: Path, duration: int = 5, freq: int = 440) -> bool:
    """Sinh tiếng sine để test no-speech path (ASR sẽ trả empty → fallback sliding)."""
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  ✓ {out_path.name} (đã có)")
        return True
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={duration}",
        "-ac", "1", "-ar", "16000",
        str(out_path),
    ]
    print(f"  ⚙ {out_path.name}", end=" ... ", flush=True)
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"{out_path.stat().st_size // 1024} KB")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAIL: {e.stderr.decode()[:100]}")
        return False


def main():
    img_dir = TEST_DATA / "images"
    vid_dir = TEST_DATA / "videos"
    aud_dir = TEST_DATA / "audio"
    for d in (img_dir, vid_dir, aud_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("\n[1/6] Tải ảnh từ Wikimedia Commons (sleep 1s giữa requests)")
    img_ok = []
    for i, (name, urls) in enumerate(IMAGES):
        if i > 0:
            time.sleep(1.0)
        if try_download(urls, img_dir / name):
            img_ok.append(img_dir / name)

    print("\n[2/6] Sinh ảnh có chữ tiếng Việt (test OCR)")
    vi_text = img_dir / "vietnamese_menu.png"
    if make_vietnamese_text_image(vi_text):
        img_ok.append(vi_text)

    print("\n[3/6] Tải Big Buck Bunny (CC-BY, ~50MB — video thực có cuts)")
    for name, url in REAL_VIDEOS:
        http_download(url, vid_dir / name, timeout=300)

    print("\n[4/6] Sinh video từ ảnh (image loop → mp4)")
    for img in img_ok[:5]:
        make_synth_video(img, vid_dir / f"{img.stem}.mp4", duration=10)

    print("\n[5/6] Tách audio từ video chess + Big Buck Bunny")
    existing = sorted(TEST_DATA.glob("*.mp4"))
    if existing:
        extract_audio_from_video(existing[0], aud_dir / "chess_extracted.mp3")
    bbb = vid_dir / "big_buck_bunny_480p.mp4"
    if bbb.exists():
        extract_audio_from_video(bbb, aud_dir / "bbb_audio.mp3")

    print("\n[6/6] Sinh audio test")
    make_synth_sine(aud_dir / "tone_440hz.wav", duration=5, freq=440)

    # Summary
    print("\n=== Summary ===")
    for d, label in [(img_dir, "images"), (vid_dir, "videos"), (aud_dir, "audio")]:
        files = sorted(f for f in d.glob("*") if f.is_file())
        total = sum(f.stat().st_size for f in files)
        ext_count: dict[str, int] = {}
        for f in files:
            ext_count[f.suffix.lower()] = ext_count.get(f.suffix.lower(), 0) + 1
        ext_str = ", ".join(f"{ext}×{n}" for ext, n in sorted(ext_count.items()))
        print(f"  {label:<8} {len(files):>2} files  {total/1024:>6.0f} KB  ({ext_str})")
    # Root chess video
    root_files = [f for f in TEST_DATA.glob("*.mp4") if f.is_file()]
    if root_files:
        for f in root_files:
            print(f"  (root)   1 file   {f.stat().st_size/1024:>6.0f} KB  ({f.name})")


if __name__ == "__main__":
    main()
