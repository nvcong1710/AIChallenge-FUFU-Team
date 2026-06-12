"""Tải và sinh dữ liệu đa phương tiện mẫu (Video, Audio, Image) cho việc test.
Gộp chung logic từ các file download cũ để giữ codebase gọn gàng.
"""

import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = ROOT / "test-data"
IMG_DIR = TEST_DATA / "images"
VID_DIR = TEST_DATA / "videos"
AUD_DIR = TEST_DATA / "audio"
for d in (IMG_DIR, VID_DIR, AUD_DIR):
    d.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "FUFUTest/1.0"}

VIDEOS = [
    ("big_buck_bunny_480p.mp4", "https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_480p_h264.mov"),
    ("tears_of_steel.mp4", "https://download.blender.org/demo/movies/ToS/tos.mp4")
]

VN_SCRIPTS = [
    ("tin_kinh_te.mp3", "Thị trường chứng khoán Việt Nam hôm nay tăng ba phần trăm. Giá vàng tiếp tục lập đỉnh mới."),
    ("tin_giao_thong.mp3", "Đường Nguyễn Trãi tắc nghẽn nghiêm trọng giờ cao điểm."),
    ("tin_du_lich.mp3", "Vịnh Hạ Long đón hơn một triệu khách quốc tế trong sáu tháng đầu năm."),
]

def http_download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ✓ {dest.name} (đã có)")
        return True
    print(f"  ↓ {dest.name}", end=" ... ", flush=True)
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            with open(dest, "wb") as f:
                while chunk := r.read(1 << 16):
                    f.write(chunk)
        print(f"{dest.stat().st_size / (1<<20):.1f} MB")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        if dest.exists(): dest.unlink()
        return False

def gen_gtts_audio():
    print("\n[1/3] Sinh Audio tiếng Việt (gTTS)")
    try:
        from gtts import gTTS
    except ImportError:
        print("  ✗ Thư viện gtts chưa cài. (pip install gtts)")
        return
    for fname, text in VN_SCRIPTS:
        out = AUD_DIR / fname
        if out.exists():
            print(f"  ✓ {fname}")
            continue
        print(f"  ⚙ {fname} ...", end=" ")
        try:
            gTTS(text=text, lang="vi").save(str(out))
            print("OK")
            time.sleep(0.5)
        except Exception as e:
            print(f"FAIL: {e}")

def download_videos():
    print("\n[2/3] Tải Video mẫu")
    for name, url in VIDEOS:
        http_download(url, VID_DIR / name)

def gen_synth_images():
    print("\n[3/3] Sinh ảnh có chữ (OCR Test)")
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return
    img_path = IMG_DIR / "vn_menu.png"
    if img_path.exists():
        print(f"  ✓ {img_path.name}")
        return
    
    img = Image.new("RGB", (800, 600), color=(255, 250, 235))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 60)
    except:
        font = ImageFont.load_default()
        
    draw.text((60, 100), "PHỞ BÒ HÀ NỘI", fill=(40, 40, 40), font=font)
    draw.text((60, 200), "Đặc sản truyền thống Việt Nam", fill=(80, 80, 80), font=font)
    img.save(img_path)
    print(f"  ⚙ {img_path.name} OK")

def main():
    gen_gtts_audio()
    download_videos()
    gen_synth_images()
    
    total_bytes = sum(f.stat().st_size for f in TEST_DATA.rglob("*") if f.is_file())
    n_files = sum(1 for f in TEST_DATA.rglob("*") if f.is_file())
    print(f"\n=== Tổng dữ liệu: {n_files} files, {total_bytes / (1<<20):.1f} MB ===")

if __name__ == "__main__":
    main()