"""Tải / sinh nhiều data hơn cho test:

  1) Wikimedia API search → lấy URL ảnh THẬT (không guess hardcoded URL)
  2) Sinh ~20 mẫu VN gTTS với nội dung đa dạng
  3) Thử download Blender open movies (CC-BY, multi-scene, multi-character)
  4) Sinh nhiều ảnh có chữ VN với layout khác nhau

Total target: < 1GB.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = ROOT / "test-data"
IMG_DIR = TEST_DATA / "images"
VID_DIR = TEST_DATA / "videos"
AUD_DIR = TEST_DATA / "audio" / "vn_speech"
for d in (IMG_DIR, VID_DIR, AUD_DIR):
    d.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "BetterDayTest/0.2 (research)"}


# ----- Wikimedia API helpers -----

def wm_search_files(query: str, limit: int = 6) -> list[str]:
    """Search Wikimedia Commons cho file ảnh; trả về list 'File:...' titles."""
    url = (
        "https://commons.wikimedia.org/w/api.php"
        "?action=query&format=json&list=search"
        f"&srsearch=filetype:bitmap+{urllib.parse.quote(query)}"
        f"&srnamespace=6&srlimit={limit}"
    )
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
        return [item["title"] for item in data.get("query", {}).get("search", [])]
    except Exception as e:
        print(f"  search '{query}' fail: {e}")
        return []


def wm_get_file_url(title: str) -> str | None:
    """Lookup direct file URL cho 1 'File:...' title."""
    url = (
        "https://commons.wikimedia.org/w/api.php"
        "?action=query&format=json"
        f"&titles={urllib.parse.quote(title)}"
        "&prop=imageinfo&iiprop=url|size|mime"
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
        return None
    return None


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


# ----- 1. Wikimedia search ảnh -----

QUERIES = [
    "Hanoi street",
    "Ho Chi Minh City skyline",
    "Vietnam pho noodle",
    "Vietnam traffic motorbike",
    "Vietnam market",
    "Vietnam landscape rice field",
    "Vietnam temple pagoda",
    "Saigon old quarter",
]


def download_wikimedia_images(per_query: int = 2, max_total: int = 15) -> int:
    print("\n[1/4] Wikimedia API search ảnh VN")
    downloaded = 0
    for q in QUERIES:
        if downloaded >= max_total:
            break
        titles = wm_search_files(q, limit=per_query + 2)
        print(f"  search '{q}': {len(titles)} match")
        time.sleep(0.5)  # né rate limit API
        n_this_query = 0
        for title in titles:
            if n_this_query >= per_query or downloaded >= max_total:
                break
            # Skip SVG/PDF
            if any(title.lower().endswith(ext) for ext in (".svg", ".pdf", ".tif", ".tiff")):
                continue
            time.sleep(0.5)
            url = wm_get_file_url(title)
            if not url:
                continue
            # Tên file gọn
            fname = title.replace("File:", "").replace(" ", "_")
            # Bỏ ký tự không an toàn cho filesystem
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in fname)[:80]
            if not safe.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            dest = IMG_DIR / f"wm_{safe}"
            time.sleep(0.5)
            if http_download(url, dest):
                size = dest.stat().st_size // 1024
                print(f"    ✓ {dest.name} ({size} KB)")
                downloaded += 1
                n_this_query += 1
            else:
                print(f"    ✗ {fname[:50]}")
    return downloaded


# ----- 2. gTTS bulk VN samples -----

VN_SCRIPTS_EXTRA = [
    ("tin_chinh_tri.mp3",
     "Quốc hội thông qua nghị quyết về phát triển kinh tế xã hội. "
     "Thủ tướng Phạm Minh Chính chỉ đạo các bộ ngành tập trung tháo gỡ khó khăn cho doanh nghiệp. "
     "Bí thư Thành ủy Hà Nội yêu cầu tăng cường công tác phòng chống tham nhũng."),

    ("tin_y_te.mp3",
     "Bệnh viện Bạch Mai triển khai phương pháp điều trị mới cho bệnh nhân ung thư. "
     "Bộ Y tế khuyến cáo người dân tiêm vắc xin phòng cúm mùa. "
     "Số ca mắc sốt xuất huyết tại Hà Nội tăng cao trong tuần qua."),

    ("tin_giao_duc.mp3",
     "Bộ Giáo dục và Đào tạo công bố quy chế thi tốt nghiệp trung học phổ thông năm nay. "
     "Đại học Quốc gia Hà Nội mở thêm ngành học mới về trí tuệ nhân tạo. "
     "Học sinh tiểu học bắt đầu năm học mới với chương trình thay đổi."),

    ("tin_cong_nghe.mp3",
     "Viettel ra mắt mạng năm G phủ sóng toàn quốc. "
     "Tập đoàn FPT đầu tư vào công nghệ trí tuệ nhân tạo. "
     "Vinfast giới thiệu mẫu xe điện mới tại triển lãm quốc tế Detroit."),

    ("tin_van_hoa.mp3",
     "Liên hoan phim quốc tế Hà Nội khai mạc với hơn năm mươi phim tham gia. "
     "Ca sĩ Mỹ Linh tổ chức đêm nhạc kỷ niệm ba mươi năm sự nghiệp. "
     "Triển lãm tranh đông hồ thu hút đông đảo khách tham quan tại Hà Nội."),

    ("tin_du_lich.mp3",
     "Vịnh Hạ Long đón hơn một triệu khách quốc tế trong sáu tháng đầu năm. "
     "Phố cổ Hội An tổ chức đêm hội đèn lồng vào rằm tháng tám. "
     "Sa Pa khôi phục hoạt động du lịch sau mưa bão, đường lên đỉnh Fansipan đã thông."),

    ("tin_an_ninh.mp3",
     "Công an Thành phố Hồ Chí Minh triệt phá đường dây ma túy lớn. "
     "Cảnh sát giao thông xử lý nghiêm vi phạm nồng độ cồn dịp lễ. "
     "Lực lượng phòng cháy chữa cháy diễn tập tại quận Hoàn Kiếm."),

    ("hoi_thoai_taxi.mp3",
     "Anh ơi cho tôi đến phố Bát Đàn nhé. "
     "Anh có thể đi nhanh hơn được không, tôi muộn cuộc họp rồi. "
     "Bao nhiêu tiền vậy anh? Tổng cộng tám mươi ngàn đồng."),

    ("hoi_thoai_chua.mp3",
     "Mình đi chùa Bà Đen ở Tây Ninh đi. "
     "Có cáp treo lên đỉnh núi không? Có chứ, vé cáp treo hai trăm năm mươi nghìn một người. "
     "Chùa ở đỉnh núi cao một nghìn mét."),

    ("hoi_thoai_hoc_lap_trinh.mp3",
     "Em đang học lập trình Python. "
     "Anh có thể chia sẻ kinh nghiệm học máy học không? "
     "Em muốn làm việc trong lĩnh vực trí tuệ nhân tạo. "
     "Anh khuyên em nên đọc nhiều paper trên arXiv."),

    ("ke_chuyen_lich_su.mp3",
     "Hai Bà Trưng khởi nghĩa chống quân Hán năm bốn mươi sau công nguyên. "
     "Ngô Quyền đánh tan quân Nam Hán trên sông Bạch Đằng năm chín trăm ba mươi tám. "
     "Lý Thái Tổ dời đô về Thăng Long năm một nghìn không trăm mười."),

    ("ke_chuyen_khoa_hoc.mp3",
     "Mặt trời cách trái đất một trăm năm mươi triệu ki lô mét. "
     "Tốc độ ánh sáng trong chân không là ba trăm nghìn ki lô mét trên giây. "
     "Nước sôi ở một trăm độ C dưới áp suất khí quyển bình thường."),

    ("tin_thoi_tiet.mp3",
     "Dự báo thời tiết các tỉnh phía Bắc. "
     "Hà Nội mưa rào và dông, nhiệt độ từ hai mươi tới hai mươi tám độ. "
     "Đà Nẵng nắng nóng, có nơi trên ba mươi lăm độ. "
     "Vùng biển từ Quảng Ngãi tới Bình Thuận sóng cao hai đến ba mét."),

    ("tin_giai_tri.mp3",
     "Sơn Tùng MTP chuẩn bị ra album mới sau hai năm im hơi lặng tiếng. "
     "Diễn viên Trấn Thành dẫn dắt game show mới vào giờ vàng cuối tuần. "
     "Cuộc thi Hoa hậu Việt Nam vòng chung kết diễn ra tối nay tại Đà Lạt."),

    ("tin_nong_nghiep.mp3",
     "Nông dân miền Tây thu hoạch lúa hè thu, năng suất bình quân sáu tấn một héc ta. "
     "Giá cà phê Tây Nguyên tăng cao kỷ lục, đạt sáu mươi lăm nghìn đồng một ki lô. "
     "Bộ Nông nghiệp cảnh báo sâu bệnh hại lúa tại đồng bằng sông Hồng."),
]


def gen_gtts_bulk() -> int:
    print("\n[2/4] gTTS — sinh thêm mẫu VN đa dạng")
    try:
        from gtts import gTTS
    except ImportError:
        print("  ✗ gtts chưa cài. pip install gtts")
        return 0
    n_ok = 0
    for fname, text in VN_SCRIPTS_EXTRA:
        out = AUD_DIR / fname
        if out.exists() and out.stat().st_size > 0:
            n_ok += 1
            continue
        try:
            print(f"  ⚙ {fname}", end=" ... ", flush=True)
            gTTS(text=text, lang="vi", slow=False).save(str(out))
            print(f"{out.stat().st_size // 1024} KB")
            n_ok += 1
            time.sleep(0.2)  # né rate limit
        except Exception as e:
            print(f"FAIL: {e}")
    return n_ok


# ----- 3. Blender / Wikimedia video attempts -----

VIDEO_CANDIDATES = [
    # Blender Foundation CDN — CC-BY animated open movies
    ("sintel_trailer.mp4",
     "https://download.blender.org/durian/trailer/sintel_trailer-720p.mp4"),
    ("tears_of_steel_trailer.mp4",
     "https://download.blender.org/mango/download.blender.org/ToS-4k-1920.mov"),
    # Wikimedia hosts some CC video
    ("caminandes_2.ogv",
     "https://upload.wikimedia.org/wikipedia/commons/5/55/Caminandes_2_-_Gran_Dillama.ogv"),
    ("caminandes_3.webm",
     "https://upload.wikimedia.org/wikipedia/commons/2/27/Caminandes_-_Llamigos.webm"),
    # NASA public domain
    ("nasa_iss_tour.webm",
     "https://upload.wikimedia.org/wikipedia/commons/9/9c/ISS_Walkthrough_HD.ogv"),
]


def download_video_candidates() -> int:
    print("\n[3/4] Tải video Blender / Wikimedia (CC, multi-scene)")
    n_ok = 0
    for name, url in VIDEO_CANDIDATES:
        dest = VID_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  ✓ {name} (đã có)")
            n_ok += 1
            continue
        print(f"  ↓ {name}", end=" ... ", flush=True)
        ok = http_download(url, dest, timeout=180)
        if ok:
            size_mb = dest.stat().st_size / 1024 / 1024
            print(f"{size_mb:.1f} MB")
            n_ok += 1
        else:
            print("FAIL")
        time.sleep(1.0)
    return n_ok


# ----- 4. Thêm synth ảnh VN text -----

VN_SYNTH_IMAGES = [
    ("vn_menu_2.png", [
        ("BÁNH MÌ HÀ NỘI",          70),
        ("Thịt nướng – Pate – Trứng",36),
        ("Giá: 25.000 đ – 35.000 đ",36),
        ("Hotline: 0912 345 678",    32),
    ]),
    ("vn_sign_street.png", [
        ("ĐƯỜNG NGUYỄN TRÃI",       60),
        ("HÀ NỘI · QUẬN THANH XUÂN", 32),
        ("←  CẦU GIẤY",              42),
        ("HÀ ĐÔNG →",                42),
    ]),
    ("vn_news_ticker.png", [
        ("THỜI SỰ 19H",             50),
        ("VTV1 – ĐÀI TRUYỀN HÌNH VIỆT NAM", 28),
        ("Thị trường chứng khoán tăng điểm", 26),
        ("VN-Index: 1245 (+12 điểm)", 26),
    ]),
    ("vn_shop.png", [
        ("CỬA HÀNG ĐIỆN MÁY XANH",  56),
        ("Khuyến mãi cuối tuần",     32),
        ("iPhone 15 Pro Max",        36),
        ("Samsung – Xiaomi – Oppo",  32),
    ]),
]


def gen_synth_vn_images() -> int:
    print("\n[4/4] Sinh ảnh VN có chữ (test OCR đa layout)")
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return 0

    # Tìm font hỗ trợ VN
    font_paths = ["arial.ttf", "C:/Windows/Fonts/arial.ttf",
                  "C:/Windows/Fonts/segoeui.ttf",
                  "/System/Library/Fonts/Helvetica.ttc",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]

    def get_font(size: int):
        for fp in font_paths:
            try:
                return ImageFont.truetype(fp, size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()

    n_ok = 0
    for name, lines in VN_SYNTH_IMAGES:
        out = IMG_DIR / name
        if out.exists() and out.stat().st_size > 0:
            n_ok += 1
            continue
        img = Image.new("RGB", (800, 600), color=(252, 248, 240))
        draw = ImageDraw.Draw(img)
        y = 60
        for text, fsize in lines:
            font = get_font(fsize)
            draw.text((60, y), text, fill=(40, 40, 50), font=font)
            y += fsize + 16
        img.save(out)
        print(f"  ⚙ {name} ({out.stat().st_size // 1024} KB)")
        n_ok += 1
    return n_ok


def main():
    n_img = download_wikimedia_images(per_query=2, max_total=12)
    n_aud = gen_gtts_bulk()
    n_vid = download_video_candidates()
    n_synth_img = gen_synth_vn_images()

    print("\n=== Tổng kết phiên tải mới ===")
    print(f"  Wikimedia images: {n_img}")
    print(f"  gTTS VN audio:    {n_aud}")
    print(f"  Real videos:      {n_vid}")
    print(f"  Synth VN images:  {n_synth_img}")

    # Stats tổng
    total_bytes = sum(f.stat().st_size for f in TEST_DATA.rglob("*") if f.is_file())
    n_files = sum(1 for f in TEST_DATA.rglob("*") if f.is_file())
    print(f"\nTotal test-data: {n_files} files, {total_bytes / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
