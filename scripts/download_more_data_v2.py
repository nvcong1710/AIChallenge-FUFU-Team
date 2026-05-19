"""Phiên 2: tải thêm sau khi rate limit Wikimedia clear + thử nguồn khác.

  - Wikimedia API search với delay dài hơn
  - Archive.org direct URLs (stable CDN, CC content)
  - Blender open movies (retry với URL khác)
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

UA = {"User-Agent": "BetterDayTest/0.3 (research)"}


def http_download(url: str, dest: Path, timeout: int = 180) -> bool:
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
    except Exception as e:
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


# ----- Wikimedia retry với delay 2s -----

QUERIES_RETRY = [
    "Vietnam traffic motorbike",
    "Vietnam market food",
    "Vietnam rice paddy",
    "Vietnam temple pagoda",
    "Saigon old quarter",
    "Vietnam dragon dance",
    "Halong Bay scenery",
    "Vietnamese coffee shop",
    "Hue imperial city",
    "Da Nang beach",
]


def wm_search(query: str, limit: int = 4):
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
    except Exception as e:
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
    return f"wm_{safe}"


def fetch_wikimedia_more(per_query: int = 2, max_total: int = 15) -> int:
    print("\n[1] Wikimedia retry (delay 2s/request)")
    n = 0
    for q in QUERIES_RETRY:
        if n >= max_total:
            break
        titles = wm_search(q, limit=per_query + 2)
        print(f"  '{q}': {len(titles)} match")
        time.sleep(2.0)
        added_this = 0
        for t in titles:
            if added_this >= per_query or n >= max_total:
                break
            if any(t.lower().endswith(ext) for ext in (".svg", ".pdf", ".tif", ".tiff", ".gif")):
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
                print(f"    ✓ {dest.name} ({size_mb:.1f} MB)")
                n += 1
                added_this += 1
    return n


# ----- Archive.org + Blender stable URLs -----

VIDEO_CANDIDATES_V2 = [
    # Blender CDN
    ("tears_of_steel_short.mp4",
     "https://download.blender.org/demo/movies/ToS/tos.mp4"),
    ("sintel_demo.mp4",
     "https://download.blender.org/demo/movies/Sintel.2010.720p.mkv"),
    # archive.org Blender mirror
    ("caminandes_1.mp4",
     "https://archive.org/download/Caminandes-Llamasinhats/Caminandes_Llamasinhats.mp4"),
    ("caminandes_llamigos.mp4",
     "https://archive.org/download/Caminandes_Llamigos/Caminandes_Llamigos_1080p.mp4"),
    # NASA short videos (public domain)
    ("nasa_earth.mp4",
     "https://archive.org/download/NASA-A-Roll-2014-12-09/A%20Roll%20-%20The%20Year%202014%20%28NASA%29.mp4"),
]


def fetch_videos_v2() -> int:
    print("\n[2] Video retry — Blender + archive.org")
    n = 0
    for name, url in VIDEO_CANDIDATES_V2:
        dest = VID_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  ✓ {name} (đã có, {dest.stat().st_size/1024/1024:.1f} MB)")
            n += 1
            continue
        print(f"  ↓ {name}", end=" ... ", flush=True)
        ok = http_download(url, dest, timeout=300)
        if ok:
            print(f"{dest.stat().st_size/1024/1024:.1f} MB")
            n += 1
        else:
            print("FAIL")
        time.sleep(1.0)
    return n


# ----- More VN gTTS -----

VN_SCRIPTS_V2 = [
    ("hoi_thoai_buu_dien.mp3",
     "Chào chị, em muốn gửi bưu phẩm đi Thành phố Hồ Chí Minh. "
     "Cân lên hai ki lô gram, phí ba mươi năm nghìn đồng. "
     "Khi nào hàng tới nơi vậy chị? Hai ngày tới sẽ tới."),
    ("hoi_thoai_kham_benh.mp3",
     "Em bị đau đầu mấy hôm nay, kèm theo sốt nhẹ. "
     "Anh đo huyết áp giúp em. Một trăm hai mươi trên tám mươi, bình thường. "
     "Tôi kê đơn paracetamol uống ba lần một ngày sau bữa ăn."),
    ("hoi_thoai_nha_hang.mp3",
     "Cho tôi đặt bàn cho bốn người tối nay lúc bảy giờ. "
     "Nhà hàng có món hải sản tươi không? "
     "Có ạ, cua ghẹ tôm cá tươi sống đầy đủ. "
     "Cho tôi xem thực đơn nhé."),
    ("hoi_thoai_san_bay.mp3",
     "Chuyến bay Vietnam Airlines VN một ba ba bốn đi Tokyo. "
     "Cửa khởi hành số mười hai, máy bay cất cánh lúc hai mươi mốt giờ. "
     "Hành lý xách tay không quá bảy ki lô gram."),
    ("ke_chuyen_co_tich.mp3",
     "Ngày xửa ngày xưa, có một nàng công chúa tên là Tấm. "
     "Tấm mồ côi mẹ từ nhỏ, sống với dì ghẻ và em cùng cha khác mẹ là Cám. "
     "Hàng ngày Tấm phải làm việc nặng nhọc, còn Cám chỉ chơi đùa."),
    ("ke_chuyen_dia_ly.mp3",
     "Việt Nam nằm ở Đông Nam Á, có diện tích ba trăm ba mươi nghìn ki lô mét vuông. "
     "Dân số khoảng một trăm triệu người. "
     "Thủ đô là Hà Nội, thành phố lớn nhất là Thành phố Hồ Chí Minh."),
    ("tin_moi_truong.mp3",
     "Bộ Tài nguyên và Môi trường cảnh báo chất lượng không khí Hà Nội ở mức kém. "
     "Người dân nên đeo khẩu trang khi ra đường. "
     "Sông Tô Lịch tiếp tục được nạo vét, kỳ vọng giảm ô nhiễm vào cuối năm."),
    ("tin_xay_dung.mp3",
     "Khu đô thị mới Thủ Thiêm chính thức đi vào hoạt động. "
     "Tuyến metro số một Bến Thành Suối Tiên đã chạy thử nghiệm thành công. "
     "Sân bay Long Thành dự kiến hoàn thành giai đoạn một vào năm hai nghìn hai mươi sáu."),
    ("hoi_thoai_mua_sam.mp3",
     "Em ơi, áo dài này bao nhiêu tiền? "
     "Áo này tám trăm năm mươi nghìn ạ. "
     "Có giảm giá không em? "
     "Chị mua hai cái em giảm còn một triệu năm nghìn."),
    ("ke_chuyen_van_hoa.mp3",
     "Tết Nguyên Đán là dịp lễ quan trọng nhất trong năm của người Việt. "
     "Mọi nhà gói bánh chưng, dọn dẹp nhà cửa, đón giao thừa cùng gia đình. "
     "Trẻ con háo hức nhận lì xì từ ông bà, cha mẹ."),
]


def gen_more_gtts() -> int:
    print("\n[3] gTTS — sinh thêm 10 mẫu VN")
    try:
        from gtts import gTTS
    except ImportError:
        return 0
    n_ok = 0
    for fname, text in VN_SCRIPTS_V2:
        out = AUD_DIR / fname
        if out.exists() and out.stat().st_size > 0:
            n_ok += 1
            continue
        try:
            print(f"  ⚙ {fname}", end=" ... ", flush=True)
            gTTS(text=text, lang="vi", slow=False).save(str(out))
            print(f"{out.stat().st_size // 1024} KB")
            n_ok += 1
            time.sleep(0.2)
        except Exception as e:
            print(f"FAIL: {e}")
    return n_ok


def main():
    n_wm = fetch_wikimedia_more(per_query=2, max_total=15)
    n_v = fetch_videos_v2()
    n_a = gen_more_gtts()

    print("\n=== Tổng kết phiên 2 ===")
    print(f"  Wikimedia images thêm: {n_wm}")
    print(f"  Videos thêm:           {n_v}")
    print(f"  gTTS VN audio thêm:    {n_a}")

    total = sum(f.stat().st_size for f in TEST_DATA.rglob("*") if f.is_file())
    n_files = sum(1 for f in TEST_DATA.rglob("*") if f.is_file())
    print(f"\nTotal test-data: {n_files} files, {total / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
