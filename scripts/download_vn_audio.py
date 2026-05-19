"""Sinh / tải Vietnamese audio để test ASR + BM25 ASR.

Chiến lược: thử HF VIVOS trước, fallback sang gTTS (Google TTS) sinh tổng hợp.
gTTS không cần auth, sinh file MP3 thật từ text Việt Nam.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "test-data" / "audio" / "vn_speech"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Mẫu kịch bản tin tức / nội dung VN đa dạng (test các query khác nhau)
VN_SCRIPTS = [
    (
        "tin_kinh_te.mp3",
        "Chào quý vị và các bạn. Đây là bản tin kinh tế buổi sáng. "
        "Thị trường chứng khoán Việt Nam hôm nay tăng ba phần trăm. "
        "Giá vàng tiếp tục lập đỉnh mới, đạt mức tám mươi lăm triệu đồng một lượng. "
        "Tỷ giá đô la Mỹ ổn định ở mức hai mươi lăm nghìn đồng.",
    ),
    (
        "tin_giao_thong.mp3",
        "Cập nhật giao thông Hà Nội. Đường Nguyễn Trãi tắc nghẽn nghiêm trọng giờ cao điểm. "
        "Tuyến đường Trường Chinh đang sửa chữa, đề nghị người dân tránh đi qua khu vực này. "
        "Phà Bình Khánh hoạt động bình thường, mưa rào rải rác trong chiều nay.",
    ),
    (
        "tin_the_thao.mp3",
        "Tin thể thao. Đội tuyển bóng đá Việt Nam vừa giành chiến thắng ba không trước đối thủ. "
        "Vận động viên cờ vua Lê Quang Liêm đứng thứ hai bảng xếp hạng thế giới. "
        "Giải bóng chuyền nữ vô địch quốc gia khai mạc tại thành phố Hồ Chí Minh.",
    ),
    (
        "doi_thoai_pho_bo.mp3",
        "Anh muốn ăn gì? Tôi gọi cho mình một bát phở bò tái nạm. "
        "Cô cho tôi thêm rau thơm và chanh nhé. "
        "Bao nhiêu tiền tất cả vậy? Tổng cộng năm mươi ngàn đồng.",
    ),
]


def try_hf_vivos() -> int:
    """Thử tải VIVOS từ HF (datasets format)."""
    print("\n[try VIVOS] AILAB-VNUHCM/vivos (HF datasets snapshot)")
    try:
        from huggingface_hub import snapshot_download

        local_dir = OUT_DIR / "_vivos_snapshot"
        snapshot_download(
            repo_id="AILAB-VNUHCM/vivos",
            repo_type="dataset",
            local_dir=str(local_dir),
            allow_patterns=["*.wav", "*.mp3", "*.flac"],
            max_workers=4,
        )
        audios = list(local_dir.rglob("*.wav")) + list(local_dir.rglob("*.mp3"))
        print(f"  → tải về {len(audios)} file audio")
        return len(audios)
    except Exception as e:
        print(f"  ✗ {e}")
        return 0


def gen_gtts_samples() -> int:
    """Sinh các mẫu VN qua gTTS."""
    print("\n[gTTS] Sinh 4 mẫu tiếng Việt qua Google TTS")
    try:
        from gtts import gTTS
    except ImportError:
        print("  ✗ gtts chưa cài. pip install gtts")
        return 0
    n_ok = 0
    for fname, text in VN_SCRIPTS:
        out = OUT_DIR / fname
        if out.exists():
            print(f"  ✓ {fname} (đã có)")
            n_ok += 1
            continue
        print(f"  ⚙ {fname}", end=" ... ", flush=True)
        try:
            gTTS(text=text, lang="vi", slow=False).save(str(out))
            size = out.stat().st_size
            print(f"{size // 1024} KB")
            n_ok += 1
        except Exception as e:
            print(f"FAIL: {e}")
    return n_ok


def main():
    n_hf = try_hf_vivos()
    n_tts = gen_gtts_samples() if n_hf == 0 else 0
    files = sorted(
        f for f in OUT_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg")
    )
    total_kb = sum(f.stat().st_size for f in files) / 1024
    print(f"\n=== {len(files)} file audio VN, total {total_kb:.0f} KB ===")
    for f in files[:10]:
        print(f"  {f.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
