# FUFU v2 — Multimedia Search

Hệ thống tìm kiếm **multimedia** (video / audio / image) theo ngôn ngữ tự nhiên.

## Stack

| Tầng | Công nghệ | Vai trò |
|---|---|---|
| Visual encoder | **SigLIP-2 Base 384** | Embed image / video frame + text query, multilingual native |
| OCR | **PaddleOCR (VN/EN)** | Chữ trên màn (biển hiệu, banner, phụ đề) |
| Caption + Scene | **Qwen2.5-VL-7B INT4** | Mô tả semantic per frame bằng tiếng Việt |
| Object detection | **YOLO-World v2 (open-vocab)** | Đối tượng bất kỳ theo text prompt |
| ASR | **PhoWhisper-medium** | Lời thoại VN từ audio file + audio track của video |
| Translation | **NLLB-200 distilled** | Query expansion VI ↔ EN |
| Paraphrase | **Qwen2.5-3B INT4** | Query paraphrase |
| Vector DB | **FAISS HNSW** | Visual retrieval |
| Text BM25 | **SQLite FTS5** × 2 | Frame annotations (OCR+caption+labels) + ASR transcripts |
| Backend | **FastAPI** | API + serve thumbnails |
| Frontend | **React 18 + Vite** | UI 3 loại card (video/audio/image) |

## Cấu trúc thư mục

```
app/
├── common/
│   ├── config.py
│   ├── encoder.py            (SigLIP-2 shared image+text)
│   ├── types.py              (MediaType, FrameAnnotation, ASRSegment, ...)
│   └── audio_io.py           (ffmpeg PCM mono 16k load)
├── extractors/               (cross-modal, lazy singletons)
│   ├── ocr.py                (PaddleOCR)
│   ├── caption.py            (Qwen2.5-VL)
│   ├── detection.py          (YOLO-World)
│   └── asr.py                (PhoWhisper)
├── ingest/
│   ├── pipeline.py           (router theo media_type)
│   ├── cli.py
│   ├── storage.py            (IndexWriter — schema items/frames/asr + 2 FTS5)
│   ├── utils.py
│   ├── image/ingest.py
│   ├── audio/ingest.py
│   └── video/
│       ├── ingest.py
│       ├── shots.py
│       ├── keyframes.py
│       └── segments.py
└── backend/
    ├── main.py
    ├── api/{search,health}.py
    └── services/{encoder,translator,paraphraser,retrieval,rerank,search_engine}.py
```

## Ingest flow theo media type

### Image (.jpg, .png, .webp, ...)
```
image
  → resize keep-aspect (max 1024)
  → SigLIP-2 encode → 1 vector
  → OCR (PaddleOCR)
  → Caption (Qwen2.5-VL)
  → Detection (YOLO-World)
  → 1 item, 1 segment, 1 frame, mọi annotation
  → FAISS + frame_text FTS5
```

### Audio (.mp3, .wav, .m4a, ...)
```
audio
  → ffmpeg → PCM mono 16k
  → PhoWhisper ASR → ASR chunks (start, end, text)
  → merge chunks cách nhau ≤ 0.5s (tránh phân mảnh)
  → segments = ASR chunks (mỗi đoạn lời = 1 'cảnh tự nhiên')
  → chunk > 15s → subdivide đều
  → asr_text FTS5 (KHÔNG có vector visual)
  → fallback sliding window 10s/5s nếu KHÔNG phát hiện speech
```

**Tại sao ASR chunks làm segments?** Cùng triết lý với shots-as-segments của video — mỗi segment là 1 đơn vị nội dung tự nhiên (đoạn lời pause-bounded), không phải cắt cứng theo timer. Operator nhảy đến đoạn lời cụ thể thay vì cửa sổ 10s arbitrary.

Audio không speech (nhạc thuần, ambient) → sliding window placeholder, item được log nhưng không retrievable qua text query (cần CLAP audio-event detection — chưa có).

### Video (.mp4, .mkv, ...)
```
video
  → PySceneDetect → shots (start/end CHÍNH XÁC từng cảnh)
  → adaptive keyframes (mật độ 1 frame / 1s × duration shot)
  → mỗi keyframe:
    SigLIP-2 encode + OCR + Detection
    (caption Qwen-VL tắt mặc định — SigLIP đủ làm tầng dense)
  → segments = shots (shot > 15s subdivide)
  → ffmpeg → audio track
  → PhoWhisper ASR → ASR segments gán vào shot overlap nhiều nhất
  → FAISS + frame_text FTS5 + asr_text FTS5
```

**Tại sao shots làm segments thay vì sliding window?**
- Mỗi segment = 1 cảnh nguyên vẹn (semantically coherent unit) thay vì cắt ngang giữa cảnh
- Operator nhảy navigate theo cảnh trực quan hơn
- start/end là ranh giới camera cut thật, không phải timestamp tròn số

**Bật lại caption?** Sửa `extractors.enable_caption: true` trong `settings.yaml`. Sẽ chậm ingest ~5-10× vì Qwen-VL bottleneck 1.5s/frame.

## Query flow (3 kênh hybrid)

```
Query VI
  ├─ NLLB → EN
  ├─ Qwen-3B → 3 paraphrase
  └─ SigLIP-2 text encode → mean → L2-norm
                            ↓
  ┌──────────────┬─────────────┬──────────────┐
  │ Dense FAISS  │ BM25 visual │ BM25 ASR     │
  │ top-500      │ frame_text  │ asr_text     │
  │ (frames)     │ top-200     │ top-200      │
  └──────┬───────┴──────┬──────┴──────┬───────┘
         ↓              ↓             ↓
    Min-max norm    Min-max norm  Min-max norm
         ↓              ↓             ↓
    w_d=0.60       w_v=0.25      w_a=0.15
         └──────────────┴─────────────┘
                      ↓
        Aggregate theo (item_id, segment_id)
        max-pool mỗi kênh → cộng có trọng số
                      ↓
        Top-K Hits với best_frame + best_asr
                      ↓
              JSON response
```

## Quickstart

```bash
# 1. Cài deps
pip install -r requirements.txt
cd frontend && npm install && cd ..

# 2. Pre-download model (~25GB tổng — SigLIP 0.4 + Qwen-VL-7B INT4 5 + Qwen-3B 2.5 + NLLB 1.3 + PhoWhisper 3 + YOLO-World 0.6)
python scripts/download_models.py

# 3. Ingest multimedia (mix bất kỳ video / audio / image)
python -m app.ingest.cli /path/to/media/
# Chỉ 1 loại:
python -m app.ingest.cli /path/to/media/ --only video

# 4. Chạy backend + frontend
uvicorn app.backend.main:app --host 0.0.0.0 --port 8080
cd frontend && npm run dev    # → http://localhost:3006
```

## Test với 3090 24GB (khuyến nghị)

Stack online query (paraphrase + translation + SigLIP):
- SigLIP-2 Base 384 (fp16): 0.4 GB
- NLLB-200 (fp16): 1.3 GB
- Qwen2.5-3B INT4: 2.5 GB
- **Tổng online**: ~5 GB

Stack ingest (1 video tại 1 thời điểm):
- SigLIP-2 + OCR + YOLO-World + Qwen-VL-7B INT4 + PhoWhisper-medium: ~13 GB
- **Vẫn dư ~10 GB** trên 3090 cho activations.

Xem `HARDWARE.md` cho GPU thấp hơn.

## Storage schema

```sql
items          (id, path, media_type, duration_sec)
segments       (id, item_id, seg_idx, start_sec, end_sec)
frames         (id, item_id, timestamp, thumbnail_path, faiss_id, caption, objects_json)
frame_segments (frame_id, segment_id)
asr_segments   (id, item_id, start, end, text, segment_id)
frame_text     FTS5 (ocr_text, caption, labels)        -- rowid = frame_id
asr_text       FTS5 (transcript)                       -- rowid = asr_id
```

## Thumbnails có giữ lại sau ingest

Có — frontend hiển thị `best_frame.thumbnail` cho mỗi kết quả search. FastAPI mount `/thumbnails` để serve. Nếu xóa thì grid kết quả không có ảnh preview.

Disk overhead: ~30 KB / keyframe JPEG q=85. 1000 frame ≈ 30 MB.

## API response (multimedia)

```json
{
  "query": "...",
  "results": [
    {
      "item_id": 1,
      "media_type": "video",
      "item_path": "...",
      "segment_id": 42,
      "segment_start": 10.0,
      "segment_end": 15.0,
      "score": 0.62,
      "score_breakdown": {"dense": 0.85, "bm25_visual": 0.3, "bm25_asr": 0.1},
      "best_frame": {
        "frame_id": 891,
        "timestamp": 12.5,
        "thumbnail": "...",
        "caption": "Người chơi cờ vua đang di chuyển quân...",
        "objects": [{"label":"chess piece","conf":0.92,"bbox":[...]}],
        "raw_cosine": 0.14
      },
      "best_asr": {
        "asr_id": 12,
        "start": 11.2,
        "end": 14.0,
        "text": "Nước này là tốt nhất ở đây"
      }
    },
    {
      "item_id": 5,
      "media_type": "audio",
      "best_frame": null,
      "best_asr": {"start": 5.5, "end": 9.2, "text": "..."}
    },
    {
      "item_id": 7,
      "media_type": "image",
      "best_frame": {"caption": "...", "objects": [...]}
    }
  ]
}
```

## Khoảng trống còn lại (so với combo đầy đủ trong báo cáo)

- ❌ Scene graph (sáng tạo #3 báo cáo) — chưa có
- ❌ Session adapter (sáng tạo #2) — operator phải gõ lại
- ❌ Temporal event graph cross-frame — chỉ có sliding window
- ❌ Synthetic query augmentation (15/frame) — có thể thêm dễ qua Qwen-VL
- ❌ Audio event detection (CLAP) — chỉ có ASR cho audio
