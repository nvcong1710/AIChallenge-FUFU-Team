# FUFU — Project Context (cho AI đọc)

> **File này là gì:** tài liệu ngữ cảnh đầy đủ, tự-chứa, mô tả toàn bộ hệ thống FUFU
> như **code đang thực sự chạy** (không phải như README cũ mô tả). Mục đích: đưa cho một
> AI/người mới để họ hiểu hệ thống và bắt tay sửa code mà không cần đọc lại từng file.
>
> **Quy tắc đọc:**
> 1. Code trong `app/` + `config/settings.yaml` là **source of truth duy nhất**.
> 2. `README-V2.md` và `HARDWARE.md` **đã lỗi thời ở vài chỗ** — xem mục [§2 Điểm lệch tài liệu](#2-điểm-lệch-tài-liệu-đọc-trước-khi-tin-readme-cũ).
> 3. Trước khi sửa, **verify lại bằng cách mở file được trỏ tới** — file/dòng có thể đã đổi sau ngày cập nhật doc này.
>
> **Cập nhật:** 2026-06-12 · phản ánh commit `d4bf91e` (chunked persist + signal handler).

---

## 1. TL;DR (đọc trong 30 giây)

**FUFU** = công cụ **tìm kiếm multimedia bằng ngôn ngữ tự nhiên tiếng Việt** cho cuộc thi
**HCM AI Challenge 2026** (bài toán Video Browsing / Known-Item Search — người dùng gõ mô tả,
hệ thống trả về đoạn video/audio/ảnh khớp nhất kèm timestamp để nhảy đến).

Một câu: **gõ câu tiếng Việt → mở rộng query (dịch + paraphrase) → tìm song song 3 kênh
(SigLIP dense visual + BM25 trên annotation hình + BM25 trên lời thoại) → hợp nhất có trọng số →
rerank cross-encoder → trả top-K đoạn kèm thumbnail + timestamp.**

| | |
|---|---|
| **Ngôn ngữ** | Python 3.10 (backend/ingest), JS/React (frontend). Comment code = tiếng Việt. |
| **Chạy được chưa** | Có. Ingest + backend + frontend đều hoạt động. Cần GPU (CUDA) để đầy đủ tính năng. |
| **Hardware đích** | RTX 3090 24GB (ingest). Online query ~5GB VRAM. |
| **Storage** | FAISS (vector) + SQLite (metadata + 2× FTS5 BM25). Không có DB ngoài. |
| **Entry points** | Ingest: `python -m app.ingest.cli <path>` · Backend: `uvicorn app.backend.main:app --port 8080` · Frontend: `cd frontend && npm run dev` (:3006) |

---

## 2. Điểm lệch tài liệu (ĐỌC TRƯỚC khi tin README cũ)

Các file `README-V2.md`, `HARDWARE.md` viết ở giai đoạn "v2 plan" và **chưa đồng bộ với code/config hiện tại**:

| Chỗ | README/HARDWARE nói | Code/config THỰC TẾ | Nguồn sự thật |
|---|---|---|---|
| OCR engine | PaddleOCR | **EasyOCR** (`[vi, en]`, py3.12 compat) | `app/extractors/ocr.py`, `requirements.txt` |
| Visual encoder | SigLIP-2 **Base** 384 | SigLIP-2 **Large** patch16-384 | `config/settings.yaml:8` |
| Trọng số hybrid | dense 0.60 / visual 0.25 / asr 0.15 | **dense 0.40 / visual 0.25 / asr 0.50** (đang ưu tiên ASR) | `config/settings.yaml:69-72` |
| Caption | "tắt mặc định" | **đang BẬT** (`enable_caption: true`) — bottleneck ingest | `config/settings.yaml:16` |
| Scene graph | "❌ chưa có" | **ĐÃ CÓ**: bảng `scenes` + clustering shot theo cosine | `app/ingest/video/scenes.py`, `storage.py` |
| Reranker | không nhắc | **ĐÃ CÓ**: BGE-reranker-v2-m3 cross-encoder, rerank top-50 | `app/backend/services/reranker.py` |

> Khi sửa, lấy `config/settings.yaml` + file code làm chuẩn. Nếu sửa hành vi, **cập nhật cả file này**.

---

## 3. Bối cảnh bài toán

- **Cuộc thi:** HCM AI Challenge 2026 — Multimedia Event Retrieval / Video Browsing.
- **Input thi:** một corpus lớn video (kèm audio), ảnh; truy vấn là **mô tả ngôn ngữ tự nhiên** (thường tiếng Việt) về một cảnh/sự kiện cần tìm ("known-item search").
- **Output cần:** danh sách đoạn (video + mốc thời gian / ảnh) xếp theo độ liên quan, để operator nhảy nhanh đến đúng cảnh.
- **Thách thức đặc thù:**
  - Truy vấn tiếng Việt nhưng nhiều model visual mạnh ở tiếng Anh → cần **query expansion song ngữ**.
  - Cảnh có thể chỉ phân biệt được qua **chữ trên màn (OCR)** hoặc **lời thoại (ASR)**, không chỉ hình ảnh → cần **hybrid đa kênh**.
  - Cần nhảy đến **đoạn/cảnh cụ thể**, không chỉ "video nào" → cắt theo **shot/scene tự nhiên** thay vì cửa sổ thời gian cứng.

---

## 4. Tech stack thực tế

| Tầng | Công nghệ (model id) | Vai trò | File |
|---|---|---|---|
| Visual encoder | `google/siglip2-large-patch16-384` (fp16) | Embed ảnh/frame **và** text query vào cùng không gian, multilingual | `app/common/encoder.py` |
| OCR | **EasyOCR** `[vi, en]` | Chữ trên màn (biển hiệu, banner, phụ đề) | `app/extractors/ocr.py` |
| Caption | `Qwen/Qwen2.5-VL-7B-Instruct` (INT4 nf4) | Mô tả semantic per-frame bằng tiếng Việt | `app/extractors/caption.py` |
| Object detection | YOLO-World v2 (`yolov8l-world.pt`, open-vocab) | 62 lớp đối tượng theo text prompt (`DEFAULT_CLASSES`) | `app/extractors/detection.py` |
| ASR | `vinai/PhoWhisper-medium` | Lời thoại VN từ audio + audio track của video | `app/extractors/asr.py` |
| Translation | `facebook/nllb-200-distilled-600M` | Query expansion VI→EN | `app/backend/services/translator.py` |
| Paraphrase | `Qwen/Qwen2.5-3B-Instruct` (INT4) | Sinh 3 cách diễn đạt khác cho query | `app/backend/services/paraphraser.py` |
| Reranker | `BAAI/bge-reranker-v2-m3` (cross-encoder) | Rerank top-50 hits cuối | `app/backend/services/reranker.py` |
| Vector DB | **FAISS** `IndexHNSWFlat`, inner-product | Dense visual retrieval | `app/ingest/storage.py`, `services/retrieval.py` |
| Text search | **SQLite FTS5** × 2 (BM25) | Annotation hình (OCR+caption+labels) & transcript ASR | cùng trên |
| Backend | **FastAPI** + uvicorn (:8080) | API search/stats + serve thumbnail | `app/backend/main.py` |
| Frontend | **React 18 + Vite** (:3006) | UI gõ query + grid kết quả | `frontend/` |

Pin quan trọng: `transformers==4.50.0` (4.49 thiếu SigLIP-2; 5.x phá API). `numpy<2`. `faiss-cpu` (FAISS chạy CPU, model chạy CUDA).

---

## 5. Bản đồ repository

```
app/
├── common/
│   ├── config.py          # get_config() đọc settings.yaml (lru_cache), resolve path tương đối → tuyệt đối
│   ├── encoder.py          # SiglipEncoder: encode_images() + encode_text(), tự detect dim, L2-normalize
│   ├── types.py            # MediaType enum, detect_media_type(), FrameAnnotation, ASRSegment, DetectionBox
│   └── audio_io.py         # load_audio_mono_16k() via ffmpeg, audio_duration()
├── extractors/             # cross-modal, LAZY SINGLETON (load 1 lần, share ingest↔backend)
│   ├── __init__.py         # get_ocr/get_caption/get_detection/get_asr(cfg) — cache trong _singletons dict
│   ├── ocr.py              # EasyOCR → annotation.ocr_text + ocr_lines
│   ├── caption.py          # Qwen2.5-VL → annotation.caption (greedy, ≤96 tokens)
│   ├── detection.py        # YOLO-World → annotation.objects (DEFAULT_CLASSES ~70 lớp)
│   └── asr.py              # PhoWhisper pipeline → List[ASRSegment]
├── ingest/
│   ├── cli.py              # `python -m app.ingest.cli <paths> [--only video|audio|image]`
│   ├── pipeline.py         # run_ingest(): dispatch theo media_type + SIGINT/SIGTERM handler → persist trước khi die
│   ├── storage.py          # IndexWriter: SCHEMA (SQL+FTS5) + FAISS, add_frames/add_asr/add_scenes, persist()
│   ├── utils.py            # collect_files, group_by_type, resize_keep_aspect, save_thumbnail
│   ├── image/ingest.py     # 1 ảnh → 1 item/1 seg/1 frame + mọi annotation
│   ├── audio/
│   │   ├── ingest.py       # ASR → segments = đoạn lời; fallback sliding window
│   │   └── segments.py     # asr_chunks_to_segments, merge_close_chunks, build_sliding_segments
│   └── video/
│       ├── ingest.py       # PIPELINE CHÍNH video — CHUNKED persist mỗi 16 frame
│       ├── shots.py         # detect_shots() qua PySceneDetect
│       ├── keyframes.py     # extract_keyframes_adaptive() density 1/s, clamp [1,12]/shot
│       ├── segments.py      # shots_to_segments (subdivide >15s), assign_frames_to_segments, sliding fallback
│       └── scenes.py        # cluster_shots_into_scenes(): gom shot kề cosine≥0.85 → scene
└── backend/
    ├── main.py             # FastAPI app, mount /thumbnails, CORS *, include routers
    ├── api/
    │   ├── search.py        # POST /api/search, GET /api/stats; SearchEngine lru_cache singleton
    │   └── health.py        # GET /health
    └── services/
        ├── search_engine.py # SearchEngine: expand_query() + search() — ORCHESTRATOR chính của query
        ├── retrieval.py      # Retriever: faiss_search + bm25_visual + bm25_asr + enrichment SQL
        ├── rerank.py         # fuse_and_aggregate(): hybrid score fusion + gom frame→segment (KHÔNG phải cross-encoder)
        ├── reranker.py       # BGEReranker: cross-encoder rerank (LƯU Ý tên gần giống rerank.py)
        ├── translator.py     # NLLB VI→EN
        └── paraphraser.py    # Qwen2.5-3B paraphrase

config/
├── settings.yaml           # CONFIG CHÍNH (source of truth cho mọi tham số)
└── settings_local.yaml     # override local (FUFU_CONFIG env hoặc path arg)

frontend/src/
├── App.jsx                 # state + gọi searchAPI/fetchStats
├── api.js                  # searchAPI(), fetchStats(), thumbnailURL() — BASE = VITE_API_BASE || :8080
├── components/SearchBox.jsx, ResultGrid.jsx
└── styles.css

scripts/                    # tiện ích: download_models.py, eval_accuracy.py, eval_html_report.py,
                            # search_demo.py, db_inspector.py, download_msrvtt.py, translate_msrvtt_to_vn.py ...
data/                       # OUTPUT runtime: index.faiss, meta.sqlite, thumbnails/ (gitignored)
```

> ⚠️ **Bẫy tên file:** `rerank.py` (hợp nhất điểm số, hàm `fuse_and_aggregate`) ≠ `reranker.py`
> (cross-encoder BGE). Dễ nhầm. Khi nói "rerank" cần rõ là **score fusion** hay **cross-encoder**.

---

## 6. Mô hình dữ liệu (SQLite + FAISS)

Định nghĩa đầy đủ trong `app/ingest/storage.py` (`SCHEMA`). Quan hệ:

```
items (1) ──< segments (N) ──> scenes (mỗi segment có thể thuộc 1 scene)
items (1) ──< frames (N)
frames (M) ──< frame_segments >── (N) segments     # 1 frame có thể thuộc nhiều segment (overlap)
items (1) ──< asr_segments (N) ──> segment_id (gán theo overlap lớn nhất)

FAISS index            : vector visual (image frame). faiss_id ↔ frames.faiss_id
frame_text (FTS5)      : rowid = frames.id ; cột (ocr_text, caption, labels)
asr_text   (FTS5)      : rowid = asr_segments.id ; cột (transcript)
```

| Bảng | Cột chính | Ghi chú |
|---|---|---|
| `items` | id, path (UNIQUE), media_type (video/audio/image), duration_sec | 1 file = 1 item |
| `segments` | id, item_id, seg_idx, start_sec, end_sec, **scene_id** | đơn vị nhảy-đến. Video: 1 shot. Audio: 1 đoạn lời. Image: [0,0] |
| `scenes` | id, item_id, scene_idx, start/end, n_shots | gom nhiều shot kề nhau (chỉ video) |
| `frames` | id, item_id, timestamp, thumbnail_path, **faiss_id** (UNIQUE), caption, objects_json | mỗi keyframe = 1 hàng + 1 vector FAISS |
| `frame_segments` | frame_id, segment_id | M-N |
| `asr_segments` | id, item_id, start/end, text, segment_id | lời thoại; audio-only item KHÔNG có vector visual |
| `frame_text` FTS5 | ocr_text, caption, labels | `tokenize='unicode61 remove_diacritics 0'` → **GIỮ dấu tiếng Việt** |
| `asr_text` FTS5 | transcript | như trên |

**Bất biến quan trọng:**
- `faiss_id` được gán tuần tự = `index.ntotal` tại thời điểm add (xem `storage.py:add_frames`). FAISS và SQLite phải đồng bộ — **đừng add vào FAISS mà không ghi `frames` tương ứng**.
- Audio thuần (không speech) → có `items`/`segments` placeholder nhưng **không retrievable qua text** (cần CLAP audio-event, chưa có).
- Vector trong FAISS đã **L2-normalize**; metric = inner product = cosine.

---

## 7. Ingest pipeline (3 nhánh theo media type)

`run_ingest()` (`pipeline.py`) tạo 1 `SiglipEncoder` + 1 `IndexWriter` dùng chung, đăng ký
signal handler (Ctrl+C/SIGTERM → `writer.persist()` rồi exit sạch — không mất work), rồi
dispatch từng file theo `detect_media_type()`.

### 7.1 Image — `image/ingest.py`
```
ảnh → resize keep-aspect (max 1024)
    → 1 item / 1 segment [0,0] / 1 frame
    → OCR + Caption + Detection (annotation)
    → SigLIP encode → 1 vector → FAISS
    → frame_text FTS5
```

### 7.2 Audio — `audio/ingest.py`
```
audio → ffmpeg PCM mono 16k
      → PhoWhisper ASR → chunks (start,end,text)
      → merge chunks gap ≤ 0.5s (chống phân mảnh)
      → segments = các đoạn lời (chunk >15s thì subdivide đều)
      → asr_text FTS5    (KHÔNG có vector visual)
      → nếu KHÔNG có speech: fallback sliding window 10s/5s (item không tìm được qua text)
```
**Triết lý:** mỗi đoạn lời (pause-bounded) = 1 "cảnh tự nhiên" → operator nhảy đến lời cụ thể.

### 7.3 Video — `video/ingest.py` (pipeline phức tạp nhất, CHUNKED)
```
video
 1. PySceneDetect → shots (ranh giới camera-cut, threshold 27.0)
 2. extract_keyframes_adaptive: mỗi shot lấy ceil(duration×1.0) frame, clamp [1,12]
 3. segments = shots (shot >15s subdivide); assign_frames_to_segments theo timestamp
 4. LẶP theo chunk 16 frame:
      annotate (OCR + Caption + Detection) → SigLIP encode → save thumbnail
      → writer.add_frames(...) → writer.persist()   ← kill ở đây chỉ mất ≤16 frame
 5. cluster_shots_into_scenes: gom shot kề nếu cosine(frame cuối shot_i, frame đầu shot_i+1) ≥ 0.85
 6. ffmpeg tách audio track → PhoWhisper ASR → gán mỗi ASR segment vào shot overlap lớn nhất
```
**Tại sao shot-as-segment thay vì cửa sổ cứng:** mỗi segment là 1 cảnh nguyên vẹn, start/end là
camera-cut thật → navigate trực quan, không cắt ngang cảnh.

**Caption là BOTTLENECK** (~1.5s/frame với Qwen-VL INT4). Tắt = sửa `extractors.enable_caption: false`
→ nhanh ~5-10× nhưng mất signal semantic (giảm recall ~5-10% trên query mơ hồ).

---

## 8. Query / Search pipeline

Orchestrator: `SearchEngine.search()` trong `search_engine.py`. Luồng:

```
query VI (vd "người chơi cờ vua")
  │
  ├─ expand_query():
  │     ├─ NLLB: VI→EN (num_beams=2)              → translated
  │     ├─ Qwen2.5-3B: 3 paraphrase (temp 0.7)    → paraphrases
  │     ├─ "all"  = [original, translated, *paraphrases]  (cho DENSE)
  │     └─ "bm25" = [original, translated]                (BM25 — BỎ paraphrase để không nhiễu phrase)
  │
  ├─ encode_text("all") → mean → L2-norm → q_vec   (1 vector đại diện)
  │
  ├──────────────┬────────────────────┬─────────────────────┐
  │ DENSE FAISS  │ BM25 visual         │ BM25 ASR            │
  │ q_vec top-500│ frame_text top-200  │ asr_text top-200    │
  │ (cosine)     │ OR các token        │ OR các token        │
  └──────┬───────┴─────────┬───────────┴──────────┬──────────┘
         │ min-max norm     │ raw/8.0 cap 1.0       │ raw/8.0 cap 1.0
         │                  │ (filter raw < 3.0)    │ (filter raw < 3.0)
         └──────────────────┴───────────────────────┘
                            │
   fuse_and_aggregate(): gom theo (item_id, segment_id), max-pool mỗi kênh,
   score = 0.40·dense + 0.25·bm25_visual + 0.50·bm25_asr   (weights từ config)
                            │
   BGE-reranker cross-encoder: rerank top-50
   (passage = caption + "objects: ..." + ASR text) → reorder
                            │
   top-K (mặc định 20) → enrich segment/item/scene meta → JSON
```

**Chi tiết then chốt:**
- **Dense dùng tất cả biến thể; BM25 chỉ original+translated.** Lý do (comment trong code): paraphrase dài làm phrase-match OCR/ASR ngắn bị nhiễu.
- **FTS5 query = OR các token** (không phải phrase match) — xem `Retriever._build_fts_or_query`. Token <2 ký tự bị loại; lọc ký tự đặc biệt FTS5; giữ ký tự tiếng Việt có dấu.
- **Filter nhiễu:** raw BM25 < `MIN_BM25_RAW = 3.0` bị bỏ (loại match 1-token rác).
- **Chuẩn hoá điểm bất đối xứng:** dense = min-max (cosine bounded), BM25 = `raw / BM25_SCALE(8.0)` cap 1.0 → giữ "độ mạnh tuyệt đối" của BM25 thay vì equalize về 1.0 khi chỉ có 1 hit.
- **KHÔNG renormalize final score** → item match nhiều kênh ăn điểm cao hơn item 1 kênh (đúng intent). Weights tune sao cho audio (chỉ ASR) vẫn cạnh tranh được với video (chỉ dense).
- **best_frame / best_asr**: mỗi Hit giữ frame đại diện (cosine cao nhất) + snippet ASR mạnh nhất để hiển thị.

---

## 9. Tham số config (giải thích từng khối — `config/settings.yaml`)

```yaml
storage:        # path output; config.py tự resolve tương đối → tuyệt đối theo PROJECT_ROOT
models:
  siglip: google/siglip2-large-patch16-384   # đổi sang base-384 nếu thiếu VRAM
  device: cuda                                # "cpu" để chạy không GPU (chậm, paraphrase sẽ tắt)
extractors:
  enable_ocr/caption/detection/asr: true      # tắt từng cái để giảm VRAM/thời gian ingest
  ocr_min_confidence: 0.4
  caption_quant_4bit: true                     # INT4 ~5GB; false = bf16 ~14GB nhanh hơn
  asr_model: vinai/PhoWhisper-medium           # -small nếu GPU nhỏ
ingest:
  video:
    shot_detect_threshold: 27.0                # PySceneDetect: thấp = nhiều shot hơn
    keyframe_density_per_sec: 1.0              # 1 frame/giây/shot, clamp [min=1, max=12]
    use_shots_as_segments: true
    max_segment_len_sec: 15.0                  # shot dài hơn → subdivide
    chunk_size_frames: 16                      # persist mỗi 16 frame (resilience)
  audio:
    use_asr_as_segments: true
    merge_close_chunks_sec: 0.5
retrieval:
  top_k_dense: 500 / top_k_bm25_*: 200 / top_k_final: 20
  enable_reranker: true / rerank_top_k: 50
  weights: {dense: 0.4, bm25_visual: 0.25, bm25_asr: 0.5}   # ⭐ tham số tune chính
  hnsw_m: 32 / ef_construct: 200 / ef_search: 128            # FAISS HNSW; ef_search cao = recall cao, chậm hơn
query_expansion:
  enable_translation/paraphrase: true / num_paraphrases: 3
```

**Override config:** đặt biến môi trường `FUFU_CONFIG=/path/to/file.yaml`, hoặc truyền path vào `get_config(path)`. `get_config` dùng `lru_cache` → đổi config phải restart process.

---

## 10. API contract (FastAPI :8080)

| Endpoint | Method | Body / Param | Trả về |
|---|---|---|---|
| `/health` | GET | — | health check |
| `/api/stats` | GET | — | `{items, items_video/audio/image, frames, segments, scenes, asr_segments, faiss_total}` |
| `/api/search` | POST | `{"query": str(1..500), "top_k": int(1..100, default 20)}` | xem dưới |
| `/thumbnails/...` | GET | static | file JPEG thumbnail |
| `/` | GET | — | liệt kê endpoint |

**Response `/api/search`** (rút gọn — đầy đủ trong `search_engine.py:search` cuối hàm):
```json
{
  "query": "...",
  "expanded_queries": ["...","..."],     // biến thể dùng cho dense
  "bm25_queries": ["...","..."],
  "translated": "english version",
  "num_dense": 500, "num_bm25_visual": 12, "num_bm25_asr": 3,
  "results": [
    {
      "item_id": 1, "media_type": "video", "item_path": "...",
      "segment_id": 42, "segment_start": 10.0, "segment_end": 15.0,
      "scene_id": 5, "scene_start": 8.0, "scene_end": 20.0, "scene_n_shots": 3,
      "score": 0.62,
      "score_breakdown": {"dense": 0.85, "bm25_visual": 0.3, "bm25_asr": 0.1},
      "best_frame": {"frame_id","timestamp","thumbnail","caption","objects":[{"label","conf","bbox"}],"raw_cosine"},
      "best_asr":   {"asr_id","start","end","text"}
    }
  ],
  "timing_ms": {"expand_ms","encode_ms","faiss_ms","bm25_visual_ms","bm25_asr_ms","fetch_meta_ms","rerank_ms","cross_rerank_ms"}
}
```
`best_frame` = null với audio; `best_asr` = null với image/video không lời.

---

## 11. Frontend (React + Vite :3006)

- `api.js`: `BASE = VITE_API_BASE || "http://localhost:8080"`. `thumbnailURL()` cắt path tuyệt đối server thành URL `/thumbnails/...`.
- `App.jsx`: gọi `fetchStats()` lúc mount (hiển thị đếm 🎥🎵🖼/frames/scenes/asr), `searchAPI()` khi submit. Có panel `<details>` debug: biến thể query, số hit mỗi kênh, `timing_ms`.
- `ResultGrid.jsx`: 3 loại card (video/audio/image). `SearchBox.jsx`: ô nhập + nút.
- Đổi port/host backend: set `VITE_API_BASE` khi build/dev.

---

## 12. Hardware & hiệu năng (xem thêm `HARDWARE.md`, lưu ý nó dùng số của SigLIP Base)

- **Ingest đồng thời** (3090, full): SigLIP 0.4G + EasyOCR 1-2G + YOLO-World 1.5G + Qwen-VL INT4 5G + PhoWhisper 3G ≈ **13GB** → dư ~10GB.
- **Online query**: SigLIP 0.4G + NLLB 1.3G + Qwen-3B INT4 2.5G ≈ **5GB** → vừa GPU 8GB.
- **Bottleneck ingest = Qwen-VL caption ~1.5s/frame.** 1 phút video (~30 keyframe) ≈ 3-4 phút wall-time. 100h video ≈ ~24h trên 1×3090.
- **Giảm tải:** tắt `enable_caption` (nhanh ~10×), dùng PhoWhisper-small, hoặc multi-process ingest.
- **Disk:** models ~25GB; FAISS ~3GB/1M vec; thumbnail ~30KB/frame.

---

## 13. Cách chạy (quickstart)

```bash
# 1. deps
pip install -r requirements.txt
cd frontend && npm install && cd ..

# 2. tải model trước (~25GB)
python scripts/download_models.py

# 3. ingest (tự nhận diện loại; --only để lọc)
python -m app.ingest.cli /path/to/media/
python -m app.ingest.cli /path/to/media/ --only video

# 4. backend + frontend
uvicorn app.backend.main:app --host 0.0.0.0 --port 8080
cd frontend && npm run dev        # http://localhost:3006

# Tiện ích:
python scripts/db_inspector.py            # soi nội dung meta.sqlite
python scripts/search_demo.py "<query>"   # test search không cần frontend
python scripts/eval_accuracy.py           # đánh giá (MSR-VTT translated VN)
```
Ingest **idempotent** ở mức item: file đã ingest (có frame hoặc asr) sẽ bị bỏ qua (`item_already_ingested`).

---

## 14. Khoảng trống / roadmap (so với báo cáo giải pháp)

Trong `BAO-CAO-TONG-HOP.md` / `CHECKLIST-GIAI-PHAP.md` có nhiều "sáng tạo" — trạng thái thực tế:

| Tính năng | Trạng thái |
|---|---|
| Scene clustering (gom shot) | ✅ đã có (`scenes.py`) — đơn giản theo cosine biên, chưa phải scene graph đầy đủ |
| Cross-encoder rerank | ✅ đã có (BGE-reranker) |
| Query expansion VI↔EN + paraphrase | ✅ đã có |
| Temporal event graph cross-frame | ❌ chưa (chỉ shot/scene tuyến tính) |
| Session adapter (nhớ ngữ cảnh truy vấn) | ❌ chưa — mỗi query độc lập |
| Synthetic query augmentation (sinh query/frame) | ❌ chưa (dễ thêm qua Qwen-VL) |
| Audio event detection (CLAP) cho audio không lời | ❌ chưa — audio không speech không tìm được |

---

## 15. Glossary

- **item**: 1 file media (1 video / 1 audio / 1 ảnh).
- **frame**: 1 keyframe đã trích, có 1 vector SigLIP + annotation (OCR/caption/objects). Chỉ video/image có frame.
- **segment**: đơn vị nhảy-đến. Video = 1 shot (camera-cut). Audio = 1 đoạn lời. Image = [0,0].
- **scene**: nhóm các shot kề nhau giống nhau về visual (chỉ video).
- **shot**: đoạn liên tục giữa 2 camera-cut (PySceneDetect).
- **dense channel**: tìm bằng cosine giữa q_vec (text) và vector frame (SigLIP).
- **BM25 visual / BM25 asr**: full-text search (FTS5) trên annotation hình / transcript lời.
- **fuse/aggregate** (`rerank.py`): hợp nhất 3 kênh + gom frame→segment. **Khác** cross-encoder rerank (`reranker.py`).

---

## 16. Tôi (AI) cần sửa X — vào đâu?

| Muốn làm | Bắt đầu từ |
|---|---|
| Đổi model encoder / thêm model | `config/settings.yaml` + `app/common/encoder.py` |
| Chỉnh trọng số / ngưỡng tìm kiếm | `config/settings.yaml` (`retrieval.weights`, `MIN_BM25_RAW`/`BM25_SCALE` trong `retrieval.py`+`rerank.py`) |
| Sửa cách hợp nhất điểm 3 kênh | `app/backend/services/rerank.py` (`fuse_and_aggregate`) |
| Sửa logic query expansion | `app/backend/services/search_engine.py` (`expand_query`) |
| Thêm/sửa extractor (OCR/caption/...) | `app/extractors/*.py` + bật/tắt ở `config.extractors` |
| Đổi cách cắt segment/shot/scene video | `app/ingest/video/{shots,keyframes,segments,scenes}.py` |
| Đổi schema DB / cách lưu | `app/ingest/storage.py` (`SCHEMA` + `add_*`) — nhớ đồng bộ FAISS↔SQLite |
| Thêm endpoint API | `app/backend/api/*.py` + `main.py` |
| Sửa UI / hiển thị kết quả | `frontend/src/components/*.jsx`, `api.js` |
| Debug "sao query này không ra kết quả" | `scripts/search_demo.py` + xem `timing_ms`/`score_breakdown` + soi FTS5 token qua `db_inspector.py` |

> **Trước mọi thay đổi hành vi:** chạy `scripts/search_demo.py` trước/sau để so sánh, và **cập nhật file này** nếu đổi luồng/tham số mặc định.
```
