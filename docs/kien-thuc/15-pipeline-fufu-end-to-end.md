# Chương 15 — Pipeline FUFU end-to-end: ráp mọi mảnh lại

---

## 1. Vì sao chương này tồn tại

14 chương vừa rồi, mỗi chương mổ xẻ MỘT mảnh: SigLIP là gì (ch07), PhoWhisper
nghe tiếng Việt thế nào (ch09), FAISS tìm hàng xóm gần nhất ra sao (ch13),
BM25 cộng điểm kiểu gì (ch14)... Nhưng khi bạn mở repo FUFU lên, bạn không thấy
"14 mảnh" — bạn thấy MỘT dòng chảy dữ liệu: file video đi vào, JSON kết quả đi ra.

Chương này là **bản đồ ráp hình**. Không dạy lý thuyết mới — mỗi bước chỉ trỏ
"kiến thức nằm ở chương XX", còn nhiệm vụ chính là trả lời:

- Một video đi qua ingest thì **chuyện gì xảy ra theo thứ tự nào**, model nào
  chạy, kết quả ghi vào bảng nào, mất bao lâu?
- Một query tiếng Việt đi qua backend thì **biến hình qua những dạng nào** trước
  khi thành 20 card kết quả trên UI?
- Khi cần sửa/tune chỗ nào, **mở file nào, đổi config key nào**?

Chương này được viết để đọc KÈM `PROJECT-CONTEXT.md` (đặc biệt §7 ingest và
§8 search) — file đó là "đặc tả", chương này là "chuyến tham quan có hướng dẫn viên".

> 🔗 **Trong FUFU:** toàn bộ chương này bám theo 2 file orchestrator:
> `app/ingest/video/ingest.py` (hàm `ingest_video`) và
> `app/backend/services/search_engine.py` (class `SearchEngine`). Mọi con đường
> đều dẫn về 2 file này.

---

## 2. Cần biết trước

Chương này RÁP TẤT CẢ, nên về nguyên tắc cần **toàn bộ chương 01-14**. Tối thiểu:

| Nhóm | Chương | Dùng ở bước nào trong chương này |
|---|---|---|
| Nền tảng | 01-06 (NN, huấn luyện, CNN, transformer, tokenization, ViT) | nền của mọi model bên dưới |
| Visual embedding | **07** (CLIP/SigLIP) | encode frame lúc ingest + encode query lúc search |
| Trích xuất nội dung | **08** (VLM caption), **09** (ASR), **10** (OCR + detection) | bước annotate frame + tách lời thoại |
| Query expansion | **11** (NLLB + paraphrase) | bước đầu tiên của mọi query |
| Rerank | **12** (bi/cross-encoder) | bước cuối cùng của mọi query |
| Lưu trữ & tìm kiếm | **13** (FAISS), **14** (BM25 + fusion) | nơi mọi thứ được ghi xuống và đọc lên |

Và đọc kèm: `PROJECT-CONTEXT.md` (ít nhất §5 bản đồ repo, §6 mô hình dữ liệu,
§7 ingest, §8 search).

Hai "nhân vật" sẽ đi xuyên suốt chương:

- **Video giả định:** `thoi_su_18h.mp4` — bản tin thời sự 3 phút (180s), có
  banner chữ chạy dưới màn hình, phát thanh viên đọc tin, vài phóng sự hiện trường.
- **Query giả định:** `"người dân xếp hàng mua vàng ở phố Trần Nhân Tông"`.

Mọi con số trong ví dụ (24 shots, ~45 keyframe, timing...) là **con số giả định
hợp lý** để dễ theo dõi — số thật phụ thuộc nội dung video và config của bạn.

---

## PHẦN A — Hành trình một video qua ingest

Lệnh khởi đầu:

```bash
python -m app.ingest.cli /data/news/thoi_su_18h.mp4
```

`run_ingest()` (trong `app/ingest/pipeline.py`) làm 3 việc trước khi đụng tới
video: tạo **một** `SiglipEncoder` + **một** `IndexWriter` dùng chung cho mọi file,
và đăng ký **signal handler** — Ctrl+C/SIGTERM sẽ gọi `writer.persist()` rồi mới
thoát (xem Phần D.3 vì sao điều này quan trọng). Sau đó `detect_media_type()`
thấy `.mp4` → dispatch sang `ingest_video()`.

> 🔗 **Trong FUFU:** `app/ingest/pipeline.py` (hàm `run_ingest`, signal handler),
> `app/ingest/cli.py` (entry point), `app/common/types.py` (`detect_media_type`).

Ngay đầu `ingest_video()`: tạo row trong bảng `items`
(`writer.add_or_get_item`) và check **idempotent** — nếu item đã có frame hoặc
ASR (`item_already_ingested`) thì bỏ qua, nên chạy lại lệnh ingest trên cùng thư
mục là an toàn.

### A.1 — PySceneDetect: cắt video thành shot (~15-20s)

```
thoi_su_18h.mp4 (180s) → detect_shots(threshold=27.0) → 24 shots
   shot 0: [0.0, 12.4]   phát thanh viên mở đầu
   shot 1: [12.4, 15.1]  cắt sang cảnh phố
   shot 2: [15.1, 22.8]  người dân xếp hàng trước tiệm vàng  ← cảnh ta sẽ tìm ở Phần B
   ...
   shot 23: [171.2, 180.0]
```

Đây là bước **không dùng deep learning**: PySceneDetect so sai khác màu giữa
frame liên tiếp, vượt ngưỡng `27.0` thì coi là camera-cut (xử lý ảnh cổ điển,
tinh thần chương 03). Threshold thấp hơn → nhiều shot hơn.

> 🔗 **Trong FUFU:** `app/ingest/video/shots.py` (`detect_shots`), config
> `ingest.video.shot_detect_threshold` trong `config/settings.yaml`.

### A.2 — Keyframes adaptive: ~45 frame đại diện

Mỗi shot lấy `ceil(duration × keyframe_density_per_sec)` frame, clamp về
`[min=1, max=12]` frame/shot. Bản tin có nhiều shot ngắn 2-3s (chỉ lấy 1-3 frame)
nên 24 shots cho ra **~45 keyframe** thay vì 180. Mỗi keyframe nhớ kèm
`(timestamp, ảnh, shot_index)`.

Lưu ý kỹ thuật trong code: **toàn bộ keyframe được decode upfront** (comment
trong `ingest.py`: "load ALL upfront — bottleneck nếu video lớn") — RAM tỉ lệ
với số keyframe, không phải độ dài video.

> 🔗 **Trong FUFU:** `app/ingest/video/keyframes.py`
> (`extract_keyframes_adaptive`), config `ingest.video.keyframe_density_per_sec`,
> `min/max_keyframes_per_shot`.

### A.3 — Segments = shots, rồi gán frame vào segment

Vì `use_shots_as_segments: true`, **mỗi shot trở thành 1 segment** — đơn vị
"nhảy-đến" của hệ thống (Phần D.1 giải thích vì sao). Shot nào dài hơn
`max_segment_len_sec: 15.0` bị chia đều. 24 shots của ta → giả định 26 segments
(2 shot dài bị subdivide). `assign_frames_to_segments` gán từng keyframe vào
segment chứa timestamp của nó, ghi vào bảng `segments` + quan hệ M-N
`frame_segments`.

> 🔗 **Trong FUFU:** `app/ingest/video/segments.py` (`shots_to_segments`,
> `assign_frames_to_segments`), schema trong `app/ingest/storage.py`.

### A.4 — Vòng lặp CHUNKED: 16 frame một, annotate → encode → persist

Đây là trái tim của ingest. 45 frame chia thành 3 chunk (16 + 16 + 13), mỗi chunk
đi qua đủ 5 bước rồi **ghi xuống disk ngay**:

```
chunk 16 frame
 ├─ 1. OCR (EasyOCR, ch10)         → annotation.ocr_text     ~0.3s/frame
 ├─ 2. Caption (Qwen2.5-VL, ch08)  → annotation.caption      ~1.5s/frame ← BOTTLENECK
 ├─ 3. Detection (YOLO-World, ch10)→ annotation.objects      ~0.1s/frame
 ├─ 4. SigLIP encode batch (ch07)  → 16 vector 1024-d, L2-norm  ~1s/chunk
 ├─ 5. Save thumbnail JPEG          → data/thumbnails/thoi_su_18h/s0002_f000007_t16.50.jpg
 └─ writer.add_frames(...) + writer.persist()   ← kill SAU dòng này = chunk an toàn
```

Với frame tại t=16.5s (trong shot 2 — cảnh tiệm vàng), annotation giả định:

- **OCR**: `"GIÁ VÀNG TĂNG KỶ LỤC | PHỐ TRẦN NHÂN TÔNG, HÀ NỘI"` (banner tin tức)
- **Caption** (Qwen-VL, tiếng Việt): `"đông người dân đứng xếp hàng dài trước cửa hàng vàng trên vỉa hè"`
- **Objects** (YOLO-World): `person ×14, storefront, signboard`

Mỗi frame ghi 1 row bảng `frames` (kèm `faiss_id` gán tuần tự = `index.ntotal`
— bất biến FAISS↔SQLite, xem PROJECT-CONTEXT §6), 1 vector vào FAISS
`IndexHNSWFlat` (ch13), và 1 row FTS5 `frame_text` gồm `(ocr_text, caption,
labels)` cho BM25 visual (ch14). Tokenizer FTS5 là `unicode61
remove_diacritics 0` — **giữ dấu tiếng Việt**.

Timing 45 frame: caption ~68s + OCR ~14s + detection ~5s + SigLIP ~3s ≈ **~90s**
cho cả 3 chunk. Tắt `enable_caption` → còn ~20s (nhanh ~5-10×, đổi lại mất
signal semantic, giảm recall ~5-10% trên query mơ hồ — PROJECT-CONTEXT §7.3).

> 🔗 **Trong FUFU:** vòng lặp chunk nằm trọn trong
> `app/ingest/video/ingest.py` (mục `# 4. CHUNKED annotate + encode + persist`).
> Extractor: `app/extractors/{ocr,caption,detection}.py` — lazy singleton qua
> `app/extractors/__init__.py`. Ghi DB: `app/ingest/storage.py`
> (`add_frames`, `persist`). Config: `ingest.video.chunk_size_frames: 16`.

### A.5 — Scene clustering: gom shot kề thành scene (~1s)

Sau khi mọi frame đã có vector, `cluster_shots_into_scenes` so cosine giữa
**frame cuối shot i** và **frame đầu shot i+1**; nếu ≥ 0.85 → cùng scene. Bản tin
hay cắt qua-lại giữa 2 góc máy cùng hiện trường nên 26 segments gom còn giả định
**9 scenes** (shot 1-2-3 cùng cảnh tiệm vàng → 1 scene). Ghi bảng `scenes` +
cập nhật `segments.scene_id`, rồi `persist()` lần nữa.

> 🔗 **Trong FUFU:** `app/ingest/video/scenes.py` (`cluster_shots_into_scenes`,
> threshold 0.85 hard-code tại call-site trong `ingest.py`).

### A.6 — Tách audio + PhoWhisper ASR + gán lời vào shot (~30-40s)

Cuối cùng: ffmpeg tách audio track về PCM mono 16kHz
(`load_audio_mono_16k`), PhoWhisper-medium (ch09) chuyển 180s audio thành các
đoạn lời `(start, end, text)` — giả định **22 đoạn**, ví dụ:

```
[15.3 → 21.8] "ngay từ sáng sớm rất đông người dân đã xếp hàng trước các
               cửa hàng vàng trên phố Trần Nhân Tông để chờ mua vàng"
```

Mỗi đoạn lời được gán vào **shot có overlap thời gian lớn nhất**
(`segment_id` trong bảng `asr_segments` — đoạn trên gán vào shot 2). Text vào
FTS5 `asr_text` cho kênh BM25 ASR. `persist()` lần cuối.

> 🔗 **Trong FUFU:** `app/extractors/asr.py`, `app/common/audio_io.py`,
> `app/ingest/storage.py` (`add_asr_segments` — logic gán overlap lớn nhất).
> Config: `extractors.asr_model: vinai/PhoWhisper-medium`.

### A.7 — Tổng kết hành trình video

| # | Bước | Model | Ghi vào | Thời gian (3-min video, 3090) | Chương |
|---|---|---|---|---|---|
| 1 | Shot detection | PySceneDetect (không-DL) | (RAM) | ~15-20s | 03 |
| 2 | Keyframes ~45 | OpenCV decode | (RAM) | ~5s | — |
| 3 | Segments = shots | — | `segments`, `frame_segments` | <1s | — |
| 4a | OCR | EasyOCR [vi,en] | `frame_text.ocr_text` | ~14s | 10 |
| 4b | Caption | Qwen2.5-VL-7B INT4 | `frames.caption`, `frame_text` | **~68s** | 08 |
| 4c | Detection | YOLO-World v2 | `frames.objects_json`, `frame_text.labels` | ~5s | 10 |
| 4d | Encode | SigLIP-2 Large fp16 | FAISS (`frames.faiss_id`) | ~3s | 07, 13 |
| 4e | Thumbnail + persist ×3 chunk | — | `data/thumbnails/`, disk | ~3s | — |
| 5 | Scene clustering | cosine trên vector đã có | `scenes`, `segments.scene_id` | ~1s | 07 |
| 6 | ASR | PhoWhisper-medium | `asr_segments`, `asr_text` | ~30-40s | 09 |
| | **Tổng** | | | **~2.5-3 phút** (≈ 1× realtime khi caption BẬT) | |

Khớp ước lượng PROJECT-CONTEXT §12: "1 phút video ≈ 3-4 phút wall-time" cho
video dày keyframe; bản tin nhiều shot ngắn nên nhẹ hơn.

---

## PHẦN B — Hành trình một query

Backend đang chạy (`uvicorn app.backend.main:app --port 8080`), người dùng gõ:

```
"người dân xếp hàng mua vàng ở phố Trần Nhân Tông"
```

Frontend POST `/api/search` → `SearchEngine.search()` (singleton `lru_cache`
trong `app/backend/api/search.py`). Đi từng bước theo PROJECT-CONTEXT §8.

### B.1 — `expand_query()`: 1 query → 5 biến thể (ch11)

```
original    : "người dân xếp hàng mua vàng ở phố Trần Nhân Tông"
translated  : "people queuing to buy gold on Tran Nhan Tong street"   (NLLB, beam=2)
paraphrase 1: "đám đông chờ mua vàng trước tiệm vàng phố Trần Nhân Tông" (Qwen2.5-3B, temp 0.7)
paraphrase 2: "dòng người xếp hàng dài tại cửa hàng vàng Hà Nội"
paraphrase 3: "cảnh mua bán vàng đông đúc trên phố Trần Nhân Tông"

"all"  = cả 5            → cho kênh DENSE
"bm25" = original + translated  → cho 2 kênh BM25 (BỎ paraphrase — xem D, và comment trong code)
```

> 🔗 **Trong FUFU:** `search_engine.py` (`expand_query`),
> `app/backend/services/translator.py` (NLLB),
> `app/backend/services/paraphraser.py` (Qwen-3B INT4). Config:
> khối `query_expansion` trong `config/settings.yaml`.

### B.2 — Encode query: 5 biến thể → 1 vector (ch07)

`encoder.encode_text(qe["all"])` → 5 vector SigLIP → **mean → L2-normalize** →
`q_vec` duy nhất. Trung bình hóa làm vector "tròn nghĩa" hơn: phần chung của 5
cách diễn đạt (đám đông + xếp hàng + vàng) được giữ, phần riêng lẻ bị pha loãng.

### B.3 — Ba kênh chạy tuần tự (nhưng độc lập về logic)

```
DENSE (ch13)              BM25 visual (ch14)            BM25 ASR (ch14)
q_vec → FAISS HNSW        FTS5 frame_text                FTS5 asr_text
top-500, cosine           "người OR dân OR xếp OR        (cùng query OR-token)
ef_search=128             hàng OR ... OR gold OR ..."    top-200
                          top-200, token <2 ký tự loại
norm: min-max             norm: raw/8.0, cap 1.0         norm: raw/8.0, cap 1.0
                          filter raw < 3.0               filter raw < 3.0
```

Với query của ta, frame t=16.5s của `thoi_su_18h.mp4` trúng **cả 3 kênh**:

- Dense: caption-ảnh khớp ngữ nghĩa "đám đông xếp hàng" → cosine cao trong top-500.
- BM25 visual: OCR banner chứa nguyên cụm "PHỐ TRẦN NHÂN TÔNG" + caption chứa
  "xếp hàng", "cửa hàng vàng" → raw BM25 cao.
- BM25 ASR: đoạn lời 15.3-21.8s chứa "xếp hàng... cửa hàng vàng... Trần Nhân Tông".

Lưu ý vì sao chuẩn hoá **bất đối xứng** (dense min-max, BM25 chia 8.0): cosine
bị chặn sẵn trong [-1,1] nên min-max ổn; BM25 thì cần giữ "độ mạnh tuyệt đối" —
nếu min-max hóa BM25, một query chỉ có 1 hit rác cũng bị kéo lên 1.0
(PROJECT-CONTEXT §8, chi tiết lý thuyết ở ch14).

> 🔗 **Trong FUFU:** `app/backend/services/retrieval.py` — `faiss_search`,
> `bm25_visual`, `bm25_asr`, `_build_fts_or_query`, hằng `MIN_BM25_RAW=3.0`,
> `BM25_SCALE=8.0`. Config: `retrieval.top_k_dense/top_k_bm25_*`, `hnsw_ef_search`.

### B.4 — `fuse_and_aggregate()`: gom frame → segment, trộn 3 kênh (ch14)

Mọi hit (frame-level hoặc asr-level) được gom theo khóa `(item_id, segment_id)`,
**max-pool** mỗi kênh trong segment (Phần D.4), rồi:

```
score = 0.40·dense + 0.25·bm25_visual + 0.50·bm25_asr      (KHÔNG renormalize)
```

Segment shot-2 của ta (giả định):

```json
"score_breakdown": {"dense": 0.78, "bm25_visual": 0.45, "bm25_asr": 0.81}
score = 0.40×0.78 + 0.25×0.45 + 0.50×0.81 = 0.312 + 0.113 + 0.405 = 0.83
```

So với đối thủ hạng 2 — một video khác chỉ có cảnh tiệm vàng nhưng **không có
lời thoại nhắc địa danh**: `{"dense": 0.82, "bm25_visual": 0.20, "bm25_asr": 0.0}`
→ score = 0.328 + 0.05 + 0 = **0.38**. Match nhiều kênh thắng áp đảo — đúng
intent thiết kế "không renormalize". Mỗi Hit giữ kèm `best_frame` (frame cosine
cao nhất) và `best_asr` (snippet mạnh nhất) để UI hiển thị.

> 🔗 **Trong FUFU:** `app/backend/services/rerank.py` (`fuse_and_aggregate`) —
> **không phải** cross-encoder, xem D.5. Weights: `retrieval.weights` trong
> `config/settings.yaml` — tham số tune chính của cả hệ thống.

### B.5 — BGE cross-encoder rerank top-50 (ch12)

Top-50 hit sau fusion được dựng thành passage text:

```
"đông người dân đứng xếp hàng dài trước cửa hàng vàng trên vỉa hè |
 objects: person, signboard, storefront |
 ngay từ sáng sớm rất đông người dân đã xếp hàng trước các cửa hàng vàng..."
```

BGE-reranker-v2-m3 đọc **(query, passage) cùng lúc** (cross-attention — chính là
lý do nó chính xác hơn nhưng đắt hơn, ch12) → reorder 50 candidate. Hit ngoài
top-50 giữ nguyên thứ tự cũ.

> 🔗 **Trong FUFU:** `app/backend/services/reranker.py` (`BGEReranker`), được gọi
> cuối `search_engine.py:search()`. Config: `retrieval.enable_reranker`,
> `rerank_top_k: 50`.

### B.6 — Top-20 + enrich + JSON

Lấy `top_k=20`, query thêm SQLite để gắn meta `segments` / `items` / `scenes`
(start/end, scene chứa shot, path file) → trả JSON đúng contract PROJECT-CONTEXT
§10. UI React render card, click → nhảy đến `segment_start=15.1s`.

`timing_ms` thực tế điển hình (3090, models đã warm):

| Khóa | ms | Ai tốn |
|---|---|---|
| `expand_ms` | ~1200 | Qwen-3B sinh 3 paraphrase — **chậm nhất pipeline query** |
| `encode_ms` | ~40 | SigLIP encode 5 câu |
| `faiss_ms` | ~5 | HNSW top-500 — gần như miễn phí (ch13) |
| `bm25_visual_ms` | ~15 | FTS5 |
| `bm25_asr_ms` | ~10 | FTS5 |
| `fetch_meta_ms` | ~20 | SQLite enrich |
| `rerank_ms` | ~5 | fuse_and_aggregate (pure Python) |
| `cross_rerank_ms` | ~400 | BGE chấm 50 passage |
| **Tổng** | **~1.7s** | muốn nhanh: tắt paraphrase → ~0.5s |

---

## PHẦN C — Bảng tra: model → chương → file → config

| # | Model (id) | Vai trò | Chương | File code | Config key (`settings.yaml`) |
|---|---|---|---|---|---|
| 1 | SigLIP-2 Large (`google/siglip2-large-patch16-384`) | embed frame + query cùng không gian | 07 (nền 06) | `app/common/encoder.py` | `models.siglip`, `models.device` |
| 2 | EasyOCR `[vi, en]` | chữ trên màn → BM25 visual | 10 | `app/extractors/ocr.py` | `extractors.enable_ocr`, `ocr_min_confidence` |
| 3 | Qwen2.5-VL-7B-Instruct INT4 | caption tiếng Việt per-frame | 08 (nền 04) | `app/extractors/caption.py` | `extractors.enable_caption`, `caption_quant_4bit` |
| 4 | YOLO-World v2 (`yolov8l-world.pt`) | ~70 lớp object open-vocab | 10 (nền 03) | `app/extractors/detection.py` | `extractors.enable_detection` |
| 5 | PhoWhisper-medium (`vinai/PhoWhisper-medium`) | lời thoại VN → BM25 ASR | 09 | `app/extractors/asr.py` | `extractors.enable_asr`, `asr_model` |
| 6 | NLLB-200 distilled 600M | dịch query VI→EN | 11 | `app/backend/services/translator.py` | `models.translator`, `query_expansion.enable_translation` |
| 7 | Qwen2.5-3B-Instruct INT4 | 3 paraphrase cho dense | 11 (nền 05) | `app/backend/services/paraphraser.py` | `models.paraphraser`, `query_expansion.enable_paraphrase`, `num_paraphrases` |
| 8 | BGE-reranker-v2-m3 | cross-encoder rerank top-50 | 12 | `app/backend/services/reranker.py` | `models.reranker`, `retrieval.enable_reranker`, `rerank_top_k` |

Hai "hạ tầng" không phải model nhưng cùng vai vế: **FAISS HNSW** (ch13 —
`app/ingest/storage.py` + `services/retrieval.py`, config `retrieval.hnsw_*`) và
**SQLite FTS5 ×2** (ch14 — cùng 2 file đó, schema trong `storage.py`).

Ingest cần model 1-5 (~13GB VRAM); query online cần 1, 6, 7, 8 (~5GB).

---

## PHẦN D — 5 quyết định thiết kế đáng hiểu sâu

### D.1 Shots-as-segments (thay vì cửa sổ thời gian cứng)

Cách hiển nhiên là cắt video mỗi 10s. FUFU không làm vậy: **mỗi shot
(camera-cut thật) = 1 segment**, shot >15s mới subdivide. Lý do (PROJECT-CONTEXT
§7.3): bài toán Known-Item Search chấm theo việc operator nhảy đến **đúng cảnh**;
segment trùng ranh giới cảnh thật → start/end có nghĩa, thumbnail đại diện đúng,
không bao giờ cắt ngang một cảnh làm đôi. Trade-off: phụ thuộc chất lượng
PySceneDetect — video không có cut rõ (livestream tĩnh) sẽ ra ít shot dài, phải
nhờ subdivide. Config: `ingest.video.use_shots_as_segments`, `max_segment_len_sec`.

### D.2 ASR-chunks-as-segments (audio thuần)

Cùng triết lý cho audio: **mỗi đoạn lời pause-bounded = 1 segment**
(PROJECT-CONTEXT §7.2), chunk cách nhau ≤0.5s được merge chống phân mảnh. "Cảnh"
của audio chính là một lượt nói — operator nhảy đến đúng câu. Hệ quả cần nhớ:
audio không có vector visual → **chỉ tìm được qua kênh BM25 ASR**; audio không có
speech thì hiện không tìm được (gap CLAP, §14). File:
`app/ingest/audio/segments.py`, config `ingest.audio.*`.

### D.3 Chunked persist + signal handler (commit `d4bf91e`)

Ingest 100h video chạy ~24h — chắc chắn sẽ có lúc bị kill (OOM, Ctrl+C, mất
điện ảo). Hai tầng phòng thủ: (1) **persist mỗi 16 frame** — kill ở frame
170/555 chỉ mất ≤16 frame của chunk dở; (2) **SIGINT/SIGTERM handler** trong
`pipeline.py` gọi `writer.persist()` trước khi thoát. Giá phải trả: ~5% overhead
do `faiss.write_index()` mỗi chunk (docstring `video/ingest.py`). Điều kiện sống
còn: `add_frames` ghi FAISS và SQLite **trong cùng một chunk** để bất biến
`faiss_id ↔ frames` không bao giờ lệch (§6). Cộng với idempotent-check ở mức
item → quy trình phục hồi = chạy lại đúng lệnh cũ.

### D.4 Max-pool aggregate (frame → segment)

Một segment có nhiều frame; gom điểm bằng **max** chứ không phải mean
(`fuse_and_aggregate` trong `rerank.py`). Trực giác: query "xếp hàng mua vàng"
chỉ cần MỘT frame trong shot khớp mạnh là cả shot đáng trả về — mean sẽ bị các
frame phụ (mặt phát thanh viên, cảnh chuyển) kéo tụt. Max-pool per-channel còn
cho phép kênh khác nhau "bầu" frame/asr khác nhau trong cùng segment. Nhược điểm
đã biết của max: nhạy với 1 frame nhiễu cosine cao — được bù bằng BGE rerank
phía sau (ch12) đọc lại nội dung thật.

### D.5 Hai nghĩa của "rerank" — bẫy `rerank.py` vs `reranker.py`

Bẫy kinh điển của repo này (PROJECT-CONTEXT §5 đã cảnh báo):

| File | Là gì | Chương | Key trong `timing_ms` |
|---|---|---|---|
| `services/rerank.py` | `fuse_and_aggregate` — **score fusion** 3 kênh + gom frame→segment. Không có model nào. | 14 | `rerank_ms` (~5ms) |
| `services/reranker.py` | `BGEReranker` — **cross-encoder** chấm lại top-50. Có model thật. | 12 | `cross_rerank_ms` (~400ms) |

Khi teammate nói "rerank đang làm sai", câu hỏi đầu tiên LUÔN là: *fusion hay
cross-encoder?* Nhìn `timing_ms` cũng phân biệt được: 5ms là fusion, 400ms là BGE.

---

## Tóm tắt 10 giây

**Ingest:** video → PySceneDetect cắt shot → keyframe adaptive → mỗi 16 frame:
OCR + caption + detection + SigLIP → ghi FAISS + SQLite + FTS5 + persist ngay →
scene clustering → PhoWhisper gán lời vào shot. **Query:** expand (NLLB +
paraphrase) → encode mean → 3 kênh dense/BM25v/BM25a song song → fuse có trọng
số 0.40/0.25/0.50 + max-pool theo segment → BGE rerank top-50 → top-20 JSON.
Mọi tham số ở `config/settings.yaml`; mọi đường đi qua `video/ingest.py` và
`search_engine.py`.

---

## Câu hỏi tự kiểm tra

**1. Kill ingest bằng Ctrl+C khi đang ở frame 40/45 của `thoi_su_18h.mp4`. Mất gì, và chạy lại thế nào?**

<details><summary>Đáp án</summary>

Mất tối đa chunk đang dở (≤16 frame, ở đây là chunk 3 gồm 13 frame nếu chưa
persist) — 32 frame của chunk 1-2 đã an toàn trên disk nhờ chunked persist;
signal handler còn gọi `persist()` lần cuối trước khi thoát. NHƯNG: item này
đã có frame → check `item_already_ingested` sẽ **bỏ qua khi chạy lại** — file
sẽ thiếu chunk cuối + scenes + ASR. Muốn ingest lại trọn vẹn phải xóa item khỏi
DB (hoặc xóa `data/` nếu mới test). Đây là giới hạn đáng biết của idempotent
mức-item.
</details>

**2. Query "người dân xếp hàng mua vàng" — vì sao paraphrase được dùng cho dense nhưng KHÔNG dùng cho BM25?**

<details><summary>Đáp án</summary>

Dense (ch07) so sánh ngữ nghĩa: thêm biến thể rồi mean vector làm query "tròn
nghĩa" hơn, không hại. BM25 (ch14) match token: paraphrase dài sinh thêm token
("đám đông", "dòng người", "Hà Nội"...) OR vào FTS5 → match rác với OCR/ASR
ngắn, gây nhiễu phrase. Nên `expand_query` trả 2 list riêng: `"all"` cho dense,
`"bm25"` chỉ gồm original + translated (comment ngay trong
`search_engine.py:expand_query`).
</details>

**3. Một audio podcast 10 phút được ingest. Nó xuất hiện ở những kênh nào lúc search? Vì sao?**

<details><summary>Đáp án</summary>

Chỉ kênh **BM25 ASR**. Audio không có keyframe → không có vector SigLIP trong
FAISS (không vào dense) và không có row `frame_text` (không vào BM25 visual).
Weights `bm25_asr: 0.5` được tune cao một phần chính để item chỉ-có-ASR vẫn
cạnh tranh được với video trúng dense (PROJECT-CONTEXT §8). Nếu audio không có
speech → hoàn toàn không tìm được (gap CLAP, §14).
</details>

**4. `score_breakdown = {"dense": 0.85, "bm25_visual": 0.0, "bm25_asr": 0.0}` và `{"dense": 0.55, "bm25_visual": 0.40, "bm25_asr": 0.62}` — cái nào xếp trên sau fusion? Tính cụ thể.**

<details><summary>Đáp án</summary>

Hit 1: 0.40×0.85 = **0.34**. Hit 2: 0.40×0.55 + 0.25×0.40 + 0.50×0.62 = 0.22 +
0.10 + 0.31 = **0.63**. Hit 2 thắng gần gấp đôi dù dense thấp hơn — vì final
score KHÔNG renormalize, match đa kênh được thưởng. (BGE rerank sau đó vẫn có
thể đảo lại nếu nội dung text của hit 1 thật sự khớp query hơn.)
</details>

**5. `timing_ms` cho thấy `rerank_ms: 4.8` và `cross_rerank_ms: 412`. Hai số này đến từ 2 file nào, và nếu muốn giảm 412ms thì chỉnh gì?**

<details><summary>Đáp án</summary>

`rerank_ms` = `fuse_and_aggregate` trong `services/rerank.py` (score fusion,
pure Python). `cross_rerank_ms` = `BGEReranker.rerank` trong
`services/reranker.py` (cross-encoder, ch12). Giảm 412ms: hạ
`retrieval.rerank_top_k` (50 → 20, ít passage phải chấm) hoặc tắt hẳn
`enable_reranker: false` (đổi precision lấy tốc độ).
</details>

**6. Vì sao 1 frame có thể thuộc NHIỀU segment (bảng `frame_segments` là M-N)?**

<details><summary>Đáp án</summary>

Shot dài >15s bị subdivide thành nhiều segment; cộng với fallback sliding window
(stride < length) tạo segment chồng lấn — một timestamp có thể nằm trong ≥2
segment. `assign_frames_to_segments` vì thế trả list segment cho mỗi frame, và
schema cần bảng nối M-N thay vì cột `segment_id` đơn trên `frames`
(PROJECT-CONTEXT §6).
</details>

**7. Đội quyết định tắt caption để ingest nhanh gấp ~10×. Liệt kê CHÍNH XÁC những gì hệ thống mất ở phía search.**

<details><summary>Đáp án</summary>

(1) Cột `caption` trong FTS5 `frame_text` trống → BM25 visual chỉ còn dựa OCR +
labels detection; (2) passage cho BGE rerank mất phần caption → cross-encoder ít
ngữ liệu hơn; (3) UI mất dòng mô tả trên card. Dense KHÔNG bị ảnh hưởng (SigLIP
encode pixel trực tiếp, không qua caption). Tổng thiệt hại ước tính: recall
giảm ~5-10% trên query mơ hồ/semantic (PROJECT-CONTEXT §7.3).
</details>

**8. Một frame có cosine rất cao với query nhưng segment của nó vẫn xếp dưới một segment có cosine thấp hơn. Nêu 2 cơ chế trong pipeline có thể gây ra điều này.**

<details><summary>Đáp án</summary>

(1) **Fusion đa kênh**: segment kia trúng thêm BM25 visual/ASR — với weight
0.25/0.50, vài điểm BM25 bù được chênh lệch dense (xem câu 4). (2) **BGE
cross-encoder**: trong top-50, thứ hạng cuối do reranker quyết — nếu
caption/ASR của segment kia khớp query hơn về nội dung text, nó được đẩy lên
bất chấp cosine. (Cơ chế phụ: dense được min-max normalize trong batch nên
"cosine cao" tuyệt đối chưa chắc cao sau chuẩn hoá.)
</details>

---

## PHẦN E — BÀI TẬP THỰC HÀNH

Mục tiêu: tự tay đẩy 1 video qua toàn bộ pipeline và **đọc hiểu từng con số**
hệ thống nhả ra. Cần GPU CUDA; nếu chưa tải model: `python scripts/download_models.py`.

### Bước 1 — Ingest 1 video mẫu

```bash
# lấy 1 video bất kỳ ~1-3 phút (bản tin càng tốt: có chữ + có lời thoại)
python -m app.ingest.cli /path/to/video_mau.mp4
```

Quan sát log và đối chiếu Phần A: số shots? số keyframes? bao nhiêu chunk
(`chunk 1/3...`)? `OCR=?/45 Caption=?/45`? bao nhiêu scenes gom từ bao nhiêu
shots? bao nhiêu đoạn lời ASR? **Ghi lại các số này.**

*Thử nghiệm phụ:* nhấn Ctrl+C giữa chừng rồi xem log signal handler; kiểm tra
`data/index.faiss` và `data/meta.sqlite` vẫn tồn tại và nhất quán.

### Bước 2 — Soi DB bằng `scripts/db_inspector.py`

```bash
python scripts/db_inspector.py
```

Tự trả lời bằng dữ liệu thật: (a) bảng `segments` có đúng số shot bạn đếm ở
bước 1? (b) mở vài row `frames` — caption tiếng Việt có hợp lý không, `faiss_id`
có tuần tự không? (c) row `asr_segments` nào có `segment_id` — nghĩa là đoạn lời
đã được gán vào shot nào?

### Bước 3 — Chạy 1 query không cần frontend

```bash
# nghĩ 1 query mô tả MỘT cảnh cụ thể trong video của bạn
python scripts/search_demo.py "mô tả cảnh trong video của bạn"

# hoặc qua API:
uvicorn app.backend.main:app --port 8080
curl -s -X POST http://localhost:8080/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "mô tả cảnh trong video của bạn", "top_k": 5}'
```

Đọc output theo Phần B: `translated` dịch có đúng không? `expanded_queries` có
mấy biến thể? `num_dense` / `num_bm25_visual` / `num_bm25_asr` mỗi kênh ra bao
nhiêu hit? `timing_ms` — bước nào chậm nhất, có khớp bảng B.6 không?

### Bước 4 — Giải thích vì sao #1 thắng #2

Lấy `score_breakdown` của kết quả #1 và #2, **tự tính lại bằng tay**:

```
score = 0.40·dense + 0.25·bm25_visual + 0.50·bm25_asr
```

Viết 2-3 câu trả lời: #1 thắng nhờ kênh nào? Nếu hai score sát nhau, thứ tự có
thể do BGE rerank đảo — kiểm tra bằng cách so passage (caption + objects +
best_asr) của 2 kết quả với query của bạn.

### Bước 5 — Đổi 1 weight, restart, so sánh

```yaml
# config/settings.yaml — ví dụ: tăng mạnh kênh ASR
retrieval:
  weights:
    dense: 0.4
    bm25_visual: 0.25
    bm25_asr: 0.9     # thử 0.5 → 0.9
```

Restart backend (bắt buộc — `get_config` có `lru_cache`, PROJECT-CONTEXT §9),
chạy lại đúng query ở bước 3. So sánh: top-5 đổi thứ tự thế nào? `score` của
các item có lời thoại tăng bao nhiêu? Trả weight về cũ khi xong.

*Nâng cao:* lặp bước 5 với `enable_reranker: false` để cô lập ảnh hưởng của
cross-encoder; hoặc `enable_paraphrase: false` để xem `expand_ms` giảm còn bao
nhiêu. Bạn vừa làm thủ công những gì chương 17 (tuning) và 19 (eval) sẽ làm
một cách có hệ thống.

---

*Chương tiếp theo: 16 — LoRA: fine-tune model lớn với ngân sách nhỏ.*
