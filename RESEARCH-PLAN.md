# RESEARCH-PLAN — FUFU cho HCM AI Challenge 2026

> **Mục đích:** tổng hợp research về các cuộc thi liên quan (VBS, HCM AI Challenge các mùa) +
> kỹ thuật của các đội top, đối chiếu với FUFU v2 hiện tại ([PROJECT-CONTEXT.md](PROJECT-CONTEXT.md)),
> và đưa ra **menu ý tưởng** xếp hạng theo impact/effort để chọn lựa thực thi.
>
> Cập nhật: 2026-06-12. Trạng thái: **chờ chọn ý tưởng** (đánh dấu ✅ vào cột "Chọn" rồi triển khai theo §6).
>
> 🆕 **2026-06-26:** menu A1–F2 dưới đây = **sao chép kỹ thuật đội thắng** (để ngang mặt sàn). Phần **kỹ thuật SÁNG TẠO để VƯỢT lên** (querybank-norm, domain-adapt encoder, visual-HyDE, chính sách submit hiệu chỉnh…) chuyển sang [KE-HOACH-SANG-TAO-2026.md](KE-HOACH-SANG-TAO-2026.md). Đọc file đó cho hướng "đốt ingest → thắng query".

---

## 1. Bức tranh cuộc thi (research mới, 2024→2026)

### 1.1 Video Browser Showdown (VBS) — chuẩn quốc tế mà HCM AIC mô phỏng

| Năm | Top 3 | Ghi chú |
|---|---|---|
| **VBS 2026** (Prague, 17 đội) | 🥇 **PraK V4** · 🥈 **NII-UIT** · 🥉 Exquisitor | |
| **VBS 2025** (Nara) | 🥇 **NII-UIT** (Best Expert) · 🥉 VEAGLE (Best Novice) | NII-UIT = đội gốc VN (UIT + NII Tokyo); VEAGLE dùng eye-gaze |
| **VBS 2024** | 🥇 **VISIONE 5.0** (ISTI-CNR) · 🥈 Vibro · 🥉 diveXplore | VISIONE = ensemble nhiều embedding (xem §1.3) |
| 2022-2023 | 🥇 Vibro | CLIP + browsing UI cực nhanh |

**Kỹ thuật của các hệ top (đã verify qua paper):**

- **NII-UIT** (vô địch 2025, á quân 2026):
  - **LLM query expansion** — dùng LLM viết lại/mở rộng query thay vì paraphrase mù.
  - **Dynamic temporal search** — chấm điểm frame relevance cả **trước và sau** kết quả trước đó (truy vấn dạng "A rồi đến B").
  - **Visual query bằng Stable Diffusion** — sinh ảnh từ mô tả → query-by-image.
  - 2026 thêm bộ **VQA**: answer span prediction + candidate answer suggestion + **in-video retrieval** (tìm trong 1 video).
- **PraK V4** (vô địch 2026):
  - **Localized queries** — text/texture gắn vào **vị trí không gian trong frame** ("người áo đỏ ở góc trái"), kết hợp **spatial conjunction** (AND giữa nhiều vùng).
  - **Within-video browsing/querying** — player riêng + search trong 1 video.
  - **Backend song song hoá, latency thấp** + keyframe layout tối ưu cho mắt operator.
  - Nền tảng: CLIP + **temporal fusion** cho cặp query liên tiếp.
- **VISIONE** (CNR-ISTI, top ổn định nhiều năm): **ensemble nhiều embedding** (OpenCLIP ViT-L/14 + CLIP2Video + ALADIN) hợp nhất qua dot-product chung; object detection; query-by-image nội bộ.
- **Exquisitor** (🥉 2026): **interactive learning / relevance feedback** — hệ học từ click của operator qua từng vòng.
- **SnapSeek 2.0**: human feedback + sketch "magic brush".

**Meta-bài học từ VBS** (lặp lại mọi năm): chênh lệch model giữa các đội top là nhỏ;
thắng thua nằm ở **(1) tốc độ end-to-end, (2) truy vấn thời gian (A→B), (3) UI browsing
hiệu quả, (4) nhiều "cửa" truy vấn bổ trợ nhau** (text/image/OCR/ASR/sketch).

### 1.2 HCM AI Challenge (mùa 2023→2025)

- **Format 2024** (paper tổng kết chính thức): 1.471 video tin tức / 328 giờ; nhiệm vụ **Textual KIS, Visual KIS (clip mẫu), Q&A** — trả về đúng video+timestamp; chấm theo thời gian + số lần submit sai.
- **Format 2025**: thêm **TRAKE** (Temporal Retrieval and Alignment of Key Events) — cho 1 chuỗi mô tả nhiều "moment" trong **một** video, phải trả về **đúng frame cho từng moment**. Đây là dạng "multi-event temporal" mà các đội yếu temporal sẽ chết.
- **2026 Sơ Tuyển (Official PDF)**: Có 3 dạng truy vấn: **Textual KIS, Q&A, và TRAKE**. Metric đánh giá KHÔNG phải thời gian/phạt nộp sai, mà là **Trung bình của Top-k R-Score (R@1, R@5, R@20, R@50, R@100)** — cho phép nộp tới 100 đáp án/câu hỏi. Hỗ trợ batch 1: Video, Keyframes, Objects (Faster R-CNN), CLIP (ViT-B-32), Metadata.

**Kỹ thuật các đội top VN (qua paper SOICT/arXiv):**

| Đội / Paper | Kỹ thuật đáng chú ý |
|---|---|
| **RAPID** (giải Ba 2024, SOICT'24) | BLIP-2 Q-Former; **envit5 translation bridging** VI→EN; YOLO-Worldv2 + PP-OCR; parallel inference drafting |
| **AIO_Owlgorithms 2025** (arXiv 2512.13169) | TransNetV2 shot detect; **BEiT-3** embeddings + Milvus; **Gemini OCR** + Elasticsearch (vietnamese analyzer); **QUEST** = LLM query rewriting + **Google Image Search fallback** cho entity ngoài tri thức model (người nổi tiếng, địa danh VN); **DANTE** = dynamic programming align chuỗi moment cho TRAKE |
| **MADTempo / AIO_Trinh 2025** (arXiv 2512.12929) | Multi-event temporal: **cộng dồn similarity qua các segment liên tiếp**; Google-Image-Search query augmentation |
| **Moment Retrieval nhẹ** (arXiv 2504.09298) | **SuperGlobal reranking** (rerank không cần model nặng) + **Adaptive Bidirectional Temporal Search** |
| Các đội SOICT'24 khác | CLIP + **BEiT-3** hybrid; GPT-4 multimodal query expansion + open image search; reweighting đa kênh (ReViMM) |

**Pattern chung của các đội VN top:** ensemble 2-3 embedding (CLIP-family + BEiT-3) ·
Elasticsearch cho OCR/ASR tiếng Việt · LLM rewrite query · translation bridging ·
temporal scoring qua segment liên tiếp · external image search cho entity VN.

### 1.3 Insight về model embedding (benchmark 2026)

- Frame-averaged image encoder (kiểu SigLIP) **mất thông tin chuyển động** — model có cross-frame attention (X-CLIP/CLIP2Video) tốt hơn rõ trên query hành động.
- Tuy nhiên với format KIS-trên-keyframe (VBS/AIC), **image-text encoder mạnh vẫn là trụ chính** — các đội top đều dùng CLIP-family per-frame, bù temporal bằng **thuật toán scoring** (không phải bằng video encoder).
- **Ensemble nhiều encoder > một encoder tốt nhất** (VISIONE đã chứng minh nhiều năm).

### 1.4 Kiến trúc tham chiếu mã nguồn mở & "menu cửa truy vấn" (từ vitrivr / lifeXplore / LSC)

> Bổ sung 2026-06-15 sau khi đọc: vitrivr (ACM MM 2016), lifeXplore (LSC'20 & LSC'24),
> MEMORIA (LSC'23), kết quả VBS 2025 (arXiv 2509.12000) + roster VBS teams.

**vitrivr — stack tham chiếu mở của giới nghiên cứu** (Rossetto et al., ACM MM 2016; vẫn dự thi VBS 2025/2026 dưới tên *vitrivr-engine*):

- 3 tầng tách bạch: **Cineast** (engine trích đặc trưng + truy vấn) → **Cottontail DB / ADAMpro** (CSDL vector + boolean) → **vitrivr-ng** (UI web). Đây đúng mô hình "extractor / index / UI tách rời" mà FUFU đang theo → đáng đọc để học cách họ tổ chức multi-feature và fuse nhiều kênh trong 1 query.
- Hỗ trợ **nhiều "cửa" truy vấn cùng lúc rồi fuse**: query-by-example, **query-by-sketch** (sketch màu + sketch cạnh), **query-by-motion** (vẽ đường chuyển động), text/concept, audio, boolean/metadata filter, temporal. → minh chứng kinh điển cho luận điểm "nhiều cửa bổ trợ" ở §1.1.

**Catalog modality truy vấn / duyệt mà FUFU CHƯA có** (gộp từ vitrivr, lifeXplore, MEMORIA, roster VBS):

| Modality (từ khóa mới) | Là gì | Đội dùng |
|---|---|---|
| **Query-by-sketch** (color/edge) + "magic brush" | vẽ phác bố cục/màu → match frame | vitrivr, SnapSeek, lifeXplore |
| **Query-by-motion** (motion sketch) | vẽ quỹ đạo chuyển động của đối tượng trong frame | vitrivr |
| **SOM / feature-map / grid browsing** | xếp keyframe tương tự thành lưới 2D (Self-Organizing Map) để mắt quét nhanh thay vì cuộn list | lifeXplore, diveXplore |
| **Concept search + filter** | lọc theo bộ concept định sẵn (vd YOLO9000) + metadata | lifeXplore, MEMORIA |
| **Uniform sampling** vs shot-detect | lấy keyframe đều theo thời gian thay vì theo shot (đối nghịch TransNetV2 §1.2) | lifeXplore |
| **Eye-gaze-assisted** | dùng hướng nhìn operator để dẫn hướng kết quả | VEAGLE (Best Novice VBS2025) |
| **Emotion-based retrieval** | truy theo cảm xúc của cảnh | vitrivr-engine (VBS2026) |
| **VR / mixed-reality / panoramic** | duyệt trong không gian 3D/VR | vitrivr-VR, MediaMix, TapesVRy |
| **Autonomous agent loop** | agent tự gọi search/verify lặp đến khi tự tin | "Autonomous Agents" (Ho-Le, VBS2026) — khớp **Automated track** AIC2026 |

**LSC (Lifelog Search Challenge)** — cuộc thi "anh em" của VBS, cùng tổ chức tại ICMR, cùng ban (Schoeffmann, Gurrin). Domain là ảnh đời sống cá nhân nhưng **kỹ thuật chuyển thẳng sang KIS video**: concept search, OCR, lọc metadata/thời gian, **moment retrieval**. lifeXplore & MEMORIA là 2 hệ tiêu biểu — đáng tham khảo phần UI duyệt nhanh (SOM) và moment retrieval.

**Roster VBS 2025/2026 — hệ/đội mới đáng để mắt** (ngoài top đã nêu §1.1):

- **VBS2024 winner = VISIONE 5.0** (ISTI-CNR) — bổ sung mốc còn thiếu trong bảng §1.1; 🥈 Vibro (best textual KIS), 🥉 diveXplore (best QA/novice).
- **HORUS / VIREO / ViewsInsight 2.0** — đều dùng **MLLM (multimodal LLM)** để caption/sinh query tự động → cùng hướng B1.
- **EAGLE → H-EAGLE** — retrieval ngữ nghĩa **đa tầng/phân cấp** (hierarchical multi-level semantic).
- **Fusionista (2.0)** — "3D information fusion" cho truy vấn hiệu quả.

#### Đọc sâu — lifeXplore 2024 & MEMORIA (file local trong [docs/nguon-tham-khao/](nguon-tham-khao/README.md) — đã đọc full-text 2026-06-15)

**lifeXplore 2024** (vô địch LSC2023; FAISS + OpenCLIP) — nhiều chi tiết triển khai **dùng ngay được**:

- **Temporal search bằng dấu `<`**: query nhiều "phần" cách nhau bởi `<` (vd `man with red shirt -o person < plane flight < -h 120+`). Thuật toán xử lý **ngược từ phần CUỐI**: lấy kết quả phần cuối → với mỗi kết quả, kiểm tra phần trước có kết quả **cùng ngày & thời gian sớm hơn** không; không có thì loại. → mẫu cụ thể cho **A1/A4**, khác cách "cộng dồn similarity" của MADTempo.
- **Kết hợp FAISS + MongoDB** (dense + filter có cấu trúc): lấy top-K lớn từ FAISS (vd 5000) → dùng list image-id làm filter đầu vào cho truy vấn MongoDB (object/concept/text/metadata). → mẫu cụ thể cho **E7** và hybrid.
- **Position sub-filter**: tiền-tính vị trí bounding box (`top-left`/`bottom-right`) lúc ingest → cho phép `-o person|position:top-left`. Bản nhẹ của **localized query** kiểu PraK V4.
- **Query-building UI dạng block**: mỗi block = 1 phần temporal, kéo ngang để nối; trong block thêm nhiều filter/sub-filter → tham khảo cho **E1**.
- **Eval chọn model**: so OpenCLIP ViT-B/32 → bigG/14 trên query LSC bằng **Recall@K theo từng hint tăng dần** (KIS thêm hint mỗi 30s); **ViT-L/14 (LAION2B) thắng** → cách làm **F1** + lưu ý chọn checkpoint.
- Extractor stack: EfficientNet-B2 (Places365 concept), YOLOv7 (COCO object), CRAFT (text detect+OCR), OpenCLIP ViT-L/14. Gộp query-server + index-server vào 1 process Python để bỏ overhead IPC (bài học latency **E2**).

**MEMORIA LSC2023:**

- Đổi **CSDL quan hệ → graph database (Neo4j)** + full-text search; tác giả đo Neo4j > PostgreSQL cho multimedia retrieval (ý tưởng index thay thế — tham khảo).
- **Event segmentation phân cấp** theo semantic location / activity / transport mode (clustering GPS) — chia chuỗi thành "sự kiện"; tư duy "segment theo ngữ nghĩa" có thể vận vào nhóm keyframe.
- Annotation: CLIP + **ClipCap** (sinh caption) + YOLO concept — cùng hướng caption/**D1**.

---

## 2. Đối chiếu: FUFU v2 đang có gì / thiếu gì

| Năng lực | FUFU v2 | Đội top làm |
|---|---|---|
| Dense visual | ✅ SigLIP-2 Large (1 encoder) | Ensemble 2-3 encoder (VISIONE, đội VN dùng thêm BEiT-3) |
| OCR / ASR / Caption / Detection | ✅ đủ 4 | ✅ tương đương (họ dùng Gemini OCR/PP-OCR — mạnh hơn EasyOCR) |
| BM25 text | ✅ FTS5 ×2 | Elasticsearch + vietnamese analyzer (word-segmentation, ASCII-folding) |
| Query expansion | ✅ NLLB dịch + Qwen-3B paraphrase | **LLM rewrite có cấu trúc** (tách entity/OCR/ASR keywords) — hơn paraphrase mù |
| Rerank | ✅ BGE cross-encoder (text-only) | + VLM rerank ảnh thật, SuperGlobal |
| **Temporal query (A→B)** | ❌ **KHÔNG CÓ** | NII-UIT, PraK, MADTempo, DANTE — **vũ khí chính 2025+** |
| **TRAKE alignment** | ❌ | DANTE (DP alignment) |
| Query-by-image / external image | ❌ | QUEST/MADTempo (Google Images), NII-UIT (SDXL) |
| Relevance feedback | ❌ | Exquisitor, SnapSeek |
| Within-video search | ❌ | PraK V4, NII-UIT 2026 |
| UI thi đấu (submit, player, keyboard) | ⚠ UI demo cơ bản | Tối ưu từng giây (TycheVid, PraK) |
| **Eval harness nội bộ** | ⚠ chỉ MSR-VTT dịch | Đội mạnh đo recall@k mỗi thay đổi trên query giống đề thi |
| Q&A / VQA module | ❌ | NII-UIT 2026 (nếu 2026 có task QA — 2024 đã có) |

**Kết luận chiến lược:** phần "nhận diện nội dung" (extractors) của FUFU đã ngang
mặt bằng. Khoảng cách lớn nhất nằm ở **(1) temporal query, (2) query understanding
thông minh hơn, (3) tooling thi đấu (UI + tốc độ + eval), (4) ensemble**.

---

## 3. MENU Ý TƯỞNG (chọn ở đây)

> Impact đánh giá theo: bằng chứng từ đội thắng × độ khớp format 2026 × mức FUFU đang thiếu.
> Effort: S = 1-2 ngày · M = 3-7 ngày · L = 1-3 tuần.

### Nhóm A — Temporal (khoảng trống lớn nhất, killer cho KIS chuỗi + TRAKE)

| # | Ý tưởng | Bằng chứng | Impact | Effort | Chọn |
|---|---|---|---|---|---|
| **A1** | **Temporal pair query** "A rồi B": 2 ô query, score(seg_i, A) + max score(seg_{i+1..i+k}, B), cộng dồn → rank cặp | PraK temporal fusion, NII-UIT dynamic temporal search, MADTempo | 🔥🔥🔥 | M | ☐ |
| **A2** | **DANTE-style DP alignment** cho TRAKE: cho N mô tả moment + 1 video → DP tìm chuỗi frame tăng dần theo thời gian có tổng similarity max | AIO_Owlgorithms 2025 | 🔥🔥🔥 (TRAKE đã CHỐT sơ tuyển 2026) | M | ☐ |
| **A3** | **Within-video search mode**: chọn 1 video → search/browse chỉ trong video đó (timeline strip + query) | PraK V4, NII-UIT 2026 | 🔥🔥 | M | ☐ |
| A4 | Bidirectional adaptive temporal search (mở rộng A1 hai chiều, độ rộng cửa sổ thích nghi) | arXiv 2504.09298 | 🔥 | S (sau A1) | ☐ |

### Nhóm B — Query understanding (nâng cấp não query)

| # | Ý tưởng | Bằng chứng | Impact | Effort | Chọn |
|---|---|---|---|---|---|
| **B1** | **LLM structured query rewriting** thay paraphrase mù: 1 lần gọi LLM tách query thành {scene_en cho SigLIP, ocr_keywords, asr_keywords, objects, entity_names, temporal_parts[]} → đánh đúng kênh | QUEST, GPT-4 expansion (SOICT'24), NII-UIT | 🔥🔥🔥 | M | ☐ |
| **B2** | **External image search fallback** cho entity ngoài tri thức (người nổi tiếng VN, địa danh): query → Google/Bing Images top-5 → SigLIP encode → query-by-image vào FAISS | QUEST + MADTempo (cả 2 đội 2025 đều làm) | 🔥🔥🔥 cho video tin tức | M | ☐ |
| **B3** | **Query-by-image upload** (operator dán ảnh/URL) — nền tảng cho B2, Visual KIS dùng trực tiếp | VISIONE, PraK, mọi hệ top | 🔥🔥 | S | ☐ |
| B4 | Sinh ảnh từ query bằng SDXL-turbo → query-by-image (khi không có ảnh thật) | NII-UIT 2025 | 🔥 | M | ☐ |
| B5 | **Query-by-sketch**: operator vẽ phác bố cục/màu → encode (SigLIP image hoặc kênh color/edge) → query-by-image vào FAISS | vitrivr (QBS chuẩn), SnapSeek "magic brush", lifeXplore | 🔥 | M | ☐ |

### Nhóm C — Retrieval core (chất lượng ranking)

| # | Ý tưởng | Bằng chứng | Impact | Effort | Chọn |
|---|---|---|---|---|---|
| **C1** | **Ensemble encoder thứ 2 (BEiT-3 hoặc OpenCLIP/EVA-CLIP)** song song SigLIP-2, fuse điểm 2 kênh dense | VISIONE (nhiều năm), đội VN 2024-25 | 🔥🔥 | M | ☐ |
| **C2** | **VLM rerank top-20** — Qwen-VL chấm "frame này khớp query?" trên ảnh thật (thay vì BGE chỉ đọc text) | RAPID, chuẩn 2-tầng VBS | 🔥🔥 | M | ☐ |
| C3 | SuperGlobal reranking (rerank bằng neighborhood expansion trên embedding, không cần model) | arXiv 2504.09298 | 🔥 | S | ☐ |
| C4 | Video-native encoder (X-CLIP/CLIP2Video) làm kênh thứ 3 cho query hành động | benchmark 2026, VISIONE dùng CLIP2Video | 🔥 | L | ☐ |
| C5 | Tune trọng số hybrid bằng eval harness (hiện asr=0.5 là giá trị chưa kiểm chứng) | — | 🔥 | S (cần F1) | ☐ |

### Nhóm D — Index enrichment (chất lượng dữ liệu)

| # | Ý tưởng | Bằng chứng | Impact | Effort | Chọn |
|---|---|---|---|---|---|
| **D1** | **Synthetic query augmentation** (doc2query): Qwen-VL sinh 10-15 query/frame lúc ingest → embed + FTS | BAO-CAO §9.3; doc2query/HyDE trong IR | 🔥🔥 | M | ☐ |
| D2 | Nâng OCR: VietOCR/PaddleOCR-VL hoặc Gemini-OCR API cho keyframe có text (EasyOCR yếu chữ Việt cách điệu) | AIO_Owlgorithms dùng Gemini OCR | 🔥🔥 | S-M | ☐ |
| D3 | FTS tiếng Việt xịn hơn: dual-index có dấu + không dấu, hoặc chuyển Elasticsearch + vietnamese analyzer | mọi đội VN dùng ES | 🔥 | M | ☐ |
| D4 | Dedup keyframe gần trùng (CNN/pHash) → index gọn, browse đỡ rác | đội CLIP+BEiT-3 SOICT'24 | 🔥 | S | ☐ |

### Nhóm E — Interaction & thi đấu (Traditional track — nơi thắng thua thật sự)

| # | Ý tưởng | Bằng chứng | Impact | Effort | Chọn |
|---|---|---|---|---|---|
| **E1** | **UI thi đấu**: video player nhảy đúng segment, phím tắt, temporal context strip (frame ±lân cận), nút submit (nối API BTC khi có), lịch sử query | TycheVid (tốc độ = vũ khí), PraK keyframe layout | 🔥🔥🔥 | M-L | ☐ |
| **E2** | **Latency tuning**: profile end-to-end, song song hoá 3 kênh (hiện chạy tuần tự), cache query expansion, mục tiêu <500ms | TycheVid, PraK V4 parallelized backend | 🔥🔥 | S-M | ☐ |
| E3 | Relevance feedback (✓/✗ → dịch query vector, Rocchio hoặc adapter δ) | Exquisitor 🥉 VBS2026, SnapSeek; BAO-CAO §9.2 | 🔥🔥 | M | ☐ |
| E4 | QA-assist: VLM đọc top frame + ASR → đề xuất đáp án cho task Q&A | NII-UIT 2026 VQA | 🔥🔥 (Q&A đã CHỐT sơ tuyển 2026) | M | ☐ |
| E5 | Agent mode cho **Automated track**: LLM orchestrator gọi search/verify/temporal-check tools tự loop đến khi confident | BAO-CAO GP#3; **"Autonomous Agents" VBS2026** (Ho-Le); xu hướng 2026 "trợ lý ảo" | 🔥🔥 | L | ☐ |
| **E6** | **SOM / grid browsing**: xếp top-K + lân cận embedding thành lưới 2D self-organizing để operator quét bằng mắt thay vì cuộn list dọc | lifeXplore, diveXplore (UI duyệt = vũ khí LSC/VBS) | 🔥🔥 | M | ☐ |
| E7 | **Concept/metadata filter panel**: lọc nhanh song song search theo object/OCR/thời lượng/khung giờ | vitrivr, lifeXplore, MEMORIA | 🔥 | S | ☐ |

### Nhóm F — Nền tảng (làm trước mọi thứ khác)

| # | Ý tưởng | Bằng chứng | Impact | Effort | Chọn |
|---|---|---|---|---|---|
| **F1** | **Eval harness theo format thi**: tự tạo ~50-100 query KIS/TRAKE tiếng Việt trên dataset mẫu (lấy đề các mùa cũ làm template), đo recall@1/5/20 + MRR tự động cho MỌI thay đổi | cách mọi đội nghiêm túc làm; không có nó thì mọi tuning là mò | 🔥🔥🔥 | M | ☐ |
| F2 | Ingest scale-test: đo throughput trên 10-20h video thật, quyết định bật/tắt caption, multi-process | HARDWARE.md ước tính nhưng chưa đo thật | 🔥 | S | ☐ |

---

## 4. Gợi ý combo (nếu cần default để chọn nhanh)

- **Combo "Thi đấu thực dụng"** (≈4-5 tuần): **F1 → A1 + B1 + B3 → E1 + E2 → C5** — lấp đúng 3 khoảng trống lớn (temporal, query brain, tooling), mọi thứ đo được bằng F1.
- **Combo "Kỹ thuật khác biệt"** (cộng thêm ≈2-3 tuần): + **A2 (TRAKE DP) + B2 (image fallback) + D1 (synthetic query)** — 3 thứ ít đội làm tốt, có evidence từ đội 2025.
- **Tối thiểu nếu chỉ chọn 3**: **F1, A1, B1** — eval + temporal + query rewriting là bộ ba ROI cao nhất.

## 5. Thứ tự thực thi đề xuất (khung 8 tuần, điều chỉnh theo lựa chọn)

| Tuần | Việc | Phụ thuộc |
|---|---|---|
| 1 | **F1 eval harness** + F2 scale test + C5 tune weights | — |
| 2-3 | A1 temporal pair (backend scoring + 2 ô query UI) → A4 nếu dư | F1 |
| 3-4 | B1 LLM rewriting (thay paraphraser) + B3 query-by-image | F1 |
| 4-5 | B2 external image fallback · D2 nâng OCR (chạy song song) | B3 |
| 5-6 | E1 UI thi đấu + E2 latency | A1, B3 |
| 6-7 | A2 TRAKE alignment · D1 synthetic queries (nếu chọn) | A1 |
| 7-8 | C1/C2 ensemble+rerank (nếu chọn) · E4/E5 theo track · tập dợt mock contest | tất cả |

**Nguyên tắc xuyên suốt:** mỗi thay đổi phải qua F1 đo trước/sau; không merge thứ làm giảm recall@5; giữ latency <1s.

## 6. Quy trình khi bắt đầu thực thi 1 ý tưởng

1. Đọc mục tương ứng trong file này + section liên quan trong [PROJECT-CONTEXT.md](PROJECT-CONTEXT.md) (§16 chỉ chỗ sửa).
2. Chạy eval baseline (F1) → ghi số.
3. Implement → chạy lại eval → ghi số vào bảng dưới.
4. Cập nhật PROJECT-CONTEXT.md nếu đổi hành vi/luồng.

| Ý tưởng | Ngày | recall@1 | recall@5 | recall@20 | Latency | Ghi chú |
|---|---|---|---|---|---|---|
| (baseline) | | | | | | |

---

## 7. Nguồn

- [Results of the 2025 Video Browser Showdown (arXiv 2509.12000)](https://arxiv.org/html/2509.12000v1) · [VBS Hall of Fame](https://videobrowsershowdown.org/hall-of-fame/) · [VBS Teams & Papers](https://videobrowsershowdown.org/teams/)
- [PraK V4 at VBS 2026 (Springer)](https://link.springer.com/chapter/10.1007/978-981-95-6963-2_25) · [PraK Tool V3](https://link.springer.com/chapter/10.1007/978-981-96-2074-6_39)
- [NII-UIT at VBS2025: LLM Integration + Dynamic Temporal Search](https://dl.acm.org/doi/10.1007/978-981-96-2074-6_38) · [NII-UIT at VBS2026: VQA](https://link.springer.com/chapter/10.1007/978-981-95-6963-2_26)
- [VISIONE 5.0 (MMM 2024)](https://link.springer.com/chapter/10.1007/978-3-031-53302-0_29) · [VISIONE feature repository](https://zenodo.org/records/8188570)
- [SnapSeek 2.0 at VBS 2025](https://link.springer.com/chapter/10.1007/978-981-96-2074-6_41)
- [Event Retrieval from Large Video Collection in HCMC AI Challenge 2024 (Springer)](https://link.springer.com/chapter/10.1007/978-981-96-4291-5_1)
- [RAPID (arXiv 2501.16303)](https://arxiv.org/pdf/2501.16303)
- [Integrated Semantic and Temporal Alignment — AIO_Owlgorithms, AIC 2025 (arXiv 2512.13169)](https://arxiv.org/abs/2512.13169)
- [MADTempo — AIO_Trinh (arXiv 2512.12929)](https://arxiv.org/abs/2512.12929)
- [Lightweight Moment Retrieval: SuperGlobal + ABTS (arXiv 2504.09298)](https://arxiv.org/abs/2504.09298)
- [Hybrid CLIP + BEiT-3 system (SOICT 2024)](https://link.springer.com/chapter/10.1007/978-981-96-4291-5_17)
- [Video Embedding Benchmark 2026 (Mixpeek)](https://mixpeek.com/blog/video-embedding-benchmark-2026)
- [HCM AI Challenge official site](https://aichallenge.hochiminhcity.gov.vn/en/home)

**Bổ sung 2026-06-15 (kiến trúc tham chiếu & query modalities — §1.4):**

- [vitrivr: A Flexible Retrieval Stack Supporting Multiple Query Modes (Rossetto et al., ACM MM 2016)](https://doras.dcu.ie/32428/1/ACMMM16_vitrivr.pdf) — Cineast + ADAMpro + vitrivr-ng; QBE/QBS/motion/audio
- [lifeXplore at LSC 2020 (arXiv 2508.21397)](https://arxiv.org/abs/2508.21397) — feature-map/SOM browsing, concept search, sketch, YOLO9000, OCR, uniform sampling
- [lifeXplore at LSC 2024 (DOI 10.1145/3643489.3661123)](https://dl.acm.org/doi/10.1145/3643489.3661123) — CLIP+FAISS free-text + concept/object/OCR/metadata filter
- [MEMORIA — Memory Enhancement & Moment Retrieval, LSC 2023 (DOI 10.1145/3592573.3593099)](https://dl.acm.org/doi/10.1145/3592573.3593099)
- [Results of the 2025 Video Browser Showdown (arXiv 2509.12000)](https://arxiv.org/abs/2509.12000) — kết quả chính thức VBS2025 (NII-UIT vô địch, VEAGLE best novice)
- [VBS Teams & Papers (all years)](https://videobrowsershowdown.org/teams/) · [VBS Hall of Fame](https://videobrowsershowdown.org/hall-of-fame/) · [Lifelog Search Challenge](http://lifelogsearch.org/)
