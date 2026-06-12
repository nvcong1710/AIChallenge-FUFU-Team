# FUFU-Tool — Báo cáo phân tích & Đề xuất nâng cấp cho AI Challenge HCMC

> Tài liệu tổng hợp: (1) hiểu codebase hiện tại, (2) các luồng hoạt động, (3) phần AI, (4) tóm tắt 4 paper tham khảo, (5) 3 giải pháp baseline, (6) 3 giải pháp sáng tạo nâng cấp, (7) lộ trình triển khai.

## Mục lục

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Kiến trúc &amp; cấu trúc thư mục](#2-kiến-trúc--cấu-trúc-thư-mục)
3. [Các luồng hoạt động chính](#3-các-luồng-hoạt-động-chính)
4. [Luồng AI hiện tại](#4-luồng-ai-hiện-tại)
5. [Tóm tắt 4 paper tham khảo](#5-tóm-tắt-4-paper-tham-khảo)
6. [Bối cảnh HCM AI Challenge — hệ sinh thái &amp; tiến hóa bài toán](#6-bối-cảnh-hcm-ai-challenge--hệ-sinh-thái--tiến-hóa-bài-toán)
7. [Phân tích các đội đoạt giải &amp; kiến trúc của họ](#7-phân-tích-các-đội-đoạt-giải--kiến-trúc-của-họ)
8. [3 giải pháp baseline cho AI Challenge](#8-3-giải-pháp-baseline-cho-ai-challenge)
9. [3 giải pháp sáng tạo nâng cấp FUFU](#9-3-giải-pháp-sáng-tạo-nâng-cấp-fufu)
10. [Chiến lược tối ưu thực chiến đã được kiểm chứng](#10-chiến-lược-tối-ưu-thực-chiến-đã-được-kiểm-chứng)
11. [Kiến trúc combo &amp; lộ trình 8 tuần](#11-kiến-trúc-combo--lộ-trình-8-tuần)

---

## 1. Tổng quan dự án

**FUFU** là hệ thống tìm kiếm & truy xuất video mã nguồn mở, dựa trên nền tảng SOMHunter, từng trưng bày tại **AI Challenge 2023 (TP.HCM)**. Mục đích: cho phép tìm video bằng **truy vấn ngôn ngữ tự nhiên** (semantic search).

**Đội ngũ gốc**: Tu Nguyen, Huy Dang, Van Bui, Quan Nguyen, Dat Pham. Giấy phép Apache 2.0.

**Stack:**

- Backend: Spring Boot 2.7.15, Java 17
- Frontend: React 18 + Bootstrap 5
- Vector DB: Weaviate (cho semantic search)
- RDBMS: MySQL
- Container: Docker + GitHub Actions CI/CD

---

## 2. Kiến trúc & cấu trúc thư mục

```
React Frontend (port 3006)
        ↓ HTTP
Spring Boot Backend (port 8080)
        ↓
   ┌────┴────┐
 MySQL    Weaviate (Vector DB, port 8081)
(metadata)  (embeddings cho semantic search)
```

```
FUFU-Tool/
├── backend/                          # Spring Boot backend
│   ├── pom.xml
│   └── src/main/java/com/challenge/fufu/
│       ├── Application.java
│       ├── controller/      (VideoController, WeaviateController, Advice)
│       ├── service/         (VideoService, WeaviateService, ItemService)
│       ├── model/           (Video entity, ResponseHandler, Exception)
│       ├── repository/      (VideoRepository - JPA)
│       ├── config/          (WeaviateConfig, SwaggerConfig)
│       ├── exception/       (custom: NotFound/BadRequest/...)
│       └── util/            (Constants, DateUtil)
├── frontend/
│   └── src/
│       ├── App.js
│       ├── index.js
│       └── components/      (SearchSection.jsx, ResultSection.jsx)
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/pipeline.yml
```

**Backend** dùng các dependency chính: Spring Web/Security/JPA, `io.weaviate:client:4.0.0`, MySQL Connector, Lombok, Swagger.

**Frontend** đơn giản (UI prototype): 1 trang với ô input và grid kết quả, hiện đang gọi tạm OMDB API (`omdbapi.com`) — chưa nối backend.

---

## 3. Các luồng hoạt động chính

### Luồng 1 — Khởi động ứng dụng

```
mvn package → fufutool.jar
        ↓
java -jar fufutool.jar
        ↓
Spring scan: @Configuration, @RestController, @Service, @Repository, @Entity
        ↓
Hibernate auto-create bảng (ddl-auto=update) trong MySQL
        ↓
Tomcat embedded listen :8080
```

Yêu cầu: MySQL :3306 và Weaviate :8081 phải chạy sẵn.

### Luồng 2 — CRUD Video (MySQL)

```
GET /api/videos
   VideoController.getAllVideos()
      → videoService.getAllVideos()
         → videoRepository.findAll()  (SELECT * FROM video)
   ← List<Video> JSON

POST /api/videos  body: {title, description, weaviateId}
   VideoController.createVideo()
      → videoService.saveVideo()
         → videoRepository.save()  (INSERT, id IDENTITY)
   ← Video đã lưu
```

`Video` entity có 4 trường: `id, title, description, weaviateId` — trong đó `weaviateId` là **cầu nối** sang Weaviate.

### Luồng 3 — Semantic Search (3 pha)

**3a. Tạo schema (chạy 1 lần):**

```
POST /api/weaviate/create-class?className=Video
   → POST {weaviate}/v1/schema  body: {"class": "Video"}
```

**3b. Index video:**

```
POST /api/weaviate/index-video?className=Video&id=...&title=...&description=...
   → POST {weaviate}/v1/objects
   Weaviate: vectorizer module sinh embedding ~768d → lưu vào HNSW index
```

**3c. Search:**




```
GET /api/weaviate/search-video?className=Video&query=...
   → POST {weaviate}/v1/graphql với nearText
   Weaviate: query → embedding → cosine similarity → top-K objects
```

**⚠ Bug đã phát hiện**: GraphQL query ở `WeaviateService.searchVideos` đang sai cú pháp (dùng `search` thay vì `Get` + `nearText`) và không nhúng biến `query` vào payload. Cần sửa để search chạy thật.

### Luồng 4 — Frontend (hiện chưa nối backend)

```
User gõ query → click "Generate"
   searchData(searchTerm)
      fetch("http://www.omdbapi.com?apikey=...&s=" + title)  ❗ KHÔNG gọi backend
   setQuery(data.Search)
   <ResultSection queries={queries}/> render danh sách poster
```

Cần đổi `API_URL` thành endpoint Weaviate của backend để hoàn thiện E2E.

### Luồng 5 — Xử lý lỗi

```
throw ArchitectureException (NotFound | BadRequest | Conflict | ...)
   → @RestControllerAdvice CustomExceptionHandler
      → ErrorResponse + đúng HTTP status (404/400/409/...)
   → Client nhận JSON lỗi
```

### Luồng 6 — CI/CD Deploy

```
git push origin dev
   → GitHub Actions:
      Job 1: docker build + mvn test
      Job 2: SSH server → docker pull + restart container :80
```

---

## 4. Luồng AI hiện tại

**Điểm quan trọng**: Code Java **không chứa AI nào** — không train, không inference, không gọi OpenAI/HuggingFace. Toàn bộ phần AI được uỷ thác cho **Weaviate** (external vector DB).

### Pipeline AI 2 nửa

**Nửa 1 — INDEX TIME (offline):**

```
title + description (text)
        ↓ Build JSON, POST /v1/objects
Weaviate:
   [Bước 1] Vectorizer (text2vec-transformers / text2vec-openai)
       text ──BERT/Sentence-BERT/CLIP──▶ vector ~768d
   [Bước 2] Lưu (object, vector) vào HNSW graph
```

**Nửa 2 — QUERY TIME (online):**

```
query text
        ↓ POST /v1/graphql nearText
Weaviate:
   [Bước 1] Vectorize query bằng CHÍNH model lúc index
   [Bước 2] HNSW k-NN: cosine(q, v) cho mọi v trong index
   [Bước 3] Lọc certainty ≥ 0.9, sort desc → trả top-K
```

**Insight cốt lõi**: hai câu cùng nghĩa → hai vector gần nhau trong không gian 768d, kể cả khi không trùng từ khoá nào. Đây là khác biệt cốt lõi giữa semantic search vs keyword search.

### "AI" lý thuyết theo README vs "AI" thực tế

| Thuật toán README đề cập (kế thừa SOMHunter) | Có trong code?   |
| --------------------------------------------------- | ----------------- |
| W2VV++ (text-to-video deep model)                   | ❌                |
| SOM (Self-Organizing Map) re-ranking                | ❌                |
| Relevance feedback loop                             | ❌                |
| Keyword ranker                                      | ❌                |
| Vector similarity search                            | ✅ (qua Weaviate) |

→ Bản code này **chỉ implement xương sống** (vector search). Các thuật toán nâng cao của SOMHunter gốc chưa được port.

---

## 5. Tóm tắt 4 paper tham khảo

### Paper #1 — VisionLLM ([arXiv:2305.11175](https://arxiv.org/abs/2305.11175), 2023)

**"Large Language Model is also an Open-Ended Decoder for Vision-Centric Tasks"** — Wang, Chen, Dai, ...

- **Ý tưởng**: Coi ảnh như "ngoại ngữ" — dùng **LLM làm open-ended decoder** cho task thị giác.
- **Đóng góp**: Người dùng "lập trình" task vision bằng instruction ngôn ngữ tự nhiên, không cần model chuyên dụng.
- **Kết quả**: ~60% mAP trên COCO, sánh ngang model detection chuyên dụng.

### Paper #2 — Images in Language Space ([arXiv:2305.13782](https://arxiv.org/abs/2305.13782), ACL'23 Findings)

**"Exploring the Suitability of LLMs for Vision & Language Tasks"** — Hakimov, Schlangen.

- **Ý tưởng**: **Verbalize ảnh thành text** rồi đẩy vào LLM thuần (không cần fine-tune multimodal).
- **Đóng góp**: Bảo toàn khả năng reasoning của LLM + **interpretable** (truy ngược được output đến mô tả ảnh).
- **Kết quả**: Hiệu quả ngay cả với ít sample, open-source LLM cạnh tranh được với GPT-3.

### Paper #3 — Qwen-VL ([arXiv:2308.12966](https://arxiv.org/abs/2308.12966), 2023)

**"A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond"** — Alibaba Qwen Team.

- **Ý tưởng**: VLM đa năng mở rộng từ Qwen LLM, đa ngôn ngữ.
- **Tính năng**: captioning + VQA + **visual grounding (bbox)** + **OCR** + multi-image dialog.
- **Bản open-source**: `Qwen-VL-Chat`, dùng được ngay. SOTA cho generalist model cùng kích cỡ.

### Paper #4 — LaVi ([arXiv:2506.16691](https://arxiv.org/abs/2506.16691), 2025)

**"Efficient Large Vision-Language Models via Internal Feature Modulation"** — Yue, Guo, Liu, ...

- **Ý tưởng**: Fusion vision↔language qua **feature modulation** trong LayerNorm (inject token-wise vision deltas vào affine params), thay vì concat visual tokens (làm phình context).
- **Kết quả vs LLaVA-OV-7B**: **−94% FLOPs, ×3.1 nhanh hơn, −50% memory**, vẫn giữ SOTA.

---

## 6. Bối cảnh HCM AI Challenge — hệ sinh thái & tiến hóa bài toán

### 6.1 Hệ sinh thái cuộc thi VN

HCM AI Challenge **không tồn tại đơn lẻ** mà nằm trong mạng lưới các cuộc thi liên đới, cung cấp cả dataset, talent pipeline và cảm hứng kỹ thuật:

| Cuộc thi                   | Tổ chức                   | Bài toán tiêu biểu                 | Đóng góp                     |
| --------------------------- | --------------------------- | -------------------------------------- | ------------------------------- |
| **HCM AI Challenge**  | Sở TT&TT TP.HCM, ĐHQG-HCM | OCR tiếng Việt, Event Retrieval      | Đô thị thông minh           |
| **Zalo AI Challenge** | Zalo (VNG)                  | 5K Compliance, Liveness, Generative AI | Xử lý đặc thù tiếng Việt |
| **Viet Solutions**    | Bộ TT&TT + Viettel         | Quản trị dữ liệu, lọc nội dung   | Ươm mầm SaaS                 |
| **I-Star Awards**     | UBND TP.HCM                 | 3D/360, cảnh báo giao thông         | Thương mại hóa              |
| **AI for Impact**     | Western Sydney + Launch Pad | AI cho SDGs                            | STEM cho THPT                   |

Mùa 2024 HCM AI Challenge thu hút **3.000+ thí sinh sơ tuyển → 74 đội vào chung kết**. Tới mùa 2026 đã là mùa thứ 5-6, format chia 2 bảng: **Bảng A (sinh viên/chuyên gia)** và **Bảng B (THPT)**.

### 6.2 Tiến hóa bài toán

| Năm                        | Bài toán                                                                                                                                                                                                       | Mức độ          |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| **2021**              | Nhận diện chữ Tiếng Việt trong ảnh tự nhiên (Scene Text Recognition - STR). Dataset: 2.000 ảnh, 56.000 từ, 1.000 test.                                                                                 | Unimodal CV        |
| **2022-2024**         | **Event Retrieval from Visual Data** — truy vấn sự kiện từ video bằng ngôn ngữ tự nhiên. Format theo chuẩn **LSC (Lifelog Search Challenge)** và **VBS (Video Browser Showdown)**. | Multimodal         |
| **2026 (hiện tại)** | Trợ lý ảo phân tích & truy xuất thông tin từ multimedia. 2 track: Traditional + Automated.                                                                                                               | Multimodal + Agent |

**Insight cốt lõi**: cuộc thi đã chuyển dịch rõ rệt từ **AI hẹp** (nhận diện chuyên biệt 1 task) sang **AI đa phương thức + lý luận đa kênh** (multimodal reasoning). Đây chính xác là hướng các paper #1 (VisionLLM) và #3 (Qwen-VL) đang theo đuổi.

---

## 7. Phân tích các đội đoạt giải & kiến trúc của họ

> Đây là **vàng ròng** — cho biết chính xác cái gì đã thắng để mình biết phải **vượt qua** chứ không lặp lại.

### 7.1 TaQuangTu — Traffic Tracking (mùa cũ)

Kho `TaQuangTu/HCM-AI-Challenge` — pipeline cho bài toán giám sát giao thông:

- **Model**: YOLOv4 + CSPDarknet53 backbone (cân bằng mAP/FPS tốt).
- **Lớp đối tượng** giới hạn: cars, cyclists, trucks, bus (lọc bớt nhiễu từ MS COCO).
- **Sáng tạo bản địa**: **ROI polygon** lưu trong file `.json` → khi tâm bbox cắt polygon → tính trajectory để xác định direction.
- **Bài học**: prior knowledge dạng hình học (geometry) cực kỳ rẻ và mạnh khi camera tĩnh.

### 7.2 NaiveNotNice — Event Retrieval 2024 (chisngooo)

Multi-model **ensemble "chia để trị"** — đại diện điển hình cho cấu trúc VBS thực chiến:

```
Video
  ├─ OCR module      → trích chữ trên màn (biển số, biển hiệu) → metadata
  ├─ ASR module      → hội thoại + tiếng động → text
  ├─ CLIP module     → embed shared latent space (trái tim hệ thống)
  ├─ Image Caption   → mô tả khung cảnh
  └─ Object & Color  → filter theo thuộc tính vật lý
```

**Search**: query → CLIP text embed → cosine với hàng triệu frame vector → top-K.

**Đây gần như chính xác là kiến trúc FUFU đang nhắm tới**, và là **baseline phổ biến nhất trong cuộc thi**. Tức là làm theo cách này → chỉ ngang đám đông, không thắng được.

### 7.3 TycheVid — Giải Nhất Bảng A 2024 (UIT)

5 thành viên (UIT-ĐHQG TP.HCM): Trần Gia Bảo, Bùi Công Khánh Tường, Trần Nhật Khoa, Lê Thị Thanh Tâm, Hồ Trọng Hiền.

**Bí kíp thắng** (theo Ngô Đức Thành — Phó Chủ tịch BGK): không hơn về thuật toán, mà hơn về **TỐC ĐỘ + EFFICIENCY**:

- **Vector DB + ANN/HNSW** (Hierarchical Navigable Small World) → latency mili-giây.
- **Index thông minh** hàng triệu vector, tránh bottleneck bộ nhớ.
- **Khai thác triệt để OSS** uy tín toàn cầu thay vì tự sáng tạo từ đầu.
- **Thích ứng nhanh** khi thể thức thi đổi đột ngột.

**Bài học vàng**: Chênh lệch thuật toán giữa các đội top là **rất nhỏ**. Người thắng là người **có hạ tầng nhanh hơn** — sub-second response cho phép operator thử nhiều query hơn trong cùng thời gian giới hạn. → Latency là vũ khí cạnh tranh chính.

### 7.4 FriedPotatoes — Giải Nhất Bảng B 2024 (PT Năng khiếu)

Sẩm Pí Diệu, Nguyễn Quang Thiện — học sinh THPT thắng với:

- **Tối ưu bộ lọc logic sắc bén** trên công cụ BTC cung cấp.
- **Vượt hạn chế phần cứng** bằng query syntax đa lớp.

→ Cho thấy **không cần GPU khủng**, cần **logic search query mạnh**.

### 7.5 RAPID — đội URA (giải Ba 2024, xuất bản SOICT 2024) ⭐

Đây là kiến trúc **học thuật nhất, ấn tượng nhất**, đã xuất bản tại hội nghị quốc tế SOICT 2024 với tên *"Retrieval-Augmented Parallel Inference Drafting for Text-Based Video Event Retrieval"*.

Đội gốc là **học sinh THPT Ngô Quyền** (Nguyễn Gia Huy, Khưu Gia Bảo, Nguyễn Thanh Tuấn), bảo trợ bởi URA (GS. Quan Thanh Thọ, ThS. Nguyễn Song Thiên Long).

**4 trụ cột kiến trúc RAPID**:

1. **BLIP-2 với Q-Former** cho nhúng đa phương thức — mạnh hơn CLIP, ít tham số huấn luyện hơn.
2. **envit5-translation bridging** — dịch tự động query tiếng Việt → tiếng Anh trước khi đưa vào BLIP-2, tránh "catastrophic forgetting" của VLM huấn luyện trên tiếng Anh.
3. **YOLOWorldv2** (Open-Vocabulary Object Detection) + **PP-OCR** — tìm bất kỳ object nào không cần định nghĩa trước, không bị giới hạn nhãn cố định như YOLOv4.
4. **KPI** (Knowledge-based Processing for Interactive Video Retrieval) — tổng hợp text + ASR + OCR cho **time-based segment search** và **elevated keyframe selection**.

**Insight vàng cho cuộc thi 2026**:

- **Translation bridging** là pattern *thực dụng* mà FUFU (và đa số đội) đang bỏ lỡ.
- **Open-vocabulary detection** > closed-set detection trong mọi trường hợp event retrieval.
- **Time-based segment search** ≈ giải pháp Temporal Event Graph mà tôi đã đề xuất ở Section 9.3 — RAPID đã đi trước theo hướng này.

### 7.6 Thành tích VN tại đấu trường quốc tế

| Năm | Cuộc thi                                               | Đội VN                    | Hạng                 | Đối thủ                      |
| ---- | ------------------------------------------------------- | --------------------------- | --------------------- | ------------------------------- |
| 2021 | NVIDIA AI City Challenge Track 5 (NL Vehicle Retrieval) | HCMUS                       | **#2** (0.1741) | Alibaba, Baidu, ByteDance, HUST |
| 2022 | NVIDIA AI City Challenge                                | HCMIU-CVIP                  | **#3**          | MegVideo, Terminus-AI           |
| 2022 | NVIDIA AI City Challenge                                | HCMUS                       | Top 5                 | Toyota, Huawei                  |
| 2025 | Global AI Challenge HK                                  | VN (AIZ + Viettel + VCCorp) | **HVàng**      | 200+ đội từ 26 quốc gia     |

→ VN có truyền thống mạnh trong **multimodal retrieval** và **edge optimization**. Đây là nền tảng tốt để target Top.

---

## 8. 3 giải pháp baseline cho AI Challenge

> Bối cảnh: AI Challenge HCMC theo format **Video Browser Showdown / Lifelog Search Challenge** — cho dataset video lớn, truy vấn ngôn ngữ tự nhiên, trả về frame/đoạn đúng. Hai track: **Traditional** (operator vận hành) và **Automated** (AI tự chạy).

### GP #1 — "Verbalize-then-Search" (paper #2 + #3)

> Biến mọi keyframe thành đoạn text giàu thông tin offline → search text rẻ và chính xác online.

```
OFFLINE: video → keyframes → Qwen-VL-Chat → {caption, OCR, entities, action}
       → Vietnamese-SBERT embed → Weaviate + BM25 index

ONLINE: query → hybrid retrieve (0.7·dense + 0.3·BM25) → top-200
      → cross-encoder rerank → top-20 → display caption highlight
```

**Mạnh:** tận dụng được code FUFU; OCR Qwen-VL xử lý chữ tiếng Việt tốt; interpretable cho operator; search siêu nhanh.

**Yếu:** chất lượng phụ thuộc caption offline; tốn GPU lúc ingest (1 lần).

### GP #2 — "CLIP-First, VLM-Rerank" (paper #3 + #4)

> Vector multimodal cho first-stage rộng → VLM mạnh rerank top-K nhỏ.

```
INGEST: keyframes → CLIP-ViT-L/14 multilingual → vector 768d → Weaviate

QUERY: q → CLIP text encoder → q_vec
     → Weaviate top-500 (~10ms)
     → LaVi-style efficient VLM rerank ("Does this match: <query>?")
     → top-20 (~2s trên 1×A100)
     → nếu query có grounding → Qwen-VL bbox → boost
```

**Mạnh:** kiến trúc 2-tầng là chuẩn mực thắng VBS thực tế; LaVi giúp rerank đủ nhanh; xử lý query không gian qua grounding.

**Yếu:** cần GPU mạnh khi inference.

### GP #3 — "Agentic Search Operator" (paper #1 + #4, cho Automated Track)

> LLM = decoder tổng quát, biến cả việc search thành 1 agent tự lập kế hoạch + gọi tool.

```
LLM Orchestrator (Qwen2.5-14B/72B) gọi tools:
   • search_clip(q, k) → top-K id
   • verbalize_frame(frame_id) → Qwen-VL desc chi tiết
   • check_temporal(frame, ±t) → before/after context

Loop tới khi LLM confident → submit frame_id + reasoning trace
```

**Mạnh:** phù hợp track Automated; reasoning trace giúp giám khảo verify; ấn tượng demo.

**Yếu:** phức tạp nhất, dễ over-engineering, cần prompt engineering kỹ.

### So sánh nhanh

| Tiêu chí                   | #1 Verbalize | #2 CLIP+Rerank | #3 Agent            |
| ---------------------------- | ------------ | -------------- | ------------------- |
| Độ khó                    | thấp        | trung bình    | cao                 |
| GPU query                    | rẻ          | trung bình    | tốn                |
| GPU ingest                   | tốn         | rẻ            | rẻ                 |
| Query phức tạp             | ⚠           | ✅             | ✅✅                |
| OCR tiếng Việt             | ✅✅         | ⚠             | ✅                  |
| Query không gian/thời gian | ❌           | ✅             | ✅✅                |
| Interpretable                | ✅✅         | ⚠             | ✅✅                |
| Phù hợp track              | Traditional  | Cả 2          | **Automated** |

---

## 9. 3 giải pháp sáng tạo nâng cấp FUFU

> Để có **tính mới thực sự**, phải tấn công đúng khoảng trống của cả ngành VBS, không chỉ làm CLIP+rerank như mọi đội (NaiveNotNice đã làm), và phải đi xa hơn RAPID — kiến trúc mạnh nhất hiện tại.

### Khoảng trống cần lấp

| Điểm yếu                                     | FUFU  | Cả ngành VBS                                          |
| ----------------------------------------------- | ---------- | ------------------------------------------------------- |
| 1 vector / 1 frame                              | ✗         | Hầu hết → mất chi tiết ai/làm gì/text trên màn |
| Frame độc lập, không hiểu chuỗi sự kiện | ✗         | "X xảy ra rồi Y" → fail                              |
| Operator click không dạy được hệ thống   | ✗         | Vài hệ dùng Rocchio thủ công (1965!)               |
| Caption là bottleneck một chiều              | ✗         | Verbalize → đè detail thị giác                     |
| Mù tiếng Việt văn hoá                      | ✗ (OMDB!) | CLIP gốc training tiếng Anh                           |

### Sáng tạo #1 — **Disentangled Multi-Axis Retrieval (DMAR)**

> Thay vì 1 vector/frame, lưu **6 vector song song** đại diện cho 6 trục ngữ nghĩa độc lập. LLM phân rã query thành **weighted compositional query** → đánh đúng trục.

```
        MỘT KEYFRAME
              │
   ┌────┬────┼────┬────┬────┐
   ▼    ▼    ▼    ▼    ▼    ▼
 SCENE ACTION ENT  OCR AUDIO TEMP
 (CLIP)(Act- (NER)(Qwen(Whisp(±5s
       CLIP)      VL)  er)   clip)
   └────┴────┴──┬─┴────┴────┘
                ▼
   6 vector × 512d → Weaviate multi-vector


QUERY "Tổng Bí thư phát biểu về kinh tế trên VTV1 năm 2023"
              ↓ LLM Query Decomposer
   {
     entity: ("Tổng Bí thư", w=0.35),
     ocr:    ("VTV1, 2023",  w=0.30),
     audio:  ("kinh tế",     w=0.20),
     scene:  ("studio TV",   w=0.10),
     action: ("phát biểu",   w=0.05)
   }
              ↓ Multi-vector search có trọng số → fuse → top-K
```

**Cảm hứng**: LaVi (paper #4) — chứng minh có thể tách vision feed vào *vị trí cụ thể* trong mạng. Áp triết lý đó ở cấp **retrieval index**.

**Mới ở chỗ**:

- VBS hiện nhồi mọi thông tin vào 1 vector → query "VTV1" và "studio TV" trộn lẫn không phân biệt.
- Operator có thể **chỉnh trọng số trục bằng UI slider** → trải nghiệm mới chưa hệ nào có.

**Diff vs FUFU**: thay class `Video` đơn lẻ → 6 class song song link nhau qua `frame_id`; thêm Python AI service cho ingest; React thêm slider.

### Sáng tạo #2 — **Session Adapter: hệ học từ click realtime**

> Mỗi session, train 1 **tiny adapter δ** (vài KB) trên embedding query, cập nhật theo từng click ✓/✗ của operator. Sau 3-5 click, query hội tụ về ý đồ thực — **không cần gõ lại**.

```
Lượt 1: gõ "người phụ nữ áo đỏ"
   q₀ = embed(...) → search → 20 ảnh

Operator click:
   ✓ #3   ✗ #7 (áo cam)   ✗ #12 (đàn ông)
              ↓
   Online Gradient Update:
     L = -log σ(sim(q+δ, v_pos)) + Σ log σ(sim(q+δ, v_neg))
     δ ← δ - η · ∇_δ L     (3-5 steps)
              ↓
   q₁ = q₀ + δ   ← ngầm encode "ĐỎ chứ không CAM, NỮ chứ không NAM"
              ↓
   Re-search Weaviate (<100ms) → kết quả chuẩn hơn
```

**Cảm hứng**: VisionLLM (paper #1) — LLM là decoder "lập trình runtime". Kết hợp LaVi modulation ở dạng vi mô: δ chính là 1 modulation vector áp vào query.

**Bonus "Click-to-Explanation"**: LLM nhận `δ` và sinh giải thích cho operator: *"Hệ thống đã hiểu bạn ưu tiên màu đỏ thuần thay vì cam, đối tượng nữ giới"* → tăng niềm tin, đậm chất AI Challenge.

**Mới ở chỗ**:

- VBS hiện chỉ có Rocchio relevance feedback (cộng/trừ trung bình vector, từ 1965!). Đây là **gradient descent thực sự** trên adapter — neural, online, không retrain index.
- δ trôi nổi trong session, reset khi sang query mới → không drift.

**Diff vs FUFU**: thêm `SessionAdapterService` (in-memory `Map<sessionId, δ>`); React thêm nút ✓/✗ → POST `/api/feedback` → re-rank tự động.

### Sáng tạo #3 — **Temporal Event Graph + Synthetic Query Augmentation**

> Build offline đồ thị thời gian (frame=node, "rồi-thì"=edge). Mỗi frame đi kèm **K query tổng hợp** (LLM tự sinh). Cho phép **pattern matching theo chuỗi sự kiện**.

**Phần A — Synthetic Query Pre-generation:**

```
OFFLINE, mỗi keyframe:
   Qwen-VL prompt: "Sinh 15 truy vấn tự nhiên mà người Việt
                    có thể dùng để tìm ảnh này"
   →
   ["xe máy đỏ trước cửa quán phở",
    "Hà Nội phố cổ giờ tan tầm",
    "biển hiệu Phở Thìn",
    ... 15 cái]
   Embed cả 15 → lưu Weaviate (cùng frame_id)
```

Query user matching với 15 synthetic ("query-style") mạnh hơn match với caption thô ("description-style") — đây là *query-document distribution gap*, vấn đề thực trong IR.

**Phần B — Temporal Event Graph:**

```
Node = shot/keyframe
Edge:
  • next  (shot kế tiếp cùng video)
  • cut   (chuyển cảnh đột ngột)
  • cont  (tiếp diễn ngữ nghĩa, cosine ≥0.85)
Attributes: {scene, action, entities, ocr, asr, start, end}

QUERY "Cảnh phỏng vấn ông X, ngay sau đó là biểu đồ chứng khoán"
                ↓ LLM Pattern Parser (VisionLLM style)
   pattern = MATCH (a)-[:next*1..3]->(b)
             WHERE a.entities CONTAINS "ông X"
               AND a.action ~ "phỏng vấn"
               AND b.scene ~ "biểu đồ"
               AND b.ocr ~ "VN-Index|chứng khoán"
                ↓ Cypher-like query trên graph
   → CẶP (a, b) → submit
```

**Mới ở chỗ**:

- **Synthetic query augmentation** đã có trong IR text (doc2query, HyDE) nhưng **chưa hệ VBS nào áp dụng cho video**.
- **Temporal graph** giải quyết query "before/after" — Automated track killer feature.

**Diff vs FUFU**: thêm Neo4j (hoặc PostgreSQL recursive CTE) song song Weaviate; thêm step `generate_15_queries` trong ingest; backend thêm `GraphQueryService` translate LLM-output → Cypher.

---

## 10. Chiến lược tối ưu thực chiến đã được kiểm chứng

> 3 trụ cột rút từ phân tích các đội thắng giải HCM AI Challenge + AI City Challenge.

### 10.1 Multimodal Latent Space + Contrastive Learning

Đây là **chuẩn mực ngầm** của ngành:

- **CLIP / BLIP-2 Q-Former** thay vì BM25/tf-idf cổ điển.
- **InfoNCE loss** kéo cặp ảnh-text khớp lại gần, đẩy cặp không khớp ra xa.
- **Embed toàn bộ keyframe offline** → online chỉ là phép nhân ma trận (cosine/L2) → tốc độ mili-giây.

**Áp dụng cho FUFU**: phải thay vectorizer mặc định của Weaviate bằng **CLIP-Vi multilingual** hoặc **BLIP-2** + bridging translation.

### 10.2 Cascading Multi-Stage Filtering (kiến trúc 3 tầng)

Không thể chạy model nặng trên mọi frame → out-of-memory. Pipeline thắng:

```
Tầng 1 — COARSE (Elasticsearch trên metadata OCR/ASR)
  Hàng triệu frame → hàng chục nghìn ứng viên
                            ↓
Tầng 2 — DENSE (vector similarity với CLIP/BLIP-2)
  Hàng chục nghìn → top-K (~500-1000)
                            ↓
Tầng 3 — RE-RANK (model nặng: BLIP-2 / KPI / cross-encoder)
  Top-K → top-20 chính xác
```

**Áp dụng cho FUFU**: hiện tại chỉ có 1 tầng (Weaviate semantic search). Phải thêm Elasticsearch index tiền OCR/ASR + 1 tầng rerank cuối.

### 10.3 Robustness — chống nhiễu thực tế

| Nguồn nhiễu                                         | Đối phó                                        |
| ----------------------------------------------------- | ------------------------------------------------- |
| Cảm biến lỗi, dữ liệu thiếu (HK Global AI 2025) | Smoothing + imputation                            |
| Ánh sáng đèn xe / occlusion (TaQuangTu)           | ROI polygon JSON + tracking trajectory            |
| VLM mù tiếng Việt (RAPID)                          | **Translation bridging** envit5-translation |
| Bbox chập chờn giữa các frame                     | Multi-frame fusion / temporal smoothing           |

**Áp dụng cho FUFU**: trước khi index → preprocess data + translation bridging tiếng Việt → tiếng Anh.

### 10.4 ANN/HNSW cho latency mili-giây (TycheVid's secret)

TycheVid thắng nhờ **HNSW** trong Weaviate (đã có sẵn!) nhưng cần:

- Tune `efConstruction`, `efSearch`, `maxConnections` cho dataset cụ thể.
- Pre-warm cache, mmap index.
- Batch search nếu có nhiều query song song.

Weaviate của FUFU đã có HNSW nhưng đang dùng default config → có thể tăng 5-10× tốc độ chỉ bằng tuning.

---

## 11. Kiến trúc combo & lộ trình 8 tuần

### Kiến trúc kết hợp cả 3 sáng tạo trên xương FUFU

```
              ┌─────────────────────────────────┐
              │   React UI (FUFU v2)       │
              │  - axis sliders   (từ DMAR)     │
              │  - ✓/✗ buttons     (từ Session)  │
              │  - timeline grid  (từ TEG)      │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │   Spring Boot API Gateway        │
              │     (giữ nguyên FUFU)       │
              └────────────────┬────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
 ┌─────────────┐       ┌─────────────┐       ┌──────────────┐
 │  Weaviate   │       │  Session    │       │   Neo4j      │
 │ 6-axis multi│       │  Adapter    │       │ Event Graph  │
 │   vector +  │       │  Service    │       │    (TEG)     │
 │  synthetic  │       │ (in-mem δ)  │       │              │
 │   queries   │       │             │       │              │
 └─────────────┘       └─────────────┘       └──────────────┘
                               ▲
                               │ gRPC
              ┌────────────────┴────────────────┐
              │   Python AI Service              │
              │  • Qwen-VL (caption, OCR,        │
              │    grounding, synthetic query)   │
              │  • Whisper (Vietnamese ASR)      │
              │  • CLIP-Vi (multilingual)        │
              │  • LaVi-style efficient reranker │
              └──────────────────────────────────┘
```

### "Wow factor" cho ban giám khảo

1. **Live demo Session Adapter**: query mơ hồ → click 3 lần → kết quả chính xác như phép màu. **Chưa đội VBS/HCM AI Challenge nào có**.
2. **Slider axis sáng đèn realtime**: operator nhìn được ngay trục nào đang chi phối → interpretable hơn cả RAPID.
3. **Temporal query "X rồi Y"**: NaiveNotNice không có, RAPID có nhưng dạng KPI tuyến tính — Event Graph là superset đầy đủ hơn.
4. **Bản tiếng Việt thuần** + translation bridging (kế thừa pattern envit5 từ RAPID nhưng đảo chiều cho cả captioning).
5. **Sub-second latency** (kế thừa bài học TycheVid): tuning HNSW + caching → response < 100ms cho mọi query → operator thử nhiều query → tìm đúng nhanh hơn.

### So sánh với các đội đã đoạt giải

| Tính năng           | NaiveNotNice (2024) | TycheVid (#1 2024)  | RAPID (giải 3, SOICT'24) | **FUFU v2 (đề xuất)** |
| --------------------- | ------------------- | ------------------- | ------------------------- | ----------------------------------- |
| Multi-modal embedding | CLIP                | CLIP + ANN          | BLIP-2 Q-Former           | **6-axis DMAR + BLIP-2**      |
| OCR                   | ✅                  | ✅                  | PP-OCR                    | ✅ + dedicated axis                 |
| ASR                   | ✅                  | -                   | ✅ (KPI)                  | ✅ + Whisper-VN                     |
| Translation bridging  | ❌                  | ❌                  | ✅ envit5                 | ✅ envit5 bidirectional             |
| Object detection      | Closed-set          | -                   | YOLOWorldv2 (open)        | YOLOWorldv2 (open)                  |
| Temporal reasoning    | ❌                  | ❌                  | KPI (segment)             | **Event Graph (full Cypher)** |
| Operator feedback     | Manual              | Manual              | Manual                    | **Session Adapter (neural)**  |
| Synthetic query       | ❌                  | ❌                  | ❌                        | **15 queries/frame**          |
| HNSW tuning           | Default             | **Optimized** | -                         | Optimized + cached                  |
| Latency target        | ~1s                 | <100ms              | ~500ms                    | **<100ms**                    |

### Lộ trình 8 tuần thực tế (cập nhật với insight từ các đội đoạt giải)

| Tuần | Việc                                                                                                                                                     | Inspired by                               |
| ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| 1-2   | Refactor FUFU: tách Python AI service, sửa GraphQL bug, dựng ingest pipeline cơ bản.**Thêm envit5-translation bridging** ngay từ đầu. | RAPID                                     |
| 3-4   | Implement DMAR (6 axis): CLIP-Vi + Qwen-VL OCR + Whisper-VN ASR + NER tiếng Việt + YOLOWorldv2 (open-vocab)                                             | RAPID + đề xuất riêng                 |
| 5     | Synthetic query generation (15/frame) +**Cascading 3-stage filter** (Elasticsearch + Weaviate + rerank)                                             | Đề xuất riêng + chuẩn ngành         |
| 6     | Session Adapter + UI feedback buttons +**HNSW tuning** (`efSearch`, batch search)                                                                 | Đề xuất riêng + TycheVid              |
| 7     | Temporal Event Graph + LLM pattern parser (Cypher-like)                                                                                                   | Đề xuất riêng (vượt KPI của RAPID) |
| 8     | LaVi-style efficient reranker, polish UI,**stress-test latency < 100ms**, viết slide thuyết trình + ablation study                               | TycheVid + LaVi                           |

### Tóm tắt 1 dòng

> Đừng chạy đua CLIP+rerank như mọi đội — hãy **đập vỡ embedding monolithic (DMAR)**, **làm hệ thống tự học trong session (Session Adapter)**, và **lên một tầng cao hơn về thời gian (Event Graph)**. Ba thứ này chưa có hệ VBS nào kết hợp đầy đủ, đủ "mới" để ấn tượng, đủ thực tế để chạy được trong 8 tuần dựa trên FUFU sẵn có.

### Định vị cạnh tranh

- **Vượt NaiveNotNice** ở chỗ: không phải multi-model ensemble đơn giản, mà là **multi-axis disentangled** với weighted query decomposition.
- **Vượt TycheVid** ở chỗ: cùng tốc độ sub-second nhưng có thêm **interpretability + neural feedback** + **temporal reasoning**.
- **Vượt RAPID** ở chỗ: kế thừa translation bridging và open-vocab detection của họ, nhưng **đi xa hơn** với Event Graph (thay KPI tuyến tính) + Session Adapter (chưa có trong RAPID) + Synthetic Queries (đóng góp mới).
- **Bài học từ FriedPotatoes**: kết hợp **logic filter sắc bén** với GPU tối thiểu → có thể demo mượt ngay cả khi BTC giới hạn hạ tầng.

---

## Phụ lục A — Các điểm cần fix trước khi mở rộng

1. **Bug GraphQL** ở `WeaviateService.searchVideos` (sai cú pháp, không nhúng `query` vào payload).
2. **Frontend chưa nối backend**: `SearchSection.jsx` đang gọi OMDB, cần đổi sang `/api/weaviate/search-video`.
3. **Secrets hardcode** trong `application.properties` (MySQL password, Weaviate token) → move sang env var.
4. **docker-compose thiếu service**: chỉ có backend, chưa khai báo MySQL/Weaviate/frontend.
5. **Chưa đồng bộ MySQL ↔ Weaviate**: index-video không tự lưu vào MySQL → cần wrap trong cùng transaction hoặc dùng outbox pattern.

---

## Phụ lục B — Tài nguyên tham khảo từ cộng đồng HCM AI Challenge

| Repository                                                                       | Tác giả           | Lĩnh vực       | Kiến trúc cốt lõi     |
| -------------------------------------------------------------------------------- | ------------------- | ---------------- | ------------------------- |
| [TaQuangTu/HCM-AI-Challenge](https://github.com/TaQuangTu/HCM-AI-Challenge)         | TaQuangTu           | Traffic Tracking | YOLOv4 + ROI polygon JSON |
| [chisngooo](https://github.com/chisngooo)                                           | NaiveNotNice (2024) | Event Retrieval  | CLIP + OCR + ASR ensemble |
| [anminhhung/Pipeline_HCM_AI](https://github.com/anminhhung/Pipeline_HCM_AI)         | anminhhung          | Setup pipeline   | Python env packaging      |
| [iamthaoly/UIT-AI-Challenge2020](https://github.com/iamthaoly/UIT-AI-Challenge2020) | iamthaoly           | Colab notebooks  | Google Colab templates    |

**Paper học thuật quan trọng nhất**: RAPID (URA, SOICT 2024) — *"Retrieval-Augmented Parallel Inference Drafting for Text-Based Video Event Retrieval"* — đây là baseline học thuật cao nhất hiện có cho bài toán.

**Models cần tải sẵn** cho FUFU v2:

- `Salesforce/blip2-opt-2.7b` (HuggingFace) — VLM chính
- `VietAI/envit5-translation` — bridging vi↔en
- `Qwen/Qwen-VL-Chat` — captioning + OCR
- `openai/whisper-large-v3` (hoặc `vinai/PhoWhisper`) — ASR tiếng Việt
- `dangvantuan/vietnamese-sbert` — text embedding tiếng Việt
- `AILab-CVC/YOLO-World` — open-vocabulary detection
- `OFA-Sys/chinese-clip-vit-large-patch14` hoặc `M-CLIP` — multilingual CLIP
