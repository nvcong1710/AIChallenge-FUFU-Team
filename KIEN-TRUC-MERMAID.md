# Sơ đồ kiến trúc các hệ thống — Mermaid

> Tài liệu mô tả luồng hoạt động **từng bước** của các hệ thống đã thắng tại HCM AI Challenge + các kiến trúc đề xuất.
> Tất cả diagram đều theo cú pháp Mermaid — render được trên GitHub, VSCode, Obsidian.

## Mục lục

**Các đội đã thi:**
1. [DoppelSearch (2023) — CLIP-only baseline](#1-doppelsearch-2023--clip-only-baseline)
2. [Vi-ATISO (2023) — Microservices 4-channel](#2-vi-atiso-2023--microservices-4-channel)
3. [NaiveNotNice (2024) — Multi-model Ensemble](#3-naivenotnice-2024--multi-model-ensemble)
4. [NewsInsight 2.0 (2024) — BLIP + Elasticsearch + LLM](#4-newsinsight-20-2024--blip--elasticsearch--llm)
5. [VizQuest (2024) — Fusion + Temporal Modeling](#5-vizquest-2024--fusion--temporal-modeling)
6. [SnapSeek (2024) — Milvus + Context-aware](#6-snapseek-2024--milvus--context-aware)
7. [RAPID (URA, SOICT 2024) — BLIP-2 + Translation Bridging](#7-rapid-ura-soict-2024--blip-2--translation-bridging)
8. [TycheVid (Giải Nhất 2024) — Speed-optimized HNSW](#8-tychevid-giải-nhất-2024--speed-optimized-hnsw)

**Kiến trúc đề xuất:**
9. [DMAR — Disentangled Multi-Axis Retrieval](#9-dmar--disentangled-multi-axis-retrieval)
10. [Session Adapter — Online Learning from Clicks](#10-session-adapter--online-learning-from-clicks)
11. [Temporal Event Graph + Synthetic Queries](#11-temporal-event-graph--synthetic-queries)
12. [Combo cuối: BetterDay v2 tổng hợp](#12-combo-cuối-betterday-v2-tổng-hợp)

---

## 1. DoppelSearch (2023) — CLIP-only baseline

> Đơn giản nhất: chỉ CLIP + Faiss. Là baseline mà mọi đội nên hiểu trước khi mở rộng.

```mermaid
flowchart LR
 subgraph OFFLINE["OFFLINE INGEST"]
 V1[Video] --> KF1[Extract Keyframes]
 KF1 --> CLIP1[CLIP ViT-B/32<br/>Image Encoder]
 CLIP1 --> VEC1[Vector 512d]
 VEC1 --> FAISS[(Faiss Index)]
 end

 subgraph ONLINE["ONLINE QUERY"]
 Q1([Query text]) --> CLIP2[CLIP ViT-B/32<br/>Text Encoder]
 CLIP2 --> QVEC[Query Vector]
 QVEC --> SEARCH1[Nearest Neighbor Search]
 FAISS -.-> SEARCH1
 SEARCH1 --> TOPK1[Top-K Frames]
 TOPK1 --> UI1[Operator UI]
 end
```

**Các bước:**
1. **Offline**: tách keyframe → CLIP image encoder → vector → đẩy vào Faiss.
2. **Online**: query text → CLIP text encoder → vector → Faiss search → top-K.
3. **Điểm mạnh**: nhanh, đơn giản. **Yếu**: bỏ qua OCR, ASR, temporal.

---

## 2. Vi-ATISO (2023) — Microservices 4-channel

> Kiến trúc microservice: 4 search engine riêng biệt, operator chọn loại search.

```mermaid
flowchart TB
 subgraph INGEST["INGEST (parallel)"]
 V2[Video] --> KF2[Keyframes]
 KF2 --> P1[CLIP + BEiT-3<br/>image embed]
 KF2 --> P2[CLIP2Video<br/>video embed]
 KF2 --> P3[VFNet<br/>object detect]
 KF2 --> P4[Vietnamese OCR<br/>Toolbox]
 P1 --> I1[(Image Index)]
 P2 --> I2[(Video Index)]
 P3 --> I3[(Object Metadata)]
 P4 --> I4[(OCR Text Index)]
 end

 subgraph QUERY["QUERY ROUTING"]
 OP[Operator] --> CHOICE{Search Type?}
 CHOICE -->|text→image| S1[Image Service]
 CHOICE -->|text→video| S2[Video Service]
 CHOICE -->|object+count| S3[Object Service]
 CHOICE -->|keyword| S4[OCR Service]
 I1 -.-> S1
 I2 -.-> S2
 I3 -.-> S3
 I4 -.-> S4
 S1 --> R[Result Grid]
 S2 --> R
 S3 --> R
 S4 --> R
 end
```

**Các bước:**
1. **Offline**: 4 pipeline song song xử lý keyframe → 4 index riêng.
2. **Online**: operator **chọn loại search** → route đến service tương ứng → trả kết quả.
3. **Điểm mạnh**: linh hoạt, mỗi service tối ưu cho 1 task. **Yếu**: operator phải biết chọn loại search nào.

---

## 3. NaiveNotNice (2024) — Multi-model Ensemble

> "Chia để trị": trích xuất mọi tầng thông tin có thể → fuse khi search.

```mermaid
flowchart TB
 subgraph INGEST3["INGEST (5 modules song song)"]
 V3[Video] --> KF3[Keyframes + Audio]
 KF3 --> M1[OCR Module<br/>biển số/biển hiệu]
 KF3 --> M2[ASR Module<br/>Whisper/PhoWhisper]
 KF3 --> M3[CLIP Encoder<br/>shared latent]
 KF3 --> M4[Image Captioning<br/>BLIP]
 KF3 --> M5[Object + Color<br/>Detection]
 M1 --> META[(Metadata Store)]
 M2 --> META
 M4 --> META
 M5 --> META
 M3 --> VDB[(Vector DB)]
 end

 subgraph QUERY3["QUERY"]
 Q3([Query]) --> P31[CLIP text embed]
 Q3 --> P32[Keyword extract]
 P31 --> SR1[Vector Search]
 P32 --> SR2[Metadata Filter]
 VDB -.-> SR1
 META -.-> SR2
 SR1 --> FUSION[Score Fusion]
 SR2 --> FUSION
 FUSION --> TOPK3[Top-K]
 end
```

**Các bước:**
1. **Ingest**: chạy 5 module song song → CLIP vector vào Vector DB, OCR/ASR/caption/object vào metadata.
2. **Query**: query → CLIP embed (vector path) + keyword extract (metadata path) → fuse score.
3. **Điểm mạnh**: bao quát nhiều modality. **Yếu**: fusion score đơn giản, không có temporal.

---

## 4. NewsInsight 2.0 (2024) — BLIP + Elasticsearch + LLM

> Kiến trúc 2-stage filter: Elasticsearch lọc thô + BLIP rerank tinh + LLM optimize query.

```mermaid
flowchart TB
 subgraph ING4["INGEST"]
 V4[Video] --> KF4[Keyframes]
 KF4 --> BLIP4[BLIP<br/>zero-shot encoder]
 KF4 --> EXT4[Extract metadata<br/>OCR/ASR/caption]
 BLIP4 --> VDB4[(BLIP Vector Index)]
 EXT4 --> ES4[(Elasticsearch)]
 end

 subgraph QRY4["QUERY"]
 Q4([Query VI/EN]) --> LLM4[LLM Query Optimizer<br/>paraphrase + expand]
 LLM4 --> OPT_Q[Optimized Query]
 OPT_Q --> STAGE1[Stage 1:<br/>Elasticsearch Filter]
 ES4 -.-> STAGE1
 STAGE1 --> CAND[~10K Candidates]
 CAND --> STAGE2[Stage 2:<br/>BLIP Vector Rerank]
 VDB4 -.-> STAGE2
 STAGE2 --> STAGE3[Stage 3:<br/>Temporal Mechanism]
 STAGE3 --> TOPK4[Top-K Final]
 end
```

**Các bước:**
1. **Ingest**: BLIP embed → Vector index; metadata text → Elasticsearch.
2. **Query**:
   - LLM nhận query → tinh chỉnh (mở rộng từ đồng nghĩa, thêm ngữ cảnh).
   - Stage 1: ES filter coarse → giảm xuống ~10K candidate.
   - Stage 2: BLIP rerank → giảm xuống top-K.
   - Stage 3: temporal alignment cho chuỗi sự kiện.
3. **Điểm mạnh**: chính xác cao, có LLM giúp xử lý query mơ hồ.

---

## 5. VizQuest (2024) — Fusion + Temporal Modeling

> Top 10. Cốt lõi: fusion 3 modality + temporal cho chuỗi sự kiện.

```mermaid
flowchart TB
 subgraph ING5["INGEST (3 modality)"]
 V5[Video] --> SP5[Split: frames + audio + on-screen text]
 SP5 --> VE5[Visual Encoder<br/>CLIP-like]
 SP5 --> AE5[Audio Encoder<br/>ASR + embed]
 SP5 --> TE5[Text/OCR Encoder]
 VE5 --> VI5[(Visual Index)]
 AE5 --> AI5[(Audio Index)]
 TE5 --> TI5[(Text Index)]
 end

 subgraph QRY5["QUERY"]
 Q5([Query]) --> PAR5[Parallel Search]
 PAR5 --> SV5[Visual Search]
 PAR5 --> SA5[Audio Search]
 PAR5 --> ST5[Text Search]
 VI5 -.-> SV5
 AI5 -.-> SA5
 TI5 -.-> ST5
 SV5 --> R5[Rank Fusion<br/>RRF/Weighted]
 SA5 --> R5
 ST5 --> R5
 R5 --> TM5[Temporal Modeling<br/>shot sequence]
 TM5 --> TOPK5[Top-K Events]
 end
```

**Các bước:**
1. **Ingest**: tách video thành 3 stream (visual/audio/text) → 3 encoder riêng → 3 index.
2. **Query**: query → 3 search song song → fusion rank.
3. **Bước đặc biệt**: tầng **temporal modeling** ráp các shot liên tiếp thành sự kiện.

---

## 6. SnapSeek (2024) — Milvus + Context-aware

> Milvus (vector DB chuyên dụng) + Elasticsearch + "contextual news extraction" để hiểu ngữ cảnh tin tức.

```mermaid
flowchart TB
 subgraph ING6["INGEST + Enrichment"]
 V6[Video] --> KF6[Keyframes]
 KF6 --> EMB6[CLIP Embeddings]
 KF6 --> META6[Metadata Extraction]
 KF6 --> CTX6[Contextual News<br/>Extraction<br/>topic/entity/date]
 EMB6 --> MILVUS[(Milvus<br/>Vector DB)]
 META6 --> ES6[(Elasticsearch)]
 CTX6 --> ES6
 DSET[Dataset<br/>Expansion] -.-> KF6
 end

 subgraph QRY6["QUERY"]
 Q6([Multimodal Query]) --> P6A[Vector path]
 Q6 --> P6B[Text + context path]
 P6A --> MS[Milvus Search]
 P6B --> ESS[ES Search]
 MILVUS -.-> MS
 ES6 -.-> ESS
 MS --> CF[Context-aware Fusion<br/>topic/time match]
 ESS --> CF
 CF --> TOPK6[Top-K]
 end
```

**Các bước:**
1. **Ingest**: thêm bước **enrichment** — trích topic/entity/date từ video (rất hợp với dataset news VN).
2. **Query**: vector path + text/context path → fuse có trọng số ưu tiên match topic và thời gian.
3. **Điểm đặc biệt**: dataset expansion (tự bổ sung dữ liệu ngoài để cải thiện).

---

## 7. RAPID (URA, SOICT 2024) — BLIP-2 + Translation Bridging

> Kiến trúc học thuật mạnh nhất, đã publish. Điểm độc đáo: **dịch tiếng Việt sang tiếng Anh** trước khi đưa vào VLM.

```mermaid
flowchart TB
 subgraph ING7["INGEST"]
 V7[Video] --> KF7[Keyframes]
 KF7 --> BLIP7[BLIP-2<br/>Q-Former]
 KF7 --> YOLO7[YOLOWorldv2<br/>open-vocab detect]
 KF7 --> OCR7[PP-OCR<br/>chữ trên màn]
 KF7 --> ASR7[ASR<br/>hội thoại]
 BLIP7 --> VDB7[(Vector Index)]
 YOLO7 --> KB7[(Knowledge Base<br/>objects)]
 OCR7 --> KB7
 ASR7 --> KB7
 end

 subgraph QRY7["QUERY"]
 Q7([Query tiếng Việt]) --> TRANS[envit5-translation<br/>VI → EN]
 TRANS --> EN_Q[English Query]
 EN_Q --> BLIPT[BLIP-2 Text Encoder]
 BLIPT --> SEARCH7[Vector Search]
 VDB7 -.-> SEARCH7
 SEARCH7 --> CAND7[Candidates]
 CAND7 --> KPI[KPI System<br/>time-based segment]
 KB7 -.-> KPI
 KPI --> ELEV[Elevated<br/>Keyframe Selection]
 ELEV --> TOPK7[Top-K]
 end
```

**Các bước:**
1. **Ingest**: BLIP-2 embed + open-vocab object + OCR + ASR → 2 storage (vector + KB).
2. **Query**:
   - **Translation bridge**: query VI → envit5 → EN (tránh VLM mù tiếng Việt).
   - BLIP-2 encode EN query → vector → search.
   - **KPI**: gom shot theo segment thời gian + tổng hợp text/ASR/OCR.
   - **Elevated keyframe**: chọn frame đại diện tốt nhất cho mỗi segment.
3. **Điểm độc đáo**: translation bridging — pattern thực dụng, **bắt buộc** kế thừa nếu làm dataset VN.

---

## 8. TycheVid (Giải Nhất 2024) — Speed-optimized HNSW

> Bí kíp: không sáng tạo về model, mà **siêu nhanh** nhờ tune HNSW. Latency là vũ khí.

```mermaid
flowchart LR
 subgraph ING8["INGEST"]
 V8[Video] --> KF8[Keyframes]
 KF8 --> EMB8[CLIP / BLIP-2<br/>Embedding]
 EMB8 --> TUNED["Tuned HNSW Index<br/>efConstruction=400<br/>maxConnections=64<br/>efSearch=tuned"]
 TUNED --> VDB8[(Vector DB)]
 end

 subgraph QRY8["QUERY sub-100ms"]
 Q8([Query]) --> CACHE{Embedding<br/>Cache hit?}
 CACHE -->|Yes| QVEC8[Cached Vector]
 CACHE -->|No| TENC8[Text Encoder]
 TENC8 --> QVEC8
 QVEC8 --> ANN[ANN Search<br/>HNSW]
 VDB8 -.-> ANN
 ANN --> TOPK8[Top-K]
 TOPK8 --> PRELOAD[Pre-fetch<br/>thumbnails]
 PRELOAD --> UI8[UI Grid]
 end
```

**Các bước:**
1. **Ingest**: tune HNSW params kỹ (`efConstruction`, `maxConnections`) — không default!
2. **Query**:
   - Cache embedding cho query trùng.
   - HNSW search với `efSearch` tối ưu → mili-giây.
   - Pre-fetch thumbnail (predictive load) → UI render tức thì.
3. **Triết lý**: chất lượng thuật toán giữa top đội sát nhau → ai nhanh hơn, operator thử được nhiều query hơn → thắng.

---

## 9. DMAR — Disentangled Multi-Axis Retrieval

> **Đề xuất 1**: thay vì 1 vector/frame, lưu **6 vector song song** đại diện 6 trục ngữ nghĩa độc lập.

```mermaid
flowchart TB
 subgraph ING9["INGEST — 6 axis song song"]
 V9[Keyframe] --> A1[SCENE<br/>CLIP]
 V9 --> A2[ACTION<br/>Action-CLIP]
 V9 --> A3[ENTITY<br/>NER trên VL caption]
 V9 --> A4[OCR<br/>PaddleOCR-VN]
 V9 --> A5[AUDIO<br/>Whisper-VN]
 V9 --> A6[TEMPORAL<br/>CLIP của ±5s clip]
 A1 --> W1[(Class:Scene)]
 A2 --> W2[(Class:Action)]
 A3 --> W3[(Class:Entity)]
 A4 --> W4[(Class:OCR)]
 A5 --> W5[(Class:Audio)]
 A6 --> W6[(Class:Temporal)]
 end

 subgraph QRY9["QUERY"]
 Q9(["Tổng Bí thư phát biểu<br/>kinh tế trên VTV1"]) --> LLM9[LLM Query<br/>Decomposer]
 LLM9 --> DEC{Weighted<br/>Decomposition}
 DEC -->|w=0.35| S91[Search Entity Axis]
 DEC -->|w=0.30| S94[Search OCR Axis]
 DEC -->|w=0.20| S95[Search Audio Axis]
 DEC -->|w=0.10| S91s[Search Scene Axis]
 DEC -->|w=0.05| S92[Search Action Axis]
 W3 -.-> S91
 W4 -.-> S94
 W5 -.-> S95
 W1 -.-> S91s
 W2 -.-> S92
 S91 --> FUSE9[Weighted Rank Fusion]
 S94 --> FUSE9
 S95 --> FUSE9
 S91s --> FUSE9
 S92 --> FUSE9
 FUSE9 --> TOPK9[Top-K]
 TOPK9 --> SLIDER[UI: axis weight sliders]
 end
```

**Các bước:**
1. **Ingest**: 6 encoder song song cho mỗi keyframe → 6 class riêng trong Weaviate.
2. **Query**:
   - LLM phân rã query thành **weighted compositional query** (entity/ocr/audio/...).
   - Search **6 trục** song song với trọng số.
   - Fuse rank theo trọng số.
3. **UI**: operator có thể chỉnh slider trọng số trục → tinh chỉnh kết quả real-time.

---

## 10. Session Adapter — Online Learning from Clicks

> **Đề xuất 2**: hệ thống **tự học từ click ✓/✗ của operator** trong session, không cần gõ lại query.

```mermaid
sequenceDiagram
 autonumber
 actor Op as Operator
 participant UI
 participant Adapt as Session Adapter
 participant Embed as Text Encoder
 participant Search as Vector Search
 participant VDB as Vector DB

 Op->>UI: Gõ query "người phụ nữ áo đỏ"
 UI->>Embed: encode(query)
 Embed-->>UI: q₀
 UI->>Adapt: init δ = 0
 UI->>Search: search(q₀ + δ)
 Search->>VDB: HNSW nearest
 VDB-->>Search: top-20
 Search-->>UI: 20 ảnh
 UI->>Op: Hiển thị grid

 Op->>UI: Click #3, #7 (áo cam), #12 (đàn ông)
 UI->>Adapt: feedback(pos=[3], neg=[7,12])
 Note over Adapt: L = -log σ(sim(q+δ, v_pos))<br/>+ Σ log σ(sim(q+δ, v_neg))<br/>δ ← δ - η·∇_δ L
 Adapt-->>UI: δ updated

 UI->>Search: search(q₀ + δ)
 Search->>VDB: HNSW nearest
 VDB-->>Search: top-20 (chuẩn hơn)
 Search-->>UI: 20 ảnh mới
 UI->>Op: Grid v2

 UI->>Embed: explain(δ)
 Embed-->>UI: "hệ thống đã hiểu: ưu tiên ĐỎ thuần, đối tượng nữ"
 UI->>Op: Hiển thị giải thích
```

**Các bước (đánh số trong diagram):**
1-5. Initial: operator gõ query, hệ thống embed q₀, search lần đầu, hiển thị.
6-8. Operator click ✓/✗ → adapter tính gradient → cập nhật δ.
9-12. Re-search với q₀+δ → kết quả chính xác hơn.
13-14. (Bonus) LLM giải thích δ đã encode gì → operator hiểu hệ thống "học" gì.

**Điểm đặc biệt**: δ chỉ tồn tại trong session, reset khi sang query mới → an toàn, không drift toàn cục.

---

## 11. Temporal Event Graph + Synthetic Queries

> **Đề xuất 3**: build offline đồ thị thời gian + 15 query tổng hợp/frame để xử lý **chuỗi sự kiện** ("X rồi Y").

```mermaid
flowchart TB
 subgraph ING11["INGEST"]
 V11[Video] --> SHOT[Shot Detection<br/>TransNetV2]
 SHOT --> KF11[Keyframe per shot]
 KF11 --> VLM11[Qwen-VL: sinh 15<br/>synthetic queries<br/>per frame]
 VLM11 --> EMB11[Embed 15 queries]
 EMB11 --> VDB11[(Weaviate<br/>multi-query index)]
 SHOT --> GRAPH[Graph Builder]
 GRAPH --> NODE[Node: shot]
 GRAPH --> EDGE_N[Edge: next]
 GRAPH --> EDGE_C[Edge: cut]
 GRAPH --> EDGE_S[Edge: cont semantic]
 NODE --> NEO[(Neo4j)]
 EDGE_N --> NEO
 EDGE_C --> NEO
 EDGE_S --> NEO
 end

 subgraph QRY11["QUERY chuỗi sự kiện"]
 Q11(["Phỏng vấn ông X,<br/>rồi biểu đồ chứng khoán"]) --> LLM11[LLM Pattern Parser]
 LLM11 --> PATTERN["MATCH a-[:next*1..3]→b<br/>WHERE a.entity~'ông X'<br/>AND a.action~'phỏng vấn'<br/>AND b.scene~'biểu đồ'<br/>AND b.ocr~'VN-Index'"]
 PATTERN --> CYPHER[Run on Neo4j]
 NEO -.-> CYPHER
 CYPHER --> PAIRS[Matching pairs a,b]
 PAIRS --> RANK11[Rank by combined score]
 RANK11 --> SUB[Submit start_a or middle]
 end

 subgraph QRY11B["QUERY đơn lẻ"]
 Q11B([Query đơn]) --> EMB_Q[Embed query]
 EMB_Q --> SEARCH11[Match với 15 synthetic<br/>queries của mỗi frame]
 VDB11 -.-> SEARCH11
 SEARCH11 --> TOPK11[Top-K]
 end
```

**Các bước:**
1. **Ingest part A — Synthetic queries**:
   - Mỗi keyframe → Qwen-VL prompt sinh 15 truy vấn theo style operator.
   - Embed 15 query → lưu Weaviate (link cùng frame_id).
   - **Lý do**: query của user khớp với "query-style" tốt hơn caption "description-style".
2. **Ingest part B — Graph**:
   - Shot detection → node trong Neo4j.
   - Edge: next (cùng video, kế tiếp), cut (chuyển cảnh), cont (cosine ≥0.85).
3. **Query đơn lẻ** (right side): khớp query với 15 synthetic → top-K nhanh.
4. **Query chuỗi** (left side): LLM parse query thành **Cypher pattern** → chạy trên Neo4j → trả cặp (a,b) liền nhau.

---

## 12. Combo cuối: BetterDay v2 tổng hợp

> Gộp 3 kiến trúc đề xuất + kế thừa best practices từ RAPID/TycheVid/NewsInsight.

```mermaid
flowchart TB
 subgraph CLIENT["Frontend (React)"]
 UI12[Search UI]
 SLIDER12[Axis weight sliders]
 FEEDBACK[Feedback buttons]
 TIMELINE[Timeline grid]
 end

 subgraph GW["Spring Boot API Gateway"]
 API[REST + WebSocket]
 end

 subgraph STORE["Storage Layer"]
 WEA[(Weaviate<br/>6-axis multi-vector<br/>+ synthetic queries)]
 NEO12[(Neo4j<br/>Temporal Graph)]
 ES12[(Elasticsearch<br/>OCR/ASR full-text)]
 ADAPT[(Session Adapter Store<br/>in-memory δ)]
 end

 subgraph AISVC["Python AI Service (gRPC)"]
 BLIP_S[BLIP-2 + envit5<br/>translation bridging]
 QWEN_S[Qwen-VL<br/>caption + OCR + grounding]
 WHISP[Whisper-VN<br/>ASR]
 CLIP_S[CLIP-Vi<br/>multilingual]
 LAVI[LaVi-efficient<br/>reranker]
 end

 UI12 --> API
 SLIDER12 --> API
 FEEDBACK --> API
 API --> WEA
 API --> NEO12
 API --> ES12
 API --> ADAPT
 API -.gRPC.-> AISVC

 INGEST_PIPE[Ingest Pipeline] --> AISVC
 AISVC --> WEA
 AISVC --> NEO12
 AISVC --> ES12

 style CLIENT fill:#e1f5ff
 style GW fill:#fff4e1
 style STORE fill:#f0e1ff
 style AISVC fill:#e1ffe1
```

**Các bước hoạt động end-to-end cho 1 query:**

```mermaid
sequenceDiagram
 autonumber
 actor Op as Operator
 participant UI as React UI
 participant API as Spring Boot
 participant AI as Python AI Service
 participant ES as Elasticsearch
 participant WEA as Weaviate
 participant NEO as Neo4j

 Op->>UI: Gõ query VI + chỉnh slider axis
 UI->>API: POST /search {query, weights}

 par Parallel preprocessing
 API->>AI: translate(VI→EN)
 API->>AI: parse temporal pattern?
 end
 AI-->>API: EN query + (optional) pattern

 alt Có temporal pattern
 API->>NEO: Cypher query
 NEO-->>API: matching pairs
 else Query đơn
 par Cascading filter
 API->>ES: coarse filter OCR/ASR
 API->>WEA: 6-axis weighted search
 end
 ES-->>API: candidates A
 WEA-->>API: candidates B
 API->>AI: LaVi rerank top 500
 AI-->>API: top-20
 end

 API-->>UI: Top-K + explanation
 UI->>Op: Grid hiển thị

 Op->>UI: Click #3, #7
 UI->>API: POST /feedback
 API->>API: Update session δ
 API->>WEA: Re-search với q+δ
 WEA-->>API: Top-K v2
 API-->>UI: Update grid
 UI->>Op: Kết quả tốt hơn

 Op->>UI: Click submit #3
 UI->>API: POST /submit {frame_id}
```

**Các bước (numbered):**
1-2. Operator gõ query + chỉnh slider → gửi lên API.
3-5. Song song: translation bridging + parse temporal pattern (nếu có).
6-9. **Nếu temporal**: query Neo4j; **nếu đơn**: cascading filter (ES coarse + Weaviate dense).
10-11. LaVi rerank → top-20.
12-13. UI hiển thị + giải thích.
14-17. Operator click feedback → update δ → re-search nhanh.
18-19. Submit khi đúng.

---

## Ghi chú render Mermaid

- **GitHub**: tự động render khi mở file `.md`.
- **VSCode**: cài extension *Markdown Preview Mermaid Support*.
- **Obsidian**: render native.
- **Online**: paste vào https://mermaid.live để chỉnh sửa.

## Bảng tham chiếu nhanh

| Đội/Kiến trúc | Năm | Stack chính | Đặc trưng |
|---|---|---|---|
| DoppelSearch | 2023 | CLIP + Faiss | Baseline đơn giản |
| Vi-ATISO | 2023 | CLIP+BEiT-3, CLIP2Video, VFNet, OCR | 4 microservice |
| NaiveNotNice | 2024 | CLIP + OCR + ASR + caption + object | Multi-model ensemble |
| NewsInsight 2.0 | 2024 | BLIP + ES + LLM optimizer | 2-stage filter + LLM |
| VizQuest | 2024 | Visual+Audio+Text fusion + temporal | Top 10 |
| SnapSeek | 2024 | Milvus + ES + context news | Context-aware |
| RAPID | 2024 | BLIP-2 + envit5 + YOLOWorldv2 + KPI | Translation bridging, SOICT'24 |
| TycheVid | 2024 | Vector DB + HNSW tuned | Giải Nhất, speed-first |
| **DMAR** | 2026 | 6-axis disentangled | Đề xuất, axis sliders |
| **Session Adapter** | 2026 | Online gradient δ | Đề xuất, neural feedback |
| **Temporal Event Graph** | 2026 | Neo4j + synthetic queries | Đề xuất, chuỗi sự kiện |
