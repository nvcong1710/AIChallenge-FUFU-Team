# Checklist Vấn Đề & Phương Án Giải Quyết — Hệ Thống Phân Tích Nội Dung Video

> Tài liệu này gồm 2 phần: (1) Checklist các vấn đề chia 3 phase ưu tiên, (2) Phương án giải quyết cho từng vấn đề.

---

# PHẦN I — CHECKLIST CÁC VẤN ĐỀ

## 🔴 PHASE 1 — BẮT BUỘC (không có là không chạy được)

### Tiền xử lý video
- [ ] **P1.1** Phân cảnh / shot boundary detection
- [ ] **P1.2** Chọn keyframe đại diện cho mỗi shot
- [ ] **P1.3** Sampling rate hợp lý
- [ ] **P1.4** Chuẩn hoá định dạng đầu vào

### Hiểu nội dung frame
- [ ] **P1.5** Hiểu cảnh tổng thể (scene semantic)
- [ ] **P1.6** Phát hiện đối tượng
- [ ] **P1.7** OCR chữ trên màn (rất quan trọng với dataset VN)
- [ ] **P1.8** Embedding ngữ nghĩa frame

### Hiểu câu truy vấn
- [ ] **P1.9** Xử lý truy vấn đồng nghĩa
- [ ] **P1.10** Truy vấn đa ngôn ngữ VI/EN

### Khớp & xếp hạng
- [ ] **P1.11** Chọn frame khớp nhất từ candidate set
- [ ] **P1.12** Top-K threshold
- [ ] **P1.13** Fusion đa modal

### Lý luận thời gian
- [ ] **P1.14** Định nghĩa "sự kiện"
- [ ] **P1.15** Chọn frame đại diện để submit

### Hiệu năng
- [ ] **P1.16** Storage plan
- [ ] **P1.17** Index memory
- [ ] **P1.18** Latency truy vấn
- [ ] **P1.19** Throughput ingest

### Operator UX
- [ ] **P1.20** Refine query
- [ ] **P1.21** Trình bày kết quả
- [ ] **P1.22** Submit nhanh

### Đặc thù cuộc thi
- [ ] **P1.23** Kiến trúc tổng quát (dataset bị giấu)
- [ ] **P1.24** Tối ưu giới hạn submission
- [ ] **P1.25** Tối ưu giới hạn thời gian

### Đánh giá nội bộ
- [ ] **P1.26** Tạo dev set
- [ ] **P1.27** Metric đại diện

---

## 🟡 PHASE 2 — NÊN CÓ (tạo lợi thế cạnh tranh)

### Tiền xử lý video
- [ ] **P2.1** Khử frame trùng lặp
- [ ] **P2.2** Lọc frame chất lượng kém

### Hiểu nội dung frame
- [ ] **P2.3** Action recognition
- [ ] **P2.4** Thuộc tính vật lý (màu, vị trí)
- [ ] **P2.5** Nhận diện người nổi tiếng / landmark VN
- [ ] **P2.6** ASR speech-to-text
- [ ] **P2.7** Quan hệ không gian giữa đối tượng

### Hiểu câu truy vấn
- [ ] **P2.8** Xử lý query mơ hồ
- [ ] **P2.9** Ngữ cảnh văn hoá VN
- [ ] **P2.10** Tổ hợp logic AND/OR
- [ ] **P2.11** Truy vấn không gian ("góc trái", "phía sau")
- [ ] **P2.12** Truy vấn thời gian / chuỗi ("X rồi Y")
- [ ] **P2.13** Query-document distribution gap

### Khớp & xếp hạng
- [ ] **P2.14** Phát hiện false positive ở similarity cao
- [ ] **P2.15** Diversification top-K
- [ ] **P2.16** Re-ranking 2-stage
- [ ] **P2.17** Cross-modal score calibration

### Lý luận thời gian
- [ ] **P2.18** Khoảng cách thời gian linh hoạt

### Hiệu năng
- [ ] **P2.19** GPU memory rerank
- [ ] **P2.20** Cache query

### Robustness
- [ ] **P2.21** Bù sai số OCR/ASR
- [ ] **P2.22** Occlusion, motion blur, ánh sáng kém
- [ ] **P2.23** Detect đối tượng nhỏ/xa
- [ ] **P2.24** Hạn chế error propagation
- [ ] **P2.25** Phát hiện VLM hallucinate

### Operator UX
- [ ] **P2.26** Giải thích vì sao match
- [ ] **P2.27** Relevance feedback

### Đặc thù cuộc thi
- [ ] **P2.28** Linh hoạt khi format đổi
- [ ] **P2.29** Tối ưu phần cứng giới hạn
- [ ] **P2.30** Hiểu cách BGK đặt query

### Đánh giá
- [ ] **P2.31** Debug pipeline

---

## 🟢 PHASE 3 — MỞ RỘNG (làm khi dư thời gian)

### Hiểu nội dung
- [ ] **P3.1** Đếm đối tượng chính xác
- [ ] **P3.2** Phát hiện cảm xúc / biểu cảm

### Hiểu query
- [ ] **P3.3** Phủ định (negation)
- [ ] **P3.4** Truy vấn so sánh
- [ ] **P3.5** Truy vấn đếm

### Khớp & xếp hạng
- [ ] **P3.6** Long-tail concepts

### Lý luận thời gian
- [ ] **P3.7** Duration matching
- [ ] **P3.8** Causal reasoning

### Hiệu năng
- [ ] **P3.9** Versioning + incremental re-index

### Robustness
- [ ] **P3.10** Khử compression artifacts

### Operator UX
- [ ] **P3.11** Session memory

### Đánh giá
- [ ] **P3.12** Ablation study chính thức

---

# PHẦN II — PHƯƠNG ÁN GIẢI QUYẾT

> Mỗi vấn đề có 1-3 phương án. Phương án được sắp xếp theo thứ tự: đơn giản nhất → mạnh nhất.

## 🔴 PHASE 1

### P1.1 — Phân cảnh / shot boundary detection
- **A. Histogram diff** giữa frame liên tiếp + threshold — đơn giản, không cần model, đủ dùng cho video cut rõ ràng.
- **B. PySceneDetect** (content-aware detector) — mã nguồn mở, hoạt động tốt cho phần lớn trường hợp.
- **C. TransNetV2** (deep learning) — chính xác cao, xử lý được dissolve/fade, cần GPU.

### P1.2 — Chọn keyframe đại diện
- **A. Frame giữa shot** — đơn giản nhất, đủ dùng cho shot ngắn.
- **B. Frame ít blur nhất** trong shot (Laplacian variance cao nhất).
- **C. K-means** trên CLIP embedding các frame trong shot → chọn frame gần centroid nhất.
- **D. Lấy nhiều keyframe** (uniform N frame) nếu shot dài > 5s.

### P1.3 — Sampling rate
- **A. Uniform 1 fps** — baseline, dễ tính storage.
- **B. Adaptive theo motion** (optical flow lớn → sample dày) — cân bằng storage và information.
- **C. Per-shot strategy** — 1-3 frame mỗi shot, độc lập với độ dài.

### P1.4 — Chuẩn hoá định dạng đầu vào
- **A. FFmpeg pre-process** script: resize, fps standardize, codec convert.
- **B. PyAV** (Python binding của libav) — fine control trong code.

### P1.5 — Hiểu cảnh tổng thể (scene)
- **A. CLIP zero-shot classification** với danh sách scene labels (indoor/outdoor/...).
- **B. Places365 pretrained** ResNet/ViT — 365 nhãn scene chuyên dụng.
- **C. VLM caption** rồi NLP extract scene keyword.

### P1.6 — Phát hiện đối tượng
- **A. YOLOv8** (closed-set, fast, 80 class COCO).
- **B. YOLO-World v2** (open-vocabulary, RAPID dùng).
- **C. Grounding DINO** (text-prompted detection — query trực tiếp bằng text).
- **D. OWL-ViT** (open-vocab, transformer).

### P1.7 — OCR chữ trên màn
- **A. EasyOCR** — đa ngôn ngữ, nhẹ, dễ setup.
- **B. PaddleOCR / PP-OCR** — RAPID dùng, mạnh, có pre-trained VN.
- **C. VietOCR** — chuyên tiếng Việt, độ chính xác cao nhất cho dataset VN.

### P1.8 — Embedding ngữ nghĩa frame
- **A. CLIP-ViT-L/14** — chuẩn ngành, nhanh.
- **B. OpenCLIP / M-CLIP** — multilingual, hỗ trợ tiếng Việt.
- **C. SigLIP** — mới, hiệu năng cao hơn CLIP.
- **D. BLIP-2 Q-Former** — mạnh nhất, RAPID dùng, nặng hơn.
- **E. InternVideo2** — chuyên video (không chỉ frame tĩnh).

### P1.9 — Truy vấn đồng nghĩa
- **A. Shared embedding space**: dùng cùng encoder cho query và index → vector đồng nghĩa gần nhau.
- **B. Query expansion**: LLM sinh 5-10 paraphrase → embed tất cả → average.
- **C. HyDE** (Hypothetical Document Embedding): LLM sinh đoạn mô tả giả → embed đoạn đó thay vì query.

### P1.10 — Truy vấn đa ngôn ngữ VI/EN
- **A. Multilingual CLIP** (M-CLIP, OpenCLIP-XLM-R) — 1 model xử lý cả 2.
- **B. Translation bridging**: dịch VI→EN trước khi embed (envit5-translation / NLLB) — pattern RAPID.
- **C. Language detection** → route đến model VN-specific hoặc EN-specific.

### P1.11 — Chọn frame khớp nhất
- **A. Cosine similarity rank** — chuẩn, đơn giản.
- **B. Cross-encoder re-rank** (BERT-style) — tốt hơn nhưng chậm hơn.
- **C. LLM judge**: prompt VLM "trong các frame sau, frame nào khớp query nhất?" — chính xác nhất, đắt nhất.

### P1.12 — Top-K threshold
- **A. Fixed K** (vd K=20) — đơn giản.
- **B. Dynamic K**: cắt khi gap rank K và K+1 lớn — adaptive.
- **C. Certainty threshold** (vd ≥0.7) — bỏ qua kết quả yếu, có thể trả 0 nếu không có gì khớp.

### P1.13 — Fusion đa modal
- **A. Weighted sum** score: `w1·CLIP + w2·OCR + w3·ASR` — tuning trọng số trên dev set.
- **B. Reciprocal Rank Fusion (RRF)** — không cần calibrate score, chỉ dùng rank.
- **C. Learn-to-rank** (LightGBM/XGBoost) — học fusion từ dev set.

### P1.14 — Định nghĩa "sự kiện"
- **A. Đơn vị = keyframe** — đơn giản, mỗi frame là 1 ứng viên.
- **B. Đơn vị = shot** — gom keyframe theo shot boundary.
- **C. Đơn vị = clip 5-10s** — sliding window có overlap.

### P1.15 — Chọn frame đại diện để submit
- **A. Frame giữa shot** — đơn giản.
- **B. Frame max similarity** với query trong shot.
- **C. Median frame** của top-K trong cùng shot.

### P1.16 — Storage plan
- **A. Local file system** + Parquet cho metadata — đơn giản.
- **B. MinIO/S3** cho thumbnail + Parquet cho metadata — scalable.
- **C. Tách layer**: Vector DB (Weaviate/Qdrant) + Object storage (S3) + RDBMS (metadata).

### P1.17 — Index memory
- **A. HNSW** (Weaviate default) — chuẩn, đánh đổi memory để có recall cao.
- **B. IVF-PQ** (Faiss) — memory < 1/10 HNSW, recall giảm nhẹ.
- **C. DiskANN** — index không vừa RAM, đẩy ra SSD.

### P1.18 — Latency truy vấn
- **A. Tune HNSW** params (`efSearch`, `maxConnections`) — bí kíp TycheVid.
- **B. Embedding cache** cho query phổ biến.
- **C. Quantize embedding** (PQ, scalar quant) — nhỏ hơn, nhanh hơn.
- **D. GPU inference** cho text encoder.

### P1.19 — Throughput ingest
- **A. Batch inference** 32-128 frame/batch.
- **B. Data parallel** trên nhiều GPU.
- **C. Async pipeline** (decode → embed → index) chạy song song.

### P1.20 — Refine query
- **A. Text box** cho phép sửa & search lại.
- **B. Auto-suggest** từ query đã thử trong session.
- **C. Query history** với thumbnail preview.

### P1.21 — Trình bày kết quả
- **A. Grid thumbnail** 4-5 cột.
- **B. Hover preview** video segment (3-5s clip).
- **C. Hiển thị similarity score** + breakdown per modality.

### P1.22 — Submit nhanh
- **A. Keyboard shortcut** (Enter để submit, số 1-9 để chọn top-K).
- **B. 1-click submit** trên thumbnail.
- **C. Auto-fill timestamp** từ frame metadata.

### P1.23 — Kiến trúc tổng quát (dataset bị giấu)
- **A. Modular pipeline**: mỗi modality 1 service riêng, có thể bật/tắt.
- **B. Config-driven** (YAML) cho threshold, weight, model path.
- **C. Fallback chain**: nếu modality A fail → dùng B.

### P1.24 — Tối ưu giới hạn submission
- **A. Show top-K rộng** (K=20-50) trước khi submit.
- **B. Re-rank kỹ** bằng VLM ở tầng cuối.
- **C. Operator double-check** với thumbnail kích thước lớn.

### P1.25 — Tối ưu giới hạn thời gian
- **A. Optimize latency** (như P1.18).
- **B. Predictive pre-fetch** lúc operator đang gõ.
- **C. Streaming results** (trả top-K nhanh trước, rerank sau).

### P1.26 — Tạo dev set
- **A. Tải video public** (YouTube VN, VTV archive) + tự viết 100-200 query.
- **B. Dùng query mẫu** từ các mùa HCM AI Challenge trước.
- **C. LLM tự sinh query** từ ground-truth caption — scale lên 1000+ query nhanh.

### P1.27 — Metric đại diện
- **A. Recall@1, Recall@5, Recall@10** — chuẩn IR.
- **B. MRR** (Mean Reciprocal Rank).
- **C. nDCG** nếu có relevance score liên tục.
- **D. Custom metric** mô phỏng cách BTC chấm (top-N submission).

---

## 🟡 PHASE 2

### P2.1 — Khử frame trùng lặp
- **A. Perceptual hash (pHash)** — siêu nhanh, đơn giản.
- **B. CLIP cosine threshold** (≥0.95 = trùng) — chính xác hơn pHash.

### P2.2 — Lọc frame chất lượng kém
- **A. Laplacian variance** < threshold = blur.
- **B. Histogram analysis** cho quá tối / quá sáng.
- **C. VLM zero-shot**: "is this image clear?".

### P2.3 — Action recognition
- **A. SlowFast / MViT** — video action model truyền thống.
- **B. VideoMAE** — masked autoencoder cho video.
- **C. Action-CLIP** — extend CLIP cho action.
- **D. VLM trên clip ngắn** — generic, đắt.

### P2.4 — Thuộc tính vật lý (màu, vị trí)
- **A. Color histogram** per bbox.
- **B. CLIP zero-shot classify** thuộc tính ("red car", "blue shirt").
- **C. Grounded VLM** ("the red car at the left").

### P2.5 — Nhận diện người nổi tiếng / landmark VN
- **A. Face recognition** + CSDL người VN tự build (lãnh đạo, KOL).
- **B. Landmark detection** với Google Landmarks pretrained.
- **C. VLM** với prompt context VN ("Đây có phải Tổng Bí thư...?").

### P2.6 — ASR speech-to-text
- **A. Whisper large-v3** — multilingual, mạnh.
- **B. PhoWhisper** (VinAI) — chuyên tiếng Việt, accuracy cao hơn cho VN.
- **C. wav2vec2** fine-tuned VN.

### P2.7 — Quan hệ không gian giữa đối tượng
- **A. Bbox geometry**: tính left/right/above/below từ tọa độ bbox.
- **B. Scene graph generation** (RelTR, Neural Motifs).
- **C. Grounded VLM** với prompt spatial.

### P2.8 — Xử lý query mơ hồ
- **A. LLM detect ambiguity** → hỏi clarifying question.
- **B. Trả top-K diverse** → operator chọn.
- **C. Auto-add modifier** (vd: "người đàn ông" → "người đàn ông trưởng thành").

### P2.9 — Ngữ cảnh văn hoá VN
- **A. Knowledge base** entities VN (wiki, từ điển tự build).
- **B. LLM fine-tuned** cho ngữ cảnh VN (PhoGPT, VinaLLaMA).
- **C. NER tiếng Việt** (PhoBERT-NER, VnCoreNLP) → expand query với context.

### P2.10 — Tổ hợp logic AND/OR
- **A. LLM parse** query → boolean tree → evaluate trên candidates.
- **B. Multi-vector retrieval** với intersection (AND) / union (OR).
- **C. Symbolic + vector hybrid**: filter cứng + similarity mềm.

### P2.11 — Truy vấn không gian
- **A. Grounding model** (Qwen-VL grounding, Grounding DINO) trả bbox.
- **B. Bbox post-filter** sau retrieval (rank cao hơn nếu bbox đúng vị trí).

### P2.12 — Truy vấn thời gian / chuỗi
- **A. Time-based segment search** (RAPID KPI style) — gom shot liền nhau theo query.
- **B. Temporal graph** (frame → next → frame) + Cypher-like pattern matching.
- **C. LLM parse query** thành temporal pattern → execute.

### P2.13 — Query-document distribution gap
- **A. HyDE**: LLM sinh hypothetical document → embed cái đó.
- **B. doc2query**: với mỗi frame, sinh K query mô tả → embed cả K query → index.
- **C. Query expansion** bằng LLM lúc search.

### P2.14 — Phát hiện false positive ở similarity cao
- **A. VLM verifier**: "does this image contain X?" trên top-K.
- **B. Multi-modal cross-check**: nếu query có chữ → OCR phải match.
- **C. Ensemble**: nhiều model embed → consensus.

### P2.15 — Diversification top-K
- **A. MMR** (Maximal Marginal Relevance) — balance relevance vs novelty.
- **B. Cluster top-100** → chọn 1 per cluster.
- **C. Shot-based dedup** (1 frame per shot trong top-K).

### P2.16 — Re-ranking 2-stage
- **A. Tier 1 CLIP/HNSW** → top 500 → **Tier 2 BLIP-2** → top 20.
- **B. Tier 1 + cross-encoder** thay BLIP-2 (nhẹ hơn).
- **C. Tier 1 + LLM judge** (đắt, cao chính xác).

### P2.17 — Cross-modal score calibration
- **A. Min-max normalize** trên dev set.
- **B. Sigmoid/softmax** với temperature tuning.
- **C. Learn calibration** bằng isotonic regression.

### P2.18 — Khoảng cách thời gian linh hoạt
- **A. Window expansion**: search frame X, mở rộng ±N giây.
- **B. LLM parse "ngay sau" = ±5s, "vài phút sau" = ±300s**.

### P2.19 — GPU memory rerank
- **A. Dynamic batching** theo VRAM khả dụng.
- **B. Quantize VLM** (INT8 / INT4) — giảm 2-4× memory.
- **C. LaVi-style efficient VLM** — feature modulation thay vì concat token.

### P2.20 — Cache query
- **A. LRU cache** cho top-K results.
- **B. Embedding cache** cho query string trùng.
- **C. Approximate cache**: query gần giống → trả kết quả gần đúng.

### P2.21 — Bù sai số OCR/ASR
- **A. Fuzzy match** (Levenshtein distance).
- **B. Phonetic match** cho ASR (Soundex / Metaphone VN).
- **C. Spelling correction** tiếng Việt (Hunspell-VN, language model).

### P2.22 — Occlusion, motion blur, ánh sáng kém
- **A. Multi-frame fusion**: gom embedding của 3-5 frame liền nhau.
- **B. Skip frame quá blur** (Laplacian variance < threshold).
- **C. Enhancement pre-process** (denoise, contrast).

### P2.23 — Detect đối tượng nhỏ / xa
- **A. Multi-scale detection** (FPN).
- **B. Crop & zoom + re-detect** trên patch.
- **C. Higher resolution input** (1280p thay vì 640p).

### P2.24 — Hạn chế error propagation
- **A. Mỗi tầng output confidence** → downstream weight.
- **B. Fallback path** khi tầng A fail.
- **C. End-to-end model** thay vì pipeline (giảm error compounding).

### P2.25 — Phát hiện VLM hallucinate
- **A. Cross-verify** caption với object detection (object có thật không?).
- **B. Multi-VLM consensus** (BLIP-2 + Qwen-VL + LLaVA cùng caption).
- **C. Confidence threshold** trên VLM logits.

### P2.26 — Giải thích vì sao match
- **A. Highlight phần khớp** trong caption / OCR text.
- **B. Show component scores** (CLIP=0.8, OCR=0.6, ASR=0.4).
- **C. Heatmap attention** (Grad-CAM trên CLIP).

### P2.27 — Relevance feedback
- **A. Rocchio** classic — cộng/trừ trung bình vector.
- **B. Online gradient** (Session Adapter — δ vector cập nhật theo click).
- **C. LLM re-formulate query** dựa vào click history.

### P2.28 — Linh hoạt khi format đổi
- **A. Config-driven pipeline** (YAML), không hard-code.
- **B. Hot-swap modality** qua API.
- **C. Versioned configs** cho từng round thi.

### P2.29 — Tối ưu phần cứng giới hạn
- **A. Distill model nhỏ hơn** (TinyCLIP, DistilBERT).
- **B. CPU inference fallback** với ONNX.
- **C. Quantize aggressively** (INT4, GPTQ).

### P2.30 — Hiểu cách BGK đặt query
- **A. Phân tích query mùa trước** → identify patterns (cấu trúc câu, length, style).
- **B. Fine-tune query encoder** trên query mẫu.
- **C. Few-shot prompt** LLM mô phỏng style BGK.

### P2.31 — Debug pipeline
- **A. Logging per stage** (timestamp, input, output, score).
- **B. Trace UI** để xem từng tầng góp gì vào kết quả cuối.
- **C. Diff hai version** (A/B compare).

---

## 🟢 PHASE 3

### P3.1 — Đếm đối tượng chính xác
- **A. Object detector** + count bbox sau NMS.
- **B. Specialized counting model** (CSRNet cho crowd).

### P3.2 — Phát hiện cảm xúc / biểu cảm
- **A. Face emotion classifier** (FER+, AffectNet).
- **B. VLM zero-shot** ("what emotion is this person showing?").

### P3.3 — Phủ định (negation)
- **A. LLM parse**: tách positive + negative parts.
- **B. Score** = sim(positive) − α·sim(negative).
- **C. Filter cứng**: loại frame có object negative.

### P3.4 — Truy vấn so sánh
- **A. LLM parse** → run multiple queries → compare attributes.
- **B. Rank theo thuộc tính** sau retrieval (vd "cao nhất" → sort by height bbox).

### P3.5 — Truy vấn đếm
- **A. Như P3.1** + filter theo count đúng.

### P3.6 — Long-tail concepts
- **A. Few-shot prompting** VLM với ví dụ.
- **B. Manual seed examples** vào index.

### P3.7 — Duration matching
- **A. Tag shot/segment với duration** → filter theo duration.

### P3.8 — Causal reasoning
- **A. LLM chain-of-thought** trên top-K segment.
- **B. Temporal graph** + causal edges (đắt, khó).

### P3.9 — Versioning + incremental re-index
- **A. Diff embedding** → chỉ re-index frame thay đổi.
- **B. Model versioning** với tagged collections trong vector DB.

### P3.10 — Khử compression artifacts
- **A. Denoise pre-process** (BM3D, neural denoiser).
- **B. Super-resolution** (ESRGAN, Real-ESRGAN) cho frame quan trọng.
- **C. Skip artifact-heavy frames**.

### P3.11 — Session memory
- **A. LLM context window** cho session.
- **B. Embedding-based session state** (gom query trong session thành 1 vector).

### P3.12 — Ablation study chính thức
- **A. Run experiments** với từng modality on/off.
- **B. Report contribution percentage** trong bài thuyết trình.

---

## Ghi chú

- Phương án **A** thường là baseline đơn giản nhất, đủ cho prototype.
- Phương án **B/C** là nâng cấp, chọn theo budget thời gian / GPU.
- Một số vấn đề (vd P1.13 fusion, P2.16 re-ranking) có thể **kết hợp nhiều phương án** thay vì chọn 1.
- Tên model / công cụ cụ thể là **gợi ý**, có thể thay bằng tương đương nếu có model mới hơn.
