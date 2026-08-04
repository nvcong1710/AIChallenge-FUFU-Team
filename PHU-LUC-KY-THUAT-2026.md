# PHỤ LỤC KỸ THUẬT THỰC THI — FUFU @ HCM AI Challenge 2026

> **File này là gì:** lớp "code được" của [KE-HOACH-SANG-TAO-2026.md](KE-HOACH-SANG-TAO-2026.md). Bản kế hoạch sáng tạo nói *làm gì & vì sao*; file này nói *làm thế nào* — công thức chính xác, siêu tham số, pseudocode, số benchmark — rút từ vòng deep-research thứ 2 (5 luồng × ~5 góc, ~115 nguồn, đa số là primary: arXiv + GitHub chính chủ + trang Codabench/gov.vn chính thức).
>
> Đọc cùng: [KE-HOACH-SANG-TAO-2026.md](KE-HOACH-SANG-TAO-2026.md) (chiến lược N1–N8) · [RESEARCH-PLAN.md](RESEARCH-PLAN.md) (menu A1–F2 nền) · [PROJECT-CONTEXT.md](PROJECT-CONTEXT.md) (code hiện tại).
>
> Cập nhật: 2026-06-27.

---

## 0. Trạng thái xác minh (đọc trước — quan trọng)

Vòng research thứ 2 **thu thập dữ liệu thành công** (search + fetch + trích claim từ nguồn gốc), nhưng **bước verify đối-kháng bị một giới hạn rate của API chặn giữa chừng** (reset 3:10am Asia/Bangkok). Hệ quả: report tự động ghi "all claims refuted / inconclusive" — **đây là báo động giả**: phiếu `0-0 (3 abstain)` nghĩa là *không ai bỏ phiếu được*, KHÔNG phải claim sai. Sau khi rate-limit hồi phục, tôi **tự fetch lại các sự thật then-chốt** để xác nhận.

| Mức | Nghĩa | Áp cho |
|---|---|---|
| ✅ **Đã xác minh** | Tôi tự fetch lại nguồn gốc, hoặc ≥2 nguồn độc lập trùng khớp | Chấm điểm AIC 2025; công thức QB-Norm + β; nguồn gốc DANTE; KIS phạt thời gian/nộp-sai |
| 🟡 **Nguồn gốc đáng tin** | Trích từ arXiv/GitHub chính chủ (workflow), chưa tôi tự fetch lại | Recipe CLIP-LoRA; gains LaCLIP/VeCLIP/MLLM-caption; benchmark CG-DETR/TR-DETR; WER PhoWhisper; số OCR; guideline FAISS; MADTempo |
| 🟠 **Hợp lý, cần chốt với thể lệ 2026** | Cụ thể, từ primary, nhưng có thể đổi theo mùa | Time-limit vòng 600/1200/1600s; vô địch 2025 OpenCubee/UIT; cú pháp dòng submit TRAKE |

> Quy tắc thép (giữ từ kế hoạch gốc): **trước khi code một công thức, mở lại paper/repo gốc lấy đúng ký hiệu** — danh sách URL ở §9. Đừng code theo trí nhớ hay theo bảng này.

---

## 1. SỰ THẬT CUỘC THI ĐÃ KHOÁ — và nó đổi chiến lược thế nào

### 1.1 Chấm điểm (✅ xác minh, 2 nguồn độc lập: Codabench + arXiv 2603.02888)

> **`Mean of Top-k R-Scores`, trung bình qua k ∈ {1, 5, 20, 50, 100}.**

Đây **không phải** đua-nộp-trước kiểu VBS live thuần. Đây là metric **chất lượng xếp hạng**: với mỗi truy vấn, lấy max R-Score trong top-k cho từng ngưỡng k rồi trung bình 5 ngưỡng. Hệ quả số học cực quan trọng:

- Đưa item đúng lên **rank 1** → ăn điểm đầy ở cả 5 ngưỡng.
- Item đúng ở **rank 6–20** → mất sạch bucket k=1 và k=5 (mất 40% điểm câu đó).
- Item đúng ở **rank 51–100** → chỉ còn bucket k=100 (còn ~20%).

→ **Toàn bộ giá trị nằm ở việc kéo item đúng từ rank ~6–50 lên top-1/5.** Đây chính xác là thứ **N1 (QB-Norm, +2–6 điểm R@1)**, **N7 (rerank)** và **N3 (HyDE)** làm. Metric này *được sinh ra* cho combo của ta.

R-Score theo task:
- **Textual-KIS:** đúng nếu predicted video khớp tên video ground-truth **và** frame index nằm trong khoảng frame đúng.
- **VQA:** ngoài khớp frame, **đáp án text phải khớp chính xác** reference.
- **TRAKE:** R-Score = tỉ lệ frame dự đoán nằm trong đoạn thời gian ground-truth (có cửa sổ tolerance). 🟠 Cú pháp dòng submit báo cáo là `<Video name>, <Frame ID_1>, …, <Frame ID_N>`.

### 1.2 Dữ liệu (✅): ~**250 GB** video **broadcast + documentary**, đa miền — news, education, travel, culture, sports. **Rộng hơn hẳn 2024** (1.471 video tin tức / 328h). Tổ chức **cung cấp sẵn** keyframe + object detection + **CLIP features** + metadata (theo mùa 2023; khả năng cao giữ).

### 1.3 Hạ tầng & vòng (🟠): nộp/chấm trên **Codabench** (Group A; 2024 còn dùng CodaLab). 3 vòng online, **public leaderboard = 50% ground-truth, private/final = 100%**. Time-budget mỗi vòng (báo cáo 600/1200/1600s). KIS truyền thống (gov.vn ✅): điểm tối đa bằng nhau mỗi câu (vd 100đ), **giảm theo thời gian tìm + số lần nộp sai**.

### 1.4 Bảng "sự thật → đổi gì trong kế hoạch"

| Sự thật đã khoá | Hệ quả thực thi |
|---|---|
| Metric = Mean Top-k R-Score k∈{1,5,20,50,100} | **F1 phải đo ĐÚNG metric này**, không phải recall@k chung. Tối ưu kéo rank 6–50 → 1–5. |
| Rank-1/5 ăn điểm nặng nhất | **N1 + N7 + N3 lên ưu tiên cao nhất** (đều nhắm top-rank). |
| Dataset đa miền (không chỉ news) | **N2 domain-adapt càng giá trị**; OCR/ASR phải chịu được nội dung đa dạng. |
| Tổ chức phát CLIP features sẵn | Cân nhắc dùng để **ensemble** (C1) thay vì chỉ tự trích — có encoder "miễn phí". |
| Public=50%/private=100% GT | **Đừng overfit public leaderboard**; giữ tập eval riêng. |
| VQA cần đáp án text khớp chính xác | Cần module QA (E4) đọc frame+ASR → sinh đáp án ngắn, chuẩn hoá khớp. |
| TRAKE = nhiều frame/đoạn, R-Score theo tỉ lệ trúng | **N4 + DANTE** (DP align) trực tiếp tối ưu metric này. |

---

## 2. N1 — Querybank-Norm + khử hubness (đòn rẻ nhất, làm NGAY sau F1)

### 2.1 Dynamic Inverted Softmax (✅ công thức + β tự xác nhận từ repo `ioanacroi/qb-norm`)

Cho query `q`, gallery item `j`, similarity thô `s_q(j)`. Định nghĩa **probe vector** của gallery `j` trên querybank N mẫu: `p_j(i) = sim(querybank_i, gallery_j)`. Khi đó:

```
φ_q(j) = exp(β·s_q(j)) / (1ᵀ · exp[β·p_j])        # Inverted Softmax (IS)
```

**Dynamic IS (DIS) — "do no harm", bản khuyến nghị:** chỉ áp chuẩn hoá khi **top-1 của query rơi vào "gallery activation set" A** (tập item là top-k của ít nhất 1 mẫu querybank, k=1); ngược lại giữ `s_q(j)` thô.

```
φ_q(j) = exp(β·s_q(j))/(1ᵀexp[β·p_j])   nếu argmaxₗ s_q(l) ∈ A
       = s_q(j)                          ngược lại
```

- **β (inverse temperature):** mặc định **20**; CLIP2Video dùng **1/1.99**; CLIP4Clip trên LSMDC dùng **0.8**. → với SigLIP của ta, **tune β trên F1** (quét 1, 5, 10, 20, 40).
- **Querybank:** lấy từ **query-modality** (text). KHÔNG cần test queries. Dùng được: caption VLM của frame + synthetic query (D1/N3) + đề mùa cũ. Cỡ **vài nghìn → 60k**, càng lớn càng tốt (bão hoà). Cần 2 ma trận similarity tiền-tính: (train/querybank captions × gallery) và (test query × gallery).
- **Gains đã xác minh (text→video):** MSR-VTT R@1 **14.9→17.3** (+2.4), MSVD **25.4→28.9** (+3.5), DiDeMo **21.6→24.2** (+2.6). Trên MSCOCO CLIP: +4.5 (🟡). Trên SigLIP-news của ta: **phải đo** — nhưng cơ chế (hubness) gần như chắc tồn tại.
- Implement chuẩn: file `dynamic_inverted_softmax.py` trong repo gốc.

### 2.2 Khử hubness bổ trợ (rẻ, làm cùng)

- **All-but-the-top (ABTT)** (🟡, Mu & Viswanath ICLR'18): (1) trừ vector trung bình `μ` khỏi mọi embedding; (2) PCA, **chiếu bỏ D thành phần trội nhất**. Quy tắc **D ≈ d/100** → với SigLIP d=768 ⇒ **D ≈ 7–8**. Áp lên tập embedding **image** lúc ingest (1 lần).
- **CSLS** (🟡, Conneau et al.): `CSLS(x,y) = 2·cos(x,y) − r_T(x) − r_S(y)`, với `r(·)` = cosine trung bình tới K láng giềng. Nâng điểm vector cô lập, hạ điểm hub. Thay cosine ở bước ranking. (Tốn hơn IS chút; thử nếu IS chưa đủ.)

> **Verify N1:** A/B trên F1 đo **Mean Top-k R-Score**: baseline → +IS → +DIS(β tune) → +ABTT. Đây là thí nghiệm đầu tiên sau F1. Downside ~0 (tắt được).

---

## 3. N2 — Domain-adapt SigLIP bằng LoRA (đốt GPU, edge cấu trúc)

### 3.1 Recipe LoRA (🟡, từ CLIP-LoRA arXiv 2405.18541)

- **Áp LoRA vào ma trận q,k,v của attention ở CẢ hai tháp** (vision + text) → tốt nhất.
- Rank **r = 2** (CLIP-LoRA few-shot); với dữ liệu domain lớn hơn của ta, thử **r = 8–16**.
- LR **2e-4**, batch **32**, cosine scheduler, **dropout p=0.25** ở input mỗi module LoRA (chống overfit), budget ~`500·(N/K)` iter (few-shot; ta nhiều dữ liệu hơn → theo early-stop).
- Loss: contrastive (InfoNCE/sigmoid kiểu SigLIP) trên cặp (frame, caption).

### 3.2 Cảnh báo & cách chặn downside (🟡, quan trọng)

- **LP-FT pitfall** (arXiv 2202.10054): full fine-tune **làm méo feature pretrained**, OOD **thấp hơn 7%** so với linear-probe. → **không full-FT**; LoRA rank thấp + LR nhỏ + early-stop trên F1.
- **Misalignment khi tune tháp ảnh** (arXiv 2409.01936): nếu chỉ tune image tower kiểu metric-learning → **hỏng alignment text↔image**. Cách an toàn: **đóng băng text tower** (LiT-style) hoặc tune nhẹ cả hai bằng contrastive (giữ alignment).
- **Caption tự sinh nhiễu** (arXiv 2212.07086 NLIP): lọc cặp nhiễu bằng GMM 2-thành-phần trên loss ITC; hoặc đơn giản lọc theo độ tin caption.
- **LUÔN giữ base làm thành viên ensemble** (C1) → adapter kém cũng không tụt.

### 3.3 Caption-augmentation có cơ sở (🟡, gains thực đo)

- **LaCLIP** (NeurIPS'23): viết lại caption bằng LLM, sample ngẫu nhiên gốc/viết-lại lúc train. Gain zero-shot ImageNet: CC3M **+5.7**, CC12M **+8.2** (gain co lại khi data lớn/sạch).
- **MLLM-caption** (arXiv 2311.18765): thêm 4 caption sinh bởi MLLM/ảnh (Qwen-VL, LLaVA…) → MSCOCO t2i R@1 **18.3→33.2 (+14.9)**, Flickr t2i **28.4→58.0 (+29.6)** (pretrain CC3M). **Lưu ý "text shearing": cắt ~30 token, giữ mệnh đề đầu** — caption dài >30 token làm tụt.
- **VeCLIP** (arXiv 2310.07699): COCO i2t R@1 **24.52→47.78 (+23.26)** ở quy mô 12M.
- **CLIPS** (arXiv 2411.16828): **"inverse effect" — feed caption NGẮN cho text encoder retrieve TỐT hơn caption đầy đủ.** → khi sinh caption train/HyDE, **ưu tiên ngắn gọn**.

> **Verify N2:** F1 trước/sau, **không được regress**. Giữ checkpoint base; chỉ promote adapter nếu thắng rõ trên tập held-out (query không dùng khi train). Train vài giờ trên 3090 hoặc thuê A100 Vast.ai.

---

## 4. N3 — Visual-HyDE (sinh caption giả khớp phân phối index)

- Nền: **HyDE** (repo `texttron/hyde`). LLM sinh "tài liệu giả định" = **caption-frame giả định** theo đúng văn phong caption VLM của index → encode bằng SigLIP text tower → search.
- **Tận dụng phát hiện CLIPS (§3.3): caption NGẮN retrieve tốt hơn** → prompt LLM sinh caption **ngắn, đặc tả** (~20–30 token), không lan man.
- Sinh **nhiều** caption giả → **mean-pool embedding** (giảm phương sai), rồi **fuse với query gốc** (đừng thay hẳn): `q_final = norm(α·q_orig + (1−α)·q_hyde)`, tune α trên F1.
- Cache theo query; sinh sẵn cho đề mẫu. Latency thêm ~vài trăm ms → chấp nhận được offline-prep.

> **Verify N3:** F1 phân nhóm loại query; HyDE lợi nhất ở **query visual mơ hồ**, ít lợi ở query OCR/entity.

---

## 5. N4 — Temporal & TRAKE: DANTE (DP) + event-index

### 5.1 DANTE — DP align cho TRAKE (✅ nguồn gốc; 🟡 công thức từ arXiv 2512.13169, AIO_Owlgorithms AIC'25)

Cho N mô tả sự kiện có thứ tự, keyframe `t`, `S[i,t]` = cosine(event embedding `u_i`, keyframe `E[t]`), `λ` = phạt khoảng cách thời gian (ép thứ tự tăng dần):

```
DP[i,t] = S[i,t] + max_{τ ∈ [s_v, t-1]} ( DP[i-1, τ] − λ·(t − τ) )
```

Tối ưu **O(N·T)** bằng **running-max** (bỏ vòng max trong):
```
running_max = max(running_max, DP[i-1, t-1] + λ·(t-1))
DP[i, t]    = S[i, t] + running_max − λ·t
```
Điểm cuối mỗi video: `DANTE[v] = max_{t∈[s_v,e_v]} DP[N, t]`; **backtrack** để dựng lại chuỗi keyframe tối ưu → đúng định dạng submit TRAKE (nhiều frame ID). Per-video phức tạp `O(N·(e_v − s_v + 1))`.

### 5.2 Biến thể cộng-dồn-có-suy-giảm (🟡, arXiv 2512.12929 MADTempo / 2512.12935)

```
SS = Σ_{i=1..K} s_i · e^{−α·(t_i − t_{i-1})}        # additive + exponential time-decay
```
Giải bằng **beam search** (giữ top-B chuỗi từng bước), ràng buộc cùng video + thời gian tăng dần + gap tối đa τ; phức tạp `O(B·K·M)`. Tác giả lập luận **cộng (additive) bền hơn nhân (product)** — product quá nhạy với 1 transition điểm thấp. (MADTempo dùng CLIP-Laion, không SigLIP.)

→ **Khuyến nghị:** dùng **DANTE DP (exact, O(NT))** làm lõi TRAKE; giữ beam-search-decay làm fallback/đối chiếu.

### 5.3 Event-index dựng sẵn (mở rộng "đốt ingest")

Lúc ingest, pool keyframe theo cửa sổ shot liên tiếp (2–5) → **event embedding** vào FAISS index thứ 2. Query TRAKE/temporal match event-vectors trước → DANTE chỉ chạy trên tập ứng viên nhỏ, sạch. Pooling: mean hoặc attention.

### 5.4 Nếu cần moment-grounding model riêng (🟡, benchmark QVHighlights)

- **CG-DETR**: MR R1@0.5 **65.4**, R1@0.7 48.4, mAP avg 42.9 (> QD-DETR, UniVTG, Moment-DETR). Charades-STA R1@0.5 58.4.
- **TR-DETR**: QVHighlights R1@0.5 **64.66**, R1@0.7 48.96.
→ Chỉ cân nhắc nếu cần localize chính xác hơn trong-video; nặng hơn DANTE. Để sau.

> **Verify N4:** eval riêng F1-TRAKE đo **R-Score = tỉ lệ frame trúng đoạn GT** (đúng metric BTC).

---

## 6. N5 / N6 / N7 — fusion động, submit, rerank

### 6.1 N5 — fusion routing động
Phân loại loại query (OCR-heavy / ASR-heavy / visual / entity) bằng luật + parse LLM (B1 đã gọi LLM sẵn → "free") → đặt trọng số fusion theo từng query; "không chắc" → về trọng số tĩnh. Học trọng số bằng logistic regression trên F1. (Giải luôn C5.)

### 6.2 N6 — chính sách submit **(REFRAME theo metric Codabench)**
Vì 2025 chấm **Mean Top-k R-Score** (xếp hạng, không đua-nộp-trước thuần), N6 đổi khung:
- **Rank-aware:** tối đa hoá *thứ hạng* item đúng (đẩy lên top-1/5) — đây là việc của N1/N7, N6 hỗ trợ chọn thứ tự đưa vào danh sách nộp.
- **Throughput triage:** trong **time-budget mỗi vòng**, ưu tiên câu dễ-chắc trước, câu khó để sau → giải được nhiều câu hơn. Bộ phân loại độ-khó/độ-tin (Platt/Isotonic calibration, Guo et al. ICML'17) trên feature {gap top1−top2, điểm tuyệt đối, đồng thuận đa kênh, hubness}.
- Nếu vòng final là **live KIS (time + nộp-sai)** như gov.vn mô tả → đồng hồ "submit ngay khi P(top-1 đúng) > ngưỡng" như kế hoạch gốc.
→ **Phải chốt cơ chế chấm vòng final 2026** rồi mới cố định N6.

### 6.3 N7 — learned reranker đa kênh
GBDT/MLP nhỏ trên feature {điểm từng kênh, đồng thuận, biên rerank, hubness N1, trùng object, exact-match OCR} huấn luyện trên F1 — thay/bổ trợ BGE text-only; chia sẻ feature với N6. Tuỳ chọn VLM-rerank top-10 ảnh thật (C2) khi cần độ chính xác cao.

---

## 7. Nâng cấp STACK (đã có số — đổi được ngay)

| Thành phần | FUFU hiện tại | Nâng cấp (số đo) | Ghi chú |
|---|---|---|---|
| **OCR** | EasyOCR (acc ~**49%** trên video text 🟡) | **VLM OCR** GPT-4o/Gemini acc **~76%** (+27đ); hoặc **VietOCR/PaddleOCR** CER **~0.22** offline | EasyOCR yếu nhất; VLM tốt nhưng tốn API. Cân nhắc VietOCR cho on-prem. |
| **ASR** | PhoWhisper-medium | **PhoWhisper-large** WER 4.67 (VIVOS)/8.14 (CMV) 🟡; **faster-whisper** (CTranslate2) **4× nhanh**, int8 ~2.9GB | faster-whisper để ingest nhanh; large nếu cần WER thấp nhất. |
| **FTS tiếng Việt** | SQLite FTS5 (k1=1.2, b=0.75 **cố định**) | ⚠ `unicode61 remove_diacritics`: bản `=1` **gộp sai dấu** tiếng Việt; cần `=2` (folding đầy đủ) hoặc `=0` (giữ dấu). Hoặc **ES + analyzer CocCoc** (`vi_analyzer`), hoặc custom FTS5 tokenizer (coccoc/VnCoreNLP) | Word-seg: **RDRsegmenter F1 97.9%, 62k từ/s** (VnCoreNLP). Cân nhắc dual-index có/không dấu. |
| **FAISS** | HNSW (M=32, ef_search=128) | <1M: IVF với K=4–16·√N; **1–10M: `IVF65536_HNSW32`** (🟡 FAISS guideline) | Quy mô 250GB → ước ~1–5M keyframe. HNSW hiện vẫn ổn ≤vài triệu; cân nhắc IVF khi RAM căng. |
| **Eval/live practice** | chỉ MSR-VTT dịch | **DRES** (`dres-dev/DRES`) để tập live KIS; tự dựng query set | F1 §8. |
| **Doc2query** | ❌ | Sinh 10–15 query/frame bằng Qwen-VL lúc ingest (caption ngắn, §3.3) → embed + FTS | Nuôi cả querybank N1. |

---

## 8. F1 — Eval harness (điều kiện tiên quyết, làm TRƯỚC mọi mũi N)

1. **Đo đúng metric thi:** implement `Mean of Top-k R-Scores` với **k∈{1,5,20,50,100}** cho cả 3 task (KIS/VQA/TRAKE), KHÔNG dùng recall@k chung. Đây là điểm dễ sai nhất.
2. **Tập query:** lấy đề các mùa cũ (2023/2024/2025) làm template, tự viết 50–100 query tiếng Việt + ground-truth (video+frame range) trên dataset mẫu. Giữ **tập test riêng** không dùng để tune (chống overfit, mô phỏng public 50% / private 100%).
3. **TRAKE eval:** chuỗi nhiều moment, chấm tỉ lệ frame trúng đoạn GT.
4. **DRES** (`dres-dev/DRES`) dựng local để tập phản xạ live + đo time-to-find.
5. Mỗi mũi N: chạy baseline → sửa → chạy lại → ghi vào bảng tiến độ ([RESEARCH-PLAN §6](RESEARCH-PLAN.md)). Không merge thứ giảm Mean-Top-k-R-Score.

---

## 9. Nguồn (đã verify / primary — đọc lại trước khi code)

**Cuộc thi (✅):**
- HCMC AI Challenge 2025 — Codabench: https://www.codabench.org/competitions/10187/
- Thể lệ chính thức (KIS scoring): https://aichallenge.hochiminhcity.gov.vn/en/huong-dan
- LLandMark / AI VIETNAM (xác nhận metric Mean Top-k R-Score + 3 task + dataset 250GB): https://arxiv.org/html/2603.02888
- AIC 2024 overview: https://link.springer.com/chapter/10.1007/978-981-96-4291-5_1 · AIC 2023: https://dl.acm.org/doi/10.1145/3628797.3628940

**N1 QB-Norm & hubness:**
- Repo gốc (✅ β, gains): https://github.com/ioanacroi/qb-norm · paper CVPR'22: https://arxiv.org/abs/2112.12777 · slides: https://samuelalbanie.com/files/digest-slides/2022-06-qb-norm.pdf
- All-but-the-top (ICLR'18): https://openreview.net/pdf?id=HkuGJ3kCb · CSLS/hubness survey: https://arxiv.org/pdf/1706.04902

**N2 domain-adapt:**
- CLIP-LoRA: https://arxiv.org/pdf/2405.18541 · LP-FT pitfall: https://arxiv.org/abs/2202.10054 · misalignment/LiT: https://arxiv.org/pdf/2409.01936 · NLIP (lọc nhiễu): https://arxiv.org/pdf/2212.07086
- LaCLIP: https://arxiv.org/abs/2305.20088 · MLLM-caption: https://arxiv.org/html/2311.18765v3 · VeCLIP: https://ar5iv.labs.arxiv.org/html/2310.07699 · CLIPS (short-caption): https://arxiv.org/pdf/2411.16828

**N3 HyDE:** https://github.com/texttron/hyde · https://aclanthology.org/2023.emnlp-main.585.pdf

**N4 temporal:**
- DANTE (AIO_Owlgorithms AIC'25): https://arxiv.org/abs/2512.13169 · MADTempo: https://arxiv.org/abs/2512.12929 · temporal-decay: https://arxiv.org/pdf/2512.12935
- CG-DETR: https://arxiv.org/html/2311.08835v4 · TR-DETR: https://arxiv.org/html/2401.02309v1

**N6 calibration:** Guo et al. ICML'17: https://arxiv.org/abs/1706.04599

**Stack VN & engineering:**
- OCR VLM-vs-traditional: https://arxiv.org/html/2502.06445v1 · VLM VN OCR: https://arxiv.org/html/2508.13680v1 · VietOCR/Paddle MC-OCR: https://arxiv.org/html/2506.05061v1
- PhoWhisper: https://arxiv.org/abs/2406.02555 · faster-whisper: https://github.com/SYSTRAN/faster-whisper
- ES analyzer VN (CocCoc): https://github.com/duydo/elasticsearch-analysis-vietnamese · RDRsegmenter: https://arxiv.org/pdf/1709.06307 · FTS5: https://sqlite.org/fts5.html
- FAISS guideline: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index · DRES: https://github.com/dres-dev/DRES

> ⚠️ Các mục 🟡/🟠 (recipe, benchmark, time-limit, tên đội thắng) là từ nguồn gốc nhưng **bước verify đối-kháng tự động chưa hoàn tất** (rate-limit). Khi cần dựa vào con số cụ thể để quyết định, mở lại đúng URL ở trên xác nhận. Có thể chạy lại verify sạch sau khi rate-limit reset bằng cách resume 5 workflow (script đã lưu trong session).
