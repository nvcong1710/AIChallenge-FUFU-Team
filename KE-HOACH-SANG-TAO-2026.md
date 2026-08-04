# KẾ HOẠCH SÁNG TẠO ĐỂ THẮNG — FUFU @ HCM AI Challenge 2026

> **File này khác gì [RESEARCH-PLAN.md](RESEARCH-PLAN.md)?**
> RESEARCH-PLAN là **menu sao chép kỹ thuật của đội thắng** (A1=NII-UIT/PraK temporal, B1=QUEST rewrite, B2=MADTempo image-search…). Những thứ đó cần thiết để **ngang bằng mặt sàn**, nhưng nếu chỉ làm chúng thì ta đi sau đội đã publish 1 năm.
>
> File này trả lời câu hỏi khác: **làm gì để VƯỢT lên khi mọi đội top đều đã có temporal + LLM rewrite + ensemble?** Trọng tâm là các kỹ thuật **chưa thấy đội VBS/AIC nào báo cáo**, nhưng (a) có cơ sở học thuật vững trong IR/retrieval ML, (b) chạy được trên 3090/Vast.ai, (c) **đo được lời/lỗ ngay trên eval harness F1**.
>
> **Triết lý xuyên suốt:** *"Trả giá ở ingest — thắng ở query."* User đã bật đèn xanh đốt compute lúc ingest. Đây là đòn bẩy quyết định, vì **dữ liệu thi được phát hành trước vòng live** → mọi thứ tính sẵn được lúc ingest là lợi thế không đối thủ nào lấy lại được trong 3 giờ thi.
>
> Cập nhật: 2026-06-26 · dựa trên deep-research đã verify (VBS 2024-2026, HCM AIC 2024-2025, TRECVID AVS) + đối chiếu [PROJECT-CONTEXT.md](PROJECT-CONTEXT.md).
>
> 🔧 **2026-06-27 — lớp "code được":** công thức chính xác, recipe, pseudocode, số benchmark cho N1–N8 (từ vòng deep-research thứ 2, ~115 nguồn primary) đã chuyển sang **[PHU-LUC-KY-THUAT-2026.md](PHU-LUC-KY-THUAT-2026.md)**. Quan trọng: vòng đó **xác minh metric chấm điểm AIC 2025 = `Mean of Top-k R-Scores`, k∈{1,5,20,50,100}** (xếp hạng, không đua-nộp-trước thuần) → metric này *được sinh ra* cho combo N1+N7+N3. Đọc phụ lục §1 trước khi triển khai.

---

## 0. Luận điểm trung tâm (đọc 60 giây)

Ba sự thật đã verify từ research, ghép lại thành một chiến lược:

1. **"Chênh lệch model giữa các đội top là rất nhỏ; thắng thua nằm ở tốc độ, temporal, UI, và số cửa truy vấn."** (meta-bài học VBS, lặp lại mọi năm). → Nghĩa là: **đua model mạnh hơn là cuộc đua thua**. Phải thắng ở chỗ khác.
2. **HCM AIC & VBS đều phát hành corpus TRƯỚC vòng thi live.** Ta ingest, dựng index, rồi vài ngày sau mới thi. → **Mọi tính toán nặng làm sẵn được trên CHÍNH corpus thi đều hợp lệ và không thể bị đối thủ bắt kịp lúc live.** Hầu hết các đội không khai thác điều này vì nó "tốn kém" — đúng cái user cho phép.
3. **Scoring KIS phạt submit sai + thưởng tốc độ** (HCM 2024: chấm theo thời gian + số lần submit sai; VBS KIS: first-correct). → **Một tín hiệu "nên submit bây giờ" được hiệu chỉnh tốt = điểm thật**, mà gần như không đội nào làm.

> **Kết luận:** đặt cược vào 3 nơi đối thủ bỏ trống — **(I) hiệu chỉnh không gian embedding theo chính corpus thi**, **(II) thu hẹp khoảng cách phân phối giữa query và index**, **(III) biến độ tin cậy thành quyết định submit**. Đây là 3 đòn "đốt compute ingest → thắng query" mà RESEARCH-PLAN chưa có.

---

## 1. Bản đồ "khe hở" — đối thủ đang bỏ trống chỗ nào

| Vùng | Đội top đã bão hoà | Khe hở còn trống (← ta đánh vào đây) |
|---|---|---|
| Encoder | CLIP/SigLIP/BEiT-3 ensemble | **Hiệu chỉnh hậu-kỳ không gian embedding** (hubness, whitening, querybank-norm) — gần như 0 đội VBS báo cáo |
| Query expansion | LLM rewrite, dịch, paraphrase | **Khớp phân phối query↔caption-index** (visual-HyDE) — khác paraphrase mù |
| Domain | dùng checkpoint web đông cứng | **Fine-tune/adapter trên CHÍNH corpus thi** bằng caption tự sinh — rất ít đội dám vì tốn |
| Temporal | pair-query, DP align (DANTE) | **Index event-level dựng sẵn** để DP chạy trên không gian nhỏ & sạch |
| Fusion | trọng số cố định | **Định tuyến trọng số động theo loại query** |
| Rerank | cross-encoder text / VLM ảnh | **Reranker học từ feature đa kênh** (rẻ, hiệu chỉnh được, nuôi tín hiệu submit) |
| Thi đấu | UI nhanh | **Chính sách submit hiệu chỉnh xác suất** — biến confidence thành điểm |

Bảy mũi nhọn dưới đây tấn công đúng 7 ô bên phải.

---

## 2. Bảy mũi nhọn sáng tạo (xếp theo ROI: impact × độ-mới × độ-khả-thi)

> Mỗi mũi có: **cơ chế** (làm gì) · **vì sao mới** (không có trong menu A1–F2 & chưa thấy đội nào publish) · **vì sao khả thi** (chạy được trên phần cứng nào, chi phí ingest/query) · **bằng chứng** (paper nền tảng, kèm mức độ tự tin) · **cách verify** (đo thắng/thua thế nào trên F1) · **rủi ro**.

---

### 🥇 N1 — Querybank Normalisation + khử hubness trên index (CROWN JEWEL)

**Cơ chế.** Embedding SigLIP/CLIP bị bệnh **hubness**: một số ít frame (bầu trời, đám đông, frame nhiều chữ, frame "trung tính") nằm trong top-K của **gần như mọi** query → đẩy frame đúng ra khỏi top-1/5. Đây là vấn đề kinh điển của cross-modal retrieval. Cách chữa:
- **Querybank Normalisation (QB-Norm)** — dựng một "ngân hàng truy vấn" Q (vài nghìn text mẫu: chính các caption VLM của frame + synthetic query D1 + đề các mùa cũ). Lúc query, thay vì dùng `sim(q, frame)` thô, chuẩn hoá bằng **Dynamic Inverted Softmax**: phạt frame nào "giống tất cả mọi thứ" trong Q. Frame hub bị hạ điểm, frame đặc thù được nâng.
- Bổ trợ rẻ tiền: **whitening / "all-but-the-top"** (bỏ vài principal component trội của tập embedding image lúc ingest) để khử dị hướng (anisotropy).

**Vì sao mới.** Không có trong A1–F2. Tôi **không thấy đội VBS/AIC nào báo cáo querybank-norm hay khử-hubness** — đây là kỹ thuật từ giới retrieval-ML chưa "vượt biên" sang cộng đồng thi video. Đối thủ chạy cosine thô trên index thô.

**Vì sao khả thi.** Cực rẻ. Lúc ingest: tính trước, với mỗi frame, thống kê độ tương đồng của nó với querybank (1 hằng số chuẩn hoá hoặc top-m sim) — vài MB, vài phút. Lúc query: thêm **1 phép chuẩn hoá**, latency tăng <5ms. Chạy trên **bất kỳ GPU nào**, kể cả CPU.

**Bằng chứng.** *Querybank Normalisation* (Bogolin et al., **CVPR 2022**) báo cáo cải thiện R@1 đáng kể (thường vài điểm %) trên text→video retrieval (MSR-VTT/MSVD/…). *Dual-Softmax* (CAMoE) & inverted-softmax cùng họ, dùng phổ biến lúc inference của X-CLIP. *"All-but-the-top"* (Mu & Viswanath, ICLR 2018) cho post-processing embedding. **Độ tự tin: CAO** — đây là phương pháp đã công bố, rõ ràng, dễ tái lập. *(Cần đọc lại paper QB-Norm để lấy đúng công thức β-γ trước khi code.)*

**Verify.** A/B trên F1: bật/tắt QB-Norm + whitening, đo `recall@1/5/20` + MRR. Đây là phép thử rẻ nhất, làm **đầu tiên ngay sau F1**. Kỳ vọng: +2 đến +6 điểm R@1 nếu index có hubness rõ (gần như chắc có).

**Rủi ro.** Thấp. Querybank lệch phân phối → chuẩn hoá quá tay; mitigate bằng tune cường độ chuẩn hoá trên F1 (1 hyperparam). Nếu không cải thiện thì cũng không phá gì (tắt đi).

---

### 🥈 N2 — Domain-adaptation SigLIP trên CHÍNH corpus thi (LoRA, self-distill)

**Cơ chế.** Sau khi ingest xong, ta có **hàng trăm nghìn cặp (frame, caption VLM + OCR + objects + ASR)** thuộc **đúng domain thi** (video tin tức VN). Dùng chúng làm dữ liệu contrastive **fine-tune nhẹ SigLIP bằng LoRA** (adapter rank thấp trên text tower, tuỳ chọn cả image tower) → kéo không gian alignment vốn học từ web về **đúng phân phối news-video tiếng Việt-dịch-Anh**. Đây là "test-domain adaptation không nhãn" hợp lệ vì corpus đã công khai.

**Vì sao mới.** Không có trong menu. Các đội VBS dùng checkpoint **đông cứng** (frozen). Tôi **không thấy đội nào fine-tune encoder trên chính corpus thi bằng caption tự sinh** — vì nó tốn compute và rủi ro, đúng thứ user cho phép đánh đổi. Đây là edge cấu trúc: ta có một encoder "biết" dataset, họ thì không.

**Vì sao khả thi.** LoRA trên SigLIP-Large với vài trăm nghìn cặp: **vài giờ trên 1×3090** (hoặc thuê A100 trên Vast.ai cho nhanh). Adapter ~vài chục MB. Lúc online: nạp base + adapter, VRAM gần như không đổi. Giữ **base làm thành viên ensemble** (xem C1) → không mất gì kể cả khi adapter kém.

**Bằng chứng.** *LaCLIP* (Fan et al., **NeurIPS 2023**) — train/adapt CLIP bằng caption viết lại, cải thiện retrieval. Caption-augmented contrastive là hướng đã được chứng minh. Domain-adapt CLIP bằng pseudo-caption là thực hành chuẩn trong transfer learning. **Độ tự tin: TRUNG BÌNH-CAO** về tính khả thi; **độ lợi cần đo** (phụ thuộc chất lượng caption tự sinh).

**Verify.** F1 trước/sau, **bắt buộc không được giảm recall**. Giữ checkpoint base; chỉ promote adapter nếu thắng F1 rõ. Theo dõi overfit bằng tập query held-out không dùng khi train.

**Rủi ro.** TRUNG BÌNH — caption tự sinh nhiễu → adapter học sai; catastrophic drift. Mitigate: rank thấp (8–16), learning rate nhỏ, early-stop trên F1, **luôn ensemble với base** nên downside có chặn. Đây là mũi "high-upside, bounded-downside".

---

### 🥉 N3 — Visual-HyDE: sinh caption giả-lập KHỚP phân phối index

**Cơ chế.** Query người dùng ("người đàn ông áo đỏ đang chạy") nằm ở **phân phối ngôn ngữ khác** với thứ mà index "thích" — vì frame trong index được căn chỉnh tốt nhất với **văn phong caption VLM** (câu mô tả đầy đủ kiểu "A man wearing a red shirt is running on a street, daytime, urban background…"). Thay vì paraphrase mù, dùng LLM sinh **caption-giả-định theo đúng văn phong caption của bộ extractor** rồi encode caption đó để search. Đây là **HyDE (Hypothetical Document Embeddings)** áp cho cross-modal: "tài liệu giả định" = caption-frame giả định.

**Vì sao mới.** Khác B1 (rewrite tách kênh) và khác paraphrase hiện tại (3 cách diễn đạt chung chung). N3 **cố ý mô phỏng văn phong của caption-index** để thu hẹp gap query↔index — một biến thể distribution-aware chưa thấy đội nào nêu đích danh.

**Vì sao khả thi.** Chỉ là lời gọi LLM (Qwen-3B đã có, hoặc API) + encode text. Latency thêm ~vài trăm ms → **cache theo query** và sinh sẵn cho đề mẫu. Không thêm VRAM ingest.

**Bằng chứng.** *HyDE — Precise Zero-Shot Dense Retrieval without Relevance Labels* (Gao et al., **2022/2023**), cải thiện rõ dense retrieval zero-shot. Cùng họ doc2query/query2doc. **Độ tự tin: CAO** rằng phương pháp tồn tại; lợi ích trên SigLIP-news cần đo.

**Verify.** F1: so 3 chế độ expand — (a) hiện tại, (b) +HyDE, (c) HyDE-only. Đo theo từng loại query (visual mơ hồ vs entity vs OCR-heavy) vì HyDE lợi nhất ở **query visual mơ hồ**.

**Rủi ro.** Thấp-trung bình. LLM "ảo" chi tiết sai → nhiễu; mitigate: sinh **nhiều** caption giả rồi mean-pool embedding (giảm phương sai), và **fuse** với query gốc thay vì thay thế.

---

### N4 — Index event-level dựng sẵn cho TRAKE / temporal (đốt ingest)

**Cơ chế.** A1/A2 (menu) chấm temporal **hậu kỳ** trên frame đơn. Bổ sung: lúc ingest, tính **embedding mức "sự kiện"** bằng cách attention-pool / mean-pool các keyframe của **cửa sổ shot liên tiếp** (2–5 shot) → một FAISS index thứ 2 ở mức event. Query TRAKE nhiều moment match thẳng vào event-vectors → DP alignment (A2) chỉ chạy trên **tập ứng viên nhỏ, đã gom ngữ nghĩa**, vừa nhanh vừa chính xác hơn.

**Vì sao mới.** Vượt "cộng dồn similarity" (MADTempo) và "DP trên frame" (DANTE): ta cho DP một **không gian tìm kiếm cấp sự kiện dựng sẵn**, giảm nhiễu frame lẻ. Index event-level cho TRAKE chưa thấy đội nào báo cáo.

**Vì sao khả thi.** Chỉ thêm embedding + 1 index lúc ingest (compute rẻ so với caption). Storage tăng ~10–20%.

**Bằng chứng.** Temporal mean/attention-pooling là chuẩn trong video retrieval; điểm mới là **dùng nó làm tầng index cho TRAKE**. **Độ tự tin: TRUNG BÌNH** (kỹ thuật nền chắc; thiết kế cụ thể cần thử nghiệm). **Điều kiện:** chỉ làm nếu 2026 giữ TRAKE (2025 có; cần xác nhận đề 2026).

**Verify.** Eval riêng cho TRAKE (chuỗi nhiều moment) trên F1-TRAKE: đo "đúng frame cho từng moment" theo đúng cách BTC chấm.

**Rủi ro.** Phụ thuộc đề 2026 giữ TRAKE. Là **enhancement của A1/A2**, không thay thế — cần làm temporal core trước.

---

### N5 — Định tuyến trọng số fusion ĐỘNG theo loại query

**Cơ chế.** Hiện fusion cố định `dense .4 / visual .25 / asr .5`. Nhưng query "biển hiệu ghi "Phở Hoà"" cần dồn OCR; "cảnh rượt đuổi xe máy" cần dồn dense. Dùng **bộ phân loại nhẹ** (hoặc chính kết quả parse có cấu trúc của B1) để **đặt trọng số fusion theo từng query**. Học trọng số bằng logistic regression trên F1 với feature loại-query.

**Vì sao mới.** Hầu hết hệ dùng trọng số **tĩnh** (kể cả FUFU). Định tuyến động theo intent là cải tiến rẻ, ít đội khai thác.

**Vì sao khả thi.** Phân loại bằng vài luật + LLM (B1 đã gọi LLM sẵn → "free"). Không thêm model nặng.

**Bằng chứng.** Adaptive/learned fusion phổ biến trong IR; điểm mới là **per-query routing theo intent** trong ngữ cảnh thi. **Độ tự tin: CAO** về khả thi; lợi ích cần đo.

**Verify.** F1 phân nhóm theo loại query (OCR-heavy / ASR-heavy / visual). Trọng số động phải thắng trọng số tĩnh **trên trung bình toàn tập**, không chỉ một nhóm.

**Rủi ro.** Thấp. Sai loại → tệ hơn tĩnh ở vài query; mitigate: phân loại "không chắc" → về trọng số tĩnh mặc định.

---

### N6 — Chiến lược tối ưu xếp hạng 100 đáp án (Maximize Top-k R-Score)

**Cơ chế.** Vòng sơ tuyển 2026 cho phép nộp tối đa **100 câu trả lời** và chấm điểm bằng **Trung bình Top-k R-Score (k ∈ {1, 5, 20, 50, 100})**, hoàn toàn không phạt submit sai hay tính thời gian. Do đó, thay vì "đồng hồ tin cậy nộp bài", N6 chuyển thành **chính sách đa dạng hoá và xếp hạng top 100**: dùng bộ phân loại (đã hiệu chỉnh xác suất) để (1) chọn câu trả lời tự tin nhất đưa lên hạng 1, (2) ở các hạng 2-100, ưu tiên **đa dạng hoá** (diversity) các frame/đoạn video khác nhau thay vì nộp 100 frame gần nhau của cùng 1 video sai.

**Vì sao mới.** Hầu hết các hệ thống chỉ nộp top-K thô từ thuật toán search. Việc chủ động phân bổ đáp án vào 100 slot để tối đa hoá kỳ vọng "trúng ít nhất 1 slot ở các mốc k" là một chiến thuật tối ưu metric đặc thù của cuộc thi.

**Vì sao khả thi.** Rất nhẹ. Chỉ là thuật toán re-ranking/diversification (như MMR - Maximal Marginal Relevance) chạy trên danh sách top 1000 kết quả ban đầu trước khi submit.

**Bằng chứng.** Đa dạng hoá kết quả (Search Result Diversification) là kinh điển trong IR. Áp dụng MMR để phủ tối đa các mốc k là chiến thuật toán học thuần tuý. **Độ tự tin: CAO**.

**Verify.** **Mô phỏng giải đấu** trên F1: chạy chính sách MMR diversify top 100, đo **trung bình Top-k R-Score** so với việc nộp top 100 thô từ retriever.

**Rủi ro.** Rất thấp. Việc đa dạng hoá các hạng sau (từ hạng 2 đến 100) hầu như chỉ có lợi trong metric R@k vì nếu top 1 sai mà các kết quả sau quá giống top 1 thì cũng sai nốt.

---

### N7 — Reranker học từ feature đa kênh (thay/bổ trợ BGE text-only)

**Cơ chế.** Rerank hiện tại = BGE cross-encoder chỉ đọc **text** (caption+objects+ASR), **mù ảnh thật**. Hai nâng cấp, đều hơn hiện tại:
- (a) **VLM rerank top-20** trên ảnh thật (C2 trong menu — mạnh nhưng chậm).
- (b) **Learned fusion reranker**: GBDT/MLP nhỏ trên feature giàu (điểm từng kênh, mức đồng thuận, biên rerank, hubness N1, độ trùng object, exact-match OCR) huấn luyện trên F1. Rẻ, hiệu chỉnh được, và **chia sẻ feature với N6** (đồng hồ submit).

**Vì sao mới.** Reranker **học trên eval set tự dựng** từ feature đa kênh ít gặp trong các hệ thi (họ dùng cross-encoder off-the-shelf hoặc VLM). Kết hợp (b) làm "bộ não" chung cho cả rank lẫn quyết định submit là thiết kế gọn.

**Vì sao khả thi.** GBDT train tức thì; feature đã có sẵn từ pipeline. VLM rerank (a) tốn hơn → chỉ top-10/20.

**Bằng chứng.** Learning-to-rank (LambdaMART/GBDT) là kinh điển IR. **Độ tự tin: CAO**.

**Verify.** F1: so (hiện tại BGE) vs (b GBDT) vs (a VLM) vs (a+b). Đo recall + latency. Chọn theo Pareto.

**Rủi ro.** Thấp. (b) cần đủ dữ liệu eval (gắn với F1 phải đủ lớn). Overfit eval nhỏ → giữ tập test riêng.

---

### N8 *(tuỳ chọn, nghiên cứu)* — Late-interaction kiểu ColBERT cho frame

**Cơ chế.** Lưu **nhiều embedding vùng/patch** mỗi frame, query bằng **MaxSim** (như ColBERT) → bắt được "vật nhỏ trong góc khung" mà embedding toàn-frame làm mờ. Đốt ingest + storage mạnh.

**Vì sao mới / khả thi / rủi ro.** Mới so với menu, nhưng **storage & latency nặng** (gấp nhiều lần). **Chỉ làm nếu N1–N7 đã xong và còn thời gian.** Đây cũng là bản "localized query" của PraK V4 nhưng theo hướng dense. **Độ tự tin: TRUNG BÌNH-THẤP** về tỉ lệ lời/lỗ trong khung thời gian thi → để cuối.

---

## 3. Bảng tổng hợp — ưu tiên & đặt cược

| # | Mũi nhọn | Impact | Độ mới | Effort | Chi phí ingest | Chi phí query | Tự tin | Khi làm |
|---|---|---|---|---|---|---|---|---|
| **N1** | Querybank-norm + khử hubness | 🔥🔥🔥 | ⭐⭐⭐ | S | rẻ | ~0 | CAO | **ngay sau F1** |
| **N6** | Tối ưu hạng 100 (Diversity) | 🔥🔥🔥 | ⭐⭐⭐ | S-M | 0 | ~0 | CAO | làm thuật toán MMR |
| **N2** | Domain-adapt SigLIP (LoRA) | 🔥🔥🔥 | ⭐⭐⭐ | M-L | **cao** (vài giờ GPU) | ~0 | TB-CAO | sau N1, có thời gian |
| **N3** | Visual-HyDE | 🔥🔥 | ⭐⭐ | S-M | 0 | thấp (cache) | CAO | song song N1 |
| **N5** | Fusion routing động | 🔥🔥 | ⭐⭐ | S | 0 | ~0 | CAO | sau B1 |
| **N7** | Learned multi-channel reranker | 🔥🔥 | ⭐⭐ | M | 0 | thấp | CAO | sau F1 đủ lớn |
| **N4** | Event-index cho TRAKE | 🔥🔥 | ⭐⭐ | M | TB | thấp | TB | **Đã chốt TRAKE sơ tuyển** |
| N8 | ColBERT late-interaction | 🔥 | ⭐⭐ | L | rất cao | cao | TB-thấp | cuối, nếu dư |

**Combo đặt cược chính (ROI cao nhất, ít rủi ro):** **N1 + N6 + N3 + N5** — toàn bộ rẻ, nhanh, độ tự tin cao, đánh đúng 4 ô đối thủ bỏ trống. Đây là phần "thắng mà không cần model mạnh hơn".

**Combo khác biệt hoá (nếu có thời gian + dám đốt ingest):** + **N2 (domain-adapt)** — đây là mũi duy nhất tạo lợi thế *cấu trúc* (encoder biết dataset), khó bị copy trong vòng live.

**Mở rộng theo đề:** + **N4** nếu TRAKE còn; + **N7** khi eval đủ lớn để học reranker.

---

## 4. Quan hệ với menu A1–F2 cũ (giữ gì / thay gì)

- **F1 (eval harness) vẫn là điều kiện tiên quyết tuyệt đối** — mọi mũi N1–N8 chỉ chứng minh được bằng F1. **Làm F1 trước hết, không bàn cãi.**
- **A1 (temporal) + B1 (LLM rewrite) vẫn cần** để ngang mặt sàn — nhưng coi chúng là **nền**, không phải vũ khí thắng. N-series là lớp vượt lên trên.
- N3 **thay** cách paraphrase mù hiện tại bằng HyDE distribution-aware.
- N5 **thay** trọng số tĩnh (giải quyết luôn C5).
- N7 **nâng cấp** rerank (gồm C2 như một lựa chọn).
- N2 **bổ sung** ensemble (C1): base + adapter + (tuỳ chọn) encoder thứ 2.

> Nói ngắn: **A1/B1/F1 để không thua; N1/N6/N2/N3 để thắng.**

---

## 5. Lộ trình thực thi (8–10 tuần, eval-driven)

| Tuần | Việc | Phụ thuộc | Verify |
|---|---|---|---|
| 1 | **F1 eval harness** (50–100 query KIS + TRAKE tiếng Việt theo format đề) + baseline số | — | có baseline recall@1/5/20 + MRR |
| 1–2 | **N1** querybank-norm + whitening (đòn rẻ nhất, làm ngay) | F1 | A/B recall, kỳ vọng +2..6 R@1 |
| 2 | **N3** visual-HyDE + **N5** fusion routing (đều rẻ) | F1, B1 | A/B theo loại query |
| 2–3 | A1 temporal core + B1 LLM rewrite (nền mặt sàn) | F1 | recall trên query chuỗi |
| 3–4 | **N6** chính sách submit (cần luật điểm BTC → tạm dùng luật 2024) | F1 + mô phỏng điểm | **tổng điểm** mô phỏng giải |
| 4–6 | **N2** domain-adapt SigLIP (LoRA) — đốt GPU Vast.ai | ingest xong, F1 | F1 trước/sau, không regress |
| 5–6 | **N7** learned reranker (khi F1 đủ lớn) | F1 lớn | Pareto recall/latency |
| 6–7 | E1 UI thi đấu + E2 latency (<500ms) + tích hợp đồng hồ N6 | A1, N6 | latency, mock contest |
| 7–8 | **N4** event-index (nếu TRAKE) · ensemble N2+base+enc2 | A1/A2, N2 | F1-TRAKE |
| 8–10 | Mock contest lặp, tune ngưỡng submit, N8 nếu dư | tất cả | tổng điểm mock |

**Nguyên tắc (giữ từ P2/P3):** mỗi mũi qua F1 đo trước/sau; không merge thứ giảm recall@5; latency online <1s (mục tiêu <500ms).

---

## 6. Xác minh "khả năng đạt giải" — thành thật

**Không thể hứa giải.** Nhưng có thể lập luận **vì sao các mũi này tạo lợi thế ở đúng nơi cuộc đua đang hoà**:

- **Cơ sở thắng:** research đã verify rằng đội top **cách nhau rất ít về model**. Khi mọi người hoà ở retrieval thô, **lợi thế biên** (N1 khử hubness +vài % R@1, N6 không bao giờ submit sai + submit sớm, N2 encoder biết dataset) **trực tiếp đổi thành thứ hạng**. Đây là chiến lược "thắng ở rìa" — phù hợp khi không thể thắng ở lõi.
- **Bằng chứng từng mũi (mức tự tin):** N1 (CVPR'22, CAO), N3 (HyDE, CAO), N6 (calibration kinh điển, CAO), N5/N7 (LTR/adaptive fusion, CAO), N2 (LaCLIP NeurIPS'23, TB-CAO), N4 (pooling chuẩn, TB). **Không mũi nào là bịa** — tất cả dựa trên phương pháp đã công bố, chỉ là **chưa ai ghép chúng vào ngữ cảnh AIC/VBS**.
- **Cách tự xác minh trước khi tin:** F1 là trọng tài. N1/N3/N5/N7 cho con số trong **vài ngày**. N6 cho **tổng điểm mô phỏng** — thước đo gần "đạt giải" nhất ta có offline. Nếu một mũi không thắng trên F1 → bỏ, không tiếc.
- **Rủi ro lớn nhất:** (1) **luật điểm & đề 2026 chưa chốt** → N4/N6 phải chờ thể lệ; (2) **F1 phải đủ giống đề thật** — đây là điểm chết người, nếu eval set lệch thì mọi tuning sai hướng → **đầu tư mạnh vào F1, lấy đề các mùa cũ làm template, ưu tiên query tiếng Việt thật**; (3) **N2 overfit** → chặn downside bằng ensemble với base.

> **Đặt cược một câu:** nếu chỉ có thời gian cho **3 thứ**, làm **F1 → N1 → N6**. Eval tốt + khử hubness + submit thông minh là bộ ba "đốt rất ít công, đổi trực tiếp ra thứ hạng" mà gần như không đội nào tối ưu.

---

## 7. Nguồn (nền tảng các mũi N — cần đọc kỹ trước khi code)

- **N1:** Bogolin et al., *Cross-Modal Retrieval with Querybank Normalisation*, CVPR 2022 · Mu & Viswanath, *All-but-the-Top*, ICLR 2018 · Cheng et al., *CAMoE / Dual-Softmax* (inference-time inverted softmax).
- **N2:** Fan et al., *LaCLIP: Improving CLIP Training with Language Rewrites*, NeurIPS 2023 · LoRA (Hu et al., 2021) cho fine-tune adapter.
- **N3:** Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)*, 2022/2023 · doc2query / query2doc (họ query-expansion sinh tài liệu).
- **N4:** temporal mean/attention-pooling (chuẩn video retrieval) + DANTE/DP align (AIO_Owlgorithms, arXiv 2512.13169 — đã có trong RESEARCH-PLAN).
- **N5:** learning-to-rank / adaptive fusion (IR kinh điển).
- **N6:** Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 (Platt/temperature scaling) · isotonic regression.
- **N7:** LambdaMART / GBDT learning-to-rank.
- **N8:** Khattab & Zaharia, *ColBERT: Late Interaction*, SIGIR 2020 (áp cho region embedding của frame).
- **Bối cảnh thi (đã verify, xem RESEARCH-PLAN §7):** VBS 2025 results (arXiv 2509.12000) · HCMC AIC 2024 overview (Springer 978-981-96-4291-5_1) · TRECVID AVS (NIST tv2023) · NII-UIT / PraK / VISIONE papers.

> ⚠️ Mọi citation trên là **phương pháp đã công bố thật**, nhưng **đọc lại paper gốc để lấy đúng công thức** trước khi triển khai — đặc biệt QB-Norm (β-γ) và HyDE (prompt sinh caption). Không code theo trí nhớ.
