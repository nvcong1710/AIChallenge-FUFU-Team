# Chương 19 — Đánh giá hệ retrieval: đo trước khi tin

> *"In God we trust; all others must bring data."* — câu treo tường của dân đo lường.
> Phiên bản FUFU: *demo thì vui, nhưng muốn merge thì mang bảng số tới.*

## 1. Vì sao chương này tồn tại trong FUFU

Hãy nhìn lại những gì team đã/sắp làm: tune trọng số hybrid (chương 17), thêm encoder thứ hai để ensemble (chương 18), có thể finetune LoRA (chương 16), đổi OCR engine, viết lại query expansion... Mỗi việc đều có chung một câu hỏi sống còn:

> **"Sau khi sửa, hệ thống TỐT LÊN hay TỆ ĐI?"**

Nếu không trả lời được câu này bằng **con số**, thì mọi việc trên đều là **mò trong bóng tối**. Đổi `weights.bm25_asr` từ 0.5 xuống 0.3 — tốt hơn không? Tắt caption để ingest nhanh gấp 10 — mất bao nhiêu recall? Thêm reranker — đáng giá 200ms latency không? Không có eval harness thì câu trả lời duy nhất là "chắc là... ổn?", và đó không phải kỹ thuật, đó là cầu nguyện.

RESEARCH-PLAN của team đã nói thẳng điều này ở ý **F1** (nhóm Nền tảng — làm trước mọi thứ khác):

> *"Eval harness theo format thi... — cách mọi đội nghiêm túc làm; **không có nó thì mọi tuning là mò**."*

Và §5 chốt nguyên tắc xuyên suốt: *"mỗi thay đổi phải qua F1 đo trước/sau; không merge thứ làm giảm recall@5; giữ latency <1s."*

Các đội thắng VBS / HCM AI Challenge đều có một điểm chung không nằm trong model: họ đo recall@k trên bộ query giống đề thi **sau mỗi thay đổi**. Chênh lệch model giữa các đội top vốn nhỏ; cái tách họ ra là kỷ luật đo đạc. Chương này dạy bạn xây kỷ luật đó cho FUFU. Đây là chương nền tảng: chương 16 (LoRA), 17 (tuning), 18 (ensemble) đều **giả định bạn đã có** thước đo từ chương này.

---

## 2. Cần biết trước

- **Precision / Recall / Accuracy** từ ML cổ điển (sklearn) — ta sẽ liên hệ trực tiếp.
- Chương 15 (pipeline FUFU): hiểu query đi qua dense + BM25 ×2 → fusion → rerank, và response có `timing_ms`.
- Khái niệm KIS (Known-Item Search) từ chương 15: mỗi query có **đúng một** đáp án (1 video + 1 khoảng thời gian).
- Biết sơ format chấm của cuộc thi (RESEARCH-PLAN §1.2): điểm tính theo **thời gian tìm thấy + số lần submit sai**, không phải theo độ "đẹp" của ranking.
- Không cần: đạo hàm, code dài, hay bất kỳ thư viện mới nào. Toàn bộ toán trong chương là phép chia.

---

## 3. "Demo thấy ổn" là ảo giác

Quy trình quen thuộc của mọi team mới: sửa code → mở frontend → gõ 3-4 query → "ồ ra đúng nè" → merge. Quy trình này hỏng vì ba lý do:

1. **Bias chọn query dễ.** Bạn vô thức gõ những query mà bạn *biết* hệ làm tốt ("người chơi cờ vua" — query bạn đã test 50 lần). Query khó (cần OCR chữ Việt cách điệu, tên riêng, cảnh tối) không bao giờ được gõ, nên điểm mù không bao giờ lộ ra.
2. **Không lặp lại được.** Tuần sau, người khác gõ 4 query *khác*, thấy kết quả tệ. Ai đúng? Không ai biết, vì không có bộ query cố định.
3. **Không so sánh trước/sau được.** "Hình như nhanh hơn" và "hình như đúng hơn" không cộng trừ được. Thay đổi A tăng query nhóm 1 nhưng giảm nhóm 2 — mắt thường không bao giờ thấy trade-off này.

Một tình huống cụ thể để thấy cái bẫy. Giả sử bạn đổi trọng số `bm25_asr` từ 0.5 → 0.3 và thử tay 4 query:

| Query thử tay | Trước | Sau | Cảm nhận |
|---|---|---|---|
| "người chơi cờ vua" | hạng 1 | hạng 1 | "vẫn ổn" |
| "bản tin thời tiết" | hạng 2 | hạng 4 | "hơi tụt, chắc không sao" |
| "cô gái múa lân" | hạng 8 | hạng 3 | "tốt lên nè!" |
| "biển hiệu quán cà phê" | ∅ | hạng 12 | "tốt lên nữa!" |

Cảm giác: "2 tốt lên, 1 tụt nhẹ, 1 giữ nguyên → merge". Nhưng 4 query này nói gì về 96 query còn lại của đề thi? **Không gì cả.** Có thể thay đổi này đánh sập cả nhóm query ASR (vốn là 25% đề) mà 4 query trên không đại diện. Chỉ một bộ eval đủ lớn, cố định, phủ đủ loại query mới trả lời được.

Chuyện thật trong ngành: ở mọi team search/recommendation nghiêm túc (Google, Spotify, các lab IR), một thay đổi ranking **không có bảng số đính kèm thì không được review**. Câu cửa miệng là *"What's the metric delta?"*. Team FUFU từ chương này sẽ làm y vậy — quy mô nhỏ hơn, kỷ luật y nguyên.

---

## 4. Eval phân loại (đã biết) vs eval retrieval (mới)

Với phân loại, bạn quen kiểu này: mỗi sample có 1 nhãn đúng, model đoán 1 nhãn → so sánh được ngay đúng/sai → đếm ra accuracy, confusion matrix, precision/recall theo lớp.

Retrieval khác ở chỗ: hệ không trả về *một* đáp án, mà trả về **một danh sách xếp hạng** (top-20 segment). Không có "đúng/sai" toàn cục cho cả danh sách. Câu hỏi đổi thành hai câu:

1. **Đáp án đúng có NẰM TRONG top-K không?** (có/không — giống recall)
2. **Nếu có, nó đứng thứ mấy?** (hạng 1 khác xa hạng 19, dù cả hai đều "trong top-20")

Liên hệ với precision/recall đã biết:

- **Recall** quen thuộc trong phân loại = TP / (TP + FN) = "trong số những cái thực sự dương, bắt được bao nhiêu". **Recall@K** là đúng khái niệm đó, nhưng tính trên "tập trả về = top-K của ranking": trong số các item relevant, bao nhiêu cái lọt vào top-K. Với KIS chỉ có 1 đáp án đúng mỗi query, recall@K của 1 query chỉ có thể là 0 hoặc 1 → trung bình trên nhiều query, nó thành **hit rate**: "bao nhiêu % query tìm thấy đáp án trong top-K".
- **Precision@K** (= bao nhiêu trong K kết quả là relevant) **ít có nghĩa với KIS**: chỉ có 1 đáp án đúng nên precision@20 tối đa là 1/20 = 5% — con số trông xấu nhưng vô hại. Vì vậy với FUFU ta gần như không dùng precision, đây là khác biệt lớn nhất so với eval phân loại.
- **Accuracy / confusion matrix** không có chỗ đứng: không có "lớp dự đoán" để đối chiếu — đầu ra là một thứ tự, không phải một nhãn.
- Một khác biệt nữa: trong phân loại, ngưỡng quyết định (threshold 0.5) là của model; trong retrieval, "ngưỡng" chính là **K** — và K do *trải nghiệm người dùng* quyết định (màn hình hiển thị được bao nhiêu kết quả), không phải do model. Đó là lý do ta báo cáo nhiều K cùng lúc (1, 5, 20) thay vì một con số duy nhất.

Tóm lại bảng quy đổi tư duy:

| Phân loại (quen) | Retrieval KIS (mới) |
|---|---|
| 1 sample → 1 nhãn dự đoán | 1 query → 1 danh sách xếp hạng |
| Đúng/sai tức thì | "Có trong top-K?" + "hạng mấy?" |
| Accuracy, confusion matrix | Recall@K (hit rate), MRR |
| Threshold của model | K của màn hình/người dùng |

---

## 5. Các metric — tính tay trên cùng một bảng

Toàn bộ mục này dùng **một bảng kết quả duy nhất** để mọi metric so sánh được với nhau. Cách tạo bảng: lấy 5 query KIS (mỗi query đã biết trước đáp án đúng là video nào + đoạn nào), chạy từng query qua FUFU, rồi dò danh sách top-20 trả về xem đáp án đúng đứng ở **hạng** mấy (∅ = không có trong top-20). 5 query là quá ít cho eval thật (xem §8.1) — ở đây chỉ để tính tay cho gọn:

| Query | Loại | Hạng của đáp án đúng |
|---|---|---|
| Q1 "người đàn ông nấu phở trong bếp" | visual | **1** |
| Q2 "biển hiệu 'Cơm tấm Ba Ghiền'" | ocr | **3** |
| Q3 "phát thanh viên nói về giá xăng tăng" | asr | **7** |
| Q4 "đoàn xe đạp đi qua cầu lúc hoàng hôn" | visual | **15** |
| Q5 "ca sĩ mặc áo dài đỏ hát trên sân khấu nổi" | visual | **∅** (không thấy) |

### 5.1 Recall@K (= hit rate với KIS)

Quy tắc: query "đậu" ở mức K nếu hạng ≤ K. Recall@K = số query đậu / tổng số query.

**Recall@1** — chỉ Q1 có hạng ≤ 1:

```
Recall@1 = 1/5 = 0.20  (20%)
```

**Recall@5** — Q1 (hạng 1) và Q2 (hạng 3) đậu; Q3 hạng 7 > 5 nên rớt:

```
Recall@5 = 2/5 = 0.40  (40%)
```

**Recall@20** — Q1, Q2, Q3, Q4 đều ≤ 20; Q5 không thấy:

```
Recall@20 = 4/5 = 0.80  (80%)
```

Nhận xét trực giác: Recall@K **đơn điệu tăng theo K** (nới lỏng tiêu chí thì chỉ thêm người đậu). Khoảng cách giữa Recall@5 (40%) và Recall@20 (80%) cho biết: rất nhiều đáp án "có tìm thấy nhưng đứng thấp" — đây chính là tín hiệu *"retrieval ổn, ranking yếu"* → đáng đầu tư vào rerank/fusion hơn là vào encoder.

Một cách nhìn bổ ích khác là **phân bố hạng** — thay vì 3 con số, vẽ histogram đáp án rơi vào vùng nào:

```
hạng 1      █████ 1 query     ← thấy ngay
hạng 2-5    █████ 1 query     ← màn hình đầu
hạng 6-20   ██████████ 2 query ← phải cuộn
rớt top-20  █████ 1 query     ← phải viết lại query
```

Hai hệ có cùng Recall@20 = 80% nhưng phân bố khác nhau (dồn về hạng 1-5 vs rải đều 6-20) cho trải nghiệm thi khác hẳn nhau — và MRR (mục tiếp) chính là cách nén phân bố này thành 1 con số.

### 5.2 MRR — Mean Reciprocal Rank

Recall@K thô ở một điểm: hạng 1 và hạng 5 được tính như nhau trong Recall@5, nhưng với operator đang thi, hạng 1 (liếc phát thấy ngay) khác hẳn hạng 5 (phải quét mắt). MRR sửa điều này bằng **điểm = 1/hạng** (reciprocal rank), không tìm thấy = 0:

| Query | Hạng | Reciprocal rank |
|---|---|---|
| Q1 | 1 | 1/1 = 1.000 |
| Q2 | 3 | 1/3 ≈ 0.333 |
| Q3 | 7 | 1/7 ≈ 0.143 |
| Q4 | 15 | 1/15 ≈ 0.067 |
| Q5 | ∅ | 0 |

```
MRR = (1.000 + 0.333 + 0.143 + 0.067 + 0) / 5 = 1.543 / 5 ≈ 0.309
```

Trực giác về thang đo: MRR = 1.0 nghĩa là *mọi* query đều hạng 1; MRR ≈ 0.5 đại khái là "đáp án trung bình quanh hạng 2"; phạt rất nặng việc tụt hạng đầu (từ hạng 1 → 2 mất 0.5 điểm, từ hạng 10 → 11 chỉ mất 0.009). Tính chất "phạt nặng phần đầu" này khớp tâm lý thi đấu: vài hạng đầu là vùng mắt operator nhìn đầu tiên.

### 5.3 nDCG — biết để đọc paper, ít dùng cho KIS

Khi mỗi query có **nhiều** kết quả relevant với **nhiều mức** độ liên quan (rel = 2: rất khớp, 1: tạm, 0: sai), Recall@K và MRR không phân biệt được. nDCG (normalized Discounted Cumulative Gain) xử lý bằng cách cộng điểm relevance có **chiết khấu theo hạng**:

```
DCG@K = Σ rel_i / log2(i + 1)    (i = hạng, từ 1 đến K)
nDCG@K = DCG@K / IDCG@K          (IDCG = DCG của thứ tự lý tưởng)
```

Ví dụ tính tay nhanh: hệ trả 3 kết quả có rel lần lượt [1, 2, 0]:

```
DCG  = 1/log2(2) + 2/log2(3) + 0/log2(4) = 1.000 + 1.262 + 0 = 2.262
Thứ tự lý tưởng là [2, 1, 0]:
IDCG = 2/log2(2) + 1/log2(3) + 0          = 2.000 + 0.631 + 0 = 2.631
nDCG = 2.262 / 2.631 ≈ 0.86
```

Với KIS của FUFU chỉ có 1 đáp án đúng (rel chỉ có 0/1) → nDCG suy biến gần thành 1 biến thể của MRR, nên **team không cần dùng**; biết khái niệm để khi đọc paper retrieval (MS MARCO, BEIR...) thấy "nDCG@10" thì hiểu nó đo gì.

### 5.4 Metric nào phản ánh trải nghiệm thi?

Trên UI FUFU, ~5 kết quả đầu nằm ngay màn hình đầu tiên, không cần cuộn. Operator thi sẽ liếc top-5 trước, cuộn nếu chưa thấy, đổi query nếu hết top-20.

| Metric | Trả lời câu hỏi | Phản ánh trải nghiệm thi? |
|---|---|---|
| Recall@1 | "Kết quả đầu tiên đúng luôn?" | Tốt nhưng khắt khe — thi không cần đúng ngay hạng 1 |
| **Recall@5** | "Đáp án trong màn hình đầu?" | ⭐ **Sát nhất** — thấy là click, gần như không tốn thời gian |
| Recall@20 | "Có cứu được bằng cách cuộn?" | Đo "trần" của retrieval; rớt @20 = phải đổi query (tốn nhiều giây) |
| MRR | "Trung bình đứng cao cỡ nào?" | Nhạy để so 2 phiên bản khi Recall@5 bằng nhau |
| nDCG | "Xếp hạng nhiều mức relevance tốt không?" | Không cần cho KIS |

**Quy ước team:** Recall@5 là metric chính để ra quyết định merge; Recall@1/@20 và MRR là metric phụ để chẩn đoán.

### 5.5 Đọc một bảng eval như thế nào

Giả sử eval harness in ra bảng so sánh trước/sau một thay đổi (bộ 80 query):

| Metric | Baseline | Sau thay đổi | Δ |
|---|---|---|---|
| Recall@1 | 31.3% | 32.5% | +1.2 |
| Recall@5 | 56.3% | 57.5% | +1.2 |
| Recall@20 | 76.3% | 68.8% | **−7.5** |
| MRR | 0.412 | 0.428 | +0.016 |

Đọc vội thì "Recall@5 với MRR đều tăng, merge thôi". Đọc kỹ thì thấy chuyện đáng lo: **Recall@20 giảm 7.5%** = 6 query trước đây còn cứu được bằng cách cuộn, giờ rớt hẳn khỏi top-20 → operator phải viết lại query (chi phí lớn nhất). Thay đổi này đánh đổi "đứng cao hơn một chút khi tìm thấy" lấy "mất hẳn vài đáp án". Bước tiếp theo bắt buộc: mở **diff theo từng query id** xem 6 query nào rớt, chúng thuộc type nào — nếu cùng một type (ví dụ toàn `asr`) thì thay đổi đã làm hỏng một kênh. Bài học: **không bao giờ đọc một con số đơn lẻ; đọc cả bộ + phân rã theo loại query + diff từng query.**

---

## 6. Metric riêng cho format thi: thời gian và lần submit sai

Đề thi HCM AIC chấm theo **thời gian tìm thấy** + **số lần submit sai** (xem RESEARCH-PLAN §1.2) — không chấm recall. Recall offline chỉ là *proxy*. Thời gian thực tế của operator gồm:

```
T_tổng = T_gõ query + T_hệ thống trả lời + T_mắt quét tìm đáp án trên UI + T_verify + T_submit
            (người)      (LATENCY — đo được)    (phụ thuộc VỊ TRÍ đáp án)      (người)
```

Hai thành phần hệ thống kiểm soát được, và **phải đo cùng lúc với recall**:

1. **Latency** — FUFU đã có sẵn `timing_ms` trong response `/api/search` (PROJECT-CONTEXT §10): `expand_ms`, `encode_ms`, `faiss_ms`, `bm25_visual_ms`, `bm25_asr_ms`, `fetch_meta_ms`, `rerank_ms`, `cross_rerank_ms`. Eval script chỉ việc cộng dồn và báo cáo **trung bình + p95**. Vì sao cần p95 chứ không chỉ trung bình: trải nghiệm thi bị quyết định bởi *query chậm nhất bạn gặp lúc đang căng thẳng*, không phải query trung bình — trung bình 0.6s nhưng p95 = 4s nghĩa là cứ ~20 query lại có 1 lần ngồi chờ 4 giây. Một thay đổi tăng Recall@5 thêm 2% nhưng tăng latency từ 0.5s lên 3s có thể **lỗ** trong giờ thi (mỗi query thử chậm thêm 2.5s × hàng chục query thử = nhiều phút).
2. **Vị trí đáp án trên UI** — hạng 1-5: không cuộn (~0s); hạng 6-20: cuộn + quét (~5-15s); rớt top-20: viết lại query (~30-60s + rủi ro submit nhầm). MRR và phân bố hạng chính là proxy số hoá của chi phí này.

Còn **submit sai** thì sao? Nó xảy ra khi một kết quả *trông giống* đáp án (cùng bối cảnh, khác đoạn). Hệ thống giảm rủi ro này bằng thông tin giúp operator verify nhanh: thumbnail rõ, `segment_start/end` chính xác, `score_breakdown` (thấy hit đến từ kênh nào — một hit thuần BM25 ASR với điểm dense thấp đáng nghi hơn với query visual). Eval offline không đo trực tiếp được submit sai, nhưng match rule "đúng video + đúng khoảng thời gian" (§7 bước 3) chính là phiên bản tự động của hành vi verify đó.

Breakdown `timing_ms` còn cho biết *tiền đang tốn ở đâu* khi cần tối ưu. Ví dụ một dòng đo thực tế có dạng:

| Thành phần | ms | Nhận xét |
|---|---|---|
| `expand_ms` (dịch + paraphrase) | 350 | thành phần nặng nhất — ứng viên cache/tắt bớt |
| `encode_ms` (SigLIP text) | 40 | rẻ |
| `faiss_ms` | 15 | gần như miễn phí |
| `bm25_visual_ms` + `bm25_asr_ms` | 25 | rẻ |
| `cross_rerank_ms` (BGE top-50) | 220 | đáng giá nếu nó kéo MRR lên — eval sẽ trả lời |
| **Tổng** | ~650 | đạt mục tiêu <1s |

Khi bảng eval ghi cả cột latency, câu hỏi "reranker có đáng 220ms không?" trở thành phép trừ hai dòng trong bảng thay vì cuộc tranh luận cảm tính.

Cuối cùng, có một lớp đo mà script không thay được: **mock contest** — một người làm operator, người khác bấm giờ, chạy 10-15 query đề cũ trên UI thật. Nó đo những thứ offline metric mù: thời gian gõ, thời gian quét mắt trên grid thumbnail, số lần suýt submit nhầm. Recall@5 offline là vòng lặp hàng ngày (rẻ, chạy mọi PR); mock contest là vòng lặp hàng tuần/trước thi (đắt, nhưng là thứ duy nhất đo đúng cái BTC chấm). RESEARCH-PLAN §5 đã xếp "tập dợt mock contest" vào tuần 7-8 — đừng cắt nó.

Vậy bảng eval chuẩn của FUFU có **3 cột nhóm**: Recall@{1,5,20} + MRR | latency trung bình/p95 | phân bố hạng. Một dòng cho mỗi phiên bản hệ thống.

> 🔗 **Trong FUFU:** field `timing_ms` được build ở cuối `app/backend/services/search_engine.py` (hàm `search`) và `scripts/eval_accuracy.py` hiện đã đo `elapsed_ms` mỗi query (hàm `eval_one`, dòng `t0 = time.time()`). Khi viết eval harness mới, tái dùng đúng cơ chế này.

---

## 7. Xây eval set cho FUFU — hướng dẫn từng bước

Đây là phần thực dụng nhất chương. Mục tiêu: một bộ `eval_set.json` + một script, chạy 1 lệnh ra bảng số.

### Bước 1 — Chọn 30-50 video đa dạng từ dataset

Không chọn ngẫu nhiên hoàn toàn — chọn **có chủ đích để phủ các loại tín hiệu** mà các kênh của FUFU dựa vào:

| Nhóm video | Tín hiệu chính | Vì sao cần trong eval set |
|---|---|---|
| Tin tức trường quay | ASR dày + chữ chạy trên màn | Kiểm kênh BM25 ASR + OCR phụ đề |
| Phóng sự ngoài trời | Thuần visual | Kiểm kênh dense SigLIP "trần trụi" |
| Sự kiện có banner/biển hiệu | OCR chữ Việt (có dấu, cách điệu) | Điểm yếu đã biết của EasyOCR — cần theo dõi |
| Phỏng vấn | ASR + khuôn mặt | Query kiểu "ông X nói về Y" |
| Cảnh tối / mưa / đám đông | Visual khó | Giữ bộ không "kịch trần", phân biệt được các phiên bản |

Ghi lại danh sách video này vào file — eval set gắn chặt với snapshot dataset (xem §8.4).

### Bước 2 — Viết 50-100 query kiểu đề thi

Lấy format đề các mùa cũ làm template (mô tả một cảnh/sự kiện cụ thể, đủ chi tiết để chỉ khớp 1 chỗ). Phân bố loại query nên **phản ánh đề thật** — đề AIC trên corpus tin tức thường nặng cả 4 loại:

| Loại (`type`) | Tín hiệu cần | Tỷ lệ gợi ý | Ví dụ |
|---|---|---|---|
| `visual` | Thuần hình ảnh | ~40% | "ba đứa trẻ thả diều trên đồi cỏ, một con diều hình cá" |
| `ocr` | Chữ trên màn | ~20% | "đoạn có dòng chữ 'KHU VỰC CẤM LỬA' trên hàng rào" |
| `asr` | Lời thoại | ~25% | "người dẫn nói 'mức tăng cao nhất trong mười năm'" |
| `entity` | Tên riêng (người/địa danh) | ~15% | "ông Nguyễn Văn A phát biểu tại lễ khánh thành cầu X" |

Mẹo viết query không bị "quá dễ": người viết query **xem video trước**, sau đó **đóng video lại và mô tả theo trí nhớ** — bắt chước cách BTC ra đề (mô tả tự nhiên, không copy nguyên văn caption/transcript). Xem thêm pitfall §8.2.

Checklist chất lượng cho từng query trước khi đưa vào bộ:

- [ ] Mô tả đủ cụ thể để chỉ khớp ~1 chỗ trong corpus (không phải "một người đang nói" — khớp cả nghìn segment).
- [ ] Viết bằng tiếng Việt tự nhiên như giám khảo đọc đề, có cả query dài lẫn ngắn.
- [ ] Trộn độ khó: ~1/3 dễ (cảnh đặc trưng), ~1/2 trung bình, ~1/5 khó (cảnh tối, đông người, chữ mờ) — bộ toàn query dễ sẽ "kịch trần" sớm (mọi phiên bản đều đạt 95%, không phân biệt được gì).
- [ ] Vài query có **biến thể diễn đạt** của cùng một cảnh (kiểm tra độ bền với cách dùng từ — proxy cho query expansion).
- [ ] Không chứa nguyên văn cụm từ trong caption/transcript (trừ loại `ocr`/`asr` nơi trích lời là chủ đích của đề).

### Bước 3 — Ground truth = (video, khoảng thời gian chấp nhận)

Với KIS, ground truth của mỗi query là: **video nào** + **từ giây nào đến giây nào**. Nới khoảng thời gian một chút (±2-5s quanh sự kiện) để không phạt oan segment cắt lệch biên. Tiêu chí match khi chấm:

```
match(kết_quả, GT) =  đúng video
                   AND khoảng [segment_start, segment_end] GIAO với [gt.start, gt.end]
```

(So sánh: `eval_accuracy.py` hiện tại chỉ match **substring trên tên file** — đủ cho smoke test, nhưng không kiểm tra timestamp → một hit ở phút 12 trong khi đáp án ở phút 3 vẫn được tính "đúng". Bộ mới phải chặt hơn.)

### Bước 4 — Lưu JSON theo schema cố định

Đề xuất schema (file `eval/eval_set.json`):

```json
{
  "version": "2026-06-15",
  "dataset_note": "batch-01, 42 video, ingest commit d4bf91e",
  "queries": [
    {
      "id": "Q001",
      "type": "visual",
      "query": "người đàn ông áo trắng nấu phở, hơi nước bốc lên trong bếp nhà hàng đông khách",
      "gt": {"video": "L01_V0012.mp4", "start": 125.0, "end": 143.0}
    },
    {
      "id": "Q002",
      "type": "ocr",
      "query": "cảnh quay biển hiệu quán 'Cơm tấm Ba Ghiền' lúc trời tối, đèn neon màu vàng",
      "gt": {"video": "L03_V0007.mp4", "start": 41.0, "end": 52.0}
    },
    {
      "id": "Q003",
      "type": "asr",
      "query": "phát thanh viên nữ nói giá xăng tăng lần thứ ba liên tiếp trong tháng",
      "gt": {"video": "L02_V0031.mp4", "start": 300.0, "end": 330.0}
    }
  ]
}
```

Lưu ý: `version` + `dataset_note` bắt buộc — kết quả eval chỉ có nghĩa khi biết nó đo trên snapshot nào (xem pitfall §8.4). Mỗi query có `id` cố định để diff kết quả giữa 2 lần chạy ("Q017 từ hạng 3 tụt xuống ∅ sau thay đổi X").

### Bước 5 — Script chấm

Logic cốt lõi (ngắn gọn — đầy đủ thì kế thừa khung của `eval_accuracy.py`):

```python
for case in eval_set["queries"]:
    res = engine.search(case["query"], top_k=20)
    rank = None
    for i, r in enumerate(res["results"], 1):
        same_video = case["gt"]["video"] in (r["item_path"] or "")
        overlap = r["segment_start"] <= case["gt"]["end"] and \
                  r["segment_end"]   >= case["gt"]["start"]
        if same_video and overlap:
            rank = i
            break
    # ghi (id, type, rank, timing_ms) → tính Recall@K, MRR, latency theo §5-6
```

Output: bảng tổng (Recall@1/5/20, MRR, avg/p95 latency) + bảng **theo từng `type`** (visual/ocr/asr/entity) + danh sách failure kèm top-3 trả về. Bảng theo type là công cụ chẩn đoán chính: "Recall@5 tổng giảm 3%" ít thông tin; "nhóm `ocr` sập từ 60% → 20% sau khi đổi OCR engine" chỉ thẳng thủ phạm.

### Bước 6 — Chạy và đọc kết quả

Một lần chạy chuẩn cho output dạng (mô phỏng):

```
$ python scripts/eval_kis.py --cases eval/eval_set.json
OVERALL (80 queries)              By type — Recall@5:
  Recall@1 : 25/80 (31.3%)          visual (32q): 62.5%
  Recall@5 : 45/80 (56.3%)          ocr    (16q): 43.8%   ← yếu nhất
  Recall@20: 61/80 (76.3%)          asr    (20q): 65.0%
  MRR:       0.412                  entity (12q): 41.7%   ← yếu nhì
  Latency:   avg 640ms · p95 1100ms

Failures (19): [ocr] Q014 'biển hiệu CẤM LỬA...' → top1: v:L05_V003 ...
```

Cách đọc: cột phải nói cho team biết tuần tới nên làm gì — `ocr` và `entity` yếu nhất gợi ý ưu tiên D2 (nâng OCR) và B2 (external image fallback cho entity) trong RESEARCH-PLAN §3, thay vì tiếp tục đánh bóng kênh visual vốn đã ổn. Eval harness không chỉ để gác cổng merge — nó còn là **la bàn chọn việc**.

Lưu mỗi lần chạy ra file (`eval/runs/2026-06-15_baseline.json` chứa rank từng query) để lần sau diff được từng query id.

### Ghi chú mở rộng — eval cho temporal/TRAKE (khi team làm A1/A2)

Bộ KIS ở trên là nền. Khi team triển khai temporal pair query (A1) hay TRAKE alignment (A2) theo RESEARCH-PLAN, eval set cần thêm 2 loại entry — chuẩn bị schema từ bây giờ thì sau không phải đập đi xây lại:

- **Temporal pair**: query = cặp mô tả `["xe cứu hỏa chạy đến", "ngọn lửa được dập tắt"]`, GT = video + 2 khoảng thời gian theo thứ tự. Match = đúng video + cả 2 khoảng đều được "bắt" đúng thứ tự.
- **TRAKE**: query = chuỗi N moment trong 1 video cho trước, GT = N mốc frame; metric tự nhiên là "% moment được align đúng (trong dung sai ±t giây)" — vẫn là tinh thần hit-rate, chỉ đổi đơn vị từ query sang moment.

Cách đo không đổi (so trước/sau, dán vào §6 RESEARCH-PLAN); chỉ định nghĩa "match" là mở rộng.

> 🔗 **Trong FUFU:** khung sẵn có để kế thừa nằm ở `scripts/eval_accuracy.py` (vòng lặp eval, Recall@K + MRR + báo cáo theo channel + failure list — chính là 80% script ở bước 5). Chỗ cần sửa: thay match substring bằng match video + overlap thời gian (bước 3), và đọc schema mới (bước 4). Kết quả dán vào bảng ở `RESEARCH-PLAN.md` §6.

---

## 8. Pitfalls — những cách tự lừa mình bằng con số

### 8.1 Eval set quá nhỏ

Mỗi query trong bộ là một "lá phiếu" 0/1, nên độ phân giải của Recall bị chặn bởi kích thước bộ:

| Kích thước bộ | 1 query đổi kết quả = | Chênh lệch tối thiểu nên tin (~3 query) |
|---|---|---|
| 30 query | ±3.3% | ~10% |
| 50 query | ±2.0% | ~6% |
| 100 query | ±1.0% | ~3% |

Nghĩa là chênh lệch "Recall@5: 62% vs 64%" trên bộ 50 query chỉ là **1 query đổi phe** — nhiễu thống kê, không phải cải tiến (que đó có thể đổi phe vì bất kỳ lý do vớ vẩn nào: một paraphrase khác đi, một tie-break trong sort). Quy tắc thô: chỉ tin chênh lệch ≥ 2-3 query tuyệt đối; muốn phát hiện cải tiến nhỏ thì cần bộ lớn hơn (100+) hoặc nhìn thêm MRR (liên tục, mịn hơn hit-rate) + diff từng query theo `id`.

### 8.2 Query viết bởi người đã biết đáp án

Người vừa xem caption "a man cooking noodle soup in a kitchen" rồi viết query "người đàn ông nấu súp mì trong bếp" → query khớp gần nguyên văn annotation → điểm eval cao **ảo**, không phản ánh query của giám khảo (người mô tả cảnh theo trí nhớ, dùng từ khác). Phòng tránh: người viết query không nhìn caption/transcript khi viết (đóng video, mô tả theo trí nhớ — §7 bước 2); lý tưởng là người A chọn cảnh, người B (chỉ được xem cảnh) viết query.

### 8.3 Overfit chính eval set

Tune trọng số 30 vòng trên cùng 1 bộ 50 query (chương 17) → bạn đang "học thuộc" bộ đó: tham số tốt nhất cho 50 query này chưa chắc tốt cho đề thi. Đây chính là overfitting quen thuộc, chỉ khác chỗ "tham số" là weights/threshold thay vì trọng số mạng. Phòng tránh kinh điển: **chia dev/holdout** — ví dụ với bộ 100 query: 70 vào dev (tune thoải mái, chạy bao nhiêu vòng cũng được), 30 vào holdout (khoá lại, chỉ chạy khi chốt phiên bản trước milestone). Chia **phân tầng theo `type`** (mỗi bên giữ đúng tỷ lệ visual/ocr/asr/entity) để holdout không lệch. Nếu dev tăng mà holdout đứng yên → bạn đang overfit dev set, các "cải tiến" gần nhất là ảo.

### 8.4 Eval set không cập nhật khi dataset đổi

Re-ingest với threshold shot khác, thêm batch video mới, đổi cách cắt segment → khoảng thời gian GT có thể không còn khớp segment nào, hoặc Recall thay đổi chỉ vì corpus to lên (nhiều distractor hơn). Số đo trên snapshot cũ **không so sánh được** với snapshot mới. Vì vậy schema §7 có `version`/`dataset_note`: mọi bảng số phải ghi rõ đo trên snapshot nào; dataset đổi → bump version, chạy lại baseline, không so chéo version.

### 8.5 Quên rằng pipeline có thành phần ngẫu nhiên

Query expansion của FUFU sinh paraphrase bằng sampling (temperature 0.7) → **hai lần chạy cùng query có thể cho paraphrase khác nhau → q_vec khác → ranking khác**. Nếu eval bật paraphrase, chênh lệch 1-2 query giữa hai lần chạy có thể chỉ là xổ số sampling, không phải thay đổi code. Lựa chọn: (a) tắt paraphrase khi eval (như `eval_accuracy.py` đang làm — đo "lõi" deterministic, nhưng lệch với cấu hình lúc thi), (b) cố định seed sampling, hoặc (c) chạy 3 lần lấy trung bình khi cần đo đúng cấu hình thi. Tối thiểu: **biết rõ mình đang eval với cấu hình nào** và ghi vào Ghi chú của bảng số — so sánh hai dòng đo với hai cấu hình expansion khác nhau là so táo với cam.

---

## 9. Quy trình làm việc chuẩn của team (từ nay)

1. **Trước khi sửa** bất cứ thứ gì đụng retrieval: chạy eval → ghi dòng **baseline**.
2. **Implement** thay đổi.
3. **Chạy lại eval** cùng bộ query, cùng snapshot dataset.
4. **Dán bảng số vào RESEARCH-PLAN.md §6** (bảng "Ý tưởng / Ngày / recall@1 / recall@5 / recall@20 / Latency / Ghi chú") — đây là sổ cái thí nghiệm của team.
5. **Luật merge:** không merge thứ làm giảm Recall@5; latency giữ <1s (RESEARCH-PLAN §5). Trade-off (recall tăng nhưng latency tăng) phải nêu rõ trong Ghi chú để cả team quyết.

Bảng §6 của RESEARCH-PLAN sau vài tuần sẽ trông như thế này — và chính nó là lịch sử kỹ thuật đáng tin nhất của dự án:

| Ý tưởng | Ngày | recall@1 | recall@5 | recall@20 | Latency | Ghi chú |
|---|---|---|---|---|---|---|
| (baseline) | 15/06 | 31.3% | 56.3% | 76.3% | 640ms | eval_set v2026-06-15, paraphrase OFF |
| C5 tune weights | 18/06 | 33.8% | 61.3% | 77.5% | 640ms | asr 0.5→0.35; nhóm visual +8% |
| B1 LLM rewrite | 24/06 | 35.0% | 63.8% | 80.0% | 780ms | +140ms expand; nhóm ocr +12% |

Mọi PR liên quan retrieval mà thiếu bảng số → reviewer có quyền (và nghĩa vụ) trả lại. Quy tắc này nghe quan liêu nhưng rẻ hơn rất nhiều so với việc phát hiện regression vào tuần thi.

Checklist 20 giây cho reviewer trước khi bấm merge một PR retrieval:

- [ ] Có bảng số trước/sau? Cùng eval set version, cùng cấu hình expansion?
- [ ] Recall@5 không giảm? (chênh lệch có ≥ 2-3 query không, hay là nhiễu §8.1?)
- [ ] Recall@20 và latency p95 không xấu đi âm thầm?
- [ ] Bảng theo `type` không có nhóm nào sập?
- [ ] Số đã được dán vào RESEARCH-PLAN §6?

Hai mẹo để quy trình không bị bỏ rơi sau 2 tuần hứng khởi:

- **Một lệnh duy nhất.** Toàn bộ eval phải chạy được bằng đúng 1 lệnh không tham số bắt buộc (`python scripts/eval_kis.py`) và xong trong vài phút. Mỗi bước thủ công thêm vào (sửa config, copy file, chờ nửa tiếng) giảm một nửa xác suất nó được chạy.
- **Kết quả tự lưu, tự diff.** Script tự ghi `eval/runs/<ngày>_<nhãn>.json` và in sẵn diff với baseline gần nhất — người chạy chỉ việc copy bảng vào PR. Đừng bắt con người làm việc mà máy làm tốt hơn.

**Còn `scripts/eval_accuracy.py` + MSR-VTT dịch thì sao?** Hiện team có sẵn pipeline eval trên MSR-VTT đã dịch sang tiếng Việt (`scripts/download_msrvtt.py`, `translate_msrvtt_to_vn.py`). Nó **dùng tạm được** làm smoke test (hệ có chạy không, thay đổi có phá vỡ gì thô không) — nhưng **lệch domain** so với đề thi: MSR-VTT là caption mô tả chung chung kiểu "a man is singing" (dịch máy lại càng phẳng), video ngắn không có cấu trúc tin tức, không có query OCR/ASR/entity tiếng Việt, và match chỉ ở mức video (không có timestamp). Đề AIC là mô tả cảnh cụ thể trên corpus tin tức VN dài, chấm đúng video + đúng đoạn. Tối ưu hệ theo MSR-VTT có thể kéo hệ **lệch khỏi** đề thật. Vì vậy: giữ MSR-VTT làm smoke test, còn quyết định merge dựa trên **bộ eval riêng theo format thi** xây ở §7 — đó chính là deliverable của ý F1.

---

## 10. Tóm tắt 10 giây

- Không đo được = không cải tiến được; eval harness (F1) đi trước mọi tuning/ensemble/finetune.
- KIS: Recall@K = hit rate ("đáp án có trong top-K?"), MRR = 1/hạng trung bình (phạt nặng đứng thấp); **Recall@5 là metric quyết định merge** vì khớp màn hình đầu của operator.
- Thi chấm theo thời gian + submit sai → đo kèm latency (`timing_ms`) và vị trí đáp án, không chỉ recall.
- Eval set: 30-50 video đa dạng, 50-100 query 4 loại (visual/ocr/asr/entity) phản ánh đề thật, GT = (video, khoảng thời gian), JSON có version, match = đúng video + overlap thời gian.
- Cảnh giác: bộ nhỏ → nhiễu ±2-3%/query; query viết khi nhìn caption → quá dễ; tune nhiều → giữ holdout; dataset đổi → bump version eval set; paraphrase sampling → ghi rõ cấu hình khi đo.
- Mọi thay đổi retrieval: chạy eval trước/sau → dán số vào RESEARCH-PLAN §6. Không có bảng số = không review.
- MSR-VTT dịch = smoke test; quyết định merge dựa trên bộ eval tự xây theo format thi.

---

## 11. Câu hỏi tự kiểm tra

**1. Hệ trả top-20 cho 4 query, đáp án đúng đứng hạng: 2, 5, ∅, 10. Tính Recall@1, Recall@5, Recall@20 và MRR.**

<details><summary>Đáp án</summary>

- Recall@1: không query nào hạng ≤1 → 0/4 = **0%**
- Recall@5: hạng 2 và 5 đậu → 2/4 = **50%**
- Recall@20: hạng 2, 5, 10 đậu → 3/4 = **75%**
- MRR = (1/2 + 1/5 + 0 + 1/10)/4 = (0.5 + 0.2 + 0 + 0.1)/4 = 0.8/4 = **0.2**
</details>

**2. Vì sao precision@20 gần như vô nghĩa với bài toán KIS của FUFU?**

<details><summary>Đáp án</summary>

KIS chỉ có 1 đáp án đúng cho mỗi query, nên trong 20 kết quả trả về tối đa 1 cái relevant → precision@20 tối đa = 1/20 = 5% dù hệ làm hoàn hảo. Con số luôn "xấu" và không phân biệt được hệ tốt/kém. Thông tin hữu ích nằm ở chỗ đáp án *có mặt không* (Recall@K) và *đứng thứ mấy* (MRR), không phải tỷ lệ relevant trong danh sách.
</details>

**3. Hai phiên bản có cùng Recall@5 = 60% trên cùng bộ query. Phiên bản A có MRR 0.55, B có MRR 0.41. Chọn cái nào cho thi đấu, vì sao?**

<details><summary>Đáp án</summary>

Chọn **A**. Cùng tỷ lệ "đáp án trong màn hình đầu", nhưng MRR cao hơn nghĩa là khi tìm thấy, đáp án của A đứng cao hơn (gần hạng 1 hơn) → operator quét mắt nhanh hơn, tiết kiệm thời gian — đúng tiêu chí chấm của cuộc thi (thời gian tìm thấy). MRR là tiêu chí phân thắng bại khi Recall@K hoà.
</details>

**4. Recall@5 = 40% nhưng Recall@20 = 80%. Khoảng cách lớn này gợi ý nên đầu tư vào phần nào của pipeline FUFU?**

<details><summary>Đáp án</summary>

Đáp án thường *được tìm thấy* (80% trong top-20) nhưng *xếp hạng thấp* (chỉ 40% lọt top-5) → khâu **ranking/fusion/rerank** yếu chứ không phải khâu retrieval (recall trần đã khá). Đáng đầu tư: tune trọng số hybrid (chương 17), cải thiện cross-encoder rerank, hơn là đổi encoder. Ngược lại nếu Recall@20 cũng thấp → vấn đề ở retrieval/index/annotation.
</details>

**5. Vì sao match theo substring tên file (như `eval_accuracy.py` hiện tại) là chưa đủ cho eval theo format thi, và tiêu chí match đúng là gì?**

<details><summary>Đáp án</summary>

Đề thi yêu cầu đúng video **và đúng đoạn thời gian**; match theo tên file sẽ tính "đúng" cả khi hệ trả về phút 12 trong khi đáp án ở phút 3 — thổi phồng điểm và không phát hiện lỗi cắt/gán segment. Tiêu chí đúng: cùng video **và** khoảng `[segment_start, segment_end]` của kết quả giao với khoảng thời gian ground truth (đã nới ±vài giây).
</details>

**6. Team tune trọng số hybrid 25 vòng trên bộ 60 query, Recall@5 tăng từ 55% → 70%. Có nên tin con số 70%? Phải làm gì?**

<details><summary>Đáp án</summary>

Không tin hoàn toàn — tune nhiều vòng trên cùng một bộ là overfit chính bộ đó (tham số "học thuộc" 60 query này). Cách kiểm tra: giữ một bộ **holdout** (ví dụ ~20-30 query tách riêng từ đầu, không bao giờ dùng để tune) và chỉ chạy nó khi chốt. Nếu holdout cũng tăng tương ứng → cải tiến thật; nếu holdout đứng yên trong khi dev tăng → đang overfit eval set.
</details>

**7. Trên bộ 40 query, thay đổi X làm Recall@5 tăng từ 60.0% lên 62.5%. Có nên kết luận X tốt hơn?**

<details><summary>Đáp án</summary>

Chưa. 2.5% trên 40 query = đúng **1 query** đổi trạng thái — hoàn toàn có thể là nhiễu (1 query nhạy cảm với thay đổi nhỏ bất kỳ). Nên: (a) xem diff từng query theo `id` để hiểu query nào đổi và vì sao, (b) nhìn thêm MRR và Recall@1/@20 có cùng chiều không, (c) nếu vẫn mơ hồ, mở rộng bộ query. Quy tắc thô: chỉ tin chênh lệch ≥ 2-3 query tuyệt đối.
</details>

**8. Vì sao không nên dùng kết quả eval trên MSR-VTT dịch để quyết định merge, dù nó có sẵn?**

<details><summary>Đáp án</summary>

Lệch domain so với đề thi ở mọi chiều: caption chung chung + dịch máy (khác văn phong mô tả cảnh cụ thể của giám khảo), video ngắn không phải tin tức VN, không có query loại OCR/ASR/tên riêng tiếng Việt, và chỉ match mức video (không kiểm timestamp). Tối ưu theo nó có thể kéo hệ lệch khỏi đề thật. Vai trò phù hợp: smoke test nhanh. Quyết định merge phải dựa trên bộ eval tự xây theo format thi (§7).
</details>

---

## 12. Đọc thêm

- **RESEARCH-PLAN.md** — ý F1 (§3 nhóm F), nguyên tắc §5, quy trình + bảng số §6. Sổ cái thí nghiệm của team nằm ở đó.
- **PROJECT-CONTEXT.md §10** — schema response `/api/search` (field `timing_ms`, `score_breakdown` dùng khi chẩn đoán failure).
- `scripts/eval_accuracy.py` — khung eval hiện có (Recall@K + MRR + per-channel + failure list) để kế thừa.
- *Introduction to Information Retrieval* (Manning, Raghavan, Schütze) — chương 8 "Evaluation in IR": chuẩn mực học thuật của các metric trong chương này (bản online miễn phí).
- [Results of the 2025 Video Browser Showdown (arXiv 2509.12000)](https://arxiv.org/html/2509.12000v1) — xem cách VBS chấm điểm theo thời gian + submit sai, đúng format mà eval của ta phải mô phỏng.
- [Event Retrieval from Large Video Collection in HCMC AI Challenge 2024 (Springer)](https://link.springer.com/chapter/10.1007/978-981-96-4291-5_1) — paper tổng kết chính thức: format đề + cách chấm của chính cuộc thi ta tham gia, là "spec" cho eval set §7.
- BEIR benchmark (arXiv 2104.08663) — ví dụ điển hình về vì sao eval lệch domain cho kết luận sai (zero-shot retrieval đổ vỡ khi đổi domain) — đúng bài học MSR-VTT §9.
