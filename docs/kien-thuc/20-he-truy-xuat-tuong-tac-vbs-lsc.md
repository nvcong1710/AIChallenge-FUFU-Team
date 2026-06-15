# Chương 20 — Hệ truy xuất tương tác & kỹ thuật thi đấu (VBS / LSC)

> *"Model thắng giải không phải model mạnh nhất, mà là model gắn vào đúng hệ thống và đúng đôi tay operator nhanh nhất."* — bài học lặp lại mỗi mùa Video Browser Showdown.

## 1. Vì sao chương này tồn tại trong FUFU

Mười chín chương trước dạy **bên trong** FUFU: encoder, retrieval, fusion, rerank, eval. Nhưng FUFU không tồn tại trong chân không — nó là một thí sinh trong một **dòng họ cuộc thi** đã chạy cả thập kỷ: **VBS** (Video Browser Showdown) và **LSC** (Lifelog Search Challenge). Hai cuộc thi này chính là phòng thí nghiệm nơi bài toán "tìm một cảnh trong biển video bằng mô tả ngôn ngữ" được mài giũa năm này qua năm khác. HCM AI Challenge mô phỏng gần như nguyên xi luật chơi của chúng.

Team vừa tải về và đọc một loạt tài liệu gốc của các hệ vô địch/kỳ cựu: **vitrivr** (stack tham chiếu mở của giới nghiên cứu), **lifeXplore** (vô địch LSC2023), **MEMORIA**, và bản tổng kết **VBS 2025**. Chương này biến đống tài liệu đó thành kiến thức dùng được: không phải để bắt chước y nguyên, mà để **hiểu các ý tưởng kiến trúc và "cửa truy vấn"** mà các đội mạnh đã hội tụ về sau nhiều năm thử-sai — rồi soi lại xem FUFU đang đứng đâu.

> 🔗 **Trong FUFU:** đây là phần "kiến thức nền" cho [RESEARCH-PLAN.md](../du-an/RESEARCH-PLAN.md) (§1.4 đặc biệt) và [KIEN-TRUC-VA-NGUYEN-TAC.md](../du-an/KIEN-TRUC-VA-NGUYEN-TAC.md). RESEARCH-PLAN nói *"làm gì"*; chương này nói *"vì sao những thứ đó là chuẩn mực"*. File gốc nằm ở [docs/nguon-tham-khao/](../nguon-tham-khao/README.md).

---

## 2. Cần biết trước

- **Chương 7** (CLIP/SigLIP): mọi hệ trong chương này đều lấy CLIP-family làm trụ dense — hiểu embedding chung không gian ảnh-text là bắt buộc.
- **Chương 13-14** (FAISS, BM25/hybrid): các hệ đều ghép dense (vector) với text/metadata filter — đúng mô hình hybrid đã học.
- **Chương 15** (pipeline FUFU): để đối chiếu "họ làm vs ta làm".
- **Chương 19** (eval): vì sao "đội thắng đo recall mỗi thay đổi" — chương này bổ sung *họ đo trên format thi nào*.
- Không cần: toán mới. Chương này nặng về **ý tưởng hệ thống**, nhẹ về công thức.

---

## 3. VBS và LSC là gì — và vì sao ta phải quan tâm

| | **VBS** (Video Browser Showdown) | **LSC** (Lifelog Search Challenge) |
|---|---|---|
| Bắt đầu | 2012 | 2018 |
| Tổ chức tại | hội nghị MMM | hội nghị ACM ICMR |
| Dữ liệu | video chung (tin tức V3C, biển, đời thường…) | ảnh đời sống cá nhân (egocentric, đeo camera) — 18 tháng, ~700k ảnh |
| Truy vấn | mô tả cảnh bằng ngôn ngữ | mô tả khoảnh khắc cá nhân |
| Loại task | **KIS** (Known-Item Search), **AVS/AS** (Ad-hoc — tìm *nhiều* cảnh cùng chủ đề), **Q&A** | KIS, AS, Q&A |
| Chấm | theo **thời gian tìm thấy** + **phạt submit sai** (live, vài giờ) | tương tự |

Điểm mấu chốt: **format chấm là live và theo thời gian**. Không có chuyện "nộp file kết quả rồi về" — operator ngồi trước màn hình, gõ query, hệ trả về grid kết quả, operator quét mắt, click, submit; sai bị trừ điểm; nhanh được điểm cao. Đây chính xác là kịch bản HCM AIC. Hệ quả trực tiếp cho FUFU: **tốc độ end-to-end và chất lượng top-5 quan trọng ngang ngửa độ chính xác top-100** (xem lại chương 19 §6 về latency và vị trí đáp án).

Vì sao LSC (ảnh đời sống) lại liên quan tới FUFU (video tin tức)? Vì **kỹ thuật chuyển thẳng**: cùng là "dùng CLIP làm dense + các kênh phụ (OCR, concept, metadata) + UI duyệt nhanh + truy vấn thời gian trong một ngày/một video". Domain khác, xương sống giống. Hai hệ ta đọc kỹ nhất (lifeXplore, MEMORIA) đến từ LSC chính vì lý do đó.

---

## 4. Mẫu kiến trúc tham chiếu: vitrivr

**vitrivr** (Rossetto và cộng sự, ACM MM 2016) là hệ truy xuất multimedia **mã nguồn mở** được trích dẫn nhiều nhất trong giới này — vẫn dự thi VBS 2025/2026 dưới tên *vitrivr-engine*. Đáng học vì nó là bản mẫu sạch sẽ của đúng mô hình FUFU đang theo.

Ba tầng **tách bạch hoàn toàn**:

```
┌─────────────┐   ┌──────────────────────┐   ┌──────────────┐
│  Cineast    │   │ Cottontail DB /       │   │  vitrivr-ng  │
│ trích đặc   │──▶│ ADAMpro               │◀──│  UI web      │
│ trưng +     │   │ (lưu vector + boolean,│   │ (gõ query,   │
│ truy vấn    │   │  tra cứu ANN)         │   │  duyệt KQ)   │
└─────────────┘   └──────────────────────┘   └──────────────┘
   EXTRACTOR            INDEX/STORE                UI
```

Nghe quen không? Đây **đúng** ba tầng trong [KIEN-TRUC-VA-NGUYEN-TAC.md §3](../du-an/KIEN-TRUC-VA-NGUYEN-TAC.md): extractor / index / UI nói chuyện qua contract dữ liệu. vitrivr đã chứng minh từ 2016 rằng tách tầng kiểu này cho phép thay từng phần độc lập — đó là lý do nó sống được 10 năm qua nhiều thế hệ model (từ feature thủ công màu/cạnh → CLIP).

**Bài học kiến trúc số 1:** một hệ truy xuất tốt không phải một khối liền — nó là ba hộp rời gắn bằng contract. FUFU sinh ra đã đúng hướng này; vitrivr cho ta sự tự tin rằng hướng đó bền.

---

## 5. "Nhiều cửa truy vấn" — và mỗi cửa là một concept

Điểm chung lớn nhất của mọi hệ top: chúng không chỉ có **một** ô gõ text. Chúng có **nhiều cách hỏi**, và hợp nhất kết quả. Đây là hiện thực của nguyên tắc P3 ([KIEN-TRUC §2](../du-an/KIEN-TRUC-VA-NGUYEN-TAC.md)). Dưới đây là từng "cửa" như một khái niệm — hiểu nó làm gì và khi nào hữu ích:

| Cửa truy vấn | Operator làm gì | Khớp tốt khi | Hệ tiêu biểu |
|---|---|---|---|
| **Query-by-text** | gõ mô tả ngôn ngữ tự nhiên | mặc định, luôn có | tất cả |
| **Query-by-example (QBE)** | dán 1 ảnh → tìm ảnh giống | đã có 1 frame mẫu (Visual KIS) | vitrivr, mọi hệ |
| **Query-by-sketch (QBS)** | vẽ phác **màu + cạnh** bố cục cảnh | nhớ bố cục/màu nhưng không có ảnh | vitrivr, lifeXplore, SnapSeek |
| **Query-by-motion** | vẽ **quỹ đạo chuyển động** của vật | nhớ "vật đi từ trái sang phải" | vitrivr |
| **Concept / object filter** | lọc theo nhãn định sẵn ("có con chó", "ngoài trời") | thu hẹp nhanh không gian tìm | lifeXplore, MEMORIA |
| **Metadata filter** | lọc theo thời gian/địa điểm/… | biết bối cảnh ("buổi tối, ở Hà Nội") | lifeXplore, MEMORIA |
| **Temporal query (A→B)** | mô tả **chuỗi** sự kiện theo thứ tự | nhớ "X xảy ra rồi mới tới Y" | lifeXplore, NII-UIT, PraK |

Và bên **đầu ra/duyệt** (không phải cách hỏi mà là cách *nhìn* kết quả) cũng là một mặt trận:

- **SOM / feature-map browsing**: thay vì danh sách dọc, xếp các keyframe **giống nhau cạnh nhau** thành một lưới 2D (Self-Organizing Map). Mắt người quét lưới-tương-đồng nhanh hơn nhiều so với cuộn list — vì cảnh cần tìm thường nằm gần các cảnh giống nó. lifeXplore và diveXplore dựa nhiều vào kiểu duyệt này.

> 🔑 **Trực giác cốt lõi:** mỗi cửa truy vấn = **một nguồn bằng chứng độc lập** về cùng một cảnh. Cảnh khó-cho-text có thể dễ-cho-sketch hoặc dễ-cho-OCR. Thêm cửa = thêm cơ hội bắt trúng, miễn là khâu fusion (chương 14, 18) biết hợp nhất chúng. Đó là vì sao "ensemble nhiều cửa" gần như luôn thắng "một cửa hoàn hảo".

FUFU hiện có: text, BM25 trên OCR/caption (≈ concept filter dạng full-text), BM25 trên ASR. **Chưa có**: QBE/QBS, query-by-motion, temporal, SOM browsing. Đó là bản đồ khoảng trống — khớp đúng các nhóm B/E của RESEARCH-PLAN.

---

## 6. Học sâu một kỹ thuật: temporal search kiểu lifeXplore

Đây là phần đáng tiền nhất chương, vì temporal là khoảng trống lớn nhất của FUFU và lifeXplore cho ta một **thuật toán cụ thể**.

**Bài toán:** operator nhớ một *chuỗi*: "Tôi thấy người đàn ông áo đỏ, **rồi sau đó** lên máy bay, **rồi** chạy marathon." Một query text gộp chung ("người áo đỏ máy bay marathon") sẽ thất bại — không frame nào chứa cả ba. Cần hỏi theo **thứ tự thời gian**.

lifeXplore cho phép gõ: `man with red shirt < plane flight < marathon` (dấu `<` ngăn các "phần"). Cách xử lý — **đi ngược từ phần CUỐI**:

```
1. Tìm phần CUỐI ("marathon")        → được danh sách frame ứng viên + thời điểm.
2. Với mỗi ứng viên marathon:
      tìm phần trước ("plane flight") CÙNG NGÀY và THỜI GIAN SỚM HƠN.
      Không có → loại ứng viên này.
3. Lặp lùi tiếp ("man with red shirt") với cùng ràng buộc sớm-hơn.
4. Ứng viên sống sót qua hết các phần = một chuỗi khớp đúng thứ tự → trả về.
```

Ví dụ số nhỏ (giả định trong 1 ngày, đơn vị = phút):

| Phần | Frame ứng viên (thời điểm) |
|---|---|
| áo đỏ | t=10, t=200 |
| máy bay | t=60, t=240 |
| marathon | t=120, t=300 |

Đi ngược từ marathon:
- marathon@120 → cần máy bay <120 → có @60 ✓ → cần áo đỏ <60 → có @10 ✓ → **chuỗi (10, 60, 120) hợp lệ**.
- marathon@300 → máy bay <300 → @240 ✓ → áo đỏ <240 → @200 ✓ → **chuỗi (200, 240, 300) hợp lệ**.

Hai chuỗi cùng ngày, đúng thứ tự → đều là ứng viên trả về.

**Vì sao đi ngược, không đi xuôi?** Vì phần cuối thường là cái operator nhớ rõ nhất / đặc trưng nhất ("marathon" hiếm hơn "người áo đỏ"). Bắt đầu từ tập ứng viên nhỏ nhất → ít nhánh phải kiểm tra hơn. (Một heuristic; có thể bắt đầu từ phần *đặc trưng nhất* thay vì luôn phần cuối.)

So sánh với cách khác trong RESEARCH-PLAN — **MADTempo cộng dồn similarity** qua các segment liên tiếp (mềm, không ràng buộc cứng "cùng ngày"). Hai triết lý:

| | lifeXplore `<` (lọc cứng) | MADTempo (cộng dồn mềm) |
|---|---|---|
| Bản chất | filter theo thứ tự + cùng-cửa-sổ | điểm số = tổng similarity các phần |
| Mạnh khi | mốc rõ ràng, cần đúng thứ tự | mô tả mờ, muốn xếp hạng mềm |
| Rủi ro | quá chặt → rớt cả chuỗi nếu 1 phần yếu | quá lỏng → khớp nhầm thứ tự |

> 🔗 **Trong FUFU:** đây là chất liệu trực tiếp cho ý **A1/A4** ([RESEARCH-PLAN §3](../du-an/RESEARCH-PLAN.md)). FUFU đã có `segments` với `start_sec/end_sec` và `item_id` (PROJECT-CONTEXT §6) — đủ dữ liệu để hiện thực ràng buộc "cùng item, thời gian tăng dần". Cửa sổ thích nghi (window rộng hẹp) là phần A4.

---

## 7. Học sâu kỹ thuật 2: ghép dense với filter có cấu trúc

lifeXplore mặc định dùng **FAISS + OpenCLIP** cho free-text, nhưng cho phép kèm filter object/concept/metadata (lưu trong **MongoDB**). Câu hỏi kỹ thuật: làm sao ghép "top-K từ vector search" với "lọc theo điều kiện cấu trúc"? Có hai chiến lược, và lifeXplore nêu rõ cả hai:

**Chiến lược 1 — vector trước, filter sau (lifeXplore dùng mặc định):**
```
"đang nấu ăn trong bếp -o banana"   (free-text + cần có object 'banana')
 1. FAISS: lấy top-K LỚN cho "đang nấu ăn trong bếp"  (vd K=5000 frame)
 2. MongoDB: trong 5000 id đó, lọc cái nào có object 'banana' → kết quả cuối
```
Ưu: tận dụng độ mạnh ngữ nghĩa của dense trước; phân trang gọn (giữ nguyên 5000 id, lật trang trên MongoDB). Nhược: nếu 'banana' hiếm, 5000 có thể chưa đủ → phải tăng K.

**Chiến lược 2 — phân trang nội bộ FAISS:** lấy ít trước, cần thêm thì nới K dần (dùng cho temporal, nơi cần nhiều ứng viên).

> 🔑 Bài học: "dense + filter cấu trúc" **không cần một CSDL thần kỳ** — chỉ cần một quy ước rõ ràng về *ai lọc trước*. FUFU hiện fuse dense và BM25 bằng cộng điểm có trọng số (chương 14); chiến lược "vector trước, filter sau" là một *cách khác* hữu ích khi filter là điều kiện cứng (phải có object X, phải trong khung giờ Y) thay vì tín hiệu mềm.

> 🔗 **Trong FUFU:** liên quan ý **E7** (concept/metadata filter panel) trong RESEARCH-PLAN. FUFU lưu `objects_json` trên `frames` và có FTS5 — đủ để thử chiến lược 1 mà không cần thêm DB.

Một mẹo phụ đáng giá từ lifeXplore: **position sub-filter**. Lúc ingest, từ bounding box của object họ tính sẵn vị trí thô (`top-left`, `bottom-right`…) và lưu lại, cho phép hỏi `-o person|position:top-left` ("người ở góc trên trái"). Đây là bản nhẹ của **localized query** mà PraK V4 (vô địch VBS2026) làm bài bản — gắn truy vấn vào *vị trí không gian* trong frame.

---

## 8. Học sâu kỹ thuật 3: graph database & event segmentation (MEMORIA)

**MEMORIA** (Ribeiro và cộng sự, U. Aveiro, LSC2023) đóng góp hai ý tưởng khác biệt:

**(a) Graph database thay vì quan hệ.** Họ so Neo4j (graph DB) với PostgreSQL cho truy xuất multimedia và thấy Neo4j nhanh hơn cho các truy vấn nhiều quan hệ (ảnh ↔ địa điểm ↔ hoạt động ↔ thời gian). Trực giác: khi dữ liệu là **mạng lưới quan hệ** ("ảnh này ở địa điểm A, cùng sự kiện với ảnh kia"), truy vấn theo đường đi trên graph tự nhiên hơn là JOIN nhiều bảng. **Khi nào đáng với FUFU?** Hiện tại *chưa* — quan hệ của FUFU còn đơn giản (item→segment→frame, tuyến tính). Nhưng nếu sau này thêm "scene graph" thật (đối tượng ↔ quan hệ ↔ đối tượng) hoặc liên kết entity xuyên video, graph DB là hướng đáng nhớ. Đây là kiến thức "để dành", không phải việc-làm-ngay.

**(b) Event segmentation phân cấp.** MEMORIA không cắt theo cửa sổ thời gian cứng — họ gom ảnh thành **"sự kiện"** dựa trên *ngữ nghĩa*: cùng địa điểm (từ GPS clustering), cùng hoạt động, cùng phương tiện di chuyển. Một "sự kiện" = một mạch tự nhiên của ngày. Liên hệ FUFU: đây đúng tinh thần `scenes` (gom shot kề nhau giống nhau, PROJECT-CONTEXT §6) — nhưng MEMORIA gom theo *tín hiệu ngữ nghĩa đa chiều*, không chỉ độ giống visual. Gợi ý nâng cấp: gom segment thành "sự kiện" dựa trên cả ASR (cùng chủ đề lời thoại) + visual + thời gian, không chỉ cosine biên.

MEMORIA cũng dùng **CLIP + ClipCap** (sinh caption từ CLIP feature) + YOLO concept — cùng họ với caption của FUFU (Qwen-VL), củng cố hướng **D1** (synthetic caption/query augmentation).

---

## 9. Bài học meta: vì sao các đội thắng giống nhau

Đọc đủ nhiều báo cáo VBS/LSC, bốn điều lặp lại mọi năm (đã đúc trong RESEARCH-PLAN §1.1, nhắc lại ở đây với lý do):

1. **Chênh lệch model giữa các đội top là nhỏ.** Ai cũng dùng CLIP-family mạnh. Model không còn là yếu tố phân thắng bại.
2. **Tốc độ end-to-end là vũ khí.** Thi tính giờ → hệ trả 0.5s thắng hệ trả 3s dù recall ngang nhau. (lifeXplore còn gộp query-server + index-server vào *một* process để bỏ overhead giao tiếp — chương 19 §6 đã bàn latency.)
3. **Nhiều cửa truy vấn bổ trợ** (§5) — không đội mạnh nào đặt cược một kênh.
4. **UI duyệt hiệu quả** (SOM, layout keyframe tối ưu mắt, phím tắt, submit nhanh) — nơi giây phút thật sự được tiết kiệm.

> 🔑 Hệ quả cho FUFU: sau khi phần "nhận diện nội dung" đã ngang mặt bằng (FUFU đã đủ OCR/ASR/caption/detection), **biên độ cải thiện lớn nhất KHÔNG nằm ở model tốt hơn** mà ở temporal + nhiều cửa + tốc độ + UI + eval kỷ luật. Đây là kim chỉ nam khi chọn việc.

---

## 10. FUFU đứng đâu trên bản đồ này

| Năng lực (theo chương này) | FUFU | Ghi chú |
|---|---|---|
| Kiến trúc 3 tầng tách rời (§4) | ✅ | đúng mô hình vitrivr |
| Dense CLIP-family (§5) | ✅ | SigLIP-2 Large (1 encoder; ensemble = C1) |
| Concept/OCR/ASR filter (§5) | ✅ | qua BM25 FTS5 |
| Metadata/position filter (§7) | ⚠️ | có `objects_json` nhưng chưa khai thác vị trí/thời gian làm filter |
| Temporal A→B (§6) | ❌ | khoảng trống lớn nhất — A1/A4 |
| QBE / QBS / motion (§5) | ❌ | B3/B5 |
| SOM / grid browsing (§5) | ❌ | E6 |
| Event segmentation ngữ nghĩa (§8) | ⚠️ | có `scenes` theo cosine, chưa đa tín hiệu |
| Tốc độ tối ưu cho thi (§9) | ⚠️ | chạy được, chưa tối ưu song song — E2 |

Đọc bảng này cùng [RESEARCH-PLAN §2](../du-an/RESEARCH-PLAN.md) (bảng đối chiếu chi tiết hơn) — chúng kể cùng một câu chuyện từ hai góc: chương này từ *"các hệ kinh điển làm gì"*, RESEARCH-PLAN từ *"ta nên làm gì trước"*.

---

## 11. Tóm tắt 10 giây

- VBS/LSC = phòng thí nghiệm của bài toán FUFU; chấm **live theo thời gian + phạt submit sai** → tốc độ và top-5 quý ngang độ chính xác.
- **vitrivr** = bản mẫu kiến trúc 3 tầng rời (Cineast/store/UI) — chính là mô hình FUFU; đã bền 10 năm vì tách tầng theo contract.
- **Nhiều cửa truy vấn** (text, QBE, sketch, motion, concept/metadata filter, temporal) + **SOM browsing**: mỗi cửa là một nguồn bằng chứng độc lập; ensemble nhiều cửa thắng một cửa hoàn hảo.
- **Temporal kiểu lifeXplore** (`<`, xử lý ngược từ phần đặc trưng nhất, ràng buộc cùng-item-thời-gian-tăng) — chất liệu cho A1; khác cách cộng dồn mềm của MADTempo.
- **Dense + filter cấu trúc**: "vector trước, filter sau" — không cần DB thần kỳ; position sub-filter = localized query nhẹ.
- **MEMORIA**: graph DB (để dành khi quan hệ phức tạp) + event segmentation theo ngữ nghĩa đa chiều (nâng cấp `scenes`).
- Bài học meta: model đã ngang nhau → thắng nhờ **temporal + nhiều cửa + tốc độ + UI + eval**, không nhờ model to hơn.

---

## 12. Câu hỏi tự kiểm tra

**1. Vì sao kỹ thuật của LSC (ảnh đời sống cá nhân) lại áp dụng được cho FUFU (video tin tức)?**

<details><summary>Đáp án</summary>

Domain khác nhưng xương sống giống hệt: CLIP-family làm dense + các kênh phụ (OCR, concept, metadata) + UI duyệt nhanh + truy vấn thời gian. Cùng loại task (KIS/AS/QA), cùng cách chấm (live, theo thời gian). Vì vậy thuật toán (temporal `<`, dense+filter, SOM browsing) chuyển thẳng; chỉ có dữ liệu và extractor cụ thể là khác.
</details>

**2. vitrivr chia hệ thành Cineast / Cottontail-DB / vitrivr-ng. Ánh xạ ba phần này sang ba tầng trong KIEN-TRUC-VA-NGUYEN-TAC §3 của FUFU.**

<details><summary>Đáp án</summary>

Cineast = tầng **extractor** (trích đặc trưng + xử lý truy vấn). Cottontail DB/ADAMpro = tầng **index/store** (lưu vector + boolean, tra cứu ANN). vitrivr-ng = tầng **UI**. Bài học: ba hộp rời nói chuyện qua contract dữ liệu → thay từng phần độc lập.
</details>

**3. Trong temporal search của lifeXplore, vì sao xử lý đi NGƯỢC từ phần cuối (hoặc phần đặc trưng nhất) thay vì từ phần đầu?**

<details><summary>Đáp án</summary>

Để bắt đầu từ **tập ứng viên nhỏ nhất**. Phần đặc trưng/hiếm nhất ("marathon") cho ít frame ứng viên hơn phần phổ biến ("người áo đỏ"). Khởi đầu từ tập nhỏ → ít nhánh phải kiểm tra ràng buộc thời gian → nhanh hơn. Đi xuôi từ phần phổ biến sẽ phải duyệt rất nhiều ứng viên vô ích.
</details>

**4. So sánh temporal "lọc cứng" (`<` của lifeXplore) và "cộng dồn similarity mềm" (MADTempo). Khi nào dùng cái nào?**

<details><summary>Đáp án</summary>

Lọc cứng: ràng buộc đúng thứ tự + cùng cửa sổ thời gian, hợp khi các mốc rõ ràng và thứ tự quan trọng; rủi ro rớt cả chuỗi nếu một phần yếu tín hiệu. Cộng dồn mềm: điểm = tổng similarity, hợp khi mô tả mờ và muốn xếp hạng linh hoạt; rủi ro khớp nhầm thứ tự. Thực tế có thể kết hợp: lọc cứng để loại, mềm để xếp hạng phần còn lại.
</details>

**5. FUFU cần lọc kết quả "đang nấu ăn" mà PHẢI có object 'dao'. Mô tả chiến lược "vector trước, filter sau" và một rủi ro của nó.**

<details><summary>Đáp án</summary>

Lấy top-K lớn từ dense cho "đang nấu ăn" (vd 5000 frame), rồi trong tập đó lọc các frame có object 'dao' (từ `objects_json`/FTS5). Rủi ro: nếu 'dao' hiếm, top-5000 có thể không chứa đủ frame có dao → bỏ sót; phải tăng K hoặc đảo chiến lược (lọc 'dao' trước rồi mới dense).
</details>

**6. Bài học meta nói "model đã ngang nhau giữa các đội top". Nó đổi cách team FUFU chọn việc ưu tiên như thế nào?**

<details><summary>Đáp án</summary>

Vì đổi sang model to hơn cho lợi nhuận biên nhỏ (ai cũng có CLIP mạnh), biên độ cải thiện lớn nằm ở: temporal query (A), nhiều cửa truy vấn + query understanding (B), tốc độ/UI thi đấu (E), và eval kỷ luật (F). Nên ưu tiên các nhóm này thay vì dành thời gian thay encoder — đúng combo "F1 + A1 + B1" mà RESEARCH-PLAN đề xuất.
</details>

---

## 13. Đọc thêm

File gốc đã tải trong [docs/nguon-tham-khao/](../nguon-tham-khao/README.md):

- **lifeXplore at LSC 2024** (Rader & Schoeffmann) — `lifeXplore-LSC2024.pdf`: temporal `<`, FAISS+MongoDB, position sub-filter, query-building UI, eval OpenCLIP. Nguồn chính của §6-7.
- **MEMORIA at LSC 2023** (Ribeiro et al.) — `MEMORIA-LSC2023.html`: graph DB Neo4j, event segmentation, CLIP+ClipCap. Nguồn của §8.

Đọc online:

- [vitrivr: A Flexible Retrieval Stack (ACM MM 2016)](https://doras.dcu.ie/32428/1/ACMMM16_vitrivr.pdf) — kiến trúc 3 tầng (§4) và các query mode (§5).
- [lifeXplore at LSC 2020 (arXiv 2508.21397)](https://arxiv.org/abs/2508.21397) — feature-map/SOM browsing, sketch, concept search (§5).
- [Results of the 2025 Video Browser Showdown (arXiv 2509.12000)](https://arxiv.org/abs/2509.12000) — bức tranh đội/kỹ thuật + cách chấm (§3, §9).
- [VBS Teams & Papers](https://videobrowsershowdown.org/teams/) — roster đầy đủ các hệ và kỹ thuật mới.

Trong dự án:

- [RESEARCH-PLAN.md](../du-an/RESEARCH-PLAN.md) — đặc biệt §1.4 (catalog modality) và §2-3 (đối chiếu + menu ý tưởng A-F).
- [KIEN-TRUC-VA-NGUYEN-TAC.md](../du-an/KIEN-TRUC-VA-NGUYEN-TAC.md) — contracts & nguyên tắc; chương này là phần "vì sao" của các contract đó.
- **Chương 19** (eval) — cách đo các kỹ thuật trên theo đúng format thi trước khi tin.
