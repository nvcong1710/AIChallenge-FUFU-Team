# Chương 14 — BM25 & hybrid fusion: hợp nhất 3 kênh tìm kiếm

---

## 1. Vì sao chương này tồn tại trong FUFU

Chương 13 cho ta một cỗ máy mạnh: FAISS tìm trong hàng triệu vector SigLIP để
trả về frame "giống về ngữ nghĩa" với câu query. Vậy là xong rồi chứ? Chưa.
Hãy thử query thật trong cuộc thi:

> *"bản tin VTV1 nói về cơn bão số 3"*

Vector SigLIP hiểu rất tốt "bản tin truyền hình", "thời tiết xấu", "phát thanh
viên ngồi trước màn hình xanh" — nhưng nó **mù chuỗi ký tự chính xác**. Với nó,
logo "VTV1" và logo "HTV7" chỉ là hai mảng pixel na ná nhau; "bão số 3" và
"bão số 9" gần như cùng một điểm trong không gian embedding. Trong khi đó,
chính những chuỗi đó — **tên kênh, tên riêng, con số, địa danh** — lại là thứ
phân biệt đúng/sai trong bài Known-Item Search.

May mắn là FUFU không chỉ có vector. Lúc ingest, mỗi frame đã được gắn chữ
(OCR + caption + nhãn detection) và mỗi đoạn audio đã có transcript ASR
(chương 09, 10). Đó là **văn bản thuần** — và tìm kiếm văn bản theo từ khoá
là bài toán đã được giải rất tốt từ thập niên 90 bằng **BM25**: keyword search
giỏi đúng chỗ dense mù, và ngược lại dense cứu keyword khi người dùng diễn đạt
khác từ ngữ trong tài liệu. **Hai kênh bù nhau.**

Thế là pipeline search của FUFU chạy **3 kênh song song**:

```
              query (đã expand — chương 11)
        ┌──────────────┼──────────────────┐
   DENSE FAISS    BM25 visual         BM25 ASR
   (cosine,       (FTS5 trên          (FTS5 trên
   chương 13)     ocr+caption+labels) transcript)
        └──────────────┼──────────────────┘
              fuse_and_aggregate()  ← CHƯƠNG NÀY
                       │
              cross-encoder rerank (chương 12)
```

Chương này trả lời 2 câu hỏi: **BM25 chấm điểm thế nào**, và — khó hơn nhiều —
**làm sao cộng được điểm của 3 kênh có thang đo hoàn toàn khác nhau** mà không
để kênh này "nuốt" kênh kia.

> 🔗 **Trong FUFU:** 2 kênh BM25 nằm ở `app/backend/services/retrieval.py`
> (hàm `bm25_visual`, `bm25_asr`, và `_build_fts_or_query` xây query FTS5);
> phần hợp nhất điểm nằm ở `app/backend/services/rerank.py` (hàm
> `fuse_and_aggregate`). ⚠️ Đừng nhầm `rerank.py` (score fusion — chương này)
> với `reranker.py` (cross-encoder BGE — chương 12). Trọng số 3 kênh ở
> `config/settings.yaml` → `retrieval.weights`.

---

## 2. Cần biết trước

- **TF-IDF** từ ML cổ điển: từ xuất hiện nhiều trong document thì quan trọng
  (TF), từ xuất hiện trong ít document thì đáng giá (IDF). BM25 chính là
  TF-IDF "trưởng thành" — nếu bạn từng vector hoá văn bản bằng
  `TfidfVectorizer` của sklearn thì đã đi được 70% chương này.
- **Chương 07 + 13**: dense retrieval — query và frame cùng không gian vector,
  điểm = cosine ∈ [-1, 1]. Chỉ cần nhớ tính chất *bounded* (bị chặn) của nó.
- **Chương 09, 10**: ASR và OCR sinh ra văn bản mà BM25 sẽ tìm trên đó.
- Khái niệm **chuẩn hoá min-max** (rescale về [0,1]) — quen thuộc từ
  preprocessing ML cổ điển.

Chương này KHÔNG dạy cross-encoder rerank (chương 12), không dạy FAISS
(chương 13), và chỉ *nhắc tên* Reciprocal Rank Fusion — so sánh các chiến
lược ensemble/fusion là việc của chương 18.

---

## 3. Dense giỏi gì, mù gì — vì sao phải có kênh chữ

| Tình huống | Dense (SigLIP) | Keyword (BM25) |
|---|---|---|
| "người đàn ông chơi cờ" nhưng caption ghi "hai kỳ thủ thi đấu" | ✅ hiểu đồng nghĩa | ❌ không token nào trùng |
| "VTV1", "Sơn Tùng M-TP", "bão số 3" | ❌ tên riêng/số bị nhoè | ✅ match chính xác chuỗi |
| Query mô tả cảnh, không từ khoá đặc biệt | ✅ | ⚠️ ra toàn từ phổ biến |
| Cảnh chỉ phân biệt được qua lời thoại | ❌ (vector là của ảnh) | ✅ BM25 trên ASR |
| Audio thuần (podcast, radio) | ❌ không có frame nào | ✅ kênh duy nhất tìm được |

Hàng cuối quan trọng: trong FUFU, **item audio không có vector visual nào cả**
(PROJECT-CONTEXT §6) — nếu không có kênh BM25 ASR thì cả tệp audio đó vô hình
với search. Đây là lý do trọng số kênh ASR được đặt cao (xem §8).

---

## 4. BM25 — TF-IDF trưởng thành: 3 trực giác

BM25 (Best Matching 25, từ họ mô hình Okapi) chấm điểm document *d* cho query
*q* bằng tổng theo từng từ của query. So với TF-IDF thô, nó thêm đúng 3 ý:

### 4.1 TF saturation — lần thứ 10 không đáng giá gấp 10 lần thứ nhất

TF-IDF thô: từ xuất hiện 10 lần → điểm TF gấp 10 lần xuất hiện 1 lần. Trực
giác sai: một đoạn ASR nhắc "bão" 10 lần không *về bão* gấp 10 lần đoạn nhắc
1 lần — nó chỉ *chắc chắn hơn một chút* là về bão. BM25 thay `tf` bằng:

```
tf_component = tf · (k1 + 1) / (tf + k1)        (k1 ≈ 1.2)
```

Tính tay với k1 = 1.2:

| tf | tf_component | nhận xét |
|---|---|---|
| 1 | 1.00 | |
| 2 | 1.38 | lần 2 chỉ thêm +0.38 |
| 3 | 1.57 | |
| 10 | 1.96 | gấp 10 lần tf nhưng điểm chưa gấp đôi |
| 100 | 2.17 | tiệm cận trần k1+1 = 2.2 |

Đường cong **bão hoà** (diminishing returns) — giống sigmoid bị cắt nửa trên.
Hệ quả thực dụng: spam từ khoá không ăn điểm.

### 4.2 IDF — từ hiếm đáng giá hơn từ phổ biến

Giống hệt TF-IDF cổ điển. Giả sử kho FUFU có **N = 10.000** đoạn ASR:

| từ | xuất hiện trong (df) | idf ≈ log₁₀(N/df) |
|---|---|---|
| "VTV1" | 12 đoạn | log₁₀(833) ≈ **2.92** |
| "bão" | 400 đoạn | log₁₀(25) ≈ **1.40** |
| "người" | 6.000 đoạn | log₁₀(1.67) ≈ **0.22** |

Match được "VTV1" đáng giá gấp ~13 lần match "người". (BM25 thật dùng công
thức idf có smoothing, nhưng trực giác y nguyên.) Đây chính là cái phao cứu
chiến lược OR-tokens của FUFU ở §6: match nhầm toàn từ phổ biến → idf thấp
→ điểm thấp → bị ngưỡng lọc cắt.

### 4.3 Length normalization — đoạn ngắn chứa từ khoá thì "đậm đặc" hơn

Một đoạn ASR 8 từ chứa "VTV1" gần như chắc chắn *về* VTV1; một đoạn 60 từ
chứa "VTV1" có thể chỉ nhắc lướt qua. BM25 chia điểm TF cho hệ số độ dài
`1 - b + b·(dl/avgdl)` với b ≈ 0.75: document dài hơn trung bình bị phạt,
ngắn hơn được thưởng.

### 4.4 Ví dụ tính tay trọn vẹn

Query **"VTV1 bão"**, hai đoạn ASR, avgdl = 20 từ, k1 = 1.2, b = 0.75,
idf lấy từ bảng §4.2:

- **A** (8 từ): *"bản tin VTV1 tối nay có bão rất lớn"* — chứa VTV1 ×1, bão ×1
- **B** (40 từ): phóng sự dài về mưa gió, chứa "bão" ×3, **không** có VTV1

**Đoạn A** — hệ số độ dài: 1 − 0.75 + 0.75·(8/20) = **0.55** (ngắn → được thưởng).
Mỗi từ tf = 1: tf_comp = 1·2.2 / (1 + 1.2·0.55) = 2.2/1.66 ≈ **1.33**.

```
score(A) = idf(VTV1)·1.33 + idf(bão)·1.33
         = 2.92·1.33 + 1.40·1.33 ≈ 3.87 + 1.86 = 5.73
```

**Đoạn B** — hệ số độ dài: 0.25 + 0.75·(40/20) = **1.75** (dài → bị phạt).
"bão" tf = 3: tf_comp = 3·2.2 / (3 + 1.2·1.75) = 6.6/5.1 ≈ **1.29**.

```
score(B) = idf(bão)·1.29 = 1.40·1.29 ≈ 1.81
```

**A = 5.73 thắng B = 1.81** dù B nhắc "bão" gấp 3 lần — vì A có từ hiếm
(IDF), ngắn gọn (length norm), còn 3 lần "bão" của B bị saturation đè xuống.
Cả 3 trực giác cùng hiện hình trong một con tính. Để ý thêm: B = 1.81 **dưới
ngưỡng MIN_BM25_RAW = 3.0 của FUFU** → trong hệ thật nó bị lọc luôn (§6).

Hai điểm cuối về thang đo, sẽ thành chuyện lớn ở §7:
- Cosine bị chặn trong [-1, 1]; **BM25 không có trần** — query nhiều từ hiếm
  có thể ra raw 15-20.
- BM25 chỉ có nghĩa **tương đối trong một lần truy vấn** — 5.73 của query này
  không so được với 5.73 của query khác.

---

## 5. SQLite FTS5 — BM25 chạy ở đâu trong FUFU

### 5.1 Inverted index: cuốn mục lục ngược

Tìm "VTV1" bằng cách quét tuần tự 10.000 transcript thì quá chậm. FTS5 (mô-đun
full-text search của SQLite) xây **inverted index** — đúng nghĩa "mục lục
ngược" cuối sách giáo trình: thay vì *document → các từ trong nó*, lưu
*từ → danh sách document chứa nó*:

```
"vtv1" → [đoạn 17, đoạn 502, đoạn 9981, ...]      (12 đoạn)
"bão"  → [đoạn 3, đoạn 17, đoạn 88, ...]           (400 đoạn)
```

Tra "VTV1 bão" = lấy 2 danh sách, gộp lại, chấm BM25 trên vài trăm ứng viên
thay vì 10.000 — cùng triết lý "đánh index để khỏi quét hết" như HNSW của
FAISS, chỉ khác cấu trúc dữ liệu.

FUFU có **2 bảng FTS5** (định nghĩa trong `app/ingest/storage.py`):

| Bảng | Nội dung | Là "mắt" hay "tai" |
|---|---|---|
| `frame_text` | `ocr_text` + `caption` + `labels` của mỗi frame | chữ-trên-hình: biển hiệu, phụ đề, mô tả Qwen-VL, nhãn YOLO |
| `asr_text` | `transcript` của mỗi đoạn ASR | lời thoại PhoWhisper |

### 5.2 Tokenizer: `unicode61 remove_diacritics 0` — GIỮ dấu tiếng Việt

Cả 2 bảng khai báo `tokenize='unicode61 remove_diacritics 0'`. Mặc định FTS5
*xoá dấu* khi index ("bão" → "bao"); FUFU tắt hành vi đó. Vì sao? Tiếng Việt
mất dấu là mất nghĩa: **"bão" / "bao" / "báo" / "bảo"** là 4 từ khác hẳn nhau
— xoá dấu thì query "bão" sẽ match cả bài về "báo chí" và "bao bì".

Trade-off đáng biết: **query gõ không dấu sẽ không match gì cả.** Người dùng
gõ "ban tin vtv1" → token "ban", "tin" không khớp "bản", "tin"... thực ra
"tin" khớp nhưng "bản" thì không. FUFU chọn chính xác > dễ dãi, và dựa vào
việc operator thi đấu gõ có dấu. Nếu sau này muốn cứu query không dấu, phải
xử lý ở tầng query expansion chứ tokenizer hiện tại không tha thứ.

Một chi tiết kỹ thuật dễ vấp khi đọc code: hàm `bm25()` của FTS5 trả về số
**âm** (càng âm càng tốt — để `ORDER BY` tăng dần ra kết quả tốt trước).
`retrieval.py` đảo dấu ngay (`-float(s)`) nên từ đó trở đi trong FUFU,
"BM25 raw" luôn là số dương, càng lớn càng tốt.

---

## 6. Chiến lược query của FUFU: OR các token + ngưỡng lọc rác

Có bảng FTS5 rồi, câu hỏi tiếp: **đưa cái gì vào `MATCH`?** Cách ngây thơ là
phrase match cả câu — `"bản tin VTV1 tối nay"` — yêu cầu 4-5 từ đứng liên
tiếp đúng thứ tự. FUFU đã thử và bỏ. Comment trong chính `_build_fts_or_query`
giải thích: *"Phrase match cũ quá strict — đoạn lời ASR khó match 4-5 từ liên
tiếp với query gốc."* ASR có lỗi nhận dạng, người nói chêm từ đệm, query của
operator không bao giờ trùng nguyên văn lời thoại. Phrase match → recall ≈ 0.

Thay vào đó, `_build_fts_or_query` (trong `retrieval.py`) làm 4 việc:

1. **Lowercase + tách token** theo whitespace, từ *cả* các biến thể query
   (gốc + bản dịch EN — lưu ý kênh BM25 không nhận paraphrase, vì paraphrase
   dài chỉ bơm thêm token nhiễu).
2. **Lọc ký tự đặc biệt** của cú pháp FTS5 (`"`, `(`, `)`, `:`, `-`...) — chỉ
   giữ chữ-số và nguyên bảng chữ cái tiếng Việt có dấu (hard-code trong hàm).
3. **Bỏ token < 2 ký tự** ("ở", "à", mảnh vụn sau khi lọc) và dedupe bằng set.
4. Bọc mỗi token trong nháy kép rồi nối bằng `OR`:

```
"bản tin VTV1 tối nay" + "VTV1 news tonight"
  → "bản" OR "tin" OR "vtv1" OR "tối" OR "nay" OR "news" OR "tonight"
```

OR nghĩa là *match 1 token cũng được tính* — nghe có vẻ lỏng lẻo nguy hiểm,
nhưng nhớ lại §4: BM25 tự xếp hạng — đoạn khớp **nhiều** token và token
**hiếm** sẽ nổi lên đầu, đoạn chỉ khớp mỗi "tin" thì idf thấp, điểm bèo.

Phòng tuyến cuối là ngưỡng cứng trong `bm25_visual` / `bm25_asr`:

```python
MIN_BM25_RAW = 3.0     # raw score < 3.0 → coi là noise, vứt
```

Match 1 token phổ biến cho raw cỡ 0.2-1.5 (như đoạn B = 1.81 ở §4.4) — dưới
3.0 hết. Muốn sống sót qua ngưỡng phải khớp 1 từ hiếm hoặc ≥2-3 từ thường.
OR-rộng-rãi + ngưỡng-chặt = recall cao mà rác vẫn bị chặn ở cửa.

---

## 7. VẤN ĐỀ TRUNG TÂM — trộn điểm khác thang đo

Giờ ta có 3 danh sách kết quả với 3 loại điểm:

| Kênh | Điểm | Thang |
|---|---|---|
| Dense FAISS | cosine | **bounded**, thực tế SigLIP ra ~0.05-0.35 |
| BM25 visual | raw BM25 | **unbounded**, thực tế ~3-15 |
| BM25 ASR | raw BM25 | unbounded, ~3-15 |

Thử cộng thẳng xem chuyện gì xảy ra. Hai ứng viên cho query "bản tin VTV1":

- Segment **X**: đúng cảnh bản tin, dense cosine = **0.31** (rất cao với
  SigLIP), không lọt BM25 (logo bị OCR trượt).
- Segment **Y**: phóng sự bất kỳ mà transcript tình cờ chứa "bản", "tin",
  "tối", "nay" — BM25 ASR raw = **7.2**, dense = 0.04 (chẳng giống gì).

```
cộng thẳng:  score(X) = 0.31 + 0     = 0.31
             score(Y) = 0.04 + 7.2   = 7.24     ← Y "thắng" gấp 23 lần?!
```

Kênh BM25 **nuốt chửng** kênh dense — không phải vì Y liên quan hơn mà thuần
tuý vì đơn vị đo khác nhau, như cộng 38°C với 100°F rồi kết luận 100°F sốt
hơn. Bài học ML cổ điển hiện về: **chưa chuẩn hoá feature thì đừng cộng** —
y như lý do phải scale feature trước khi cho vào kNN hay SVM.

---

## 8. Hai cách chuẩn hoá của FUFU — và một design decision tinh tế

`fuse_and_aggregate` trong `rerank.py` chuẩn hoá **bất đối xứng** — mỗi kênh
một kiểu, có chủ đích:

### 8.1 Dense: min-max trong lần truy vấn

```python
dense_norm = (s - min) / (max - min)      # hàm _minmax
```

Top-500 cosine của FAISS được kéo về [0,1]: hit tốt nhất = 1.0, tệ nhất = 0.0.
Hợp lý vì cosine vốn bị chặn và phân bố trong lần truy vấn khá dày — min-max
chỉ "kéo giãn" cho dùng hết thang, không bóp méo thứ tự.

### 8.2 BM25: chia cho hằng số, KHÔNG min-max — vì sao?

```python
BM25_SCALE = 8.0
bm25_norm = min(raw / BM25_SCALE, 1.0)    # hàm _raw_scaled_bm25
```

Câu hỏi tự nhiên: sao không min-max luôn cho đồng bộ? Comment trong code trả
lời: *min-max sẽ equalize tất cả về 1.0 khi chỉ có 1 hit — mất "độ mạnh tuyệt
đối"*. Cụ thể bằng số:

Query hiếm chỉ có **đúng 1 hit ASR**, raw = 3.2 — tức là *vừa lết qua* ngưỡng
3.0, match yếu xìu. Nếu min-max: 1 hit → max = min → quy ước trả 1.0 → nhân
trọng số ASR 0.5 → đóng góp **0.5**, đè bẹp cả frame dense hoàn hảo
(1.0 × 0.4 = 0.4). Một match rác lên ngôi chỉ vì nó... cô đơn.

Với raw-scale: 3.2/8.0 = **0.4** → đóng góp 0.5 × 0.4 = 0.2 — đúng tầm một
match yếu. Còn match thật sự mạnh (raw ≥ 8, nhiều token + từ hiếm) chạm trần
1.0. Hằng số 8.0 được chọn theo quan sát: raw 4-5 = match cụm từ tốt,
raw 8-12 = match nhiều token rất mạnh.

Bài học khái quát: **min-max chỉ giữ thông tin *thứ hạng tương đối*; phép
chia-hằng-số giữ thông tin *cường độ tuyệt đối*.** BM25 raw vốn mang nghĩa
tuyệt đối trong một query (3 = yếu, 8 = mạnh) — vứt thông tin đó đi là phí.
Dense thì ngược lại: cosine SigLIP 0.31 hay 0.25 "mạnh" đến đâu còn tuỳ query,
nên min-max tương đối lại hợp. Mỗi kênh một phép chuẩn hoá — không phải cẩu
thả mà là thiết kế.

### 8.3 Weighted sum — và quyết định KHÔNG renormalize

Sau chuẩn hoá, cả 3 kênh đều nằm trong [0,1]. Hợp nhất bằng tổng có trọng số:

```
score = 0.40·dense + 0.25·bm25_visual + 0.50·bm25_asr
```

(trọng số từ `config/settings.yaml` → `retrieval.weights` — tham số tune
chính của cả hệ thống). Hai điều đáng đọc ra từ bộ số này:

**Tổng trọng số = 1.15 > 1, và cố ý không renormalize.** Item match cả 3 kênh
có thể đạt 1.15; item một kênh tối đa bằng đúng trọng số kênh đó. Ví dụ:

| Hit | dense | bm25v | bm25a | score |
|---|---|---|---|---|
| X: khớp dense + OCR | 0.9 | 0.8 | 0 | 0.4·0.9 + 0.25·0.8 = **0.56** |
| Y: chỉ khớp ASR (mạnh) | 0 | 0 | 1.0 | 0.5·1.0 = **0.50** |
| Z: chỉ khớp dense (hoàn hảo) | 1.0 | 0 | 0 | 0.4·1.0 = **0.40** |

X thắng dù không kênh nào của nó đạt tuyệt đối — **bằng chứng độc lập từ
nhiều kênh đáng tin hơn một kênh đơn lẻ**, đúng intent (comment trong
`fuse_and_aggregate` nói thẳng điều này).

**w_asr = 0.5 cao nhất** — vì item audio chỉ có đúng 1 kênh ASR để sống
(hàng Y): nếu w_asr nhỏ, mọi audio sẽ chìm dưới đáy ranking trước các video
có kênh dense. Bộ 0.4/0.25/0.5 được tune để Y (audio, ASR khớp mạnh) vẫn
cạnh tranh được với Z (video, dense khớp hoàn hảo). README cũ ghi
0.6/0.25/0.15 — đã lỗi thời, tin `settings.yaml`.

---

## 9. Aggregate frame → segment: max-pool từng kênh

Còn một lệch pha cuối: dense và BM25 visual chấm điểm **frame**, BM25 ASR chấm
**đoạn lời**, nhưng người dùng cần **segment** (đoạn video nhảy-đến được).
`fuse_and_aggregate` gom mọi thứ về khoá `(item_id, segment_id)` và
**max-pool từng kênh riêng rẽ**: điểm kênh của segment = điểm cao nhất mà một
frame/đoạn-lời nào đó của nó đạt được trong kênh ấy.

Ví dụ segment 42 có 3 keyframe, và 1 đoạn ASR được gán vào nó:

| nguồn | dense (norm) | bm25v (norm) | bm25a (norm) |
|---|---|---|---|
| frame f1 | 0.62 | — | |
| frame f2 | **0.91** | **0.75** | |
| frame f3 | 0.45 | — | |
| đoạn ASR a7 | | | **0.60** |
| **segment 42 (max-pool)** | **0.91** | **0.75** | **0.60** |

```
score(seg 42) = 0.4·0.91 + 0.25·0.75 + 0.5·0.60 = 0.364 + 0.188 + 0.300 ≈ 0.85
```

Vì sao max chứ không mean? Một shot 12 giây có thể chỉ có **1 frame** thực sự
chứa thứ ta tìm (logo hiện 1 giây rồi biến mất) — mean sẽ để 11 frame "vô tội"
pha loãng bằng chứng. Triết lý known-item search: *một khoảnh khắc khớp mạnh
đủ kết luận cả segment đáng xem.*

Trong lúc max-pool, mỗi `Hit` giữ lại **`best_frame`** (frame có raw cosine
cao nhất — làm thumbnail đại diện) và **`best_asr`** (snippet lời thoại mạnh
nhất) — để UI hiển thị *vì sao* segment này được trả về, và để cross-encoder
ở chương 12 có văn bản mà đọc. Chi tiết nhỏ trong code: 1 frame có thể thuộc
nhiều segment (bảng `frame_segments` là M-N do overlap) → điểm của frame được
**broadcast** cho mọi segment chứa nó.

---

## 10. Có cách fusion nào khác không?

Có. Weighted sum trên điểm chuẩn hoá chỉ là một trường phái. Trường phái lớn
còn lại là **Reciprocal Rank Fusion (RRF)** — vứt hẳn điểm số, chỉ dùng *thứ
hạng* trong mỗi kênh (hit đứng top-1 của kênh nào cũng đáng giá như nhau),
nhờ đó né luôn toàn bộ vấn đề chuẩn hoá của §7-8 nhưng đổi lại mù "độ mạnh
tuyệt đối". So sánh RRF vs weighted sum, khi nào cái nào thắng, và các chiêu
ensemble nhiều model — hẹn ở **chương 18**.

---

## Tóm tắt 10 giây

- Dense hiểu ngữ nghĩa nhưng mù chuỗi ký tự ("VTV1", số, tên riêng); BM25
  ngược lại → FUFU chạy 3 kênh song song: dense + BM25 visual + BM25 ASR.
- BM25 = TF-IDF + 3 nâng cấp: TF bão hoà (trần k1+1), IDF (từ hiếm ăn điểm),
  phạt document dài.
- FTS5 (SQLite) giữ dấu tiếng Việt (`remove_diacritics 0`) — query không dấu
  không match; query xây kiểu OR-tokens + ngưỡng `MIN_BM25_RAW = 3.0` chặn rác.
- Cấm cộng điểm khác thang: dense → min-max; BM25 → `raw/8.0` cap 1.0 (giữ
  cường độ tuyệt đối, tránh bẫy min-max-1-hit = 1.0).
- `score = 0.4·dense + 0.25·bm25v + 0.5·bm25a`, không renormalize → match
  nhiều kênh thắng; max-pool frame→segment, giữ best_frame/best_asr.

---

## Câu hỏi ôn tập

**1. Query "ca sĩ Sơn Tùng hát trên sân khấu" — kênh nào bắt được "Sơn Tùng",
kênh nào bắt được "hát trên sân khấu"? Vì sao?**

<details><summary>Đáp án</summary>

"Sơn Tùng" là tên riêng — embedding SigLIP gần như không phân biệt được ca sĩ
này với ca sĩ khác, nhưng BM25 (trên caption/OCR nếu có chữ trên màn, hoặc ASR
nếu MC giới thiệu tên) match chính xác chuỗi và "sơn"/"tùng" có IDF khá cao.
"hát trên sân khấu" là khái niệm thị giác tổng quát — sở trường của dense
(ánh đèn, micro, đám đông), trong khi BM25 chỉ ăn nếu transcript/caption tình
cờ chứa đúng các từ đó. Hai kênh bù nhau đúng nghĩa.
</details>

**2. Với k1 = 1.2, từ xuất hiện 100 lần được TF component bao nhiêu? Điều đó
chống lại hành vi gì?**

<details><summary>Đáp án</summary>

100·2.2/(100+1.2) ≈ 2.17, gần trần k1+1 = 2.2 — chỉ hơn gấp đôi so với
xuất hiện 1 lần (1.0) dù tf gấp 100. Saturation chống keyword spamming và
phản ánh trực giác: lần lặp thứ 100 hầu như không thêm thông tin gì về chủ
đề document.
</details>

**3. Vì sao FUFU dùng OR các token thay vì phrase match cả câu trong FTS5?
Cái gì cứu OR khỏi việc trả về toàn rác?**

<details><summary>Đáp án</summary>

Phrase match đòi 4-5 từ liên tiếp đúng thứ tự — transcript ASR có lỗi nhận
dạng, từ đệm, và operator không bao giờ gõ trùng nguyên văn → recall ≈ 0
(comment trong `_build_fts_or_query`). OR cho match lỏng, và 2 thứ chặn rác:
(1) bản thân BM25 xếp hạng — match nhiều token/token hiếm nổi lên, match 1 từ
phổ biến điểm bèo do IDF; (2) ngưỡng cứng `MIN_BM25_RAW = 3.0` cắt mọi match
yếu trước khi vào fusion.
</details>

**4. Query chỉ có 1 hit BM25 ASR với raw = 3.1. So sánh đóng góp vào final
score nếu chuẩn hoá bằng min-max vs raw/8.0 (w_asr = 0.5).**

<details><summary>Đáp án</summary>

Min-max với 1 hit: max = min → quy ước 1.0 → đóng góp 0.5·1.0 = **0.5** —
match suýt-bị-lọc bỗng đè bẹp cả dense hoàn hảo (0.4). Raw-scale:
3.1/8.0 ≈ 0.39 → đóng góp 0.5·0.39 ≈ **0.19** — đúng tầm match yếu. Đây
chính là lý do FUFU không min-max BM25: giữ "độ mạnh tuyệt đối" của raw score.
</details>

**5. Tổng trọng số 0.4 + 0.25 + 0.5 = 1.15 > 1. Đây là bug? Hệ quả của việc
không renormalize là gì?**

<details><summary>Đáp án</summary>

Không phải bug — cố ý. Không renormalize nghĩa là item match nhiều kênh có
trần điểm cao hơn item một kênh: bằng chứng độc lập từ 2-3 kênh đáng tin hơn
một kênh đơn. Ví dụ dense 0.9 + bm25v 0.8 → 0.56, thắng cả dense hoàn hảo
đơn độc (0.40). Đồng thời w_asr = 0.5 cao nhất để item audio (chỉ có duy nhất
kênh ASR) còn cạnh tranh được với video.
</details>

**6. Vì sao aggregate frame → segment dùng max-pool chứ không mean-pool từng
kênh?**

<details><summary>Đáp án</summary>

Trong known-item search, một segment "đúng" có thể chỉ chứa 1 frame thực sự
khớp (logo hiện 1 giây) giữa nhiều frame thường — mean sẽ pha loãng bằng
chứng mạnh đó bằng các frame không liên quan. Max nói: "khoảnh khắc tốt nhất
của segment khớp đến đâu?" — đủ để quyết định segment đáng nhảy đến. Frame
đạt max được giữ làm `best_frame` để hiển thị và đưa vào cross-encoder.
</details>

**7. Người dùng gõ "ban tin vtv1" (không dấu) và kênh BM25 trả về 0 kết quả
dù DB đầy bản tin. Giải thích, và vì sao FUFU vẫn chọn thiết kế này?**

<details><summary>Đáp án</summary>

Cả 2 bảng FTS5 dùng `unicode61 remove_diacritics 0` — index giữ nguyên dấu,
nên token "ban" không khớp "bản" (riêng "vtv1" vẫn khớp vì không có dấu, nhưng
điểm 1-token thường dưới ngưỡng nếu là từ thường — "vtv1" hiếm nên có thể
sống). Thiết kế này đổi lấy độ chính xác: bỏ dấu thì "bão"/"báo"/"bảo"/"bao"
trộn làm một, nhiễu nặng hơn nhiều. Muốn hỗ trợ query không dấu phải xử lý ở
tầng query expansion, không phải ở tokenizer.
</details>

---

## Đọc thêm

- Robertson & Zaragoza (2009), *The Probabilistic Relevance Framework: BM25
  and Beyond* — survey chuẩn về BM25 từ chính tác giả.
- SQLite docs: [FTS5 Extension](https://www.sqlite.org/fts5.html) — mục
  `bm25()` và tokenizer `unicode61` (giải thích `remove_diacritics`).
- Cormack, Clarke & Buettcher (2009), *Reciprocal Rank Fusion outperforms
  Condorcet and individual rank learning methods* — bài gốc RRF (chương 18).
- Blog Pinecone/Weaviate về *hybrid search* — các hệ vector DB thương mại
  cũng phải giải đúng bài chuẩn hoá điểm dense + BM25 như FUFU.
- Trong repo: `PROJECT-CONTEXT.md` §8 (sơ đồ luồng search đầy đủ),
  `scripts/search_demo.py` (chạy thử và soi `score_breakdown` của từng hit).
