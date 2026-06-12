# Chương 11 — LLM sinh văn bản: dịch máy & paraphrase cho query expansion

> **Vị trí trong lộ trình:** Phần II — Các model thành phần. Đứng sau chương 10
> (OCR + detection), trước chương 12 (bi-encoder vs cross-encoder). Đây là chương
> đầu tiên model trong FUFU **sinh ra văn bản mới** thay vì chỉ "nhìn" (ch06-08),
> "nghe" (ch09) hay "đọc" (ch10). Hai nhân vật chính: **NLLB-200** (dịch VI→EN)
> và **Qwen2.5-3B-Instruct** (paraphrase tiếng Việt) — cặp đôi đứng ngay cổng vào
> của mọi truy vấn, trước cả khi FAISS hay BM25 kịp chạy.

---

## 1. Vì sao chương này tồn tại trong FUFU

Hãy bắt đầu từ hai sự thật khó chịu mà team nào thi HCM AI Challenge cũng vấp phải.

**Sự thật 1 — query tiếng Việt, nhưng model visual "mơ bằng tiếng Anh".**
SigLIP (chương 07) được train trên hàng tỷ cặp (ảnh, caption) lấy từ web — mà web
caption thì áp đảo là tiếng Anh. Nó *có* hiểu tiếng Việt (multilingual), nhưng
vùng "tiếng Anh" trong không gian embedding của nó được mài giũa kỹ hơn nhiều.
Kết quả thực nghiệm: cùng một cảnh, query `"a man playing chess"` thường cho
cosine với frame đúng **cao hơn** query `"người đàn ông chơi cờ vua"`. Vứt đi
phần lợi thế đó là tự bịt mắt mình.

**Sự thật 2 — một cách diễn đạt là một lần gieo xúc xắc.**
Người ra đề mô tả cảnh theo cách của *họ*; caption/OCR/ASR trong index mô tả theo
cách của *dữ liệu*. Query `"em bé khóc trong siêu thị"` có thể trượt, trong khi
`"đứa trẻ đang quấy khóc ở cửa hàng"` lại trúng — vì caption do Qwen-VL sinh ra
(chương 08) tình cờ dùng từ "đứa trẻ". Một query duy nhất = một điểm duy nhất
trong không gian embedding; **nhiều biến thể = một đám mây điểm**, xác suất có
điểm nào đó rơi gần frame đúng tăng lên rõ rệt.

FUFU giải cả hai bằng **query expansion**: mỗi query tiếng Việt được (a) dịch
sang tiếng Anh bằng NLLB, (b) paraphrase thành 3 cách nói khác bằng Qwen2.5-3B,
rồi cả chùm biến thể cùng đi tìm kiếm. Chương này dạy hai kỹ thuật sinh văn bản
đứng sau đó — và quan trọng hơn, dạy **các nút vặn** (beam, temperature, top_p,
prompt) mà team sẽ phải tự tay chỉnh khi tune hệ thống.

> 🔗 **Trong FUFU:** toàn bộ chương này gói trong 3 file —
> `app/backend/services/translator.py` (NLLB, 44 dòng),
> `app/backend/services/paraphraser.py` (Qwen 3B, 85 dòng), và hàm
> `expand_query()` trong `app/backend/services/search_engine.py` (dòng 55-105).
> Bật/tắt qua `query_expansion.*` trong `config/settings.yaml`.

---

## 2. Cần biết trước

- **Chương 04-05:** transformer; 3 họ kiến trúc **encoder-only / decoder-only /
  encoder-decoder**. Chương này dùng đúng bản đồ đó: NLLB là encoder-decoder,
  Qwen là decoder-only. Tokenization (subword) cũng từ chương 05.
- **Chương 07:** SigLIP embed text và ảnh vào cùng không gian — để hiểu vì sao
  "dịch sang EN" lại giúp kênh dense.
- **Chương 08:** Qwen-VL sinh caption bằng **greedy decoding** — chương này sẽ
  đối chiếu trực tiếp với cách FUFU sinh paraphrase (sampling).
- Không cần đọc code; mọi tham số quan trọng đều được trích ra thành bảng.

---

## 3. Dịch máy neural (NMT): encoder-decoder vào việc

### 3.1 Trực giác: "nén rồi viết lại"

Dịch máy là bài toán **sequence-to-sequence** kinh điển: vào một câu, ra một câu
khác, độ dài khác nhau, thứ tự từ khác nhau. Kiến trúc encoder-decoder (chương 05)
sinh ra cho đúng việc này:

```
"người chơi cờ vua"
      │ ENCODER: đọc TOÀN BỘ câu nguồn, hai chiều,
      ▼          nén thành dãy vector ngữ cảnh
[vector ngữ cảnh]
      │ DECODER: viết câu đích TỪNG TOKEN một, mỗi bước
      ▼          vừa nhìn vector nguồn (cross-attention) vừa nhìn phần đã viết
"a person playing chess"
```

Encoder được phép nhìn hai chiều vì câu nguồn đã có đủ — giống BERT. Decoder phải
viết tuần tự trái-sang-phải — giống GPT. NMT là nơi hai nửa bắt tay.

### 3.2 NLLB-200: một model, 200 ngôn ngữ

**NLLB** (*No Language Left Behind*, Meta 2022) là một encoder-decoder duy nhất
dịch qua lại giữa **200 ngôn ngữ** — trong đó tiếng Việt là công dân hạng nhất
(dữ liệu train VI-EN thuộc nhóm dồi dào). FUFU dùng bản
`facebook/nllb-200-distilled-600M`: bản "chưng cất" (distill) từ model 54B xuống
600M tham số, đủ nhỏ để thường trú trên GPU (~1.3GB fp16) mà chất lượng dịch câu
ngắn vẫn tốt.

Vì 200 ngôn ngữ chung một model, phải **nói cho nó biết dịch từ đâu sang đâu**
bằng *language code*:

- `vie_Latn` = tiếng Việt, chữ Latinh (nguồn)
- `eng_Latn` = tiếng Anh, chữ Latinh (đích)

Cơ chế: code nguồn gắn vào input cho encoder; code đích được **ép làm token đầu
tiên** mà decoder phải sinh (`forced_bos_token_id` trong code) — như viết sẵn chữ
cái đầu vào trang giấy, ép decoder "vào vai" tiếng Anh ngay từ token thứ nhất.

### 3.3 Greedy vs beam search: viết câu thế nào cho khỏi hối hận

Decoder mỗi bước cho ra một **phân phối xác suất** trên toàn từ điển. Chọn token
nào?

**Greedy:** mỗi bước lấy token xác suất cao nhất. Nhanh, nhưng *tham lam cục bộ*:
chọn tốt nhất ở bước này có thể dồn mình vào ngõ cụt ở bước sau. Ví dụ dịch
`"anh ấy đá bóng rất hay"` — nếu bước đầu greedy chốt `"He kicks..."` (vì "kicks"
nhỉnh hơn "plays" một chút tại thời điểm đó), các bước sau buộc phải chữa cháy
quanh "kicks the ball very well", trong khi nhánh `"He plays football very well"`
— tổng xác suất cả câu cao hơn — đã bị vứt từ bước 1, không bao giờ quay lại được.

**Beam search:** thay vì giữ 1 ứng viên, giữ **N ứng viên tốt nhất** (N = beam
width) ở mỗi bước; bước sau mở rộng cả N nhánh rồi lại cắt về N nhánh có tổng
xác suất cao nhất; kết thúc chọn câu trọn vẹn điểm cao nhất. Như chơi cờ có nghĩ
trước vài nước thay vì ăn quân ngay khi thấy.

FUFU đặt `num_beams=2` (trong `translator.py`) — beam nhỏ nhất có ý nghĩa:
đủ để thoát các bẫy greedy phổ biến, mà chi phí chỉ ~gấp đôi greedy. Câu query
ngắn (≤128 token) nên không cần beam 5-10 như dịch văn bản dài.

Chú ý điểm tương phản sẽ gặp lại ở mục 4: **dịch máy muốn MỘT câu đúng nhất**
→ tìm kiếm tất định (beam). **Paraphrase muốn NHIỀU câu khác nhau** → ngẫu nhiên
(sampling). Cùng là "decoder sinh text" nhưng hai mục tiêu ngược nhau.

### 3.4 Translation bridging — pattern chung của các đội thi VN

Chiêu "dịch query VI→EN rồi tìm" có tên gọi: **translation bridging** — bắc cầu
qua ngôn ngữ mà model visual mạnh nhất. Đây không phải sáng kiến riêng của FUFU:
gần như **mọi đội Việt Nam** ở các kỳ AI Challenge / Video Browser Showdown đều
làm, ví dụ đội RAPID dùng **EnViT5** (model dịch VI-EN chuyên biệt do VietAI
train). FUFU chọn NLLB vì: đa ngữ (sau này cần thêm ngôn ngữ khác thì khỏi đổi
model), bản distilled 600M nhẹ, và chất lượng VI→EN cho *câu mô tả cảnh ngắn*
là quá đủ.

Lưu ý: FUFU **không vứt câu gốc tiếng Việt** — cả VI lẫn EN cùng được encode
(mục 6). Bản dịch là *thêm cầu*, không phải *đốt thuyền*: nếu NLLB dịch trật,
câu gốc vẫn còn đó kéo lại.

---

## 4. LLM sinh văn bản: autoregressive và các nút vặn

### 4.1 Sinh từng token một

Qwen2.5-3B-Instruct là **decoder-only** (chương 05) — họ nhà GPT. Sinh văn bản
kiểu **autoregressive**: đưa prompt vào, model dự đoán phân phối xác suất cho
token *kế tiếp*; chọn một token; nối vào cuối; lặp lại đến khi gặp token kết thúc.
Mỗi token sinh ra là một lần chạy cả model — vì vậy câu trả lời càng dài càng tốn
thời gian *tuyến tính theo số token* (lý do FUFU giới hạn
`paraphrase_max_tokens=120`).

Câu hỏi then chốt giống mục 3.3: có phân phối rồi, **chọn token thế nào?** Với
paraphrase, greedy là tự sát: chạy 10 lần ra đúng 1 kết quả, mà ta cần 3 cách nói
*khác nhau*. Phải **lấy mẫu ngẫu nhiên** (sampling) — và điều khiển độ ngẫu nhiên
bằng hai nút: `temperature` và `top_p`.

### 4.2 Temperature: nút "liều"

Temperature co giãn phân phối trước khi lấy mẫu. Ví dụ cụ thể: model đang viết dở
`"người đàn ông đang ___"` và phân phối gốc cho 5 ứng viên đầu là:

| token | gốc (T=1.0) | **T=0.1** | **T=0.7** | **T=1.5** |
|---|---|---|---|---|
| đi    | 0.50 | **≈1.00** | 0.61 | 0.40 |
| chạy  | 0.25 | ≈0.00 | 0.23 | 0.25 |
| nấu   | 0.15 | ≈0.00 | 0.11 | 0.18 |
| bay   | 0.07 | ≈0.00 | 0.04 | 0.11 |
| hát   | 0.03 | ≈0.00 | 0.01 | 0.06 |

- **T=0.1** (lạnh): phân phối bị bóp nhọn — token dẫn đầu nuốt gần hết xác suất.
  Sinh 10 lần ra 10 câu gần giống nhau. An toàn, nhàm chán.
- **T=0.7** (ấm): vẫn ưu tiên ứng viên tốt nhưng chừa cửa cho hạng 2, hạng 3.
  Đa dạng *có kiểm soát*.
- **T=1.5** (nóng): phân phối bị là phẳng — "bay" từ 7% lên 11%, "hát" từ 3% lên
  6%. Sáng tạo đấy, nhưng xác suất sinh ra `"người đàn ông đang bay"` cho một
  query về cảnh nấu ăn cũng tăng theo. Với query expansion, đó là **lệch nghĩa**
  — tội nặng nhất (mục 7).

### 4.3 Top-p: lưới an toàn chặn đuôi rác

Temperature chỉnh *hình dáng* phân phối nhưng không chặn được **cái đuôi dài**:
hàng chục nghìn token mỗi cái 0.001% xác suất — cộng lại vẫn đáng kể, và thỉnh
thoảng xúc xắc sẽ rơi trúng một token vô nghĩa. **Top-p (nucleus sampling)** xử
lý: sắp xếp token giảm dần, **chỉ giữ nhóm nhỏ nhất có tổng xác suất ≥ p**, vứt
sạch phần còn lại, rồi mới lấy mẫu.

Với phân phối cột T=0.7 ở trên và `top_p=0.9`: đi (0.61) + chạy (0.23) = 0.84
chưa đủ; cộng nấu (0.11) = 0.95 ≥ 0.9 → giữ {đi, chạy, nấu}, **"bay" và "hát" bị
loại hẳn** dù temperature có cho chúng cơ hội. Hay ở chỗ ngưỡng *tự thích nghi*:
chỗ model chắc chắn thì nucleus chỉ còn 1-2 token (gần như greedy), chỗ model
phân vân thì nucleus rộng ra cho thoải mái lựa.

### 4.4 Hai chế độ sinh trong FUFU: ổn định vs đa dạng

Cùng một cơ chế autoregressive, FUFU dùng hai cấu hình **đối lập** tùy mục tiêu:

| | Caption (ch08, `extractors/caption.py`) | Paraphrase (`paraphraser.py`) |
|---|---|---|
| Model | Qwen2.5-**VL-7B** (nhìn ảnh) | Qwen2.5-**3B** (chỉ text) |
| Decoding | **greedy** | **sampling, T=0.7, top_p=0.9** |
| Vì sao | Index phải **ổn định** — cùng frame ingest 2 lần phải ra cùng caption, BM25 mới nhất quán | Cần **3 câu khác nhau** — greedy chạy mấy lần cũng chỉ ra 1 câu |
| Chạy lúc nào | Ingest (offline, chậm được) | Mỗi query (online, phải nhanh) |

Bài học khái quát: **temperature không có giá trị "đúng" tuyệt đối** — nó là
tuyên bố về việc bạn cần *tái lập* hay cần *biến thể*.

---

## 5. Prompt engineering: học từ prompt thật của FUFU

Qwen2.5-3B-**Instruct** là model đã được instruction-tune — nó làm theo lời dặn.
Vấn đề: lời dặn lỏng thì kết quả lỏng. Đây là prompt thật trong `paraphraser.py`:

```python
SYSTEM = "Bạn là trợ lý tạo các cách diễn đạt khác nhau cho truy vấn tìm kiếm video tiếng Việt."

USER_TEMPLATE = """Cho truy vấn sau, sinh {n} cách diễn đạt khác mà người Việt
thường dùng để mô tả CÙNG cảnh đó. Mỗi cách trên 1 dòng, KHÔNG đánh số,
KHÔNG giải thích, KHÔNG thêm dấu gạch đầu dòng. Mỗi diễn đạt ngắn gọn,
tự nhiên, sát nghĩa gốc.

Truy vấn gốc: {q}

{n} cách diễn đạt khác:"""
```

Mỗi cụm từ trong đó là một nguyên tắc prompt engineering — không phải văn trang trí:

1. **System prompt đặt vai** ("trợ lý tạo cách diễn đạt... tìm kiếm video tiếng
   Việt") — neo ngữ cảnh để model không lạc sang chế độ chatbot tám chuyện.
2. **Chỉ định số lượng tường minh** (`{n}` xuất hiện 2 lần) — không nói rõ thì
   model tự quyết, lúc 2 lúc 7.
3. **Chỉ định format máy-đọc-được** ("mỗi cách trên 1 dòng") — vì code phía sau
   sẽ `split("\n")`. Format là *hợp đồng* giữa prompt và parser.
4. **Cấm tường minh các thói quen xấu** ("KHÔNG đánh số, KHÔNG giải thích,
   KHÔNG thêm dấu gạch đầu dòng") — viết hoa để nhấn. Model instruct *rất* thích
   trả lời kiểu `"Dưới đây là 3 cách: 1. ..."`; mỗi chữ KHÔNG ở đây là một vết
   sẹo từ một lần output hỏng.
5. **Ràng buộc ngữ nghĩa** ("CÙNG cảnh đó", "sát nghĩa gốc", "ngắn gọn") — chống
   lệch nghĩa và chống câu dài lê thê làm loãng embedding.
6. **Mồi câu trả lời** (kết thúc bằng `"{n} cách diễn đạt khác:"`) — token đầu
   tiên model sinh ra nhiều khả năng là nội dung luôn, không phải lời dạo đầu.

### Model không luôn nghe lời → hậu xử lý là bắt buộc

Dù prompt kỹ đến đâu, với sampling T=0.7 thỉnh thoảng model vẫn đánh số, vẫn thêm
`- ` đầu dòng, vẫn bọc ngoặc kép. **Quy tắc vàng: đừng tin output thô của LLM.**
`paraphraser.py` có hàm `_clean_line()` lột sạch số thứ tự, gạch đầu dòng, dấu
câu thừa ở mỗi dòng; sau đó còn một vòng **dedup** (lowercase rồi so) loại dòng
trùng nhau *và loại cả dòng trùng query gốc* — model hay "paraphrase" bằng cách...
chép lại nguyên văn. Prompt là tuyến phòng thủ thứ nhất, code là tuyến thứ hai;
hệ thống production cần cả hai.

---

## 6. Ghép lại: chiến lược `expand_query()` của FUFU

Giờ xem hai model trên được phối thế nào (hàm `expand_query`,
`search_engine.py:55`). Với query `"người chơi cờ vua"`:

```
original    = "người chơi cờ vua"
translated  = "a person playing chess"            ← NLLB, num_beams=2
paraphrases = ["hai người đánh cờ",               ← Qwen-3B, T=0.7/top_p=0.9
               "ván cờ vua đang diễn ra",
               "người ngồi chơi cờ"]

"all"  = [original, translated, *paraphrases]   → kênh DENSE (5 biến thể)
"bm25" = [original, translated]                 → 2 kênh BM25 (chỉ 2)
```

**Kênh dense ăn cả 5 biến thể:** SigLIP encode từng câu, **lấy trung bình các
vector rồi L2-normalize** thành một `q_vec` duy nhất đem đi tìm FAISS. Trực giác:
mỗi biến thể là một "góc nhìn" về cùng một cảnh; tâm của đám mây góc nhìn ổn định
hơn bất kỳ điểm đơn lẻ nào — nhiễu diễn đạt của từng câu triệt tiêu lẫn nhau,
phần "nghĩa chung" được giữ lại.

**Kênh BM25 chỉ ăn original + translated — cố tình bỏ paraphrase.** Lý do nằm
ngay trong comment của code: *"phrase match với paraphrase dài thường không khớp
OCR/ASR ngắn, gây nhiễu"*. BM25 (chương 14) khớp **đúng chữ**: OCR trên màn hình
là vài chữ cộc lốc ("KHUYẾN MÃI 50%"), ASR là lời nói tự nhiên. Paraphrase bơm
thêm từ đồng nghĩa mà *không có trong tài liệu* → thêm token rác vào câu OR-query,
kéo theo match 1-token vớ vẩn. Embedding tha thứ cho khác chữ-cùng-nghĩa; BM25
thì không — nên mỗi kênh được cho ăn đúng khẩu phần của nó.

**Dedup ở mọi cửa:** translated trùng original (query vốn là tiếng Anh?) → bỏ;
paraphrase trùng nhau hoặc trùng gốc → bỏ. So sánh sau khi lowercase. Biến thể
trùng lặp không thêm thông tin, chỉ kéo mean vector **lệch về phía bản bị lặp**.

Mọi lỗi ở tầng này đều **fail-soft**: NLLB lỗi → bỏ qua translation; Qwen lỗi
(hoặc máy không có CUDA — paraphraser từ chối chạy CPU) → bỏ qua paraphrase;
tệ nhất hệ thống vẫn tìm được bằng query gốc.

> 🔗 **Trong FUFU:** muốn xem expansion bằng mắt, gọi `POST /api/search` rồi nhìn
> các trường `expanded_queries`, `bm25_queries`, `translated` trong response —
> frontend có sẵn panel debug `<details>` hiển thị chúng (xem `frontend/src/App.jsx`).
> Test nhanh không cần UI: `python scripts/search_demo.py "người chơi cờ vua"`.

---

## 7. Rủi ro & trade-off — thứ phải nhớ khi tune

**1. Dịch sai / paraphrase lệch nghĩa → mean vector bị kéo lệch.** Vì dense dùng
*trung bình* các vector, một biến thể hỏng không bị bỏ phiếu loại — nó **kéo cả
đám mây** về phía sai. Query `"cầu thủ đá phạt"` mà NLLB dịch nhầm "đá phạt" theo
hướng "punish" thay vì "free kick", hoặc Qwen paraphrase thành "cầu thủ bị phạt
thẻ" (cảnh khác hẳn!), thì q_vec đã nhiễm độc trước khi FAISS chạy. Càng nhiều
biến thể tốt thì một biến thể hỏng càng bị pha loãng (1/5 trọng số) — nhưng không
bao giờ bằng 0.

**2. Hallucination.** Qwen-3B là model nhỏ chạy INT4; với query chứa tên riêng
hoặc khái niệm hiếm, nó có thể "sáng tác" chi tiết không có trong gốc (thêm địa
điểm, đổi số lượng người). Prompt "sát nghĩa gốc" giảm chứ không diệt được.
Đây là lý do paraphrase chỉ chiếm 3/5 phiếu trong mean, và bị cấm cửa hoàn toàn
ở BM25.

**3. Latency.** Mỗi query trả phí **1 lần NLLB generate + 1 lần Qwen-3B generate**
(tuần tự, autoregressive từng token) — thường là khoản lớn nhất trong `timing_ms.expand_ms`,
có thể vài trăm ms tới hơn 1 giây, trong khi FAISS chỉ mất vài ms. Với vòng thi
tính giờ, đây là tiền tươi.

**4. Trade-off bật/tắt.** Tất cả là công tắc trong `config/settings.yaml`:

```yaml
query_expansion:
  enable_translation: true    # tắt → nhanh hơn, mất cầu EN (đau nhất với dense)
  enable_paraphrase: true     # tắt → nhanh hơn nhiều, bắt buộc tắt nếu chạy CPU
  num_paraphrases: 3          # tăng = đa dạng hơn + chậm hơn + rủi ro lệch nghĩa hơn
```

Kinh nghiệm tune: nếu phải hy sinh một thứ cho tốc độ, **giữ translation, bỏ
paraphrase** — bản dịch EN là biến thể giá trị nhất trên mỗi ms bỏ ra (đánh thẳng
vào vùng mạnh của SigLIP), còn paraphrase là gia vị.

### Hướng nâng cấp: structured rewriting

Paraphrase hiện tại là "mù" — sinh biến thể *bề mặt chữ* mà không biết hệ thống
có 3 kênh khác nhau. Hướng nâng cấp đã được vạch trong `RESEARCH-PLAN.md` (mục
**B1**): một lần gọi LLM **tách query thành cấu trúc** — câu mô tả cảnh tiếng Anh
cho SigLIP, keywords cho kênh OCR, keywords cho kênh ASR, danh sách object, tên
riêng — rồi đánh *đúng kênh* thay vì rải đều. Các đội mạnh (QUEST, NII-UIT) đã
chứng minh hiệu quả. Chi tiết để dành khi bắt tay làm B1; ở đây chỉ cần nhớ:
mọi kỹ thuật prompt + decoding của chương này chính là nền để viết phiên bản đó.

---

## 8. Tóm tắt 10 giây

- **NLLB-200** (encoder-decoder, 200 ngôn ngữ, `vie_Latn→eng_Latn`) dịch query
  VI→EN để đánh vào vùng mạnh tiếng Anh của SigLIP — *translation bridging*,
  pattern mọi đội thi VN đều dùng. Beam search `num_beams=2`: giữ 2 ứng viên mỗi
  bước, thoát bẫy tham lam của greedy.
- **Qwen2.5-3B** sinh 3 paraphrase bằng **sampling T=0.7 + top_p=0.9** — cần ĐA
  DẠNG, đối lập caption ch08 dùng greedy — cần ỔN ĐỊNH. Temperature chỉnh độ
  liều; top_p chặt đuôi token rác.
- Prompt tốt = đặt vai + chỉ định số lượng/format + cấm tường minh + ràng nghĩa;
  nhưng model không luôn nghe lời → `_clean_line` + dedup là tuyến phòng thủ hai.
- `expand_query`: **dense ăn cả 5 biến thể** (mean vector), **BM25 chỉ ăn
  original + translated** (paraphrase phá phrase match).
- Giá phải trả: biến thể hỏng kéo lệch mean vector, hallucination, ~nửa giây
  latency mỗi query. Tất cả tắt được trong `settings.yaml`.

---

## 9. Câu hỏi tự kiểm tra

**1. Vì sao dịch query VI→EN lại tăng recall của kênh dense, trong khi SigLIP vốn đã multilingual?**

<details><summary>Đáp án</summary>

Multilingual không có nghĩa là đều tay. Dữ liệu pretrain của họ CLIP/SigLIP là
cặp (ảnh, caption) cào từ web, áp đảo tiếng Anh — nên vùng tiếng Anh trong không
gian embedding được align với ảnh tốt hơn hẳn. Cùng một nghĩa, câu EN thường cho
cosine với frame đúng cao hơn câu VI. Dịch sang EN là "bắc cầu" vào vùng mạnh đó
(translation bridging), và FUFU giữ cả câu VI gốc nên không mất gì nếu bản dịch tệ.
</details>

**2. Beam search với num_beams=2 khác greedy thế nào? Vì sao FUFU không để beam 10 cho "chắc"?**

<details><summary>Đáp án</summary>

Greedy mỗi bước chốt 1 token tốt nhất — quyết định sớm sai thì không quay lại
được. Beam giữ 2 chuỗi ứng viên tốt nhất song song mỗi bước, cuối cùng chọn chuỗi
có tổng xác suất cao nhất → tránh được các bẫy "tốt cục bộ, dở toàn cục". Beam 10
tốn ~5× compute so với beam 2 mà với câu query ngắn (≤128 token) gần như không
cải thiện — beam lớn chỉ đáng tiền với văn bản dài, cấu trúc phức tạp. Latency
là tài nguyên thi đấu.
</details>

**3. Caption (ch08) dùng greedy, paraphrase dùng T=0.7 + top_p=0.9. Nếu hoán đổi hai cấu hình thì hỏng kiểu gì?**

<details><summary>Đáp án</summary>

Caption mà sampling: cùng một frame, mỗi lần ingest ra một caption khác nhau →
index không tái lập được, BM25 visual thành trò may rủi, debug "sao hôm qua tìm
ra hôm nay không" bất khả thi. Paraphrase mà greedy: chạy bao nhiêu lần cũng ra
đúng 1 câu (thường na ná query gốc) → mất hoàn toàn mục đích đa dạng hóa, dedup
lọc xong còn 0-1 biến thể. Cấu hình decoding phải đi theo mục tiêu: index cần
ổn định, expansion cần biến thể.
</details>

**4. top_p=0.9 làm gì mà temperature một mình không làm được?**

<details><summary>Đáp án</summary>

Temperature đổi *hình dáng* phân phối nhưng mọi token vẫn giữ xác suất dương —
cái đuôi hàng chục nghìn token rác cộng dồn vẫn có cửa được chọn. Top_p cắt cứng:
chỉ giữ nhóm token nhỏ nhất có tổng xác suất ≥ 0.9, phần đuôi bị loại tuyệt đối.
Ngưỡng còn tự thích nghi: model chắc chắn → nucleus hẹp (gần greedy); model phân
vân → nucleus rộng (đa dạng). Cặp T + top_p cho "sáng tạo có rào chắn".
</details>

**5. Vì sao paraphrase bị loại khỏi kênh BM25 nhưng vẫn được dùng cho dense?**

<details><summary>Đáp án</summary>

BM25 khớp đúng mặt chữ với tài liệu ngắn (OCR vài chữ, ASR một câu nói). Paraphrase
đưa vào từ đồng nghĩa *không tồn tại trong tài liệu* → chỉ thêm token nhiễu vào
OR-query, sinh match 1-token rác. Dense thì ngược lại: embedding đo nghĩa chứ
không đo chữ, nên biến thể đồng nghĩa là tài sản — chúng được mean lại thành q_vec
ổn định hơn. Cùng một dữ liệu expansion, hai kênh có "khẩu vị" ngược nhau.
</details>

**6. Trong prompt của paraphraser có dòng "KHÔNG đánh số, KHÔNG giải thích". Vì sao vẫn cần hàm `_clean_line`?**

<details><summary>Đáp án</summary>

Vì sinh bằng sampling (T=0.7) nên việc tuân thủ prompt cũng... có xác suất: thỉnh
thoảng model vẫn đánh số, thêm gạch đầu dòng, bọc ngoặc kép, hoặc chép lại nguyên
query gốc. Prompt giảm tần suất lỗi, không đưa nó về 0. `_clean_line` (regex lột
prefix số/gạch/dấu) + vòng dedup là tuyến phòng thủ thứ hai để parser không vỡ.
Nguyên tắc production: không bao giờ tin output thô của LLM.
</details>

**7. Một paraphrase lệch nghĩa lọt vào danh sách 5 biến thể. Nó phá kết quả qua cơ chế nào?**

<details><summary>Đáp án</summary>

Kênh dense lấy trung bình 5 vector rồi L2-normalize thành 1 q_vec. Trung bình
không có cơ chế bỏ phiếu loại outlier — vector lệch nghĩa kéo q_vec dịch về phía
nó với trọng số 1/5, làm mọi kết quả FAISS lệch theo một chút. May là nó bị pha
loãng (4 biến thể đúng kéo lại) và không lọt được vào BM25 (kênh này chỉ dùng
original + translated). Đây là lý do prompt nhấn "sát nghĩa gốc" và num_paraphrases
chỉ để 3.
</details>

**8. Backend triển khai trên máy không có GPU. Query expansion sẽ ra sao?**

<details><summary>Đáp án</summary>

Translator tự fallback CPU (NLLB 600M chạy CPU được, chậm hơn). Paraphraser thì
**từ chối khởi tạo** — nó cần bitsandbytes INT4 trên CUDA, constructor raise
RuntimeError với gợi ý tắt `query_expansion.enable_paraphrase`. SearchEngine bắt
lỗi fail-soft: hệ thống vẫn chạy với expansion = [original, translated]. Đúng
khuyến nghị tune: nếu phải bỏ một thứ, bỏ paraphrase, giữ translation.
</details>

---

## 10. Đọc thêm

- **NLLB Team (Meta), 2022** — *No Language Left Behind: Scaling Human-Centered
  Machine Translation* — paper gốc NLLB-200; đọc phần distillation để hiểu bản 600M.
- **Holtzman et al., 2020** — *The Curious Case of Neural Text Degeneration* —
  paper đề xuất nucleus (top-p) sampling; hình minh họa greedy bị lặp vô hạn rất đáng xem.
- **Qwen Team, 2024** — *Qwen2.5 Technical Report* — họ model dùng cho cả
  paraphrase (3B) lẫn caption (VL-7B, chương 08).
- **VietAI EnViT5** — model dịch VI-EN chuyên biệt mà đội RAPID dùng; ứng viên
  thay NLLB nếu muốn benchmark chất lượng dịch query.
- **`RESEARCH-PLAN.md` mục B1** trong repo — kế hoạch structured query rewriting,
  bước tiến hóa tiếp theo của toàn bộ chương này.
- Chương kế tiếp trong giáo trình: **Chương 12 — bi-encoder vs cross-encoder** —
  vì sao chùm query mở rộng ở đây mới chỉ là "lưới quét thô", còn cần một model
  chấm điểm tinh ở cuối pipeline.
