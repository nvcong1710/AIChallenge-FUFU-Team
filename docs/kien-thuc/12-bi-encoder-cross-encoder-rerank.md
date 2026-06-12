# Chương 12 — Bi-encoder vs Cross-encoder: rerank để chính xác hơn

---

## 1. Vì sao chương này tồn tại trong FUFU

Nhìn lại pipeline search của FUFU (PROJECT-CONTEXT.md §8): sau khi 3 kênh
(dense SigLIP + BM25 visual + BM25 ASR) được hợp nhất điểm, kết quả **không trả
về ngay**. Còn một bước cuối:

```
fuse_and_aggregate (hợp nhất 3 kênh)
        │
BGE-reranker cross-encoder: rerank top-50   ← chương này nói về bước NÀY
        │
top-K (20) → JSON
```

Câu hỏi tự nhiên: SigLIP đã so query với frame rồi, BM25 cũng đã chấm điểm rồi,
**tại sao còn cần thêm một model nữa chấm lại?** Và nếu model đó chính xác hơn,
**tại sao không dùng nó ngay từ đầu cho toàn bộ kho dữ liệu?**

Trả lời được 2 câu này = hiểu được sự đánh đổi **bi-encoder vs cross-encoder** —
một trong những trade-off quan trọng nhất của ngành retrieval, và là lý do mọi
hệ thống tìm kiếm nghiêm túc (Google, Bing, các đội thi VBS) đều có kiến trúc
2 tầng: **retrieve rẻ → rerank đắt**.

> 🔗 **Trong FUFU:** cross-encoder nằm ở `app/backend/services/reranker.py`
> (class `BGEReranker`), được gọi từ `app/backend/services/search_engine.py`
> (cuối hàm `search()`). Bật/tắt bằng `retrieval.enable_reranker` trong
> `config/settings.yaml`.

---

## 2. Cần biết trước

- **Chương 04-05**: attention, transformer encoder, BERT-style model (đọc cả câu
  một lượt, mỗi token "nhìn" mọi token khác).
- **Chương 07**: SigLIP/CLIP — encode ảnh và text vào cùng không gian vector,
  so bằng cosine. SigLIP chính là nhân vật "bi-encoder" của chương này.
- ML cổ điển: **kNN**, **feature extraction**, khái niệm **classifier chấm điểm
  từng mẫu**, và **cascade classifier** (kiểu Viola-Jones: tầng rẻ lọc thô, tầng
  đắt xét kỹ).

Chương này KHÔNG dạy lại cách hợp nhất điểm 3 kênh (đó là chương 14) hay cách
FAISS tìm kNN nhanh (chương 13). Ta chỉ quan tâm: **hai cách cho model so sánh
query với document, và khi nào dùng cách nào.**

---

## 3. Bi-encoder — kiến trúc "2 tháp": encode riêng, so sau

### 3.1 Trực giác

Tưởng tượng bạn cần ghép đôi 1 câu hỏi với 1 triệu tài liệu. Cách "2 tháp"
(two-tower) làm thế này:

```
   query                    document
     │                          │
 [Encoder Q]               [Encoder D]      ← 2 "tháp", có thể cùng/khác weights
     │                          │
   vector q  ──── dot product ──── vector d   → similarity
```

Điểm mấu chốt: **query và document KHÔNG BAO GIỜ gặp nhau bên trong model.**
Mỗi bên tự nén toàn bộ ý nghĩa của mình vào MỘT vector (vd 1024 chiều), rồi
mới so hai vector bằng dot product (với vector đã L2-normalize thì = cosine).

### 3.2 Vì sao điều đó là siêu năng lực

Vì document không cần biết query là gì, ta có thể **encode toàn bộ kho dữ liệu
TRƯỚC, lúc ingest** — làm một lần, lưu vector vào index. Lúc người dùng gõ query:

1. Encode query → 1 vector (**1 lần forward pass duy nhất**, vài chục ms).
2. Tìm k vector gần nhất trong index (kNN — với FAISS HNSW chỉ vài ms kể cả
   hàng triệu vector, chi tiết ở chương 13).

Tổng chi phí mỗi query gần như **không phụ thuộc kích thước kho dữ liệu**.
Đó là lý do bi-encoder scale được tới hàng triệu, hàng tỷ document.

### 3.3 Liên hệ ML cổ điển

Đây chính là **feature extraction + kNN** mà bạn đã quen: thay vì hand-crafted
feature (HOG, TF-IDF...), ta dùng neural network học ra feature; thay vì
Euclidean, ta dùng cosine. Phần "tìm hàng xóm gần nhất" vẫn y nguyên tư duy kNN.

### 3.4 SigLIP của FUFU là bi-encoder

SigLIP (chương 07) là bi-encoder **ảnh-text**: tháp ảnh encode keyframe lúc
ingest → FAISS; tháp text encode query lúc search. Hai tháp khác kiến trúc
(ViT vs text transformer) nhưng được huấn luyện để output rơi vào **cùng không
gian** — nên dot product giữa "vector của câu tiếng Việt" và "vector của bức
ảnh" mới có nghĩa.

### 3.5 Cái giá phải trả: nén mất thông tin

Toàn bộ một bức ảnh phức tạp (hay một đoạn văn dài) phải nhét vào **1 vector
cố định**. Giống như tóm tắt cả cuốn sách thành 1 câu rồi mới đem so sánh —
nhanh, nhưng:

- Chi tiết nhỏ (số nhà, tên riêng, "áo đỏ hay áo cam") dễ bị nén mất.
- Quan hệ giữa các thành phần ("người ĐUỔI chó" vs "chó ĐUỔI người") thường
  mờ nhạt — hai câu này ra vector khá gần nhau.
- Model không thể "đối chiếu từng từ của query với từng phần của document",
  vì lúc encode mỗi bên, nó chưa biết bên kia tồn tại.

Hệ quả thực tế: top-500 của dense channel thường **chứa** đáp án đúng (recall
tốt), nhưng đáp án có thể nằm ở hạng 30 thay vì hạng 1 (precision ở top đầu
chưa tốt). Đây chính là chỗ cross-encoder vào cuộc.

---

## 4. Cross-encoder — đọc CHUNG một lượt, chấm 1 điểm

### 4.1 Trực giác

Cross-encoder làm điều ngược lại: **ghép query và document thành MỘT input**,
đưa qua MỘT transformer encoder (BERT-style, chương 05):

```
input:  [CLS] người chơi cờ vua [SEP] hai người đàn ông ngồi bên bàn cờ ... [SEP]
            │
   [ 1 transformer encoder — attention trên TOÀN BỘ chuỗi ghép ]
            │
        head tuyến tính → 1 số duy nhất = relevance score
```

Vì query và document nằm chung một chuỗi, **attention được "nhìn chéo"**: token
"cờ vua" trong query attend trực tiếp vào "bàn cờ" trong document, "người chơi"
attend vào "hai người đàn ông". Model đối chiếu từng mẩu thông tin của bên này
với bên kia, qua nhiều lớp — thay vì so 2 bản tóm tắt đã nén.

Kết quả: cross-encoder phân biệt được những thứ bi-encoder hay lẫn — phủ định
("không đội mũ"), thứ tự quan hệ (ai làm gì với ai), chi tiết hiếm (tên riêng,
con số). Trên các benchmark passage ranking (vd MS MARCO), cross-encoder thường
hơn bi-encoder cùng cỡ một khoảng rõ rệt về độ chính xác top đầu.

### 4.2 Liên hệ ML cổ điển

Cross-encoder = một **classifier chấm từng CẶP**: input là cặp (query, document),
output là 1 score "cặp này khớp hay không". Giống logistic regression trên
feature của cặp — chỉ khác là "feature của cặp" được transformer tự học, cực
giàu, và việc tính nó rất đắt.

### 4.3 Cái giá phải trả: KHÔNG precompute được

Đây là điểm chí mạng. Score phụ thuộc đồng thời vào query **và** document, nên:

- Không thể tính trước lúc ingest (chưa biết query).
- Mỗi query mới phải chạy lại forward pass **cho TỪNG document**.
- Không có khái niệm "vector của document" để bỏ vào FAISS — không kNN được.

---

## 5. Tính tay chi phí — vì sao không cross-encoder tất cả

Lấy số thật của FUFU: BGE-reranker-v2-m3 mất **~5ms/passage trên RTX 3090**
(~50ms/passage trên CPU — ghi ngay trong docstring của `reranker.py`). Giả sử
kho dữ liệu cỡ thi đấu ~1 triệu đơn vị tìm kiếm, và tính tròn 10ms/passage:

| Phương án | Chi phí mỗi query | Thời gian |
|---|---|---|
| Cross-encoder chấm TẤT CẢ | 1.000.000 × 10ms = 10.000 giây | **~2,8 GIỜ / 1 query** |
| Bi-encoder + FAISS (top-500) | 1 lần encode + kNN | **vài chục ms** |
| Bi-encoder retrieve → cross-encoder rerank top-50 | vài chục ms + 50 × 5ms | **~vài trăm ms tổng** |

Hai giờ rưỡi cho một query là vô nghĩa trong cuộc thi mà operator cần kết quả
trong ~1 giây. Nhưng bỏ hẳn cross-encoder thì phí mất độ chính xác. Lời giải
chuẩn ngành: **đừng chọn, hãy XẾP TẦNG.**

### Kiến trúc phễu: RETRIEVE (rẻ, rộng) → RERANK (đắt, hẹp)

```
 1.000.000 đơn vị trong kho
        │  bi-encoder + FAISS / BM25  (vài ms — quét RỘNG, mục tiêu: ĐỪNG BỎ SÓT)
   top-500 ứng viên
        │  fuse 3 kênh (chương 14)
   top-50 ứng viên
        │  cross-encoder  (50 × 5ms ≈ 250ms — xét KỸ, mục tiêu: XẾP ĐÚNG THỨ HẠNG)
   top-20 trả về
```

Phân công rõ ràng:
- **Tầng retrieve** chịu trách nhiệm về **recall**: đáp án đúng phải lọt vào
  top-500. Rẻ nên quét được cả kho.
- **Tầng rerank** chịu trách nhiệm về **precision ở top đầu**: trong 50 ứng
  viên, đẩy đáp án đúng lên hạng 1-3. Đắt nhưng chỉ chạy trên 50 mẫu.

Nếu đáp án không lọt top-50 thì rerank giỏi mấy cũng vô ích — **rerank không
cứu được recall**, nó chỉ sửa thứ hạng.

**Liên hệ ML cổ điển:** đây là **cascade** kinh điển (Viola-Jones face detection:
tầng đầu là classifier siêu rẻ loại 99% cửa sổ ảnh, tầng sau đắt dần chỉ xét
phần còn lại). Cùng một triết lý: chi tiền tính toán đúng chỗ — chỗ ít ứng viên
nhưng cần quyết định khó.

---

## 6. BGE-reranker-v2-m3 trong FUFU

### 6.1 Model

`BAAI/bge-reranker-v2-m3` — cross-encoder **multilingual** (nền XLM-RoBERTa
của họ M3). "Multilingual" quan trọng với FUFU: nó chấm trực tiếp cặp
(query tiếng Việt, passage tiếng Việt) mà **không cần dịch sang tiếng Anh** —
khác với dense channel phải expansion VI→EN vì SigLIP mạnh tiếng Anh hơn.

### 6.2 Nó chấm CÁI GÌ? — passage ghép từ text mô tả

Cross-encoder này là model **text-text**. Nhưng đơn vị kết quả của FUFU là
segment video/ảnh — vậy lấy đâu ra "document text"? Câu trả lời (trong
`search_engine.py`, cuối hàm `search()`): với mỗi hit trong top-50, FUFU **ghép
một passage** từ những gì các extractor đã viết ra lúc ingest:

```
passage = caption của best_frame
        + "objects: person, chessboard, table"   (nhãn YOLO, khử trùng lặp)
        + text ASR mạnh nhất của segment
        # các phần nối bằng " | " ; nếu rỗng hết → "(no text)"
```

Ví dụ với query *"người chơi cờ vua"*, một hit video có thể thành cặp:

```
("người chơi cờ vua",
 "hai người đàn ông ngồi đối diện bên bàn cờ trong công viên | objects: person, chair, bench | nước này hay đấy, chiếu tướng luôn")
```

Cross-encoder đọc chung cặp này và cho 1 logit; 50 logits → sắp lại thứ tự.

### 6.3 Hệ quả thiết kế QUAN TRỌNG: reranker bị "mù" ảnh

Để ý kỹ: BGE **không hề nhìn pixel nào của frame**. Nó chỉ đọc *text mô tả*
ảnh — caption do Qwen-VL viết, nhãn do YOLO đoán, lời thoại do PhoWhisper nghe.
Suy ra chuỗi hệ quả:

1. **Chất lượng rerank bị chặn trên bởi chất lượng annotation.** Caption sai →
   rerank sai theo. Caption bỏ sót chi tiết ("áo đỏ") → reranker không thể
   thưởng điểm cho chi tiết đó dù ảnh thật có.
2. **Frame không có caption/OCR/ASR → passage = `"(no text)"`** → reranker chấm
   một chuỗi vô nghĩa và gần như chắc chắn **đẩy hit đó xuống đáy top-50**, kể
   cả khi dense cosine của nó rất cao (tức là ảnh thật rất khớp query!). Nếu
   team tắt `enable_caption` để ingest nhanh (§7.3 PROJECT-CONTEXT), reranker
   sẽ hoạt động chủ yếu trên ASR — với video không lời thì gần như mù hoàn toàn.
3. Một chi tiết "đọc code mới thấy": docstring của `reranker.py` nói passage
   gồm cả OCR, nhưng code thật trong `search_engine.py` hiện **chỉ ghép caption
   + objects + ASR, KHÔNG có ocr_text** — chữ trên màn hình không đến được
   reranker (nó chỉ phục vụ kênh BM25 visual). Đây là một khoảng hở có thể vá.

**Hướng khắc phục đã nằm trong kế hoạch:** mục **C2 của `RESEARCH-PLAN.md`** —
*VLM rerank top-20*: dùng Qwen-VL nhìn **ảnh thật** + query và chấm "frame này
có khớp không", thay vì BGE đọc text mô tả. Đắt hơn nhiều (nên chỉ top-20),
nhưng gỡ được toàn bộ điểm mù ở trên. Đó là cross-encoder "xịn" cho bài toán
ảnh — còn BGE hiện tại là giải pháp text-proxy thực dụng.

> 🔗 **Trong FUFU:** đoạn build passage + gọi rerank nằm ở
> `app/backend/services/search_engine.py` (tìm `rerank_top_n` trong hàm
> `search()`); thời gian chạy được trả về trong `timing_ms.cross_rerank_ms`
> của response `/api/search` — soi số này để biết reranker tốn bao lâu thật.

### 6.4 Đọc hiểu tham số (config + code)

| Tham số | Giá trị | Ý nghĩa & vì sao |
|---|---|---|
| `retrieval.enable_reranker` | `true` | Công tắc tổng. `BGEReranker` init fail (thiếu model, hết VRAM) → tự `enabled=False`, search vẫn chạy, chỉ mất bước rerank (degrade mềm, không crash). |
| `retrieval.rerank_top_k` | `50` | Số hit đầu được rerank. To hơn = cơ hội "vớt" hit hạng sâu, nhưng tuyến tính đắt hơn (100 hit ≈ 500ms GPU). Hit từ hạng 51 trở đi **giữ nguyên thứ tự cũ**, nối vào sau. |
| cap passage `512` token | hard-code trong `reranker.py` | Cắt passage (`p[:512]` ký tự + `truncation=True, max_length=512` token) để batch 50 cặp không OOM và latency ổn định. Passage của FUFU (caption ~96 token + nhãn + 1 câu ASR) hiếm khi chạm trần. |
| `models.reranker` | `BAAI/bge-reranker-v2-m3` | Đổi model reranker tại đây (vd bản nhỏ hơn nếu thiếu VRAM). |

Lưu ý vận hành: reranker chạy fp16 trên CUDA, fallback fp32 CPU (~50ms/passage
→ top-50 ≈ 2,5s — lúc đó nên giảm `rerank_top_k` hoặc tắt hẳn).

---

## 7. Cảnh báo bẫy tên: hai thứ cùng gọi là "rerank" trong FUFU

Trong `app/backend/services/` có **hai file tên gần giống nhau, làm hai việc
hoàn toàn khác nhau**:

| | `rerank.py` | `reranker.py` |
|---|---|---|
| Hàm/class chính | `fuse_and_aggregate()` | `BGEReranker` |
| Bản chất | **Số học**: hợp nhất điểm 3 kênh theo trọng số + gom frame→segment | **Model**: cross-encoder chấm lại từng cặp (query, passage) |
| Có neural network không | Không (chỉ cộng/nhân/max-pool) | Có (transformer ~568M tham số) |
| Thuộc chương | **14** (BM25 + fusion) | **12** (chương này) |
| Chạy trên | toàn bộ hits của 3 kênh | chỉ top-50 sau fuse |

Khi teammate nói "sửa rerank đi", **hỏi lại ngay**: score fusion hay
cross-encoder? Sửa nhầm file là chuyện đã-được-dự-báo (PROJECT-CONTEXT §5 có
hẳn cảnh báo ⚠️ về cặp file này).

---

## 8. Tóm tắt 10 giây

- **Bi-encoder** (SigLIP): encode query/document RIÊNG → dot product. Document
  encode trước lúc ingest → query time chỉ 1 encode + kNN → **nhanh, scale
  triệu, nhưng nén mất chi tiết**.
- **Cross-encoder** (BGE-reranker): ghép (query, doc) vào 1 transformer,
  attention nhìn chéo từng từ → **chính xác hơn hẳn, nhưng không precompute
  được**: 1M doc ≈ 3 giờ/query.
- Giải pháp chuẩn ngành = **phễu cascade**: retrieve rẻ lấy top-500 → rerank
  đắt top-50 → trả top-20. Rerank sửa thứ hạng, **không cứu được recall**.
- Trong FUFU, BGE chấm **passage text** (caption + objects + ASR), **không nhìn
  ảnh** → frame thiếu annotation thì rerank mù (`"(no text)"`); lối ra là VLM
  rerank (RESEARCH-PLAN C2).
- `rerank.py` (fusion, ch14) ≠ `reranker.py` (cross-encoder, chương này).

---

## 9. Câu hỏi tự kiểm tra

**Câu 1.** Vì sao bi-encoder cho phép tìm kiếm trên 1 triệu document trong vài
ms, còn cross-encoder thì không?

<details><summary>Đáp án</summary>

Bi-encoder encode document **độc lập với query**, nên toàn bộ kho được encode
sẵn lúc ingest và lưu vào index; mỗi query chỉ tốn 1 lần encode + 1 lần kNN.
Cross-encoder cho score phụ thuộc **đồng thời** vào query và document — không
thể tính trước, không có vector document để đánh index — nên mỗi query phải
forward pass lại cho từng document: 1M × ~10ms ≈ 2,8 giờ.
</details>

**Câu 2.** Về mặt cơ chế attention, điều gì khiến cross-encoder chính xác hơn
bi-encoder trên cùng một cặp (query, document)?

<details><summary>Đáp án</summary>

Trong cross-encoder, query và document nằm chung một chuỗi input, nên attention
**nhìn chéo**: mỗi token của query attend trực tiếp vào từng token của document
qua nhiều lớp — model đối chiếu chi tiết với chi tiết (phủ định, quan hệ chủ-vị,
tên riêng). Bi-encoder buộc mỗi bên nén toàn bộ ý nghĩa vào 1 vector **trước
khi** gặp bên kia, nên chỉ so được hai "bản tóm tắt" đã mất chi tiết.
</details>

**Câu 3.** Query của bạn có đáp án đúng nhưng nó nằm ở hạng 73 sau bước fuse.
`rerank_top_k=50`. Reranker có cứu được không? Vặn tham số nào?

<details><summary>Đáp án</summary>

Không — reranker chỉ xếp lại **top-50**, các hit từ 51 trở đi giữ nguyên thứ tự
và nối vào sau. Đây là minh hoạ "rerank không cứu được recall". Cách vặn: tăng
`rerank_top_k` (vd 100 — chấp nhận thêm ~250ms GPU), hoặc sửa tầng retrieve/fuse
để đáp án vào được top-50 ngay từ đầu (tăng `top_k_dense`, chỉnh weights — ch14).
</details>

**Câu 4.** Một frame có dense cosine rất cao với query nhưng được ingest lúc
`enable_caption=false`, video không lời. Sau bước cross-encoder rerank, hạng của
nó nhiều khả năng thay đổi thế nào và vì sao?

<details><summary>Đáp án</summary>

Nhiều khả năng **tụt xuống đáy top-50**. Passage của nó chỉ còn nhãn YOLO (hoặc
rỗng hoàn toàn → `"(no text)"`), vì BGE không nhìn ảnh mà chỉ đọc text mô tả.
Cross-encoder chấm chuỗi nghèo thông tin đó rất thấp, bất kể ảnh thật khớp đến
đâu — điểm mù cố hữu của text-proxy rerank, và là động cơ của VLM rerank
(RESEARCH-PLAN C2: Qwen-VL nhìn ảnh thật để chấm top-20).
</details>

**Câu 5.** Kiến trúc retrieve→rerank giống kỹ thuật cổ điển nào trong computer
vision, và nguyên tắc chung là gì?

<details><summary>Đáp án</summary>

Giống **cascade classifier** (Viola-Jones face detection): tầng đầu rẻ, quét
rộng, loại đại đa số ứng viên dễ; tầng sau đắt, chỉ xét số ít ứng viên khó.
Nguyên tắc: phân bổ chi phí tính toán theo độ khó của quyết định — đừng trả
giá cross-encoder cho 999.950 document mà tầng rẻ đã đủ sức loại.
</details>

**Câu 6.** SigLIP và BGE-reranker đều "so query với nội dung". Nêu 2 khác biệt
căn bản giữa chúng trong FUFU.

<details><summary>Đáp án</summary>

(1) **Kiến trúc**: SigLIP là bi-encoder 2 tháp (encode riêng, so cosine);
BGE là cross-encoder (đọc chung 1 chuỗi, output 1 score). (2) **Modality đầu
vào**: SigLIP so text với **pixel ảnh thật**; BGE so text với **text mô tả**
(caption/objects/ASR) — không thấy ảnh. Ngoài ra khác vai trò: SigLIP phụ trách
recall trên cả kho, BGE phụ trách precision trên top-50.
</details>

**Câu 7.** Teammate bảo: "tăng trọng số rerank lên đi". Câu này mơ hồ ở đâu?

<details><summary>Đáp án</summary>

FUFU có 2 thứ tên "rerank": `rerank.py` chứa `fuse_and_aggregate` — hợp nhất
điểm 3 kênh, ở đó mới có "trọng số" (`retrieval.weights`, chương 14); còn
`reranker.py` chứa `BGEReranker` — cross-encoder, **không có trọng số** để
tăng (nó thay thế hoàn toàn thứ tự của top-50, không trộn điểm). Phải hỏi lại
ý họ là chỉnh weights fusion, hay chỉnh `rerank_top_k`/bật-tắt cross-encoder.
</details>

**Câu 8.** Vì sao passage gửi vào BGE bị cắt ở 512 token, và trong FUFU giới
hạn này có thường gây mất thông tin không?

<details><summary>Đáp án</summary>

Cắt 512 token để (a) batch 50 cặp không OOM VRAM, (b) latency ổn định, (c) khớp
giới hạn context hiệu quả của model. Trong FUFU passage = caption (Qwen-VL sinh
≤96 token) + 1 dòng nhãn objects + 1 snippet ASR — thường ngắn hơn 512 nhiều,
nên hiếm khi mất thông tin vì cắt; giới hạn này chủ yếu là lưới an toàn.
</details>

---

## 10. Đọc thêm

- Reimers & Gurevych, *Sentence-BERT* (2019) — paper khai sinh thuật ngữ
  bi-encoder vs cross-encoder cho sentence retrieval, có phân tích chi phí.
- Nogueira & Cho, *Passage Re-ranking with BERT* (2019) — cross-encoder rerank
  trên MS MARCO, khuôn mẫu của kiến trúc retrieve→rerank hiện đại.
- Chen et al., *BGE M3-Embedding* (2024) + model card `BAAI/bge-reranker-v2-m3`
  trên HuggingFace — chính model FUFU đang dùng.
- `RESEARCH-PLAN.md` mục **C2** (VLM rerank) và **C3** (SuperGlobal — rerank
  không cần model) — hai hướng nâng cấp tầng rerank của FUFU.
- Trong repo: `app/backend/services/reranker.py` (≈60 dòng, đọc 5 phút) và đoạn
  build passage trong `search_engine.py` — hai chỗ duy nhất cần đọc để nắm
  toàn bộ implementation.
- Chương liên quan: **07** (SigLIP — bi-encoder ảnh-text), **13** (FAISS — vì
  sao kNN nhanh), **14** (fusion — bước ngay trước cross-encoder).
