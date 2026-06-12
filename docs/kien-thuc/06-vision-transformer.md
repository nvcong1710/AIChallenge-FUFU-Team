# Chương 06 — Vision Transformer (ViT)

> **Một câu tóm tắt:** ViT cắt ảnh thành các ô vuông nhỏ, biến mỗi ô thành 1 token,
> rồi đưa cả chuỗi token đó vào đúng cái Transformer mà bạn đã học ở chương 04 —
> từ đây ảnh và văn bản được xử lý bằng **cùng một loại kiến trúc**.

---

## 1. Vì sao chương này tồn tại trong FUFU

Trái tim của FUFU là khả năng **so khớp câu tiếng Việt với khung hình video**. Muốn so khớp
được, vector của câu và vector của ảnh phải nằm trong **cùng một không gian** — và cách
tự nhiên nhất để làm điều đó là dùng **cùng một họ kiến trúc** cho cả hai phía.

Chương 04 đã gieo ý tưởng: *Transformer không quan tâm token đến từ đâu — chỉ cần là
một chuỗi vector, nó xử lý được*. Chương 05 cho thấy văn bản thành chuỗi token thế nào.
Chương này trả lời nốt nửa còn lại: **ảnh thành chuỗi token thế nào?** Đó chính là
Vision Transformer (ViT, Dosovitskiy et al. 2020).

Trong FUFU, ViT xuất hiện ở **hai chỗ xương sống**:

1. **Image encoder của SigLIP-2** (`google/siglip2-large-patch16-384`) — model embed mọi
   keyframe video/ảnh vào FAISS, và embed query text để tìm kiếm dense (chương 07).
2. **Vision encoder của Qwen2.5-VL** — "con mắt" của model sinh caption tiếng Việt
   cho từng frame (chương 08).

Hiểu ViT, bạn sẽ tự giải mã được vì sao model của FUFU tên là `large-patch16-384`,
vì sao đổi resolution lại tốn VRAM bình phương, và vì sao FUFU vẫn cần OCR riêng
dù ViT "nhìn" cả tấm ảnh.

> 🔗 **Trong FUFU:** model ViT chính được khai báo tại `config/settings.yaml` dòng 8
> (`siglip: google/siglip2-large-patch16-384`) và được load/chạy trong
> `app/common/encoder.py` (class `SiglipEncoder`, hàm `encode_images()`).

---

## 2. Cần biết trước

- **Chương 03 (CNN):** convolution, locality, weight sharing — để so sánh với ViT ở mục 6.
- **Chương 04 (Transformer):** self-attention, multi-head, positional encoding, khái niệm
  "chuỗi token vào → chuỗi vector ra". Chương này **không dạy lại** attention — chỉ thay
  đầu vào từ chữ sang ảnh.
- **Chương 05:** ý niệm token & embedding. Token văn bản tra bảng embedding; token ảnh
  thì *tính ra* bằng phép chiếu tuyến tính — khác biệt duy nhất nằm ở đó.

---

## 3. Ý tưởng then chốt: ảnh = chuỗi token

Transformer chỉ ăn một thứ: **chuỗi các vector cùng chiều**. Văn bản đã có cách tự nhiên
để thành chuỗi (từ/subword). Ảnh thì sao? Ảnh là lưới pixel 2D, không phải chuỗi.

Cách ngây thơ: coi **mỗi pixel là 1 token**. Ảnh 384×384 → 147.456 token. Nhớ lại chương 04:
self-attention tốn O(N²) theo độ dài chuỗi → 147.456² ≈ 21,7 **tỷ** cặp attention. Bất khả thi.

Ý tưởng của ViT đơn giản đến bất ngờ: **đừng lấy pixel, hãy lấy cả Ô (patch)**. Cắt ảnh
thành lưới các ô vuông 16×16 pixel không chồng lấn, mỗi ô = 1 token. Giống như đọc văn bản
theo *từ* thay vì theo *chữ cái* — mỗi đơn vị đủ lớn để mang nghĩa, và chuỗi đủ ngắn để
attention xử lý nổi.

```
Ảnh 384×384 ──cắt ô 16×16──> lưới 24×24 ô ──flatten từng ô──> chuỗi 576 token
                                                                    │
                                                          Transformer (chương 04)
                                                                    │
                                                        1 vector đại diện ảnh
```

Tên paper gốc nói hộ tất cả: *"An Image is Worth 16×16 Words"* — một tấm ảnh đáng giá
vài trăm "từ", mỗi từ là một ô 16×16.

---

## 4. Patch embedding — tính tay từng con số

### 4.1 Cắt ảnh thành ô

Lấy đúng cấu hình FUFU đang chạy: ảnh đầu vào **384×384**, patch **16×16**.

- Mỗi chiều có 384 / 16 = **24 ô**.
- Tổng số token: 24 × 24 = **(384/16)² = 576 token**.

So sánh: một câu query tiếng Việt dài ~10-20 token (chương 05). Vậy với Transformer,
một tấm ảnh "dài" cỡ một đoạn văn ~576 từ. Hoàn toàn trong tầm xử lý.

### 4.2 Mỗi ô thành 1 vector

Mỗi ô 16×16 có 3 kênh màu (RGB) → flatten thành vector thô dài:

$$16 \times 16 \times 3 = 768 \text{ số}$$

Vector thô này đi qua **một lớp linear duy nhất** (đúng nghĩa: nhân ma trận W kích thước
768 × D rồi cộng bias — như một perceptron của chương 01) để chiếu về chiều ẩn D của model.
Với SigLIP-2 **Large**, D = 1024:

$$\underbrace{x_{\text{patch}}}_{1 \times 768} \cdot \underbrace{W}_{768 \times 1024} + b = \underbrace{token}_{1 \times 1024}$$

Riêng lớp chiếu này có 768 × 1024 ≈ **0,79 triệu tham số** — tí hon so với cả model, nhưng
là cánh cửa duy nhất pixel đi vào thế giới Transformer.

**Liên hệ ML cổ điển:** patch embedding giống hệt bước *feature extraction thủ công* ngày xưa
(HOG, SIFT cho từng vùng ảnh) — chỉ khác là phép trích đặc trưng giờ là một phép chiếu tuyến tính
**học được**, và phần "hiểu" dồn hết cho attention phía sau.

**Mẹo cài đặt:** trong code thực tế (kể cả HuggingFace), bước "cắt + flatten + linear" được
gói gọn bằng **một lớp Conv2d với kernel 16×16, stride 16** — về mặt toán học y hệt nhau.
Đây là phép convolution *duy nhất* trong ViT.

### 4.3 Giải mã tên model của FUFU

Giờ bạn đọc được tên `google/siglip2-large-patch16-384` như đọc bảng số xe:

| Mảnh tên | Nghĩa | Hệ quả |
|---|---|---|
| `large` | cỡ model ViT (số tầng, hidden dim — xem mục 7) | chất lượng ↑, VRAM/tốc độ ↓ |
| `patch16` | mỗi ô 16×16 pixel | quyết định số token = (res/16)² |
| `384` | resolution đầu vào 384×384 | cùng patch16: res ↑ → token ↑ → chi tiết ↑, chi phí ↑↑ |

Thử tính nhanh các biến thể (cùng patch16):

| Resolution | Số token | Số cặp attention (N²) |
|---|---|---|
| 224×224 | 14² = 196 | ~38 nghìn |
| 384×384 (FUFU) | 24² = **576** | ~332 nghìn |
| 512×512 | 32² = 1024 | ~1,05 triệu |

Tăng resolution từ 224 → 384 (gấp ~1,7 lần mỗi chiều) làm số token tăng ~3 lần và chi phí
attention tăng ~8,7 lần. Đó là cái giá của việc "nhìn rõ hơn".

---

## 5. Position embedding — token ảnh cần biết mình ở đâu

Chương 04 đã nói: self-attention **mù vị trí** — xáo trộn thứ tự token, kết quả không đổi.
Với văn bản điều đó đã tệ ("chó cắn người" ≠ "người cắn chó"); với ảnh còn tệ hơn:
bầu trời ở trên, mặt đất ở dưới — đảo 576 ô lung tung thì "bãi biển hoàng hôn" và một
đống nhiễu màu cam là như nhau.

Giải pháp giống chương 04: **cộng position embedding vào từng token** trước khi vào
Transformer. Khác biệt ở chỗ ViT thường dùng position embedding **học được** (learned):
một bảng 576 vector (mỗi vector 1024 chiều với Large), vector thứ i cộng vào token thứ i.
Model tự học ra rằng "vị trí 0 là góc trên-trái, vị trí 25 là hàng thứ hai..." — thực nghiệm
cho thấy các embedding học được tự sắp xếp phản ánh đúng cấu trúc lưới 2D (các vị trí cùng
hàng/cột có embedding giống nhau hơn).

Hệ quả thực dụng đáng nhớ: bảng position embedding có kích thước **cố định theo số token**
→ muốn chạy resolution khác lúc inference phải **nội suy (interpolate)** bảng này. Đây là lý do
các model ViT công bố kèm resolution trong tên (`-384`) — nó không "co giãn tự do" như CNN.

---

## 6. Phần còn lại: y hệt chương 04, và một vector đại diện ảnh

Sau patch embedding + position embedding, không có gì mới: chuỗi 576 token đi qua
**N transformer block** (self-attention + FFN + residual + LayerNorm — đúng từng chi tiết
chương 04). Với SigLIP-2 Large: 24 block, 16 head, hidden 1024.

Mỗi tầng, mọi ô "nhìn" mọi ô khác. Ô chứa mặt người ở góc trái có thể attend thẳng đến
ô chứa quả bóng ở góc phải ngay từ **tầng 1** — ghi nhớ điều này để so với CNN ở mục sau.

Đầu ra là 576 vector (mỗi ô một vector, đã "ngấm" ngữ cảnh toàn ảnh). Nhưng FUFU cần
**một vector cho cả tấm ảnh** để nhét vào FAISS. Hai cách gom phổ biến:

- **CLS token** (ViT gốc, giống BERT chương 05): thêm 1 token "rỗng" đặc biệt vào đầu chuỗi;
  sau 24 tầng attention nó đã gom thông tin toàn ảnh → lấy vector của nó.
- **Pooling trên toàn bộ token**: mean pooling, hoặc tinh vi hơn là **attention pooling**
  (một lớp attention nhỏ học cách "chấm điểm" token nào quan trọng rồi lấy trung bình
  có trọng số). Họ SigLIP dùng kiểu này (MAP head).

Chi tiết SigLIP huấn luyện vector này *khớp với text* ra sao là chuyện của chương 07.
Chương này chỉ cần chốt: **ViT = máy biến ảnh → 1 vector**, và vector đó trong FUFU được
L2-normalize rồi ghi vào FAISS.

> 🔗 **Trong FUFU:** toàn bộ chuỗi "ảnh → 576 patch → 24 block → 1 vector → L2-normalize"
> chạy bên trong `SiglipEncoder.encode_images()` ở `app/common/encoder.py` (fp16, batch).
> Mỗi keyframe video qua đây thành đúng 1 vector, gán `faiss_id` trong
> `app/ingest/storage.py:add_frames`. Khi bạn search, câu query cũng thành 1 vector
> (`encode_text()`) và FAISS so cosine giữa hai bên — xem chương 13.

---

## 7. ViT vs CNN — cuộc đổi chác inductive bias lấy trần cao

Chương 03 cho thấy CNN thắng nhờ hai **inductive bias** (giả định cài sẵn vào kiến trúc):

1. **Locality:** pixel gần nhau thì liên quan — kernel 3×3 chỉ nhìn hàng xóm.
2. **Weight sharing / translation equivariance:** một bộ lọc phát hiện cạnh dùng chung
   cho mọi vị trí — con mèo ở góc nào cũng là con mèo.

Hai giả định này gần như luôn đúng với ảnh, nên CNN **học nhanh với ít dữ liệu** — giống
như Naive Bayes thắng mô hình phức tạp khi data nhỏ vì giả định độc lập "đủ đúng".

ViT vứt gần hết các giả định đó: attention toàn cục không ưu ái hàng xóm, không cài sẵn
khái niệm tịnh tiến. Hệ quả hai mặt:

| | CNN | ViT |
|---|---|---|
| Inductive bias | nhiều (locality, weight sharing) | rất ít (chỉ còn cấu trúc patch) |
| Ít data pretrain (~1M ảnh) | **thắng** | thua — phải tự học lại "pixel gần nhau liên quan" |
| Rất nhiều data (100M–10B ảnh) | bão hoà sớm hơn | **thắng — trần cao hơn** |
| Tầm nhìn | receptive field lớn dần qua từng tầng | **toàn cục ngay tầng 1** |
| Đổi resolution | khá thoải mái (conv trượt được) | phải nội suy position embedding |

Điểm "tầm nhìn" đáng nói nhất cho bài toán của FUFU: với CNN, hai vùng ảnh cách xa nhau
phải chờ nhiều tầng conv mới "gặp nhau" trong receptive field. Với ViT, **hai vùng xa nhau
nói chuyện trực tiếp ngay từ đầu**. Query kiểu *"người đàn ông bên trái chỉ tay về phía
tấm bảng bên phải"* đòi hỏi đúng loại quan hệ xa này.

Còn điểm yếu "cần nhiều data" thì sao? FUFU không bao giờ train ViT từ đầu — SigLIP-2 đã
được Google pretrain trên **~10 tỷ cặp ảnh-text (WebLI)**, vùng dữ liệu mà ViT vượt CNN.
Ta chỉ hưởng thành quả (tinh thần pretrain/finetune của chương 05, sẽ đào sâu ở chương 16).

---

## 8. Base / Large / Huge — kích cỡ nghĩa là gì, và lựa chọn của FUFU

"Cỡ áo" của ViT quy ước theo paper gốc:

| Cỡ | Số block | Hidden dim D | Số head | FFN dim | ~Tham số (phần vision) |
|---|---|---|---|---|---|
| Base (B) | 12 | 768 | 12 | 3072 | ~86M |
| Large (L) | 24 | 1024 | 16 | 4096 | ~307M |
| Huge (H) | 32 | 1280 | 16 | 5120 | ~632M |

(Con số chính xác lệch nhẹ giữa các họ model; tỷ lệ tương quan là thứ cần nhớ.)

Trade-off thực dụng:

- **VRAM & tốc độ:** Large ≈ 3,5× tham số của Base → encode chậm hơn ~2-3×, chiếm VRAM hơn.
  Trong FUFU (fp16) phần SigLIP vẫn chỉ ~0,4-0,8GB — nhỏ so với Qwen-VL 5GB, nên không phải
  bottleneck.
- **Chất lượng:** Large cho embedding "mịn" hơn — phân biệt được các cảnh na ná nhau
  (hai bản tin cùng studio, khác người dẫn) mà Base dễ lẫn. Với known-item search,
  recall@K của kênh dense ăn thẳng vào kết quả thi.

**Câu chuyện thật của FUFU:** kế hoạch v1 chốt SigLIP-2 **Base** 384 (xem memory
`project_stack_v1.md`), README-V2 cũng viết Base. Nhưng code hiện tại đã nâng lên **Large**
— đây chính là điểm lệch tài liệu số 2 trong PROJECT-CONTEXT §2. Lý do nâng: GPU đích
(RTX 3090 24GB) còn dư dả sau khi cộng mọi model ingest (~13GB tổng), nên đổi vài trăm MB
VRAM + chút tốc độ lấy chất lượng embedding là món hời. Nếu một ngày phải chạy máy yếu,
hạ về `base-384` chỉ là sửa một dòng trong `config/settings.yaml` — nhưng nhớ:
**đổi encoder = phải ingest lại toàn bộ** (vector Base và Large nằm ở hai không gian khác
nhau, thậm chí khác số chiều: 768 vs 1024 — FAISS index cũ vô dụng).

---

## 9. Resolution và chi tiết nhỏ — vì sao vẫn cần OCR

Một tấm ảnh 1920×1080 bị resize về 384×384 trước khi vào ViT. Làm phép tính cho một
biển hiệu chiếm 5% bề ngang khung hình:

- Trên ảnh gốc: biển rộng ~96 pixel — chữ đọc rõ.
- Sau resize về 384: biển còn ~19 pixel — **nhỏ hơn cả một patch 16×16**.
- Toàn bộ dòng chữ bị nén vào ~1-2 token trong số 576 token, hoà lẫn với pixel nền xung quanh.

Kết luận: ViT "thấy" *có một cái biển màu xanh ở đó*, nhưng **không đọc nổi chữ gì trên biển**.
Embedding SigLIP do đó giỏi semantic tổng thể (cảnh gì, ai làm gì, không khí ra sao) nhưng
mù chữ nhỏ. Đây chính là lỗ hổng mà kênh **BM25 visual** của FUFU trám vào: EasyOCR
(chương 10) chạy trên frame **độ phân giải cao hơn**, trích đúng dòng chữ trên biển hiệu/banner/
phụ đề và đánh index FTS5 — nên query *"quán phở Thìn"* match qua kênh OCR text chứ không
phải kênh dense.

Cùng logic, tăng resolution đầu vào ViT giúp chi tiết nhỏ "sống sót" qua patch embedding
(384 thay vì 224 chính là quyết định theo hướng này), nhưng chi phí tăng bình phương (mục 4.3)
và vẫn không thay thế được OCR cho chữ — bài toán *đọc ký tự* khác bài toán *hiểu cảnh*.

---

## 10. ViT đứng ở đâu trong các chương tiếp theo

- **Chương 07 (CLIP/SigLIP):** lấy ViT của chương này làm image encoder + text encoder
  (chương 04-05), huấn luyện contrastive để hai bên chung không gian → trái tim FUFU.
- **Chương 08 (Qwen-VL):** cũng dùng một ViT làm "con mắt", nhưng nối đầu ra vào một LLM
  để *sinh chữ* mô tả ảnh thay vì chỉ embed.

Nghĩa là một khái niệm chương này — patch embedding — đứng sau cả hai kênh hiểu-ảnh
của FUFU (dense vector + caption).

---

## 11. Tóm tắt 10 giây

1. ViT = cắt ảnh thành ô 16×16, mỗi ô flatten + linear projection thành 1 token,
   cộng position embedding, đưa vào Transformer y hệt chương 04, gom thành 1 vector ảnh.
2. `siglip2-large-patch16-384` của FUFU: large = 24 block/hidden 1024, patch16 + res 384
   → (384/16)² = **576 token/ảnh**.
3. CNN nhiều inductive bias → giỏi ít data; ViT ít bias → cần pretrain khổng lồ nhưng
   trần cao hơn + attention toàn cục từ tầng 1.
4. Resolution quyết định chi tiết nhỏ sống hay chết qua patch embedding; chữ trên biển hiệu
   thường chết → FUFU cần kênh OCR riêng.
5. ViT là image encoder của SigLIP-2 (ch.07) và vision encoder của Qwen-VL (ch.08).

---

## 12. Câu hỏi tự kiểm tra

**1. Ảnh 224×224, patch 14×14 — chuỗi đầu vào Transformer dài bao nhiêu token?**

<details><summary>Đáp án</summary>

Mỗi chiều 224/14 = 16 ô → 16² = **256 token**. (Đây là cấu hình patch14 thật của họ
CLIP ViT-L/14.) Công thức chung: (resolution/patch)².
</details>

**2. Vì sao không coi mỗi pixel là một token cho "chi tiết tối đa"?**

<details><summary>Đáp án</summary>

Self-attention tốn O(N²). Ảnh 384×384 = 147.456 pixel → ~21,7 tỷ cặp attention mỗi tầng,
bất khả thi về compute lẫn bộ nhớ. Patch 16×16 giảm N xuống 576 → ~332 nghìn cặp,
giảm ~65.000 lần, mà mỗi token vẫn đủ lớn để mang nghĩa (như từ so với chữ cái).
</details>

**3. Patch embedding của SigLIP-2 Large biến vector thô bao nhiêu chiều thành bao nhiêu chiều? Bằng phép gì?**

<details><summary>Đáp án</summary>

Vector thô 16×16×3 = **768** chiều → chiếu về hidden dim **1024** bằng **một lớp linear
duy nhất** (W: 768×1024 + bias) — thường cài đặt bằng Conv2d kernel 16 stride 16,
về toán học tương đương.
</details>

**4. Bỏ position embedding khỏi ViT thì model "mất" khả năng gì? Cho ví dụ kiểu query của FUFU bị ảnh hưởng.**

<details><summary>Đáp án</summary>

Attention mù vị trí → model chỉ thấy "túi các ô" (bag of patches), không biết ô nào ở đâu
trong lưới 2D. Mọi quan hệ không gian biến mất: query như *"người đứng **bên trái** chiếc
ô tô đỏ"* hay *"chữ ở **góc trên** màn hình"* không thể phân biệt với bố cục đảo ngược.
</details>

**5. Nêu 2 inductive bias của CNN mà ViT vứt bỏ, và hệ quả của việc vứt bỏ đó với (a) lượng data cần pretrain, (b) khả năng nắm quan hệ xa trong ảnh.**

<details><summary>Đáp án</summary>

Locality và weight sharing (translation equivariance). Hệ quả: (a) ViT phải tự học các
quy luật mà CNN được "cài sẵn" → cần data pretrain lớn hơn nhiều mới vượt CNN (SigLIP-2
pretrain ~10 tỷ cặp ảnh-text); (b) đổi lại, attention toàn cục cho phép 2 vùng xa nhau
tương tác trực tiếp ngay tầng 1, thay vì chờ receptive field của CNN lớn dần qua nhiều tầng.
</details>

**6. Đọc tên `siglip2-large-patch16-384`: nếu đổi sang `base-384`, những gì thay đổi trong FUFU và phải làm lại bước nào?**

<details><summary>Đáp án</summary>

`large→base`: 24 block→12, hidden 1024→768, ~307M→~86M params phần vision → nhanh hơn,
ít VRAM hơn, embedding kém mịn hơn. Số token không đổi (vẫn patch16, res 384 → 576 token).
Vì không gian vector và số chiều đổi (1024→768), **FAISS index cũ vô dụng — phải ingest lại
toàn bộ corpus**. Chỉ sửa 1 dòng `config/settings.yaml` nhưng trả giá bằng ingest lại.
</details>

**7. Vì sao FUFU vẫn cần EasyOCR dù SigLIP-ViT đã "nhìn" toàn bộ frame?**

<details><summary>Đáp án</summary>

Frame bị resize về 384×384 trước khi vào ViT; chữ nhỏ (biển hiệu, phụ đề) co lại còn vài
pixel — nhỏ hơn 1 patch 16×16 — nên bị nén mất trong 1-2 token, embedding chỉ giữ semantic
tổng thể chứ không "đọc" được ký tự. OCR chạy trên ảnh phân giải cao, trích đúng text và
index vào FTS5 (kênh BM25 visual) — hai kênh bù nhau, đúng triết lý hybrid của FUFU.
</details>

**8. Tăng resolution 384→768 (giữ patch16): số token và chi phí attention thay đổi thế nào?**

<details><summary>Đáp án</summary>

Token: (768/16)² = 48² = **2304**, gấp 4 lần 576. Chi phí attention ~N²: 2304²/576² =
**16 lần**. Resolution tăng k lần mỗi chiều → token tăng k², attention tăng k⁴. Ngoài ra
phải nội suy bảng position embedding vì số vị trí đổi.
</details>

---

## 13. Đọc thêm

- Dosovitskiy et al., *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale* (2020) — paper ViT gốc, đọc phần 3 (Method) là đủ.
- Bài giảng minh hoạ: *The Illustrated Vision Transformer* (nhiều bản trên web, tìm theo tên) — hình hoá patch embedding rất trực quan.
- Steiner et al., *How to train your ViT?* (2021) — thực nghiệm data/augmentation vs cỡ model, củng cố mục 7.
- Beyer et al., *Better plain ViT baselines for ImageNet-1k* (2022) — ViT "thuần" với pooling thay CLS, gần cách họ SigLIP làm.
- Tài liệu nội bộ: PROJECT-CONTEXT.md §2 (điểm lệch Base→Large), §4 (tech stack); chương kế tiếp **07 — Contrastive learning & CLIP/SigLIP** để xem vector ảnh từ chương này được "dạy" khớp với text ra sao.
