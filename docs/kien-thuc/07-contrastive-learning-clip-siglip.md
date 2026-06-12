# Chương 07 — Contrastive learning & CLIP/SigLIP: trái tim của FUFU

> **Vị trí trong giáo trình:** Phần II — Các model nền tảng. Đây là chương quan trọng nhất
> của cả giáo trình: mọi thứ khác trong FUFU (FAISS, fusion, rerank, query expansion...)
> đều xoay quanh một ý tưởng duy nhất được trình bày ở đây.

---

## 1. Vì sao chương này tồn tại trong FUFU

Hãy quay lại bài toán gốc của FUFU: người dùng gõ một **câu tiếng Việt** —
*"người đàn ông chơi cờ vua trong công viên"* — và hệ thống phải tìm ra **khung hình video**
khớp nhất trong hàng triệu khung hình.

Dừng lại một giây và cảm nhận độ "quái" của bài toán này:

- Một câu văn là **chuỗi ký tự**: rời rạc, có ngữ pháp, có thứ tự từ.
- Một khung hình là **lưới pixel**: liên tục, 384×384×3 con số biểu thị màu sắc.

Hai thứ này **khác loài hoàn toàn**. Không có phép toán tự nhiên nào để "so sánh" một chuỗi
ký tự với một lưới pixel — giống như hỏi "con số 7 và màu xanh lá, cái nào *giống* bài hát này hơn?"

Với ML cổ điển, bạn đã quen với việc: muốn so sánh hai thứ, hãy đưa chúng về **cùng một
không gian đặc trưng** (feature space), rồi đo khoảng cách. SVM, k-NN, k-means — tất cả đều
hoạt động trên giả định "mọi điểm dữ liệu sống trong cùng một không gian, gần nhau = giống nhau."

CLIP/SigLIP chính là câu trả lời cho câu hỏi: **làm sao ép cả câu văn lẫn bức ảnh vào CÙNG
một không gian vector**, sao cho cặp (ảnh, mô tả đúng của ảnh đó) nằm gần nhau, còn cặp
(ảnh, mô tả sai) nằm xa nhau?

Khi đã có không gian chung đó, bài toán tìm kiếm của FUFU trở nên tầm thường về mặt khái niệm:

```
Lúc ingest:  mỗi keyframe video  → vector 1152 chiều  → cất vào FAISS
Lúc query:   câu tiếng Việt      → vector 1152 chiều  → hỏi FAISS: "vector nào gần tôi nhất?"
```

Toàn bộ "phép màu" nằm ở chỗ hai mũi tên `→` kia do **cùng một model SigLIP-2** thực hiện,
và model đó đã được huấn luyện để hai vector "nói cùng một ngôn ngữ".

> 🔗 **Trong FUFU:** cả hai mũi tên trên đều đi qua đúng một class —
> `SiglipEncoder` trong `app/common/encoder.py`. Hàm `encode_images()` chạy lúc ingest,
> `encode_text()` chạy lúc query, cùng load một model
> `google/siglip2-large-patch16-384` (khai báo tại `config/settings.yaml`, khối `models:`).
> Cả hai hàm đều trả về vector **đã L2-normalize** — chi tiết này quan trọng đến mức
> có riêng mục §6 bên dưới.

---

## 2. Cần biết trước

- **Chương 04 (Transformer)** — text encoder của SigLIP là một transformer encoder.
- **Chương 05 (Tokenization/BERT/GPT)** — câu văn vào model dưới dạng token như thế nào.
- **Chương 06 (ViT)** — image encoder của SigLIP là một Vision Transformer.
- Từ ML cổ điển: **logistic regression** (sigmoid + binary cross-entropy) và ý tưởng
  **margin** của SVM. Hai thứ này sẽ được tái sử dụng trực tiếp.

Chương này KHÔNG dạy lại transformer hay ViT — ta coi chúng là hai "hộp đen biến input
thành vector" và tập trung vào câu hỏi: *huấn luyện kiểu gì để hai hộp đen đó cho ra
vector so sánh được với nhau?*

---

## 3. Dual-encoder: hai tháp, một không gian

### 3.1 Kiến trúc

CLIP/SigLIP gồm **hai encoder hoàn toàn tách biệt**, thường vẽ như hai cái tháp:

```
   "a man playing chess"            [ảnh người chơi cờ]
            │                                │
   ┌────────▼────────┐              ┌────────▼────────┐
   │  TEXT ENCODER   │              │  IMAGE ENCODER  │
   │  (transformer,  │              │  (ViT,          │
   │   chương 05)    │              │   chương 06)    │
   └────────┬────────┘              └────────┬────────┘
            │                                │
        vector t ∈ R^1152            vector v ∈ R^1152
            └───────────┬────────────────────┘
                        ▼
              cosine(t, v) = "độ khớp"
```

Ba điểm cần khắc cốt ghi tâm:

1. **Không chia sẻ tham số.** Tháp text và tháp ảnh là hai mạng riêng, kiến trúc khác nhau,
   trọng số khác nhau. Thứ duy nhất chúng "chung" là **không gian đầu ra**: cả hai đều
   nhả ra vector cùng số chiều (1152 với SigLIP-2 Large), và được train sao cho các vector
   đó so sánh được với nhau.
2. **Hai tháp chạy độc lập.** Encode ảnh không cần biết query là gì; encode query không cần
   nhìn ảnh. Đây chính là kiến trúc **bi-encoder** — và là lý do FUFU có thể encode hàng triệu
   frame *một lần lúc ingest*, rồi lúc query chỉ cần encode đúng một câu. (So với
   **cross-encoder** phải chạy lại model cho từng cặp query-ảnh — đắt hơn hàng triệu lần,
   nhưng chính xác hơn; FUFU dùng nó để rerank top-50 — xem chương 12.)
3. **"Độ khớp" = một con số duy nhất:** cosine similarity giữa hai vector. Không có gì
   huyền bí hơn.

### 3.2 Liên hệ ML cổ điển

Nếu bạn từng làm feature engineering: dual-encoder chính là **hai hàm trích đặc trưng
học được** (learned feature extractors). Thay vì bạn tự nghĩ ra "đặc trưng số 1 = độ sáng
trung bình, đặc trưng số 2 = số cạnh dọc...", model tự học ra 1152 đặc trưng sao cho
chúng đồng thời (a) mô tả tốt nội dung ảnh và (b) "ăn khớp" với đặc trưng của câu văn mô tả nó.

Câu hỏi tiếp theo — và là câu hỏi trung tâm của chương — là: **train kiểu gì để được như vậy?**

---

## 4. Contrastive learning: kéo và đẩy

### 4.1 Trực giác

Dữ liệu huấn luyện của CLIP/SigLIP không phải là nhãn lớp (như ImageNet "đây là con mèo"),
mà là **hàng tỷ cặp (ảnh, caption)** vớt từ web: ảnh kèm alt-text, ảnh kèm chú thích bài báo,
thumbnail kèm tiêu đề... Rẻ, bẩn, nhưng nhiều vô tận.

Mục tiêu huấn luyện phát biểu bằng một câu:

> **Kéo** vector của cặp đúng (ảnh + caption của chính nó) lại gần nhau,
> **đẩy** vector của cặp sai (ảnh + caption của ảnh khác) ra xa nhau.

Đó là toàn bộ "contrastive learning" — học bằng cách **tương phản** cặp đúng với cặp sai.

### 4.2 Liên hệ SVM

Bạn đã gặp tinh thần này ở SVM: SVM không chỉ tìm một đường phân lớp *đúng*, mà tìm đường
**tối đa hoá margin** — đẩy hai lớp ra xa ranh giới nhất có thể. Contrastive learning làm
điều tương tự nhưng trong không gian embedding: không chỉ cần cosine(cặp đúng) > cosine(cặp sai),
mà muốn khoảng cách giữa hai nhóm điểm số đó **càng rộng càng tốt**. Hàm loss (mục 5, 6)
chính là công cụ tạo áp lực "nới margin" đó — điểm khác là ở SVM ta tối ưu trực tiếp trên
feature có sẵn, còn ở đây ta tối ưu **chính hàm sinh ra feature** (hai encoder), bằng
gradient descent (chương 02).

### 4.3 Vì sao không cần nhãn người gán?

Đây là điểm đẹp nhất: cặp (ảnh, caption đi kèm) là **nhãn miễn phí** — web tự sinh ra.
Cặp sai cũng miễn phí nốt: lấy ảnh này ghép với caption của ảnh *khác* trong cùng batch.
Không cần ai ngồi gán nhãn "con mèo / con chó". Nhờ vậy mới scale lên hàng tỷ cặp được —
và chính quy mô đó tạo ra khả năng zero-shot (mục 8).

---

## 5. CLIP loss: ma trận batch N×N và softmax

### 5.1 Setup

Mỗi bước huấn luyện, CLIP lấy một batch N cặp (ảnh, caption). Encode hết: được N vector ảnh
và N vector text. Tính cosine giữa **mọi** ảnh với **mọi** text → ma trận N×N.

Ví dụ batch N=3:

| | "người chơi cờ" | "con mèo trên ghế" | "bãi biển hoàng hôn" |
|---|---|---|---|
| **ảnh cờ vua** | **0.30** | 0.10 | 0.05 |
| **ảnh con mèo** | 0.08 | **0.25** | 0.12 |
| **ảnh bãi biển** | 0.02 | 0.06 | **0.28** |

**Đường chéo = cặp đúng** (in đậm). Mọi ô ngoài đường chéo = cặp sai (negative) — chú ý
ta được 3 positive và 6 negative *miễn phí* từ một batch 3 cặp.

### 5.2 Tính tay một hàng

CLIP biến mỗi **hàng** thành một bài phân lớp softmax (chương 01): "trong 3 caption này,
caption nào là của ảnh này?" Cosine được nhân với hệ số nhiệt độ (logit scale, model tự học,
ở đây lấy 10 cho dễ tính):

Hàng 1 (ảnh cờ vua): logits = (3.0, 1.0, 0.5)

```
exp(3.0) = 20.09     exp(1.0) = 2.72     exp(0.5) = 1.65
tổng = 24.46
P(caption đúng) = 20.09 / 24.46 ≈ 0.82
loss hàng 1 = −ln(0.82) ≈ 0.20
```

Model bị phạt 0.20 — chưa hoàn hảo vì hai caption sai vẫn "hút" mất 18% xác suất. Gradient
sẽ đẩy 0.30 lên và 0.10, 0.05 xuống: **kéo và đẩy**, đúng như trực giác mục 4.

Làm tương tự cho mỗi **cột** ("trong 3 ảnh, ảnh nào là của caption này?"), cộng trung bình
hai chiều → CLIP loss hoàn chỉnh. Tên kỹ thuật của dạng loss này là **InfoNCE** — bạn chỉ
cần nhớ: *softmax trên một hàng/cột của ma trận batch, đáp án đúng là ô đường chéo*.

### 5.3 Điểm yếu: nghiện batch khổng lồ

Để ý: mỗi positive chỉ phải "đấu" với N−1 negative **trong cùng batch**. N=3 thì bài thi
quá dễ — chọn 1 trong 3. Muốn model học phân biệt tinh (chess vs cờ tướng vs checkers),
phải cho nó đấu với hàng nghìn negative cùng lúc → CLIP gốc train với batch 32.768.
Batch to = cần cụm GPU to + các trick kỹ thuật để tính softmax phân tán toàn batch.
Đây chính là chỗ SigLIP cải tiến.

---

## 6. SigLIP: thay softmax bằng sigmoid — mỗi cặp là một bài logistic regression

### 6.1 Ý tưởng

SigLIP (Sigmoid Loss for Language-Image Pre-training, Google 2023) hỏi: tại sao phải bắt
các cặp trong batch thi đấu *với nhau* qua softmax? Sao không chấm **từng cặp một, độc lập**?

Với mỗi ô (i, j) của ma trận N×N, đặt một bài toán **phân lớp nhị phân**:

> "Cặp (ảnh i, text j) này là cặp ĐÚNG hay SAI?" — nhãn y = +1 nếu i = j (đường chéo), −1 nếu khác.

Logit của bài toán: `z = t·cosine(i,j) + b`, với t (scale) và b (bias) là hai số model tự học.
Xác suất "đúng" = sigmoid(z). Loss = binary cross-entropy.

**Đây chính là logistic regression** mà bạn đã thuộc lòng — với đúng *một* feature đầu vào
là cosine similarity, cộng thêm việc gradient chảy ngược qua cả hai encoder để feature đó
ngày càng "dễ phân lớp" hơn.

### 6.2 Tính tay

Giả sử model đã học được t = 20, b = −5 (bias âm là cố ý: trong ma trận N×N tuyệt đại đa số
ô là negative, bias âm giúp model "mặc định nghi ngờ" — giống điều chỉnh threshold khi dữ
liệu mất cân bằng lớp trong ML cổ điển):

```
Cặp đúng,  cosine = 0.30:  z = 20·0.30 − 5 = +1.0 → σ(+1.0) ≈ 0.73 → loss = −ln(0.73) ≈ 0.31
Cặp sai,   cosine = 0.05:  z = 20·0.05 − 5 = −4.0 → σ(−4.0) ≈ 0.018
                           P(đoán "sai" đúng) = 1 − 0.018 = 0.982 → loss ≈ 0.018
Cặp sai,   cosine = 0.25:  z = 0.0 → σ(0) = 0.5 → loss = −ln(0.5) ≈ 0.69  ← bị phạt nặng!
```

Cặp sai mà cosine cao (hard negative) bị phạt nặng nhất — model dồn lực đẩy đúng những cặp
"gây nhầm lẫn", thêm một lần nữa gợi nhớ SVM (chỉ các support vector — điểm gần ranh giới —
mới quyết định nghiệm).

### 6.3 Vì sao sigmoid thắng softmax về hiệu quả

| | CLIP (softmax) | SigLIP (sigmoid) |
|---|---|---|
| Mỗi cặp phụ thuộc | cả hàng/cột (phải tính tổng exp toàn batch) | **chỉ chính nó** |
| Batch nhỏ | điểm số "lạm phát" — thi đấu quá dễ | vẫn hợp lệ — mỗi cặp là bài thi riêng |
| Cài đặt phân tán | phức tạp (gather toàn batch) | đơn giản, ít bộ nhớ |
| Kết quả thực nghiệm | cần batch ~32k | đạt chất lượng tương đương/hơn với batch nhỏ hơn nhiều |

### 6.4 SigLIP-2 — thứ FUFU thực sự đang chạy

SigLIP-2 (2025) giữ nguyên sigmoid loss, thêm:

- **Multilingual:** train trên dữ liệu đa ngôn ngữ → encode trực tiếp tiếng Việt **vẫn ra
  vector có nghĩa**. Nhưng dữ liệu tiếng Anh trên web vẫn áp đảo về lượng và chất, nên
  embedding tiếng Anh "sắc" hơn. Đây là lý do FUFU làm **cả hai**: encode query gốc tiếng Việt
  *và* bản dịch tiếng Anh (NLLB — chương 11), rồi gộp lại (mục 10). Không chọn một, lấy cả hai.
- **Các objective phụ** bên cạnh contrastive: captioning-based pretraining, self-distillation,
  masked prediction — giúp feature ảnh giàu chi tiết hơn. Biết tên là đủ; chi tiết không cần
  cho việc vận hành FUFU.

> 🔗 **Trong FUFU:** model id `google/siglip2-large-patch16-384` được pin tại
> `config/settings.yaml` (khối `models:`). "patch16" và "384" là khái niệm ViT của chương 06
> (patch 16×16, ảnh resize 384×384). Lưu ý pin `transformers==4.50.0` trong
> `requirements.txt` — bản 4.49 chưa có SigLIP-2, bản 5.x đổi API.

---

## 7. Cosine similarity & L2 normalize: vì sao FUFU normalize mọi vector

### 7.1 Ví dụ số 2 chiều

Lấy hai vector 2 chiều: **a** = (3, 4) và **b** = (4, 3).

```
cosine(a, b) = (3·4 + 4·3) / (‖a‖·‖b‖) = 24 / (5 · 5) = 0.96
```

L2-normalize từng vector (chia cho độ dài của chính nó):

```
a' = (3/5, 4/5) = (0.6, 0.8)        ‖a'‖ = 1
b' = (4/5, 3/5) = (0.8, 0.6)        ‖b'‖ = 1

a' · b' = 0.6·0.8 + 0.8·0.6 = 0.48 + 0.48 = 0.96   ← đúng bằng cosine!
```

**Bài học:** trên vector đã normalize, **inner product (tích vô hướng) = cosine similarity**.
Cosine chỉ quan tâm *hướng* của vector, bỏ qua *độ dài* — hai frame cùng nội dung nhưng một
vector "dài" hơn (vì lý do số học bên trong model) vẫn được chấm điểm như nhau.

### 7.2 Hệ quả kiến trúc trong FUFU

Vì sao điều này đáng một mục riêng? Vì FAISS (chương 13) chỉ hỗ trợ vài metric, trong đó
`METRIC_INNER_PRODUCT` là rẻ nhất. Chuỗi suy luận:

```
muốn xếp hạng bằng cosine
  → normalize MỌI vector trước khi cất/so
  → inner product ≡ cosine
  → dùng thẳng FAISS IndexHNSWFlat + inner product, không cần metric đặc biệt
```

> 🔗 **Trong FUFU:** dòng `torch.nn.functional.normalize(feats, dim=-1)` xuất hiện trong **cả**
> `encode_images()` lẫn `encode_text()` của `app/common/encoder.py` — không có đường nào để
> một vector chưa normalize lọt vào hệ thống. FAISS index tạo với inner product trong
> `app/ingest/storage.py`. Bất biến này được ghi hẳn vào `PROJECT-CONTEXT.md` §6:
> *"Vector trong FAISS đã L2-normalize; metric = inner product = cosine."*
> Nếu một ngày bạn thêm encoder mới mà quên normalize, điểm số sẽ sai một cách *âm thầm* —
> kết quả vẫn ra, chỉ là xếp hạng vô nghĩa.

---

## 8. Zero-shot: vì sao chưa từng thấy video tin tức VN vẫn tìm được

SigLIP-2 chưa bao giờ được train trên corpus video của HCM AI Challenge. Vậy sao FUFU vẫn
tìm được cảnh "phát thanh viên mặc áo dài đọc bản tin"?

Vì model không học "phân lớp K lớp cố định" như ResNet trên ImageNet — nó học một **không gian
ngữ nghĩa tổng quát** từ hàng tỷ cặp ảnh-text đủ mọi chủ đề. Trong không gian đó, vùng
"người mặc áo dài", vùng "trường quay", vùng "đọc tin" đã tồn tại sẵn; ảnh mới chỉ việc
"rơi" vào đúng vùng. So với ML cổ điển: như thể bạn có một bộ feature extractor tốt đến mức
**k-NN trên feature đó hoạt động luôn với lớp chưa từng thấy** — không cần train lại gì cả.
Đó là "zero-shot": không (zero) lượt huấn luyện nào trên domain đích.

**Nhưng zero-shot có trần.** Những thứ hiếm/không xuất hiện trong dữ liệu web bị mù:
nhân vật địa phương, địa danh ít ảnh, sự kiện sau thời điểm train, và đặc biệt là các *quan hệ
tinh vi* trong ảnh (mục 9 ngay dưới). Khi nào zero-shot không đủ → đó là lúc nghĩ đến
fine-tune bằng LoRA (chương 16) hoặc bù bằng kênh khác.

---

## 9. Thang giá trị cosine thực tế + danh sách điểm mù

### 9.1 Đừng kỳ vọng cosine = 0.9

Một nhầm lẫn kinh điển của người mới: thấy cosine giữa query và frame "đúng" chỉ có 0.22,
tưởng hệ thống hỏng. Không — **cosine text-image của họ CLIP/SigLIP thường chỉ 0.05–0.3**,
kể cả với cặp khớp hoàn hảo. Lý do: text và ảnh là hai modality khác nhau, embedding của
chúng chiếm hai "nón" tách biệt trong không gian (hiện tượng *modality gap*); thêm nữa,
loss có scale t và bias b (mục 6.2) nên model chỉ cần cosine cặp đúng *cao hơn tương đối*
cặp sai, không cần tiến về 1.0.

**Hệ quả thực hành:** giá trị cosine tuyệt đối gần như vô nghĩa; chỉ **thứ tự tương đối**
trong một lần truy vấn là có nghĩa. 0.28 vs 0.13 trong cùng một query = khác biệt lớn;
0.28 của query này vs 0.30 của query khác = không so được.

Đây chính là lý do FUFU **min-max normalize** điểm dense trong mỗi lần search (kéo dải điểm
của top-500 kết quả về [0, 1]) trước khi trộn với điểm BM25 — vì 0.22 "thô" mà đem cộng
thẳng với điểm BM25 thì kênh dense sẽ luôn lép vế một cách giả tạo. Chi tiết công thức
fusion và vì sao BM25 lại được chuẩn hoá *kiểu khác* (chia 8.0, cap 1.0) thuộc về chương 14.

### 9.2 Bốn điểm mù của CLIP-family — và vì sao FUFU có tới 3 kênh

Họ CLIP/SigLIP nhìn ảnh theo kiểu "túi khái niệm" (bag of concepts) — rất giỏi nhận *cái gì
có mặt*, rất dở các thứ sau:

| Điểm mù | Ví dụ thất bại | FUFU bù bằng |
|---|---|---|
| **Đếm** | "ba người" vs "năm người" — embedding gần như nhau | Caption Qwen-VL có thể đếm (chương 08) + object detection đếm box (chương 10) |
| **Quan hệ không gian** | "người bên TRÁI xe" vs "bên PHẢI xe" | Caption VLM mô tả tường minh (chương 08) |
| **Chữ trong ảnh** | biển hiệu "PHỞ HÒA", banner, phụ đề — SigLIP gần như không đọc được | **OCR** EasyOCR → kênh BM25 visual (chương 10) |
| **Phủ định** | "KHÔNG đội mũ bảo hiểm" — từ "không" hầu như bị bỏ qua | Caption + rerank cross-encoder hiểu ngôn ngữ tốt hơn (chương 12) |

Và một điểm mù thứ năm hiển nhiên: SigLIP **không có tai** — mọi thông tin chỉ nằm trong
lời thoại ("phát biểu của bộ trưởng về...") cần kênh ASR (chương 09).

**Đây là câu trả lời cho câu hỏi thiết kế lớn nhất của FUFU:** *"đã có SigLIP thần thánh,
sao còn bày ra OCR + caption + detection + ASR + BM25 cho rườm rà?"* — Vì SigLIP một mình
mù 5 thứ trên. Ba kênh tìm kiếm song song (dense / BM25 visual / BM25 ASR — sơ đồ đầy đủ ở
chương 15) không phải trang trí: mỗi kênh che một vùng mù của kênh kia.

---

## 10. Mean của nhiều query variant: trọng tâm của chùm vector

Mảnh ghép cuối. FUFU không encode một câu query — nó encode **một chùm biến thể**:
câu gốc tiếng Việt + bản dịch tiếng Anh + 3 paraphrase (chương 11), rồi:

```python
# app/backend/services/search_engine.py (rút gọn)
text_vecs = self.encoder.encode_text(qe["all"])   # (5, 1152) — 5 biến thể
q_vec = text_vecs.mean(axis=0)                    # trung bình từng chiều
q_vec = q_vec / np.linalg.norm(q_vec)             # re-normalize!
```

**Trực giác:** mỗi cách diễn đạt là một điểm hơi lệch nhau trong không gian ngữ nghĩa
(cách dùng từ khác → vector khác chút). Lấy mean = lấy **trọng tâm của chùm điểm** —
phần "lõi nghĩa" chung được giữ lại và cộng hưởng, phần nhiễu riêng của từng cách diễn đạt
(từ ngẫu nhiên paraphrase chêm vào) triệt tiêu lẫn nhau. Cùng tinh thần với ensemble/bagging
trong ML cổ điển: trung bình nhiều ước lượng nhiễu → ước lượng ổn định hơn.

**Vì sao phải re-normalize?** Tính tay với hai biến thể từ ví dụ mục 7:

```
v1 = (0.6, 0.8),  v2 = (0.8, 0.6)        — cả hai có ‖·‖ = 1
mean = (0.7, 0.7)                         — nhưng ‖mean‖ = √0.98 ≈ 0.9899 < 1 !
re-normalize: (0.7, 0.7)/0.9899 ≈ (0.7071, 0.7071)
```

Trung bình của các vector đơn vị **không còn là vector đơn vị** (trừ khi chúng trùng nhau
hoàn toàn — các vector càng "tãi" rộng, mean càng ngắn). Mà toàn bộ hệ thống (mục 7.2) đứng
trên bất biến "mọi vector đều normalize, inner product = cosine" — nên phải chuẩn hoá lại
trước khi ném `q_vec` vào FAISS.

> 🔗 **Trong FUFU:** ba dòng trên nằm tại `app/backend/services/search_engine.py`
> (quanh dòng 119–125, hàm `search()`). Lưu ý tinh tế: chùm "all" (cả paraphrase) chỉ dùng
> cho kênh dense; kênh BM25 chỉ dùng [gốc, bản dịch] — lý do thuộc chương 14.

---

## Tóm tắt 10 giây

1. FUFU so sánh được **câu tiếng Việt với khung hình** vì SigLIP-2 ép cả hai vào **cùng
   không gian vector 1152 chiều**.
2. Kiến trúc: **dual-encoder** — ViT cho ảnh + transformer cho text, tham số riêng,
   không gian đầu ra chung; encode ảnh một lần lúc ingest, encode query lúc search.
3. Huấn luyện = **contrastive**: kéo cặp (ảnh, caption đúng) lại, đẩy cặp sai ra —
   tinh thần margin của SVM, dữ liệu là hàng tỷ cặp web miễn phí nhãn.
4. **CLIP** = softmax trên hàng/cột ma trận batch N×N (đường chéo là đáp án) → nghiện batch
   khổng lồ. **SigLIP** = sigmoid từng cặp độc lập — mỗi cặp là một bài **logistic
   regression** với feature duy nhất là cosine → train hiệu quả hơn; SigLIP-2 thêm
   multilingual nên query tiếng Việt chạy trực tiếp được (nhưng vẫn nên dịch EN kèm theo).
5. **L2-normalize mọi vector** → inner product = cosine → FAISS dùng metric rẻ nhất.
   Cosine text-image thực tế chỉ 0.05–0.3 — chỉ so **tương đối**, nên FUFU min-max normalize.
6. CLIP-family mù: đếm, quan hệ không gian, chữ trong ảnh, phủ định, âm thanh →
   **đó là lý do tồn tại** của caption (ch08), ASR (ch09), OCR (ch10) và fusion 3 kênh (ch14).
7. Nhiều biến thể query → **mean → re-normalize** = lấy trọng tâm chùm vector, khử nhiễu
   diễn đạt (cùng triết lý ensemble).

---

## Câu hỏi ôn tập

**Câu 1.** Vì sao FUFU có thể encode hàng triệu frame *trước* khi biết người dùng sẽ hỏi gì?
Tính chất kiến trúc nào của dual-encoder cho phép điều đó?

<details><summary>Đáp án</summary>

Vì hai tháp encoder chạy **độc lập**: vector của một frame không phụ thuộc query. Image
encoder chỉ cần pixel, text encoder chỉ cần câu chữ — chúng chỉ "gặp nhau" ở phép cosine
cuối cùng. Nên FUFU encode toàn bộ frame một lần lúc ingest, cất vào FAISS; lúc search chỉ
encode đúng câu query rồi tra cứu. Nếu dùng cross-encoder (query và ảnh phải vào model
*cùng nhau*) thì mỗi query sẽ phải chạy lại model trên từng frame — bất khả thi với hàng
triệu frame; vì thế cross-encoder chỉ dùng rerank top-50 (chương 12).
</details>

**Câu 2.** Trong ma trận batch 3×3 ở mục 5, hàng 1 có logits (3.0, 1.0, 0.5) sau khi scale.
Nếu model train tốt hơn và logits thành (5.0, 0.5, 0.0), xác suất softmax của caption đúng
là bao nhiêu? (Tính tay, exp(5)≈148.4, exp(0.5)≈1.65, exp(0)=1.)

<details><summary>Đáp án</summary>

P = 148.4 / (148.4 + 1.65 + 1) = 148.4 / 151.05 ≈ **0.982**. Loss = −ln(0.982) ≈ 0.018,
gần 0 — model gần như chắc chắn chọn đúng caption, đúng mục tiêu "kéo đường chéo lên,
đẩy phần còn lại xuống".
</details>

**Câu 3.** Giải thích vì sao SigLIP loss được ví là "mỗi cặp là một bài logistic regression".
Feature đầu vào của bài logistic regression đó là gì?

<details><summary>Đáp án</summary>

Với mỗi cặp (ảnh i, text j), SigLIP đặt bài phân lớp nhị phân "cặp đúng hay sai" với
logit `z = t·cosine(i,j) + b` và loss binary cross-entropy qua sigmoid — đúng công thức
logistic regression với **một feature duy nhất: cosine similarity** (t, b đóng vai trò
weight và bias). Khác biệt: gradient không dừng ở t, b mà chảy ngược vào cả hai encoder,
"uốn" không gian embedding để feature cosine ngày càng tách lớp tốt hơn. Vì mỗi cặp độc lập
(không cần tổng softmax toàn batch như CLIP), SigLIP không nghiện batch khổng lồ.
</details>

**Câu 4.** Hai vector đã L2-normalize **u** = (1, 0) và **w** = (0.6, 0.8). Tính inner
product và cho biết đó có phải cosine của chúng không. Nếu nhân đôi w thành (1.2, 1.6) thì
inner product với u đổi thế nào, cosine đổi thế nào?

<details><summary>Đáp án</summary>

u·w = 1·0.6 + 0·0.8 = **0.6** — và vì cả hai có độ dài 1, đây chính là cosine. Nhân đôi w:
inner product thành 1.2 (tăng theo độ dài) nhưng cosine **vẫn là 0.6** (cosine bỏ qua độ
dài). Bài học: inner product chỉ thay được cosine khi vector đã normalize — đúng bất biến
mà `encoder.py` của FUFU cưỡng chế, để FAISS inner-product cho ra xếp hạng cosine.
</details>

**Câu 5.** Bạn debug FUFU, thấy frame đúng có raw_cosine = 0.21 với query. Đồng đội bảo
"0.21 thấp quá, model hỏng rồi". Bạn phản biện thế nào?

<details><summary>Đáp án</summary>

0.21 là **bình thường** với họ CLIP/SigLIP: do modality gap giữa text và ảnh cộng với việc
loss có scale/bias riêng, cosine text-image của cặp khớp hoàn hảo cũng thường chỉ 0.05–0.3.
Con số tuyệt đối không nói lên gì; phải hỏi: *các frame khác trong cùng query được bao
nhiêu?* Nếu frame đúng 0.21 còn đám sai 0.08–0.12 thì hệ thống đang chạy tốt. Đây cũng là
lý do FUFU min-max normalize điểm dense theo từng lần search trước khi fusion (chương 14).
</details>

**Câu 6.** Query "biển hiệu quán PHỞ HÒA trên đường phố" gần như chắc chắn thất bại nếu chỉ
dùng kênh dense SigLIP. Vì sao, và FUFU thiết kế gì để cứu?

<details><summary>Đáp án</summary>

"Chữ trong ảnh" là điểm mù kinh điển của CLIP-family — SigLIP thấy "có biển hiệu, có quán
ăn" nhưng không *đọc* được chữ "PHỞ HÒA" trên biển. FUFU cứu bằng kênh OCR: lúc ingest,
EasyOCR đọc chữ trên mọi keyframe và ghi vào FTS5 `frame_text`; lúc query, kênh BM25 visual
match token "PHỞ HÒA" trực tiếp. Tương tự, đếm/quan hệ không gian/phủ định được bù bằng
caption Qwen-VL (chương 08), thông tin chỉ có trong lời thoại bù bằng ASR (chương 09) —
mỗi kênh che một vùng mù của SigLIP.
</details>

**Câu 7.** Ba biến thể query cho ra ba vector đơn vị, mean của chúng có độ dài 0.93.
Con số 0.93 (so với, ví dụ, 0.999) hé lộ điều gì về ba biến thể đó? Và vì sao vẫn phải
chia mean cho 0.93 trước khi đưa vào FAISS?

<details><summary>Đáp án</summary>

Độ dài mean < 1 cho biết các vector **không trùng hướng**; càng ngắn, chùm càng tãi rộng —
0.93 nghĩa là ba cách diễn đạt khác nhau đáng kể về ngữ nghĩa (0.999 thì gần như cùng một ý).
Vẫn phải re-normalize vì toàn hệ thống đứng trên bất biến "mọi vector độ dài 1 → inner
product = cosine"; nếu đưa vector dài 0.93 vào FAISS, mọi điểm inner-product bị co lại
0.93 lần — thứ tự trong một query không đổi nhưng bất biến bị phá, và mọi so sánh/ngưỡng
dựa trên thang cosine chuẩn sẽ sai. Code thật: `q_vec = q_vec / np.linalg.norm(q_vec)`
trong `search_engine.py`.
</details>

**Câu 8.** SigLIP-2 là multilingual, encode tiếng Việt trực tiếp được. Vậy vì sao FUFU vẫn
"đốt" thêm một model NLLB chỉ để dịch query sang tiếng Anh?

<details><summary>Đáp án</summary>

Vì "chạy được" khác "chạy tốt nhất": dữ liệu pretrain nghiêng mạnh về tiếng Anh nên
embedding text tiếng Anh thường khớp embedding ảnh chuẩn hơn embedding tiếng Việt cùng nghĩa.
FUFU không đánh đổi mà lấy cả hai: encode [query gốc VI, bản dịch EN, các paraphrase] rồi
mean → re-normalize (mục 10) — bản EN kéo trọng tâm về vùng "chuẩn" của không gian, bản VI
giữ các sắc thái mà bản dịch có thể đánh rơi. Chi tiết NLLB và paraphrase ở chương 11.
</details>

---

## Đọc thêm

- **Radford et al., 2021 — "Learning Transferable Visual Models From Natural Language
  Supervision" (CLIP).** Paper gốc; mục 2-3 đọc được ngay với nền chương 01-06.
- **Zhai et al., 2023 — "Sigmoid Loss for Language Image Pre-Training" (SigLIP).** Ngắn,
  ý chính nằm trọn trong 2 trang đầu + bảng so sánh batch size.
- **Tschannen et al., 2025 — "SigLIP 2: Multilingual Vision-Language Encoders..."** —
  chính xác model FUFU đang chạy; đọc phần objective phụ nếu tò mò mục 6.4.
- **Liang et al., 2022 — "Mind the Gap: Understanding the Modality Gap in Multi-modal
  Contrastive Representation Learning"** — giải thích sâu vì sao cosine text-image thấp (mục 9.1).
- **Trong repo:** `PROJECT-CONTEXT.md` §6 (bất biến FAISS/normalize) và §8 (sơ đồ query
  pipeline đầy đủ — bản đồ nối chương này với chương 11, 13, 14.
- **Chương kế tiếp:** chương 08 (Qwen-VL — kênh caption bù điểm mù), chương 12 (bi vs
  cross-encoder — nâng cấp khái niệm dual-encoder), chương 13 (FAISS — nơi các vector
  của chương này được cất và tra cứu).
