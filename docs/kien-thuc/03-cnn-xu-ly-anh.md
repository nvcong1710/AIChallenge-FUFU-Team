# Chương 03 — CNN: mạng xử lý ảnh

> **Vị trí trong giáo trình:** chương 03/19. Trước nó: 01 (NN cơ bản), 02 (huấn luyện).
> Sau nó: 04 (attention/transformer), 06 (ViT — "người kế nhiệm" của CNN trong vai trò encoder).

---

## 1. Vì sao chương này tồn tại trong FUFU

FUFU là hệ thống tìm kiếm multimedia: người dùng gõ câu tiếng Việt, hệ thống trả về
đoạn video/ảnh khớp nhất. Để làm được, máy phải **"nhìn hiểu" ảnh** — và CNN
(Convolutional Neural Network) là kiến trúc đầu tiên làm được điều đó ở quy mô lớn.

Trong FUFU hiện tại, encoder visual chính là SigLIP-2 — dùng ViT (chương 06), **không phải CNN**.
Vậy sao vẫn học CNN? Ba lý do:

1. **CNN vẫn sống trong các model chuyên dụng của FUFU.** EasyOCR (đọc chữ trên màn hình)
   dùng CRAFT detector + CRNN recognizer — cả hai đều là CNN. YOLO-World (phát hiện ~70 lớp
   đối tượng) cũng có backbone CNN. Hai mảnh này chạy trong pipeline ingest mỗi ngày.
2. **ViT mượn rất nhiều trực giác từ CNN** (chia ảnh thành patch, học feature phân cấp).
   Không hiểu CNN thì chương 06 sẽ như học đạo hàm mà chưa biết giới hạn.
3. **Các khái niệm cốt lõi — locality, weight sharing, pooling, transfer learning — xuất hiện
   khắp nơi trong FUFU**, kể cả ở chỗ không liên quan gì đến ảnh (max-pool trong fusion điểm số).

> 🔗 **Trong FUFU:** CNN chạy thật ở `app/extractors/ocr.py` (EasyOCR `[vi, en]` — CRAFT + CRNN)
> và `app/extractors/detection.py` (YOLO-World v2, file trọng số `yolov8l-world.pt`).
> Bật/tắt chúng qua `extractors.enable_ocr` / `enable_detection` trong `config/settings.yaml`.

---

## 2. Cần biết trước

Từ **chương 01** (NN cơ bản):
- Neuron = tổng có trọng số + hàm kích hoạt; layer = nhiều neuron; mạng = nhiều layer chồng nhau.
- Fully-connected (MLP): **mỗi** neuron nối với **mọi** đầu vào.
- Ý tưởng "mạng tự học feature thay vì ta thiết kế tay" — chương này sẽ thấy ý đó thành hình rõ nhất.

Từ **chương 02** (huấn luyện):
- Loss, gradient descent, backpropagation. Mọi thứ trong chương này vẫn được train y hệt —
  filter của CNN cũng chỉ là trọng số, học bằng gradient như thường.
- Khái niệm "mạng sâu khó train" (gradient yếu dần qua nhiều tầng) — sẽ cần khi nói về ResNet.

Nếu hai chương trên còn mơ hồ, đọc lại trước. Chương này không nhắc lại backprop.

---

## 3. Ảnh trong máy tính: ma trận pixel — và vì sao MLP "ngã ngựa"

### 3.1 Ảnh là gì với máy?

Một ảnh màu kích thước H×W là một **khối số H × W × 3**: mỗi pixel có 3 giá trị
(Red, Green, Blue), thường từ 0–255 (hoặc chuẩn hoá về 0–1). Ảnh 384×384 mà SigLIP của FUFU
nhận vào là khối:

```
384 × 384 × 3 = 442.368 con số
```

Với ML cổ điển, bạn quen việc đưa cho Random Forest một vector ~vài chục feature
(tuổi, thu nhập, số lần mua...). Ảnh thì khác: gần **nửa triệu** "feature" thô, và từng con số
riêng lẻ (pixel (217, 53) có R=142) **vô nghĩa** — nghĩa chỉ nảy sinh từ **quan hệ không gian**
giữa các pixel lân cận.

### 3.2 Thử cách ngây thơ: MLP thẳng trên pixel

Duỗi ảnh thành vector 442.368 chiều, nối vào 1 hidden layer 1.024 neuron (khiêm tốn):

```
Số trọng số = 442.368 × 1.024 ≈ 453 triệu tham số — CHỈ Ở LAYER ĐẦU TIÊN
```

453 triệu số float32 ≈ 1,8 GB chỉ để chứa trọng số một layer. Để so sánh: **toàn bộ**
SigLIP-2 Large (encoder chính của FUFU, hiểu được cả ảnh lẫn text đa ngôn ngữ) chỉ có
~1,1 tỷ tham số. Một layer MLP ngây thơ đã "đốt" gần nửa ngân sách đó mà chưa học được gì.

Nhưng số tham số mới là nửa vấn đề. Nửa còn lại tệ hơn:

**MLP phá huỷ cấu trúc không gian.** Khi duỗi ảnh thành vector, pixel (5,5) và pixel (5,6)
— hai hàng xóm sát vách — trở thành hai chiều input chẳng liên quan, ngang hàng với pixel
(300, 17) ở góc kia. MLP phải tự "khám phá lại" từ dữ liệu rằng pixel gần nhau thì liên quan
— một việc ta vốn **biết chắc từ trước**. Tệ nữa: con mèo ở góc trái và con mèo ở góc phải
kích hoạt hai bộ trọng số hoàn toàn khác nhau, nên MLP phải học "khái niệm mèo" lại từ đầu
cho **từng vị trí**.

Liên hệ ML cổ điển: đây giống hệt lý do bạn không ném 442 nghìn cột thô vào Logistic
Regression — quá nhiều chiều, quá ít cấu trúc, overfit chắc chắn. Thời tiền-deep-learning,
người ta giải bằng **feature engineering tay** (HOG, SIFT — trích cạnh/góc thủ công) rồi mới
đưa vào SVM. CNN chính là câu trả lời: **để mạng tự học bộ trích feature đó**, nhưng với một
kiến trúc được "nhồi sẵn" hiểu biết về ảnh.

---

## 4. Convolution: một filter trượt trên ảnh

### 4.1 Ý tưởng

Thay vì mỗi neuron nhìn **toàn bộ** ảnh, ta dùng một **filter** (kernel) — một ma trận nhỏ,
ví dụ 3×3 — **trượt** qua từng vị trí trên ảnh. Tại mỗi vị trí: nhân từng-phần-tử filter với
vùng ảnh nó đè lên, cộng tất cả lại → ra **một con số**. Số đó nói: *"vùng này giống mẫu
hình mà filter đang tìm đến mức nào"*.

Giống một người soi tem phiếu: cầm một tấm "khuôn" nhỏ, áp lần lượt lên từng góc tờ giấy,
chỗ nào khớp khuôn thì đánh dấu đậm.

### 4.2 Ví dụ số: filter phát hiện cạnh dọc — tính tay từng bước

Ảnh xám 5×5 (một "bức tường": nửa trái tối = 0, nửa phải sáng = 10 → có **cạnh dọc** ở giữa):

```
Ảnh I (5×5):          Filter F (3×3) — "dò cạnh dọc":
 0  0 10 10 10              -1  0  1
 0  0 10 10 10              -1  0  1
 0  0 10 10 10              -1  0  1
 0  0 10 10 10
 0  0 10 10 10
```

Đọc filter bằng lời: *"lấy độ sáng cột phải TRỪ độ sáng cột trái"*. Nếu phải sáng hơn trái
nhiều → kết quả lớn → có cạnh dọc.

**Bước 1 — đặt filter ở góc trên-trái** (đè lên các cột 1–3, hàng 1–3 của ảnh):

```
Vùng ảnh:    Nhân từng phần tử với F:
 0  0 10      0×(-1) + 0×0 + 10×1 = 10
 0  0 10      0×(-1) + 0×0 + 10×1 = 10
 0  0 10      0×(-1) + 0×0 + 10×1 = 10
                              Tổng = 30
```

**Bước 2 — trượt sang phải 1 pixel** (cột 2–4):

```
Vùng ảnh:    Tính:
 0 10 10      0×(-1) + 10×0 + 10×1 = 10   ← ×3 hàng
 0 10 10      Tổng = 30
 0 10 10
```

**Bước 3 — trượt tiếp** (cột 3–5, vùng toàn 10 — phẳng, không có cạnh):

```
10 10 10      10×(-1) + 10×0 + 10×1 = 0   ← trái phải bằng nhau, triệt tiêu
10 10 10      Tổng = 0
10 10 10
```

Trượt hết ảnh (cả chiều dọc — các hàng cho kết quả y hệt vì ảnh đồng nhất theo chiều dọc),
ta được **feature map** 3×3:

```
30 30 0
30 30 0
30 30 0
```

Đọc kết quả: cột giá trị lớn (30) đánh dấu **đúng nơi có cạnh dọc**; vùng phẳng ra 0.
Filter này "mù" với cạnh ngang — xoay ảnh 90° thì output toàn 0. Muốn bắt cạnh ngang?
Dùng filter F xoay 90°. Muốn bắt chấm tròn, góc nhọn, vệt chéo? Mỗi mẫu một filter.

**Điểm then chốt:** 9 con số trong filter này, ta vừa *thiết kế tay* để minh hoạ. Trong CNN
thật, 9 số đó là **trọng số học được bằng gradient descent** (chương 02). Mạng tự khám phá ra
nên dò mẫu hình gì — và thực nghiệm cho thấy layer đầu của CNN train xong gần như luôn tự
học ra các bộ dò cạnh/màu giống hệt thứ con người từng thiết kế tay suốt 40 năm computer vision.

### 4.3 Hai vũ khí làm nên CNN

Convolution thắng MLP nhờ "nhồi" sẵn hai giả định đúng-về-ảnh vào kiến trúc:

**Vũ khí 1 — Locality (tính cục bộ).** Mỗi output chỉ nhìn một vùng 3×3 lân cận, vì
*pixel gần nhau mới liên quan nhau*. Một cái mép bàn là chuyện của vài pixel cạnh nhau,
không phải chuyện giữa góc trái và góc phải ảnh. So với ML cổ điển: giống như bạn biết trước
feature nào tương tác với feature nào và chỉ cho model học đúng các tương tác đó — một dạng
**inductive bias** (kiến thức tiên nghiệm nhúng vào kiến trúc).

**Vũ khí 2 — Weight sharing (chia sẻ trọng số).** Cùng MỘT bộ 9 trọng số dùng lại ở **mọi
vị trí** trên ảnh, vì *một cái cạnh thì ở góc nào cũng là cái cạnh*. Hệ quả: mẫu hình học được
ở vị trí này tự động nhận ra được ở mọi vị trí khác (tính bất biến tịnh tiến).

Đếm lại tham số cho ảnh 384×384×3:

| Kiến trúc | Layer đầu tiên | Số tham số |
|---|---|---|
| MLP, hidden 1.024 | 442.368 → 1.024, full | **≈ 453.000.000** |
| CNN, 64 filter 3×3 | filter 3×3×3 (×64) + bias | 64 × (27+1) = **1.792** |

Chênh nhau **~250.000 lần** — đó là cái giá MLP trả cho việc "không biết gì về ảnh".
Ít tham số hơn = ít dữ liệu hơn để train, ít overfit hơn (cùng logic khiến bạn prune
Decision Tree hay tăng regularization cho Logistic Regression).

### 4.4 Feature map, nhiều filter, stride, padding (ngắn gọn)

- **Feature map:** lưới output của một filter (ma trận 3×3 toàn 30 và 0 ở trên). Nó là một
  "bản đồ nhiệt": chỗ nào sáng = chỗ đó có mẫu hình filter tìm.
- **Nhiều filter = nhiều detector.** Một conv layer thực tế có 64–512 filter chạy song song,
  mỗi filter một feature map → output là khối H'×W'×(số filter). Mỗi filter là một "chuyên gia":
  anh dò cạnh dọc, anh dò cạnh ngang, anh dò màu đỏ...
- **Stride:** bước nhảy khi trượt. Stride 1 = nhích từng pixel; stride 2 = nhảy 2 → output
  nhỏ đi một nửa mỗi chiều (vừa tính nhanh hơn vừa "zoom out").
- **Padding:** đệm viền 0 quanh ảnh để filter đặt được cả ở mép → output giữ nguyên kích thước
  thay vì co lại (5×5 qua filter 3×3 không padding còn 3×3; pad 1 viền thì vẫn 5×5).

---

## 5. Pooling: tóm tắt vùng, giữ tín hiệu mạnh nhất

**Max pooling 2×2**: chia feature map thành các ô 2×2 không chồng nhau, mỗi ô **giữ lại đúng
giá trị lớn nhất**:

```
Feature map 4×4:        Max pool 2×2 → 2×2:
 1  3 | 2  0
 4  2 | 1  1               4  2
 ----- ------       →      6  8
 0  6 | 5  8
 1  2 | 3  3
```

Trực giác: feature map trả lời "mẫu hình X có ở đây không?". Sau pooling, câu hỏi nới lỏng
thành "mẫu hình X có ở **đâu đó trong vùng này** không?" — ta giữ tiếng nói của detector
hăng hái nhất và vứt phần còn lại. Lợi ích kép: (1) output nhỏ đi 4 lần → các tầng sau rẻ hơn,
(2) mạng **bớt nhạy với xê dịch nhỏ** — con mèo lệch 1 pixel vẫn cho output gần như cũ.

Tư duy "max = lấy tín hiệu mạnh nhất, bỏ qua phần im lặng" này xuất hiện ở FUFU tại một chỗ
chẳng dính gì đến ảnh:

> 🔗 **Trong FUFU:** hàm `fuse_and_aggregate()` trong `app/backend/services/rerank.py` gom
> điểm của nhiều frame về một segment bằng **max-pool**: điểm dense của segment = cosine **cao
> nhất** trong các frame của nó (tương tự với BM25 visual/ASR). Logic y hệt max pooling:
> *"trong cảnh này có MỘT khung hình khớp mạnh là đủ; không bắt mọi khung hình phải khớp"*.
> Một shot 10 giây quay người chơi cờ chỉ cần 1 frame chụp rõ bàn cờ để cả segment được điểm cao.

(Còn average pooling — lấy trung bình thay vì max — dùng khi muốn "ý kiến tập thể" thay vì
"người to mồm nhất"; ít gặp hơn trong các CNN hiện đại, trừ tầng cuối.)

---

## 6. Xếp tầng: hệ phân cấp feature — lời hứa của chương 01 thành hình

Một CNN thật là chuỗi: `[Conv → ReLU → Pool] × N → Flatten → FC → output`. Điều kỳ diệu
xảy ra khi xếp **nhiều** tầng conv chồng nhau:

```
Ảnh pixel thô
  └─ Tầng thấp (conv 1–2):   cạnh, góc, vệt màu          ← như ví dụ §4.2
       └─ Tầng giữa (conv 3–5):  texture, hoạ tiết, bộ phận  (mắt, bánh xe, lông vằn)
            └─ Tầng cao (conv 6+):  đối tượng, bố cục          ("mặt mèo", "ô tô", "bàn cờ")
```

Vì sao tự nhiên có phân cấp? Mỗi tầng nhìn output của tầng dưới, nên **vùng ảnh gốc mà một
neuron "thấy được" (receptive field) phình to dần**: neuron tầng 1 thấy 3×3 pixel; neuron
tầng 2 nhìn 3×3 ô của tầng 1 → thấy 5×5 pixel gốc; càng lên cao càng thấy rộng (pooling còn
tăng tốc độ phình). Tầng 1 chỉ đủ tầm nhìn để dò cạnh; tầng 5 đủ tầm nhìn để dò "cụm cạnh +
texture xếp thành hình con mắt"; tầng 10 thấy gần cả ảnh — đủ để dò "mặt mèo".

Đây chính là **feature learning phân cấp** mà chương 01 hứa hẹn, nay nhìn thấy cụ thể:
không ai dạy mạng khái niệm "mắt" — nó **tự nổi lên** ở tầng giữa vì đó là cách hiệu quả
nhất để các tầng trên lắp ráp thành "mặt". So với ML cổ điển: bạn từng phải tự nghĩ ra
feature "tỷ lệ khoảng cách hai mắt / chiều rộng mặt" rồi đưa vào SVM; CNN tự xây chuỗi
feature đó, từ thô đến tinh, chỉ từ ảnh + nhãn.

---

## 7. ResNet và skip connection: cho mạng sâu thở được

Phân cấp càng sâu càng giàu → cứ chồng thêm tầng là tốt? Thực nghiệm năm 2015 nói không:
mạng 56 tầng tệ hơn mạng 20 tầng **ngay trên tập train** — tức không phải overfit, mà là
**không train nổi**. Hai thủ phạm:

1. **Gradient yếu dần** (chương 02): tín hiệu lỗi truyền ngược qua 56 tầng bị nhân liên tiếp
   bởi các đạo hàm nhỏ, đến tầng đầu thì gần như tắt — các tầng đầu mù đường, không học được.
2. **Học "không làm gì" cũng khó:** nếu 36 tầng thừa chỉ cần copy nguyên input (hàm đồng nhất
   `f(x) = x`) thì mạng 56 tầng ít nhất phải bằng mạng 20 tầng. Nhưng ép một chồng conv +
   activation phi tuyến xấp xỉ chính xác hàm đồng nhất lại là việc khó một cách oái oăm.

**ResNet** (Residual Network) sửa bằng một đường nối nghe đơn giản đến khó tin —
**skip connection**:

```
   x ──────────────┐
   │               │  (đường tắt: bê nguyên x qua)
 [Conv→ReLU→Conv]  │
   │               │
   F(x)            │
   └───── (+) ←────┘
          │
     y = F(x) + x
```

Mỗi khối không học "output là gì" nữa, mà học **phần chỉnh sửa** F(x) — *residual* (phần dư)
— so với input. Hai vấn đề trên tan biến:

- Muốn "không làm gì"? Chỉ cần đẩy F(x) → 0 (kéo trọng số về 0 — việc dễ nhất trần đời),
  vì `y = 0 + x = x`. Tầng thừa giờ vô hại thay vì phá hoại.
- Gradient có **đường cao tốc** chảy thẳng về các tầng đầu qua phép cộng (đạo hàm của `+x`
  là 1 — không bị nhân nhỏ đi), không còn tắt giữa đường.

Liên hệ ML cổ điển: rất giống **Gradient Boosting** — mỗi cây mới không dự đoán lại từ đầu
mà chỉ học phần dư (residual) của các cây trước. ResNet làm điều tương tự theo chiều sâu của
một mạng. Kết quả: mạng 152 tầng train ngon lành, và từ 2015 đến nay skip connection có mặt
trong **hầu hết** mọi kiến trúc sâu — kể cả Transformer (chương 04) và ViT (chương 06) mà
SigLIP của FUFU dựa trên. Học một lần, dùng cả giáo trình.

---

## 8. Transfer learning: backbone pretrained = feature engineering miễn phí

Nhận xét quan trọng từ §6: các tầng thấp/giữa của CNN học cạnh, texture, bộ phận — những thứ
**chung cho mọi bài toán ảnh**, chẳng riêng gì tập dữ liệu nào. Vậy thay vì train từ đầu
(cần hàng triệu ảnh có nhãn), ta:

1. Lấy một CNN **đã train sẵn** (pretrained) trên tập khổng lồ như ImageNet — gọi là **backbone**;
2. **Cắt bỏ** tầng phân loại cuối (tầng duy nhất gắn với bài toán cũ);
3. Đưa ảnh của mình qua backbone → nhận về một **vector feature** giàu ngữ nghĩa;
4. Gắn lên đó phần riêng của mình: một classifier nhỏ, một detection head, hoặc — như FUFU —
   dùng thẳng vector để so khớp.

Với dân ML cổ điển, đây là khoảnh khắc "à há": vector feature từ backbone đóng **đúng vai trò**
của feature engineering tay ngày xưa. Trước kia: ảnh → HOG/SIFT tự thiết kế → SVM. Bây giờ:
ảnh → backbone pretrained (bộ trích feature **học được**, tốt hơn tay nghề người) → model nhẹ
phía trên. Bạn thậm chí có thể lấy vector từ backbone rồi train... Random Forest lên trên —
hoàn toàn hợp lệ, và là cách rất phổ biến khi dữ liệu có nhãn ít.

Hai mức dùng backbone:
- **Feature extractor (đóng băng):** không train lại backbone, chỉ train phần đầu mới. Rẻ, nhanh,
  phù hợp khi ít dữ liệu. FUFU dùng mọi model ở chế độ này — *không train gì cả khi ingest/search*.
- **Fine-tuning:** mở băng một phần backbone, train tiếp với learning rate nhỏ trên dữ liệu
  của mình. Mạnh hơn nhưng đắt hơn — chi tiết để dành **chương 16 (fine-tuning/LoRA)**.

> 🔗 **Trong FUFU:** toàn bộ hệ thống là một bữa tiệc pretrained-backbone: SigLIP-2 Large
> (`config/settings.yaml`, key `models.siglip: google/siglip2-large-patch16-384`) embed
> frame thành vector 1 lần lúc ingest, lưu vào FAISS (`app/ingest/storage.py`); lúc search chỉ
> so cosine — backbone đóng vai feature extractor thuần tuý, không ai fine-tune gì.
> YOLO-World và EasyOCR cũng tải trọng số pretrained về dùng thẳng (`scripts/download_models.py`).

---

## 9. CNN trong FUFU hôm nay: nghỉ hưu ở "ghế chính", sống khoẻ ở "ghế chuyên gia"

Bức tranh để khép chương:

| Vai trò trong FUFU | Model | Kiến trúc lõi |
|---|---|---|
| Encoder visual chính (embed frame + query) | SigLIP-2 Large | **ViT** (chương 06–07) |
| Đọc chữ trên màn — phát hiện vùng chữ | EasyOCR / CRAFT | **CNN** |
| Đọc chữ trên màn — nhận dạng ký tự | EasyOCR / CRNN | **CNN** + RNN |
| Phát hiện đối tượng (~70 lớp) | YOLO-World v2 | backbone **CNN** |

Xu hướng chung của ngành: ViT đã thay CNN ở vai trò **encoder ngữ nghĩa tổng quát** (nhờ scale
tốt hơn với dữ liệu khổng lồ — chương 06 giải thích). Nhưng ở các bài toán **chuyên dụng, cần
nhanh, cần chi tiết không gian mịn** — dò vùng chữ, đếm vật thể, chạy realtime — CNN với hai
vũ khí locality + weight sharing vẫn là lựa chọn thực dụng. FUFU phản ánh đúng cục diện đó:
ViT làm "bộ não hiểu cảnh", CNN làm "giác quan chuyên trách". Output của cả hai phái gặp nhau
trong bảng `frame_text` FTS5 (OCR text + caption + labels) và index FAISS — nơi chương 13–14
sẽ tiếp quản câu chuyện.

---

## 10. Tóm tắt 10 giây

- Ảnh = khối số H×W×3; MLP thẳng trên pixel chết vì **~453 triệu tham số/layer** + mất cấu trúc không gian.
- **Convolution** = filter nhỏ trượt khắp ảnh, dò một mẫu hình; output là **feature map**.
- Hai vũ khí: **locality** (chỉ nhìn lân cận) + **weight sharing** (1 filter dùng mọi nơi) → ít tham số hơn MLP ~250.000 lần.
- **Max pooling** = tóm tắt vùng bằng tín hiệu mạnh nhất — cùng tư duy với max-pool frame→segment trong `fuse_and_aggregate` của FUFU.
- Xếp tầng → feature **phân cấp**: cạnh → texture/bộ phận → object.
- **ResNet/skip connection**: học phần dư `F(x)+x` → mạng trăm tầng train được; ý tưởng sống trong cả Transformer.
- **Transfer learning**: backbone pretrained = feature engineering tự học — cách FUFU dùng mọi model.
- Trong FUFU: ViT giữ ghế encoder chính; CNN vẫn chạy trong EasyOCR + YOLO-World.

---

## 11. Câu hỏi tự kiểm tra

**Câu 1.** Ảnh 384×384×3 nối thẳng vào hidden layer MLP 1.024 neuron cần bao nhiêu trọng số?
Nêu **hai** lý do (không phải một) khiến cách này thất bại.

<details>
<summary>Đáp án</summary>

384×384×3 = 442.368 input × 1.024 neuron ≈ **453 triệu trọng số** chỉ ở layer đầu.
Hai lý do thất bại: (1) **quá nhiều tham số** → cần lượng dữ liệu khổng lồ, overfit, tốn bộ nhớ;
(2) **mất cấu trúc không gian** — duỗi ảnh thành vector làm pixel lân cận trở thành các chiều
độc lập, và mẫu hình học ở vị trí này không tái dùng được ở vị trí khác (phải học lại "con mèo"
cho từng góc ảnh).
</details>

**Câu 2.** Dùng filter dò cạnh dọc `[[-1,0,1],[-1,0,1],[-1,0,1]]` đặt lên vùng ảnh 3×3 toàn
giá trị 7 (vùng phẳng). Tính output tại đó và giải thích ý nghĩa.

<details>
<summary>Đáp án</summary>

Mỗi hàng: 7×(−1) + 7×0 + 7×1 = 0. Tổng 3 hàng = **0**. Ý nghĩa: vùng phẳng (trái phải sáng
bằng nhau) không có cạnh dọc → filter "im lặng". Filter này chỉ kêu to khi cột phải sáng
hơn (hoặc tối hơn — ra số âm) cột trái, tức đúng nơi có cạnh dọc.
</details>

**Câu 3.** Locality và weight sharing mỗi cái mã hoá giả định gì về ảnh? Điều này giống khái
niệm gì khi bạn thiết kế feature cho model ML cổ điển?

<details>
<summary>Đáp án</summary>

Locality: *pixel gần nhau mới liên quan nhau* — mẫu hình thị giác là chuyện cục bộ.
Weight sharing: *một mẫu hình ở đâu cũng là mẫu hình đó* — cái cạnh ở góc trái hay góc phải
đều là cái cạnh (bất biến tịnh tiến). Cả hai là **inductive bias**: kiến thức tiên nghiệm
nhúng vào kiến trúc — tương tự việc bạn dùng hiểu biết domain để chọn/chế feature cho SVM
hay Random Forest thay vì ném dữ liệu thô vào; khác ở chỗ ở đây kiến thức nhúng vào *cấu trúc
mạng*, còn nội dung filter vẫn được học từ dữ liệu.
</details>

**Câu 4.** Max-pool 2×2 trên feature map `[[1,5],[3,2]]` cho ra gì? Vì sao FUFU lấy **max**
(chứ không phải trung bình) điểm các frame khi gom về segment trong `fuse_and_aggregate`?

<details>
<summary>Đáp án</summary>

Ra **5** (giá trị lớn nhất của ô 2×2). FUFU lấy max vì bài toán là known-item search: chỉ cần
**một** khung hình trong segment khớp mạnh với query là segment đó đáng trả về; lấy trung bình
sẽ bị các frame "phụ hoạ" (góc quay khác, cảnh chuyển) kéo điểm xuống và nhấn chìm tín hiệu
đúng. Cùng trực giác với max pooling: giữ detector kêu to nhất, bỏ qua phần im lặng.
</details>

**Câu 5.** Vì sao mạng CNN 56 tầng (không skip connection) lại tệ hơn mạng 20 tầng **ngay trên
tập train**, và `y = F(x) + x` giải quyết thế nào?

<details>
<summary>Đáp án</summary>

Tệ trên tập train → không phải overfit mà là **không tối ưu nổi**: gradient truyền ngược qua
quá nhiều tầng bị suy yếu (tầng đầu không học được), và các tầng "thừa" không tài nào học nổi
hàm đồng nhất để vô hại hoá chính mình. Skip connection cho mỗi khối học **phần dư** F(x):
muốn vô hại chỉ cần F(x)→0 (dễ), và gradient có đường cộng thẳng (`+x`, đạo hàm = 1) chảy về
các tầng đầu không suy hao. Tương tự Gradient Boosting: mỗi cây mới chỉ học phần dư của các
cây trước.
</details>

**Câu 6.** "Dùng backbone pretrained làm feature extractor" thay thế cho công đoạn nào trong
quy trình ML cổ điển? Cho ví dụ cụ thể trong FUFU.

<details>
<summary>Đáp án</summary>

Thay cho **feature engineering tay** (kiểu HOG/SIFT cho ảnh): backbone biến ảnh thô thành
vector feature giàu ngữ nghĩa, model/logic phía trên chỉ làm việc với vector đó. Trong FUFU:
SigLIP-2 Large (pretrained, không fine-tune) embed mỗi keyframe thành 1 vector lúc ingest,
lưu vào FAISS; lúc search chỉ so cosine giữa vector query và vector frame — backbone là
feature extractor đóng băng 100%.
</details>

**Câu 7.** FUFU dùng ViT làm encoder chính — vậy kể tên hai chỗ trong pipeline FUFU mà CNN
vẫn đang chạy thật, kèm file tương ứng.

<details>
<summary>Đáp án</summary>

(1) **EasyOCR** trong `app/extractors/ocr.py`: CRAFT detector (CNN tìm vùng chứa chữ) +
CRNN recognizer (CNN trích feature ký tự + RNN đọc chuỗi).
(2) **YOLO-World v2** trong `app/extractors/detection.py`: backbone CNN trích feature để
phát hiện ~70 lớp đối tượng open-vocabulary. Cả hai chạy ở bước annotate của ingest pipeline,
output đổ vào bảng `frame_text` FTS5 cho kênh BM25 visual.
</details>

**Câu 8.** Receptive field là gì, và vì sao nó giải thích được hiện tượng "tầng thấp học cạnh,
tầng cao học object"?

<details>
<summary>Đáp án</summary>

Receptive field = vùng ảnh gốc mà một neuron "nhìn thấy được". Tầng 1 với filter 3×3 chỉ thấy
3×3 pixel — vùng nhỏ vậy chỉ đủ chứa thông tin cỡ cái cạnh/chấm màu. Mỗi tầng chồng thêm
(cộng với pooling) làm receptive field phình ra; tầng giữa thấy vài chục pixel (đủ chứa một
con mắt, một bánh xe), tầng cao thấy gần cả ảnh (đủ chứa cả khuôn mặt, cả chiếc xe). Mạng
*buộc phải* học phân cấp từ nhỏ đến lớn vì tầm nhìn vật lý của từng tầng quy định như vậy.
</details>

---

## 12. Tài liệu đọc thêm

- **CS231n — Convolutional Neural Networks for Visual Recognition** (Stanford):
  https://cs231n.github.io/convolutional-networks/ — giáo trình kinh điển, có hình minh hoạ
  convolution tương tác.
- **Hướng dẫn trực quan convolution**: "A guide to convolution arithmetic for deep learning"
  (Dumoulin & Visin, 2016) — https://arxiv.org/abs/1603.07285 — mọi biến thể stride/padding bằng hình.
- **ResNet (bài gốc):** "Deep Residual Learning for Image Recognition" (He et al., 2015) —
  https://arxiv.org/abs/1512.03385 — đọc phần 1 và 3 là đủ lấy trực giác.
- **Feature phân cấp nhìn bằng mắt:** "Visualizing and Understanding Convolutional Networks"
  (Zeiler & Fergus, 2013) — https://arxiv.org/abs/1311.2901 — ảnh chụp "filter tầng 1 dò cạnh,
  tầng 5 dò mặt chó" nổi tiếng.
- **CRAFT** (detector trong EasyOCR): https://arxiv.org/abs/1904.01941 — đọc lướt để thấy CNN
  chuyên dụng cho chữ trông thế nào.
- Trong repo: `PROJECT-CONTEXT.md` §4 (tech stack) + §7.3 (chỗ OCR/detection chạy trong
  ingest video).

---

*Chương tiếp theo: **04 — Attention & Transformer** — cơ chế đứng sau mọi model ngôn ngữ
(và sau cả ViT ở chương 06).*
