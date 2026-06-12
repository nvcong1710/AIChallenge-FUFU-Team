# Chương 02 — Huấn luyện mạng neural

## Vì sao chương này tồn tại trong FUFU

FUFU không tự train model nào from scratch — mọi model trong stack (SigLIP-2, PhoWhisper, Qwen-VL, NLLB, BGE-reranker...) đều là **pretrained**, tức là người khác đã huấn luyện sẵn và ta chỉ tải về dùng. Vậy tại sao vẫn cần học cách huấn luyện?

Vì ba lý do thực dụng:

1. **Đọc hiểu model card và paper.** Khi chọn giữa SigLIP-2 Base và Large, ta cần hiểu "trained with batch size 32k, AdamW, cosine schedule" nghĩa là gì để đánh giá model.
2. **Chuẩn bị cho fine-tuning.** Nếu sau này team cần fine-tune reranker hoặc encoder trên dữ liệu tiếng Việt của cuộc thi (xem chương 16 về LoRA), toàn bộ khái niệm loss / optimizer / overfitting ở chương này sẽ dùng trực tiếp.
3. **Debug trực giác.** Hiểu vì sao model "học" được giúp ta đoán đúng khi nó fail: ví dụ vì sao SigLIP yếu với chữ trên màn hình (nó không được train để đọc chữ) nên FUFU phải bù bằng OCR.

Tóm lại: chương này là "bằng lái xe" — ta không chế tạo ô tô, nhưng phải hiểu động cơ hoạt động thế nào.

## Cần biết trước

- **Chương 01 — Mạng neural cơ bản:** neuron, weight, bias, activation function, forward pass. Chương này nói về chuyện *tìm ra* các weight đó bằng cách nào.
- Từ ML cổ điển: bạn đã biết linear regression (tối thiểu hóa sai số bình phương), logistic regression (xác suất + log loss), ridge regression (phạt L2), và overfitting của decision tree. Ta sẽ liên hệ liên tục với những thứ này.

---

## 1. Loss function — "thước đo độ sai"

Huấn luyện = tìm bộ weight làm model **ít sai nhất**. Muốn vậy phải định nghĩa "sai" bằng một con số: đó là **loss function**. Loss càng nhỏ → model càng tốt. Toàn bộ quá trình train chỉ là trò chơi: *chỉnh weight để kéo loss xuống*.

Tin tốt: bạn đã dùng loss function từ lâu mà có thể không gọi tên nó.

### 1.1 MSE — chính là thứ linear regression tối thiểu hóa

Khi bạn fit linear regression, thuật toán tìm đường thẳng tối thiểu hóa **tổng bình phương sai số**. Đó chính là **MSE (Mean Squared Error)**:

```
MSE = trung bình của (y_thật − y_dự_đoán)²
```

**Ví dụ số tính tay:** dự đoán giá nhà (đơn vị: trăm triệu), 3 căn:

| Căn | y thật | y dự đoán | sai số | sai số² |
|---|---|---|---|---|
| A | 30 | 28 | 2 | 4 |
| B | 25 | 29 | −4 | 16 |
| C | 40 | 39 | 1 | 1 |

MSE = (4 + 16 + 1) / 3 = **7.0**

Nếu chỉnh model để căn B dự đoán 27 thay vì 29 (sai số −2 → sai số² = 4), MSE mới = (4 + 4 + 1) / 3 = **3.0**. Loss giảm → model tốt lên. Lưu ý bình phương khiến sai số lớn bị phạt *rất* nặng (sai 4 bị phạt 16, gấp 4 lần sai 2) — giống hệt trực giác least squares bạn đã biết.

MSE dùng cho bài toán **hồi quy** (dự đoán số liên tục).

### 1.2 Cross-entropy — chính là log loss của logistic regression

Với bài toán **phân loại**, model xuất ra xác suất (qua sigmoid/softmax — chương 01). Loss phù hợp là **cross-entropy**, và nó *chính là* negative log-likelihood mà logistic regression tối ưu — chỉ đổi tên.

Ý tưởng: phạt model dựa trên **xác suất nó gán cho đáp án đúng**:

```
loss = −ln(xác suất model gán cho lớp đúng)
```

**Ví dụ số tính tay:** model phân loại ảnh là "mèo" hay "chó", ảnh thật là **mèo**:

| Model nói P(mèo) | loss = −ln(P) | Nhận xét |
|---|---|---|
| 0.9 | −ln(0.9) ≈ **0.105** | tự tin đúng → phạt nhẹ |
| 0.5 | −ln(0.5) ≈ **0.693** | ba phải → phạt vừa |
| 0.1 | −ln(0.1) ≈ **2.303** | tự tin sai → phạt nặng |
| 0.01 | −ln(0.01) ≈ **4.605** | rất tự tin sai → phạt RẤT nặng |

Điểm hay: loss tăng *phi tuyến* khi model tự tin sai. Nói P(mèo)=0.01 bị phạt gấp ~44 lần nói 0.9 — cross-entropy ép model "khiêm tốn khi không chắc", đúng tinh thần maximum likelihood của logistic regression.

> 🔗 **Trong FUFU:** Mọi model FUFU dùng đều được train bằng các loss họ hàng với cross-entropy. SigLIP-2 (`google/siglip2-large-patch16-384`, khai báo ở `config/settings.yaml`, load trong `app/common/encoder.py`) train bằng **sigmoid contrastive loss** — một biến thể cross-entropy trên cặp (ảnh, text) — chi tiết xem chương 07. PhoWhisper (`app/extractors/asr.py`) và Qwen-VL (`app/extractors/caption.py`) train bằng cross-entropy dự đoán token kế tiếp — xem chương 05 và 09.

### 1.3 Điểm chung cần nhớ

Linear regression, logistic regression, SVM (hinge loss), và mạng neural khổng lồ 7 tỷ tham số như Qwen-VL đều theo **cùng một công thức tổng quát**:

```
model tốt = argmin (loss trên dữ liệu train + regularization)
```

Khác biệt duy nhất: hàm dự đoán của NN phức tạp hơn nhiều. Câu hỏi tiếp theo: *tối thiểu hóa bằng cách nào?*

---

## 2. Gradient descent — "đi xuống đồi trong sương mù"

### 2.1 Trực giác

Tưởng tượng loss là một địa hình đồi núi: mỗi điểm trên bản đồ là một bộ weight, độ cao là loss tại bộ weight đó. Ta muốn tìm thung lũng thấp nhất, nhưng trời đầy sương mù — không nhìn thấy toàn cảnh, **chỉ cảm nhận được độ dốc dưới chân mình**.

Chiến lược hiển nhiên: nhìn hướng nào dốc xuống nhất, **bước một bước nhỏ** theo hướng đó, rồi lặp lại. Đó là gradient descent:

```
weight_mới = weight_cũ − learning_rate × gradient
```

- **Gradient** = độ dốc của loss theo từng weight ("tăng weight này lên 1 chút thì loss tăng hay giảm, nhanh cỡ nào").
- **Learning rate (lr)** = độ dài bước chân.

**Ví dụ số tính tay:** giả sử loss đơn giản L(w) = w², đang đứng ở w = 4 (loss = 16). Gradient của w² là 2w = 8 (dốc lên về phía w dương → phải đi ngược lại). Với lr = 0.1:

```
w mới = 4 − 0.1 × 8 = 3.2    → loss mới = 3.2² = 10.24  (giảm! 16 → 10.24)
lặp:    3.2 − 0.1 × 6.4 = 2.56 → loss = 6.55
lặp:    2.56 − 0.1 × 5.12 = 2.05 → loss = 4.20 ... tiến dần về w = 0 (đáy)
```

### 2.2 Learning rate quá to / quá nhỏ thì sao?

Vẽ bằng chữ, cùng ví dụ L(w) = w², bắt đầu w = 4:

```
lr vừa (0.1):   4 → 3.2 → 2.56 → 2.05 → ... → êm ái trượt xuống đáy ✓

lr quá nhỏ (0.001):  4 → 3.992 → 3.984 → ...
                     vẫn đi đúng hướng nhưng RÙA BÒ — tốn hàng nghìn bước,
                     và dễ kẹt ở vũng trũng nhỏ giữa đường.

lr quá to (1.1):  4 → 4 − 1.1×8 = −4.8 → −4.8 + 1.1×9.6 = 5.76 → −6.9 → ...
                  NHẢY QUA ĐÁY sang sườn đối diện, mỗi lần văng XA HƠN
                  → loss bùng nổ (diverge). Trong thực tế: loss = NaN.
```

Learning rate là hyperparameter quan trọng số một khi train/fine-tune (cách chọn có hệ thống: xem chương 17).

### 2.3 Liên hệ ML cổ điển

Đây không phải khái niệm mới với bạn: **logistic regression và SVM (dạng primal) cũng được giải bằng gradient descent** hoặc họ hàng của nó (sklearn `SGDClassifier` đúng nghĩa đen là vậy). Linear regression có nghiệm đóng (công thức ma trận giải một phát ra luôn), nhưng với dữ liệu lớn người ta vẫn dùng gradient descent.

Khác biệt với NN chỉ là: loss của logistic regression **lồi** (một thung lũng duy nhất — đi xuống chắc chắn đến đáy toàn cục), còn loss của NN **không lồi** (nhiều thung lũng, yên ngựa, vùng phẳng). Nghe đáng sợ, nhưng thực nghiệm cho thấy với mạng đủ lớn, các thung lũng tìm được hầu hết "đủ tốt" — đây là một trong những bất ngờ dễ chịu của deep learning.

---

## 3. Backpropagation — "truy ngược trách nhiệm lỗi"

Gradient descent cần gradient của loss theo **từng weight**. Mạng neural có hàng triệu/tỷ weight xếp thành nhiều tầng — tính gradient cho từng cái bằng cách nào cho nhanh? Đáp án: **backpropagation** — thực chất chỉ là chain rule (quy tắc đạo hàm hàm hợp) được tổ chức khéo léo.

### 3.1 Trực giác: dây chuyền đổ lỗi

Tưởng tượng một dây chuyền sản xuất: nguyên liệu → tổ A xử lý → tổ B xử lý → sản phẩm cuối bị lỗi. Muốn quy trách nhiệm:

- Sản phẩm cuối sai bao nhiêu? (loss)
- Tổ B đóng góp bao nhiêu vào cái sai đó? (gradient tầng cuối)
- Tổ A đưa nguyên liệu sai cho B, nên A chịu trách nhiệm bao nhiêu? (gradient tầng trước = lỗi của B **lan ngược** về A, nhân với "mức độ A ảnh hưởng B")

Lỗi được tính ở **đầu ra**, rồi **lan ngược** từng tầng về **đầu vào** — mỗi weight nhận đúng phần trách nhiệm của mình. Vì thế tên là *back*-propagation.

### 3.2 Ví dụ mạng 2 tầng tí hon

Mạng nhỏ nhất có thể (mỗi tầng 1 neuron, bỏ qua bias và activation cho gọn):

```
x ──(×w1)──> h ──(×w2)──> y_pred        loss = (y_thật − y_pred)²

Số cụ thể:  x = 2,  w1 = 3,  w2 = 0.5,  y_thật = 5

FORWARD (tính xuôi):
  h      = 2 × 3   = 6
  y_pred = 6 × 0.5 = 3
  loss   = (5 − 3)² = 4         → model đoán thấp 2 đơn vị

BACKWARD (truy ngược trách nhiệm):
  Tầng cuối:  y_pred đang THẤP hơn y_thật → cần TĂNG y_pred.
  w2: y_pred = h × w2, mà h = 6 (dương) → TĂNG w2 sẽ tăng y_pred. ✓ tăng w2
  Lan ngược qua w2 về h: tăng h cũng tăng y_pred (vì w2 = 0.5 dương)
       → "h ơi, cậu cần lớn hơn".
  w1: h = x × w1, mà x = 2 (dương) → TĂNG w1 sẽ tăng h, tức tăng y_pred. ✓ tăng w1
```

Để ý hai điều: (1) **hướng cập nhật của w1 phải đi qua w2** — nếu w2 âm, "tăng h" sẽ đảo thành "giảm h", và w1 phải giảm; trách nhiệm của tầng trước *phụ thuộc dây chuyền phía sau*. (2) Mỗi đại lượng trung gian (h) chỉ cần tính trách nhiệm **một lần** rồi chia tiếp cho tầng trước — nhờ vậy backprop tính được gradient của *tỷ* weight chỉ tốn cỡ một lần forward nữa. Đó là toàn bộ phép màu.

Trong thực tế bạn **không bao giờ tự tính** thứ này — PyTorch làm tự động (`loss.backward()`). Việc của bạn là hiểu trực giác để debug (ví dụ hiểu "vanishing gradient": trách nhiệm lan qua quá nhiều tầng bị nhân với số nhỏ liên tục → về đến tầng đầu thì còn ~0, tầng đầu không học được gì — một lý do ra đời của kiến trúc transformer, chương 04).

---

## 4. SGD, mini-batch, epoch — học theo "nhóm nhỏ"

Gradient "chuẩn" phải tính trên **toàn bộ** dữ liệu train (full-batch). Với 1 triệu ảnh thì mỗi bước đi tốn quá đắt. Hai thái cực và một thỏa hiệp:

| Chiến lược | Mỗi bước dùng | Ưu | Nhược |
|---|---|---|---|
| Full-batch GD | toàn bộ dataset | hướng đi chính xác | quá chậm, không vừa bộ nhớ |
| SGD thuần | **1 mẫu** | siêu rẻ mỗi bước | hướng đi nhiễu loạn, zigzag |
| **Mini-batch SGD** ✓ | một nhóm nhỏ (vd 32, 256) | cân bằng: gần đúng hướng + tận dụng GPU tính song song | phải chọn batch size |

Thực tế ngày nay "SGD" gần như luôn nghĩa là mini-batch. Nhiễu của mini-batch hóa ra còn **có lợi**: cú lắc ngẫu nhiên giúp nhảy thoát các vũng trũng nông.

**Bộ từ vựng bắt buộc thuộc** (xuất hiện trong mọi log train):

- **batch size**: số mẫu mỗi bước. Ví dụ 32.
- **step (iteration)**: 1 lần cập nhật weight (1 mini-batch).
- **epoch**: model đã "nhìn qua" toàn bộ dataset 1 lượt.

**Ví dụ số:** dataset 10 000 ảnh, batch size 32 → 1 epoch = ⌈10 000 / 32⌉ = **313 step**. Train 5 epoch = 1 565 step, mỗi ảnh được nhìn 5 lần.

Liên hệ ML cổ điển: random forest nhìn toàn bộ data một lần là xong (không có epoch); NN học **lặp đi lặp lại** — giống cách `SGDClassifier` của sklearn có tham số `max_iter` vậy.

---

## 5. Optimizer hiện đại — Momentum, Adam, AdamW và lịch learning rate

Gradient descent thuần có hai bệnh: zigzag trong "khe núi" hẹp, và dùng chung một learning rate cho mọi weight dù có weight cần bước to, weight cần bước nhỏ. Các optimizer hiện đại chữa hai bệnh này.

### 5.1 Momentum — viên bi lăn có đà

Thay vì bước theo gradient *hiện tại*, giữ một **vận tốc** tích lũy: bước mới = phần lớn vận tốc cũ + một phần gradient mới (hệ số quán tính thường là 0.9). Trực giác: viên bi lăn xuống đồi **có đà** — các cú zigzag trái/phải triệt tiêu nhau, còn hướng xuống dốc ổn định thì được cộng dồn → đi nhanh và mượt hơn, lăn qua được ổ gà nhỏ.

### 5.2 Adam / AdamW — vì sao mọi người mặc định dùng

**Adam** = momentum + **learning rate tự thích nghi cho từng weight**: weight nào có gradient thường xuyên lớn thì tự động bước nhỏ lại, weight có gradient bé thì bước to lên. Kết quả: gần như "cắm là chạy" — ít phải tinh chỉnh lr thủ công, hội tụ nhanh trên đủ loại bài toán. Đó là lý do Adam thành mặc định.

**AdamW** = Adam + sửa một lỗi kỹ thuật về cách áp **weight decay** (xem §6.2): tách phần phạt weight ra khỏi cơ chế thích nghi, giúp regularization hoạt động đúng. Ngày nay hầu hết model lớn — bao gồm các model FUFU dùng như SigLIP và họ Qwen — đều được train bằng **AdamW**. Khi fine-tune (chương 16), AdamW cũng là lựa chọn mặc định của bạn.

### 5.3 Learning rate schedule — đổi độ dài bước chân theo thời gian

Không ai giữ lr cố định suốt quá trình train. Hai kỹ thuật phổ biến (mức khái niệm):

- **Warmup**: vài trăm/nghìn step đầu, lr tăng dần từ ~0 lên giá trị đích. Lý do: lúc mới khởi tạo, weight ngẫu nhiên, gradient hỗn loạn — bước to ngay dễ văng (như lr quá to ở §2.2). Khởi động nhẹ nhàng đã.
- **Cosine decay**: sau warmup, lr giảm dần mượt theo hình nửa sóng cosine về ~0. Trực giác: càng gần đáy thung lũng càng nên **bước ngắn lại** để dò vào điểm thấp nhất thay vì nhảy qua nhảy lại quanh nó.

Hình dạng tổng thể vẽ bằng chữ: `lr: /‾‾\___` — leo nhanh (warmup), giữ đỉnh ngắn, rồi trượt dài xuống (decay). Bạn sẽ thấy cụm "linear warmup + cosine decay" trong gần như mọi paper/model card.

---

## 6. Overfitting trong deep learning và các "thuốc chữa"

### 6.1 Bệnh quen mà liều cao hơn

Bạn đã biết overfitting từ decision tree: cây mọc sâu không giới hạn sẽ **học thuộc lòng** tập train (mỗi lá 1 mẫu, train accuracy 100%) nhưng fail trên dữ liệu mới — chữa bằng `max_depth`, pruning. NN còn dễ overfit hơn: một mạng vài triệu tham số thừa sức nhớ vẹt vài chục nghìn mẫu. Triệu chứng kinh điển: **train loss tiếp tục giảm, validation loss chạm đáy rồi quay đầu tăng**.

### 6.2 Tủ thuốc

| Thuốc | Cơ chế | Liên hệ ML cổ điển |
|---|---|---|
| **Weight decay** | cộng phạt tỷ lệ với bình phương độ lớn weight vào loss → ép weight nhỏ, hàm "mượt" hơn | **chính là L2 regularization của ridge regression** — cùng công thức, λ‖w‖². Tham số `C` của SVM/logistic trong sklearn cũng là họ hàng (C nhỏ = phạt mạnh) |
| **Dropout** | lúc train, mỗi step **tắt ngẫu nhiên** một tỷ lệ neuron (vd 10–50%) → không neuron nào được ỷ lại neuron khác, mạng buộc học đặc trưng dư thừa, bền vững. Lúc inference bật lại hết | giống tinh thần **random forest**: mỗi cây chỉ thấy tập con feature ngẫu nhiên → ensemble bớt phụ thuộc một feature; dropout = "ensemble ngầm" của vô số mạng con |
| **Early stopping** | theo dõi validation loss, **dừng train khi nó hết giảm** (giữ checkpoint tốt nhất) | giống chọn số cây/độ sâu bằng validation thay vì cho mọc tối đa |
| **Data augmentation** | nhân tạo thêm dữ liệu bằng biến đổi giữ nguyên nhãn: lật/cắt/xoay/đổi màu ảnh, thêm nhiễu vào audio... | trong ML cổ điển ít gặp; trực giác = "cho học sinh nhiều đề biến thể để không học tủ" |

Lưu ý nghịch lý hiện đại: các model nền tảng (foundation model) được train trên dataset **khổng lồ** (hàng tỷ cặp ảnh–text) nên ít overfit theo nghĩa cổ điển — nhưng khi **bạn fine-tune** chúng trên vài nghìn mẫu của mình thì overfitting quay lại ngay, và cả tủ thuốc trên lại cần dùng (chương 16).

---

## 7. BatchNorm / LayerNorm — giữ tín hiệu ở thang ổn định

Vấn đề: trong mạng sâu, đầu ra mỗi tầng là đầu vào tầng sau. Nếu tầng 3 bỗng xuất ra số toàn cỡ 0.001 còn tầng 7 xuất số cỡ 500, gradient lan ngược qua chúng sẽ chỗ teo chỗ nổ → train chậm và bất ổn (lr nào cũng sai với ai đó).

Giải pháp: chèn các tầng **chuẩn hóa** — đưa tín hiệu về thang ổn định (trung bình ≈ 0, độ lệch chuẩn ≈ 1) trước khi đi tiếp. Bạn đã làm việc tương tự khi `StandardScaler` dữ liệu đầu vào cho SVM/logistic; ở đây chỉ khác là chuẩn hóa **cả các tầng giữa**, và liên tục trong lúc train:

- **BatchNorm**: chuẩn hóa theo **batch** — lấy mean/std của từng feature *tính trên các mẫu trong cùng mini-batch*. Hợp với CNN (chương 03); nhược điểm: hành vi phụ thuộc batch size, lúc inference phải dùng thống kê lưu sẵn.
- **LayerNorm**: chuẩn hóa theo **từng mẫu riêng lẻ** — mean/std tính trên các chiều feature của chính mẫu đó, không liên quan mẫu khác trong batch. Đơn giản, ổn định → là chuẩn hóa **mặc định trong transformer** (bạn sẽ gặp lại nó ở chương 04, và nó nằm bên trong mọi model transformer FUFU dùng: SigLIP, Qwen, PhoWhisper, NLLB, BGE).

Mức cần nhớ: norm layer = "bộ ổn áp" giữa các tầng. Không có nó, mạng sâu cỡ transformer gần như không train nổi.

---

## 8. Train/val/test và learning curve — bạn đã biết, chỉ thêm vài thói quen

Quy tắc chia 3 tập **giống hệt** ML cổ điển:

- **Train**: để gradient descent học weight.
- **Validation**: để chọn hyperparameter (lr, batch size, lúc nào early stop...). Model *không* học từ tập này, nhưng *bạn* thì có — chọn đi chọn lại theo val là một dạng overfit lên val.
- **Test**: niêm phong, chỉ đụng vào một lần cuối cùng để báo cáo.

Khác biệt thói quen so với ML cổ điển: vì train NN kéo dài hàng giờ/ngày, người ta **vẽ learning curve theo thời gian train** (loss theo step/epoch) chứ không chỉ theo kích thước dữ liệu như learning curve của sklearn. Cách đọc:

```
train loss giảm, val loss giảm theo  → đang học tốt, cứ tiếp ✓
train loss giảm, val loss quay đầu ↑ → overfitting, dùng tủ thuốc §6 / early stop
cả hai cùng cao, lì không giảm       → underfit: model quá bé, lr sai, hoặc bug
loss = NaN / nhảy loạn               → lr quá to (§2.2) hoặc dữ liệu lỗi
```

Riêng cho FUFU: hệ của ta là hệ **retrieval**, nên "tập test" của ta không phải accuracy phân loại mà là bộ query + đáp án đúng, đo bằng recall@K / MRR — toàn bộ chuyện đánh giá retrieval ở chương 19.

> 🔗 **Trong FUFU:** `scripts/eval_accuracy.py` chính là vòng "validation" của hệ thống ở mức *cấu hình*: mỗi khi team đổi tham số trong `config/settings.yaml` (ví dụ `retrieval.weights: {dense: 0.4, bm25_visual: 0.25, bm25_asr: 0.5}` — tham số tune chính của hệ), ta chạy eval trên bộ MSR-VTT đã dịch tiếng Việt để xem điểm tăng hay giảm. Tinh thần "đừng tin cảm giác, hãy đo trên tập giữ riêng" của mục này áp dụng nguyên xi, dù thứ được 'tune' là trọng số fusion chứ không phải weight của mạng.

---

## 9. Thực tế FUFU: vì sao gần như không bao giờ train from scratch

Nhìn lại stack trong `PROJECT-CONTEXT.md` §4: SigLIP-2 Large, EasyOCR, Qwen2.5-VL-7B, YOLO-World, PhoWhisper-medium, NLLB-600M, Qwen2.5-3B, BGE-reranker — **8 model, 0 model do team train**. Lý do lạnh lùng bằng số:

1. **Dữ liệu:** SigLIP được train trên cỡ **hàng tỷ** cặp ảnh–text đa ngôn ngữ. Team có... vài trăm giờ video cuộc thi. Train from scratch với từng đó dữ liệu = overfit thảm họa (§6).
2. **Compute:** train một model cỡ SigLIP Large tốn hàng trăm nghìn GPU-hour trên TPU/A100 cluster. Team có 1× RTX 3090. Riêng *chạy inference* ingest 100h video đã mất ~24h.
3. **Không cần thiết:** kiến thức các model này học (vật thể trông thế nào, tiếng Việt nói ra sao) là **kiến thức chung**, không riêng gì dữ liệu cuộc thi. Tải về dùng lại là hợp lý tuyệt đối.

Đây là **paradigm pretrain → finetune** thống trị deep learning hiện đại: ai đó giàu compute train model nền trên dữ liệu khổng lồ (pretrain), mọi người tải về dùng nguyên trạng (zero-shot, như FUFU đang làm) hoặc tinh chỉnh nhẹ trên dữ liệu riêng (fine-tune). Bức tranh đầy đủ của paradigm này: **chương 05**; kỹ thuật fine-tune tiết kiệm VRAM (LoRA — thứ khả thi trên 3090 của team): **chương 16**.

Việc của team FUFU vì thế không phải train, mà là: **chọn model đúng, ghép pipeline đúng, và tune các tham số "phía trên" model** (trọng số fusion, ngưỡng BM25, threshold shot detect...) — những thứ tối ưu bằng eval + tay/grid search chứ không phải gradient descent.

> 🔗 **Trong FUFU:** Toàn bộ "huấn luyện" mà team từng làm thực chất là *tải weight đã train* về máy: `python scripts/download_models.py` (~25GB). Các model id nằm ở `config/settings.yaml` (`models.siglip`, `extractors.asr_model`...) và được load trong `app/common/encoder.py`, `app/extractors/*.py`, `app/backend/services/{translator,paraphraser,reranker}.py`. Đổi model = đổi 1 dòng config, không phải train lại gì cả.

---

## Tóm tắt 10 giây

Train NN = định nghĩa **loss** (MSE cho hồi quy ~ linear regression; cross-entropy cho phân loại ~ logistic regression) → **gradient descent** đi xuống đồi từng bước nhỏ (learning rate to quá thì văng, nhỏ quá thì rùa) → **backprop** truy ngược trách nhiệm lỗi về từng weight bằng chain rule → chạy theo **mini-batch** (epoch/step/batch size) với **AdamW + warmup + cosine decay** → canh **overfitting** bằng val loss, chữa bằng dropout / weight decay (= L2 của ridge) / early stopping / augmentation → norm layer giữ tín hiệu ổn định. FUFU **không train from scratch** — dùng pretrained, chỉ tune tham số pipeline; muốn can thiệp vào weight thì fine-tune bằng LoRA (chương 16).

---

## Câu hỏi tự kiểm tra

**1. Model phân loại nói P(lớp đúng) = 0.25. Cross-entropy loss của mẫu này là bao nhiêu? (−ln(0.25) = ?)**

<details><summary>Đáp án</summary>

−ln(0.25) = ln(4) ≈ **1.386**. So sánh: nếu model nói 0.5 thì loss chỉ 0.693 — tự tin sai hơn bị phạt nặng hơn.

</details>

**2. Loss L(w) = w², đang ở w = 10, learning rate 0.3. Tính w sau 1 bước gradient descent. (Gradient = 2w.)**

<details><summary>Đáp án</summary>

w mới = 10 − 0.3 × (2×10) = 10 − 6 = **4**. Loss giảm từ 100 xuống 16.

</details>

**3. Train đang chạy thì loss thành NaN. Nghi phạm số một là gì và chỉnh theo hướng nào?**

<details><summary>Đáp án</summary>

Learning rate **quá to** → các bước cập nhật văng qua đáy ngày càng xa, loss bùng nổ. Giảm lr (vd chia 10) và/hoặc thêm warmup.

</details>

**4. Dataset 50 000 mẫu, batch size 100, train 3 epoch. Tổng cộng bao nhiêu step?**

<details><summary>Đáp án</summary>

1 epoch = 50 000/100 = 500 step → 3 epoch = **1 500 step**.

</details>

**5. Weight decay trong AdamW là họ hàng trực tiếp của kỹ thuật nào bạn đã biết từ ML cổ điển?**

<details><summary>Đáp án</summary>

**L2 regularization của ridge regression** — cùng ý tưởng phạt λ‖w‖² để ép weight nhỏ, hàm mượt, đỡ overfit. (AdamW chỉ sửa *cách áp* phạt này vào optimizer thích nghi cho đúng.)

</details>

**6. Train loss vẫn giảm đều nhưng validation loss bắt đầu tăng từ epoch 8. Chuyện gì đang xảy ra và bạn làm gì?**

<details><summary>Đáp án</summary>

**Overfitting** — model bắt đầu học thuộc lòng tập train (y hệt decision tree mọc quá sâu). Early stop tại checkpoint quanh epoch 8; nếu muốn train tiếp thì tăng dropout/weight decay hoặc thêm data augmentation.

</details>

**7. Vì sao backprop được gọi là "truy ngược trách nhiệm"? Trong ví dụ mạng x →(w1)→ h →(w2)→ y_pred, nếu w2 âm thì hướng cập nhật w1 có bị ảnh hưởng không?**

<details><summary>Đáp án</summary>

Lỗi tính ở đầu ra rồi lan ngược từng tầng, mỗi weight nhận phần trách nhiệm của mình qua chain rule. **Có** — trách nhiệm của w1 đi *xuyên qua* w2: nếu w2 âm thì "cần tăng y_pred" đảo thành "cần giảm h", hướng chỉnh w1 đảo theo. Gradient tầng trước luôn phụ thuộc các tầng sau.

</details>

**8. Vì sao team FUFU không train SigLIP from scratch trên dữ liệu cuộc thi, dù dữ liệu đó sát bài toán hơn?**

<details><summary>Đáp án</summary>

Thiếu cả ba thứ: **dữ liệu** (cần hàng tỷ cặp ảnh–text, ta có vài trăm giờ video → overfit), **compute** (hàng trăm nghìn GPU-hour vs 1× RTX 3090), và **lý do** (kiến thức visual–ngôn ngữ là kiến thức chung, pretrained đã có sẵn). Nếu cần thích nghi với dữ liệu riêng → fine-tune bằng LoRA (chương 16), rẻ hơn hàng nghìn lần.

</details>

---

## Tài liệu đọc thêm

- **3Blue1Brown — Neural networks (chuỗi 4 video, có phụ đề Việt):** đặc biệt tập 2 (gradient descent) và tập 3 (backpropagation) — trực giác hình ảnh đẹp nhất hiện có. youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
- **Andrej Karpathy — "Neural Networks: Zero to Hero", bài 1 (micrograd):** tự tay viết backprop ~100 dòng Python — sau khi xem, backprop hết là ma thuật. youtube.com/watch?v=VMj-3S1tku0
- **Loshchilov & Hutter (2019), "Decoupled Weight Decay Regularization":** paper khai sinh AdamW — đọc phần intro là đủ hiểu Adam sai chỗ nào với weight decay.
- **Karpathy — "A Recipe for Training Neural Networks" (blog, 2019):** checklist thực chiến chống các lỗi train kinh điển; sẽ rất hữu ích khi team bước sang chương 16-17.
- Nội bộ: `PROJECT-CONTEXT.md` §4 (danh sách pretrained model FUFU dùng) và chương 05, 16, 17, 19 của giáo trình này.
