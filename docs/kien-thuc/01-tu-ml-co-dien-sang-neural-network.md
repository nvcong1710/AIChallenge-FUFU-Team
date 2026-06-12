# Chương 01 — Từ ML cổ điển sang Neural Network

## 1. Vì sao chương này tồn tại trong FUFU

FUFU là hệ thống tìm kiếm multimedia tiếng Việt: gõ "người chơi cờ vua" → hệ thống trả về
đúng đoạn video/ảnh/audio khớp nhất. Để làm được điều đó, FUFU dùng **toàn bộ là neural
network**: SigLIP-2 để hiểu ảnh, PhoWhisper để nghe lời thoại, Qwen-VL để mô tả frame,
NLLB để dịch query, BGE-reranker để xếp hạng lại. Không có Random Forest hay SVM nào ở đây cả.

Câu hỏi tự nhiên là: **tại sao?** Team mình đều biết RF/SVM/Logistic Regression rồi — chúng
đâu có tệ. Chương này trả lời câu hỏi đó: neural network là gì (spoiler: nó là logistic
regression xếp chồng lên nhau), và vì sao trên ảnh/âm thanh/văn bản — đúng loại dữ liệu FUFU
xử lý — NN thắng áp đảo còn ML cổ điển gần như bó tay. Hiểu chương này xong, các chương sau
(CNN, Transformer, CLIP...) sẽ chỉ là "biến thể kiến trúc" của cùng một ý tưởng gốc.

## 2. Cần biết trước

- **Linear regression / Logistic regression**: biết công thức `y = w·x + b` và hàm sigmoid.
- **Khái niệm train/test, loss, overfitting** ở mức ML cổ điển.
- Nhân ma trận cơ bản (nhân hàng với cột, cộng lại).
- KHÔNG cần biết gì về deep learning — đó là mục đích của chương này.

## 3. Ôn 30 giây: model = hàm có tham số học từ data

Mọi model ML — từ linear regression đến Qwen-VL 7B — đều là **một hàm số có tham số**:

```
ŷ = f(x; θ)
```

- `x`: input (vector feature, hoặc ảnh, hoặc câu văn).
- `θ`: tham số (parameters) — những con số model "học" được từ dữ liệu.
- `ŷ`: dự đoán.

Với **linear regression**: `f(x) = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`. Tham số là `w` và `b`.
Học = tìm `w, b` sao cho dự đoán gần ground truth nhất (minimize loss).

Với **logistic regression**: y hệt như trên, nhưng bóp output về khoảng (0,1) bằng sigmoid:

```
f(x) = σ(w·x + b)      với σ(z) = 1 / (1 + e^(-z))
```

Hết phần ôn. Điều duy nhất cần nhớ: **học máy = chọn dạng hàm f, rồi chỉnh tham số θ theo
dữ liệu**. Neural network chỉ khác ML cổ điển ở chỗ **dạng hàm f phức tạp hơn và θ nhiều hơn**.
Cách "chỉnh θ" (gradient descent, backprop) để dành chương 02.

## 4. Một neuron = một logistic regression

Đây là cú "à ha" quan trọng nhất chương: **1 neuron trong neural network CHÍNH LÀ 1 logistic
regression** mà bạn đã biết. Không hơn không kém:

```
neuron(x) = activation(w·x + b)
```

Ba thành phần:

| Thành phần | Trong logistic regression | Trong neuron |
|---|---|---|
| **weights** `w` | hệ số hồi quy | y hệt — mỗi input 1 trọng số |
| **bias** `b` | intercept | y hệt — dịch ngưỡng kích hoạt |
| **activation** | sigmoid (bắt buộc, để ra xác suất) | tùy chọn: sigmoid, ReLU, GELU... |

### Ví dụ số — tính tay output 1 neuron 3 input

Cho neuron có:
- weights `w = [0.5, -0.8, 1.2]`
- bias `b = -0.3`
- activation = sigmoid

Input `x = [2.0, -1.0, 0.5]`. Tính từng bước:

```
Bước 1 — tổng có trọng số:
  z = w·x + b
    = (0.5 × 2.0) + (-0.8 × -1.0) + (1.2 × 0.5) + (-0.3)
    = 1.0 + 0.8 + 0.6 - 0.3
    = 2.1

Bước 2 — activation:
  output = σ(2.1) = 1 / (1 + e^(-2.1)) = 1 / (1 + 0.1225) ≈ 0.891
```

Neuron này "kích hoạt mạnh" (0.891, gần 1). Nếu bạn từng train logistic regression bằng
sklearn, bạn vừa tính tay đúng cái mà `model.predict_proba()` làm. Một neuron không có gì
huyền bí — nó là một bộ phân loại tuyến tính bé tí.

## 5. Từ 1 neuron → 1 tầng → MLP nhiều tầng

### 5.1 Một tầng (layer) = nhiều neuron chạy song song

Một neuron cho ra **1 con số**. Muốn ra nhiều con số? Đặt nhiều neuron cạnh nhau, mỗi neuron
có bộ `w, b` riêng, cùng nhìn vào một input:

```
Tầng 4 neuron, input 3 chiều:
  h₁ = σ(w⁽¹⁾·x + b₁)
  h₂ = σ(w⁽²⁾·x + b₂)
  h₃ = σ(w⁽³⁾·x + b₃)
  h₄ = σ(w⁽⁴⁾·x + b₄)
```

Viết gọn bằng ma trận: `h = σ(Wx + b)` với `W` là ma trận 4×3 (4 neuron × 3 input),
`b` là vector 4 chiều. **Một tầng = một phép nhân ma trận + một activation.** Đây là lý do
GPU (vốn sinh ra để nhân ma trận cho đồ họa) trở thành trái tim của deep learning.

### 5.2 MLP: xếp chồng nhiều tầng

MLP (Multi-Layer Perceptron) = lấy output của tầng này làm input của tầng kế:

```
x (input)
  → tầng 1: h¹ = σ(W¹x + b¹)        ← hidden layer 1
  → tầng 2: h² = σ(W²h¹ + b²)       ← hidden layer 2
  → tầng ra: ŷ = W³h² + b³          ← output layer
```

Các tầng giữa gọi là **hidden layer** ("ẩn" vì ta không quy định trước chúng phải tính gì —
chúng tự học ra các đại lượng trung gian hữu ích). Mỗi giá trị `h` trong hidden layer là một
**feature do model tự chế** — ý này sẽ là điểm mấu chốt ở mục 7.

### 5.3 Vì sao xếp chồng tạo được hàm phức tạp (universal approximation — trực giác)

Mỗi neuron sigmoid vẽ được một "bậc thang mềm": output thấp ở một phía, cao ở phía kia,
chuyển tiếp quanh một ngưỡng. Bây giờ:

- **Cộng 2 bậc thang ngược chiều** → được một "cái gò" (cao ở giữa, thấp 2 bên).
- **Cộng nhiều cái gò** ở các vị trí khác nhau, độ cao khác nhau → xấp xỉ được **bất kỳ
  đường cong nào**, giống như xấp xỉ một hàm bằng nhiều cột histogram.

Đó là trực giác của **định lý xấp xỉ phổ quát (universal approximation theorem)**: một MLP
1 hidden layer đủ rộng có thể xấp xỉ mọi hàm liên tục. (Không chứng minh ở đây — và thực tế
người ta không dùng 1 tầng siêu rộng mà dùng **nhiều tầng vừa phải**, vì xếp sâu cho phép
*tái sử dụng* feature trung gian, hiệu quả hơn nhiều về số tham số.)

So với ML cổ điển: đây giống vai trò của **kernel trick trong SVM** (biến không gian để dữ
liệu phân tách được) — nhưng thay vì *chọn sẵn* kernel (RBF, polynomial...), NN **tự học**
phép biến đổi từ dữ liệu.

## 6. Activation: vì sao bắt buộc phải phi tuyến

### 6.1 Nếu bỏ activation, 100 tầng = 1 tầng

Thử xếp 2 tầng **tuyến tính thuần** (không activation), ví dụ 1 chiều cho dễ:

```
tầng 1: h = 2x        (w₁ = 2)
tầng 2: y = 3h        (w₂ = 3)
→ y = 3 × (2x) = 6x   ← vẫn chỉ là MỘT hàm tuyến tính, w = 6
```

Tổng quát: `W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)` — tích hai ma trận vẫn là một ma
trận. Xếp 100 tầng tuyến tính, gộp lại vẫn chỉ là **một** linear regression. Activation
phi tuyến chính là thứ "chặn" việc gộp này, làm cho mỗi tầng thêm vào thực sự tăng sức mạnh
biểu diễn.

### 6.2 Bốn activation hay gặp

| Tên | Công thức | Ví dụ số | Ghi chú |
|---|---|---|---|
| **Sigmoid** | `1/(1+e⁻ᶻ)` | σ(2.1) ≈ 0.891; σ(0) = 0.5 | Output (0,1) — như logistic regression. Nay chủ yếu dùng ở tầng ra. |
| **Tanh** | `(eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ)` | tanh(1) ≈ 0.762; tanh(0) = 0 | Như sigmoid nhưng output (-1,1), cân quanh 0. |
| **ReLU** | `max(0, z)` | ReLU(2.1) = 2.1; ReLU(-0.7) = 0 | Đơn giản, nhanh, mặc định của hidden layer hơn 10 năm qua. |
| **GELU** | ≈ `z·σ(1.702z)` | GELU(1) ≈ 0.841; GELU(-1) ≈ -0.159 | "ReLU mềm" — cho qua một phần giá trị âm nhỏ. Chuẩn trong Transformer. |

Trực giác chọn lựa: ReLU rẻ và hiệu quả; GELU mượt hơn, là mặc định trong các Transformer
hiện đại — tức là trong **tất cả model FUFU dùng** (SigLIP, Qwen, PhoWhisper, NLLB, BGE đều
là Transformer, xem chương 04).

```python
import numpy as np
def relu(z):    return np.maximum(0, z)
def sigmoid(z): return 1 / (1 + np.exp(-z))

x = np.array([2.0, -1.0, 0.5])
w = np.array([0.5, -0.8, 1.2]); b = -0.3
print(sigmoid(w @ x + b))   # 0.8909 — đúng kết quả tính tay ở mục 4
```

## 7. ĐIỂM MẤU CHỐT: feature engineering vs feature learning

Đây là lý do thực sự khiến deep learning thống trị — và khiến FUFU không thể xây bằng RF/SVM.

### 7.1 ML cổ điển: người thiết kế feature

Khi bạn train Random Forest hay SVM, quy trình luôn là:

```
dữ liệu thô → [NGƯỜI nghĩ ra feature] → bảng feature → model học
```

Ví dụ dự đoán giá nhà: bạn *tự nghĩ ra* các cột "diện tích", "số phòng", "khoảng cách đến
trung tâm". Model chỉ học cách **kết hợp** các feature đó. Chất lượng feature do người quyết
định — đây gọi là **feature engineering**, và nó là 80% công sức của ML cổ điển.

### 7.2 Vấn đề: với ảnh/âm thanh/text, KHÔNG AI thiết kế nổi feature tay

Bài toán của FUFU: query "người chơi cờ vua" phải khớp với một frame video. Frame đó, ở dạng
thô, là một lưới **384×384×3 ≈ 442.000 con số** (mỗi số = độ sáng 1 kênh màu của 1 pixel).

Thử làm feature engineer cho bài này xem: feature nào nói lên "có người đang chơi cờ"?
"Pixel (120,85) có màu nâu"? Vô nghĩa — bàn cờ có thể nằm bất kỳ đâu, dưới bất kỳ ánh sáng,
góc quay nào. Suốt thập niên 2000, các nhà nghiên cứu thị giác máy đã thiết kế tay những
feature rất công phu (SIFT, HOG — đếm hướng cạnh, gradient...) rồi nhét vào SVM. Kết quả:
tiến bộ chậm chạp, và đến 2012 bị một neural network (AlexNet) đè bẹp trên ImageNet với cách
biệt chưa từng có. Từ đó không ai quay lại nữa.

### 7.3 Neural network: feature tự học, theo tầng

```
dữ liệu thô → [tầng 1: tự học feature thô] → [tầng 2: feature trung cấp] → ... → output
```

NN nhận **thẳng pixel thô** và tự học chuỗi biến đổi: tầng đầu học phát hiện cạnh/màu, tầng
giữa ghép cạnh thành hình dạng (ô vuông, quân cờ), tầng sâu ghép hình dạng thành khái niệm
("bàn cờ", "người ngồi"). Mỗi hidden layer ở mục 5.2 chính là một "bảng feature" — nhưng do
**gradient descent tự tìm ra**, không phải do người nghĩ. Đây là **feature learning**, và là
toàn bộ phép màu của deep learning: thay vì thuê chuyên gia thiết kế feature, ta đổ dữ liệu
và compute vào để model tự thiết kế — và nó thiết kế giỏi hơn người, miễn đủ dữ liệu.

Trong FUFU, "feature cuối cùng" mà SigLIP học ra cho mỗi frame là một **vector 1152 chiều**
(gọi là embedding) — nén 442.000 pixel thành 1152 con số mang ngữ nghĩa, sao cho ảnh "người
chơi cờ" và câu text "người chơi cờ vua" có vector **gần nhau** (cosine cao). Tìm kiếm của
FUFU về bản chất là so cosine giữa các vector feature-tự-học này (chi tiết ở chương 07 và 13).

> 🔗 **Trong FUFU:** việc "đổ ảnh thô vào NN, lấy ra vector feature" nằm ở
> `app/common/encoder.py` (class `SiglipEncoder` với `encode_images()` / `encode_text()`,
> có L2-normalize để dùng cosine). Model được chọn ở `config/settings.yaml` dòng 8
> (`siglip: google/siglip2-large-patch16-384`). Vector output được ghi vào FAISS index qua
> `app/ingest/storage.py`, và lúc query được so khớp trong
> `app/backend/services/retrieval.py` (kênh dense). Mọi thứ còn lại của hệ thống —
> caption (`app/extractors/caption.py`), ASR (`app/extractors/asr.py`), dịch
> (`app/backend/services/translator.py`), rerank (`app/backend/services/reranker.py`) —
> cũng đều là neural network nhận dữ liệu thô và tự học feature.

### 7.4 So sánh công bằng: khi nào RF/XGBoost vẫn thắng NN

Deep learning KHÔNG thắng mọi nơi. Trên **dữ liệu dạng bảng (tabular)** — đặc biệt khi bảng
nhỏ/vừa (vài nghìn đến vài trăm nghìn dòng) với feature đã có ý nghĩa sẵn (tuổi, thu nhập,
số lần mua...) — gradient boosting (XGBoost/LightGBM) và Random Forest **vẫn thường thắng
hoặc hòa NN**, với chi phí train rẻ hơn hàng trăm lần. Lý do trực giác:

- Feature bảng đã là "feature tốt" do con người chọn → lợi thế feature learning biến mất.
- Cây quyết định xử lý tự nhiên các ngưỡng cứng, feature lệch thang đo, dữ liệu thiếu.
- NN cần nhiều dữ liệu để tự học feature; bảng nhỏ thì không đủ "nguyên liệu".

Quy tắc bỏ túi:

| Dữ liệu | Nên dùng |
|---|---|
| Bảng tabular nhỏ/vừa, feature có nghĩa | RF / XGBoost (đừng cố NN) |
| Ảnh, video, âm thanh, văn bản tự nhiên | Neural network — không có cửa cho ML cổ điển |
| Bảng cực lớn + tương tác phức tạp | Thử cả hai |

FUFU toàn ảnh/audio/text → 100% neural network là lựa chọn đúng, không phải mốt.

## 8. Quy mô: "tham số" là gì, 300M / 7B nghĩa là gì

**Tham số (parameter)** = một con số học được. Mỗi weight `w` và mỗi bias `b` là 1 tham số.
Đếm thử cho 1 tầng: input 3 chiều, 4 neuron → `W` có 4×3 = 12 weight + 4 bias = **16 tham số**.
Một MLP 1152 → 4096 → 1152 (cỡ 1 khối trong Transformer thật) đã có
1152×4096 + 4096 + 4096×1152 + 1152 ≈ **9,4 triệu tham số**. Xếp vài chục khối như vậy là
ra hàng trăm triệu.

Liên hệ ML cổ điển: logistic regression với 50 feature có ~51 tham số. Random Forest 500 cây
có thể có hàng triệu "tham số" (các ngưỡng split) — nhưng chúng được chọn tham lam từng cái,
còn tham số NN được tối ưu **đồng thời toàn bộ** theo gradient (chương 02).

Số tham số quyết định 2 thứ thực dụng: **sức chứa kiến thức** và **chi phí phần cứng**.
Quy đổi bộ nhớ: mỗi tham số fp16 = 2 byte, INT4 ≈ 0,5 byte. Áp vào các model FUFU:

| Model trong FUFU | Số tham số | Bộ nhớ thực tế | Vai trò |
|---|---|---|---|
| SigLIP-2 Large | ~0,3 tỷ (300M) | ~0,4 GB VRAM (fp16) | encode ảnh + text |
| NLLB-200 distilled | 0,6 tỷ | ~1,3 GB | dịch VI→EN |
| Qwen2.5-3B-Instruct | 3 tỷ | ~2,5 GB (INT4) | paraphrase query |
| PhoWhisper-medium | ~0,8 tỷ | ~3 GB | ASR tiếng Việt |
| Qwen2.5-VL-7B | 7 tỷ | ~5 GB (INT4: 7e9 × 0,5 byte + overhead) | caption frame |

Kiểm tra tay một dòng: Qwen-VL 7B ở fp16 sẽ tốn 7×10⁹ × 2 byte = **14 GB** — quá nửa con
RTX 3090 24GB chỉ cho 1 model. Vì vậy FUFU nén INT4 (mỗi tham số ~0,5 byte) còn ~5 GB
(kỹ thuật quantization — xem chương 08). Đây là lý do `config/settings.yaml` có cờ
`caption_quant_4bit: true`, và là ví dụ đầu tiên cho thấy "đếm tham số" không phải lý thuyết
suông mà là kỹ năng quy hoạch VRAM hằng ngày của team.

Một trực giác cuối: nhiều tham số hơn = nhớ được nhiều pattern hơn, nhưng cũng **dễ overfit
hơn nếu thiếu dữ liệu** — đúng trade-off bias/variance bạn đã biết từ ML cổ điển, chỉ khác
là các model FUFU dùng đã được pretrain trên hàng tỷ mẫu nên đứng vững (chương 16 bàn chuyện
fine-tune chúng).

## 9. Forward pass / inference vs training

Hai "chế độ sống" của một neural network:

- **Forward pass (inference / suy luận):** tham số đã cố định; đưa input vào, nhân ma trận +
  activation lần lượt qua các tầng, lấy output. Mục 4 bạn đã làm forward pass tay cho 1 neuron.
  Giống `model.predict()` của sklearn. **Toàn bộ FUFU lúc chạy chỉ làm forward pass** — cả khi
  ingest (encode frame, sinh caption, nhận dạng giọng nói) lẫn khi query (encode text, rerank).
  Không tham số nào thay đổi.

- **Training (huấn luyện):** chiều ngược lại — so output với đáp án, tính loss, rồi **chỉnh
  tham số** để loss giảm (gradient descent + backpropagation). Giống `model.fit()`. Team FUFU
  hiện **không train model nào** — ta dùng model pretrain sẵn. Cơ chế training là nội dung
  chương 02; fine-tune cho domain riêng là chương 16.

Hệ quả thực dụng: inference rẻ hơn training rất nhiều (không cần lưu gradient), nên máy 8GB
VRAM chạy được query pipeline của FUFU (~5GB) dù không bao giờ train nổi các model đó.

## 10. Tóm tắt 10 giây

1. Model ML = hàm có tham số; NN chỉ là hàm phức tạp hơn, nhiều tham số hơn.
2. **1 neuron = 1 logistic regression** (weights + bias + activation).
3. Tầng = nhiều neuron = 1 phép nhân ma trận; MLP = xếp chồng tầng; tầng giữa = hidden layer.
4. **Activation phi tuyến là bắt buộc** — không có nó, 100 tầng gộp lại thành 1 tầng tuyến tính.
5. Khác biệt sống còn: ML cổ điển cần người **thiết kế feature**; NN **tự học feature** từ dữ
   liệu thô → thắng áp đảo trên ảnh/âm thanh/text (địa bàn của FUFU).
6. Tabular nhỏ → XGBoost/RF vẫn là vua. Biết chọn đúng vũ khí.
7. 300M/7B params = số weight+bias; nhân 2 byte (fp16) hoặc 0,5 byte (INT4) ra VRAM.
8. FUFU lúc chạy chỉ làm **forward pass**; training để chương 02.

## 11. Câu hỏi tự kiểm tra

**Câu 1.** Một neuron có `w = [1.0, -2.0]`, `b = 0.5`, activation ReLU. Tính output với
input `x = [3.0, 1.0]`.
<details><summary>Đáp án</summary>
z = 1.0×3.0 + (-2.0)×1.0 + 0.5 = 3 - 2 + 0.5 = 1.5. ReLU(1.5) = <b>1.5</b>.
(Nếu x = [1.0, 3.0] thì z = 1 - 6 + 0.5 = -4.5 → ReLU = 0: neuron "tắt".)
</details>

**Câu 2.** Vì sao một MLP 5 tầng KHÔNG dùng activation lại không mạnh hơn linear regression?
<details><summary>Đáp án</summary>
Hợp của các phép tuyến tính vẫn là tuyến tính: W₅(W₄(...W₁x)) = (W₅W₄...W₁)x — gộp được
thành đúng 1 ma trận. Phi tuyến giữa các tầng là thứ ngăn việc gộp này.
</details>

**Câu 3.** "1 neuron = logistic regression" — đúng hay sai, và khác nhau chỗ nào (nếu có)?
<details><summary>Đáp án</summary>
Về cấu trúc là một: σ(w·x + b). Khác biệt: neuron trong NN có thể dùng activation khác
(ReLU/GELU), và output của nó làm input cho tầng sau thay vì là dự đoán cuối; tham số của
nó được học chung với cả mạng chứ không đứng riêng.
</details>

**Câu 4.** Team được giao bài dự đoán churn từ bảng 20.000 khách hàng × 30 cột (tuổi, gói
cước, số cuộc gọi...). Nên thử NN hay XGBoost trước? Vì sao?
<details><summary>Đáp án</summary>
XGBoost/RF trước. Tabular nhỏ, feature đã có nghĩa do người chọn → lợi thế feature learning
của NN biến mất, còn chi phí thì cao hơn nhiều. NN chỉ đáng thử nếu boosting đã kịch trần.
</details>

**Câu 5.** Vì sao không ai xây được hệ thống như FUFU bằng SVM + feature thiết kế tay?
<details><summary>Đáp án</summary>
Input là pixel thô (~442.000 số/frame 384×384×3) và sóng âm thô — không ai thiết kế nổi
feature tay nói lên "người chơi cờ vua" bất biến theo góc quay/ánh sáng/vị trí. Phải để
model tự học feature theo tầng từ dữ liệu (feature learning); SIFT/HOG + SVM đã thua hướng
này từ 2012.
</details>

**Câu 6.** Qwen2.5-VL-7B cần khoảng bao nhiêu GB để nạp ở fp16? Vì sao FUFU không làm vậy?
<details><summary>Đáp án</summary>
7×10⁹ tham số × 2 byte ≈ 14 GB. Quá nặng khi phải chạy chung SigLIP + OCR + YOLO + PhoWhisper
trên 3090 24GB, nên FUFU bật <code>caption_quant_4bit: true</code> → INT4 ≈ 5 GB (chương 08).
</details>

**Câu 7.** Khi user gõ query vào FUFU, hệ thống đang làm forward pass hay training?
<details><summary>Đáp án</summary>
Chỉ forward pass (inference): encode text qua SigLIP, dịch qua NLLB, rerank qua BGE — tham
số mọi model đều đứng yên. FUFU hiện không train model nào.
</details>

**Câu 8.** Hidden layer "ẩn" theo nghĩa nào, và nó tương ứng với khái niệm gì trong quy
trình ML cổ điển?
<details><summary>Đáp án</summary>
Ẩn = ta không quy định trước nó tính gì; gradient descent tự tìm ra. Nó đóng vai trò của
bước feature engineering trong ML cổ điển — nhưng feature do model tự học thay vì người
thiết kế.
</details>

## 12. Tài liệu đọc thêm

- **3Blue1Brown — "But what is a neural network?"** (YouTube, chương 1 của series Neural
  Networks): trực quan hóa neuron/tầng/weights đẹp nhất hiện có, ~20 phút.
- **Michael Nielsen — *Neural Networks and Deep Learning*, chương 1 & 4**
  (neuralnetworksanddeeplearning.com): chương 4 có demo tương tác về universal approximation.
- **Karpathy — "Neural Networks: Zero to Hero", bài micrograd** (YouTube): tự code 1 neuron
  và MLP từ con số 0 bằng Python thuần — dành cho ai muốn "sờ tận tay".
- **Grinsztajn et al. 2022 — "Why do tree-based models still outperform deep learning on
  tabular data?"** (arXiv:2207.08815): bằng chứng thực nghiệm cho mục 7.4.
- **Goodfellow, Bengio, Courville — *Deep Learning*, chương 6** (deeplearningbook.org):
  tham khảo chuẩn về MLP khi cần độ chặt chẽ toán học.
- Tiếp theo trong giáo trình: **chương 02** (làm sao θ được học — backprop/optimizer),
  **chương 03** (CNN — NN chuyên cho ảnh), **chương 07** (SigLIP — vì sao ảnh và text chung
  một không gian vector).
