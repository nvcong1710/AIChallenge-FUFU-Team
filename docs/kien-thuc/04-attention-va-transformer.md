# Chương 04 — Attention & Transformer

> **Phần I — Nền tảng Deep Learning** · Chương quan trọng nhất của Phần I.
> Cần đọc trước: [Chương 01](01-tu-ml-co-dien-sang-neural-network.md),
> [Chương 02](02-huan-luyen-mang-neural.md), [Chương 03](03-cnn-xu-ly-anh.md).

---

## 1. Vì sao chương này tồn tại trong FUFU

Hãy mở `PROJECT-CONTEXT.md` mục §4 (Tech stack) và đếm số model trong hệ thống:

| Model trong FUFU | Vai trò | Kiến trúc bên trong |
|---|---|---|
| SigLIP-2 Large | encode ảnh + text query | **Transformer** (×2: một cho ảnh, một cho text) |
| Qwen2.5-VL-7B | caption frame tiếng Việt | **Transformer** |
| PhoWhisper-medium | nhận dạng lời thoại (ASR) | **Transformer** (encoder-decoder) |
| NLLB-200 | dịch query VI→EN | **Transformer** (encoder-decoder) |
| Qwen2.5-3B | paraphrase query | **Transformer** (decoder-only) |
| BGE-reranker-v2-m3 | cross-encoder rerank | **Transformer** (encoder) |

**Sáu trên sáu.** Model duy nhất không phải transformer là YOLO-World (detection, gốc CNN — và
ngay cả nó cũng nhúng một text encoder transformer bên trong). Nói cách khác: nếu bạn hiểu
một kiến trúc duy nhất là transformer, bạn hiểu được **bộ khung của gần như toàn bộ FUFU**.
Các chương 05–12 sau này chỉ là các biến tấu: transformer ăn text (chương 05), ăn ảnh
(chương 06), ăn cặp ảnh–text (chương 07–08), ăn âm thanh (chương 09)...

Chương này trả lời ba câu hỏi:

1. **Attention là gì** — và vì sao nó chỉ là "trung bình có trọng số" mà bạn đã gặp ở kNN, nhưng trọng số được *học*.
2. **Transformer lắp ráp từ attention như thế nào** — dùng lại đúng các viên gạch LayerNorm, residual, FFN bạn đã học ở chương 02–03.
3. **Hệ quả thực dụng nào đập thẳng vào FUFU** — ví dụ: vì sao query quá dài bị SigLIP cắt cụt ở 64 token.

---

## 2. Cần biết trước

Từ các chương trước, bạn cần nhớ (không cần thuộc công thức, chỉ cần nhớ vai trò):

- **Vector embedding** (chương 01): mỗi từ/đối tượng được biểu diễn bằng một vector số; hai vector "gần nhau" (dot product / cosine lớn) nghĩa là hai thứ giống nhau về nghĩa.
- **Softmax** (chương 01–02): biến một dãy số bất kỳ thành một dãy trọng số dương, tổng bằng 1 — số nào lớn hơn thì chiếm tỷ trọng lớn hơn (theo hàm mũ).
- **Residual connection** (chương 03, ResNet): `output = x + F(x)` — cho gradient "đường tắt" chảy về, giúp xếp chồng rất nhiều tầng mà vẫn huấn luyện được.
- **LayerNorm** (chương 02, phần normalization): chuẩn hoá lại giá trị trong một tầng để huấn luyện ổn định.
- **Ma trận trọng số học được** (chương 01): một phép nhân ma trận `W·x` mà các phần tử của `W` được gradient descent điều chỉnh.

Từ ML cổ điển, bạn cần nhớ **kNN / kernel weighting**: dự đoán cho một điểm = trung bình
có trọng số của các điểm lân cận, hàng xóm càng "gần" thì trọng số càng lớn. Giữ chặt hình
ảnh này — attention chính là nó, phiên bản có học.

---

## 3. Bài toán: nghĩa của một từ phụ thuộc ngữ cảnh

Xét từ **"đá"** trong hai câu:

- "Anh ấy nhặt một **hòn đá**." → đá = khoáng vật, danh từ.
- "Anh ấy **đá bóng** rất giỏi." → đá = động tác chân, động từ.

Cùng một từ, hai nghĩa hoàn toàn khác nhau. Nếu ta biểu diễn mỗi từ bằng **một vector cố
định** (kiểu tra từ điển: "đá" → luôn cùng một vector), model không bao giờ phân biệt được
hai câu trên. Đây chính xác là vấn đề mà FUFU đối mặt hằng ngày: query *"người đá bóng trên
bãi biển"* phải khớp với cảnh thể thao, không phải cảnh ghềnh đá ven biển.

Kết luận: vector của một từ phải được **điều chỉnh theo các từ xung quanh** — "đá" đứng cạnh
"hòn" phải cho ra vector khác với "đá" đứng cạnh "bóng". Ta cần một cơ chế cho mỗi từ
**nhìn sang các từ khác trong câu** và trộn thông tin của chúng vào vector của mình.

So với ML cổ điển: đây là điều mà các model bạn biết đều không làm được một cách tự nhiên.
SVM hay Random Forest nhận một vector feature **cố định** cho mỗi mẫu; nếu bạn đưa câu vào
dưới dạng bag-of-words, thông tin "từ nào đứng cạnh từ nào" đã mất sạch trước khi model kịp nhìn thấy.

### RNN — giải pháp cũ, và vì sao bị thay

Trước 2017, lời giải chuẩn là **RNN (Recurrent Neural Network)**: đọc câu *tuần tự* từ trái
sang phải, duy trì một "bộ nhớ" (hidden state) được cập nhật sau mỗi từ. Ý tưởng đúng, nhưng
hai điểm yếu chí mạng: (1) **tuần tự nên chậm** — phải xử lý xong từ thứ 5 mới đến từ thứ 6,
không tận dụng được GPU vốn mạnh ở tính toán song song; (2) **quên xa** — thông tin từ đầu
câu phải "sống sót" qua hàng chục bước cập nhật mới tới cuối câu, trên đường đi bị pha loãng
dần (cùng họ với vanishing gradient ở chương 02). Transformer (paper *"Attention Is All You
Need"*, 2017) thay RNN bằng cơ chế cho mọi từ nhìn **trực tiếp** mọi từ khác trong một bước —
vừa song song hoá được, vừa không có khái niệm "xa" để mà quên. Ta không đi sâu RNN hơn nữa.

---

## 4. Attention = trung bình có trọng số CÓ HỌC

### 4.1 Khởi đầu từ thứ bạn đã biết: kNN

Nhớ lại kNN regression: muốn dự đoán giá nhà cho căn nhà x, ta tìm k căn hàng xóm gần nhất,
lấy **trung bình giá của chúng** — bản nâng cấp (kernel regression / Nadaraya–Watson) thì lấy
trung bình **có trọng số**, hàng xóm càng giống x thì trọng số càng cao:

```
dự đoán(x) = Σᵢ  wᵢ · giá(hàng_xóm_i)        với wᵢ ~ độ_giống(x, hàng_xóm_i)
```

Attention là đúng công thức này, áp vào câu văn:

```
vector_mới(từ) = Σᵢ  wᵢ · thông_tin(từ_i trong câu)   với wᵢ ~ độ_giống(từ, từ_i)
```

Khác biệt quyết định nằm ở chữ **"độ giống"**:

| | kNN / kernel | Attention |
|---|---|---|
| Độ giống đo bằng | khoảng cách Euclid / kernel **cố định**, bạn chọn trước | dot product giữa các vector **đã qua phép chiếu học được** |
| Học được gì | không học gì (lazy learner) | học *nên so sánh khía cạnh nào* và *nên lấy thông tin gì* từ hàng xóm |
| "Hàng xóm" là | các điểm trong training set | các token khác **trong cùng input** |

Điểm cuối cùng đáng dừng lại một nhịp: với kNN, hàng xóm nằm trong dữ liệu huấn luyện.
Với attention, "hàng xóm" là **các từ khác trong chính câu đang xử lý** — model học cách
tra cứu nội bộ input của nó.

### 4.2 Query / Key / Value — ẩn dụ thư viện

Để "học cách chọn hàng xóm", attention tách mỗi token thành **ba vai trò**, qua ba ma trận
chiếu học được `W_Q`, `W_K`, `W_V`:

- **Query (Q)** — *câu hỏi tôi đang tra*: "tôi là 'đá', tôi cần biết quanh tôi có gì để xác định nghĩa."
- **Key (K)** — *nhãn trên gáy sách*: mỗi token tự quảng cáo "tôi chứa loại thông tin này."
- **Value (V)** — *nội dung ruột sách*: thông tin thực sự được lấy về nếu sách được chọn.

Quy trình tra cứu thư viện: cầm phiếu Query đi dọc kệ, **so phiếu với từng nhãn Key**
(dot product → điểm khớp), **softmax** các điểm khớp thành trọng số, rồi lấy về
**trung bình có trọng số của các Value**. Vì sao phải tách Key khỏi Value? Vì *tiêu chí để
được chọn* và *thứ được mang về* nên khác nhau — gáy sách ghi "Lịch sử VN" (Key) nhưng ruột
sách là 500 trang nội dung (Value). Cho token cũng vậy: từ "bóng" được "đá" chọn vì nó là
tân ngữ chỉ vật thể (Key), còn thứ "đá" cần hút về là sắc thái nghĩa thể-thao (Value).

Cả ba ma trận `W_Q, W_K, W_V` đều được học bằng backprop như mọi trọng số khác (chương 02)
— không có gì mới về cách huấn luyện, chỉ mới ở cách *dùng* kết quả.

### 4.3 Ví dụ số: tính tay attention cho 3 token

Câu: **"nó đá bóng"** — 3 token, mỗi vector 2 chiều. Giả sử sau khi nhân với `W_Q, W_K, W_V`
ta thu được (số được chọn tròn trịa để tính tay; thực tế là số thực bất kỳ):

| Token | Key k | Value v |
|---|---|---|
| nó | [1, 0] | [0, 2] |
| đá | [0, 1] | [4, 0] |
| bóng | [2, 1] | [2, 2] |

Ta tính vector mới cho token **"đá"**, có query **q = [1, 1]**.

**Bước 1 — điểm khớp (dot product q·k):**

```
score(đá → nó)   = 1·1 + 1·0 = 1
score(đá → đá)   = 1·0 + 1·1 = 1
score(đá → bóng) = 1·2 + 1·1 = 3   ← "bóng" khớp mạnh nhất
```

(Thực tế còn chia thêm cho √d — ở đây √2 — để điểm không phình theo số chiều; ta bỏ qua
cho số đẹp, ý nghĩa không đổi.)

**Bước 2 — softmax thành trọng số** (e¹ ≈ 2.72, e³ ≈ 20.09):

```
tổng = 2.72 + 2.72 + 20.09 = 25.53
w(nó)   = 2.72/25.53 ≈ 0.1
w(đá)   = 2.72/25.53 ≈ 0.1
w(bóng) = 20.09/25.53 ≈ 0.8
```

Để ý tính chất của softmax: điểm 3 chỉ gấp 3 lần điểm 1, nhưng qua hàm mũ trọng số gấp ~7.4
lần — softmax **khuếch đại** chênh lệch, làm attention "tập trung" thay vì chia đều.

**Bước 3 — trung bình có trọng số của các Value:**

```
vector_mới(đá) = 0.1·[0, 2] + 0.1·[4, 0] + 0.8·[2, 2]
               = [0, 0.2] + [0.4, 0] + [1.6, 1.6]
               = [2.0, 1.8]
```

**Đọc kết quả:** vector mới của "đá" giờ chứa **80% thông tin của "bóng"**. Nghĩa của "đá"
đã bị ngữ cảnh kéo về phía "động tác với quả bóng". Nếu câu là "hòn đá to", query của "đá"
sẽ khớp mạnh với key của "hòn", và vector mới ngả về nghĩa khoáng-vật. **Cùng một bộ trọng số
W_Q/W_K/W_V, hai input khác nhau cho hai kết quả khác nhau** — đó là toàn bộ phép màu:
trọng số attention không cố định theo *vị trí* mà tính động theo *nội dung*.

Đó là một phép attention hoàn chỉnh. Mọi thứ còn lại của chương chỉ là nhân bản và xếp chồng nó.

---

## 5. Self-attention và multi-head

### 5.1 Self-attention: mọi token nhìn mọi token

Ở ví dụ trên ta mới tính cho "đá". **Self-attention** lặp y hệt quy trình cho cả "nó" và
"bóng" — mỗi token đều mang query của mình đi so với key của *tất cả* token (kể cả chính nó),
và nhận về một vector mới. Ba phép tính này độc lập nhau nên thực hiện **song song bằng một
phép nhân ma trận duy nhất** — đây chính là lý do transformer nhanh hơn RNN trên GPU. Chữ
*"self"* nghĩa là Q, K, V đều sinh ra từ cùng một chuỗi (câu tự nhìn chính nó); biến thể
*cross-attention* — Q từ chuỗi này, K/V từ chuỗi khác — sẽ gặp lại ở mục 7 và chương 08.

Sau một tầng self-attention, mỗi token đã "ngấm" ngữ cảnh một bước. Xếp nhiều tầng, ngữ cảnh
lan xa và trừu tượng dần — tầng đầu học quan hệ ngữ pháp gần, tầng sâu học quan hệ ngữ nghĩa
toàn câu (cùng tinh thần "tầng nông học cạnh, tầng sâu học vật thể" của CNN ở chương 03).

### 5.2 Multi-head: nhiều góc nhìn song song — như nhiều cây trong Random Forest

Một phép attention chỉ học được **một kiểu quan hệ** (một cách chiếu Q/K/V → một tiêu chí
"giống nhau"). Nhưng từ trong câu quan hệ với nhau theo nhiều kiểu cùng lúc: "đá" cần nhìn
"bóng" theo quan hệ *động-từ–tân-ngữ*, nhìn "nó" theo quan hệ *chủ-ngữ*, có head khác lại
chuyên bám quan hệ phủ định, chỉ thời gian...

Giải pháp: chạy **h phép attention song song** (h "head"), mỗi head có bộ `W_Q, W_K, W_V`
riêng nên học một "góc nhìn" riêng, rồi **nối các output lại** và chiếu qua một ma trận tổng
hợp. Liên hệ thẳng với Random Forest: một cây quyết định chỉ bắt được một góc của dữ liệu;
rừng nhiều cây — mỗi cây nhìn một tập feature/mẫu khác nhau — gộp lại mạnh hơn hẳn từng cây.
Multi-head cũng vậy: **mỗi head là một "cây" chuyên bắt một khía cạnh quan hệ giữa các
token**, và phần "gộp phiếu" ở đây là phép nối + chiếu tuyến tính, được học end-to-end thay
vì vote cứng. Số head điển hình: 8–16 (SigLIP Large dùng 16 head mỗi tầng).

### 5.3 Positional encoding: attention bị "mù thứ tự"

Nhìn lại ví dụ mục 4.3: trong toàn bộ phép tính, có chỗ nào dùng đến việc "bóng" đứng *sau*
"đá" không? **Không.** Attention chỉ so nội dung vector với nhau — xáo trộn thứ tự token,
các điểm dot product không đổi, kết quả từng token giữ nguyên. Với attention thuần,
*"chó cắn người"* và *"người cắn chó"* là một (cùng một túi token — đúng điểm yếu
bag-of-words mà ta chê ở mục 3!).

Khắc phục: **cộng thêm vào mỗi vector token một "vector vị trí"** (positional encoding) —
token ở vị trí 1 được cộng một mẫu số đặc trưng cho vị-trí-1, vị trí 2 một mẫu khác, v.v.
Sau bước cộng này, key của "người" đứng đầu câu khác key của "người" đứng cuối câu, và
attention *có thể* học cách phân biệt trật tự khi cần. Cách sinh vector vị trí (sin/cos cố
định, học được, hay "xoay" như RoPE trong Qwen) là chi tiết kỹ thuật — ở mức trực giác chỉ
cần nhớ: **vị trí được tiêm vào nội dung vector, để cơ chế so-nội-dung gián tiếp thấy được
thứ tự.** Với ảnh (chương 06), chính positional encoding cho ViT biết patch nào nằm góc nào.

---

## 6. Transformer block: lắp ráp các viên gạch đã có

Attention mới giải quyết "trộn thông tin giữa các token". Một **transformer block** hoàn
chỉnh ghép thêm các viên gạch bạn đã học ở chương 02–03:

```
input (chuỗi vector token)
   │
   ├─► Multi-head self-attention ─┐
   │                              ▼
   └────────────► (+) cộng residual ──► LayerNorm
                                            │
                      ┌─────────────────────┤
                      ├─► FFN (MLP 2 tầng) ─┐
                      │                     ▼
                      └──► (+) cộng residual ──► LayerNorm ──► output
```

Đọc từng mảnh bằng kiến thức cũ:

- **Attention** = bước *giao tiếp*: các token trao đổi thông tin với nhau (phần mới duy nhất của chương này).
- **FFN** (feed-forward network) = một MLP 2 tầng quen thuộc từ chương 01, áp **riêng rẽ cho từng token** = bước *suy nghĩ một mình*: mỗi token tiêu hoá thông tin vừa thu thập. FFN thường chiếm ~2/3 số tham số của block.
- **Residual** `x + F(x)` = đường cao tốc gradient từ ResNet (chương 03) — nhờ nó mới xếp được hàng chục block mà không vanishing gradient.
- **LayerNorm** = bộ ổn áp từ chương 02, giữ giá trị trong tầm kiểm soát sau mỗi bước.

Nhịp điệu của block: **giao tiếp → ổn định → suy nghĩ → ổn định.** Một transformer hoàn
chỉnh chỉ là **N block giống hệt nhau xếp chồng** (N = "số tầng"): BERT-base 12 block,
SigLIP-2 Large ~24 block phía vision, Qwen2.5-VL-7B 28 block. Toàn bộ sự khác nhau giữa các
model khổng lồ này nằm ở: N bao nhiêu, vector mấy chiều, bao nhiêu head, và dữ liệu huấn
luyện — *bộ khung block thì y hệt nhau*.

> 🔗 **Trong FUFU:** mở `app/common/encoder.py` — class `SiglipEncoder` gọi
> `model.get_text_features(...)` / `get_image_features(...)`. Đằng sau hai lời gọi đó là
> đúng chồng block vừa vẽ: text đi qua ~24 block text-encoder, ảnh qua ~24 block
> vision-encoder, mỗi block đều là attention → residual+LayerNorm → FFN → residual+LayerNorm.
> Tương tự, `app/extractors/asr.py` (PhoWhisper), `app/backend/services/translator.py`
> (NLLB), `app/backend/services/paraphraser.py` (Qwen2.5-3B) và
> `app/backend/services/reranker.py` (BGE) cũng chỉ là các chồng block với N và kích thước khác nhau.

---

## 7. Encoder vs Decoder — hai chế độ đọc

Cùng một block, nhưng có hai cách cho token "nhìn nhau", sinh ra hai họ transformer:

**Encoder — đọc kiểu "soát toàn câu".** Mọi token nhìn mọi token, cả trước lẫn sau —
như bạn đọc xong cả câu rồi mới kết luận nghĩa từng từ. Phù hợp khi cần **hiểu/biểu diễn**
một input đã có đầy đủ: phân loại, tạo embedding để tìm kiếm. Text encoder của SigLIP và
BGE-reranker thuộc họ này.

**Decoder — đọc kiểu "viết tiếp".** Dùng khi **sinh** văn bản từng token một. Lúc sinh từ
thứ 5, từ thứ 6 chưa tồn tại — nên khi *huấn luyện* phải mô phỏng đúng điều kiện đó:
**causal mask** chặn không cho token nhìn về tương lai (kỹ thuật: đặt điểm attention với mọi
token đứng sau thành −∞ trước softmax → trọng số = 0). Mỗi token chỉ thấy quá khứ của nó.
Qwen2.5-3B (paraphrase) và phần sinh chữ của Qwen-VL thuộc họ này.

**Encoder–decoder** ghép cả hai: encoder đọc trọn input, decoder vừa sinh output vừa
*cross-attention* sang kết quả encoder (Q từ decoder, K/V từ encoder — đúng biến thể nhắc ở
mục 5.1). NLLB (đọc trọn câu Việt → sinh câu Anh) và PhoWhisper (đọc trọn audio → sinh
transcript) trong FUFU là encoder–decoder. Chi tiết hai họ này — BERT vs GPT, vì sao decoder-only
thắng thế ở LLM — để dành trọn chương 05.

---

## 8. Cái giá O(n²) — và vì sao FUFU cắt query ở 64 token

Mỗi token phải so query của mình với key của **mọi** token: chuỗi n token → **n × n** phép
so + một bảng trọng số n×n, ở *mỗi head, mỗi block*. Chi phí (tính toán lẫn bộ nhớ) tăng
**bình phương** theo độ dài chuỗi:

| Độ dài chuỗi n | Số cặp phải so (n²) | So với n = 64 |
|---|---|---|
| 64 | 4 096 | 1× |
| 256 | 65 536 | 16× |
| 2 048 | ~4.2 triệu | 1 024× |

Gấp đôi độ dài → gấp **bốn** chi phí. Đây là lý do "context dài" là cuộc đua tốn kém của
LLM, và là lý do các model embedding chọn trần ngắn: text encoder của SigLIP được huấn luyện
với trần **64 token** — caption ảnh hiếm khi dài hơn, và giữ trần thấp giúp huấn luyện trên
hàng tỷ cặp ảnh–text với chi phí chịu được.

> 🔗 **Trong FUFU:** `app/common/encoder.py`, hàm `encode_text()` gọi processor với
> `truncation=True, max_length=64` (xuất hiện ở cả probe khởi tạo lẫn vòng encode chính).
> **Hệ quả thực tế:** mọi query — và mọi biến thể sau dịch/paraphrase trong
> `expand_query()` — dài quá 64 token sẽ bị **cắt cụt phần đuôi trong im lặng**, không báo
> lỗi. Một query mô tả lê thê ("một người đàn ông mặc áo xanh đang đứng cạnh chiếc xe máy
> màu đỏ trước một quán phở có biển hiệu vàng, trời đang mưa, ...") có thể mất sạch các chi
> tiết cuối — phần bị cắt **không tham gia** vào q_vec của kênh dense. Khi viết query hoặc
> prompt paraphrase, hãy dồn chi tiết quan trọng lên đầu; khi debug "sao chi tiết X không ăn
> điểm", hãy nghi ngờ nó đã rơi ngoài cửa sổ 64 token. (Hai kênh BM25 không bị giới hạn này
> — thêm một lý do hybrid đa kênh tồn tại, xem chương 14.)

Ghi chú thêm: O(n²) cũng giải thích một thiết kế ở chương 06 — ảnh 384×384 được chia thành
patch 16×16 (576 token) thay vì coi mỗi pixel là một token (147 456 token → bảng attention
~21,7 tỷ ô, bất khả thi).

---

## 9. Bức tranh lớn: kiến trúc vạn năng

Điều khiến transformer "ăn" được mọi modality: nó **không hề biết input là gì**. Toàn bộ
kiến trúc chỉ yêu cầu một thứ — *một chuỗi vector*. Vậy nên công thức chung của mọi model
hiện đại là:

```
modality bất kỳ ──[tokenizer riêng]──► chuỗi vector ──► CHỒNG TRANSFORMER BLOCK (y hệt nhau)
```

| Modality | Cách biến thành chuỗi token | Model trong FUFU | Chương |
|---|---|---|---|
| Văn bản | tách từ-con (subword) → tra bảng embedding | SigLIP text, NLLB, Qwen, BGE | 05 |
| Ảnh | cắt thành lưới patch 16×16 → chiếu tuyến tính | SigLIP vision, Qwen-VL | 06 |
| Âm thanh | spectrogram → cắt khung thời gian | PhoWhisper | 09 |

Khác hẳn thời CNN-cho-ảnh / RNN-cho-text mỗi modality một kiến trúc riêng, giờ chỉ còn
**một kiến trúc, nhiều tokenizer**. Hệ quả quan trọng nhất cho FUFU: vì ảnh và text cùng đi
qua một loại kiến trúc, ta có thể huấn luyện chúng nói chung một "ngôn ngữ vector" — nền tảng
để query text tìm được frame ảnh (CLIP/SigLIP, chương 07).

Hai chương kế tiếp đi sâu hai nhánh: **chương 05** — biến text thành token và hai họ
BERT/GPT; **chương 06** — biến ảnh thành token (ViT).

---

## 10. Tóm tắt 10 giây

- Nghĩa của từ phụ thuộc ngữ cảnh → cần cơ chế cho mỗi token **nhìn các token khác**; RNN làm được nhưng tuần tự chậm + quên xa.
- **Attention = trung bình có trọng số kiểu kNN/kernel, nhưng trọng số được học** qua ba phép chiếu Query (hỏi gì) / Key (quảng cáo gì) / Value (giao gì): dot product Q·K → softmax → weighted sum của V.
- **Multi-head** = nhiều attention song song, mỗi head một kiểu quan hệ — như nhiều cây trong Random Forest.
- Attention mù thứ tự → **positional encoding** tiêm vị trí vào vector.
- **Transformer block** = attention → residual + LayerNorm → FFN → residual + LayerNorm; model = N block xếp chồng. **Encoder** nhìn cả câu (hiểu), **decoder** + causal mask chỉ nhìn quá khứ (sinh).
- Chi phí **O(n²)** theo độ dài chuỗi → SigLIP cap text 64 token → query FUFU dài quá bị cắt đuôi im lặng.
- Transformer ăn mọi modality miễn biến được input thành chuỗi token — vì thế **cả 6 model chính của FUFU đều là transformer**.

---

## 11. Câu hỏi tự kiểm tra

**1. Attention giống và khác kernel weighting / kNN ở điểm nào?**
<details><summary>Đáp án</summary>

Giống: cả hai đều tính output = trung bình có trọng số, trọng số tỷ lệ với "độ giống".
Khác: (1) kNN dùng độ đo khoảng cách cố định do người chọn, attention *học* độ giống qua
các ma trận chiếu W_Q/W_K (và học cả "lấy gì về" qua W_V); (2) hàng xóm của kNN nằm trong
training set, còn "hàng xóm" của attention là các token khác trong chính input đang xử lý.
</details>

**2. Vì sao phải tách Key riêng khỏi Value, thay vì dùng chung một vector cho cả hai?**
<details><summary>Đáp án</summary>

Vì *tiêu chí để được chọn* và *nội dung được mang về* là hai việc khác nhau (gáy sách ≠ ruột
sách). Token "bóng" được "đá" chọn nhờ đặc điểm "tân ngữ chỉ vật" (Key), nhưng thứ "đá" cần
hút về là sắc thái nghĩa thể-thao (Value). Dùng chung một vector sẽ ép hai vai trò này trùng
nhau, giảm độ linh hoạt của model.
</details>

**3. Tính tay: q = [2, 0]; hai token có key k₁ = [1, 0], k₂ = [0, 1] và value v₁ = [1, 3], v₂ = [5, 1]. Output xấp xỉ bao nhiêu? (bỏ qua chia √d)**
<details><summary>Đáp án</summary>

Score: q·k₁ = 2, q·k₂ = 0. Softmax: e² ≈ 7.39, e⁰ = 1 → w₁ ≈ 7.39/8.39 ≈ 0.88,
w₂ ≈ 0.12. Output = 0.88·[1, 3] + 0.12·[5, 1] = [0.88 + 0.60, 2.64 + 0.12] ≈ **[1.48, 2.76]**
— ngả mạnh về v₁ vì query khớp k₁ hơn hẳn.
</details>

**4. Xáo trộn thứ tự các từ trong câu, output của self-attention thuần (chưa có positional encoding) thay đổi thế nào? Vì sao đó là vấn đề, và cách khắc phục?**
<details><summary>Đáp án</summary>

Mỗi token vẫn nhận đúng vector output như cũ (chỉ đổi chỗ theo token) — attention chỉ so
nội dung, không biết vị trí. Vấn đề: "chó cắn người" và "người cắn chó" trở thành không phân
biệt được, y hệt điểm yếu bag-of-words. Khắc phục: cộng positional encoding (vector đặc trưng
cho từng vị trí) vào vector token trước khi đưa vào attention, để nội dung vector mang theo
thông tin vị trí.
</details>

**5. Multi-head attention tương tự Random Forest ở điểm nào, và "gộp phiếu" của nó khác vote của RF ra sao?**
<details><summary>Đáp án</summary>

Tương tự: nhiều "bộ học" song song, mỗi bộ (head / cây) chuyên bắt một khía cạnh khác nhau
của dữ liệu, gộp lại mạnh hơn từng bộ riêng lẻ. Khác: RF gộp bằng vote/average cứng sau khi
các cây học độc lập; multi-head gộp bằng nối output + một phép chiếu tuyến tính học được, và
toàn bộ các head được huấn luyện *cùng nhau* end-to-end (gradient chảy qua tất cả).
</details>

**6. Causal mask làm gì về mặt kỹ thuật, và vì sao decoder cần nó khi huấn luyện?**
<details><summary>Đáp án</summary>

Đặt điểm attention từ một token tới mọi token đứng *sau* nó thành −∞ trước softmax → trọng
số về 0 → token chỉ trộn thông tin từ quá khứ. Cần vì lúc sinh thật, tương lai chưa tồn tại;
nếu khi huấn luyện cho phép nhìn token kế tiếp (chính là đáp án cần dự đoán), model sẽ "quay
cóp" và học được thói gian lận vô dụng lúc suy luận.
</details>

**7. Query dense của FUFU sau khi dịch + paraphrase dài 90 token. Điều gì xảy ra ở kênh dense, và kênh nào không bị ảnh hưởng?**
<details><summary>Đáp án</summary>

`encode_text()` trong `app/common/encoder.py` cắt cụt còn 64 token đầu (`truncation=True,
max_length=64`) — 26 token cuối bị bỏ im lặng, các chi tiết nằm ở đuôi không tham gia vào
q_vec dense. Hai kênh BM25 (frame_text, asr_text trên SQLite FTS5) không đi qua SigLIP nên
không bị trần này — chúng vẫn thấy đủ các token (đã lọc) của query.
</details>

**8. Vì sao tăng độ dài chuỗi từ 512 lên 2048 token làm chi phí attention tăng ~16 lần chứ không phải 4 lần?**
<details><summary>Đáp án</summary>

Chi phí attention tỷ lệ n² (mỗi token so với mọi token). Độ dài tăng 4 lần → số cặp tăng
4² = 16 lần. Đây là lý do context dài đắt đỏ và các model embedding như SigLIP chọn trần
token thấp.
</details>

---

## 12. Đọc thêm

- **Jay Alammar — *The Illustrated Transformer*** (jalammar.github.io/illustrated-transformer) — bài minh hoạ trực quan kinh điển, khớp đúng trình tự chương này.
- **3Blue1Brown — *Attention in transformers, visually explained*** (YouTube, chuỗi Deep Learning chương 5–6) — hoạt hình hoá Q/K/V và phép weighted sum.
- **Vaswani et al., 2017 — *Attention Is All You Need*** — paper gốc; chỉ cần đọc Hình 1 + mục 3.2, đối chiếu với mục 6 của chương này.
- **Andrej Karpathy — *Let's build GPT from scratch*** (YouTube) — cho ai muốn thấy block transformer hiện hình từng dòng code (xem sau chương 05 sẽ thấm hơn).
- Tiếp theo trong giáo trình: [Chương 05 — Tokenization, BERT vs GPT](05-tokenization-bert-gpt.md) và [Chương 06 — Vision Transformer](06-vision-transformer.md).
