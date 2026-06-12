# Chương 05 — Tokenization, BERT vs GPT, và paradigm pretrain–finetune

> **Vị trí trong lộ trình:** Phần I — Nền tảng Deep Learning. Đứng sau chương 04
> (Attention & Transformer), trước chương 06 (ViT). Chương 04 dạy bạn "động cơ"
> transformer hoạt động thế nào; chương này dạy **nhiên liệu đầu vào** (token →
> embedding) và **3 kiểu lắp ráp động cơ** (encoder / decoder / encoder-decoder)
> mà mọi model trong FUFU đều thuộc một trong ba kiểu đó.

---

## 1. Vì sao chương này tồn tại trong FUFU

Mở `config/settings.yaml` hoặc PROJECT-CONTEXT.md, bạn sẽ gặp một loạt tên model:
`google/siglip2-large-patch16-384`, `vinai/PhoWhisper-medium`,
`facebook/nllb-200-distilled-600M`, `Qwen/Qwen2.5-3B-Instruct`,
`BAAI/bge-reranker-v2-m3`...

Tất cả đều là transformer (chương 04), nhưng chúng **không thể thay thế cho nhau**:

- Vì sao **BGE-reranker** chấm điểm rất giỏi nhưng không "viết" được câu nào?
- Vì sao **Qwen2.5-3B** sinh paraphrase trôi chảy nhưng ta không dùng nó để embed?
- Vì sao **NLLB** và **Whisper** lại có cấu trúc "hai nửa" khác hẳn hai loại trên?

Câu trả lời nằm ở chỗ chúng thuộc **3 họ kiến trúc khác nhau** (encoder-only,
decoder-only, encoder-decoder), được **pretrain với mục tiêu khác nhau**, nên giỏi
việc khác nhau. Nắm được "bản đồ nhận diện" này, bạn nhìn tên model bất kỳ trên
HuggingFace là đoán được ngay nó dùng để làm gì — kỹ năng sống còn khi team cần
thử model mới cho cuộc thi.

Ngoài ra, chương này trả lời câu hỏi nền tảng nhất: **máy không đọc được chữ, vậy
câu tiếng Việt "người chơi cờ vua" đi vào model dưới dạng gì?** Hiểu điều này thì
chương 07 (SigLIP encode text), 11 (NLLB dịch), 12 (cross-encoder) mới không bị "ảo".

---

## 2. Cần biết trước

- **Chương 01-02:** vector, ma trận trọng số, khái niệm "tham số học được", train/finetune.
- **Chương 04:** self-attention, khái niệm "mỗi vị trí trong câu nhìn các vị trí khác",
  ví dụ từ **"đá"** đổi nghĩa theo ngữ cảnh. Chương này nối tiếp trực tiếp ví dụ đó.
- Không cần biết code; vài đoạn Python xuất hiện chỉ để minh hoạ, không cần chạy.

---

## 3. Máy không đọc chữ: từ text đến vector

### 3.1 Chuỗi biến đổi tổng quát

Mọi model ngôn ngữ đều xử lý text qua 3 bước, **trước khi** transformer làm việc:

```
"người chơi cờ vua"
   │  (1) tokenizer: cắt thành token
   ▼
["▁người", "▁chơi", "▁cờ", "▁vua"]
   │  (2) tra từ điển token → số nguyên (id)
   ▼
[5012, 8730, 21044, 17263]            ← id chỉ là "số thứ tự trong từ điển"
   │  (3) embedding lookup: id → vector học được
   ▼
4 vector, mỗi vector ~768 chiều        ← đây mới là thứ transformer "ăn"
```

So với ML cổ điển: bước (1)-(2) giống hệt **label encoding** một biến categorical;
bước (3) giống **one-hot rồi nhân với một ma trận trọng số** — chỉ khác là ma trận
đó được **học cùng model** thay vì cố định.

### 3.2 Vì sao cắt theo subword chứ không theo từ hay ký tự?

Hai phương án "ngây thơ" đều dở:

- **Cắt theo từ:** từ điển phình vô hạn ("chơi", "chơi_bời", "ăn_chơi"... mỗi biến thể
  một mục), và gặp từ chưa thấy bao giờ (tên riêng "Sơn Tùng M-TP") là bó tay — lỗi
  *out-of-vocabulary* y như khi RF gặp category mới lúc inference.
- **Cắt theo ký tự:** từ điển nhỏ (vài trăm ký tự) nhưng chuỗi quá dài — câu 10 từ
  thành 50+ token, attention tốn O(n²) (chương 04) nên cực chậm, và mỗi ký tự đơn lẻ
  gần như không mang nghĩa.

Giải pháp trung dung: **subword** — cắt thành "mảnh từ". Nguyên tắc: **mảnh nào xuất
hiện thường xuyên trong dữ liệu huấn luyện thì được giữ nguyên làm 1 token; từ hiếm
bị bẻ thành nhiều mảnh nhỏ hơn**. Ba thuật toán phổ biến (khác nhau ở chi tiết, giống
nhau ở tư tưởng):

| Thuật toán | Tư tưởng | Dùng ở |
|---|---|---|
| **BPE** (Byte-Pair Encoding) | Lặp lại: ghép cặp ký tự/mảnh xuất hiện nhiều nhất thành mảnh mới | GPT, Qwen |
| **WordPiece** | Tương tự BPE nhưng chọn cặp theo likelihood thay vì tần suất thô | BERT |
| **SentencePiece** | Chạy BPE/unigram **thẳng trên chuỗi thô** (không cần tách từ trước, coi dấu cách là ký tự `▁`) | NLLB, SigLIP text, nhiều model đa ngữ |

Bạn không cần nhớ chi tiết từng thuật toán — chỉ cần nhớ: **từ điển token là sản phẩm
thống kê từ dữ liệu pretrain**, nên ngôn ngữ nào nhiều dữ liệu thì được "ưu ái" mảnh dài.

### 3.3 Ví dụ tiếng Việt cụ thể — và cái giá phải trả

Lấy câu query thật trong FUFU: **"người đàn ông đang nướng thịt trên bãi biển"**
(9 từ, 41 ký tự). Đem qua tokenizer của vài model (số liệu ước lượng, đúng cỡ độ lớn):

| Tokenizer | Số token (VI) | Câu EN tương đương "a man grilling meat on the beach" |
|---|---|---|
| GPT-2 (BPE, train chủ yếu tiếng Anh) | ~35-45 token (vỡ vụn từng byte vì dấu tiếng Việt hiếm) | ~8 token |
| Qwen2.5 (BPE đa ngữ) | ~12-16 token | ~8 token |
| NLLB (SentencePiece, 200 ngôn ngữ) | ~10-14 token | ~8-9 token |
| PhoBERT (train riêng tiếng Việt) | ~9-11 token | (không tối ưu cho EN) |

Một token tiếng Việt điển hình với tokenizer đa ngữ trông như:
`["▁người", "▁đàn", "▁ông", "▁đang", "▁nướng", "▁thịt", "▁trên", "▁bãi", "▁biển"]`
— may mắn thì mỗi âm tiết 1 token; xui (từ hiếm như "nướng") thì vỡ thành
`["▁nư", "ớng"]`.

**Hệ quả thực tế cần nhớ:**

1. **Tiếng Việt thường tốn nhiều token hơn tiếng Anh** cho cùng một nội dung
   (thường gấp ~1.3-2× với tokenizer đa ngữ, gấp 4-5× với tokenizer thuần Anh).
   → câu dài dễ chạm giới hạn độ dài chuỗi (max length) của model hơn.
2. Token nhiều hơn = attention tính nhiều hơn = **chậm hơn và tốn VRAM hơn**.
3. Từ bị bẻ vụn quá nhỏ → mỗi mảnh ít nghĩa → model "hiểu" tiếng Việt kém hơn
   tiếng Anh. Đây là một lý do FUFU **dịch query VI→EN** rồi tìm bằng cả hai bản
   (chương 11) thay vì chỉ tin bản tiếng Việt.

> 🔗 **Trong FUFU:** mỗi model tự mang tokenizer riêng của nó. Khi
> `app/backend/services/translator.py` load NLLB hay `app/extractors/asr.py` load
> PhoWhisper qua `AutoProcessor`/`AutoTokenizer`, bước tokenize xảy ra ngầm bên
> trong — code FUFU chỉ đưa string vào. Còn ở kênh BM25 (không phải neural), FUFU
> tự "tokenize" kiểu cổ điển: tách theo khoảng trắng và **giữ nguyên dấu tiếng Việt**
> — xem `tokenize='unicode61 remove_diacritics 0'` trong `app/ingest/storage.py`
> và hàm `_build_fts_or_query` trong `app/backend/services/retrieval.py`. Hai nghĩa
> của chữ "token" này khác nhau — đừng nhầm.

### 3.4 Embedding lookup: bảng tra vector học được

Sau khi có id, model tra **bảng embedding** — một ma trận kích thước
`(số token trong từ điển) × (số chiều)`. Ví dụ từ điển 250.000 token, mỗi vector
768 chiều → bảng có ~192 triệu tham số, **tất cả đều học được qua backprop**
(chương 02) y như mọi trọng số khác.

```python
# Toàn bộ "embedding lookup" chỉ là thế này:
embedding_table = ...   # ma trận [250_000, 768], học được
ids = [5012, 8730, 21044, 17263]
vectors = embedding_table[ids]   # lấy 4 hàng → 4 vector 768 chiều
```

Trực giác: sau pretrain, các token hay xuất hiện trong ngữ cảnh giống nhau sẽ có
vector gần nhau — "chó" gần "mèo", xa "vi phân". Giống như nếu bạn one-hot 250.000
category rồi cho linear layer tự nén xuống 768 chiều: model tự học cách xếp các
category "giống nhau" lại gần nhau.

### 3.5 Static vs contextual embedding — nối lại ví dụ "đá" chương 04

Bảng tra ở 3.4 cho ra **static embedding**: token "đá" → **một** vector cố định,
bất kể câu nào. Đây chính là tinh thần của **word2vec** (2013) — mỗi từ một vector,
tra xong là xong.

Vấn đề: *"đá bóng"*, *"nước đá"*, *"núi đá"* — cùng token "đá", ba nghĩa khác nhau.
Static embedding buộc cả ba nghĩa nhồi chung một vector → vector "đá" thành một
món trộn nhạt nhẽo.

**Contextual embedding** = đầu ra **sau khi** chuỗi vector static đi qua các tầng
self-attention (chương 04). Tại tầng cuối, vector ở vị trí "đá" trong *"đá bóng"*
đã hút thông tin từ "bóng" nên nằm gần vùng *hành-động-thể-thao*; còn trong
*"nước đá"* nó nằm gần vùng *vật-thể-lạnh*. **Cùng một token, vector đầu ra khác
nhau tuỳ câu** — đó là toàn bộ lý do transformer thắng word2vec.

Tóm gọn một dòng: `contextual = transformer(static)`. Static là "nghĩa trong từ
điển"; contextual là "nghĩa trong câu này".

---

## 4. Ba họ kiến trúc: encoder, decoder, encoder-decoder

Transformer gốc (chương 04) có hai nửa: **encoder** (đọc) và **decoder** (viết).
Các model đời sau chỉ lấy một nửa hoặc giữ cả hai — sinh ra 3 họ.

### 4.1 BERT — encoder-only: chuyên gia ĐỌC HIỂU

- **Cấu trúc:** chỉ chồng các tầng encoder. Self-attention **hai chiều** — token ở
  giữa câu nhìn được cả từ đứng trước lẫn đứng sau.
- **Pretrain bằng Masked Language Model (MLM)** — trò "đục lỗ điền từ": che ngẫu
  nhiên ~15% token rồi bắt model đoán lại.

  > *"Tôi uống cà [MASK] mỗi sáng"* → đoán "phê".

  Để đoán đúng, model **buộc phải** dùng ngữ cảnh cả hai phía ("cà" bên trái,
  "mỗi sáng" bên phải). Hàng tỷ câu đục lỗ như vậy ép model học ngữ pháp, ngữ
  nghĩa, kiến thức thường thức — **không cần ai gán nhãn** (nhãn chính là từ bị che,
  có sẵn trong dữ liệu). Đây gọi là *self-supervised learning*.
- **Giỏi:** mọi việc cần *hiểu* trọn câu — phân loại, chấm điểm độ liên quan,
  tạo embedding đại diện câu.
- **Dở:** *sinh* văn bản. BERT nhìn hai chiều nên không có khái niệm "viết tiếp" —
  giống người điền ô chữ rất giỏi nhưng chưa từng tập viết văn.

### 4.2 GPT / Qwen — decoder-only: chuyên gia VIẾT

- **Cấu trúc:** chỉ chồng các tầng decoder, dùng **causal mask** — token ở vị trí t
  **chỉ nhìn được** các vị trí 1..t (quá khứ), bị che hoàn toàn tương lai.
- **Pretrain bằng next-token prediction:** cho nửa câu, đoán token kế tiếp.

  > *"Hôm nay trời"* → đoán "mưa"/"nắng"/"đẹp"...
  > rồi *"Hôm nay trời mưa"* → đoán tiếp "to"/"nên"...

  Mỗi vị trí trong câu là một mẫu huấn luyện → tận dụng dữ liệu cực hiệu quả.
- **Giỏi:** sinh văn bản — vì sinh văn bản *chính là* lặp đi lặp lại "đoán token
  kế tiếp rồi nối vào". Paraphrase, dịch kiểu chat, viết caption đều là sinh.
- **Dở hơn encoder** (ở cùng cỡ) cho việc embed/chấm điểm cả câu: token đầu câu
  không bao giờ "thấy" token cuối câu, nên biểu diễn hai chiều kém tự nhiên hơn.

Liên hệ ML cổ điển: encoder giống **mô hình phân loại/hồi quy** (nhìn toàn bộ
feature vector → ra một phán đoán); decoder giống **mô hình autoregressive trong
chuỗi thời gian** (chỉ dùng quá khứ dự đoán bước kế).

### 4.3 Encoder-decoder: ĐỌC bằng một nửa, VIẾT bằng nửa kia

NLLB (dịch) và Whisper/PhoWhisper (ASR) giữ nguyên cả hai nửa:

```
NLLB:    "người chơi cờ vua" ─► ENCODER đọc trọn câu VI (2 chiều)
                                    │ cross-attention
         "a person playing chess" ◄─ DECODER viết từng token EN (1 chiều)

Whisper: dải âm thanh (spectrogram) ─► ENCODER đọc trọn đoạn audio
         "xin chào các bạn"         ◄─ DECODER viết transcript từng token
```

Hợp lý vì bài toán có **input trọn vẹn cần hiểu kỹ** (câu nguồn / đoạn audio) và
**output là chuỗi mới phải sinh dần** (câu đích / transcript). Encoder được nhìn
hai chiều thoải mái vì input đã có đủ từ đầu; decoder sinh tuần tự vì output chưa
tồn tại.

### 4.4 Bản đồ nhận diện: kiến trúc → model trong FUFU

Đây là bảng đáng nhớ nhất chương — dán nó vào đầu khi đọc tên model bất kỳ:

| Họ | Pretrain bằng | Giỏi | Model trong FUFU | File |
|---|---|---|---|---|
| **Encoder-only** | Masked LM (đục lỗ) | Hiểu / embed / chấm điểm | `BAAI/bge-reranker-v2-m3` (chấm độ liên quan query–passage); **text encoder của SigLIP** (embed query) | `app/backend/services/reranker.py`; `app/common/encoder.py` |
| **Decoder-only** | Next-token | Sinh văn bản | `Qwen/Qwen2.5-3B-Instruct` (paraphrase); `Qwen/Qwen2.5-VL-7B` (caption — phần ngôn ngữ) | `app/backend/services/paraphraser.py`; `app/extractors/caption.py` |
| **Encoder-decoder** | Dịch / transcribe có giám sát (+ denoising) | Biến chuỗi này thành chuỗi khác | `facebook/nllb-200-distilled-600M` (VI→EN); `vinai/PhoWhisper-medium` (audio→text) | `app/backend/services/translator.py`; `app/extractors/asr.py` |

Mẹo nhận diện nhanh khi gặp model lạ trên HuggingFace:

- Tên/mô tả có "embedding", "retrieval", "reranker", "BERT", "RoBERTa" → **encoder**.
- Có "Instruct", "Chat", "GPT", "LLaMA", "Qwen", "-7B/-70B" → **decoder** (LLM sinh).
- Là model dịch (NLLB, mBART, T5, MarianMT) hoặc ASR (Whisper) → **encoder-decoder**.

> 🔗 **Trong FUFU:** một query "người chơi cờ vua" đi qua đủ cả 3 họ trong **một**
> lần search: decoder (Qwen 3B) sinh 3 paraphrase, encoder-decoder (NLLB) dịch sang
> EN, encoder (SigLIP text tower) embed toàn bộ biến thể thành `q_vec`, và cuối
> pipeline encoder (BGE-reranker) chấm lại top-50. Xem orchestrator
> `app/backend/services/search_engine.py` (hàm `expand_query` và `search`) — đọc
> hàm đó sau chương này bạn sẽ gọi tên được vai trò của từng model.

### 4.5 Một vector đại diện cả câu: [CLS] và mean pooling

Encoder nhả ra **một vector cho mỗi token** — câu 10 token thì 10 vector. Nhưng
retrieval cần **một vector cho cả câu** để so cosine. Hai cách rút gọn phổ biến:

1. **Token [CLS]:** BERT chèn token đặc biệt `[CLS]` vào đầu mỗi câu khi pretrain.
   Qua các tầng attention, vị trí này hút thông tin từ mọi token khác → lấy vector
   tại `[CLS]` làm đại diện câu.
2. **Mean pooling:** lấy trung bình cộng tất cả vector token. Đơn giản, và với các
   model embedding hiện đại thường tốt ngang hoặc hơn `[CLS]`.

Giống feature engineering cổ điển: có 10 quan sát, cần 1 hàng feature → bạn lấy
mean/max. Ở đây "quan sát" là vector token.

Chi tiết này là cây cầu sang hai chương sau: **chương 07** — SigLIP text encoder
rút 1 vector/câu để so với vector ảnh; **chương 12** — bi-encoder rút 1 vector/câu
cho query và passage riêng rẽ, còn cross-encoder thì khỏi pooling kiểu này vì nó
nhét cả cặp câu vào đọc chung rồi nhả thẳng 1 điểm số.

---

## 5. Paradigm pretrain → finetune

### 5.1 Hai giai đoạn, hai túi tiền

| | Pretrain | Finetune |
|---|---|---|
| **Dữ liệu** | "Núi" text/ảnh/audio thô từ internet (hàng trăm tỷ token), không cần nhãn người gán | Nghìn → triệu mẫu **có nhãn, đúng bài toán của bạn** |
| **Chi phí** | Hàng trăm GPU × tuần/tháng — **triệu đô** | 1 GPU × giờ/ngày — vài đô tới vài trăm đô |
| **Ai làm** | Google, Meta, Alibaba, VinAI... | **Mình** (hoặc cộng đồng) |
| **Model học được gì** | Ngôn ngữ, kiến thức tổng quát, "thị giác" tổng quát | Thích nghi với domain/nhiệm vụ hẹp |

Tư tưởng: kiến thức tổng quát (ngữ pháp tiếng Việt, "mèo trông như thế nào") chỉ
cần học **một lần** trên dữ liệu khổng lồ; phần việc của ta là **kế thừa** rồi chỉnh
nhẹ. Trước kỷ nguyên này, mỗi bài toán NLP phải train từ đầu với dữ liệu gán nhãn
riêng — đắt và yếu.

Liên hệ ML cổ điển dễ nhớ: giống như bạn nhận một **Random Forest đã train sẵn**
trên 100 triệu mẫu từ nơi khác, về chỉ việc **calibrate ngưỡng quyết định** trên
vài nghìn mẫu của mình thay vì trồng lại rừng từ đầu. (Khác biệt: với NN, "chỉnh"
có thể đụng vào chính trọng số model — mức độ đụng nông hay sâu là chuyện của
chương 16 về LoRA/PEFT.)

### 5.2 Ba mức "kế thừa", từ rẻ đến đắt

1. **Zero-shot:** dùng nguyên xi, không train gì. FUFU dùng SigLIP, NLLB, Qwen,
   BGE-reranker đều ở chế độ này — tải về là chạy.
2. **Finetune:** train tiếp trên dữ liệu của mình (toàn bộ hoặc một phần trọng số —
   chương 16).
3. **Pretrain từ đầu:** gần như không bao giờ là việc của team thi đấu.

### 5.3 Ví dụ sống ngay trong FUFU: PhoWhisper

- OpenAI **pretrain Whisper** trên 680.000 giờ audio đa ngôn ngữ — encoder-decoder,
  nghe spectrogram → viết transcript. Tiếng Việt chỉ chiếm phần nhỏ → Whisper gốc
  nghe tiếng Việt ở mức "tàm tạm", hay sai dấu, sai tên riêng.
- **VinAI finetune** Whisper trên ~844 giờ audio **tiếng Việt** đủ giọng vùng miền
  → ra `vinai/PhoWhisper-medium`. Kiến trúc y hệt, tokenizer y hệt — chỉ trọng số
  được tinh chỉnh — mà WER tiếng Việt giảm rõ rệt.
- FUFU hưởng trọn thành quả: chỉ ghi một dòng `asr_model: vinai/PhoWhisper-medium`
  trong `config/settings.yaml`. Không tốn phút GPU training nào.

Đó là paradigm vận hành đúng nghĩa: OpenAI trả tiền pretrain, VinAI trả tiền
finetune, FUFU dùng zero-shot kết quả.

---

## 6. Hệ sinh thái HuggingFace — đủ để đọc code FUFU

HuggingFace (HF) là "GitHub của model": kho chứa hàng triệu model pretrain sẵn +
thư viện `transformers` để load chúng thống nhất một kiểu.

**Model id** = `tổ-chức/tên-model`, ví dụ `vinai/PhoWhisper-medium`,
`google/siglip2-large-patch16-384`. Mỗi id là một trang trên huggingface.co chứa
trọng số + config kiến trúc + tokenizer/processor đi kèm.

**Auto-class** — mẫu code bạn sẽ gặp khắp `app/`:

```python
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("google/siglip2-large-patch16-384")
processor = AutoProcessor.from_pretrained("google/siglip2-large-patch16-384")
```

- `AutoModel...` đọc file config của model id rồi **tự chọn đúng class kiến trúc**
  và nạp trọng số — bạn không cần biết trước nó là BERT hay SigLIP.
- `AutoProcessor` / `AutoTokenizer` tải đúng bộ tiền xử lý **đi cặp với model đó**
  (tokenizer cho text, resize/normalize cho ảnh, spectrogram cho audio). **Luật
  bất di bất dịch: model nào dùng tokenizer/processor của model đó** — embedding
  lookup (mục 3.4) tra theo id, mà id chỉ có nghĩa trong từ điển của chính nó.
  Tokenizer A + model B = kết quả rác không báo lỗi.

**Cache:** lần đầu `from_pretrained`, HF tải trọng số về
`~/.cache/huggingface/hub/` (Windows: `C:\Users\<user>\.cache\huggingface\`) —
các lần sau load thẳng từ đĩa, không cần mạng. Vì vậy FUFU có
`scripts/download_models.py` để "mồi cache" một thể (~25GB) trước ngày thi, tránh
cảnh đứng giữa phòng thi chờ tải model.

> 🔗 **Trong FUFU:** mở `app/common/encoder.py` — `SiglipEncoder` load model +
> processor đúng mẫu trên; `app/extractors/asr.py`, `caption.py`,
> `app/backend/services/translator.py`, `paraphraser.py`, `reranker.py` cũng cùng
> một khuôn. Sau chương này, đọc các file đó bạn chỉ còn phải hiểu phần "dùng model
> để làm gì", không vướng phần "load model kiểu gì".

---

## 7. Tóm tắt 10 giây

1. Text → **token** (mảnh subword, BPE/WordPiece/SentencePiece) → **id** →
   **embedding vector**; tiếng Việt thường tốn nhiều token hơn tiếng Anh.
2. Static embedding (word2vec): 1 từ = 1 vector chết. Contextual (transformer):
   vector đổi theo câu — "đá bóng" ≠ "nước đá".
3. **Encoder (BERT)** đọc 2 chiều, pretrain đục-lỗ → giỏi hiểu/embed/chấm điểm.
   **Decoder (GPT/Qwen)** chỉ nhìn quá khứ, pretrain next-token → giỏi sinh.
   **Encoder-decoder (NLLB/Whisper)** đọc một nửa, viết nửa kia → dịch/transcribe.
4. FUFU: BGE-reranker & SigLIP-text = encoder; Qwen = decoder; NLLB & PhoWhisper
   = encoder-decoder.
5. Pretrain (triệu đô, người khác lo) → mình finetune hoặc zero-shot. PhoWhisper
   = Whisper finetune tiếng Việt. HuggingFace = nơi nhận hàng:
   `AutoModel.from_pretrained("tổ-chức/tên-model")`, cache ở `~/.cache/huggingface`.

---

## 8. Câu hỏi tự kiểm tra

**Câu 1.** Vì sao không tokenize theo nguyên từ ("nướng" = 1 mục từ điển cố định)
mà phải dùng subword?

<details><summary>Đáp án</summary>

Từ điển theo nguyên từ sẽ phình rất lớn và vẫn bất lực trước từ chưa từng thấy
(tên riêng, từ mới, lỗi chính tả) — lỗi out-of-vocabulary, giống RF gặp category
lạ lúc inference. Subword giữ từ phổ biến nguyên vẹn, còn từ hiếm bị bẻ thành mảnh
nhỏ đã có trong từ điển → **mọi chuỗi đều tokenize được**, từ điển kích thước
kiểm soát được (vài chục đến vài trăm nghìn mục).
</details>

**Câu 2.** Cùng câu "người đàn ông nướng thịt", tokenizer GPT-2 (thuần Anh) cho
~40 token còn NLLB cho ~12. Giải thích và nêu một hệ quả thực tế.

<details><summary>Đáp án</summary>

Từ điển token được xây bằng thống kê trên dữ liệu pretrain. GPT-2 hầu như không
thấy tiếng Việt nên các mảnh có dấu phải vỡ xuống mức byte; NLLB train trên 200
ngôn ngữ nên có sẵn mảnh tiếng Việt dài. Hệ quả: với tokenizer nghèo tiếng Việt,
câu dài hơn → chậm hơn, tốn VRAM hơn (attention O(n²)), dễ chạm max length, và
model hiểu kém hơn vì mỗi mảnh gần như vô nghĩa — một lý do FUFU dịch query sang
EN để tìm song song.
</details>

**Câu 3.** "Sân khấu được lát đá" và "cầu thủ đá phạt" — static embedding và
contextual embedding xử lý từ "đá" khác nhau thế nào?

<details><summary>Đáp án</summary>

Static (word2vec): "đá" tra ra **một vector duy nhất** cho cả hai câu — trộn lẫn
nghĩa vật liệu và nghĩa hành động. Contextual: vector ban đầu giống nhau, nhưng
sau các tầng self-attention, "đá" trong câu 1 hút thông tin từ "lát"/"sân khấu"
→ vector nghiêng về vật liệu; trong câu 2 hút từ "cầu thủ"/"phạt" → nghiêng về
hành động. Cùng token, **hai vector đầu ra khác nhau**.
</details>

**Câu 4.** Vì sao BERT giỏi chấm điểm độ liên quan nhưng không sinh được văn bản,
còn GPT thì ngược lại?

<details><summary>Đáp án</summary>

BERT pretrain bằng masked LM với attention 2 chiều — nó học cách **hiểu** câu trọn
vẹn nhưng không học quy trình "viết tiếp từ trái sang phải". GPT dùng causal mask,
pretrain next-token — sinh văn bản chính là lặp lại đúng việc nó được luyện; đổi
lại mỗi token chỉ thấy quá khứ nên biểu diễn toàn câu kém tự nhiên hơn encoder
cùng cỡ. Kiến trúc + mục tiêu pretrain quyết định sở trường.
</details>

**Câu 5.** Whisper/PhoWhisper là encoder-decoder. Nửa nào "nghe", nửa nào "viết",
và vì sao bài ASR hợp với kiến trúc này?

<details><summary>Đáp án</summary>

Encoder "nghe": đọc toàn bộ spectrogram audio (input có sẵn trọn vẹn → nhìn 2 chiều
thoải mái). Decoder "viết": sinh transcript từng token, vừa nhìn output đã sinh vừa
cross-attention sang encoder. Hợp vì ASR có input trọn vẹn cần hiểu kỹ + output là
chuỗi text mới phải sinh dần — đúng khuôn "đọc chuỗi này, viết chuỗi khác" như dịch
máy (NLLB).
</details>

**Câu 6.** Xếp các model FUFU sau vào đúng họ kiến trúc: BGE-reranker-v2-m3,
Qwen2.5-3B-Instruct, NLLB-200, PhoWhisper-medium, text encoder của SigLIP.

<details><summary>Đáp án</summary>

- Encoder-only: **BGE-reranker** (chấm điểm cặp query–passage), **SigLIP text
  encoder** (embed câu thành 1 vector).
- Decoder-only: **Qwen2.5-3B** (sinh paraphrase).
- Encoder-decoder: **NLLB-200** (dịch VI→EN), **PhoWhisper** (audio→text).
</details>

**Câu 7.** Phép so sánh "pretrain–finetune giống dùng RF train sẵn rồi calibrate
threshold" đúng ở đâu và khập khiễng ở đâu?

<details><summary>Đáp án</summary>

Đúng: phần đắt nhất (học pattern tổng quát từ dữ liệu khổng lồ) đã có người làm;
mình chỉ chỉnh phần nhỏ, rẻ, trên dữ liệu của mình; tri thức cũ được kế thừa thay
vì học lại. Khập khiễng: calibrate threshold **không đụng vào model** (chỉ chỉnh
ngưỡng đầu ra), còn finetune **cập nhật chính trọng số** bằng gradient descent —
model thực sự thay đổi bên trong. Mức "đụng nông hay sâu" (full FT vs LoRA) là nội
dung chương 16.
</details>

**Câu 8.** Đồng đội load model bằng `AutoModel.from_pretrained("BAAI/bge-reranker-v2-m3")`
nhưng tokenize bằng tokenizer của Qwen "cho tiện vì đã load sẵn". Sai ở đâu?

<details><summary>Đáp án</summary>

Token id chỉ có nghĩa trong **từ điển của chính tokenizer đó**. Id 5012 của Qwen
và id 5012 của BGE trỏ tới hai token khác hẳn nhau, nên embedding lookup của BGE
sẽ tra ra vector của token sai bét — model chạy không lỗi nhưng output là rác.
Luôn lấy tokenizer/processor theo **đúng model id** (`AutoTokenizer.from_pretrained`
cùng id với model).
</details>

---

## 9. Đọc thêm

- **Jay Alammar — *The Illustrated BERT, ELMo, and co.*** và ***The Illustrated GPT-2*** — hai bài blog hình hoá đẹp nhất về encoder vs decoder.
- **Hugging Face NLP Course, chương 1-2 & 6** (huggingface.co/learn) — pipeline, Auto-class, và một chương riêng rất hay về tokenizer (BPE/WordPiece/SentencePiece, tự train tokenizer).
- **Tiktokenizer (tiktokenizer.vercel.app)** — dán câu tiếng Việt vào, xem trực tiếp nó vỡ thành token nào với từng model. 5 phút nghịch đáng giá hơn 1 giờ đọc.
- **Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers* (2018)** — đọc mục 3 (pretrain MLM) là đủ.
- **VinAI — *PhoWhisper*** (github.com/VinAIResearch/PhoWhisper) — đọc README để thấy một dự án finetune tiếng Việt thực tế trông như thế nào.
- **Tiếp theo trong giáo trình:** chương 06 (ViT — đưa *ảnh* vào transformer bằng cách "tokenize" thành patch), rồi chương 07 (SigLIP — ghép text encoder của chương này với image encoder của chương 06).
