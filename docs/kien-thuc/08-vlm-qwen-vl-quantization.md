# Chương 08 — VLM: Qwen-VL và quantization

> **Một câu tóm tắt:** VLM = ghép "con mắt" ViT (chương 06) vào "cái miệng" LLM decoder
> (chương 05) qua một lớp phiên dịch gọi là projector — model giờ **nhìn ảnh và viết ra
> cả câu văn**; và nhờ quantization INT4, con quái vật 28GB này chạy lọt 5GB VRAM trong FUFU.

---

## 1. Vì sao chương này tồn tại trong FUFU

Chương 07 cho FUFU khả năng **đo độ giống** giữa câu query và keyframe: SigLIP nhận một
cặp (text, ảnh) và trả về **một con số** cosine. Rất nhanh, rất hợp để quét hàng triệu
frame. Nhưng nó có một giới hạn cố hữu: nó chỉ *so sánh*, không *kể lại*. Nếu query của
operator dùng từ ngữ lệch hẳn khỏi phân bố huấn luyện ("cô gái áo dài đứng cạnh xích lô"),
hoặc chi tiết quan trọng nằm ở chữ trên băng-rôn, dense channel có thể trượt.

FUFU vá lỗ hổng đó bằng một kênh thứ hai: lúc ingest, mỗi keyframe được đưa qua
**Qwen2.5-VL-7B** — một Vision-Language Model — để **sinh một caption tiếng Việt**
("Một người phụ nữ mặc áo dài đỏ đứng cạnh xích lô trước chợ Bến Thành..."). Caption này
được nhét vào bảng FTS5 `frame_text`, để query tiếng Việt có thể match **chữ-với-chữ**
qua BM25 — không cần đi qua không gian embedding nào cả.

Cái giá phải trả có hai mặt, và cả hai đều là nội dung chương này:

1. **VRAM:** model 7 tỷ tham số, nguyên bản chiếm 28GB — nhiều hơn cả con RTX 3090 của
   team. Lời giải: **quantization** (trọng tâm chương).
2. **Thời gian:** ~1,5 giây/frame — caption là **bottleneck của toàn bộ ingest**
   (PROJECT-CONTEXT §7.3). Lời giải: hiểu rõ trade-off để biết khi nào bật, khi nào tắt.

> 🔗 **Trong FUFU:** toàn bộ logic caption nằm trong `app/extractors/caption.py`
> (class `CaptionExtractor`). Tham số điều khiển ở `config/settings.yaml` dòng 16, 23-26:
> `enable_caption: true`, `caption_model: Qwen/Qwen2.5-VL-7B-Instruct`,
> `caption_max_tokens: 96`, `caption_quant_4bit: true`, và `caption_prompt` (prompt
> tiếng Việt thật mà mọi frame đều đi qua).

---

## 2. Cần biết trước

- **Chương 04:** decoder Transformer, cơ chế sinh token tự hồi quy (autoregressive).
- **Chương 05:** token, embedding, chat template ở mức "LLM ăn chuỗi token và đoán token
  tiếp theo". Chương này tái dụng decoder y nguyên — chỉ đổi món khai vị.
- **Chương 06:** ViT — vision encoder của Qwen-VL chính là một ViT. Không dạy lại.
- **Chương 07:** CLIP/SigLIP — để thấy rõ VLM *khác* gì, mục 3 ngay dưới đây.

---

## 3. VLM khác CLIP thế nào: máy đo vs máy kể

Đây là phân biệt quan trọng nhất chương, vì hai họ model này hay bị gọi lẫn lộn là
"model hiểu ảnh":

| | CLIP/SigLIP (chương 07) | VLM (chương này) |
|---|---|---|
| Đầu vào | 1 ảnh + 1 câu | 1 ảnh + 1 prompt |
| Đầu ra | **1 con số** (độ giống) | **cả câu văn** (sinh từng token) |
| Câu hỏi trả lời được | "Ảnh này giống câu kia bao nhiêu?" | "Trong ảnh có gì? Chữ gì? Ai làm gì?" |
| Tốc độ | ~mili-giây/ảnh | ~giây/ảnh (chậm hơn ~1000×) |
| Dùng để | quét/lọc hàng triệu ứng viên | đọc kỹ từng ảnh một |

**Phép so sánh ML cổ điển:** SigLIP giống một hàm **scoring/similarity** (như kernel
trong SVM — chỉ trả về độ gần); VLM giống một model **sinh chuỗi có điều kiện**
(conditional sequence generation). Một bên là *discriminative về mặt sử dụng*, một bên
là *generative*. Không bên nào "xịn hơn" — chúng trả lời hai câu hỏi khác nhau.

FUFU dùng **cả hai, đúng việc của từng đứa**: SigLIP quét nhanh toàn corpus (dense
channel), Qwen-VL đọc chậm-mà-kỹ từng keyframe lúc ingest để lại "lời khai bằng văn bản"
cho BM25 dùng về sau.

---

## 4. Kiến trúc VLM: mắt → phiên dịch → miệng

Mọi VLM hiện đại (LLaVA, Qwen-VL, InternVL...) đều theo một công thức ba khối:

```
ảnh ──> [Vision Encoder]  ──> [Projector] ──> [LLM Decoder] ──> "Một người phụ nữ
         ViT, chương 06        1-2 lớp MLP     chương 04+05        mặc áo dài đỏ..."
         ảnh → ~10²-10³        "dịch" vector    sinh token
         vector patch          ảnh sang chiều   tự hồi quy
                               embedding LLM
```

1. **Vision encoder** — một ViT. Ảnh thành chuỗi vài trăm vector patch, mỗi vector tóm
   tắt một vùng ảnh. Đến đây không có gì mới so với chương 06.
2. **Projector** — phần *duy nhất* thực sự mới, và lại là phần đơn giản nhất: thường chỉ
   1-2 lớp linear/MLP. Vấn đề nó giải: vector patch của ViT sống ở không gian riêng
   (ví dụ 1024 chiều, mang "ngữ pháp thị giác"), còn LLM chỉ hiểu vector trong không gian
   embedding token của nó (ví dụ 3584 chiều với Qwen-7B). Projector là **người phiên
   dịch**: chiếu mỗi vector patch thành một vector "giả-token" mà LLM đọc được như thể
   đó là một từ.
3. **LLM decoder** — sinh văn bản y hệt chương 05, không biết và không cần biết là một
   phần input của nó vốn là ảnh.

**Trực giác đáng nhớ nhất:** với LLM, tấm ảnh trở thành **đoạn mở đầu của prompt**.
Chuỗi đầu vào thực tế trông như:

```
[img₁][img₂]...[img₅₇₆] Mô tả ngắn gọn (1-2 câu) nội dung chính của ảnh bằng tiếng Việt: ...
└── vài trăm "từ" ảnh ──┘└──────────── prompt text của FUFU ────────────┘
```

LLM đọc từ trái sang phải, attention cho phép mỗi token text "nhìn lại" các giả-token
ảnh — và nó hoàn thành đoạn văn như hoàn thành mọi prompt khác. Không có phép màu
multimodal nào ngoài việc *nối hai chuỗi vector lại với nhau*.

**Liên hệ ML cổ điển:** kiến trúc này là **feature extractor + classifier** quen thuộc,
phóng to: ViT trích đặc trưng (như PCA/HOG ngày xưa), projector là phép chiếu tuyến tính
đổi hệ tọa độ, LLM là "bộ phân loại" — chỉ khác là thay vì chọn 1 nhãn, nó chọn lần lượt
từng token trong từ điển ~150k từ, lặp lại cho đến hết câu.

---

## 5. Qwen2.5-VL-7B: con mắt biết viết tiếng Việt của FUFU

Qwen2.5-VL-7B-Instruct (Alibaba) là VLM mà FUFU chọn, vì tổ hợp khả năng hiếm:

- **Caption & VQA:** mô tả ảnh, trả lời câu hỏi về ảnh — **tốt cả tiếng Việt** (huấn
  luyện đa ngữ mạnh, hiếm trong các VLM 7B).
- **OCR-trong-ảnh:** đọc được chữ trên biển hiệu, banner — bổ trợ cho EasyOCR.
- **Grounding:** chỉ ra *vùng nào* trong ảnh chứa đối tượng (FUFU chưa khai thác,
  nhưng là tài nguyên cho roadmap).

FUFU chỉ dùng một việc: **sinh caption tiếng Việt per-frame**. Prompt thật (từ
`settings.yaml`, cũng là `DEFAULT_PROMPT` trong `caption.py`):

> *"Mô tả ngắn gọn (1-2 câu) nội dung chính của ảnh bằng tiếng Việt: đối tượng nổi bật,
> hành động, bối cảnh, văn bản trên màn nếu có."*

Mổ xẻ từng lựa chọn thiết kế trong prompt này:

| Mảnh prompt | Vì sao |
|---|---|
| "ngắn gọn (1-2 câu)" | Caption dài = thời gian sinh dài (mỗi token thêm = một lượt forward decoder) và nhiễu BM25 (nhiều từ đệm khớp lung tung) |
| "bằng tiếng Việt" | Query của operator là tiếng Việt; caption phải cùng ngôn ngữ thì BM25 mới match chữ-với-chữ. Không nhắc → Qwen có thể trả lời tiếng Anh/Trung |
| "đối tượng, hành động, bối cảnh" | Đúng 3 thứ người ta gõ khi tìm cảnh ("người đàn ông / đang câu cá / trên thuyền") |
| "văn bản trên màn nếu có" | Tận dụng khả năng OCR của VLM làm lưới thứ hai sau EasyOCR |

Và chốt chặn cứng: `caption_max_tokens: 96` — dù prompt bị lờ đi, model bị **cắt máy**
ở token thứ 96. Vì sao 96? Mỗi token sinh thêm tốn một lượt forward qua 7B tham số;
1-2 câu tiếng Việt ≈ 40-80 token; 96 là trần đủ rộng cho mô tả tử tế nhưng chặn được
trường hợp model "sa đà" viết cả đoạn văn — nhân với hàng trăm nghìn frame, mỗi token
thừa là phút-giờ ingest thừa.

---

## 6. Chat template & generation: vì sao caption dùng greedy

Hai chi tiết trong `CaptionExtractor.extract()` đáng hiểu sâu:

**Chat template.** Qwen là model *Instruct* — nó được fine-tune trên hội thoại có khuôn
dạng đặc biệt (token đánh dấu vai user/assistant). `apply_chat_template()` bọc prompt +
ảnh của ta vào đúng khuôn đó. Bỏ qua bước này, model vẫn chạy nhưng output xuống cấp
hẳn — giống nộp bài thi không theo mẫu: nội dung đúng nhưng người chấm (model) không
nhận ra đang ở "chế độ trả lời".

**Greedy decoding.** Dòng quan trọng: `do_sample=False`. Nhớ lại chương 05: ở mỗi bước,
decoder cho một phân bố xác suất trên từ điển. Có hai cách chọn:

- **Greedy** (`do_sample=False`): luôn lấy token xác suất cao nhất. **Deterministic** —
  cùng ảnh cùng prompt → cùng caption, mọi lần chạy.
- **Sampling** (`do_sample=True`, temperature > 0): rút thăm theo phân bố. Mỗi lần ra
  một biến thể khác nhau.

FUFU dùng cả hai — ở hai chỗ khác nhau, và lý do trái ngược nhau là một bài học thiết kế:

| | Caption (chương này) | Paraphrase query (chương 11) |
|---|---|---|
| Cấu hình | greedy, `do_sample=False` | sampling, temperature 0.7 |
| Cần gì | **ổn định** — caption ghi vào DB một lần, dùng mãi; chạy lại ingest phải ra cùng kết quả (reproducible, debug được) | **đa dạng** — sinh 3 cách diễn đạt *khác nhau* của cùng query; greedy sẽ ra 3 câu na ná |
| Tương tự ML cổ điển | dự đoán điểm (point estimate) | lấy mẫu từ posterior để tăng coverage |

Quy tắc bỏ túi: **ghi vào index → greedy; mở rộng không gian tìm kiếm → sampling.**

---

## 7. Quantization — nhét 28GB vào 5GB (trọng tâm chương)

### 7.1 Số thực trong máy: fp32 → fp16 → INT8 → INT4

Mỗi trọng số của mạng neural là một số thực, và máy tính phải chọn **dùng bao nhiêu bit
để lưu nó**:

| Định dạng | Bit | Byte | Trực giác độ chính xác |
|---|---|---|---|
| fp32 | 32 | 4 | ~7 chữ số thập phân — chuẩn huấn luyện cổ điển |
| fp16 / bf16 | 16 | 2 | ~3 chữ số — đủ cho inference, chuẩn de-facto |
| INT8 | 8 | 1 | 256 mức rời rạc |
| INT4 | 4 | 0.5 | **16 mức rời rạc** |

**Trực giác cốt lõi:** quantization là **làm tròn trọng số để tiết kiệm chỗ**. Giống nén
ảnh từ 16 triệu màu xuống 16 màu: từng pixel xấu đi rõ rệt, nhưng *bức tranh toàn cục*
vẫn nhận ra. Mạng neural chịu được trò này tốt bất ngờ vì kết quả của nó là **tổng của
hàng nghìn phép nhân-cộng** — sai số làm tròn từng trọng số có dương có âm, triệt tiêu
lẫn nhau phần lớn (cùng trực giác với việc ensemble nhiều model yếu, hay định lý giới
hạn trung tâm: trung bình của nhiều nhiễu nhỏ → nhiễu rất nhỏ).

Cơ chế cụ thể (mức ý tưởng): lấy một khối trọng số, tìm giá trị tuyệt đối lớn nhất
`absmax`, chia cả khối cho `absmax` để đưa về [-1, 1], rồi snap mỗi số vào mức rời rạc
gần nhất. Khi tính toán, làm ngược lại: nhân mức rời rạc với `absmax` (gọi là **scale
factor**) để khôi phục xấp xỉ giá trị gốc. Mỗi khối chỉ cần lưu: các mã 4-bit + 1 scale.

### 7.2 Tính tay cho Qwen2.5-VL-7B

7 tỷ tham số. Lấy máy tính ra:

```
fp32 :  7 × 10⁹ param × 4 byte  = 28 GB   → tràn cả RTX 3090 24GB. Loại.
fp16 :  7 × 10⁹ param × 2 byte  = 14 GB   → chạy được, nhưng ngốn hơn nửa card
INT8 :  7 × 10⁹ param × 1 byte  =  7 GB
INT4 :  7 × 10⁹ param × 0.5 byte ≈ 3.5 GB
        + scale factors + vài layer giữ nguyên 16-bit (embedding, lm_head,
          layernorm — những chỗ nhạy cảm) + activation lúc chạy
                                 ≈  5 GB thực tế
```

Hai con số **14GB (bf16)** và **~5GB (INT4)** khớp đúng docstring đầu file
`caption.py` và bảng VRAM trong PROJECT-CONTEXT §12. Đây không phải số quảng cáo —
là số đo trên máy của team.

Vì sao 5GB quan trọng sống còn: ingest của FUFU chạy **đồng thời** SigLIP (0.4G) +
EasyOCR (1-2G) + YOLO-World (1.5G) + PhoWhisper (3G) + Qwen-VL. Với Qwen-VL bf16 14GB,
tổng ≈ 21GB — sát nút 24GB, một spike activation là OOM. Với INT4 5GB, tổng ≈ 13GB —
thoải mái. **Quantization không phải tối ưu cho đẹp; nó là điều kiện để pipeline tồn tại
trên 1 GPU.**

### 7.3 nf4 và double quantization (mức trực giác)

Hai chữ bí ẩn trong config của FUFU: `nf4` và `use_double_quant`.

**nf4 (NormalFloat-4).** INT4 thường đặt 16 mức **cách đều nhau** trên [-1, 1]. Nhưng
trọng số mạng neural không phân bố đều — chúng tụ quanh 0 theo hình chuông (xấp xỉ
phân phối chuẩn, hệ quả của khởi tạo + regularization, chương 02). Đặt mức cách đều =
phí phạm: các mức ngoài rìa gần như không có trọng số nào dùng, trong khi vùng quanh 0
đông đúc lại chỉ được vài mức. nf4 đặt 16 mức theo **quantile của phân phối chuẩn** —
dày quanh 0, thưa ngoài rìa — mỗi mức "phục vụ" lượng trọng số xấp xỉ bằng nhau.
Đây chính là **equal-frequency binning** thay vì equal-width binning của ML cổ điển,
áp lên trọng số thay vì feature.

**Double quantization.** Mỗi khối 64 trọng số cần 1 scale factor fp32 → tự thân scale
chiếm 32/64 = 0.5 bit/param phụ trội — đáng kể khi mỗi param chỉ còn 4 bit! Giải pháp
đệ quy thú vị: **quantize luôn cả đám scale factor** (xuống 8-bit). Tiết kiệm thêm
~0.4 bit/param ≈ 0.3-0.4GB trên model 7B. Nén cả... metadata của phép nén.

### 7.4 Mất bao nhiêu chất lượng?

Câu hỏi vàng. Theo các benchmark công bố (QLoRA paper và các đo đạc cộng đồng trên
model 7B): nf4 + double quant thường mất **~1-3%** điểm trên các tác vụ chuẩn so với
fp16. Với việc dùng của FUFU — caption làm tài liệu BM25 — mức mất này gần như vô hình:
caption "một người đàn ông đang đá bóng trên sân cỏ" vẫn ra đúng những keyword đó, dù
một vài từ phụ có thể đổi. Đổi 1-3% chất lượng lấy **giảm 64% VRAM** (14GB → 5GB):
một trong những trade-off hời nhất toàn bộ deep learning thực dụng.

(Lưu ý chiều ngược: INT4 *chậm hơn* bf16 trên GPU dư VRAM — ~0.8-1.5s/frame vs
0.4-0.7s/frame, theo docstring `caption.py` — vì phải dequantize on-the-fly. Quantization
mua **chỗ**, không mua **tốc độ**. Nếu một ngày team có GPU 48GB, tắt `caption_quant_4bit`
là ingest nhanh gần gấp đôi.)

---

## 8. Đọc hiểu 4 dòng config thật trong caption.py

Toàn bộ mục 7 cô đặc thành đoạn code này (nguyên văn trong `app/extractors/caption.py`,
chạy khi `caption_quant_4bit: true`):

```python
from transformers import BitsAndBytesConfig
bnb = BitsAndBytesConfig(
    load_in_4bit=True,                        # bật INT4 (mục 7.1-7.2): 28GB → ~5GB
    bnb_4bit_compute_dtype=torch.bfloat16,    # LƯU 4-bit, nhưng TÍNH ở bf16:
                                              # dequantize từng khối ngay trước matmul
    bnb_4bit_quant_type="nf4",                # 16 mức theo quantile chuẩn (mục 7.3)
    bnb_4bit_use_double_quant=True,           # nén luôn scale factors (mục 7.3)
)
model = cls_load.from_pretrained(model_name, quantization_config=bnb, device_map="auto")
```

Chi tiết đáng giá nhất: `compute_dtype=bfloat16`. Trọng số chỉ *nằm im* ở dạng 4-bit;
mỗi lần cần nhân ma trận, khối trọng số liên quan được **giải nén tức thời về bf16**,
nhân xong vứt đi. Tức là: bộ nhớ trả giá 4-bit, độ chính xác phép tính vẫn là 16-bit —
sai số chỉ đến từ việc làm tròn lưu trữ, không tích lũy qua phép toán. `bitsandbytes`
là thư viện làm việc dequantize-on-the-fly này hiệu quả trên CUDA (đó là lý do
`caption.py` tự tắt mình khi không có CUDA — xem điều kiện `device != "cuda"` đầu
`__init__`).

---

## 9. Cái giá của VLM: bottleneck và kỷ luật ingest-vs-query

### 9.1 Caption là bottleneck ingest

Số liệu thật (PROJECT-CONTEXT §7.3, §12): Qwen-VL INT4 mất **~1.5s/frame**. Một phút
video → ~30 keyframe → riêng caption đã ~45 giây, chiếm áp đảo tổng thời gian xử lý
(OCR, detection, SigLIP encode mỗi thứ chỉ vài chục ms/frame). Quy mô thi đấu:
**100 giờ video ≈ ~24 giờ ingest trên 1×3090** — và phần lớn là ngồi chờ Qwen-VL viết văn.

Vì thế config có công tắc `enable_caption`. Tắt → ingest nhanh ~5-10×, nhưng mất kênh
semantic tiếng Việt → giảm recall ~5-10% trên query mơ hồ (loại query mà OCR/ASR không
cứu được). Quyết định bật/tắt là quyết định **chiến lược theo quỹ thời gian**: được phát
data trước 1 tuần → bật; chỉ có 1 đêm → tắt, sống bằng dense + OCR + ASR.

### 9.2 Vì sao VLM chỉ chạy lúc ingest, không bao giờ lúc query

Nguyên tắc thiết kế quan trọng nhất của mọi hệ retrieval: **việc đắt làm offline một
lần, việc rẻ làm online mỗi query** (sẽ thành chủ đề trung tâm của chương 15).

Tính thử để thấy "không bao giờ" nghĩa đen: nếu lúc search ta cho VLM nhìn 500 ứng viên
dense trả về → 500 × 1.5s = **12.5 phút cho một lần gõ tìm kiếm**. Trong khi toàn bộ
pipeline query hiện tại của FUFU (expand + FAISS + 2×BM25 + fuse + BGE rerank) chạy
dưới 1 giây. VLM lúc query phá vỡ hoàn toàn ngân sách latency.

**Ngoại lệ tiềm năng — VLM rerank top-20** (RESEARCH-PLAN, ý **C2**): sau khi pipeline
thường đã chốt top-20, cho Qwen-VL nhìn *ảnh thật* của 20 frame đó và chấm "frame này có
khớp query không?". 20 × 1.5s = 30s — vẫn chậm, nhưng nằm trong tầm chấp nhận cho chế độ
"tìm kỹ" khi operator bấm nút riêng. Điểm hay: BGE-reranker hiện tại (chương 12) chỉ đọc
*text mô tả* frame (caption + objects + ASR); VLM rerank nhìn *pixel thật* — bắt được
những gì mọi tầng text đã bỏ sót. Đây là kiến trúc 2 tầng chuẩn của các đội VBS mạnh.

### 9.3 Hallucination: vì sao caption mãi mãi chỉ là kênh phụ

VLM là model *sinh* — và mọi model sinh đều có thể **bịa**. Qwen-VL hoàn toàn có thể
nhìn một frame mờ và viết "người đàn ông cầm điện thoại" trong khi đó là cái ví; đếm
3 người thành 4; hay "đọc" ra chữ không tồn tại trên biển hiệu. Nó được huấn luyện để
sinh câu *hợp lý*, không phải câu *được kiểm chứng* — khi tín hiệu thị giác yếu, prior
ngôn ngữ ("cạnh bàn ăn thường có người ngồi") lấn át bằng chứng trong ảnh.

FUFU phòng thủ bằng kiến trúc, không bằng niềm tin:

- Caption chỉ là **1 trong 3 cột** của bảng FTS5 `frame_text` (ocr_text, caption,
  labels), và BM25-visual chỉ có trọng số **0.25** trong score fusion — so với dense
  0.40 và ASR 0.50 (`settings.yaml`, `retrieval.weights`).
- Dense channel (SigLIP) đo **trực tiếp trên pixel**, không qua một bước sinh văn bản
  nào → không thể bịa. Caption sai nhiều nhất chỉ làm nhiễu một kênh phụ; cảnh đúng vẫn
  được dense kéo lên.
- Caption **bổ sung recall** (thêm cách để match), không bao giờ **thay thế** dense.

Tương tự ML cổ điển: caption như một feature do con người gán nhãn bán-tự-động — hữu ích
nhưng có label noise; ta cho nó trọng số thấp trong ensemble thay vì tin tuyệt đối.

---

## Tóm tắt 10 giây

- **CLIP đo, VLM kể:** SigLIP ra 1 số similarity; Qwen-VL ra cả câu văn về ảnh. FUFU
  dùng cả hai — SigLIP quét nhanh online, Qwen-VL đọc kỹ offline.
- **Kiến trúc VLM** = ViT (mắt) → projector (phiên dịch) → LLM decoder (miệng); ảnh trở
  thành đoạn mở đầu của prompt.
- **Caption FUFU:** prompt tiếng Việt ngắn gọn, trần 96 token, greedy (`do_sample=False`)
  vì cần ổn định — khác paraphrase chương 11 cần đa dạng nên sampling.
- **Quantization:** 7B × 4B = 28GB (fp32) → 14GB (fp16) → ~5GB (INT4 nf4) — làm tròn
  trọng số xuống 16 mức đặt theo quantile chuẩn, mất ~1-3% chất lượng, đổi lấy việc cả
  pipeline ingest sống được trong 24GB.
- **Kỷ luật:** VLM ~1.5s/frame = bottleneck ingest, tuyệt đối không chạy lúc query —
  ngoại lệ duy nhất đáng cân nhắc là VLM rerank top-20 (RESEARCH-PLAN C2).
- **Hallucination:** caption có thể bịa → chỉ làm kênh BM25 phụ trọng số 0.25, không bao
  giờ thay dense.

---

## Câu hỏi ôn tập

**1. Một bạn trong team đề xuất: "Bỏ SigLIP đi, dùng luôn Qwen-VL so khớp query với từng frame cho chính xác." Phản biện bằng con số.**

<details>
<summary>Đáp án</summary>

Qwen-VL mất ~1.5s/frame. Corpus thi đấu cỡ 100h video → hàng trăm nghìn keyframe; chỉ
cần 100.000 frame × 1.5s ≈ **42 giờ cho MỘT query**. SigLIP + FAISS quét cùng lượng đó
trong vài chục ms vì so sánh vector đã tính sẵn. VLM ra cả câu văn (đắt), SigLIP ra 1 số
(rẻ) — hai công cụ cho hai pha khác nhau: VLM đọc kỹ offline lúc ingest, SigLIP quét
nhanh online lúc query.
</details>

**2. Projector trong VLM làm nhiệm vụ gì, và vì sao nó có thể nhỏ (1-2 lớp MLP) trong khi hai khối hai bên đều khổng lồ?**

<details>
<summary>Đáp án</summary>

Projector chiếu vector patch của ViT sang không gian embedding token của LLM — đổi "hệ
tọa độ" để LLM đọc patch ảnh như giả-token. Nó nhỏ được vì việc *hiểu ảnh* đã xong ở ViT
và việc *hiểu ngôn ngữ* nằm ở LLM; projector chỉ cần một phép đổi cơ sở tuyến tính (như
một phép chiếu đổi hệ tọa độ trong đại số tuyến tính), không cần học thêm tri thức gì.
</details>

**3. Tính tay: model 3B tham số (cỡ Qwen2.5-3B mà FUFU dùng paraphrase) chiếm bao nhiêu VRAM ở fp32, fp16, INT4?**

<details>
<summary>Đáp án</summary>

fp32: 3×10⁹ × 4B = **12GB**. fp16: ×2B = **6GB**. INT4: ×0.5B ≈ **1.5GB** + scale
factors + vài layer giữ 16-bit ≈ **2-2.5GB thực tế** — khớp con số "Qwen-3B INT4 2.5G"
trong PROJECT-CONTEXT §12.
</details>

**4. nf4 khác INT4 "thường" chỗ nào? Liên hệ với một kỹ thuật tiền xử lý ML cổ điển.**

<details>
<summary>Đáp án</summary>

INT4 thường đặt 16 mức cách đều (equal-width). nf4 đặt 16 mức theo quantile của phân
phối chuẩn — dày quanh 0 nơi trọng số tụ đông, thưa ngoài rìa — vì trọng số mạng neural
phân bố hình chuông quanh 0. Đây chính là **equal-frequency binning** (chia bin theo
quantile) thay vì equal-width binning khi rời rạc hóa feature trong ML cổ điển.
</details>

**5. Vì sao caption dùng `do_sample=False` còn paraphrase (chương 11) dùng temperature 0.7? Nếu đảo ngược hai cấu hình thì hỏng gì?**

<details>
<summary>Đáp án</summary>

Caption ghi vào DB một lần dùng mãi → cần deterministic để reproducible/debug được;
paraphrase cần 3 biến thể *khác nhau* để mở rộng coverage của query → cần ngẫu nhiên.
Đảo lại: caption sampling → mỗi lần ingest lại ra caption khác (index không tái lập
được, khó debug, có thể rút phải biến thể kém); paraphrase greedy → 3 lần sinh ra 3 câu
gần như trùng nhau, query expansion thành vô dụng.
</details>

**6. Trong `BitsAndBytesConfig` của FUFU, `bnb_4bit_compute_dtype=torch.bfloat16` nghĩa là gì? Nó khác `load_in_4bit` ra sao?**

<details>
<summary>Đáp án</summary>

`load_in_4bit` quy định trọng số được **lưu** ở 4-bit trong VRAM. `compute_dtype=bf16`
quy định khi nhân ma trận, khối trọng số được **dequantize tức thời về bf16** rồi mới
tính. Tức là tiết kiệm bộ nhớ ở khâu lưu trữ, nhưng phép toán vẫn chạy ở độ chính xác
16-bit — sai số chỉ đến từ làm tròn lưu trữ, không tích lũy qua tính toán. Đây cũng là
lý do INT4 chậm hơn bf16 thuần: tốn thêm bước giải nén mỗi lần dùng.
</details>

**7. Caption của Qwen-VL có thể bịa ("hallucinate"). FUFU chống đỡ rủi ro này bằng những cơ chế kiến trúc nào?**

<details>
<summary>Đáp án</summary>

(1) Caption chỉ vào kênh BM25-visual với trọng số 0.25 — thấp nhất trong 3 kênh
(dense 0.40, ASR 0.50); (2) kênh dense SigLIP đo trực tiếp trên pixel, không qua bước
sinh văn bản nên không bịa được — cảnh đúng vẫn được dense kéo lên dù caption sai;
(3) caption ngồi chung bảng FTS5 với ocr_text và labels nên một cột nhiễu không phá cả
kênh. Tinh thần: caption bổ sung recall, không bao giờ là nguồn sự thật duy nhất.
</details>

**8. Ý C2 trong RESEARCH-PLAN đề xuất "VLM rerank top-20". Vì sao 20 mà không phải 500? Và nó hơn BGE-reranker hiện tại ở điểm nào?**

<details>
<summary>Đáp án</summary>

Ngân sách thời gian: 20 × 1.5s = 30s (chấp nhận được cho chế độ "tìm kỹ"), còn
500 × 1.5s = 12.5 phút (vô dụng khi thi). Hơn BGE ở chỗ: BGE là cross-encoder text-only,
chỉ đọc caption + objects + ASR — tức đọc *mô tả* của frame; VLM rerank nhìn **pixel
thật** của frame, nên bắt được chi tiết mà mọi tầng text hóa (caption/OCR/detection) đã
bỏ sót hoặc bịa sai.
</details>

---

## Đọc thêm

- **QLoRA: Efficient Finetuning of Quantized LLMs** (Dettmers et al., 2023) — paper khai
  sinh nf4 + double quantization; mục 7.3 của chương này là bản trực giác của nó. (Phần
  LoRA của paper để dành chương 16.)
- **Qwen2.5-VL Technical Report** (Alibaba, 2025) — kiến trúc, dữ liệu huấn luyện đa ngữ,
  khả năng OCR/grounding.
- **Visual Instruction Tuning (LLaVA)** (Liu et al., 2023) — paper phổ cập công thức
  "ViT + projector MLP + LLM" mà mục 4 mô tả, dễ đọc nhất trong các paper VLM.
- **HuggingFace docs — Quantization / bitsandbytes** — tài liệu thực hành cho
  `BitsAndBytesConfig`, đối chiếu trực tiếp với code `caption.py`.
- **Object Hallucination in Image Captioning** (Rohrbach et al., 2018) — nghiên cứu sớm
  và dễ hiểu về hiện tượng caption bịa đối tượng (metric CHAIR).
- Trong repo: `PROJECT-CONTEXT.md` §7.3 + §12 (số đo bottleneck/VRAM thật),
  `RESEARCH-PLAN.md` ý C2 (VLM rerank), `app/extractors/caption.py` (toàn bộ code chương này).
