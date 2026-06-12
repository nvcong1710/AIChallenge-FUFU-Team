# Chương 09 — ASR: PhoWhisper chuyển lời nói thành văn bản

> **ASR** = Automatic Speech Recognition — nhận dạng tiếng nói tự động.
> Đầu vào: file âm thanh. Đầu ra: văn bản (kèm mốc thời gian).
> Trong FUFU, đây là kênh biến **lời thoại** thành thứ tìm kiếm được.

---

## 1. Vì sao chương này tồn tại trong FUFU

Hãy tưởng tượng bạn là operator trong phòng thi HCM AI Challenge. Đề bài:

> *"Tìm cảnh phóng viên nói: 'mực nước sông Sài Gòn dâng cao kỷ lục'."*

Mở video lên nhìn thử: một người đứng trước ống kính, phía sau là con sông. Vấn đề là
corpus có **hàng trăm cảnh** y hệt như vậy — phóng viên, micro, sông nước. SigLIP (chương 07)
nhìn frame nào cũng thấy "person standing near river". OCR (chương 10) chỉ bắt được banner
nếu may mắn. **Thông tin phân biệt duy nhất nằm trong LỜI NÓI** — thứ không hề xuất hiện
trên bất kỳ pixel nào.

Đây không phải trường hợp hiếm. Tin tức, phỏng vấn, vlog, podcast — phần lớn nội dung
"sự kiện" được **kể bằng miệng**, không được **vẽ lên hình**. Và FUFU thừa nhận điều đó
ngay trong config:

> 🔗 **Trong FUFU:** `config/settings.yaml` đặt trọng số hợp nhất 3 kênh là
> `dense: 0.40 / bm25_visual: 0.25 / bm25_asr: 0.50` — kênh ASR đang có **weight cao nhất**.
> Tức là khi query khớp với lời thoại, hệ thống tin lời thoại hơn cả hình ảnh.
> Model ASR khai báo ở `extractors.asr_model: vinai/PhoWhisper-medium`, code chạy nó nằm ở
> `app/extractors/asr.py`.

Ngoài ra, ASR còn gánh một vai trò thứ hai ít hiển nhiên hơn: với file **audio thuần**
(podcast, ghi âm), FUFU **không có frame nào để embed** — ASR là kênh index *duy nhất*.
Không có ASR, audio coi như vô hình với search. Chi tiết ở mục 3.8.

---

## 2. Cần biết trước

- **Chương 04 (Transformer):** encoder-decoder, attention, sinh token tự hồi quy — Whisper
  chính là một encoder-decoder transformer.
- **Chương 05 (Tokenization, pretrain→finetune):** PhoWhisper là ví dụ sống của paradigm
  "pretrain trên dữ liệu khổng lồ → finetune cho domain hẹp".
- **Chương 03 (CNN) — mức khái niệm:** chỉ cần nhớ ý tưởng "ảnh = lưới số 2 chiều", vì ta sắp
  biến âm thanh thành... một dạng ảnh.

Không cần biết gì về xử lý tín hiệu số. Mọi thứ cần thiết sẽ được xây từ đầu ngay dưới đây.

---

## 3. Nội dung

### 3.1 Âm thanh trong máy tính: chỉ là một mảng số

Âm thanh vật lý là sóng áp suất không khí. Micro đo áp suất đó liên tục, còn máy tính thì
**lấy mẫu** (sample) giá trị áp suất ở những thời điểm cách đều nhau. Kết quả: âm thanh
trở thành **một mảng số 1 chiều** — gọi là **waveform**.

Số lần lấy mẫu mỗi giây gọi là **sample rate**. Với sample rate 16.000 Hz (16kHz):

```
1 giây âm thanh = 16.000 con số
                = [0.01, 0.03, 0.02, -0.01, -0.04, ...]   (16.000 phần tử)
```

Một đoạn ghi âm 10 phút = mảng ~9,6 triệu số float. Thế thôi — không có gì huyền bí.

**Vì sao FUFU resample mọi thứ về 16kHz mono?**

1. **16kHz là đủ cho tiếng nói.** Định lý lấy mẫu nói rằng sample rate 16kHz ghi lại được
   các tần số đến 8kHz — mà thông tin ngữ âm của giọng người chủ yếu nằm dưới 8kHz.
   Nhạc chất lượng cao cần 44.1kHz, nhưng ta đang nhận dạng *lời nói*, không thưởng thức nhạc.
2. **Whisper được train trên audio 16kHz** — đưa input đúng định dạng model mong đợi.
3. **Mono (1 kênh):** stereo trái/phải không thêm thông tin ngữ âm, chỉ tốn gấp đôi bộ nhớ.

> 🔗 **Trong FUFU:** `app/common/audio_io.py` — hàm `load_audio_mono_16k()` gọi **ffmpeg**
> để decode *bất kỳ* file audio/video nào (mp3, mp4, mkv...) thành PCM mono 16kHz, rồi trả về
> mảng `numpy float32` chuẩn hóa về [-1, 1]. Dùng ffmpeg subprocess thay vì librosa để
> không phụ thuộc codec — video hay audio gì cũng nuốt được.

### 3.2 Mel-spectrogram: biến âm thanh thành "ảnh"

Mảng 16.000 số/giây quá "thô" để model học trực tiếp: cùng một từ "xin chào" nói bởi hai
người sẽ cho hai waveform khác nhau hoàn toàn về hình dạng. Thứ ổn định hơn là **thành phần
tần số**: nguyên âm "a" luôn có năng lượng tập trung ở những dải tần đặc trưng, bất kể ai nói.

Cách làm: cắt waveform thành các cửa sổ rất ngắn (~25ms), với mỗi cửa sổ tính xem năng lượng
phân bố ở các tần số nào (biến đổi Fourier — không cần hiểu chi tiết). Xếp các cột kết quả
cạnh nhau theo thời gian, ta được **spectrogram** — một **lưới số 2 chiều**:

```
trục ngang  = thời gian   (mỗi cột ≈ 10ms)
trục dọc    = tần số      (thấp ở dưới, cao ở trên)
độ sáng ô   = năng lượng  (tần số đó "kêu" to cỡ nào tại thời điểm đó)
```

Nói cách khác: **âm thanh đã trở thành một bức ảnh**. Nhìn spectrogram của câu nói, bạn thấy
các "vệt sáng" uốn lượn — đó chính là các formant của nguyên âm, các "nét chữ" của tiếng nói.

Còn chữ **mel**? Tai người không nghe tuyến tính: ta phân biệt 100Hz với 200Hz rất rõ
(khác nhau cả quãng tám), nhưng 7.000Hz với 7.100Hz thì gần như chịu. **Thang mel** co giãn
trục tần số theo cảm nhận của tai — dày đặc ở tần số thấp, thưa dần ở tần số cao. Mel-spectrogram
vì thế dành "độ phân giải" cho đúng vùng tai người (và ngữ âm) quan tâm. Whisper dùng
80 dải mel → input là lưới `80 × T`.

Ý nghĩa chiến lược của bước này: spectrogram là cây cầu nối cho phép **tái sử dụng toàn bộ
kho kiến trúc xử lý ảnh/chuỗi** (CNN, transformer) cho âm thanh. Ta không cần phát minh
kiến trúc riêng cho audio — chỉ cần đổi cách "chụp ảnh" đầu vào.

### 3.3 Whisper: dịch từ "tiếng-sóng-âm" sang "tiếng-chữ"

Whisper (OpenAI, 2022) là một **encoder-decoder transformer** — đúng kiến trúc dịch máy
ở chương 04, chỉ thay ngôn ngữ nguồn bằng... âm thanh:

```
mel-spectrogram (80 × 3000)                      văn bản
        │                                            ▲
        ▼                                            │
   [ ENCODER ]  ──── cross-attention ────►     [ DECODER ]
   đọc toàn bộ                                 sinh từng token:
   "bức ảnh âm thanh",                         "mực" → "nước" → "sông" → ...
   nén thành chuỗi                             (tự hồi quy, như GPT,
   biểu diễn ngữ âm                             nhưng vừa sinh vừa "liếc"
                                                sang encoder)
```

- **Encoder** nhận mel-spectrogram (qua vài lớp conv 1D để giảm độ dài, rồi các block
  transformer), cho ra chuỗi vector biểu diễn "âm thanh này chứa những âm gì, ở đâu".
- **Decoder** là một language model sinh text từng token, mỗi bước dùng cross-attention
  để "nghe lại" phần spectrogram liên quan. Hệt như decoder dịch máy nhìn sang câu nguồn.

Trực giác đáng nhớ: **ASR = dịch máy mà ngôn ngữ nguồn là sóng âm**. Mọi thứ bạn biết
về transformer sinh chuỗi đều áp dụng nguyên xi — kể cả tật xấu của nó (hallucination,
xem mục 3.7).

Sức mạnh của Whisper đến từ dữ liệu: train trên **680.000 giờ** audio đa ngôn ngữ thu thập
từ web (audio + phụ đề có sẵn). Quy mô này cho nó độ bền đáng nể với tiếng ồn, accent,
và ~100 ngôn ngữ — trong đó có tiếng Việt, dù chỉ là "công dân hạng hai" về lượng dữ liệu.

Decoder còn nhận các **token điều khiển** đầu chuỗi: ngôn ngữ (`<|vi|>`), nhiệm vụ
(`transcribe` = ghi lại nguyên văn, `translate` = dịch sang tiếng Anh). FUFU truyền
`language="vi", task="transcribe"` để model không phải tự đoán ngôn ngữ — đoán sai một lần
là cả đoạn transcript thành tiếng Indonesia.

### 3.4 Chunking 30 giây: vì sao phải cắt, và timestamps từ đâu ra

Whisper được train với input **cố định 30 giây** (spectrogram 80×3000). Audio dài hơn?
Bắt buộc phải cắt thành các khúc 30s rồi chạy lần lượt — giống như attention có context
window hữu hạn ở chương 04.

Việc cắt do `transformers.pipeline` lo tự động: chia audio thành các cửa sổ 30s
(có overlap ở mép để câu nói bị cắt ngang không mất chữ), chạy model trên từng cửa sổ,
rồi khâu kết quả lại.

```python
# app/extractors/asr.py (rút gọn)
self.pipe = pipeline(
    task="automatic-speech-recognition",
    model="vinai/PhoWhisper-medium",
    chunk_length_s=30,        # cắt audio dài thành khúc 30s
    return_timestamps=True,   # trả (start, end) cho từng câu
)
```

Còn **timestamps**? Whisper được train để tự sinh **token thời gian** xen kẽ trong output —
trong dữ liệu train (phụ đề video), mỗi câu vốn có sẵn mốc thời gian, nên model học luôn cách
"đọc vị trí" từ spectrogram. Khi đặt `return_timestamps=True`, pipeline giữ các token đó lại
và trả về dạng:

```python
{"chunks": [
    {"timestamp": (0.0, 4.2),  "text": "mực nước sông Sài Gòn dâng cao kỷ lục"},
    {"timestamp": (4.2, 7.8),  "text": "người dân ven sông khẩn trương di dời"},
]}
```

Với FUFU, timestamps **không phải tính năng phụ** — nó là xương sống: bài toán Known-Item
Search đòi trả về *mốc thời gian để nhảy đến*, và cặp `(start, end)` này chính là thứ
trở thành `asr_segments.start/end` trong SQLite.

### 3.5 PhoWhisper: Whisper nói giọng Việt

Whisper gốc nghe được tiếng Việt, nhưng "nghe được" và "nghe tốt" là hai chuyện: trong 680k
giờ train, tiếng Việt chỉ chiếm phần nhỏ, nên Whisper gốc hay vấp dấu thanh, từ ghép,
tên riêng Việt.

**PhoWhisper** (VinAI, 2024) giải quyết bằng công thức quen thuộc từ chương 05:

```
pretrain (Whisper, 680k giờ đa ngôn ngữ)  →  finetune (~844 giờ audio TIẾNG VIỆT có nhãn)
        "biết nghe nói chung"                    "chuyên nghe tiếng Việt"
```

Đây là ví dụ sống động nhất trong toàn bộ stack FUFU của paradigm **pretrain → finetune**:
không ai đủ tiền train ASR tiếng Việt from scratch, nhưng đứng trên vai Whisper thì 844 giờ
dữ liệu Việt (đa giọng vùng miền) là đủ để vượt mọi model trước đó trên benchmark ASR tiếng Việt.
Kiến trúc, input, output, cách dùng — **giống hệt Whisper**, chỉ trọng số khác. Vì thế code
FUFU gọi nó qua đúng cái pipeline `automatic-speech-recognition` tiêu chuẩn.

PhoWhisper có các cỡ tương ứng Whisper:

| Cỡ | Tham số | VRAM (fp16) | Ghi chú |
|---|---|---|---|
| tiny / base / small | 39M–244M | <1–1GB | nhanh, sai nhiều hơn |
| **medium** ⭐ | 769M | **~3GB** | **FUFU dùng** — cân bằng tốc độ/độ chính xác |
| large | 1.55B | ~6GB | tốt nhất, chậm và tốn VRAM |

> 🔗 **Trong FUFU:** `ASRExtractor` trong `app/extractors/asr.py` load PhoWhisper-medium
> fp16 lên CUDA, và là **lazy singleton** (`app/extractors/__init__.py` → `get_asr(cfg)`):
> load đúng 1 lần, dùng chung cho cả nhánh audio lẫn nhánh video. 3GB VRAM của nó nằm trong
> ngân sách ~13GB ingest trên RTX 3090. GPU nhỏ → đổi `asr_model` sang `PhoWhisper-small`
> trong `config/settings.yaml`.

### 3.6 WER: đo độ sai của ASR

**Word Error Rate** — thước đo chuẩn của ASR. So transcript model với transcript chuẩn
(người gõ), đếm số thao tác sửa tối thiểu:

```
WER = (S + D + I) / N
  S = số từ bị thay (substitution)     D = số từ bị xóa (deletion)
  I = số từ bị chèn thừa (insertion)   N = số từ trong câu chuẩn
```

Ví dụ tính tay một câu:

```
Chuẩn:  "mực nước sông sài gòn dâng cao kỷ lục"        (N = 9 từ)
Model:  "mức nước sông sài gòn dâng cao lục"

So khớp: "mực"→"mức" = 1 substitution;  "kỷ" bị mất = 1 deletion
WER = (1 + 0 + 1) / 9 ≈ 0.22 = 22%
```

Vài mốc trực giác: WER < 10% = đọc mượt; 10–25% = hiểu được nhưng lác đác lỗi;
> 40% = gần như vô dụng. PhoWhisper-medium trên benchmark tiếng Việt sạch đạt WER
một chữ số; với audio thực tế ồn ào thì cao hơn.

Lưu ý dễ chịu cho FUFU: ta **không cần WER = 0**. Kênh ASR tìm kiếm bằng **BM25**
(chương 14) — match theo *túi từ khóa*, không cần nguyên văn. Câu sai 22% như trên vẫn
match tốt query "nước sông sài gòn dâng cao". Lỗi chỉ chí mạng khi nó rơi **đúng vào từ khóa
phân biệt** (tên riêng, số liệu) — mà éo le thay, đó lại là loại từ ASR hay sai nhất.

### 3.7 Hạn chế thực tế — và FUFU đỡ đòn thế nào

Chạy ASR trên dữ liệu thi thực tế, bạn sẽ gặp đủ các bệnh sau:

**1. Nhạc nền / tiếng ồn.** Whisper khá lì đòn với nhiễu nhẹ, nhưng nhạc nền to (intro
bản tin, quán cà phê) làm WER tăng vọt — spectrogram của nhạc đè lên spectrogram giọng nói.

**2. Hallucination khi im lặng.** Bệnh nổi tiếng nhất. Decoder là language model — bản năng
của nó là *sinh text*. Gặp đoạn 30s chỉ có nhạc hoặc im lặng, không có tín hiệu ngữ âm để
bám, nó... bịa: lặp câu trước đó, hoặc tuôn ra mấy câu học từ phụ đề YouTube kiểu
*"Hãy đăng ký kênh để ủng hộ"*, *"Cảm ơn các bạn đã theo dõi"*. Đây là hallucination
của mô hình sinh (đã hứa ở 3.3) — text bịa này lọt vào FTS5 index và có thể match query
một cách oan uổng. Nhận diện: thấy transcript lặp đi lặp lại một câu, hãy nghi đoạn đó
là nhạc/im lặng.

**3. Giọng địa phương.** PhoWhisper train đa vùng miền nên đỡ hơn Whisper gốc nhiều, nhưng
giọng nặng (Quảng Nam, miền Tây đặc sệt) vẫn rớt chữ — nhất là dấu hỏi/ngã.

**4. Từ chuyên ngành & tên riêng.** "siglip" → "xích líp", tên người, tên thuốc, thuật ngữ —
model chỉ phiên âm được những gì nó từng thấy trong dữ liệu train. Trớ trêu: đây thường là
từ khóa *đắt nhất* khi search.

**5. Phân mảnh chunk.** Whisper hay cắt câu thành nhiều mẩu vụn 1–2 giây (theo nhịp ngắt hơi).
Nếu giữ nguyên, mỗi mẩu thành 1 segment lèo tèo vài từ — BM25 match kém (từ khóa của query
rải trên 3 mẩu khác nhau, chẳng mẩu nào đủ điểm) và operator phải nhảy lắt nhắt.

> 🔗 **Trong FUFU:** bệnh số 5 được chữa bằng `merge_close_chunks()` trong
> `app/ingest/audio/segments.py`: hai chunk liền kề có khoảng lặng **gap ≤ 0.5s**
> (config `ingest.audio.merge_close_chunks_sec`) được gộp làm một — text nối bằng dấu cách,
> thời gian lấy min start / max end. Trực giác: ngắt hơi dưới nửa giây là *cùng một hơi nói*;
> ngắt lâu hơn mới là ranh giới ý. Log ingest in ra `merged 47 → 12` chính là bước này.

### 3.8 Thiết kế FUFU quanh ASR: lời nói định hình cấu trúc dữ liệu

Điểm tinh tế nhất của chương này: trong FUFU, ASR không chỉ là "thêm một trường text" —
nó **quyết định cách cắt dữ liệu thành cảnh**.

**Audio: đoạn lời = segment.** Với file audio, FUFU lấy thẳng các ASR chunk (sau merge)
làm **segments** — đơn vị nhảy-đến của hệ thống:

```
podcast.mp3 ──ffmpeg──► waveform 16kHz mono
            ──PhoWhisper──► chunks (start, end, text)
            ──merge gap ≤ 0.5s──► đoạn lời gọn
            ──asr_chunks_to_segments──► segments (đoạn >15s chia đều)
            ──► FTS5 asr_text (BM25)
```

Triết lý (ghi ngay trong docstring `app/ingest/audio/ingest.py`): **mỗi đoạn lời
pause-bounded = một "cảnh tự nhiên"** — tương tự shot-as-segment của video (chương 15).
Người ta ngắt nghỉ khi chuyển ý; ranh giới im lặng chính là camera-cut của âm thanh.
Operator search trúng là nhảy đúng vào *câu nói* cần tìm, không phải một cửa sổ 10s vô hồn
cắt ngang câu.

**Video: ASR gán vào shot.** Video đã có segment riêng (shot từ PySceneDetect), nên ASR
không tạo segment mới — mỗi ASR segment được **gán vào shot có overlap thời gian lớn nhất**
(`app/ingest/video/ingest.py`, cột `asr_segments.segment_id`). Nhờ vậy khi BM25 ASR match
một câu thoại, kết quả trỏ về đúng *cảnh hình* đang chiếu lúc câu đó vang lên — và lúc fuse
điểm, shot đó được cộng cả điểm visual lẫn điểm lời thoại.

**Giới hạn thẳng thắn: audio không lời = vô hình.** Tiếng sóng biển, nhạc không lời, tiếng
còi xe — PhoWhisper trả về rỗng (hoặc tệ hơn: hallucinate, xem 3.7). FUFU fallback sliding
window 10s/5s để item vẫn tồn tại trong DB, nhưng **không có text → không match được query
nào**. Log ingest cảnh báo rõ: *"không phát hiện speech — audio sẽ không retrievable qua
text query"*. Lấp lỗ hổng này cần một model embed **âm thanh sự kiện** vào không gian chung
với text (CLAP — "CLIP cho audio", cùng tinh thần chương 07) — đã nằm trong `RESEARCH-PLAN.md`
và mục §14 của `PROJECT-CONTEXT.md`, nhưng **chưa có trong code**.

Khi search, transcript ASR còn được tái sử dụng lần nữa: nó là một phần passage đưa vào
**cross-encoder reranker** (chương 12) và snippet `best_asr` hiển thị trên UI — một lần
chạy PhoWhisper lúc ingest, giá trị dùng lại ở ba nơi.

---

## 4. Tóm tắt 10 giây

- Âm thanh = mảng số (waveform); FUFU resample **16kHz mono** vì tiếng nói chỉ cần đến đó.
- **Mel-spectrogram** biến âm thanh thành "ảnh" thời gian × tần số (thang mel = theo tai người) → tái dùng được kiến trúc transformer.
- **Whisper** = encoder-decoder transformer: encoder đọc spectrogram, decoder *sinh* text — như dịch máy từ sóng âm sang chữ; train 680k giờ.
- Audio dài bị cắt **chunk 30s**; `return_timestamps=True` cho `(start, end)` từng câu — xương sống của Known-Item Search.
- **PhoWhisper** = Whisper finetune ~844 giờ tiếng Việt (VinAI); FUFU dùng bản **medium** (~3GB).
- **WER** = (thay + xóa + chèn) / số từ chuẩn; BM25 tha thứ lỗi nhỏ, trừ khi sai đúng từ khóa.
- Bệnh thực tế: ồn, **hallucination khi im lặng**, giọng địa phương, tên riêng; FUFU **merge chunk gap ≤ 0.5s** chống phân mảnh.
- ASR định hình dữ liệu: audio lấy **đoạn lời làm segment**; video **gán ASR vào shot overlap nhất**; audio không lời → cần CLAP (chưa có).

---

## 5. Câu hỏi ôn tập

**1. Một file audio 3 phút ở 16kHz mono là mảng bao nhiêu số? Vì sao FUFU không dùng 44.1kHz cho "chất lượng cao hơn"?**

<details><summary>Đáp án</summary>

3 × 60 × 16.000 = **2.880.000 số**. Không dùng 44.1kHz vì: (1) thông tin ngữ âm của giọng
người nằm chủ yếu dưới 8kHz, mà 16kHz đã ghi được đến 8kHz; (2) Whisper/PhoWhisper được
train trên audio 16kHz — input phải khớp định dạng model mong đợi; (3) sample rate cao hơn
chỉ tốn bộ nhớ/thời gian, không thêm độ chính xác nhận dạng.
</details>

**2. Mel-spectrogram đóng vai trò "cây cầu" gì? Chữ "mel" thêm vào điều gì so với spectrogram thường?**

<details><summary>Đáp án</summary>

Nó biến âm thanh 1D thành **lưới 2D (thời gian × tần số)** — tức một dạng "ảnh" — cho phép
tái sử dụng các kiến trúc xử lý ảnh/chuỗi (conv, transformer) thay vì phát minh kiến trúc
riêng cho audio. "Mel" co giãn trục tần số theo cảm nhận tai người: dày ở tần số thấp
(nơi tai phân biệt tốt và ngữ âm tập trung), thưa ở tần số cao — dồn độ phân giải vào
đúng vùng quan trọng.
</details>

**3. Vì sao nói "ASR = dịch máy"? Bộ phận nào của Whisper tương ứng với câu nguồn / câu đích?**

<details><summary>Đáp án</summary>

Whisper là encoder-decoder transformer y như model dịch máy chương 04: **encoder** đọc
"câu nguồn" là mel-spectrogram (chuỗi cột tần số theo thời gian), **decoder** sinh "câu đích"
là text từng token tự hồi quy, dùng cross-attention liếc về encoder. Hệ quả của góc nhìn này:
decoder thừa hưởng cả tật của model sinh — nổi bật là hallucination khi input không có
tín hiệu ngữ âm.
</details>

**4. Tính WER: câu chuẩn "hôm nay trời mưa rất to ở thủ đức" (8 từ), model ra "hôm nay trời mưa to ở thủ đực".**

<details><summary>Đáp án</summary>

So khớp: "rất" bị mất = 1 deletion; "đức"→"đực" = 1 substitution. Không có insertion.
WER = (1 + 1 + 0) / 8 = **25%**. Lưu ý: với BM25, câu này vẫn match tốt query
"trời mưa to thủ đức"... nếu FTS5 không phân biệt — nhưng FUFU giữ dấu tiếng Việt
(`remove_diacritics 0`), nên "đực" ≠ "đức": lỗi rơi đúng từ khóa địa danh là lỗi đắt nhất.
</details>

**5. Hallucination khi im lặng là gì, vì sao xảy ra, và nó gây hại gì cho FUFU?**

<details><summary>Đáp án</summary>

Gặp đoạn im lặng/nhạc không lời, decoder (bản chất là language model) không có tín hiệu
ngữ âm để bám nên **bịa text** — lặp câu trước hoặc tuôn câu quen thuộc từ dữ liệu train
("Cảm ơn các bạn đã theo dõi..."). Hại: text bịa được index vào FTS5 `asr_text` và có thể
match query một cách sai lệch, đẩy segment rác lên top kết quả. Dấu hiệu nhận biết:
transcript lặp đi lặp lại một câu y hệt.
</details>

**6. Vì sao FUFU merge các ASR chunk có gap ≤ 0.5s? Nêu cả lý do search lẫn lý do UX.**

<details><summary>Đáp án</summary>

Whisper hay cắt câu thành mẩu vụn theo nhịp ngắt hơi. **Search:** từ khóa của query rải
trên nhiều mẩu → không mẩu nào gom đủ điểm BM25 → miss. Gộp lại thì cả câu nằm trong
1 document FTS5. **UX:** mỗi mẩu vụn = 1 segment = 1 kết quả lắt nhắt vài từ; gộp lại
operator nhảy đến trọn một ý. Ngưỡng 0.5s: ngắt hơi dưới nửa giây là cùng một hơi nói,
lâu hơn mới là chuyển ý. Code: `merge_close_chunks()` trong `app/ingest/audio/segments.py`.
</details>

**7. Cùng là output PhoWhisper, ASR segment của file audio và của video được dùng khác nhau thế nào trong cấu trúc dữ liệu FUFU?**

<details><summary>Đáp án</summary>

**Audio:** ASR chunks (sau merge) **trở thành chính các segments** của item — mỗi đoạn lời
= 1 cảnh tự nhiên, vì audio không có khái niệm shot. **Video:** segments đã được định nghĩa
bởi shot (camera-cut), nên mỗi ASR segment chỉ được **gán vào shot có overlap thời gian
lớn nhất** (`asr_segments.segment_id`) — để hit lời thoại trỏ về đúng cảnh hình, và lúc
fuse, shot đó nhận điểm cả kênh dense lẫn kênh ASR.
</details>

**8. File ghi âm tiếng mưa rơi (không lời) sau khi ingest có tìm được bằng query "tiếng mưa" không? Vì sao, và hướng khắc phục là gì?**

<details><summary>Đáp án</summary>

**Không.** PhoWhisper không phát hiện speech → không có transcript → không có gì trong
FTS5 `asr_text`; audio cũng không có frame nên không có vector SigLIP. FUFU chỉ tạo
sliding-window segment 10s/5s làm placeholder (log cảnh báo "không retrievable qua text").
Khắc phục: thêm model kiểu **CLAP** — embed âm thanh sự kiện và text vào cùng không gian
(tinh thần CLIP chương 07 nhưng cho audio) — đã nằm trong RESEARCH-PLAN.md, chưa có trong code.
</details>

---

## 6. Đọc thêm

- **Radford et al., 2022 — "Robust Speech Recognition via Large-Scale Weak Supervision"** (paper Whisper) — đọc phần 2 để thấy kiến trúc đơn giản đến bất ngờ.
- **VinAI — PhoWhisper:** https://github.com/VinAIResearch/PhoWhisper — README có bảng WER theo cỡ model trên các benchmark tiếng Việt.
- **Hugging Face docs — Automatic Speech Recognition pipeline:** giải thích `chunk_length_s`, `return_timestamps` mà `app/extractors/asr.py` đang dùng.
- **Trong repo:** `app/extractors/asr.py` (extractor), `app/ingest/audio/{ingest,segments}.py` (audio pipeline), `app/common/audio_io.py` (ffmpeg → 16kHz mono), `RESEARCH-PLAN.md` (kế hoạch CLAP).
- **Chương liên quan:** ch04 (encoder-decoder), ch05 (pretrain→finetune), ch12 (cross-encoder dùng ASR text), ch14 (BM25 trên transcript), ch15 (pipeline ingest tổng).
