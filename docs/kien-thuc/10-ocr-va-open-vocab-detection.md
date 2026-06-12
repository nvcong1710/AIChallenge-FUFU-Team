# Chương 10 — OCR & Open-vocabulary Detection

## 1. Vì sao chương này tồn tại trong FUFU

Ở chương 07 ta đã thấy SigLIP nhìn được "toàn cảnh" của một frame: nó biết đây là
*"một người đàn ông đứng trước tòa nhà"*. Nhưng SigLIP có một điểm mù chết người:
**nó gần như mù chữ-trong-ảnh**. Đưa cho nó hai frame — một cái có banner
*"ĐẠI HỘI ĐẢNG LẦN THỨ XIV"*, một cái có banner *"KHAI MẠC SEA GAMES 33"* — embedding
của hai frame gần như giống hệt nhau: cùng là "sân khấu, đông người, băng rôn đỏ".
Nội dung con chữ bị nén mất trong quá trình embed.

Trong khi đó, với corpus video tin tức Việt Nam của HCM AI Challenge, **chữ trên màn
hình lại chính là tín hiệu vàng**:

- **Biển hiệu, banner**: "Hội nghị APEC", "Cầu Rồng Đà Nẵng", tên cửa hàng, tên đường.
- **Phụ đề / chyron tin tức**: dòng chữ chạy dưới màn hình VTV ghi đúng tên sự kiện,
  tên người, địa danh — thứ mà query của giám khảo thường nhắc đến nguyên văn.
- **Chữ trên đồ vật**: biển số xe, số áo cầu thủ, tên sản phẩm.

Một query kiểu *"bản tin về vụ cháy ở quận 8"* gần như **chỉ** giải được bằng OCR
dòng chyron, vì về mặt hình ảnh thì vụ cháy nào trông cũng giống vụ cháy nào.

Tương tự, SigLIP nhìn toàn cảnh nên **mờ chi tiết**: nó biết "có giao thông" nhưng
không đếm được "3 chiếc xe buýt và 1 người đi xe đạp". Đó là việc của **object
detection** — tìm *cái gì* ở *đâu*, từng đối tượng một.

Chương này dạy hai "con mắt" bổ sung đó: **OCR** (Phần A) và **open-vocabulary
detection** (Phần B). Cả hai đều chạy lúc ingest, đầu ra đổ vào kênh BM25 visual
(chương 14) để bù đắp chỗ kênh dense (chương 13) bó tay.

> 🔗 **Trong FUFU:** hai extractor này nằm ở `app/extractors/ocr.py` (EasyOCR) và
> `app/extractors/detection.py` (YOLO-World v2). Chúng được gọi cho **mỗi keyframe**
> trong pipeline ingest video (`app/ingest/video/ingest.py`) và ảnh
> (`app/ingest/image/ingest.py`), kết quả ghi vào bảng FTS5 `frame_text`
> (`app/ingest/storage.py`).

---

## 2. Cần biết trước

| Kiến thức | Từ chương | Dùng để hiểu |
|---|---|---|
| CNN trích đặc trưng không gian | 03 | Backbone của text detector (CRAFT) và YOLO |
| ViT chia ảnh thành patch | 06 | Vì sao chữ nhỏ hơn 1 patch thì SigLIP "không thấy" |
| Không gian chung ảnh–text (contrastive) | 07 | Cơ chế làm YOLO-World hiểu lớp định nghĩa bằng text |
| BM25 / FTS5 | 14 (đọc sau cũng được) | Nơi đầu ra OCR/detection được tiêu thụ |

---

## 3. PHẦN A — OCR: đọc chữ trên frame

### 3.1 Trực giác: OCR là pipeline 2 bước

Con người đọc một tấm biển cũng làm 2 việc tách bạch: **(1) mắt quét tìm chỗ nào có
chữ**, rồi **(2) não đọc chỗ đó thành từ**. OCR hiện đại làm y hệt:

```
ảnh ──► [Bước 1: TEXT DETECTION]  ──► danh sách vùng-có-chữ (các bbox)
        "chỗ nào có chữ?"
     ──► [Bước 2: TEXT RECOGNITION] ──► chuỗi ký tự + confidence cho mỗi vùng
        "vùng này viết gì?"
```

**Bước 1 — Detection (tìm vùng).** EasyOCR dùng **CRAFT** (Character Region
Awareness For Text detection) — một mạng CNN (chương 03) quét ảnh và dự đoán, cho
từng pixel, xác suất "pixel này nằm trong một ký tự" và "pixel này nằm *giữa* hai ký
tự liền nhau". Nối các vùng xác suất cao lại → ra các hộp bao quanh từng dòng/từ.
Điểm hay của CRAFT: vì nó nghĩ ở mức *ký tự* chứ không phải *cả dòng*, nó bắt được
chữ cong, chữ nghiêng, chữ trên biển hiệu uốn lượn — đầy rẫy trên phố Việt Nam.

**Bước 2 — Recognition (đọc vùng).** Mỗi vùng cắt ra được đưa qua một mạng kiểu
**CRNN**: CNN trích đặc trưng theo chiều ngang → chuỗi đặc trưng → RNN/attention
giải mã thành chuỗi ký tự. Trực giác: mạng "trượt mắt" từ trái sang phải trên vùng
chữ, mỗi bước đoán một ký tự (hoặc "khoảng trống"), giống cách ta đánh vần.

Hai bước độc lập cũng giải thích hai kiểu lỗi khác nhau bạn sẽ gặp khi debug:
- Detection hụt → **cả dòng chữ biến mất** (thường do chữ quá nhỏ/mờ).
- Recognition sai → **dòng có nhưng đọc lệch** ("Hà Nội" → "Hà Nột") — vẫn có thể
  cứu được một phần nhờ BM25 match các token còn đúng.

### 3.2 EasyOCR trong FUFU và bài toán ngưỡng confidence

FUFU dùng **EasyOCR** với ngôn ngữ `[vi, en]` (hỗ trợ tiếng Việt native, chạy được
Python 3.12 — lý do đổi từ PaddleOCR). Mỗi dòng đọc được kèm một **confidence** ∈
[0,1]; FUFU lọc bằng ngưỡng `ocr_min_confidence: 0.4` trong `config/settings.yaml`.

Ngưỡng này là một trade-off kinh điển kiểu precision/recall mà bạn đã quen từ ML cổ điển:

| Ngưỡng | Được gì | Mất gì |
|---|---|---|
| Thấp (0.2) | Vớt được chữ mờ, chữ nhỏ | **Rác tràn vào FTS5**: hoa văn, logo bị "đọc" thành ký tự nhảm → BM25 match nhầm, nhiễu kênh visual |
| Cao (0.7) | Index sạch, match nào cũng đáng tin | **Sót chữ thật**: phụ đề mờ, biển hiệu xa — đúng những tín hiệu vàng ta cần |
| **0.4 (FUFU)** | Điểm cân giữa: chấp nhận ít rác để không sót chyron tin tức | — |

Lưu ý ngữ cảnh: vì FTS5 query của FUFU là OR-các-token và đã có filter
`MIN_BM25_RAW = 3.0` chặn match 1-token rác (chương 14), nên hệ chịu được một ít
noise — vì thế nghiêng về ngưỡng thấp (0.4) hợp lý hơn là ngưỡng cao.

### 3.3 Vì sao chữ Việt (trên video) khó

1. **Dấu thanh + dấu mũ chồng tầng.** "ế", "ộ", "ữ" là ký tự gốc + mũ + thanh — chỉ
   vài pixel khác nhau giữa "ơ/ô/o". Ở độ phân giải video 720p, recognition rất dễ
   rớt dấu: "Đà Nẵng" → "Da Nang". (FTS5 của FUFU dùng `remove_diacritics 0` — giữ
   dấu — nên rớt dấu là rớt match; đây là nguồn lỗi cần biết khi debug.)
2. **Font cách điệu.** Biển hiệu, karaoke, banner lễ hội dùng font trang trí mà model
   train chủ yếu trên font in chuẩn.
3. **Chữ quá nhỏ trên màn.** Nối lại chương 06: SigLIP-2 Large patch16-384 chia ảnh
   thành patch 16×16. Một dòng chyron cao ~12 pixel **nhỏ hơn 1 patch** — với ViT nó
   chỉ là vài giá trị trong 1 token, không cách nào "đọc" được. OCR thì làm việc ở mức
   pixel với detector chuyên dụng nên vẫn bắt được. Đây chính là lý do kỹ thuật sâu
   nhất khiến OCR không thể thay bằng "SigLIP tốt hơn".

### 3.4 Từ bbox 4 điểm đến chuỗi index được

EasyOCR trả mỗi dòng dưới dạng **4 điểm góc** (vì chữ có thể nghiêng — hộp là tứ giác
bất kỳ). FUFU không cần hình học chính xác, chỉ cần hộp thẳng để hiển thị, nên ép về
**xyxy** bằng min/max — đoạn duy nhất trong chương này đáng nhìn code:

```python
# app/extractors/ocr.py — 4 góc → hộp thẳng x1,y1,x2,y2
xs = [pt[0] for pt in bbox]
ys = [pt[1] for pt in bbox]
xyxy = [min(xs), min(ys), max(xs), max(ys)]
```

### 3.5 Đầu ra đi đâu?

Các dòng vượt ngưỡng được **join bằng dấu cách** thành một chuỗi `ocr_text`, gắn vào
`FrameAnnotation` của frame, rồi ghi vào cột `ocr_text` của bảng **FTS5 `frame_text`**
(cùng bảng với `caption` chương 08 và `labels` của Phần B). Khi user search, kênh
**BM25 visual** (chương 14) quét bảng này. Tức là: *chữ trên màn lúc ingest → token
trong index → match với token trong query*. Không có embedding nào ở đây cả — OCR ăn
theo đường full-text thuần túy.

> 🔗 **Trong FUFU:** luồng đầy đủ là `OCRExtractor.annotate()`
> (`app/extractors/ocr.py`) → `annotation.ocr_text` → `IndexWriter.add_frames()` ghi
> vào FTS5 `frame_text` (`app/ingest/storage.py`) → `Retriever.bm25_visual()` đọc lúc
> query (`app/backend/services/retrieval.py`).

---

## 4. PHẦN B — Object Detection: cái gì, ở đâu

### 4.1 Detection khác classification chỗ nào

Classification (chương 03) trả lời *"ảnh này là gì?"* — **một** nhãn cho **cả** ảnh.
Detection trả lời *"trong ảnh có những gì, mỗi cái ở đâu?"* — **nhiều** bộ ba
`(bbox, label, confidence)`:

```
classification:  ảnh ──► "street scene"
detection:       ảnh ──► [(xe buýt, hộp A, 0.91), (người, hộp B, 0.88),
                          (người, hộp C, 0.83), (biển báo, hộp D, 0.47)]
```

Với bài toán retrieval, cái ta khai thác chủ yếu là **danh sách label** (đếm được,
search được) và **bbox** (vẽ lên UI cho operator xác nhận nhanh).

### 4.2 Closed-set vs Open-vocabulary — bước nhảy quan trọng nhất

**YOLO truyền thống là closed-set**: train trên COCO thì chỉ biết đúng 80 lớp COCO,
mãi mãi. Giống hệt một Random Forest bạn train phân loại 10 lớp — đưa lớp thứ 11 vào
là chịu, muốn thêm phải gom data và **train lại**. Mà COCO thì không có "nón lá",
không có "áo dài", không có "xe ba gác" — các khái niệm rất Việt Nam.

**YOLO-World là open-vocabulary**: danh sách lớp được định nghĩa bằng **TEXT, lúc
chạy**. Cơ chế đằng sau chính là ý tưởng chương 07: model được train để vùng ảnh và
mô tả text nằm **cùng một không gian embedding** (kiểu CLIP). Lúc inference:

1. Bạn đưa danh sách tên lớp (chuỗi bất kỳ: `"rice bowl"`, `"áo dài"`, ...).
2. Text encoder embed từng tên lớp thành một vector — đây trở thành "classifier head"
   tạm thời.
3. Với mỗi vùng ảnh ứng viên, model so embedding vùng với embedding các tên lớp —
   lớp nào gần nhất thì gán.

Hệ quả thực dụng: **đổi danh sách lớp = đổi một list string trong config. Không train
lại, không data, không GPU-giờ nào.** Một dòng code nói lên tất cả:

```python
# app/extractors/detection.py — "train" classifier mới trong 1 dòng
self.model.set_classes(self.classes)
```

Cái giá phải trả: lớp càng hiếm/càng xa phân phối train thì độ chính xác càng giảm —
open-vocab không phải phép màu, nó nội suy từ không gian ảnh-text đã học.

### 4.3 DEFAULT_CLASSES của FUFU — chọn lớp theo ngữ cảnh tin tức VN

FUFU không dùng 80 lớp COCO mà tự soạn **~60 lớp** (`DEFAULT_CLASSES` trong
`app/extractors/detection.py`), nhóm theo đúng những gì hay xuất hiện trong corpus
tin tức/đời sống Việt Nam:

| Nhóm | Ví dụ lớp | Vì sao có mặt |
|---|---|---|
| Người + bộ phận | person, face, hand, child, elderly person | Hầu hết query nói về người; "child"/"elderly" giúp lọc nhân khẩu học |
| Giao thông | motorcycle, bus, boat, traffic light, **license plate** | Tin tức VN ngập cảnh giao thông; xe máy là "quốc xe"; biển số phục vụ query vụ việc |
| Ăn uống | food, **rice bowl, noodle bowl**, cup, bottle | "bát phở", "tô bún" — rất VN, COCO không có |
| Văn hoá / sự kiện | flag, logo, **sign, text, screen**, microphone, stage, crowd | Lễ khai mạc, họp báo, mít tinh — format chuẩn của video tin tức |
| Trò chơi / giải trí | **chess board, chess piece**, ball, guitar | Phục vụ query thể thao/giải trí cụ thể |
| Văn phòng / công nghiệp | document, money, machine, tool | Tin kinh tế, phóng sự nhà máy |
| (cùng các nhóm động vật, nội thất, kiến trúc/cảnh quan) | | |

Để ý lớp `sign`, `text`, `screen`: detection không *đọc* được chữ, nhưng nó báo
"*chỗ này có biển/màn hình*" — tín hiệu bổ trợ cho OCR và hữu ích cho query kiểu
"người đứng trước màn hình lớn".

**Thêm lớp mới = sửa config, không sửa code.** Muốn nhận diện "áo dài" và "nón lá"
cho corpus lễ hội:

```yaml
# config/settings.yaml
extractors:
  detection_classes:
    - person
    - "ao dai traditional vietnamese dress"
    - "conical hat"
    # ... giữ lại các lớp cũ cần dùng
```

Mẹo: tên lớp là *prompt* cho text encoder (train chủ yếu tiếng Anh), nên mô tả tiếng
Anh giàu ngữ nghĩa ("conical hat") thường ăn hơn từ vay mượn ("non la"). Và nhớ:
lớp mới chỉ áp dụng cho frame ingest **sau khi đổi config** — frame cũ phải ingest lại.

### 4.4 NMS và ngưỡng confidence

Detector thô luôn đề xuất **nhiều hộp gần trùng nhau** cho cùng một vật (mỗi anchor/
vị trí lân cận đều "hô lên" khi thấy vật). **NMS (Non-Maximum Suppression)** dọn dẹp
bằng trực giác đơn giản: *trong một cụm hộp chồng lấn nhiều (IoU cao) cùng nhãn, giữ
hộp tự tin nhất, vứt phần còn lại*. Như một đám đông cùng chỉ vào một con mèo — chỉ
cần nghe người chắc chắn nhất. Ultralytics làm NMS nội bộ, FUFU không phải đụng tay.

Ngưỡng confidence của detection là `detection_min_confidence: 0.25` — thấp hơn OCR
(0.4). Hợp lý vì: một label sai lọt vào FTS chỉ là 1 token nhiễu nhẹ (đã có filter
BM25 chặn), trong khi bỏ sót object là mất hẳn một tín hiệu match; và label sai vẫn
bị "phán xét" lần nữa bởi mắt operator trên UI.

### 4.5 Đầu ra đi đâu?

Mỗi frame ra `List[DetectionBox(label, confidence, bbox)]`, rồi tách đôi:

1. **Các `label`** (chuỗi text) → cột `labels` của FTS5 `frame_text` → kênh **BM25
   visual**, ngồi chung mâm với `ocr_text` và `caption`. Query "xe buýt" dịch sang
   "bus" (chương 11) sẽ match token `bus` trong labels.
2. **Toàn bộ box (label + conf + bbox)** → cột `objects_json` của bảng `frames` →
   trả về trong `best_frame.objects` của API → **frontend vẽ hộp lên thumbnail** để
   operator liếc một giây là biết kết quả có đúng không.

> 🔗 **Trong FUFU:** `DetectionExtractor.annotate()` (`app/extractors/detection.py`)
> → `annotation.objects` → `storage.py` ghi labels vào FTS5 + serialize
> `objects_json`; API trả về trong `best_frame.objects`
> (`app/backend/services/search_engine.py`).

---

## 5. Ba "con mắt" của FUFU — khi nào mắt nào cứu query nào

Đến đây FUFU đã có đủ ba kênh nhìn frame, mỗi kênh mù một kiểu và tinh một kiểu:

| | **SigLIP** (dense, ch07/13) | **OCR** (Phần A) | **Detection** (Phần B) |
|---|---|---|---|
| Nhìn thấy | Toàn cảnh, không khí, bố cục, semantic mức cảnh | Chữ trên màn, từng ký tự | Từng object + vị trí + số lượng |
| Mù với | Chữ, chi tiết nhỏ, đếm số lượng | Mọi thứ không phải chữ | Khái niệm trừu tượng, lớp ngoài danh sách |
| Dạng index | Vector FAISS (cosine) | Token FTS5 (BM25) | Token FTS5 (BM25) + JSON cho UI |
| Match kiểu | Mềm, đồng nghĩa OK | Cứng, cần trúng token (trúng cả dấu) | Cứng, cần lớp có trong danh sách |

Ba query minh họa — mỗi query chỉ một mắt cứu được:

1. **"Hoàng hôn trên bãi biển, không khí yên bình"** → chỉ **SigLIP**. Không có chữ
   nào để OCR; "không khí yên bình" không phải object. Embedding toàn cảnh bắt trọn.
2. **"Bản tin có dòng chữ 'Khai mạc SEA Games 33'"** → chỉ **OCR**. Với SigLIP mọi
   lễ khai mạc đều giống nhau; detection chỉ thấy `stage, crowd, screen`. Dòng chyron
   nằm nguyên trong `ocr_text`, BM25 match gần nguyên văn.
3. **"Cảnh có bàn cờ vua và hai người ngồi đối diện"** → **Detection** ăn điểm chắc
   nhất: `chess board`, `chess piece`, `person` ×2 là match token rõ ràng, trong khi
   SigLIP có thể lẫn bàn cờ vua với cờ tướng/board game khác (chi tiết nhỏ giữa cảnh
   rộng), và frame chẳng có chữ nào cho OCR.

Thực tế ba kênh không đấu nhau mà **cộng điểm**: chương 14 sẽ chỉ ra OCR + labels +
caption nằm chung bảng `frame_text` thành kênh BM25 visual (trọng số 0.25), hợp nhất
với dense (0.40) và BM25 ASR (0.50) — segment match nhiều mắt thì điểm càng cao.

---

## 6. Tóm tắt 10 giây

- SigLIP mù chữ và mờ chi tiết → cần **OCR** (đọc chữ) và **detection** (tìm object).
- OCR = 2 bước: **detection vùng chữ** (CRAFT/CNN) + **recognition** (CRNN đọc chuỗi);
  FUFU dùng EasyOCR `[vi,en]`, ngưỡng 0.4 = cân giữa rác-vào-FTS và sót-chữ.
- Chữ Việt khó vì dấu thanh, font cách điệu, chữ nhỏ hơn 1 patch ViT.
- **Open-vocabulary detection** (YOLO-World): lớp định nghĩa bằng text nhờ không gian
  ảnh-text chung kiểu CLIP → thêm "áo dài" chỉ cần sửa `detection_classes` trong
  config, không train lại. NMS gộp hộp trùng; ngưỡng 0.25.
- Đầu ra cả hai đổ vào FTS5 `frame_text` → kênh BM25 visual; objects_json thêm
  đường lên UI. Ba mắt SigLIP / OCR / detection mù-tinh bù nhau, fusion ở chương 14.

---

## 7. Câu hỏi ôn tập

**1. Vì sao không thể "dùng SigLIP xịn hơn" thay cho OCR?**
<details><summary>Đáp án</summary>

Hai lý do cấu trúc: (1) contrastive training nén cả ảnh thành một vector mức-cảnh,
nội dung ký tự cụ thể không sống sót qua phép nén đó — hai banner khác chữ cho
embedding gần như trùng; (2) ViT chia ảnh thành patch 16×16, dòng chữ nhỏ hơn 1 patch
thì chỉ là nhiễu trong 1 token (chương 06). OCR có detector chuyên dụng mức pixel nên
đọc được. Đây là giới hạn kiến trúc, không phải giới hạn kích thước model.
</details>

**2. OCR pipeline gồm 2 bước nào? Mỗi bước hỏng cho ra triệu chứng gì?**
<details><summary>Đáp án</summary>

Text **detection** (tìm vùng có chữ — CRAFT) và text **recognition** (đọc vùng thành
chuỗi — CRNN/attention). Detection hụt → cả dòng chữ biến mất khỏi index. Recognition
sai → dòng có nhưng lệch ký tự ("Hà Nội" → "Hà Nột") — BM25 còn cứu được phần token
đúng, nhưng với FTS5 giữ dấu của FUFU thì rớt dấu là rớt match token đó.
</details>

**3. Hạ `ocr_min_confidence` từ 0.4 xuống 0.1 thì hệ thống bị gì?**
<details><summary>Đáp án</summary>

Nhiều "chữ" ảo (hoa văn, texture bị đọc nhầm) tràn vào FTS5 `frame_text` → kênh BM25
visual match nhầm nhiều frame rác, điểm bm25_visual nhiễu kéo sai kết quả fusion. Có
filter `MIN_BM25_RAW=3.0` đỡ một phần nhưng không chặn hết khi rác trùng token query.
Ngược lại, tăng lên 0.8 thì sót phụ đề mờ — đúng tín hiệu vàng. 0.4 là điểm cân.
</details>

**4. Closed-set vs open-vocabulary detection khác nhau thế nào? Cơ chế gì cho phép open-vocab?**
<details><summary>Đáp án</summary>

Closed-set (YOLO/COCO): tập lớp cố định lúc train, thêm lớp = gom data + train lại
(như RF chỉ biết lớp đã train). Open-vocab (YOLO-World): tên lớp được text encoder
embed thành vector làm "classifier head" động lúc chạy — khả thi vì model học chung
không gian ảnh-text kiểu CLIP (chương 07), nên vùng ảnh so trực tiếp với embedding
của bất kỳ chuỗi text nào. Đổi danh sách lớp = `set_classes(list_string)`, zero train.
</details>

**5. Muốn FUFU nhận diện "nón lá", bạn sửa gì, ở đâu, và có cần GPU train không?**
<details><summary>Đáp án</summary>

Thêm `detection_classes` vào khối `extractors` trong `config/settings.yaml` (ví dụ
`"conical hat"` — prompt tiếng Anh mô tả thường ăn hơn), restart và ingest lại các
file cần lớp mới (lớp chỉ áp cho frame ingest sau đổi config). Không train, không
data, không GPU-giờ — đó chính là điểm bán của open-vocabulary.
</details>

**6. NMS giải quyết vấn đề gì? Nêu trực giác.**
<details><summary>Đáp án</summary>

Detector thô đề xuất nhiều hộp gần trùng cho cùng một vật. NMS: trong cụm hộp chồng
lấn nhiều (IoU cao) cùng nhãn, giữ hộp confidence cao nhất, loại phần còn lại — như
đám đông cùng chỉ một con mèo, chỉ cần nghe người chắc nhất. Không có NMS, một xe
buýt thành 5 token `bus` ảo trong labels và 5 hộp đè nhau trên UI.
</details>

**7. Đầu ra OCR và detection cùng đổ về đâu, và được kênh nào tiêu thụ lúc query?**
<details><summary>Đáp án</summary>

`ocr_text` và các `label` (cùng caption chương 08) ghi vào bảng FTS5 `frame_text`
(`app/ingest/storage.py`) — kênh **BM25 visual** (chương 14, trọng số 0.25) quét lúc
query. Riêng detection còn ghi đầy đủ `objects_json` (label+conf+bbox) vào bảng
`frames` để API trả về `best_frame.objects` cho frontend vẽ hộp.
</details>

**8. Query "biển quảng cáo ghi 'Trà sữa nhà làm' cạnh một chiếc xe máy" — mắt nào xử lý phần nào?**
<details><summary>Đáp án</summary>

OCR match token "Trà sữa nhà làm" trong `ocr_text` (phần định danh mạnh nhất);
detection match `motorcycle` + `sign` trong labels; SigLIP bắt bố cục "biển hiệu cạnh
xe máy trên phố". Fusion chương 14 cộng cả ba — segment trúng cả ba kênh nổi hẳn lên
so với segment chỉ trúng một.
</details>

---

## 8. Đọc thêm

- Baek et al., *Character Region Awareness for Text Detection (CRAFT)*, CVPR 2019 —
  detector mức ký tự mà EasyOCR dùng.
- Shi et al., *An End-to-End Trainable Neural Network for Image-based Sequence
  Recognition (CRNN)*, TPAMI 2017 — kiến trúc recognition kinh điển.
- Cheng et al., *YOLO-World: Real-Time Open-Vocabulary Object Detection*, CVPR 2024.
- JaidedAI EasyOCR: https://github.com/JaidedAI/EasyOCR · Ultralytics YOLO-World:
  https://docs.ultralytics.com/models/yolo-world/
- Trong repo: `app/extractors/ocr.py`, `app/extractors/detection.py`,
  `PROJECT-CONTEXT.md` §4 (tech stack) và §6 (schema `frame_text`).
- Tiếp theo: **Chương 11** — dịch NLLB + paraphrase, để query tiếng Việt match được
  labels tiếng Anh vừa tạo ở chương này.
