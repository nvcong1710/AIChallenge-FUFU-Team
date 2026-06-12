# Chương 16 — Fine-tuning: full FT vs LoRA/PEFT

---

## 1. Vì sao chương này tồn tại trong FUFU

Đến giờ, FUFU dùng **toàn model pretrain nguyên bản**: SigLIP-2 Large, BGE-reranker,
PhoWhisper, NLLB... tất cả tải về từ HuggingFace và chạy zero-shot. Câu hỏi tất yếu
sẽ xuất hiện trong team (thường là sau lần đầu thấy một query tiếng Việt trả về kết
quả tệ):

> "Hay là mình **finetune** model trên data tiếng Việt cho nó hợp với đề thi?"

Đây là câu hỏi NGUY HIỂM nhất của cả dự án — không phải vì finetune khó, mà vì nó
**nghe có vẻ là việc 'làm AI thật sự'** nên rất dễ cuốn cả team vào 2-3 tuần đốt GPU,
trong khi đa số trường hợp câu trả lời đúng là: **đừng**. Chương này tồn tại để
team có một **khung quyết định** trước khi đụng vào, và nếu quyết định làm thì làm
theo quy trình không tự bắn vào chân.

> 🔗 **Trong FUFU:** mọi model có thể là ứng viên finetune đều liệt kê ở
> `PROJECT-CONTEXT.md` §4 (tech stack). Hai ứng viên thực tế nhất sẽ phân tích ở
> §8 chương này: `BAAI/bge-reranker-v2-m3` (`app/backend/services/reranker.py`)
> và text-encoder của SigLIP-2 (`app/common/encoder.py`). Còn nguồn data tiềm năng
> nằm ở `RESEARCH-PLAN.md` §3 nhóm D (ý tưởng D1 — synthetic query).

---

## 2. Cần biết trước

- **Chương 02**: gradient descent, backprop, Adam optimizer — finetune chính là
  "tiếp tục huấn luyện", nên mọi chi phí của chương 02 áp dụng nguyên xi.
- **Chương 07**: contrastive learning (CLIP/SigLIP) — vì kiểu finetune sát FUFU
  nhất là contrastive finetune cho retrieval.
- **Chương 08**: quantization INT4 — cần cho phần QLoRA.
- **Chương 12**: cross-encoder reranker — một trong hai ứng viên finetune.
- ML cổ điển: **bias-variance trade-off** và **overfitting** — finetune sai cách
  chính là overfitting ở quy mô trăm triệu tham số.

Chương này KHÔNG dạy cách dựng eval set (chương 19), không dạy tune trọng số
hybrid của hệ thống (chương 17). Nhưng cả hai chương đó sẽ được **trỏ tới liên tục**,
vì chúng là điều kiện tiên quyết của mọi quyết định finetune.

---

## 3. Phần quan trọng nhất: khi nào KHÔNG nên finetune

Trong thực tế công nghiệp lẫn thi đấu, **đa số ý định finetune nên bị bác bỏ**.
Đi qua từng lý do:

### 3.1 Zero-shot thường đã đủ tốt

SigLIP-2 Large được train trên ~10 tỷ cặp ảnh-text đa ngôn ngữ. BGE-reranker-v2-m3
train trên hàng trăm triệu cặp đa ngôn ngữ có cả tiếng Việt. Lượng data bạn có thể
gom được (vài nghìn cặp) so với chúng là **một giọt nước trong hồ**. Model chỉ "kém"
khi phân phối data của bạn lệch hẳn khỏi phân phối pretrain — và muốn biết có lệch
không thì phải **đo** (chương 19), không đoán.

### 3.2 Không đủ data thì finetune chỉ phá

Quy tắc thô: contrastive finetune cho retrieval cần **tối thiểu vài nghìn cặp
(query, document đúng) chất lượng cao**, lý tưởng là vài chục nghìn. Dưới ngưỡng đó:

- Model "học thuộc" vài nghìn cặp → giống hệt overfit cây quyết định sâu trên
  100 mẫu trong ML cổ điển.
- Tệ hơn: nó **quên bớt** kiến thức tổng quát để nhồi mấy mẫu đó vào (catastrophic
  forgetting, §7) → điểm trên data bạn gom tăng, điểm trên đề thi thật **giảm**.

### 3.3 Vấn đề thường giải được bằng cách RẺ hơn nhiều

Phổ lựa chọn theo chi phí tăng dần — **luôn đi từ trên xuống**:

| Mức | Cách | Chi phí | Khi nào |
|---|---|---|---|
| 1 | **Prompt/query engineering** — sửa cách viết query, sửa template dịch/paraphrase | phút | Query expansion ra biến thể tệ |
| 2 | **Tune siêu tham số hệ thống** (chương 17) — weights 3 kênh, threshold, top-k | giờ | Ranking lệch giữa các kênh |
| 3 | **Finetune nhẹ — LoRA** (chương này) | ngày + GPU | Đã đo được model là nút thắt, có data |
| 4 | **Full finetune** | tuần + nhiều GPU | Hiếm khi đáng, xem §4 |
| 5 | **Pretrain từ đầu** | ❌ không bao giờ | — |

Ví dụ thật trong FUFU: nếu query "biển hiệu phở Hà Nội" không ra kết quả, khả năng
cao là do FTS5 tokenize hoặc trọng số `bm25_visual` — sửa ở mức 1-2 trong một buổi
chiều. Finetune SigLIP để "hiểu phở hơn" là dùng búa tạ đập ruồi, và con ruồi
thường... không nằm chỗ đó.

### 3.4 Checklist quyết định (dạng cây — đúng nghĩa decision tree)

Vui một chút: bài toán "có nên finetune không" tự nó là bài classification, và
model phù hợp nhất là một cây quyết định 5 node — không cần deep learning:

```
Đã có eval harness đo recall@k trên query giống đề thi chưa? (ch19, F1)
 ├─ CHƯA → ⛔ DỪNG. Làm eval harness trước. Không có thước thì đừng cưa.
 └─ RỒI
     └─ Eval chỉ ra MODEL là nút thắt? (không phải fusion weights, không phải
        tokenize, không phải query expansion — đã thử hết mức 1-2 ở §3.3?)
         ├─ KHÔNG → ⛔ Tune hệ thống (ch17) rẻ hơn 100×.
         └─ CÓ
             └─ Có ≥ vài nghìn cặp (query, kết quả đúng) chất lượng?
                 ├─ KHÔNG → ⛔ Đi gom/sinh data trước (§6.3), chưa train gì cả.
                 └─ CÓ
                     └─ Có đường rollback + thời gian chạy lại eval A/B?
                         ├─ KHÔNG → ⛔ Quá sát giờ thi, rủi ro > lợi ích.
                         └─ CÓ → ✅ LoRA trên model NHỎ nhất có thể (§8).
```

Bốn nhánh ⛔, một nhánh ✅ — tỷ lệ đó phản ánh đúng thực tế.

---

## 4. Full finetune — và phép tính VRAM cho thấy vì sao nó bất khả thi

Full finetune = tiếp tục huấn luyện, cập nhật **mọi** weight (chương 02). Nghe đơn
giản, nhưng tính tay chi phí bộ nhớ cho một model 7B tham số (cỡ Qwen2.5-VL-7B
trong FUFU) sẽ thấy vấn đề ngay.

Với mixed-precision fp16 + Adam, mỗi tham số cần:

| Thành phần | Bytes/param | Model 7B |
|---|---|---|
| Weights (fp16) | 2 | **14 GB** |
| Gradients (fp16) | 2 | **14 GB** |
| Adam moment 1 — m (fp32) | 4 | **28 GB** |
| Adam moment 2 — v (fp32) | 4 | **28 GB** |
| **Tổng (chưa tính activation)** | 12 | **≈ 84 GB** |

(Nhớ lại chương 02: Adam giữ **2 trạng thái** trên mỗi tham số — trung bình động
của gradient và của gradient bình phương. Chính 2 anh này chiếm 2/3 bộ nhớ. Thực
tế còn thêm bản master weights fp32 và activations, đẩy tổng lên ~100+ GB.)

RTX 3090 của team có **24 GB**. 84 GB không vừa — kể cả chỉ riêng optimizer states
đã gấp đôi card. Muốn full-finetune 7B cần cụm 4-8× A100, thứ team không có và
không cần có.

**Đây chính là lý do PEFT (Parameter-Efficient Fine-Tuning) ra đời:** thay vì cập
nhật 7 tỷ tham số, chỉ cập nhật một nhúm nhỏ — gradient + optimizer states chỉ tính
trên nhúm đó, phần còn lại của model đóng băng (frozen) nên chỉ tốn bộ nhớ weights.

---

## 5. LoRA — trực giác và con số

### 5.1 Ý tưởng

**LoRA (Low-Rank Adaptation)**: đóng băng weight gốc `W`, học thêm một **delta
rank thấp**:

```
W_mới = W (frozen) + ΔW,   với ΔW = B·A
```

trong đó nếu `W` có kích thước `d×d` thì `A` là `r×d` và `B` là `d×r`, với
`r` (rank) rất nhỏ (4-64). Forward pass cộng thêm nhánh `B·A·x` vào `W·x`.

Ẩn dụ: model pretrain là **một cuốn sách giáo khoa đã in**. Full finetune là in
lại cả cuốn sách. LoRA là **dán sticky note** vào những trang cần điều chỉnh —
sách gốc nguyên vẹn, sticky note nhỏ gọn, và quan trọng nhất: **bóc ra được**
(rollback = xoá file adapter vài chục MB, weight gốc không hề bị đụng).

### 5.2 Ví dụ số — tính tay

Lấy một projection layer điển hình trong transformer: `W` kích thước `1024×1024`.

- Full finetune layer này: cập nhật `1024 × 1024 = 1.048.576` ≈ **1M tham số**.
- LoRA rank 8: `A` là `8×1024`, `B` là `1024×8` → `2 × 8 × 1024 = 16.384` ≈
  **16k tham số** — chỉ **~1,6%** so với full.

Nhân lên cả model: LoRA điển hình train **0,1-1% tổng tham số**. Với model 7B,
phần trainable chỉ ~10-70M tham số → gradient + Adam states cho phần đó chỉ tốn
**vài trăm MB đến ~1 GB**, thay vì 70 GB. Bảng §4 sụp xuống còn:

| Thành phần | Full FT 7B | LoRA 7B |
|---|---|---|
| Weights frozen (fp16) | 14 GB | 14 GB |
| Trainable + grad + Adam | ~70 GB | **~0,5-1 GB** |
| Tổng | ~84 GB | **~15 GB** ✅ vừa 3090 |

Vì sao rank thấp mà vẫn đủ? Quan sát thực nghiệm (paper LoRA): khi adapt một model
pretrain sang task mới, **thay đổi cần thiết của weight có rank hiệu dụng rất thấp** —
giống PCA trong ML cổ điển: ma trận thay đổi trông to nhưng "thông tin thật" nằm
gọn trong vài chục chiều chính.

### 5.3 Ba hyperparameter cần biết (mức khái niệm)

- **`r` (rank)**: độ "rộng" của sticky note. `r=8-16` đủ cho đa số task; tăng `r`
  = nhiều capacity hơn nhưng dễ overfit hơn — đúng tinh thần bias-variance.
- **`alpha`**: hệ số scale cho nhánh LoRA (`ΔW` được nhân `alpha/r`). Quy ước hay
  gặp: `alpha = 2r`. Hiểu nôm na: volume của sticky note so với sách gốc.
- **`target_modules`**: dán sticky note vào layer NÀO — thường là các projection
  của attention (`q_proj`, `v_proj`, ...). Dán càng nhiều module càng mạnh và
  càng tốn, mặc định của thư viện `peft` thường là đủ.

### 5.4 QLoRA — LoRA trên model đã quantize

Để ý bảng §5.2: thứ còn chiếm nhiều nhất là **14 GB weights frozen**. Mà weights
frozen thì... không cần độ chính xác cao để tính gradient cho chúng (vì không có
gradient!). **QLoRA** = nén weights frozen xuống **INT4** (NF4 — đúng kỹ thuật
quantization của chương 08, thứ FUFU đang dùng để chạy Qwen-VL), giữ adapter LoRA
ở fp16/bf16:

```
7B fp16 frozen:  14 GB   →   7B INT4 frozen: ~4-5 GB
+ LoRA adapter + grad + Adam: ~1 GB
+ activations: vài GB
≈ 8-10 GB  →  finetune model 7B trên MỘT chiếc 3090 là hoàn toàn khả thi
```

Đây là lý do "team chỉ có 1×3090" không còn là lời từ chối finetune — lời từ chối
đúng phải đến từ §3 (không có eval / không có data), không phải từ hardware.

---

## 6. Finetune cho RETRIEVAL — kịch bản sát FUFU nhất

FUFU không sinh văn bản — FUFU **xếp hạng**. Nên kiểu finetune liên quan không
phải instruction-tuning kiểu chatbot, mà là **contrastive finetune** (chương 07)
cho hai vị trí:

1. **Text-encoder của SigLIP**: kéo embedding của query tiếng Việt lại gần
   embedding frame đúng.
2. **BGE-reranker (cross-encoder, chương 12)**: dạy nó chấm điểm cặp
   (query VN, passage = caption + objects + ASR) chuẩn hơn.

### 6.1 Data trông như thế nào

Mỗi mẫu train là một bộ:

```
query:         "người đàn ông áo cam chỉ tay vào bản đồ trên tường"
positive:      frame_001234  (frame/passage ĐÚNG mà operator muốn tìm)
hard negatives: frame_005678 ("người đàn ông áo cam đứng trước bảng trắng")
                frame_009012 ("người phụ nữ chỉ tay vào bản đồ")
```

**Hard negative là gì và vì sao quan trọng:** negative "dễ" (random — ví dụ frame
một con mèo) thì model phân biệt được ngay từ trước khi train, gradient ≈ 0, không
học được gì. Hard negative là mẫu **SAI nhưng GẦN GIỐNG** — giống các support
vector trong SVM: ranh giới quyết định được định hình bởi đúng những điểm khó nằm
sát biên, không phải đám điểm xa tít. Cách lấy hard negative chuẩn: chạy chính hệ
retrieval hiện tại, lấy top-10 kết quả **không đúng** làm negative.

### 6.2 Cần bao nhiêu

- **Sàn tối thiểu**: ~2.000-5.000 cặp chất lượng (cho reranker nhỏ).
- **Thoải mái**: 20k-100k cặp (cho text-encoder).
- Dưới sàn → quay lại §3.2: đừng train, đi gom data tiếp.

### 6.3 Lấy data từ đâu (trong bối cảnh FUFU)

| Nguồn | Cách | Ghi chú |
|---|---|---|
| **Log query thử nghiệm** | Mỗi lần team test search và tìm thấy đúng kết quả → ghi lại (query, segment đúng) | Chất lượng cao nhất, nhưng chậm tích luỹ |
| **Synthetic từ caption** | Qwen-VL đã sinh caption per-frame lúc ingest → nhờ LLM viết 5-10 query tiếng Việt "kiểu người dùng" cho mỗi frame | Chính là ý tưởng **D1 trong RESEARCH-PLAN.md §3** (doc2query) — một công đôi việc: vừa làm giàu index, vừa là data finetune |
| **Eval set của F1** | 50-100 query KIS tự tạo theo format đề thi | ⚠ KHÔNG được train trên cái này — nó là thước đo, train lên nó là gian lận với chính mình |

> 🔗 **Trong FUFU:** caption per-frame nằm sẵn trong cột `frames.caption` của
> `data/meta.sqlite` (schema ở `app/ingest/storage.py`), soi nhanh bằng
> `scripts/db_inspector.py`. Passage mà BGE-reranker đang đọc được ghép trong
> `app/backend/services/search_engine.py` (caption + "objects: ..." + ASR) — data
> finetune reranker phải ghép **đúng format này**, lệch format là train một đằng
> serve một nẻo.

---

## 7. Catastrophic forgetting — finetune hẹp làm mất khả năng tổng quát

Hiện tượng: finetune trên data hẹp (vd: toàn query về tin tức VN) → model giỏi hẳn
mảng đó nhưng **quên** kiến thức tổng quát (query tiếng Anh, khái niệm hiếm, cảnh
ngoài phân phối). Cơ chế: gradient descent chỉ quan tâm loss trên data HIỆN TẠI —
nó vô tư đè lên những weight đang mã hoá kiến thức cũ, vì không có gì nhắc nó rằng
weight đó "đang được dùng cho việc khác".

Hai hệ quả thực hành:

1. **LoRA đỡ hơn full FT một cách tự nhiên**: weight gốc bị đóng băng, "kiến thức
   cũ" còn nguyên trong sách — sticky note chỉ điều chỉnh được có giới hạn (rank
   thấp = ràng buộc cấu trúc, một dạng regularization). Và nếu vẫn quên quá nhiều:
   giảm `r`, giảm learning rate, hoặc bóc adapter ra là về nguyên trạng.
2. **Eval set phải có phần TỔNG QUÁT** (chương 19): nếu eval chỉ gồm query giống
   data train, bạn sẽ thấy điểm tăng đều và tin rằng mọi thứ tốt đẹp — trong khi
   model đang mục ruỗng ở mọi chỗ khác. Eval set cần cả: query giống đề thi + một
   lát query đa dạng (vd bộ MSR-VTT dịch Việt sẵn có của `scripts/eval_accuracy.py`)
   để phát hiện forgetting.

---

## 8. Quy trình finetune an toàn — 7 bước, không được đảo thứ tự

```
1. BASELINE EVAL   chạy eval harness (ch19), ghi recall@1/5/20 + MRR vào bảng.
                   ❌ Chưa có eval harness → quay lại §3.4, không đi tiếp.
2. DATA            gom cặp positive + mine hard negatives (§6). Tách hẳn
                   train / eval — không rò rỉ.
3. TRAIN LoRA      bắt đầu nhỏ: r=8, lr thấp (1e-4 ~ 2e-5), 1-3 epoch.
                   Lưu adapter riêng, KHÔNG ghi đè model gốc.
4. EVAL LẠI        đúng bộ eval của bước 1, đúng config hệ thống, đúng seed.
                   Đổi bất kỳ thứ gì khác cùng lúc = không kết luận được gì.
5. SO SÁNH         cả lát "giống đề thi" LẪN lát tổng quát (bắt forgetting §7).
                   Tăng lát 1 nhưng tụt lát 2 → nghi ngờ, chưa ăn mừng.
6. A/B TRÊN HỆ THẬT  cắm adapter vào pipeline đầy đủ (fusion + rerank), chạy
                   song song có/không adapter trên cùng bộ query. Model tốt lên
                   đơn lẻ nhưng hệ thống tệ đi là chuyện CÓ THẬT (lệch phân phối
                   điểm số → fusion weights cũ không còn đúng).
7. ROLLBACK SẴN SÀNG  kết quả tệ/lẫn lộn → gỡ adapter, về baseline, ghi lại bài
                   học. Với LoRA, rollback = xoá 1 file. Đó là lý do chọn LoRA.
```

Nguyên tắc sắt: **KHÔNG BAO GIỜ finetune khi chưa có eval harness.** Không có
bước 1 thì bước 5 là cảm tính, và cảm tính sau 3 ngày đốt GPU luôn nghiêng về
"chắc là tốt hơn rồi".

---

## 9. Lộ trình đề xuất riêng cho FUFU (xếp theo ROI)

### Bước 0 — Đừng finetune GÌ cho đến khi có eval set

Tương ứng F1 trong `RESEARCH-PLAN.md` §3 nhóm F: ~50-100 query KIS/TRAKE tiếng
Việt + đáp án, đo recall@k tự động. Mọi con số ở các bước sau đều vô nghĩa nếu
thiếu nó. (Và nhiều khả năng sau khi có F1, thứ cần sửa đầu tiên là **C5 — tune
trọng số hybrid**, chương 17, chứ chưa tới lượt finetune.)

### Bước 1 — Nếu eval chỉ ra reranker yếu tiếng Việt → LoRA BGE-reranker TRƯỚC

Lý do chọn nó làm "bài finetune đầu tay":

- **Nhỏ** (~0,6B) → train nhanh, vài giờ trên 3090, lặp thí nghiệm thoải mái.
- **Ít rủi ro hệ thống**: reranker chỉ reorder top-50 ở tầng cuối — tệ nhất thì
  thứ tự top-50 xấu đi, tắt bằng `retrieval.enable_reranker: false` là xong;
  không đụng index, không đụng embedding space.
- **Cross-encoder hưởng lợi nhiều từ data domain hẹp** (nó học pattern chấm điểm
  trực tiếp trên format passage của FUFU).

### Bước 2 — SigLIP text-encoder: để SAU CÙNG, và hiểu rõ rủi ro

Câu hỏi tự nhiên: "finetune SigLIP thì có phải re-ingest (encode lại) toàn bộ kho
frame không?" — **Không**, nếu chỉ finetune **text side**: vector frame trong FAISS
do image-encoder sinh ra, image-encoder đóng băng thì index giữ nguyên, chỉ thay
cách encode query.

Nhưng "không phải re-ingest" ≠ "an toàn". Rủi ro thật là **lệch không gian
(space drift)**: SigLIP được train contrastive để text-embedding và image-embedding
sống chung một không gian (chương 07). Khi chỉ kéo text-encoder đi theo data mới,
bạn đang dịch chuyển MỘT nửa không gian trong khi nửa kia đứng yên — query có thể
gần hơn với các frame trong data train, nhưng **xa ra một cách hệ thống** với hàng
triệu frame khác mà nó từng khớp tốt. Cosine score đổi phân phối → min-max norm
của kênh dense đổi → fusion weights (`retrieval.weights`) sai theo → phải tune lại
chương 17 từ đầu. Tóm lại: đụng vào trái tim của kênh mạnh nhất, blast radius lớn
nhất — chỉ làm khi bước 1 đã xong, eval vẫn chỉ đích danh dense channel, và còn
≥2 tuần trước hạn thi.

> 🔗 **Trong FUFU:** thứ tự ROI này khớp với nguyên tắc xuyên suốt ở
> `RESEARCH-PLAN.md` §5: "mỗi thay đổi phải qua F1 đo trước/sau; không merge thứ
> làm giảm recall@5". Điểm cắm adapter nếu làm bước 1: class `BGEReranker` trong
> `app/backend/services/reranker.py`; nếu làm bước 2: `SiglipEncoder.encode_text()`
> trong `app/common/encoder.py` (giữ nguyên `encode_images` — index FAISS không đổi).

---

## 10. Tóm tắt 10 giây

- **Mặc định là KHÔNG finetune** — đi phổ chi phí: prompt → tune hệ thống (ch17)
  → LoRA → full FT; đa số vấn đề chết ở 2 mức đầu.
- Full FT 7B cần ~84 GB (weights 14 + grad 14 + Adam 28+28) → không vừa 3090 →
  **LoRA**: đóng băng gốc, học delta rank thấp (1024×1024 ≈ 1M param → rank 8 chỉ
  ~16k, ~1,6%); **QLoRA** (gốc INT4) đưa 7B về ~8-10 GB, vừa 1×3090.
- Finetune cho retrieval = contrastive trên cặp (query, positive) + **hard
  negatives**; tối thiểu vài nghìn cặp; nguồn: log query + synthetic từ caption (D1).
- Coi chừng **catastrophic forgetting** — eval phải có lát tổng quát; LoRA đỡ hơn
  và rollback = xoá 1 file.
- Quy trình 7 bước, bắt đầu và kết thúc bằng **eval harness**. FUFU: eval set
  trước → LoRA BGE-reranker trước → SigLIP text-encoder sau cùng (không cần
  re-ingest nhưng rủi ro lệch không gian).

---

## 11. Câu hỏi ôn tập

**Câu 1.** Team thấy query "ca sĩ X hát ở phố đi bộ" không ra kết quả và đề xuất
finetune SigLIP ngay. Theo checklist §3.4, ba câu hỏi nào phải trả lời trước, và
thứ gì có khả năng là nguyên nhân thật hơn?

<details><summary>Đáp án</summary>

(1) Đã có eval harness chưa? (2) Eval có chỉ ra MODEL là nút thắt không, hay là
fusion weights / tokenize / query expansion? (3) Có đủ vài nghìn cặp data không?
Nguyên nhân khả dĩ hơn: entity "ca sĩ X" nằm ngoài tri thức model — không finetune
nào cứu được với vài nghìn mẫu; hướng đúng là external image search fallback
(ý tưởng B2 trong RESEARCH-PLAN) hoặc kênh OCR/ASR bắt được tên ca sĩ. Một query
fail là anecdote, không phải evidence.
</details>

**Câu 2.** Tính tay: full finetune model 7B với Adam mixed-precision fp16 cần
khoảng bao nhiêu GB cho weights + gradients + optimizer states? Thành phần nào
chiếm nhiều nhất và vì sao?

<details><summary>Đáp án</summary>

Weights fp16: 7B×2 = 14 GB; gradients fp16: 14 GB; Adam m (fp32): 28 GB; Adam v
(fp32): 28 GB → ≈ 84 GB (chưa tính activations). Optimizer states chiếm nhiều
nhất (56 GB = 2/3) vì Adam giữ 2 trạng thái fp32 (4 bytes) cho MỖI tham số, gấp
đôi độ rộng của weights fp16.
</details>

**Câu 3.** Layer 2048×2048 dùng LoRA rank 16: trainable param của LoRA là bao
nhiêu, bằng bao nhiêu % so với full finetune layer đó?

<details><summary>Đáp án</summary>

Full: 2048×2048 ≈ 4,19M param. LoRA: A (16×2048) + B (2048×16) = 2×16×2048 =
65.536 ≈ 65k param → 65.536/4.194.304 ≈ **1,56%**.
</details>

**Câu 4.** Vì sao hard negative quan trọng hơn random negative trong contrastive
finetune? Liên hệ với một khái niệm ML cổ điển, và nêu cách mine hard negative
trong FUFU.

<details><summary>Đáp án</summary>

Random negative quá dễ → model phân biệt được sẵn → loss ≈ 0, gradient ≈ 0, không
học gì. Hard negative (sai nhưng gần giống) ép model học đúng ranh giới — tương tự
support vectors trong SVM: ranh giới do các điểm khó sát biên quyết định. Trong
FUFU: chạy chính pipeline search hiện tại với query, lấy top-10 kết quả KHÔNG đúng
làm hard negatives.
</details>

**Câu 5.** QLoRA khác LoRA chỗ nào, và nó nối với kỹ thuật nào FUFU đang dùng sẵn
ở chỗ khác?

<details><summary>Đáp án</summary>

QLoRA = LoRA nhưng phần weights frozen được quantize xuống INT4 (NF4) thay vì giữ
fp16 — được vì frozen weights không cần gradient. 7B từ 14 GB còn ~4-5 GB →
finetune trên 1×3090. Cùng kỹ thuật NF4/INT4 (chương 08) mà FUFU đang dùng để chạy
Qwen2.5-VL-7B caption (`caption_quant_4bit: true`) và Qwen2.5-3B paraphrase.
</details>

**Câu 6.** Finetune text-encoder của SigLIP có buộc phải re-encode toàn bộ frame
trong FAISS không? Nếu không, rủi ro chính còn lại là gì?

<details><summary>Đáp án</summary>

Không — vector trong FAISS do image-encoder sinh; đóng băng image side thì index
giữ nguyên, chỉ đổi cách encode query. Rủi ro còn lại: **lệch không gian** — text
embedding bị kéo theo data hẹp trong khi image embedding đứng yên, query khớp tốt
hơn với frame giống data train nhưng xa ra hệ thống với phần còn lại; phân phối
cosine đổi kéo theo min-max norm và fusion weights sai, phải tune lại (ch17).
</details>

**Câu 7.** Sau khi LoRA reranker, recall@5 trên bộ query "giống đề thi" tăng từ
0,62 → 0,71 nhưng trên lát MSR-VTT dịch Việt giảm 0,55 → 0,41. Chẩn đoán và 2
hướng xử lý?

<details><summary>Đáp án</summary>

Catastrophic forgetting / overfit domain hẹp: model giỏi data giống train nhưng
mất tổng quát — đề thi thật sẽ có nhiều query rơi vào vùng "tổng quát". Xử lý:
(1) giảm cường độ — hạ r, hạ learning rate, ít epoch hơn, thêm data đa dạng vào
train; (2) nếu vẫn tệ → rollback (xoá adapter) và quay lại gom data tốt hơn. Tuyệt
đối không chỉ nhìn số 0,71 mà merge.
</details>

**Câu 8.** Vì sao bước 6 (A/B trên hệ thật) vẫn cần thiết khi bước 4-5 (eval model
đơn lẻ) đã cho kết quả tốt?

<details><summary>Đáp án</summary>

Vì FUFU là hệ nhiều tầng: dense + 2×BM25 → fusion có trọng số → cross-encoder.
Model mới có thể đổi PHÂN PHỐI điểm số (vd reranker mới chấm gắt hơn) → tương quan
giữa các tầng đổi → fusion/threshold tune cho model cũ không còn tối ưu → hệ thống
tổng thể tệ đi dù model đơn lẻ tốt lên. Chỉ A/B end-to-end trên cùng bộ query mới
bắt được tương tác này.
</details>

---

## 12. Đọc thêm

- **LoRA**: Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021)
  — paper gốc, phần thực nghiệm rank thấp rất đáng đọc.
- **QLoRA**: Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*
  (2023) — NF4, double quantization, paged optimizer.
- **Thư viện**: HuggingFace `peft` (https://huggingface.co/docs/peft) — LoRA/QLoRA
  vài chục dòng config; `sentence-transformers` mục *Training* — contrastive
  finetune bi-encoder/cross-encoder (MultipleNegativesRankingLoss dùng in-batch
  negatives, đúng tinh thần chương 07).
- **Finetune retriever/reranker**: tài liệu finetune của FlagEmbedding (BGE) —
  https://github.com/FlagOpen/FlagEmbedding — có sẵn script mine hard negatives.
- **Doc2query** (nguồn synthetic data, nối D1): Nogueira & Lin, *From doc2query
  to docTTTTTquery* (2019).
- Trong repo: `RESEARCH-PLAN.md` §3 (F1, C5, D1) và `PROJECT-CONTEXT.md` §16
  (bảng "muốn sửa X — vào đâu").
