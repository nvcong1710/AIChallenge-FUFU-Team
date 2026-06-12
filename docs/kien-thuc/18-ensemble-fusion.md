# Chương 18 — Kết hợp model: ensemble & fusion trong retrieval

---

## 1. Vì sao chương này tồn tại trong FUFU

Bạn đã biết một sự thật quan trọng từ ML cổ điển: **Random Forest hầu như luôn
thắng một cây quyết định đơn lẻ**, dù từng cây trong rừng chẳng có gì đặc biệt.
Sức mạnh không nằm ở từng thành viên — nằm ở việc **kết hợp nhiều thành viên
nhìn dữ liệu theo cách khác nhau**.

Retrieval cũng y hệt. Nhìn lại các đội top VBS/AIC (RESEARCH-PLAN.md §1):

- **VISIONE** (CNR-ISTI, top ổn định nhiều năm): chạy **3 embedding khác nhau
  song song** (OpenCLIP + CLIP2Video + ALADIN) rồi hợp nhất.
- Các đội VN top 2024-2025: **CLIP-family + BEiT-3 hybrid**, reweighting đa kênh.
- Bài học lặp lại trong RESEARCH-PLAN §1.3: *"Ensemble nhiều encoder > một
  encoder tốt nhất"*.

Và một điều bạn có thể chưa để ý: **FUFU vốn dĩ ĐÃ là một ensemble**. Ba kênh
dense / BM25-visual / BM25-ASR chính là ba "cây" nhìn ba khía cạnh khác nhau
của cùng một video, được vote bằng weighted sum (chương 14). Chương này đặt
thiết kế đó vào bức tranh lớn hơn — để khi team cân nhắc ý tưởng **C1
(thêm encoder thứ 2)** hay đổi cách hợp nhất sang **RRF**, bạn biết mình đang
chọn giữa những gì.

> 🔗 **Trong FUFU:** phép hợp nhất 3 kênh hiện tại nằm ở
> `app/backend/services/rerank.py` (hàm `fuse_and_aggregate`), trọng số ở
> `config/settings.yaml` (`retrieval.weights: {dense: 0.4, bm25_visual: 0.25,
> bm25_asr: 0.5}`). Menu ý tưởng ensemble nằm ở `RESEARCH-PLAN.md` §3 nhóm C
> (C1: encoder thứ 2, C2/C3: ensemble tầng rerank).

---

## 2. Cần biết trước

- **Random Forest / bagging** (ML cổ điển): nhiều cây yếu + ngẫu nhiên hoá
  (bootstrap + random feature) → vote → mạnh hơn từng cây. Đây là điểm xuất
  phát của cả chương.
- **Chương 14**: weighted sum fusion + vì sao phải chuẩn hoá thang điểm trước
  khi cộng (min-max cho dense, chia BM25_SCALE cho BM25). Chương này KHÔNG dạy
  lại — chỉ đặt nó cạnh các lựa chọn khác.
- **Chương 13**: FAISS index (để hiểu chi phí khi thêm 1 index nữa).
- **Chương 12**: cross-encoder rerank (mục 7 nói về ensemble ở tầng đó).
- **Chương 19**: eval harness — mọi quyết định ensemble trong chương này cuối
  cùng đều phải đo bằng nó.

---

## 3. Từ Random Forest đến ensemble retrieval

### 3.1 Ôn lại 30 giây: vì sao rừng thắng cây

Một cây quyết định overfit theo kiểu *riêng của nó*: cây A lỡ chia nhánh sai ở
feature X, cây B lại sai ở feature Y. Nếu các cây **sai khác chỗ nhau** (lỗi
không tương quan), thì khi vote, lỗi của từng cây bị "pha loãng" — đa số vẫn
đúng. Toán đơn giản: 5 classifier độc lập, mỗi cái đúng 70%; majority vote
đúng khi ≥3 cái đúng → xác suất ≈ 84%. Cộng thêm thành viên thứ 6, 7... con
số tiếp tục nhích lên.

Nhưng có một **điều kiện vàng** mà bagging tốn rất nhiều công để đạt được
(bootstrap sample, random feature subset): các thành viên phải **ĐA DẠNG**.
Năm cây giống hệt nhau vote = một cây. Ensemble chỉ có giá trị khi các thành
viên **sai ở những chỗ khác nhau**.

### 3.2 Dịch sang ngôn ngữ retrieval

Thay "classifier dự đoán nhãn" bằng "kênh retrieval trả về ranking":

| Random Forest | Ensemble retrieval |
|---|---|
| 1 cây quyết định | 1 kênh tìm kiếm (1 encoder, 1 BM25 index...) |
| Cây dự đoán nhãn cho 1 mẫu | Kênh chấm điểm/xếp hạng các document |
| Vote đa số / trung bình | Fusion (weighted sum, RRF...) |
| Diversity nhờ bootstrap + random feature | Diversity nhờ **modality/kiến trúc/dữ liệu train khác nhau** |
| Lỗi không tương quan → vote sửa lỗi | Kênh A trượt query này, kênh B vớt lại |

Điểm khác thú vị: trong RF ta phải **chế tạo** diversity một cách nhân tạo
(bootstrap, random subset). Trong retrieval đa phương tiện, diversity đến
**tự nhiên và rẻ**: kênh nhìn hình ảnh, kênh đọc chữ trên màn, kênh nghe lời
thoại — chúng sai khác nhau *theo thiết kế*.

### 3.3 FUFU đã là một ensemble

Soi lại pipeline search (PROJECT-CONTEXT.md §8) bằng lăng kính mới:

```
            query "biển hiệu phở Thìn, người dẫn nói về Hà Nội"
                 │
   ┌─────────────┼──────────────────┐
   │ dense SigLIP │ BM25 visual      │ BM25 ASR
   │ (HÌNH ẢNH:   │ (CHỮ TRONG HÌNH: │ (LỜI THOẠI:
   │  cảnh quán   │  OCR "PHỞ THÌN"  │  "Hà Nội" trong
   │  phở nói     │  + caption       │  transcript)
   │  chung)      │  + nhãn object)  │
   └──────┬───────┴────────┬─────────┴──────┬──────
          └────────────────┴────────────────┘
        fuse_and_aggregate: 0.40·dense + 0.25·visual + 0.50·asr
```

Ba kênh này thoả điều kiện vàng một cách lý tưởng: query chỉ phân biệt được
qua chữ trên biển hiệu → dense mù tịt nhưng BM25-visual bắt được; cảnh chung
chung nhưng lời thoại đặc trưng → ASR cứu. Mỗi kênh có "vùng mù" riêng, và
vùng mù của chúng **ít chồng lên nhau** — đó là lý do hybrid 3 kênh thắng
từng kênh đơn lẻ, cùng cơ chế với việc rừng thắng cây.

---

## 4. Bản đồ fusion: EARLY vs LATE

Có hai thời điểm để "trộn" các thành viên:

### 4.1 Early fusion — trộn TRƯỚC khi tìm

Trộn ở tầng **feature**: ví dụ concat vector SigLIP (1024 chiều) với vector
BEiT-3 (1024 chiều) thành 1 vector 2048 chiều, build **một** FAISS index trên
vector ghép, tìm một lần.

- ✅ Một lần tìm kiếm duy nhất; quan hệ giữa các feature được "nhìn chung".
- ❌ Hai không gian vector vốn **không cùng thang đo** — concat thô thì
  encoder nào có norm lớn hơn sẽ lấn át (lại bài toán chuẩn hoá!).
- ❌ Cứng nhắc: muốn thêm/bớt/thay 1 encoder → **re-ingest và build lại toàn
  bộ index**. Với 100h video tốn ~24h ingest, đây là cái giá rất đau.
- ❌ Không dùng được khi hai kênh khác bản chất (vector dense vs BM25 thưa).

### 4.2 Late fusion — mỗi kênh tìm riêng, trộn KẾT QUẢ

Mỗi kênh giữ **index riêng**, tự trả về danh sách (document, điểm/hạng) của
mình; bước fusion chỉ làm việc trên các danh sách đó.

- ✅ Mỗi kênh độc lập hoàn toàn: thêm encoder mới = thêm 1 index mới, **không
  đụng** index cũ; tắt 1 kênh = bỏ qua danh sách của nó.
- ✅ Trộn được những thứ khác bản chất (cosine, BM25, điểm reranker...).
- ✅ Debug được từng kênh riêng (FUFU đã tận dụng: `score_breakdown` trong
  response API).
- ❌ Phải chạy N lần tìm kiếm; và phải giải bài toán "điểm của các kênh không
  cùng thang đo" ở bước trộn.

Đây là lý do **late fusion thống trị thực tế** — VISIONE, các đội VN, và chính
FUFU đều dùng late fusion. Phần còn lại của chương chỉ bàn late fusion, với
câu hỏi trung tâm: **trộn các danh sách kết quả NHƯ THẾ NÀO?**

Có hai trường phái: trộn theo **ĐIỂM** (score fusion) và trộn theo **HẠNG**
(rank fusion).

---

## 5. Score fusion — đường FUFU đang đi, và gót chân Achilles

FUFU trộn theo điểm: chuẩn hoá điểm mỗi kênh về thang so sánh được (min-max
cho dense, `raw/8.0` cap 1.0 cho BM25 — chi tiết ở chương 14), rồi weighted
sum. Ưu điểm lớn nhất: **giữ được thông tin "mạnh bao nhiêu"** — match cosine
0.92 đóng góp nhiều hơn hẳn match 0.55, đúng như trực giác.

Nhưng nó đứng trên một giả định mong manh: **bước chuẩn hoá phải "công bằng"
giữa các kênh**. Giả định này vỡ theo nhiều cách:

1. **Outlier kéo lệch min-max.** Một frame rác có cosine bất thường cao →
   max bị kéo lên → mọi điểm dense khác bị nén xuống → kênh dense "yếu đi"
   so với BM25 *chỉ vì một outlier*.
2. **Thang điểm BM25 trôi theo corpus.** `BM25_SCALE = 8.0` được chọn theo
   corpus hiện tại; đề thi cho corpus khác (văn bản dài hơn, từ hiếm hơn) →
   phân bố raw BM25 đổi → phép chia 8.0 không còn "công bằng" nữa.
3. **Thêm kênh mới = thêm một bài chuẩn hoá mới.** Thêm encoder BEiT-3? Phân
   bố cosine của nó khác SigLIP (mean/spread khác) → lại phải tune cách
   chuẩn hoá + trọng số riêng cho nó.

Tóm lại: score fusion **tinh** nhưng **nhạy** — nó thưởng cho người chịu khó
tune, và trừng phạt khi phân bố điểm thay đổi mà không ai để ý.

---

## 6. Rank fusion — RRF: chỉ cần thứ hạng, quên thang điểm đi

### 6.1 Ý tưởng

**Reciprocal Rank Fusion (RRF)** vứt bỏ hoàn toàn điểm số thô, chỉ giữ lại
**thứ hạng** của document trong từng kênh:

```
RRF(d) = Σ (qua các kênh i mà d xuất hiện)  1 / (k + rank_i(d))
```

với `rank_i(d)` = hạng của d trong kênh i (1 = đầu bảng), và `k` là hằng số
làm mượt, **chuẩn cộng đồng là k = 60** (từ paper gốc Cormack et al. 2009,
hoạt động tốt một cách kỳ lạ trên đủ loại benchmark).

Trực giác: đứng đầu một kênh được cộng nhiều điểm; đứng sâu được cộng ít;
xuất hiện ở **nhiều kênh** thì các phần thưởng cộng dồn. Giống majority vote
của Random Forest, nhưng là "vote có trọng số theo hạng".

### 6.2 Ví dụ số — tính tay từng document

Hai kênh, mỗi kênh trả về 4 document A, B, C, D:

| Hạng | Kênh dense | Kênh BM25-ASR |
|---|---|---|
| 1 | **A** | **B** |
| 2 | B | D |
| 3 | C | C |
| 4 | D | A |

Chú ý kịch bản: A đứng **nhất** kênh dense nhưng **bét** kênh ASR; B đứng nhì
+ nhất — ổn định ở cả hai. Tính RRF với k = 60:

```
RRF(A) = 1/(60+1) + 1/(60+4) = 0.01639 + 0.01563 = 0.03202
RRF(B) = 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252
RRF(C) = 1/(60+3) + 1/(60+3) = 0.01587 + 0.01587 = 0.03175
RRF(D) = 1/(60+4) + 1/(60+2) = 0.01563 + 0.01613 = 0.03175
```

**Xếp hạng cuối: B (0.03252) > A (0.03202) > D (0.03175) ≈ C (0.03175).**

Hai quan sát đáng tiền:

- **B thắng A**: "nhì + nhất" ăn đứt "nhất + bét". RRF thưởng cho sự **đồng
  thuận giữa các kênh** — đúng tinh thần ensemble: một kênh khen nhiệt liệt
  nhưng kênh kia chê thậm tệ thì đáng nghi hơn là cả hai cùng khen vừa vừa.
- **D ≈ C gần như hoà** (chênh 0.0000076): với k = 60, khác biệt giữa hạng 3
  và hạng 4 gần như bị san phẳng. `k` chính là núm điều khiển độ "dốc": k nhỏ
  → hạng đầu được thưởng đậm hơn hẳn; k lớn → các hạng gần bình đẳng.

### 6.3 Vì sao RRF robust — và cái giá của nó

Robust vì **thứ hạng là bất biến với mọi phép biến đổi đơn điệu của điểm**:
nhân đôi mọi cosine, cộng 5 vào mọi BM25, có outlier kéo max lên trời — thứ
hạng không suy chuyển → RRF không suy chuyển. Toàn bộ mục 5 (chuẩn hoá, scale,
outlier) **biến mất khỏi bài toán**. Thêm kênh mới? Chỉ cần nó trả về một
danh sách có thứ tự — không cần biết điểm của nó đo bằng đơn vị gì.

Cái giá: RRF **mù với "mạnh bao nhiêu"**. Hạng 1 với cosine 0.95 (match rõ
mồn một) và hạng 1 với cosine 0.40 (đầu bảng của một kênh đang đoán mò) được
thưởng y như nhau. Score fusion phân biệt được hai tình huống đó; RRF thì không.

### 6.4 Khi nào dùng cái nào

| Tình huống | Nên nghiêng về |
|---|---|
| Ít kênh, hiểu rõ phân bố điểm từng kênh, có eval set để tune | **Weighted sum** (vắt được nhiều thông tin hơn) |
| Nhiều kênh hỗn tạp, thang điểm khó so (cosine + BM25 + điểm API ngoài...) | **RRF** |
| Corpus/phân bố điểm thay đổi thường xuyên, không kịp re-tune | **RRF** |
| Cần "độ mạnh tuyệt đối" (vd audio chỉ có 1 kênh ASR vẫn phải cạnh tranh được) | **Weighted sum** — FUFU đang cần đúng tính chất này (PROJECT-CONTEXT §8) |
| Prototype nhanh một kênh mới, chưa muốn đụng vào hệ trọng số | **RRF** |

Với FUFU: weighted sum hiện tại có lý do tồn tại (giữ độ mạnh tuyệt đối để
audio-only item không bị lép vế). Nhưng RRF là **biến thể đáng thử rẻ tiền**:
chỉ sửa vài chục dòng trong `fuse_and_aggregate`, không đụng index, và đặc
biệt hấp dẫn nếu team triển khai C1 (thêm encoder → thêm bài chuẩn hoá mà RRF
né được). Quyết định cuối cùng: chạy cả hai qua eval harness (chương 19),
con số recall@5 phân xử.

---

## 7. Multi-encoder ensemble — thêm "cây" cho kênh dense (ý C1)

### 7.1 Pattern của các đội top

Ba kênh của FUFU đa dạng về **modality**, nhưng kênh dense chỉ có **một** đôi
mắt: SigLIP-2. SigLIP có vùng mù riêng của nó (kiến trúc + dữ liệu train) —
query nào SigLIP "không hiểu" thì cả kênh dense sập, không ai vớt.

Giải pháp của VISIONE và các đội VN top: chạy **2-3 encoder visual song
song** — ví dụ SigLIP-2 + BEiT-3 (hoặc OpenCLIP/EVA-CLIP) — mỗi encoder một
FAISS index riêng, lúc query encode text bằng cả hai, tìm song song, rồi fuse
kết quả (weighted sum hoặc RRF) như thêm một kênh thứ tư. Đây chính là tăng
số cây trong rừng: hai encoder train khác nhau → sai khác chỗ nhau → fusion
vớt lỗi cho nhau. Lưu ý điều kiện vàng vẫn áp dụng: chọn encoder **khác họ**
(BEiT-3 khác hẳn kiến trúc/dữ liệu so với SigLIP) mới đáng; thêm một biến thể
SigLIP nữa thì gần như vote hai lá phiếu giống nhau.

### 7.2 Chi phí — tính tay cho corpus 100h video

Từ PROJECT-CONTEXT §12: 1 phút video ≈ 30 keyframe → 100h ≈ **180.000 frame**.
Thêm encoder thứ 2 nghĩa là:

- **Ingest**: +180.000 lần encode ảnh. Encode SigLIP-cỡ-Large ~25ms/frame
  (GPU, batch) → 180.000 × 25ms ≈ **75 phút GPU thêm** — nhỏ so với 24h tổng
  ingest (caption mới là bottleneck), nhưng VRAM phải chứa thêm ~1GB model.
- **Disk**: vector 1024-d fp32 = 4KB → 180.000 × 4KB ≈ **0,7GB FAISS thêm**
  (×2 nếu cả hai encoder, cộng overhead HNSW). Chấp nhận được.
- **Mỗi query**: +1 lần encode text (~20-30ms) + 1 lần FAISS search (vài ms)
  + fusion. Tổng cộng thêm **~30-50ms/query** — vẫn dưới ngân sách 1s.
- **Chi phí ẩn đắt nhất**: bài toán fusion phình ra (4 kênh thay vì 3 — thêm
  trọng số phải tune, hoặc lý do để chuyển RRF), và **mọi thay đổi corpus
  phải ingest qua 2 encoder** từ nay về sau.

Kết luận: chi phí máy móc khiêm tốn; chi phí thật là **độ phức tạp vận hành**.
Đáng làm khi (a) eval harness đã có để đo lợi ích thật, và (b) đã xác nhận
encoder thứ 2 đủ *khác* SigLIP (kiểm tra nhanh: lấy 20 query eval, xem hai
encoder fail trên những query **khác nhau** hay trùng nhau).

---

## 8. Ensemble ở tầng rerank (ngắn)

Tư duy ensemble không dừng ở tầng retrieve. Tầng rerank (chương 12) cũng có
thể có nhiều "giám khảo":

- **BGE cross-encoder** (FUFU đang có) chỉ đọc *text* (caption + objects +
  ASR). **VLM rerank** (ý C2: Qwen-VL nhìn *ảnh thật* của frame và chấm "khớp
  query không?") nhìn thứ BGE không thấy → hai giám khảo đa dạng, kết hợp
  điểm (hoặc RRF trên 2 ranking) thường ổn hơn từng cái.
- **SuperGlobal reranking** (ý C3): rerank không cần model mới — chỉnh điểm
  bằng lân cận trong không gian embedding sẵn có. Rẻ, là một "giám khảo" thứ
  ba gần như miễn phí.

Nguyên tắc y hệt mục 3: các giám khảo phải nhìn khía cạnh **khác nhau**, và
chỉ rerank top-50 nên đắt mấy cũng chịu được (khác tầng retrieve).

## 9. Knowledge distillation — thuật ngữ cần biết

Chiều ngược lại của ensemble: thay vì *chạy* nhiều model lúc inference, cho
**model to (teacher) dạy model nhỏ (student)** — student học bắt chước *phân
bố output* của teacher (hoặc của cả một ensemble), rồi lúc chạy thật chỉ cần
student. Được gì: tốc độ/VRAM của model nhỏ với chất lượng gần model to. Nhiều
model bạn đang dùng là sản phẩm distillation — `nllb-200-distilled-600M`
trong FUFU chính là student của NLLB to hơn. Với team, đây chủ yếu là thuật
ngữ để đọc paper/model card; tự distill là việc của người train model, không
phải của người xây hệ retrieval.

---

## 10. Checklist: trước khi thêm 1 thành viên ensemble

Mỗi lần ai đó đề xuất "thêm model X vào cho mạnh", chạy qua 4 câu:

1. **Diversity — nó có nhìn KHÁC các kênh hiện có không?** Nó bắt được loại
   query nào mà các kênh hiện tại trượt? Nếu nó fail cùng chỗ với kênh cũ
   (kiểm tra trên eval set), nó chỉ là lá phiếu trùng — bỏ.
2. **Chi phí đầy đủ?** Không chỉ ms/query: thêm VRAM, thêm disk, thêm thời
   gian ingest, thêm tham số fusion phải tune, thêm một thứ để hỏng.
3. **Fusion thế nào?** Điểm của nó có thang so được với kênh cũ không (score
   fusion + chuẩn hoá mới), hay nên né bằng RRF?
4. **Đo được không?** Có eval set (chương 19) để chứng minh recall@k tăng
   thật, hay chỉ "cảm giác tốt hơn"? Không đo được = không merge.

Câu 1 là câu quan trọng nhất — và là toàn bộ bài học của Random Forest.

---

## 11. Tóm tắt 10 giây

- Ensemble retrieval = Random Forest áp vào tìm kiếm: nhiều kênh **đa dạng**
  (sai khác chỗ nhau) + fusion → mạnh hơn kênh tốt nhất. FUFU 3 kênh đã là
  một ensemble (late fusion weighted sum).
- **Early fusion** trộn feature trước khi tìm (cứng, phải re-ingest khi đổi);
  **late fusion** trộn kết quả (linh hoạt, mỗi kênh giữ index riêng) — thực
  tế dùng late.
- **Score fusion** (weighted sum) giữ "độ mạnh" nhưng nhạy chuẩn hoá/outlier;
  **RRF** = Σ 1/(60+rank) chỉ cần thứ hạng — miễn nhiễm thang điểm, thưởng
  đồng thuận, nhưng mù "mạnh bao nhiêu".
- Thêm encoder thứ 2 (C1) = thêm cây cho kênh dense: chi phí máy nhỏ
  (~0,7GB disk, ~75' ingest, ~30ms/query cho 100h video), chi phí thật là độ
  phức tạp — chỉ làm khi đo được lợi ích.
- Trước khi thêm thành viên: hỏi diversity → chi phí → cách fuse → đo bằng gì.

---

## 12. Câu hỏi tự kiểm tra

**1. Điều kiện vàng của ensemble là gì, và Random Forest đạt nó bằng cách nào?
Retrieval đa phương tiện đạt nó bằng cách nào?**

<details><summary>Đáp án</summary>

Các thành viên phải **đa dạng** — sai ở những chỗ khác nhau (lỗi ít tương
quan), để khi vote/fuse, lỗi của thành viên này được thành viên khác bù. RF
phải *chế tạo* diversity nhân tạo bằng bootstrap sample + random feature
subset. Retrieval đa phương tiện có diversity *tự nhiên*: các kênh nhìn
modality khác nhau (hình ảnh / chữ trên màn / lời thoại), hoặc encoder khác
kiến trúc + dữ liệu train.
</details>

**2. Vì sao late fusion phổ biến hơn early fusion trong các hệ thi đấu?**

<details><summary>Đáp án</summary>

Late fusion cho mỗi kênh giữ index riêng → thêm/bớt/thay một kênh không phải
re-ingest và build lại index cũ (với 100h video, re-ingest tốn ~24h); trộn
được các kênh khác bản chất (cosine + BM25); debug được từng kênh độc lập.
Early fusion (concat feature, 1 index chung) tìm 1 lần duy nhất nhưng cứng
nhắc và gặp lại bài toán thang đo ngay ở tầng feature.
</details>

**3. Tính RRF (k=60) cho document X đứng hạng 2 ở kênh 1 và hạng 5 ở kênh 2;
document Y đứng hạng 1 ở kênh 1 và không xuất hiện ở kênh 2. Ai thắng?**

<details><summary>Đáp án</summary>

RRF(X) = 1/62 + 1/65 = 0.01613 + 0.01538 = **0.03151**.
RRF(Y) = 1/61 = **0.01639** (kênh không xuất hiện đóng góp 0).
X thắng áp đảo: xuất hiện ở **cả hai kênh** dù hạng thấp hơn vẫn ăn đứt đứng
nhất một kênh. RRF thưởng mạnh cho sự đồng thuận giữa các kênh.
</details>

**4. Một outlier cosine cực cao lọt vào kết quả dense. Score fusion kiểu
min-max và RRF, bên nào bị ảnh hưởng? Vì sao?**

<details><summary>Đáp án</summary>

Score fusion bị: outlier kéo max lên → mọi điểm dense khác bị nén nhỏ sau
min-max → cả kênh dense yếu đi tương đối so với các kênh khác. RRF miễn
nhiễm: outlier chỉ chiếm hạng 1, thứ hạng tương đối của các document còn lại
giữ nguyên — RRF chỉ nhìn thứ hạng, vốn bất biến với mọi biến đổi đơn điệu
của điểm.
</details>

**5. RRF đánh đổi cái gì để có sự robust đó? Cho ví dụ tình huống FUFU mà
weighted sum làm được còn RRF thì không.**

<details><summary>Đáp án</summary>

RRF mất thông tin "mạnh bao nhiêu": hạng 1 cosine 0.95 và hạng 1 cosine 0.40
được thưởng như nhau. Ví dụ trong FUFU: item audio chỉ match một kênh duy
nhất (BM25-ASR) nhưng match *rất mạnh* — weighted sum với trọng số asr=0.5 và
điểm chuẩn hoá giữ độ lớn tuyệt đối cho phép nó cạnh tranh với video match
nhiều kênh nhưng yếu; RRF chỉ thấy "xuất hiện ở 1 kênh, hạng X" nên thường
xếp nó dưới document xuất hiện ở nhiều kênh.
</details>

**6. Team đề xuất thêm một biến thể SigLIP-2 (cùng họ, checkpoint khác) làm
encoder thứ 2. Dùng checklist mục 10, bạn phản biện thế nào?**

<details><summary>Đáp án</summary>

Trượt ngay câu 1 (diversity): cùng kiến trúc + dữ liệu train tương tự → hai
encoder gần như fail trên cùng những query → lá phiếu trùng, trả đủ chi phí
(ingest ×2 kênh dense, disk, fusion phức tạp hơn) mà gần như không thêm thông
tin. Nên chọn encoder khác họ (BEiT-3, EVA-CLIP) — và vẫn phải xác nhận bằng
eval: hai encoder có fail trên những query *khác nhau* không.
</details>

**7. Knowledge distillation khác gì ensemble lúc inference, và FUFU đang dùng
sản phẩm distillation nào?**

<details><summary>Đáp án</summary>

Ensemble *chạy* nhiều model lúc inference (trả chi phí mỗi query); distillation
dồn tri thức của model to/ensemble vào MỘT model nhỏ lúc train, inference chỉ
chạy student → rẻ. FUFU đang dùng `facebook/nllb-200-distilled-600M`
(translator) — student được distill từ NLLB lớn hơn.
</details>

---

## 13. Đọc thêm

- Cormack, Clarke & Büttcher (2009), *Reciprocal Rank Fusion outperforms
  Condorcet and individual rank learning methods* (SIGIR) — paper gốc RRF, 2 trang.
- VISIONE 5.0 (MMM 2024) — hệ ensemble 3 embedding nhiều năm top VBS:
  https://link.springer.com/chapter/10.1007/978-3-031-53302-0_29
- AIO_Owlgorithms, AIC 2025 (arXiv 2512.13169) — BEiT-3 + CLIP hybrid của đội
  VN top, tham chiếu chính cho ý C1.
- Hinton, Vinyals & Dean (2015), *Distilling the Knowledge in a Neural
  Network* — paper khai sinh knowledge distillation.
- `RESEARCH-PLAN.md` §3 nhóm C — menu ý tưởng ensemble của chính FUFU, kèm
  impact/effort.
- Chương 14 (chuẩn hoá + weighted sum chi tiết) và chương 19 (eval harness —
  trọng tài cho mọi quyết định trong chương này).
