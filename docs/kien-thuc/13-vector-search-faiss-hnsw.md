# Chương 13 — Vector search: FAISS & HNSW

---

## 1. Vì sao chương này tồn tại trong FUFU

Chương 07 kết thúc ở chỗ: SigLIP encode mỗi keyframe thành 1 vector đã
L2-normalize, encode query thành 1 vector q_vec, và "độ giống nhau" = cosine.
Chương 12 nói tiếp: tầng retrieve phải **rẻ** để cross-encoder rerank tầng sau.

Nhưng còn một lỗ hổng giữa hai chương đó: lúc người dùng gõ query, FUFU có
**hàng trăm nghìn vector frame** nằm trong kho. Làm sao tìm 500 vector gần
q_vec nhất trong **vài mili-giây**, thay vì ngồi so cosine với từng vector một?

Đó chính là việc của **FAISS** — thư viện vector search của Meta — và cấu trúc
dữ liệu **HNSW** bên trong nó. Trong pipeline search (PROJECT-CONTEXT.md §8),
chương này nói về đúng một ô:

```
q_vec ──> │ DENSE FAISS: top-500 (cosine) │ ──> fuse ──> rerank ──> top-K
              ▲
              └── chương NÀY: bên trong ô này có gì?
```

> 🔗 **Trong FUFU:** index được TẠO ở `app/ingest/storage.py` (class
> `IndexWriter.__init__`, dòng `faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)`)
> và được ĐỌC để search ở `app/backend/services/retrieval.py` (class `Retriever`,
> hàm `faiss_search`). Tham số nằm trong `config/settings.yaml` khối `retrieval:`
> (`hnsw_m: 32`, `hnsw_ef_construct: 200`, `hnsw_ef_search: 128`).

---

## 2. Cần biết trước

- **Chương 07**: vector SigLIP đã L2-normalize (độ dài = 1), so nhau bằng cosine.
  Đây là tiền đề để mục 6 của chương này "ăn gian" được inner product = cosine.
- ML cổ điển: **kNN** — cho 1 điểm query, tìm k điểm gần nhất trong tập huấn
  luyện. Bạn đã biết kNN brute-force chậm thế nào khi N lớn; chương này là câu
  trả lời công nghiệp cho đúng vấn đề đó.
- Khái niệm **đồ thị** (node, cạnh) và **greedy search** ở mức trực giác.

Chương này KHÔNG dạy: cách sinh ra vector (ch07), cách hợp nhất điểm dense với
BM25 (ch14), hay cross-encoder rerank (ch12). Ở đây vector đã có sẵn — câu hỏi
duy nhất là **tìm hàng xóm gần nhất thật nhanh**.

---

## 3. Bài toán: kNN trên N vector — và vì sao brute force đuối

### 3.1 Phát biểu

Cho kho `N` vector `d₁..d_N` (mỗi vector `D` chiều) và 1 vector query `q`.
Tìm `k` vector có similarity với `q` cao nhất. **Đây chính là kNN** bạn đã học —
chỉ khác là "điểm dữ liệu" giờ là embedding SigLIP thay vì hàng trong bảng Iris.

### 3.2 Tính tay chi phí brute force

Brute force = tính similarity của `q` với TỪNG vector rồi lấy top-k.
Mỗi inner product D chiều ≈ `D` phép nhân + `D` phép cộng ≈ `2D` phép tính.

**Trường hợp 1M vector, D = 1024** (cỡ dim SigLIP-2 Large của FUFU):

```
1 query  =  N × 2D  =  1.000.000 × 2.048  ≈  2 tỷ phép tính
```

CPU hiện đại với SIMD làm thực dụng ~10 tỷ phép/giây/core
→ **~0,2 giây/query** (1 core). Vẫn "chạy được", nhưng:

- FUFU không search 1 lần: mỗi query người dùng → dense top-500 là 1 lần search,
  và đội thi cần phản hồi **dưới ~100ms** để operator lướt nhanh.
- Eval offline chạy hàng nghìn query → 0,2s/query thành nhiều phút.

**Trường hợp 100M vector** (corpus thi quy mô lớn): `100M × 2.048 ≈ 200 tỷ phép`
→ **~20 giây/query**. Chết hẳn. Không có cách nào "tối ưu code" cứu được —
phải đổi **thuật toán**: đừng so với tất cả.

So sánh ML cổ điển: đây y hệt lý do người ta chế ra **KD-tree / Ball-tree** cho
kNN. Nhưng các cây đó chết ở chiều cao (curse of dimensionality — với D > ~20
chúng thoái hoá về brute force). Vector 1024 chiều cần vũ khí khác.

### 3.3 Bức tranh tổng

| Quy mô N | Brute force / query | HNSW / query | Kết luận |
|---|---|---|---|
| 100k (FUFU hiện tại) | ~20 ms | < 1 ms | brute force "tạm sống", HNSW thoải mái |
| 1M | ~200 ms | ~1–2 ms | brute force bắt đầu nghẹt UI |
| 100M | ~20 s | ~vài ms* | brute force chết hẳn |

(*) với điều kiện RAM còn chứa nổi — xem mục 9; cỡ này thường phải kèm nén PQ.

Điểm đáng nhớ: chi phí HNSW tăng theo **log N**, brute force tăng **tuyến tính
theo N**. Khoảng cách giữa hai cột chỉ ngày càng doãng ra khi corpus thi phình to.

---

## 4. ANN — đổi một chút chính xác lấy 100× tốc độ

**ANN (Approximate Nearest Neighbor)**: chấp nhận rằng đôi khi kết quả trả về
**trượt mất hàng xóm thật** (ví dụ trả về hàng xóm gần thứ 2 thay vì thứ 1),
đổi lại tốc độ nhanh hơn brute force 10–1000×.

Thước đo: **recall@k** = trong k kết quả ANN trả về, bao nhiêu % trùng với k
kết quả đúng của brute force. HNSW chỉnh tham số hợp lý đạt **recall 95–99%**.

Vì sao FUFU sống khoẻ với 95–99%?

1. Dense lấy hẳn **top-500** rồi mới fuse + rerank — trượt 1-2 hàng xóm trong
   top-500 hầu như không đổi top-20 cuối cùng.
2. Bản thân embedding đã "xấp xỉ" ngữ nghĩa; sai số ANN nhỏ hơn nhiều so với
   sai số của chính model encode.

Tư duy quen thuộc từ ML cổ điển: giống **mini-batch SGD vs full-batch gradient** —
chấp nhận ước lượng nhiễu một chút để mỗi bước rẻ hơn hàng nghìn lần, tổng thể
thắng lớn.

---

## 5. HNSW — đồ thị nhiều tầng "cao tốc → quốc lộ → đường làng"

### 5.1 Trực giác

HNSW = **Hierarchical Navigable Small World**. Ý tưởng: nối các vector thành
**đồ thị**, mỗi vector là một ngã tư, có cạnh nối tới các vector gần nó. Tìm kiếm
= đứng ở một ngã tư, nhìn các ngã tư hàng xóm, **bước sang ngã tư nào gần query
hơn**, lặp đến khi không tiến thêm được (greedy search).

Một tầng đồ thị thì greedy dễ kẹt và đi chậm (toàn "đường làng", bước ngắn).
HNSW chồng **nhiều tầng**:

- **Tầng cao**: rất ít node, cạnh nối các node RẤT XA nhau → như **cao tốc**:
  vài bước nhảy là băng qua nửa không gian vector.
- **Tầng giữa**: nhiều node hơn, cạnh ngắn hơn → **quốc lộ**.
- **Tầng 0**: chứa TẤT CẢ N node, cạnh nối hàng xóm sát nhau → **đường làng**,
  tinh chỉnh đến đúng nhà.

(Ai học cấu trúc dữ liệu sẽ nhận ra: đây là **skip-list** phiên bản đồ thị —
mỗi phần tử được "thăng cấp" lên tầng trên với xác suất giảm dần theo cấp số nhân.)

### 5.2 Hình vẽ

```
Tầng 2 (cao tốc, ~vài chục node)
    A ───────────────────────── B
                                │ xuống tầng
Tầng 1 (quốc lộ, ~vài nghìn node)
    A ──────── C ─────── B ──── D
                                │ xuống tầng
Tầng 0 (đường làng, TẤT CẢ N node)
    A ── x ── C ── y ── B ── z ── D ── ★
                                       ▲
                              (★ = hàng xóm thật của query)

Search:  vào A ở tầng 2 → nhảy xa tới B
         xuống tầng 1   → B → D (gần hơn)
         xuống tầng 0   → D → ★  (tinh chỉnh từng bước ngắn)
```

Mỗi tầng đi greedy: luôn bước về phía node gần query nhất trong các hàng xóm.
Khi hết đường tiến ở tầng này → tụt xuống tầng dưới tại đúng chỗ đang đứng.
Tổng số bước ~ **O(log N)** thay vì O(N) — giống tra từ điển bằng cách mở giữa
sách thay vì lật từng trang.

### 5.3 Beam search thay vì greedy thuần

Greedy giữ đúng 1 ứng viên thì dễ kẹt ở "cực trị địa phương" (node mọi hàng xóm
đều xa query hơn nó, nhưng nó chưa phải gần nhất toàn cục — giống local minimum
khi train, ch02). HNSW khắc phục bằng cách giữ một **danh sách ef ứng viên tốt
nhất** đang mở rộng song song (beam search). `ef` to → ít kẹt, recall cao,
nhưng phải thăm nhiều node hơn → chậm hơn. Đây chính là tham số quan trọng nhất
ở mục sau.

### 5.4 Nhẩm độ phức tạp

Với N = 1.000.000 và mỗi node "thăng cấp" với xác suất giảm theo cấp số nhân,
số tầng ~ log(N) ≈ 14–20 tầng. Mỗi tầng đi vài bước, mỗi bước so query với
tối đa ~M hàng xóm. Tổng số inner product mỗi query cỡ:

```
(số node thăm, ~vài trăm với efSearch=128) × D
≈ 500 × 1024 ≈ 0,5 triệu phép nhân-cộng
```

So với brute force 2 **tỷ** phép ở mục 3.2: **nhanh hơn ~4.000 lần** trên cùng
kho 1M vector. Đó là nguồn gốc con số "100×–1000×" của mục 4 — và lý do trong
`timing_ms` của FUFU, `faiss_ms` hầu như luôn bé hơn `encode_ms`.

---

## 6. Ba tham số quyết định — bằng ẩn dụ đường xá

Giá trị THẬT của FUFU (`config/settings.yaml`, khối `retrieval:`):

| Tham số | FUFU | Ẩn dụ | Tăng lên thì... |
|---|---|---|---|
| `M` = 32 | số cạnh tối đa mỗi node | mỗi ngã tư nối đi mấy hướng | đồ thị "thoáng" hơn, recall ↑, nhưng RAM ↑ và build chậm hơn |
| `efConstruction` = 200 | beam width lúc **XÂY** | độ kỹ của đội làm đường: khảo sát bao nhiêu phương án trước khi chốt nối ngã tư nào với ngã tư nào | đồ thị chất lượng cao hơn (đường nối "đúng hàng xóm" hơn) → recall ↑, nhưng ingest chậm hơn. Trả giá MỘT lần lúc ingest |
| `efSearch` = 128 | beam width lúc **TÌM** | đi đường mà chịu khó ngó mấy ngã rẽ song song thay vì chỉ chăm chăm 1 lối | recall ↑, nhưng MỖI query chậm hơn. Chỉnh được lúc runtime, không cần rebuild |

Bảng trade-off định hướng (số liệu điển hình cho vài trăm nghìn → vài triệu vector):

| Cấu hình | recall@10 (điển hình) | tốc độ query |
|---|---|---|
| `efSearch` = 16 | ~85–90% | nhanh nhất |
| `efSearch` = 64 | ~95–97% | nhanh |
| **`efSearch` = 128 (FUFU)** | **~98–99%** | vài ms — quá đủ |
| `efSearch` = 512 | ~99,5%+ | chậm dần, lợi ích bão hoà |

Hai điều dễ nhầm:

1. **`efConstruction` chỉ tác dụng lúc add vector** (ingest); **`efSearch` chỉ
   tác dụng lúc query**. FUFU set chúng ở hai file khác nhau là vì vậy:
   `efConstruction` trong `IndexWriter` (storage.py), `efSearch` trong
   `Retriever` (retrieval.py, dòng `self.index.hnsw.efSearch = ef_search`).
2. `efSearch` phải ≥ k cần lấy. FUFU lấy dense top-500 nhưng `efSearch = 128`?
   Thực tế FAISS tự nâng beam lên `max(efSearch, k)` khi search — nên top-500
   vẫn chạy đúng; con số 128 là "sàn" chất lượng cho k nhỏ.

---

## 7. Inner product vs cosine vs L2 — vì sao FUFU dùng inner product

Ba metric phổ biến cho 2 vector `a`, `b`:

- **L2**: `‖a − b‖` — khoảng cách hình học, nhỏ = giống.
- **Cosine**: `a·b / (‖a‖·‖b‖)` — góc giữa 2 vector, lớn = giống.
- **Inner product (IP)**: `a·b` — thô nhất, rẻ nhất.

Một dòng đại số: nếu vector ĐÃ L2-normalize thì `‖a‖ = ‖b‖ = 1`, nên

```
cosine = a·b / (1 × 1) = a·b = inner product
```

Ví dụ số với 2 vector đơn vị trong 2D:

```
a = (0.6, 0.8)        ‖a‖ = √(0.36+0.64) = 1  ✓
b = (0.8, 0.6)        ‖b‖ = √(0.64+0.36) = 1  ✓
a·b = 0.6×0.8 + 0.8×0.6 = 0.48 + 0.48 = 0.96 = cosine(a,b)
```

(Bonus: trên vector đơn vị, L2 cũng quy về cùng thứ hạng vì
`‖a−b‖² = 2 − 2·a·b` — IP càng lớn thì L2 càng nhỏ. Ba metric cho cùng ranking.)

Đây là lý do FUFU làm 2 việc ăn khớp nhau:

- `app/common/encoder.py` **L2-normalize mọi vector** ngay khi encode (ch07);
- `app/ingest/storage.py` tạo index với **`faiss.METRIC_INNER_PRODUCT`** —
  metric rẻ nhất, nhưng nhờ normalize nên kết quả **chính là cosine**.

Vì vậy con số `raw_cosine` trong response API thực chất là IP do FAISS trả về,
và nó hợp lệ với tên gọi đó **chỉ vì** bất biến "mọi vector đã normalize"
(PROJECT-CONTEXT.md §6). Ai thêm vector chưa normalize vào index là phá metric
một cách thầm lặng — điểm số vẫn ra, nhưng không còn là cosine nữa.

---

## 8. FAISS thực dụng

### 8.1 `IndexHNSWFlat` — FUFU đang dùng gì

```python
# app/ingest/storage.py (rút gọn)
index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = ef_construct   # 200
```

Tên đọc là "HNSW + **Flat**": HNSW là đồ thị dẫn đường, **Flat = vector gốc
được giữ nguyên vẹn, không nén**. Khi search, độ giống được tính trên vector
thật → không thêm sai số lượng tử hoá nào ngoài sai số "trượt hàng xóm" của ANN.

Phía search chỉ cần đúng 3 dòng (rút gọn từ `Retriever`):

```python
# app/backend/services/retrieval.py (rút gọn)
self.index = faiss.read_index(str(index_path))   # load file data/index.faiss
self.index.hnsw.efSearch = ef_search             # 128
scores, ids = self.index.search(q, top_k)        # q: (1, D) — trả (1, top_k)
```

`scores` là inner product (= cosine, mục 7), `ids` là faiss_id — sẽ được JOIN
ngược về SQLite để biết đó là frame nào (mục 10.1). FAISS trả `-1` ở các slot
thừa khi index có ít hơn top_k vector — vì vậy `faiss_search` của FUFU lọc
`if i >= 0`.

### 8.2 Khi nào cần hơn Flat? (biết để gọi tên, không đi sâu)

Với dataset **cực lớn** (chục–trăm triệu vector) RAM không chứa nổi vector gốc,
FAISS có 2 họ kỹ thuật ghép thêm: **IVF** (chia kho thành các cluster bằng
k-means — đúng k-means bạn đã học — lúc search chỉ ngó vài cluster gần query)
và **PQ** (Product Quantization — nén mỗi vector từ 4 KB xuống vài chục byte,
chấp nhận tính similarity xấp xỉ trên bản nén). Ghép lại thành các index kiểu
`IVF...,PQ...`. FUFU ở cỡ vài trăm nghìn vector — chưa cần, và Flat cho recall
tốt nhất nên cứ Flat mà dùng.

### 8.3 faiss-cpu vs faiss-gpu

FUFU pin **`faiss-cpu`** (PROJECT-CONTEXT.md §4) dù mọi model chạy CUDA. Vì:

- HNSW search trên vài trăm nghìn vector mất **vài ms trên CPU** — đã nhanh hơn
  cả bước encode query (vài chục ms trên GPU). Đưa lên GPU không rút ngắn được
  đường găng, lại tốn VRAM đang cần cho NLLB + Qwen + reranker.
- faiss-gpu chủ yếu thắng ở brute-force/IVF batch lớn (hàng nghìn query song
  song) — không phải hồ sơ sử dụng của FUFU (1 query/lần, độ trễ thấp).

---

## 9. Ước lượng memory — tính tay

Công thức phần vector (Flat giữ nguyên float32 = 4 byte/chiều):

```
RAM_vector ≈ N × D × 4 byte
```

Cộng phần đồ thị HNSW: mỗi node lưu danh sách hàng xóm (~`2M` cạnh ở tầng 0 +
ít cạnh tầng trên, mỗi cạnh 1 số int 4 byte):

```
RAM_graph ≈ N × 2M × 4 byte
```

**Ví dụ FUFU: 100.000 frame, D = 1024** (dim SigLIP-2 Large — `encoder.py` tự
detect dim lúc khởi tạo, nên nếu đổi model thì thay D tương ứng), `M = 32`:

```
Vector : 100.000 × 1024 × 4  = 409.600.000 byte ≈ 0,41 GB
Đồ thị : 100.000 × 64  × 4   =  25.600.000 byte ≈ 0,026 GB
Tổng   ≈ 0,44 GB
```

Nhận xét: **vector chiếm ~94%**, đồ thị chỉ là phụ phí ~6%. Quy tắc nhẩm:
cứ **1M vector × 1024d ≈ 4,4 GB RAM** — đến mức này laptop vẫn chịu được;
qua chục triệu vector mới phải nghĩ đến PQ (mục 8.2).

Hai lưu ý thực dụng:

- File `data/index.faiss` trên đĩa ≈ đúng con số RAM trên (Flat ghi nguyên
  vector + đồ thị) — nhìn size file là đoán được ngay index đang chứa bao nhiêu
  vector.
- `M` xuất hiện trong cả công thức RAM lẫn chất lượng đồ thị: tăng M=32 → 64
  thì phụ phí đồ thị gấp đôi (vẫn nhỏ so với vector), nhưng thời gian build và
  thời gian mỗi bước search cũng tăng theo — đừng tăng M chỉ vì "RAM còn dư".

---

## 10. Vận hành trong FUFU — 3 điều phải nhớ

### 10.1 `faiss_id` gán tuần tự = `index.ntotal`

FAISS HNSW **không cho tự đặt id** — vector thứ i add vào nhận id = i. FUFU
khai thác luôn điều đó (`storage.py`, hàm `add_frames`):

```python
start_faiss = self.index.ntotal      # id của vector SẮP add
self.index.add(vectors)
# frame thứ i trong batch → faiss_id = start_faiss + i, ghi vào SQLite
```

Cột `frames.faiss_id` (UNIQUE) là cây cầu duy nhất nối "vector thứ mấy trong
FAISS" với "frame nào, video nào, giây thứ mấy". Lúc search, `Retriever`
nhận về list faiss_id từ FAISS rồi JOIN ngược qua `frames_by_faiss_ids()`.

### 10.2 Bất biến đồng bộ FAISS ↔ SQLite

Hệ quả trực tiếp của 10.1 (PROJECT-CONTEXT.md §6): **đừng bao giờ add vector
vào FAISS mà không ghi hàng `frames` tương ứng** (và ngược lại). Chỉ cần lệch 1
vector, TOÀN BỘ mapping phía sau lệch theo (off-by-one dây chuyền) — search vẫn
"chạy" nhưng trả về thumbnail của frame... hàng xóm. Bug loại này không crash,
chỉ âm thầm trả kết quả sai, rất khó lần.

### 10.3 Persist và chuyện "không xoá được"

- **Persist** = `IndexWriter.persist()`: commit SQLite + `faiss.write_index()`
  ghi nguyên index ra file `data/index.faiss`. Ingest video gọi persist **mỗi
  chunk 16 frame** (`chunk_size_frames: 16`) + có signal handler — kill giữa
  chừng chỉ mất tối đa 16 frame cuối, và quan trọng hơn: FAISS với SQLite được
  ghi cùng nhịp nên bất biến 10.2 sống sót qua crash.
- **HNSW không xoá phần tử được một cách tử tế**: xoá 1 node = đục 1 ngã tư
  khỏi mạng đường — các cạnh đi qua nó gãy, đồ thị mất tính dẫn đường; và xoá
  giữa chừng còn phá luôn quy ước id tuần tự ở 10.1. FAISS đơn giản là không
  hỗ trợ `remove_ids` cho `IndexHNSWFlat`.
- **Hệ quả vận hành**: muốn gỡ 1 video đã ingest (nhầm file, dữ liệu hỏng) →
  con đường sạch duy nhất là **xoá data và rebuild index từ đầu**. Đây cũng là
  lý do ingest được thiết kế **idempotent ở mức item** (file đã có frame/asr
  thì skip) — "chữa" bằng cách không ingest trùng, thay vì xoá-rồi-thêm-lại.

---

## 11. Tóm tắt 10 giây

Tìm k vector gần nhất = kNN; brute force O(N×D) chết ở quy mô lớn. ANN đổi
1-5% recall lấy 100× tốc độ. HNSW = đồ thị nhiều tầng kiểu cao-tốc→đường-làng,
search O(log N); ba núm vặn: `M` (độ dày đường nối), `efConstruction` (độ kỹ
khi xây), `efSearch` (độ rộng beam khi tìm — núm tune chính). Vector đã
normalize thì inner product = cosine, nên FUFU dùng `IndexHNSWFlat` +
`METRIC_INNER_PRODUCT`. RAM ≈ N×D×4 byte. `faiss_id` tuần tự = `ntotal`, phải
đồng bộ tuyệt đối với SQLite; HNSW không xoá được → muốn gỡ item thì rebuild.

---

## 12. Câu hỏi tự kiểm tra

**1. Kho 2M vector, D = 1024. Ước lượng số phép tính cho 1 query brute force, và thời gian trên CPU ~10 tỷ phép/giây?**
<details><summary>Đáp án</summary>

`2.000.000 × 2 × 1024 ≈ 4,1 tỷ phép` → ~0,4 giây/query trên 1 core. Dùng được
cho batch eval qua đêm, không dùng được cho search tương tác (~100ms budget) —
đó là lúc cần HNSW.
</details>

**2. Tăng `efSearch` từ 128 lên 512 thì recall, độ trễ query, và thời gian ingest thay đổi thế nào?**
<details><summary>Đáp án</summary>

Recall tăng (nhưng bão hoà — từ ~98-99% lên ~99,5%), độ trễ MỖI query tăng vì
beam rộng hơn phải thăm nhiều node hơn. Thời gian ingest **không đổi** —
`efSearch` chỉ tác dụng lúc search; lúc xây đồ thị là việc của `efConstruction`.
Trong FUFU chỉ cần đổi `hnsw_ef_search` trong settings.yaml và restart backend,
không phải rebuild index.
</details>

**3. Vì sao FUFU dùng `METRIC_INNER_PRODUCT` mà vẫn dám gọi điểm số là cosine? Điều kiện nào phải giữ?**
<details><summary>Đáp án</summary>

Vì `cosine = a·b/(‖a‖‖b‖)`, và khi mọi vector đã L2-normalize (‖·‖ = 1) thì
cosine = a·b = inner product. Điều kiện: **mọi vector add vào index VÀ mọi
query vector đều phải được normalize** — FUFU làm việc này trong
`app/common/encoder.py`. Add một vector chưa normalize là điểm số mất ý nghĩa
cosine một cách thầm lặng.
</details>

**4. "Flat" trong `IndexHNSWFlat` nghĩa là gì, và khi nào nên rời bỏ nó?**
<details><summary>Đáp án</summary>

Flat = giữ vector gốc float32 không nén — similarity tính trên vector thật,
sai số duy nhất là ANN trượt hàng xóm. Rời bỏ khi RAM không chứa nổi
(`N×D×4 byte` lên hàng chục GB, tức chục triệu vector trở lên) → chuyển sang
IVF (chia cluster k-means) và/hoặc PQ (nén vector), chấp nhận thêm sai số.
</details>

**5. Đồng nghiệp viết script "dọn dữ liệu": chạy `index.add()` cho một batch frame nhưng quên ghi vào bảng `frames`. Hậu quả?**
<details><summary>Đáp án</summary>

Mọi vector add SAU đó nhận faiss_id lệch so với hàng `frames` tương ứng
(off-by-N dây chuyền). Search không crash nhưng `frames_by_faiss_ids` trả về
frame sai — kết quả hiển thị thumbnail/timestamp của frame khác. Vi phạm bất
biến §6 PROJECT-CONTEXT. Cách chữa duy nhất sạch sẽ: rebuild index.
</details>

**6. Vì sao tầng cao của HNSW phải có ÍT node? Nếu mọi node đều có mặt ở mọi tầng thì sao?**
<details><summary>Đáp án</summary>

Tầng cao ít node → khoảng cách giữa các node lớn → cạnh nối là các "bước nhảy
xa" giúp băng qua không gian trong O(log N) bước. Nếu mọi node ở mọi tầng thì
mọi tầng đều là "đường làng" dày đặc — cạnh chỉ nối hàng xóm sát nhau, greedy
phải đi từng bước ngắn, mất luôn lợi thế log N (và RAM phình to vô ích). Cùng
logic với skip-list: tầng trên phải thưa thì mới "skip" được.
</details>

**7. FUFU muốn gỡ 1 video ingest nhầm khỏi hệ thống. Liệt kê vì sao không thể "xoá vector của nó khỏi FAISS" và hướng xử lý.**
<details><summary>Đáp án</summary>

(1) `IndexHNSWFlat` không hỗ trợ remove — xoá node làm gãy cạnh đồ thị, mất
tính dẫn đường; (2) kể cả xoá được thì id tuần tự phía sau dồn lại, phá mapping
faiss_id ↔ frames. Hướng xử lý: xoá `data/` (hoặc các hàng SQLite liên quan) và
**rebuild index từ đầu**; phòng ngừa bằng ingest idempotent (file đã ingest thì
skip). Mẹo nhẹ hơn cho nhu cầu "ẩn kết quả": filter item_id ở tầng SQL sau khi
search, chấp nhận vector rác vẫn nằm trong index.
</details>

**8. Trên vector đã normalize, ranking theo inner product và ranking theo L2 có khác nhau không? Chứng minh 1 dòng.**
<details><summary>Đáp án</summary>

Không khác. `‖q−d‖² = ‖q‖² + ‖d‖² − 2q·d = 2 − 2q·d` (vì ‖q‖=‖d‖=1) — L2 là
hàm giảm đơn điệu của inner product, nên d nào có IP cao nhất cũng có L2 nhỏ
nhất. Hệ quả: chọn metric IP hay L2 cho index trên vector normalize chỉ là
chuyện quy ước dấu/đọc điểm số, ranking giống hệt nhau.
</details>

---

## 13. Đọc thêm

- Malkov & Yashunin (2016), *Efficient and robust approximate nearest neighbor
  search using Hierarchical Navigable Small World graphs* — paper gốc HNSW,
  phần hình vẽ rất dễ đọc.
- [FAISS wiki — Guidelines to choose an index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
  — cây quyết định "bao nhiêu vector thì dùng index gì".
- [FAISS wiki — Indexing 1M vectors](https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors) — benchmark recall/tốc độ thật của HNSW vs IVF.
- Pinecone, *Hierarchical Navigable Small Worlds* (series "Faiss: The Missing
  Manual") — giải thích trực quan có hình động.
- Tiếp theo trong giáo trình: **chương 14** — kết quả dense top-500 từ chương
  này được trộn với BM25 như thế nào (`fuse_and_aggregate`).
