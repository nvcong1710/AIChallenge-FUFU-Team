# Chương 17 — Chỉnh siêu tham số: tune FUFU có phương pháp

---

## 1. Vì sao chương này tồn tại trong FUFU

Mở `config/settings.yaml` và đếm: có **hơn 30 con số** mà ai đó đã gõ vào.
Trọng số hợp nhất `dense: 0.4 / bm25_visual: 0.25 / bm25_asr: 0.5`. Ngưỡng
shot detect `27.0`. Mật độ keyframe `1.0/giây`. `ef_search: 128`. Ai chọn
những con số này? **Một con người, dựa trên cảm tính + vài lần thử tay.**
Không con số nào trong đó đã được kiểm chứng bằng đo đạc có hệ thống.

Điều đó nghĩa là FUFU đang để lại điểm số trên bàn. Kinh nghiệm chung của các
đội thi retrieval (VBS, các challenge tương tự): **chỉ tune trọng số fusion
trên một eval set tử tế đã có thể đổi recall@5 vài điểm phần trăm** — nhiều
khi bằng cả công sức thay một model mới, mà không tốn thêm GB VRAM nào.

Chương này trả lời 3 câu hỏi:

1. FUFU có **những núm vặn nào**, núm nào nhạy, núm nào vặn xong phải
   ingest lại từ đầu (rất đắt)?
2. **Vặn theo thứ tự nào, bằng phương pháp nào** (grid? random? Bayesian?)
   để không lãng phí lần thử?
3. Làm sao biết mình **thật sự tốt lên**, chứ không phải tự lừa mình?

> 🔗 **Trong FUFU:** toàn bộ tham số tune-được tập trung ở
> `config/settings.yaml` (giải thích từng khối trong `PROJECT-CONTEXT.md` §9).
> Một số ngưỡng bị hardcode trong code: `MIN_BM25_RAW = 3.0` ở
> `app/backend/services/retrieval.py:58`, `BM25_SCALE = 8.0` ở
> `app/backend/services/rerank.py:23`, `temperature=0.7` của paraphrase ở
> `app/backend/services/paraphraser.py:62`. Quy trình ghi kết quả mỗi lần
> thử nằm ở bảng tracking trong `RESEARCH-PLAN.md` §6.

---

## 2. Cần biết trước

- **ML cổ điển:** grid search + cross-validation từ sklearn
  (`GridSearchCV`) — bạn đã từng tune `C` của SVM hay `max_depth` của
  random forest. Chương này mở rộng đúng tư duy đó sang hệ retrieval.
- **Chương 13** (FAISS): hiểu `ef_search` là gì.
- **Chương 14** (fusion): hiểu công thức
  `score = w_d·dense + w_v·bm25_visual + w_a·bm25_asr`.
- **Chương 15** (pipeline): phân biệt giai đoạn ingest (offline, đắt) vs
  query (online, rẻ).
- **Chương 19** (eval) định nghĩa recall@K và cách dựng eval set — chương
  này *dùng* các metric đó làm hàm mục tiêu, không dạy lại cách đo.

---

## 3. Tham số vs siêu tham số — và vì sao ở FUFU mọi thứ là siêu tham số

Ôn lại từ ML cổ điển:

| | Tham số (parameters) | Siêu tham số (hyperparameters) |
|---|---|---|
| Ai quyết định | **Model tự học** từ data (gradient descent) | **Người** chọn trước khi chạy |
| Ví dụ sklearn | hệ số `coef_` của LogisticRegression | `C`, `max_depth`, `n_estimators` |
| Ví dụ deep learning | hàng tỷ weights của SigLIP | learning rate, batch size |

FUFU **không train model nào cả** (trừ khi làm LoRA — chương 16). SigLIP,
PhoWhisper, BGE-reranker... đều dùng weights có sẵn. Vậy "tham số học được"
nằm ngoài tầm tay ta.

Nhưng trong một hệ retrieval, khái niệm siêu tham số **mở rộng ra**: nó là
**mọi con số người gõ vào hệ thống** — trọng số fusion, ngưỡng lọc, top-k,
mật độ keyframe... Tất cả đều ảnh hưởng đầu ra, tất cả đều chọn bằng tay,
và do đó tất cả đều **tune được**. Với FUFU, "không gian siêu tham số"
chính là `settings.yaml` (cộng vài hằng số trong code).

Khác biệt quan trọng so với tune SVM: ở sklearn, mỗi lần thử = train lại
model (vài giây). Ở FUFU, **chi phí mỗi lần thử phụ thuộc tham số nằm ở
tầng nào** — đây là điều mục sau làm rõ.

---

## 4. Kiểm kê toàn bộ núm vặn của FUFU

Bảng dưới lấy từ `config/settings.yaml` thật (giá trị hiện tại trong ngoặc).
Cột cuối là điều quyết định chiến lược tune.

### 4.1 Nhóm A — Trọng số fusion (NHẠY NHẤT)

| Tham số | Hiện tại | Tác động | Độ nhạy | Cần re-ingest? |
|---|---|---|---|---|
| `retrieval.weights.dense` | 0.4 | Sức nặng kênh visual SigLIP | ⭐⭐⭐ | ❌ |
| `retrieval.weights.bm25_visual` | 0.25 | Sức nặng OCR + caption + labels | ⭐⭐⭐ | ❌ |
| `retrieval.weights.bm25_asr` | 0.5 | Sức nặng lời thoại | ⭐⭐⭐ | ❌ |

Bộ 0.4/0.25/0.5 hiện tại **chưa từng được kiểm chứng** trên eval set nào —
nó là phỏng đoán "ưu tiên ASR vì query VN hay nhắc lời thoại". Đây là nơi
đáng tune đầu tiên: rẻ (chỉ restart backend), nhạy (đổi trực tiếp thứ hạng
mọi kết quả).

### 4.2 Nhóm B — Top-k các tầng

| Tham số | Hiện tại | Tác động | Độ nhạy | Cần re-ingest? |
|---|---|---|---|---|
| `top_k_dense` | 500 | Số ứng viên FAISS đưa vào fusion | ⭐⭐ (tăng = recall trần cao hơn, chậm hơn chút) | ❌ |
| `top_k_bm25_visual` / `top_k_bm25_asr` | 200 / 200 | Như trên cho 2 kênh text | ⭐ | ❌ |
| `rerank_top_k` | 50 | Bao nhiêu hit được cross-encoder chấm lại | ⭐⭐ (tăng = chính xác hơn nhưng rerank chậm tuyến tính) | ❌ |
| `top_k_final` | 20 | Số kết quả trả về | ⭐ (UI, không ảnh hưởng chất lượng xếp hạng) | ❌ |

### 4.3 Nhóm C — FAISS HNSW (chương 13)

| Tham số | Hiện tại | Tác động | Độ nhạy | Cần re-ingest? |
|---|---|---|---|---|
| `hnsw_ef_search` | 128 | Độ rộng tìm kiếm lúc query: cao = recall cao, chậm hơn | ⭐⭐ | ❌ |
| `hnsw_m` / `hnsw_ef_construct` | 32 / 200 | Cấu trúc đồ thị lúc **build** index | ⭐ | ✅ (phải build lại index) |

### 4.4 Nhóm D — Ngưỡng lọc

| Tham số | Hiện tại | Ở đâu | Tác động | Cần re-ingest? |
|---|---|---|---|---|
| `MIN_BM25_RAW` | 3.0 | **hardcode** `retrieval.py:58` | Hit BM25 raw < 3.0 bị vứt (chống nhiễu 1-token). Cao quá = mất hit thật | ❌ |
| `BM25_SCALE` | 8.0 | **hardcode** `rerank.py:23` | Chia raw BM25 trước khi fuse; tương tác trực tiếp với nhóm A | ❌ |
| `ocr_min_confidence` | 0.4 | settings.yaml | Lọc dòng OCR lúc ingest. Thấp = nhiều text rác vào FTS5 | ✅ |
| `detection_min_confidence` | 0.25 | settings.yaml | Lọc box YOLO lúc ingest | ✅ |
| `shot_detect_threshold` | 27.0 | settings.yaml | Thấp = nhiều shot/segment hơn, cắt mịn hơn | ✅ |

⚠️ `BM25_SCALE` và nhóm A **không độc lập**: nhân đôi `bm25_asr` weight
≈ chia đôi `BM25_SCALE` cho kênh đó. Khi tune, **cố định BM25_SCALE, chỉ
tune weights** — tránh hai núm vặn cùng một thứ.

### 4.5 Nhóm E — Ingest (đắt nhất)

| Tham số | Hiện tại | Tác động | Cần re-ingest? |
|---|---|---|---|
| `keyframe_density_per_sec` | 1.0 (clamp 1–12/shot) | Nhiều frame = recall visual cao hơn, ingest + index to tuyến tính | ✅ |
| `max_segment_len_sec` | 15.0 | Shot dài bị chia nhỏ; ảnh hưởng độ chính xác timestamp | ✅ |
| `chunk_size_frames` | 16 | Chỉ là resilience (persist mỗi 16 frame) — **không phải núm chất lượng, đừng tune** | — |

### 4.6 Nhóm F — Query expansion

| Tham số | Hiện tại | Ở đâu | Tác động | Cần re-ingest? |
|---|---|---|---|---|
| `num_paraphrases` | 3 | settings.yaml | Nhiều = đa dạng hơn nhưng q_vec trung bình bị loãng + chậm | ❌ |
| paraphrase `temperature` | 0.7 | **hardcode** `paraphraser.py:62` | Cao = paraphrase bay bổng (dễ lạc đề), thấp = na ná nhau | ❌ |
| `enable_translation` / `enable_paraphrase` | true/true | settings.yaml | Bật/tắt cả nhánh — đáng làm ablation | ❌ |

### 4.7 Tóm gọn lằn ranh quan trọng nhất

```
KHÔNG cần re-ingest (thử = restart backend, vài giây):
   nhóm A (weights) · B (top-k) · C (ef_search) · D một phần (MIN_BM25_RAW,
   BM25_SCALE) · F (query expansion)

PHẢI re-ingest (thử = chạy lại ingest, hàng GIỜ trên corpus thi):
   nhóm E (keyframe density, segment len) · D phần ingest (ocr/detection
   confidence, shot threshold) · hnsw_m/ef_construct
```

Hệ quả chiến lược: **vắt kiệt nhóm rẻ trước**. Một lần thử weights tốn 2
phút; một lần thử `keyframe_density` tốn nửa ngày. Với cùng budget thời
gian, bạn thử được ~100 cấu hình nhóm A hoặc... 2 cấu hình nhóm E.

---

## 5. Nguyên tắc số 1: không có eval set thì ĐỪNG tune

Đây là nguyên tắc quan trọng nhất chương, nên nó đứng trước mọi thuật toán.

Kịch bản quen thuộc: bạn đổi `bm25_asr: 0.5 → 0.7`, gõ thử "người chơi cờ
vua", thấy kết quả "có vẻ hợp lý hơn", commit. Tuần sau đồng đội gõ 5 query
khác, 3 cái tệ đi. **"Nhìn có vẻ tốt hơn" trên 1-2 query là ảo giác** —
giống hệt việc đánh giá classifier bằng cách nhìn 2 mẫu thay vì đo accuracy
trên test set.

Quy tắc cứng:

1. Mọi phép so sánh A-vs-B phải là **con số (recall@1/5/20 — chương 19)
   trên CÙNG một bộ query** có ground truth, chạy tự động.
2. Trước khi vặn bất kỳ núm nào: chạy baseline, **ghi số vào bảng tracking**
   (`RESEARCH-PLAN.md` §6). Không có số baseline = không biết mình đi lên
   hay đi xuống.
3. Eval set tối thiểu ~50-100 query. Dưới mức đó, chênh lệch 1-2% giữa hai
   cấu hình chỉ là nhiễu (đổi đúng 1 query từ trượt thành trúng đã là
   +1-2%).

FUFU đã có sẵn hạ tầng: `scripts/eval_accuracy.py` chạy trên MSR-VTT đã
dịch sang tiếng Việt. Hàm mục tiêu cho mọi phần còn lại của chương = output
của script này.

---

## 6. Grid search vs Random search

### 6.1 Grid — thứ bạn đã biết, và giới hạn của nó

Từ sklearn bạn quen `GridSearchCV`: liệt kê lưới giá trị, thử hết tổ hợp.
Áp vào 3 weights của FUFU, mỗi weight 5 mức `{0.1, 0.3, 0.5, 0.7, 0.9}`:

- 5 × 5 × 5 = **125 lần chạy eval**. Mỗi eval ~2 phút (100 query × ~1.2s)
  → hơn 4 giờ. Chấp nhận được... nhưng thêm 1 tham số nữa (ef_search, 4
  mức) là 500 lần — bùng nổ tổ hợp, đúng như bạn từng thấy với SVM.

### 6.2 Random search — vì sao "thử bừa" lại thắng "thử đều"

Kết quả kinh điển của Bergstra & Bengio (2012): khi **chỉ vài tham số thật
sự quan trọng**, random search tìm điểm tốt nhanh hơn grid với cùng budget.

Trực giác bằng hình 2D. Giả sử tune 2 tham số, nhưng thực tế chỉ trục
ngang (weights) ảnh hưởng điểm số, trục dọc (top_k_dense) gần như không:

```
GRID 9 điểm (3×3)                 RANDOM 9 điểm
top_k │ ●     ●     ●             top_k │   ●        ●
      │                                 │ ●     ●
      │ ●     ●     ●                   │     ●    ●  ●
      │                                 │  ●      ●
      │ ●     ●     ●                   │
      └──────────────── weights         └──────────────── weights
   chỉ khám phá 3 GIÁ TRỊ            khám phá 9 GIÁ TRỊ
   weights khác nhau                 weights khác nhau
```

Grid 9 điểm nhưng chiếu xuống trục quan trọng chỉ còn **3 giá trị distinct**
— 6 lần thử bị "đốt" vào việc lặp lại cùng weights với top_k khác (mà top_k
chẳng đổi gì). Random 9 điểm cho **9 giá trị weights khác nhau** → quét trục
nhạy mịn gấp 3 lần, miễn phí.

FUFU đúng là trường hợp này: nhóm A nhạy ⭐⭐⭐, phần lớn nhóm B/C nhạy ⭐.
Tune chung một mẻ → **dùng random search, không grid**. (Grid vẫn ổn khi chỉ
tune đúng 1 tham số, ví dụ quét `MIN_BM25_RAW ∈ {1, 2, 3, 4, 5}`.)

Mẹo riêng cho weights: vì điểm cuối **không renormalize**
(PROJECT-CONTEXT.md §8), chỉ *tỷ lệ* giữa 3 weights là đáng kể về mặt xếp
hạng giữa các segment cùng kênh — nhưng vì các kênh khác scale, cứ sample
mỗi weight độc lập trong `[0.1, 1.0]` là đủ, đừng ép tổng = 1.

### 6.3 Bayesian optimization / Optuna — "thử thông minh dần"

Random search **mù**: lần thử thứ 30 không học gì từ 29 lần trước. Bayesian
optimization thì nhớ: nó dựng một model thống kê "vùng nào của không gian
có vẻ hứa hẹn" từ các điểm đã đo, rồi ưu tiên thử vùng hứa hẹn (xen kẽ
thăm dò vùng chưa biết). Trả giá: phức tạp hơn, và lợi thế chỉ rõ khi **mỗi
lần thử đắt** (eval lâu, hoặc tham số re-ingest).

Với FUFU: weights eval rẻ → random 30-50 lần là đủ tốt. Nhưng nếu muốn
tune đồng thời 4-5 tham số, hoặc eval set phình to làm mỗi lần chạy 10
phút, Optuna đáng dùng. Đây là chỗ duy nhất trong chương code mẫu thật sự
cần thiết:

```python
import optuna, yaml
from scripts.eval_accuracy import run_eval   # trả {"recall@5": ...} trên eval set

def objective(trial):
    cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))
    cfg["retrieval"]["weights"] = {
        "dense":       trial.suggest_float("w_dense", 0.1, 1.0),
        "bm25_visual": trial.suggest_float("w_visual", 0.05, 0.8),
        "bm25_asr":    trial.suggest_float("w_asr", 0.1, 1.0),
    }
    yaml.dump(cfg, open("config/settings_trial.yaml", "w", encoding="utf-8"))
    return run_eval(config_path="config/settings_trial.yaml")["recall@5"]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print(study.best_params, study.best_value)
```

Hai lưu ý thực dụng: (1) `get_config()` có `lru_cache` — `run_eval` phải
nhận path config và tạo SearchEngine mới mỗi trial, không dùng process
đang chạy; (2) 10 trial đầu của Optuna về bản chất là random (chưa đủ dữ
liệu để "thông minh") — nếu chỉ định chạy ~20 trial thì random search
thuần cho kết quả gần tương đương mà dễ debug hơn.

---

## 7. Phương pháp luận thực chiến cho FUFU

### 7.1 Thứ tự tune: theo độ nhạy ÷ chi phí

```
Bước 1. Dựng eval set + đo baseline, ghi bảng tracking   (1 lần, bắt buộc)
Bước 2. Nhóm A — weights (random search 30-50 lần)        rẻ, nhạy nhất
Bước 3. Nhóm D-rẻ — MIN_BM25_RAW, rồi ef_search (quét 1D) rẻ, nhạy vừa
Bước 4. Nhóm F — ablation: tắt/bật paraphrase, num_paraphrases ∈ {2,3,5}
Bước 5. Nhóm B — rerank_top_k ∈ {30, 50, 100} (cân latency)
Bước 6. Nhóm E — ingest: CHỈ KHI các bước trên bão hoà, và chỉ thử
        1-2 cấu hình đã cân nhắc kỹ (mỗi lần = re-ingest hàng giờ)
```

### 7.2 Kỷ luật khi tune

- **Một thay đổi mỗi lần** (giữa các *nhóm*). Đổi weights + ef_search +
  num_paraphrases cùng lúc rồi thấy +2% → không biết công của ai, và không
  biết có cấu hình con nào +3% không. (Trong một random search của riêng
  nhóm A thì 3 weights đổi cùng nhau là chủ đích — "một thay đổi" ở đây
  nghĩa là một *thí nghiệm có kiểm soát*.)
- **Ghi log MỌI lần chạy** vào bảng tracking (`RESEARCH-PLAN.md` §6): cấu
  hình, ngày, recall@1/5/20, latency, ghi chú. Lần chạy không ghi lại =
  lần chạy vứt đi — hai tuần sau không ai nhớ 0.55/0.2/0.6 đã thử chưa.
- Cố định mọi nguồn ngẫu nhiên đo được: cùng eval set, cùng index, cùng
  máy. Paraphrase có temperature > 0 → kết quả expand đổi giữa các run;
  khi tune thứ KHÁC ngoài nhóm F, cân nhắc cache kết quả expand_query cho
  eval set để loại nhiễu này.

### 7.3 Overfit eval set — và liều thuốc holdout

Bạn đã biết bệnh này từ ML cổ điển: tune mãi trên validation set thì model
"học thuộc" validation set — điểm validation tăng nhưng điểm test thật
giảm. Tune retrieval cũng y hệt: thử 200 cấu hình trên cùng 100 query, cấu
hình thắng cuối cùng có thể chỉ thắng nhờ **khớp ngẫu nhiên với đặc thù của
100 query đó** (vd eval set tình cờ nhiều query về lời thoại → w_asr bị đẩy
cao quá đà, vào trận gặp query thuần visual là sụp).

Thuốc giống hệt thuốc cũ:

- **Chia hold-out**: tách 20% query (vd 20/100) cất đi, KHÔNG đụng trong
  suốt quá trình tune. Tune trên 80% còn lại. Chỉ ở bước cuối — khi đã chốt
  cấu hình — mới đo 1 LẦN trên hold-out. Nếu dev-set +4% mà hold-out +0.5%
  → bạn đã overfit, kết quả thật là +0.5%.
- Nghi ngờ mọi chênh lệch < ~2% trên eval set 100 query — đó là cỡ nhiễu.
- Eval set càng giống phân phối query thật của cuộc thi càng tốt (chương 19).

### 7.4 Walk-through giả định: tune weights bằng random search 30 lần

Setup: eval set 100 query MSR-VTT-VN, chia 80 dev / 20 hold-out. Sample
mỗi weight đều trong `[0.1, 1.0]`, chạy 30 lần trên dev. Trích 5 dòng từ
bảng tracking (số minh hoạ):

| # | w_dense | w_visual | w_asr | recall@5 (dev) | Ghi chú |
|---|---|---|---|---|---|
| 0 | 0.40 | 0.25 | 0.50 | 41.2% | **baseline hiện tại** |
| 7 | 0.85 | 0.15 | 0.30 | 45.0% | dense cao vọt lên |
| 13 | 0.90 | 0.40 | 0.25 | 44.6% | dense cao, visual cao hơn → xêm xêm #7 |
| 19 | 0.20 | 0.30 | 0.95 | 38.1% | dồn ASR → tệ hơn baseline |
| 26 | 0.80 | 0.10 | 0.45 | **45.4%** | tốt nhất |

Đọc kết quả thế nào — không chỉ nhìn dòng max:

1. **Nhìn cấu trúc của top-5 cấu hình**: nếu tất cả đều có `w_dense ≥ 0.7`,
   kết luận chắc chắn nhất không phải "0.80/0.10/0.45 là số vàng" mà là
   "**w_dense đang bị để thấp quá**" — giả thuyết ban đầu "ưu tiên ASR" sai
   với phân phối query này. Đó mới là tri thức mang đi được.
2. **#7 vs #26 chênh 0.4%** trên 80 query ≈ 0.3 query — nhiễu. Hai cấu hình
   này *hoà*; chọn cái nào cũng được, đừng kể chuyện "26 tốt hơn 7".
3. **Khi nào dừng**: 10 lần thử cuối không ai vượt 45.4% quá biên nhiễu →
   vùng tốt đã bão hoà. Dừng, chốt #26.
4. **Bước cuối bắt buộc**: đo #26 trên 20 query hold-out. Dev +4.2% so với
   baseline, hold-out cho +3.1% → cải thiện là thật (nhỏ hơn chút là bình
   thường). Ghi cả hai số vào tracking, cập nhật `settings.yaml` +
   PROJECT-CONTEXT.md §9, ghi rõ "đã tune ngày X trên eval set Y".

Lưu ý cuối: kết quả tune **gắn với eval set và corpus đã ingest**. Đổi
corpus thi thật, hoặc tắt caption (đổi nội dung kênh bm25_visual) → bảng số
cũ hết hiệu lực, phải tune lại nhóm A (may là nó rẻ).

---

## 8. Tóm tắt 10 giây

FUFU không train nên không có tham số học — nhưng có >30 siêu tham số trong
`settings.yaml`, chia hai loại: **rẻ** (weights, top-k, ef_search, ngưỡng
BM25, query expansion — chỉ restart backend) và **đắt** (keyframe density,
ngưỡng ingest — phải re-ingest hàng giờ). Không có eval set thì đừng tune.
Tune nhóm rẻ-nhạy trước (weights), bằng **random search** (thắng grid khi
chỉ vài tham số nhạy), Optuna khi mỗi lần thử đắt. Một thí nghiệm mỗi lần,
ghi log mọi lần chạy, giữ 20% query hold-out chỉ đo lúc chốt để khỏi
overfit eval set.

---

## 9. Câu hỏi tự kiểm tra

**1. Vì sao nói "trong FUFU, mọi con số trong settings.yaml đều là siêu tham số" trong khi định nghĩa gốc của siêu tham số gắn với việc train model?**

<details><summary>Đáp án</summary>

Định nghĩa cốt lõi của siêu tham số là "con số do NGƯỜI chọn trước, không
do model học từ data". FUFU không train gì nên không có tham số học, nhưng
mọi con số trong settings.yaml (weights, top-k, ngưỡng, density...) đều do
người chọn và đều ảnh hưởng chất lượng đầu ra → thoả định nghĩa và đều
tune được bằng cùng phương pháp luận (eval set + search có hệ thống).
</details>

**2. Bạn muốn thử `keyframe_density_per_sec: 1.0 → 2.0` và `weights.dense: 0.4 → 0.7`. Cái nào thử trước, vì sao?**

<details><summary>Đáp án</summary>

Weights trước. Đổi weights chỉ cần restart backend (~2 phút/lần thử, thử
được hàng chục cấu hình), còn keyframe density yêu cầu re-ingest toàn bộ
corpus (hàng giờ, có thể cả ngày với corpus thi). Nguyên tắc: vắt kiệt
nhóm tham số rẻ-và-nhạy trước, nhóm phải re-ingest để cuối và chỉ thử 1-2
cấu hình đã cân nhắc kỹ.
</details>

**3. Random search 9 lần thử thường tốt hơn grid 3×3 trong tình huống nào, và trực giác là gì?**

<details><summary>Đáp án</summary>

Khi trong các tham số được tune chỉ một (vài) tham số thật sự nhạy. Grid
3×3 chiếu xuống trục nhạy chỉ cho 3 giá trị distinct — 6 lần thử bị lãng
phí vào việc lặp lại cùng giá trị trục nhạy với các giá trị khác nhau của
trục không nhạy. Random cho 9 giá trị distinct trên MỌI trục → quét trục
quan trọng mịn hơn với cùng budget (Bergstra & Bengio 2012).
</details>

**4. Vì sao không nên tune `BM25_SCALE` và `weights.bm25_asr` trong cùng một đợt search?**

<details><summary>Đáp án</summary>

Hai núm này vặn gần như cùng một thứ: điểm kênh ASR sau chuẩn hoá là
`(raw/BM25_SCALE) × w_asr`, nên giảm BM25_SCALE một nửa ≈ tăng đôi w_asr.
Tune cả hai cùng lúc tạo không gian dư thừa (nhiều cặp giá trị cho cùng
hành vi), làm search lâu hội tụ và kết quả khó diễn giải. Cố định
BM25_SCALE = 8.0, chỉ tune weights.
</details>

**5. Cấu hình A đạt recall@5 = 45.0%, cấu hình B đạt 45.4% trên eval set 80 query. Kết luận gì?**

<details><summary>Đáp án</summary>

Hoà. 0.4% trên 80 query tương đương ~0.3 query — nhỏ hơn nhiễu (đổi đúng
1 query trúng/trượt đã là 1.25%). Không được kết luận B tốt hơn A; chọn
cái nào cũng được (hoặc chọn theo tiêu chí phụ như latency). Quy tắc thô:
nghi ngờ mọi chênh lệch dưới ~2% trên eval set cỡ 100 query.
</details>

**6. Tune 200 cấu hình trên 100 query, cấu hình thắng +5% so với baseline. Rủi ro là gì và phòng thế nào?**

<details><summary>Đáp án</summary>

Overfit eval set: sau 200 lần thử, cấu hình thắng có thể chỉ khớp ngẫu
nhiên với đặc thù của 100 query đó (giống tune quá tay trên validation
set ở ML cổ điển). Phòng: tách 20% query làm hold-out KHÔNG đụng tới khi
tune, chỉ đo 1 lần lúc chốt cấu hình. Nếu dev +5% mà hold-out chỉ +0.5%
thì cải thiện thật là ~0.5%.
</details>

**7. Khi nào Optuna đáng dùng thay random search cho FUFU?**

<details><summary>Đáp án</summary>

Khi mỗi lần thử đắt (eval set lớn làm mỗi run mất ~10 phút, hoặc tham số
dính re-ingest) hoặc khi tune đồng thời nhiều tham số (4-5+) — lúc đó việc
"thử thông minh dần" (ưu tiên vùng hứa hẹn dựa trên các điểm đã đo) tiết
kiệm đáng kể số lần thử. Với 3 weights và eval ~2 phút/lần, random 30-50
lần đơn giản hơn và đủ tốt; lưu ý ~10 trial đầu của Optuna về bản chất
cũng là random.
</details>

**8. Sau khi tune xong weights và cập nhật settings.yaml, đội quyết định tắt `enable_caption` để ingest nhanh hơn. Bộ weights vừa tune còn dùng được không?**

<details><summary>Đáp án</summary>

Không tin được nữa. Tắt caption làm kênh bm25_visual đổi nội dung (mất
caption, chỉ còn OCR + labels) → độ mạnh tương đối giữa 3 kênh thay đổi,
bộ weights cũ tune cho phân phối điểm cũ. Kết quả tune luôn gắn với corpus
đã ingest + cấu hình extractor + eval set; đổi một trong ba thứ đó thì phải
đo lại baseline và tune lại nhóm A (may là nhóm này rẻ).
</details>

---

## 10. Đọc thêm

- Bergstra & Bengio, *Random Search for Hyper-Parameter Optimization*
  (JMLR 2012) — paper gốc của lập luận grid-vs-random, hình 1 chính là
  minh hoạ 2D ở mục 6.2.
- Akiba et al., *Optuna: A Next-generation Hyperparameter Optimization
  Framework* (KDD 2019) + docs: <https://optuna.readthedocs.io> — đặc biệt
  phần TPE sampler và pruning.
- scikit-learn User Guide, mục *Tuning the hyper-parameters of an
  estimator* — đối chiếu lại `GridSearchCV`/`RandomizedSearchCV` bạn đã biết.
- Trong repo: `PROJECT-CONTEXT.md` §9 (giải thích từng khối config),
  `RESEARCH-PLAN.md` §6 (quy trình + bảng tracking),
  `scripts/eval_accuracy.py` (hàm mục tiêu).
- Chương 19 (eval) — cách dựng eval set và metric; chương 13 (FAISS) —
  ý nghĩa hình học của `ef_search`; chương 14 — công thức fusion mà nhóm
  A điều khiển.
