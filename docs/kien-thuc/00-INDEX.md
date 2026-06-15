# Giáo trình kiến thức — Team FUFU

> **Mục đích:** trang bị toàn bộ kiến thức để team hiểu và phát triển hệ thống FUFU
> (multimedia search cho HCM AI Challenge 2026).
>
> **Xuất phát điểm của team:** đã nắm ML cổ điển (Random Forest, Decision Tree, SVM,
> Linear/Logistic Regression). **Chưa biết deep learning** — giáo trình dạy từ nền tảng DL
> lên đến từng model đang chạy trong FUFU, và các kỹ năng thực chiến (fine-tune, tune
> siêu tham số, kết hợp model, đánh giá).
>
> **Cách viết:** trực giác trước — công thức tối thiểu — ví dụ số cụ thể — luôn liên hệ
> với ML cổ điển đã biết — code ngắn chỉ khi thật cần — mỗi chương có box
> **"Trong FUFU"** chỉ thẳng kiến thức nằm ở file code nào.

---

## Lộ trình đọc

```
PHẦN I — NỀN TẢNG DEEP LEARNING          PHẦN II — CÁC MODEL TRONG FUFU
01 → 02 → 03 ──┬──> 04 → 05 → 06 ──────> 07 → 08..12 (đọc tùy module phụ trách)
               │
PHẦN III — HỆ RETRIEVAL                  PHẦN IV — KỸ NĂNG THỰC CHIẾN
13 → 14 → 15  (cần 07)                   16 → 17 → 18 → 19  (cần I + III)

PHẦN V — HỆ THI ĐẤU & KỸ THUẬT CUỘC THI
20  (cần 07, 13-15, 19 — đọc cùng RESEARCH-PLAN §1.4)
```

- **Đường tắt cho người gấp:** 01 → 02 → 04 → 07 → 13 → 14 → 15 → 17 → 19.
- Phần II có thể chia nhau đọc theo module mỗi người phụ trách, nhưng **07 là bắt buộc với tất cả** (CLIP/SigLIP là trái tim hệ thống).

## Mục lục

### Phần I — Nền tảng Deep Learning

| # | Chương | Nội dung chính |
|---|---|---|
| 01 | [Từ ML cổ điển sang Neural Network](01-tu-ml-co-dien-sang-neural-network.md) | Perceptron, MLP, activation, vì sao NN tổng quát hơn SVM/RF |
| 02 | [Huấn luyện mạng neural](02-huan-luyen-mang-neural.md) | Loss, gradient descent, backprop, optimizer (Adam), regularization, overfitting |
| 03 | [CNN — mạng xử lý ảnh](03-cnn-xu-ly-anh.md) | Convolution, pooling, feature map, ResNet |
| 04 | [Attention & Transformer](04-attention-va-transformer.md) | Self-attention, multi-head, positional encoding, kiến trúc Transformer |
| 05 | [Tokenization, BERT vs GPT](05-tokenization-bert-gpt.md) | Token, embedding, encoder vs decoder, pretrain/finetune paradigm |
| 06 | [Vision Transformer (ViT)](06-vision-transformer.md) | Patch embedding, ảnh thành chuỗi token, ViT vs CNN |

### Phần II — Các model trong FUFU

| # | Chương | Model trong FUFU |
|---|---|---|
| 07 | [Contrastive learning & CLIP/SigLIP](07-contrastive-learning-clip-siglip.md) | `google/siglip2-large-patch16-384` — trái tim dense retrieval |
| 08 | [VLM — Qwen-VL & quantization](08-vlm-qwen-vl-quantization.md) | `Qwen2.5-VL-7B` INT4 — caption tiếng Việt |
| 09 | [ASR — PhoWhisper](09-asr-phowhisper.md) | `vinai/PhoWhisper-medium` — lời thoại |
| 10 | [OCR & open-vocab detection](10-ocr-va-open-vocab-detection.md) | EasyOCR, YOLO-World v2 |
| 11 | [LLM sinh văn bản: dịch & paraphrase](11-llm-sinh-van-ban-dich-paraphrase.md) | NLLB-200, Qwen2.5-3B — query expansion |
| 12 | [Bi-encoder vs Cross-encoder](12-bi-encoder-cross-encoder-rerank.md) | BGE-reranker-v2-m3 — rerank |

### Phần III — Hệ retrieval

| # | Chương | Nội dung chính |
|---|---|---|
| 13 | [Vector search: FAISS & HNSW](13-vector-search-faiss-hnsw.md) | Cosine/inner product, ANN, HNSW, tham số ef/M |
| 14 | [BM25 & hybrid fusion](14-bm25-hybrid-fusion.md) | BM25, FTS5, chuẩn hoá điểm, hợp nhất đa kênh |
| 15 | [Pipeline FUFU end-to-end](15-pipeline-fufu-end-to-end.md) | Ráp toàn bộ kiến thức vào code thật — đọc cùng PROJECT-CONTEXT.md |

### Phần IV — Kỹ năng thực chiến

| # | Chương | Nội dung chính |
|---|---|---|
| 16 | [Fine-tuning: full FT vs LoRA/PEFT](16-fine-tuning-lora-peft.md) | Khi nào cần finetune, LoRA, dataset, finetune retrieval |
| 17 | [Chỉnh siêu tham số](17-hyperparameter-tuning.md) | Tham số nào của FUFU tune được, grid/random/Optuna, eval-driven |
| 18 | [Kết hợp model: ensemble & fusion](18-ensemble-fusion.md) | Score/rank fusion, RRF, multi-encoder ensemble |
| 19 | [Đánh giá hệ retrieval](19-danh-gia-retrieval-eval.md) | Recall@K, MRR, nDCG, tự xây eval set theo format thi |

### Phần V — Hệ thi đấu & kỹ thuật cuộc thi

| # | Chương | Nội dung chính |
|---|---|---|
| 20 | [Hệ truy xuất tương tác & kỹ thuật thi đấu (VBS/LSC)](20-he-truy-xuat-tuong-tac-vbs-lsc.md) | vitrivr, lifeXplore, MEMORIA; nhiều cửa truy vấn, temporal `<`, SOM browsing, graph DB; bài học VBS |

---

## Quy ước chung của giáo trình

1. **Tiếng Việt**, thuật ngữ tiếng Anh giữ nguyên kèm giải thích lần đầu xuất hiện.
2. Mỗi chương mở đầu bằng **"Vì sao chương này tồn tại trong FUFU"** và kết thúc bằng **câu hỏi tự kiểm tra + tài liệu đọc thêm**.
3. Box `> 🔗 **Trong FUFU:**` = chỉ dẫn kiến thức này nằm ở file/config nào trong repo.
4. Công thức chỉ xuất hiện khi không thể tránh; luôn có ví dụ số đi kèm.
5. Code mẫu ngắn, chỉ khi thật cần để hiểu — không phải để copy chạy.

## Liên kết tài liệu dự án

- [PROJECT-CONTEXT.md](../../PROJECT-CONTEXT.md) — hệ thống FUFU như code đang chạy (đọc cùng chương 15)
- [RESEARCH-PLAN.md](../../RESEARCH-PLAN.md) — menu ý tưởng nâng cấp (cần chương 16-19 để thực thi)
