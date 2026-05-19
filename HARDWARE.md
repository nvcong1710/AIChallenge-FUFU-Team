# Hardware Profile — Stack multimedia

User target: **3090 24GB** cho ingest, có thể dùng GPU thấp hơn cho online query.

## Ước lượng VRAM theo phase

### Phase INGEST (load tuần tự / share GPU)

| Model | VRAM | Tốc độ trên 3090 |
|---|---|---|
| SigLIP-2 Base 384 (fp16) | 0.4 GB | ~120 img/s |
| PaddleOCR (paddlepaddle-gpu nếu bật) | 1-2 GB | ~5-15 img/s |
| YOLO-World v2 large (fp16) | 1.5 GB | ~30 img/s |
| Qwen2.5-VL-7B INT4 | 5 GB | ~1 img/s (1.5s/frame) |
| PhoWhisper-medium (fp16) | 3 GB | ~30× realtime |
| **Tổng load đồng thời** | **~13 GB** | |

→ 3090 24GB dư ~10GB cho activation + batch.

### Phase ONLINE QUERY (luôn load)

| Model | VRAM |
|---|---|
| SigLIP-2 Base 384 (fp16) | 0.4 GB |
| NLLB-200 distilled 600M (fp16) | 1.3 GB |
| Qwen2.5-3B-Instruct INT4 | 2.5 GB |
| **Tổng online** | **~5 GB** |

→ Online query fit trong **8GB VRAM** dễ dàng. 3060/3070 đủ.

## Profile theo GPU

### 🟢 3090 / 4090 / A5000 (24GB) — Full ingest

```yaml
models:
  device: cuda
extractors:
  enable_ocr: true
  enable_caption: true
  enable_detection: true
  enable_asr: true
  caption_quant_4bit: true     # INT4 — đủ tốt
  asr_model: vinai/PhoWhisper-medium
```

Ingest 1 video 1 phút: ~3-5 phút (bottleneck Qwen-VL caption ~1.5s/frame).

### 🟡 3080 / 4080 (12-16GB) — Bỏ Qwen-VL caption

```yaml
extractors:
  enable_caption: false        # bỏ Qwen-VL-7B
  enable_ocr: true
  enable_detection: true
  enable_asr: true
```

Mất signal caption (semantic description). Vẫn còn OCR + detection + ASR. Mất tầm 5-10% recall trên query semantic mơ hồ.

### 🟡 3060 / 3070 (8-12GB) — Stack tối thiểu

```yaml
extractors:
  enable_caption: false
  enable_detection: false      # bỏ YOLO-World
  enable_ocr: true
  enable_asr: true             # PhoWhisper-small thay medium
  asr_model: vinai/PhoWhisper-small
```

Còn SigLIP + OCR + ASR. Đủ cho baseline ~70-80% recall.

### 🔴 3050 / 1660 (4-6GB) — Online only, ingest CPU

Online query với SigLIP + paraphrase 1.5B (5GB tổng). Ingest chạy CPU:
```yaml
models:
  device: cpu                  # ingest CPU
extractors:
  enable_caption: false
  enable_detection: false
  enable_asr: false            # hoặc bật với asr_model nhỏ
```

Ingest sẽ chậm: 1 video 1 phút mất ~30 phút trên CPU.

## Tốc độ ingest ước lượng (3090, full extractors)

| Bước | Throughput |
|---|---|
| PySceneDetect (CPU) | 5× realtime |
| Keyframe + thumbnail | 10× realtime |
| SigLIP-2 batch encode | 120 img/s |
| PaddleOCR (CPU/GPU) | 5-15 img/s |
| YOLO-World large | 30 img/s |
| **Qwen2.5-VL-7B caption** | **0.7 img/s ← BOTTLENECK** |
| PhoWhisper-medium ASR | 30× realtime |
| FAISS add | 1000 vec/s |

→ Ước lượng 1 video 1 phút (~30 keyframe sau dedup): **~3-4 phút wall time**.
→ 100 giờ video (~60k keyframe): **~24 giờ** trên 1×3090.

## Cách tăng tốc khi quá chậm

1. **Bỏ Qwen-VL** → giảm 90% thời gian ingest, vẫn giữ recall ~80%.
2. **Caption batch 4 frame cùng lúc** (cần sửa code Qwen-VL processor).
3. **Quantize Qwen-VL về INT8 thay INT4** → nhanh hơn 30%, nặng hơn 2×.
4. **Whisper → faster-whisper** (CTranslate2 backend) → nhanh hơn 3-5×.
5. **Multi-process ingest** — chạy 2-4 process song song, mỗi process 1 GPU slot.
6. **TransNetV2 GPU shot detect** thay PySceneDetect CPU → 10× nhanh hơn cho shot.

## Disk

| Component | Size |
|---|---|
| Models cache (`~/.cache/huggingface`) | ~25 GB |
| Models cache (`~/.cache/ultralytics`) | ~0.5 GB |
| FAISS index (1M vec × 768d) | ~3 GB |
| SQLite (~1KB / frame) | nhỏ |
| Thumbnails JPEG q=85 (~30 KB / keyframe) | ~2 GB / 60k frame |

Tổng cho dataset 100 giờ video: **~30 GB data + 25 GB models**.
