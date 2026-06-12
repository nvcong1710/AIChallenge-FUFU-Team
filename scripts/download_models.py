"""Pre-download mọi model HuggingFace cần — full stack v2.

Models (ước lượng total ~22-25GB):
  1) SigLIP-2 Large 384       ~1.5GB  (upgrade từ Base)
  2) NLLB-200 distilled 600M  ~2.5GB
  3) Qwen2.5-3B paraphrase    ~6GB  (INT4 chạy time)
  4) Qwen2.5-VL-7B caption    ~14GB → INT4 ~5GB chạy time
  5) PhoWhisper-medium ASR    ~3GB
  6) BGE-reranker-v2-m3       ~2.5GB
  7) YOLO-World v8l           ~0.5GB
"""

import time

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)


def banner(msg):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}", flush=True)


t0 = time.time()

banner("1/7 SigLIP-2 Large 384")
t = time.time()
AutoProcessor.from_pretrained("google/siglip2-large-patch16-384", use_fast=True)
AutoModel.from_pretrained("google/siglip2-large-patch16-384", torch_dtype=torch.float16)
print(f"  ✓ {time.time()-t:.0f}s", flush=True)

banner("2/7 NLLB-200 translator")
t = time.time()
AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", torch_dtype=torch.float16)
print(f"  ✓ {time.time()-t:.0f}s", flush=True)

banner("3/7 Qwen2.5-3B paraphraser (INT4)")
t = time.time()
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization_config=bnb,
    device_map="auto",
)
print(f"  ✓ {time.time()-t:.0f}s", flush=True)

banner("4/7 Qwen2.5-VL-7B caption (INT4)")
t = time.time()
AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
bnb_vl = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        quantization_config=bnb_vl,
        device_map="auto",
    )
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration
    Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        quantization_config=bnb_vl,
        device_map="auto",
    )
print(f"  ✓ {time.time()-t:.0f}s", flush=True)

banner("5/7 PhoWhisper-medium ASR")
t = time.time()
pipeline(
    "automatic-speech-recognition",
    model="vinai/PhoWhisper-medium",
    torch_dtype=torch.float16,
    device=0,
)
print(f"  ✓ {time.time()-t:.0f}s", flush=True)

banner("6/7 BGE-reranker-v2-m3 (cross-encoder)")
t = time.time()
AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
AutoModelForSequenceClassification.from_pretrained(
    "BAAI/bge-reranker-v2-m3", torch_dtype=torch.float16
)
print(f"  ✓ {time.time()-t:.0f}s", flush=True)

banner("7/7 YOLO-World v8l")
t = time.time()
from ultralytics import YOLOWorld
YOLOWorld("yolov8l-world.pt")
print(f"  ✓ {time.time()-t:.0f}s", flush=True)

# EasyOCR (sẽ tự tải models lần extract đầu, có thể prefetch ở đây)
banner("Bonus: EasyOCR VN reader (prefetch)")
t = time.time()
try:
    import easyocr
    easyocr.Reader(["vi", "en"], gpu=True, verbose=False)
    print(f"  ✓ {time.time()-t:.0f}s")
except Exception as e:
    print(f"  ⚠ skip: {e}")

print(f"\n=== TỔNG: {time.time()-t0:.0f}s ===", flush=True)
