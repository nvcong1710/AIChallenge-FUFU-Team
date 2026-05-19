"""Download chỉ những model cần cho config hiện tại (skip Qwen-VL caption)."""

import sys
import time
import torch
from transformers import (
    AutoModel, AutoProcessor, AutoTokenizer,
    AutoModelForSeq2SeqLM, AutoModelForCausalLM,
    BitsAndBytesConfig, pipeline,
)

def banner(msg):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}", flush=True)

t0 = time.time()

banner("1/5 SigLIP-2 Base 384")
AutoProcessor.from_pretrained("google/siglip2-base-patch16-384")
AutoModel.from_pretrained("google/siglip2-base-patch16-384", torch_dtype=torch.float16)
print(f"  ✓ {time.time()-t0:.0f}s", flush=True)

banner("2/5 NLLB-200 translator")
t = time.time()
AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", torch_dtype=torch.float16)
print(f"  ✓ {time.time()-t:.0f}s", flush=True)

banner("3/5 Qwen2.5-3B paraphraser (INT4)")
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

banner("4/5 PhoWhisper-medium ASR")
t = time.time()
pipeline(
    "automatic-speech-recognition",
    model="vinai/PhoWhisper-medium",
    torch_dtype=torch.float16,
    device=0,
)
print(f"  ✓ {time.time()-t:.0f}s", flush=True)

banner("5/5 YOLO-World v8l")
t = time.time()
from ultralytics import YOLOWorld
YOLOWorld("yolov8l-world.pt")
print(f"  ✓ {time.time()-t:.0f}s", flush=True)

print(f"\n=== TỔNG: {time.time()-t0:.0f}s ===", flush=True)
