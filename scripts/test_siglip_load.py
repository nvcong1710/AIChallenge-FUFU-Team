"""Quick test xem SigLIP-2 load được không với config hiện tại."""
import sys
import torch

# Test 1: AutoProcessor
print("Test 1: AutoProcessor")
try:
    from transformers import AutoProcessor
    p = AutoProcessor.from_pretrained("google/siglip2-base-patch16-384", use_fast=True)
    print(f"  ✓ AutoProcessor OK: {type(p).__name__}")
except Exception as e:
    print(f"  ✗ {type(e).__name__}: {e}")

# Test 2: AutoTokenizer
print("\nTest 2: AutoTokenizer")
try:
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained("google/siglip2-base-patch16-384")
    print(f"  ✓ AutoTokenizer OK: {type(t).__name__}")
except Exception as e:
    print(f"  ✗ {type(e).__name__}: {e}")

# Test 3: AutoImageProcessor separate
print("\nTest 3: AutoImageProcessor (separate)")
try:
    from transformers import AutoImageProcessor
    ip = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-384")
    print(f"  ✓ AutoImageProcessor OK: {type(ip).__name__}")
except Exception as e:
    print(f"  ✗ {type(e).__name__}: {e}")

# Test 4: Model
print("\nTest 4: AutoModel")
try:
    from transformers import AutoModel
    m = AutoModel.from_pretrained("google/siglip2-base-patch16-384", torch_dtype=torch.float16)
    print(f"  ✓ Model OK: {type(m).__name__}")
except Exception as e:
    print(f"  ✗ {type(e).__name__}: {e}")
