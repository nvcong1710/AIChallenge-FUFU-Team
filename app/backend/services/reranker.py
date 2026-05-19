"""BGE-reranker-v2-m3 cross-encoder — rerank top-K hits sau hybrid fuse.

Cách dùng: build text passage cho mỗi hit (caption + OCR + ASR + labels), gửi
cùng query qua cross-encoder, lấy lại relevance score thực, sắp xếp lại.

Cost: ~5ms/passage trên 3090, ~50ms/passage trên CPU.
Effort: rerank top-50 → return top-K cuối cùng.
"""

from __future__ import annotations

from typing import List

import torch


class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cuda"):
        self.enabled = False
        self.tokenizer = None
        self.model = None
        if device == "cuda" and not torch.cuda.is_available():
            print("[rerank] CUDA không sẵn → fallback CPU (sẽ chậm).")
            device = "cpu"
        self.device = device
        dtype = torch.float16 if device == "cuda" else torch.float32

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = (
                AutoModelForSequenceClassification.from_pretrained(
                    model_name, torch_dtype=dtype
                )
                .to(device)
                .eval()
            )
            self.enabled = True
        except Exception as e:
            print(f"[rerank] init fail: {e}; disabled.")

    @torch.inference_mode()
    def rerank(self, query: str, passages: List[str], top_k: int | None = None) -> List[int]:
        """Returns indices của passages đã sắp xếp theo relevance giảm dần."""
        if not self.enabled or self.model is None or not passages:
            return list(range(len(passages)))[:top_k] if top_k else list(range(len(passages)))

        pairs = [(query, p[:512]) for p in passages]  # cap length tránh OOM
        inputs = self.tokenizer(
            pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        logits = self.model(**inputs).logits.view(-1).float()
        scores = logits.cpu().numpy()
        order = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)
        if top_k is not None:
            order = order[:top_k]
        return order
