from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


class SiglipEncoder:
    """Wrapper cho SigLIP-2 dùng cho cả encode image (ingest) và text (query)."""

    def __init__(self, model_name: str, device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            print("[encoder] CUDA không có sẵn, fallback CPU.")
            device = "cpu"
        self.device = device
        dtype = torch.float16 if device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = (
            AutoModel.from_pretrained(model_name, torch_dtype=dtype)
            .to(device)
            .eval()
        )

        # Detect output dim từ 1 lượt encode text test
        with torch.inference_mode():
            probe = self.processor(
                text=["probe"],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=64,
            ).to(self.device)
            feat = self.model.get_text_features(**probe)
            self.dim = int(feat.shape[-1])

    @torch.inference_mode()
    def encode_images(self, images: Iterable[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """images: iterable of HxWx3 RGB uint8 arrays. Returns (N, D) float32 L2-normalized."""
        images = list(images)
        all_vecs: list[np.ndarray] = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            pil = [Image.fromarray(img) for img in batch]
            inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
            feats = self.model.get_image_features(**inputs)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            all_vecs.append(feats.cpu().float().numpy())
        if not all_vecs:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.vstack(all_vecs).astype(np.float32)

    @torch.inference_mode()
    def encode_text(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        """texts: iterable of strings. Returns (N, D) float32 L2-normalized."""
        texts = [t for t in texts if t and t.strip()]
        all_vecs: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=64,
            ).to(self.device)
            feats = self.model.get_text_features(**inputs)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            all_vecs.append(feats.cpu().float().numpy())
        if not all_vecs:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.vstack(all_vecs).astype(np.float32)
