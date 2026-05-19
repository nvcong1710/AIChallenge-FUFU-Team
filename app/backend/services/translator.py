from __future__ import annotations

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Translator:
    """NLLB-200 wrapper cho VI ↔ EN."""

    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M", device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        dtype = torch.float16 if device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype)
            .to(device)
            .eval()
        )

    @torch.inference_mode()
    def translate(
        self,
        text: str,
        src_lang: str = "vie_Latn",
        tgt_lang: str = "eng_Latn",
        max_length: int = 128,
        num_beams: int = 2,
    ) -> str:
        if not text or not text.strip():
            return ""
        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        forced_bos = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        out = self.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            max_length=max_length,
            num_beams=num_beams,
        )
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
