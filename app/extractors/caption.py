"""Qwen2.5-VL-7B captioning + scene description extractor.

Trên 3090 24GB:
- INT4 (bitsandbytes nf4): ~5GB VRAM, ~0.8-1.5s/frame
- bf16:                   ~14GB VRAM, ~0.4-0.7s/frame, chất lượng cao hơn
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from ..common.types import FrameAnnotation


DEFAULT_PROMPT = (
    "Mô tả ngắn gọn (1-2 câu) nội dung chính của ảnh bằng tiếng Việt: "
    "đối tượng nổi bật, hành động, bối cảnh, văn bản trên màn nếu có."
)


class CaptionExtractor:
    def __init__(self, cfg: dict):
        self.enabled = False
        self.model = None
        self.processor = None

        ex_cfg = cfg.get("extractors", {})
        if not ex_cfg.get("enable_caption", True):
            print("[caption] disabled by config.")
            return

        self.prompt = ex_cfg.get("caption_prompt", DEFAULT_PROMPT)
        self.max_tokens = int(ex_cfg.get("caption_max_tokens", 96))
        model_name = ex_cfg.get("caption_model", "Qwen/Qwen2.5-VL-7B-Instruct")
        use_4bit = bool(ex_cfg.get("caption_quant_4bit", True))

        device = cfg.get("models", {}).get("device", "cuda")
        if device != "cuda" or not torch.cuda.is_available():
            print("[caption] yêu cầu CUDA; disabled.")
            return

        try:
            from transformers import AutoProcessor

            try:
                from transformers import Qwen2_5_VLForConditionalGeneration  # transformers >= 4.49
                cls_load = Qwen2_5_VLForConditionalGeneration
            except ImportError:
                # fallback Qwen-VL 2 nếu transformers cũ
                from transformers import Qwen2VLForConditionalGeneration
                cls_load = Qwen2VLForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(model_name)

            if use_4bit:
                from transformers import BitsAndBytesConfig
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                self.model = cls_load.from_pretrained(
                    model_name, quantization_config=bnb, device_map="auto"
                ).eval()
            else:
                self.model = cls_load.from_pretrained(
                    model_name, torch_dtype=torch.bfloat16, device_map="auto"
                ).eval()

            self.enabled = True
        except Exception as e:
            print(f"[caption] init fail: {e}; disabled.")

    @torch.inference_mode()
    def extract(self, image_rgb: np.ndarray) -> str:
        if not self.enabled or self.model is None:
            return ""
        try:
            pil = Image.fromarray(image_rgb)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text], images=[pil], padding=True, return_tensors="pt"
            ).to(self.model.device)
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )
            trimmed = out[:, inputs.input_ids.shape[1] :]
            caption = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            return caption.strip()
        except Exception as e:
            print(f"[caption] inference fail: {e}")
            return ""

    def annotate(self, image_rgb: np.ndarray, annotation: FrameAnnotation) -> None:
        annotation.caption = self.extract(image_rgb)
