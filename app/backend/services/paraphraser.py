from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SYSTEM = "Bạn là trợ lý tạo các cách diễn đạt khác nhau cho truy vấn tìm kiếm video tiếng Việt."

USER_TEMPLATE = """Cho truy vấn sau, sinh {n} cách diễn đạt khác mà người Việt thường dùng để mô tả CÙNG cảnh đó. Mỗi cách trên 1 dòng, KHÔNG đánh số, KHÔNG giải thích, KHÔNG thêm dấu gạch đầu dòng. Mỗi diễn đạt ngắn gọn, tự nhiên, sát nghĩa gốc.

Truy vấn gốc: {q}

{n} cách diễn đạt khác:"""


def _clean_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^[\-\*\d\.\)\:\>]+\s*", "", line)
    return line.strip(" .\"'“”")


class Paraphraser:
    """Qwen2.5-3B-Instruct INT4 cho query paraphrase tiếng Việt."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", device: str = "cuda"):
        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "Paraphraser yêu cầu CUDA (bitsandbytes INT4). Tắt query_expansion.enable_paraphrase nếu chạy CPU."
            )
        self.device = device
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb,
            device_map="auto",
        ).eval()

    @torch.inference_mode()
    def paraphrase(self, query: str, n: int = 3, max_new_tokens: int = 120) -> list[str]:
        if not query or not query.strip():
            return []
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TEMPLATE.format(n=n, q=query.strip())},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.decode(
            out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        candidates = [
            _clean_line(l)
            for l in response.split("\n")
            if l.strip() and _clean_line(l)
        ]
        # Loại trùng + loại bản gốc
        seen = {query.strip().lower()}
        out_lines: list[str] = []
        for c in candidates:
            cl = c.lower()
            if cl and cl not in seen:
                seen.add(cl)
                out_lines.append(c)
            if len(out_lines) >= n:
                break
        return out_lines
