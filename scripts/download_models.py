"""Pre-download tất cả model HuggingFace để chạy offline. Chạy 1 lần sau khi cài deps."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

CFG = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / "config" / "settings.yaml").read_text(encoding="utf-8")
)


def dl(name: str, fn) -> None:
    print(f"\n→ {name}")
    try:
        fn()
        print("  ✓")
    except Exception as e:
        print(f"  ⚠ fail: {e}")


def main():
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    siglip = CFG["models"]["siglip"]
    dl(f"SigLIP-2 {siglip}", lambda: (
        AutoProcessor.from_pretrained(siglip),
        AutoModel.from_pretrained(siglip, torch_dtype=torch.float16),
    ))

    trans = CFG["models"]["translator"]
    dl(f"NLLB translator {trans}", lambda: (
        AutoTokenizer.from_pretrained(trans),
        AutoModelForSeq2SeqLM.from_pretrained(trans, torch_dtype=torch.float16),
    ))

    para = CFG["models"]["paraphraser"]
    def _dl_para():
        AutoTokenizer.from_pretrained(para)
        if torch.cuda.is_available():
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            AutoModelForCausalLM.from_pretrained(para, quantization_config=bnb, device_map="auto")
    dl(f"Paraphraser {para}", _dl_para)

    cap = CFG["extractors"]["caption_model"]
    def _dl_cap():
        AutoProcessor.from_pretrained(cap)
        if torch.cuda.is_available():
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    cap, quantization_config=bnb, device_map="auto"
                )
            except ImportError:
                from transformers import Qwen2VLForConditionalGeneration
                Qwen2VLForConditionalGeneration.from_pretrained(
                    cap, quantization_config=bnb, device_map="auto"
                )
    dl(f"Caption VLM {cap}", _dl_cap)

    asr = CFG["extractors"]["asr_model"]
    dl(f"ASR {asr}", lambda: __import__("transformers").pipeline(
        task="automatic-speech-recognition",
        model=asr,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ))

    det = CFG["extractors"]["detection_model"]
    def _dl_det():
        from ultralytics import YOLOWorld
        YOLOWorld(det)
    dl(f"YOLO-World {det}", _dl_det)

    print("\n=== Xong tất cả model ===")


if __name__ == "__main__":
    main()
