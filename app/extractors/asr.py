"""PhoWhisper / Whisper ASR cho audio + audio track của video."""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from ..common.types import ASRSegment


class ASRExtractor:
    def __init__(self, cfg: dict):
        self.enabled = False
        self.pipe = None

        ex_cfg = cfg.get("extractors", {})
        if not ex_cfg.get("enable_asr", True):
            print("[asr] disabled by config.")
            return

        model_name = ex_cfg.get("asr_model", "vinai/PhoWhisper-medium")
        lang = ex_cfg.get("asr_lang", "vi")
        chunk_length = int(ex_cfg.get("asr_chunk_length_sec", 30))

        device = cfg.get("models", {}).get("device", "cuda")
        cuda_ok = device == "cuda" and torch.cuda.is_available()
        dtype = torch.float16 if cuda_ok else torch.float32

        try:
            from transformers import pipeline

            self.pipe = pipeline(
                task="automatic-speech-recognition",
                model=model_name,
                torch_dtype=dtype,
                device=0 if cuda_ok else -1,
                chunk_length_s=chunk_length,
                return_timestamps=True,
            )
            self.lang = lang
            self.enabled = True
        except Exception as e:
            print(f"[asr] init fail: {e}; disabled.")

    def extract(self, audio: np.ndarray, sample_rate: int = 16000) -> List[ASRSegment]:
        if not self.enabled or self.pipe is None or audio.size == 0:
            return []
        try:
            generate_kwargs = {"task": "transcribe"}
            # PhoWhisper là Whisper-based, lang code 'vi' OK
            if self.lang:
                generate_kwargs["language"] = self.lang
            result = self.pipe(
                {"raw": audio, "sampling_rate": sample_rate},
                generate_kwargs=generate_kwargs,
            )
        except Exception as e:
            print(f"[asr] inference fail: {e}")
            return []

        chunks = result.get("chunks") or []
        segments: List[ASRSegment] = []
        for c in chunks:
            ts = c.get("timestamp") or (None, None)
            start = float(ts[0]) if ts[0] is not None else 0.0
            end = float(ts[1]) if ts[1] is not None else start
            text = (c.get("text") or "").strip()
            if text:
                segments.append(ASRSegment(start=start, end=end, text=text))

        # Nếu pipeline trả full text (không chunks), fallback 1 segment
        if not segments and result.get("text"):
            text = result["text"].strip()
            if text:
                segments.append(ASRSegment(start=0.0, end=len(audio) / sample_rate, text=text))

        return segments
