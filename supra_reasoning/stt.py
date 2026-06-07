from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message="The pynvml package is deprecated",
    category=FutureWarning,
)

import numpy as np
import torch
from transformers import pipeline

from .conversation import normalize_audio

WHISPER_MODEL = "openai/whisper-base"


class SpeechToText:
    _pipeline = None

    def __init__(self, model_id: str = WHISPER_MODEL, device: str = "auto") -> None:
        self.model_id = model_id
        if device == "auto":
            self.device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = 0 if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == 0 else torch.float32

    @classmethod
    def _get_pipeline(cls, model_id: str, device, dtype):
        if cls._pipeline is None:
            cls._pipeline = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                torch_dtype=dtype,
                device=device,
            )
        return cls._pipeline

    def transcribe(self, audio, *, language: str = "en") -> str:
        if audio is None:
            return ""

        if isinstance(audio, tuple) and len(audio) == 2:
            sample_rate, data = audio
            data = normalize_audio(np.asarray(data))
            audio = {"array": data, "sampling_rate": int(sample_rate)}

        asr = self._get_pipeline(self.model_id, self.device, self.dtype)
        result = asr(
            audio,
            generate_kwargs={"language": language or "en", "task": "transcribe"},
            return_timestamps=False,
        )
        text = result.get("text", "") if isinstance(result, dict) else str(result)
        return text.strip()
