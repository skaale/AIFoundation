from __future__ import annotations

import re
import warnings

warnings.filterwarnings(
    "ignore",
    message="The pynvml package is deprecated",
    category=FutureWarning,
)

import numpy as np
import torch
from kokoro import KModel, KPipeline

KOKORO_REPO = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24000

VOICES = {
    "Heart (US female)": "af_heart",
    "Bella (US female)": "af_bella",
    "Nicole (US female)": "af_nicole",
    "Sarah (US female)": "af_sarah",
    "Michael (US male)": "am_michael",
    "Adam (US male)": "am_adam",
    "Emma (UK female)": "bf_emma",
    "George (UK male)": "bm_george",
}

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def edge_tts_package_version() -> str:
    try:
        from importlib.metadata import version

        return version("edge-tts")
    except Exception:
        return "unknown"


def pop_speakable_phrases(buffer: str, *, min_chars: int = 12) -> tuple[list[str], str]:
    stripped = buffer.strip()
    if not stripped:
        return [], ""

    parts = SENTENCE_SPLIT.split(stripped)
    if len(parts) > 1:
        ready = [part.strip() for part in parts[:-1] if len(part.strip()) >= 6]
        return ready, parts[-1]

    if len(stripped) >= min_chars and stripped.endswith((",", ";", ":", "\n")):
        return [stripped], ""

    # Speak in short word groups so voice starts almost immediately.
    if len(stripped) >= 28 and " " in stripped:
        cut = stripped.rfind(" ", 0, 28)
        if cut > 8:
            return [stripped[:cut].strip()], stripped[cut:].strip()

    return [], buffer


class EdgeTTSPriest:
    _fallback_warned = False

    def __init__(
        self,
        voice: str = "da-DK-JeppeNeural",
        speed: float = 1.0,
    ) -> None:
        self.voice = voice
        self.speed = speed
        self._fallback = KokoroTTS(voice="bm_george", speed=speed, use_gpu=True)

    def _synthesize_edge(self, text: str) -> np.ndarray:
        import os
        import tempfile

        import importlib

        import torchaudio

        edge_tts = importlib.import_module("edge_tts")
        rate = f"{int((self.speed - 1.0) * 100):+d}%"
        tmp_path = tempfile.mktemp(suffix=".mp3")
        try:
            edge_tts.Communicate(text, self.voice, rate=rate).save_sync(tmp_path)
            waveform, sample_rate = torchaudio.load(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        audio = waveform[0].detach().cpu().numpy().astype(np.float32)
        if sample_rate != SAMPLE_RATE:
            audio = (
                torchaudio.functional.resample(
                    torch.from_numpy(audio),
                    sample_rate,
                    SAMPLE_RATE,
                )
                .numpy()
                .astype(np.float32)
            )
        return audio

    def stream(self, text: str):
        text = text.strip()
        if not text:
            return

        try:
            audio = self._synthesize_edge(text)
            yield SAMPLE_RATE, audio
        except Exception as exc:
            if not EdgeTTSPriest._fallback_warned:
                print(
                    "WARNING: Danish Edge TTS failed "
                    f"({exc}). Falling back to offline Kokoro voice. "
                    "Run: python -m pip install --upgrade edge-tts"
                )
                EdgeTTSPriest._fallback_warned = True
            yield from self._fallback.stream(text)

    def synthesize(self, text: str) -> tuple[int, np.ndarray]:
        chunks: list[np.ndarray] = []
        for _, audio in self.stream(text):
            chunks.append(audio)
        if not chunks:
            return SAMPLE_RATE, np.array([], dtype=np.float32)
        return SAMPLE_RATE, np.concatenate(chunks)


def make_priest_tts(
    language_code: str,
    voice_label: str,
    speed: float,
):
    from .languages import get_language

    profile = get_language(language_code)
    voice_id = profile.voices.get(voice_label) or next(iter(profile.voices.values()))
    if profile.tts_backend == "edge":
        return EdgeTTSPriest(voice=voice_id, speed=speed)
    return KokoroTTS(voice=voice_id, speed=speed, use_gpu=True)


def warmup_danish_tts(language_code: str, voice_label: str, speed: float) -> None:
    tts = make_priest_tts(language_code, voice_label, speed)
    next(tts.stream("Klar."))


class KokoroTTS:
    _model: KModel | None = None
    _pipelines: dict[str, KPipeline] = {}
    _voice_packs: dict[str, object] = {}

    def __init__(
        self,
        voice: str = "af_heart",
        speed: float = 1.0,
        *,
        use_gpu: bool = True,
    ) -> None:
        self.voice = voice
        self.speed = speed
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self._ensure_model()
        self._ensure_voice()

    @classmethod
    def _ensure_model(cls) -> None:
        if cls._model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._model = KModel(repo_id=KOKORO_REPO).to(device).eval()

    def _ensure_voice(self) -> None:
        lang = self.voice[0]
        if lang not in self._pipelines:
            self._pipelines[lang] = KPipeline(
                lang_code=lang,
                model=False,
                repo_id=KOKORO_REPO,
            )
        if self.voice not in self._voice_packs:
            self._voice_packs[self.voice] = self._pipelines[lang].load_voice(self.voice)

        self.pipeline = self._pipelines[lang]
        self.pack = self._voice_packs[self.voice]

    def stream(self, text: str):
        text = text.strip()
        if not text:
            return

        for _, ps, _ in self.pipeline(text, self.voice, self.speed):
            ref_s = self.pack[len(ps) - 1]
            with torch.no_grad():
                audio = self._model(ps, ref_s, self.speed)
            yield SAMPLE_RATE, audio.numpy()

    def synthesize(self, text: str) -> tuple[int, np.ndarray]:
        chunks: list[np.ndarray] = []
        for _, audio in self.stream(text):
            chunks.append(audio)
        if not chunks:
            return SAMPLE_RATE, np.array([], dtype=np.float32)
        return SAMPLE_RATE, np.concatenate(chunks)
