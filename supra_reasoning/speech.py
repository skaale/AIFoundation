from __future__ import annotations

import re

import numpy as np

from .tts import SAMPLE_RATE, KokoroTTS, pop_speakable_phrases

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_for_speech(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    parts = SENTENCE_SPLIT.split(text)
    phrases: list[str] = []
    buffer = ""

    for part in parts:
        buffer = f"{buffer} {part}".strip() if buffer else part.strip()
        ready, buffer = pop_speakable_phrases(buffer, min_chars=8)
        phrases.extend(ready)

    if buffer.strip():
        phrases.append(buffer.strip())

    return phrases


def stream_priest_voice(tts: KokoroTTS, text: str):
    spoken: list[np.ndarray] = []
    for phrase in split_for_speech(text):
        for _, chunk in tts.stream(phrase):
            spoken.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
            audio = np.concatenate(spoken)
            if audio.size > 0:
                yield SAMPLE_RATE, audio


def captioned_priest_voice(
    tts: KokoroTTS,
    text: str,
    words_per_step: int = 2,
):
    """Stream priest audio while revealing caption text in small word groups."""
    words = text.split()
    if not words:
        return

    revealed = 0
    for playback in stream_priest_voice(tts, text):
        revealed = min(len(words), revealed + words_per_step)
        caption = " ".join(words[:revealed])
        yield playback, caption
