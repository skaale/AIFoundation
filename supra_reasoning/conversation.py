from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SPEECH_THRESHOLD = 0.0015
INTERRUPT_THRESHOLD = 0.007
SILENCE_CHUNKS_TO_END = 2
MIN_SPEECH_CHUNKS = 1
STREAM_CHUNK_SECONDS = 0.5


def normalize_audio(data: np.ndarray) -> np.ndarray:
    audio = np.asarray(data, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak > 1.5:
        audio = audio / 32768.0
    elif peak > 1.0:
        audio = audio / peak
    return audio


@dataclass
class ConversationState:
    history: list[dict] = field(default_factory=list)
    speech_chunks: list[list[float]] = field(default_factory=list)
    sample_rate: int = 48000
    silence_streak: int = 0
    speaking: bool = False
    awaiting_response: bool = False
    interrupted: bool = False
    introduced: bool = False
    introducing: bool = False
    idle_chunks: int = 0
    draft_user: bool = False
    pending_turn: bool = False
    pending_audio: tuple[int, list[float]] | None = None
    mic_enabled: bool = False
    circle_html: str = ""
    debug_lines: list[str] = field(default_factory=list)
    debug_tick: int = 0
    mic_seen: bool = False

    def reset_utterance(self) -> None:
        self.speech_chunks = []
        self.silence_streak = 0
        self.speaking = False

    def combined_audio(self) -> tuple[int, np.ndarray] | None:
        if not self.speech_chunks:
            return None
        chunks = [
            normalize_audio(np.asarray(chunk, dtype=np.float32))
            for chunk in self.speech_chunks
        ]
        audio = np.concatenate(chunks).astype(np.float32)
        if audio.size == 0:
            return None
        return self.sample_rate, audio


def chunk_energy(chunk: np.ndarray) -> float:
    data = normalize_audio(chunk)
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(data * data)))


def _chunks_overlap(prev: np.ndarray, current: np.ndarray, atol: float = 0.02) -> bool:
    overlap = min(prev.size, current.size)
    if overlap < 32:
        return False
    return bool(np.allclose(prev[:overlap], current[:overlap], atol=atol))


def store_speech_chunk(state: ConversationState, chunk: np.ndarray) -> None:
    """Keep full utterances across Gradio's incremental stream packets."""
    current = normalize_audio(chunk)
    if current.size == 0:
        return

    chunk_list = current.astype(np.float32).tolist()
    if not state.speech_chunks:
        state.speech_chunks.append(chunk_list)
        return

    prev = normalize_audio(np.asarray(state.speech_chunks[-1], dtype=np.float32))

    # Some clients resend the whole recording-so-far in one packet.
    if current.size > prev.size and _chunks_overlap(prev, current):
        state.speech_chunks[-1] = chunk_list
        return

    if current.size > int(state.sample_rate * 0.9) and current.size >= prev.size:
        state.speech_chunks = [chunk_list]
        return

    state.speech_chunks.append(chunk_list)


def ingest_stream_chunk(
    audio: tuple[int, np.ndarray] | None,
    state: ConversationState,
) -> tuple[ConversationState, bool]:
    """Returns updated state and whether the user's turn has ended."""
    if state.introducing or not state.introduced:
        return state, False

    if not state.mic_enabled:
        return state, False

    if state.interrupted and not state.speaking:
        return state, False

    if state.awaiting_response:
        return state, False

    if audio is None:
        return state, False

    sample_rate, data = audio
    chunk = normalize_audio(data)
    if chunk.size == 0:
        return state, False

    state.sample_rate = int(sample_rate)
    energy = chunk_energy(chunk)

    if energy >= SPEECH_THRESHOLD:
        state.speaking = True
        state.silence_streak = 0
        state.idle_chunks = 0
        store_speech_chunk(state, chunk)
        return state, False

    if not state.speaking:
        if state.introduced:
            state.idle_chunks += 1
        return state, False

    state.silence_streak += 1
    store_speech_chunk(state, chunk)

    if (
        state.silence_streak >= SILENCE_CHUNKS_TO_END
        and len(state.speech_chunks) >= MIN_SPEECH_CHUNKS
    ):
        state.speaking = False
        return state, True

    return state, False
