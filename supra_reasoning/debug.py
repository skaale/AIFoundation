from __future__ import annotations

from datetime import datetime

from .conversation import ConversationState

MAX_DEBUG_LINES = 120


def debug_view(state: ConversationState) -> str:
    if not state.debug_lines:
        return "Debug log (waiting for events)…"
    return "\n".join(state.debug_lines)


def debug_log(
    state: ConversationState,
    message: str,
    level: str = "INFO",
) -> str:
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] [{level}] {message}"
    print(f"ASI-DEBUG {line}", flush=True)
    state.debug_lines.append(line)
    if len(state.debug_lines) > MAX_DEBUG_LINES:
        state.debug_lines = state.debug_lines[-MAX_DEBUG_LINES:]
    return debug_view(state)


def debug_mic_tick(
    state: ConversationState,
    energy: float,
    mic_level: float,
    sample_count: int,
    note: str = "",
) -> str | None:
    state.debug_tick += 1
    important = (
        (state.mic_enabled and state.debug_tick <= 20)
        or state.debug_tick <= 5
        or state.debug_tick % 10 == 0
        or energy >= 0.001
        or state.speaking
        or state.draft_user
        or bool(note)
    )
    if not important:
        return None
    detail = (
        f"mic #{state.debug_tick} energy={energy:.5f} level={mic_level:.3f} "
        f"samples={sample_count} chunks={len(state.speech_chunks)} "
        f"speaking={state.speaking} silence={state.silence_streak} "
        f"mic_on={state.mic_enabled} awaiting={state.awaiting_response}"
    )
    if note:
        detail = f"{detail} | {note}"
    return detail
