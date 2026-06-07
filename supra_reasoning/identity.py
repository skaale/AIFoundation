from __future__ import annotations

import re

_NAME_PREFIX = re.compile(
    r"(?:my name is|i am|i'?m|call me|this is|it'?s|name'?s|they call me)\s+"
    r"([A-Za-z][A-Za-z'-]{0,30})",
    re.IGNORECASE,
)
_NAME_ONLY = re.compile(r"^([A-Za-z][A-Za-z'-]{0,30})$")
_SKIP_WORDS = {
    "my",
    "name",
    "is",
    "the",
    "a",
    "an",
    "uh",
    "um",
    "hi",
    "hello",
    "yes",
    "yeah",
    "call",
    "me",
    "it's",
    "im",
    "i",
}


def extract_user_name(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""

    match = _NAME_PREFIX.search(raw)
    if match:
        return _clean_name(match.group(1))

    match = _NAME_ONLY.match(raw.strip())
    if match:
        return _clean_name(match.group(1))

    for word in re.findall(r"[A-Za-z]+", raw):
        if word.lower() not in _SKIP_WORDS and len(word) >= 2:
            return _clean_name(word)

    return ""


def _clean_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z'-]", "", name).strip("' -")
    if not cleaned:
        return ""
    return cleaned[0].upper() + cleaned[1:].lower()
