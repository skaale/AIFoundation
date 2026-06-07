from __future__ import annotations

import re

_NAME_PREFIX = re.compile(
    r"(?:my name is|i am|i'?m|call me|this is|it'?s|name'?s|they call me)\s+"
    r"([A-Za-z][A-Za-z'-]{0,30})",
    re.IGNORECASE,
)
_NAME_PREFIX_DA = re.compile(
    r"(?:jeg hedder|mit navn er|kalde mig|det er|jeg er)\s+"
    r"([A-Za-zÆØÅæøå][A-Za-zÆØÅæøå'-]{0,30})",
    re.IGNORECASE,
)
_NAME_ONLY = re.compile(r"^([A-Za-zÆØÅæøå][A-Za-zÆØÅæøå'-]{0,30})$")
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

_SKIP_WORDS_DA = {
    "jeg",
    "hedder",
    "mit",
    "navn",
    "er",
    "det",
    "en",
    "et",
    "ja",
    "nej",
    "kalde",
    "mig",
    "ven",
    "hej",
    "god",
    "dag",
}


def extract_user_name(text: str, *, language: str = "en") -> str:
    raw = (text or "").strip()
    if not raw:
        return ""

    lang = (language or "en").strip().lower()
    if lang == "da":
        match = _NAME_PREFIX_DA.search(raw)
        if match:
            return _clean_name(match.group(1))
    else:
        match = _NAME_PREFIX.search(raw)
        if match:
            return _clean_name(match.group(1))

    match = _NAME_ONLY.match(raw.strip())
    if match:
        return _clean_name(match.group(1))

    skip = _SKIP_WORDS_DA if lang == "da" else _SKIP_WORDS
    for word in re.findall(r"[A-Za-zÆØÅæøå]+", raw):
        if word.lower() not in skip and len(word) >= 2:
            return _clean_name(word)

    return ""


def _clean_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-zÆØÅæøå'-]", "", name).strip("' -")
    if not cleaned:
        return ""
    if len(cleaned) > 1 and cleaned[1:].isupper() and cleaned[0].isupper():
        return cleaned
    return cleaned[0].upper() + cleaned[1:].lower()
