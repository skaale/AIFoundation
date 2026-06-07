from __future__ import annotations

import json
import re
import warnings
from collections.abc import Iterator
from threading import Thread

warnings.filterwarnings(
    "ignore",
    message="The pynvml package is deprecated",
    category=FutureWarning,
)

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, TextIteratorStreamer

from .constants import (
    MODEL_ID,
    SOL_END,
    SOL_START,
    SYSTEM_PROMPT,
    THINK_END,
    THINK_START,
)


def resolve_device(device: str | None = "auto") -> tuple[str, torch.dtype]:
    choice = (device or "auto").lower()

    if choice == "auto":
        if torch.cuda.is_available():
            return "cuda", torch.bfloat16
        return "cpu", torch.float32

    if choice in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no GPU is available.")
        return "cuda", torch.bfloat16

    if choice == "cpu":
        return "cpu", torch.float32

    raise ValueError(f"Unsupported device '{device}'. Use 'auto', 'cpu', or 'cuda'.")


def load_tokenizer(model_id: str) -> PreTrainedTokenizerFast:
    tokenizer_file = hf_hub_download(model_id, "tokenizer.json")
    config_file = hf_hub_download(model_id, "tokenizer_config.json")
    with open(config_file, encoding="utf-8") as f:
        config = json.load(f)

    return PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_file,
        bos_token=config["bos_token"],
        eos_token=config["eos_token"],
        unk_token=config["unk_token"],
        pad_token=config["pad_token"],
    )


_META_ANSWER_PREFIXES = (
    re.compile(
        r"^the user(?:'s)?(?: is)? asking(?: for)?(?: me)?(?: to)?[^.?!]*[.?!]\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^the listener(?:'s)?(?: is)? asking(?: for)?(?: me)?(?: to)?[^.?!]*[.?!]\s*",
        re.IGNORECASE,
    ),
    re.compile(r"^the user wants to know[^.?!]*[.?!]\s*", re.IGNORECASE),
    re.compile(r"^the user has asked[^.?!]*[.?!]\s*", re.IGNORECASE),
    re.compile(
        r"^based on (?:the )?(?:user'?s?|listener'?s?) (?:question|request)[^.?!]*[.?!]\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^as (?:a |the )?(?:user|listener)[^.?!]*[.?!]\s*",
        re.IGNORECASE,
    ),
)

_META_SENTENCE = re.compile(
    r"\b(?:the\s+)?(?:user|listener|customer|requester)s?\b",
    re.IGNORECASE,
)


def _replace_meta_terms(text: str, listener_name: str | None) -> str:
    address = listener_name.strip() if listener_name else "friend"
    possessive = f"{address}'s" if listener_name else "your"
    replacements = (
        (re.compile(r"\bthe user's\b", re.IGNORECASE), possessive),
        (re.compile(r"\bthe listener's\b", re.IGNORECASE), possessive),
        (re.compile(r"\bthe user\b", re.IGNORECASE), address),
        (re.compile(r"\bthe listener\b", re.IGNORECASE), address),
        (re.compile(r"\ba user\b", re.IGNORECASE), "a friend"),
        (re.compile(r"\busers\b", re.IGNORECASE), "friends"),
        (re.compile(r"\blisteners\b", re.IGNORECASE), "friends"),
        (re.compile(r"\buser\b", re.IGNORECASE), address),
        (re.compile(r"\blistener\b", re.IGNORECASE), address),
    )
    for pattern, repl in replacements:
        text = pattern.sub(repl, text)
    return text


def _drop_meta_sentences(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    kept: list[str] = []
    for part in parts:
        if not part:
            continue
        if _META_SENTENCE.search(part):
            lower = part.lower()
            if any(
                phrase in lower
                for phrase in (
                    "the user",
                    "a user",
                    "the listener",
                    "user is",
                    "user wants",
                    "user has",
                    "user asked",
                )
            ):
                continue
        kept.append(part)
    return " ".join(kept).strip()


def clean_priest_answer(answer: str, listener_name: str | None = None) -> str:
    text = (answer or "").strip()
    if not text:
        return ""

    changed = True
    while changed:
        changed = False
        for pattern in _META_ANSWER_PREFIXES:
            updated = pattern.sub("", text, count=1).strip()
            if updated != text:
                text = updated
                changed = True

    text = _drop_meta_sentences(text)
    text = _replace_meta_terms(text, listener_name)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def build_prompt(
    question: str,
    history: list[dict] | None = None,
    knowledge_context: str | None = None,
    memory_context: str | None = None,
    listener_name: str | None = None,
) -> str:
    parts = [f"[SYSTEM]: {SYSTEM_PROMPT}\n"]

    if memory_context and memory_context.strip():
        parts.append(f"[MEMORY]:\n{memory_context.strip()}\n")

    if listener_name and listener_name.strip():
        parts.append(
            "[LISTENER]: You are speaking with "
            f"{listener_name.strip()}. Address them by name warmly. "
            "Never say 'user' or 'listener' in your reply.\n"
        )

    if knowledge_context and knowledge_context.strip():
        parts.append(f"[KNOWLEDGE]:\n{knowledge_context.strip()}\n")

    for msg in history or []:
        role = msg.get("role")
        content = str(msg.get("content", "")).strip()
        if not content or content == "…":
            continue
        if role == "user":
            parts.append(f"[USER]: {content}\n")
        elif role == "assistant":
            parts.append(f"[ASSISTANT]: {content}\n")

    parts.append(f"[USER]: {question}\n\n[ASSISTANT]: {THINK_START}\n")
    return "\n".join(parts)


def _clean_generated(raw: str) -> str:
    raw = raw.replace("<s>", "").replace("</s>", "").strip()
    return THINK_START + "\n" + raw


def parse_output(raw: str) -> tuple[str, str]:
    thought, answer = "", raw

    if THINK_START in raw and THINK_END in raw:
        t0 = raw.index(THINK_START) + len(THINK_START)
        t1 = raw.index(THINK_END)
        thought = raw[t0:t1].strip()

    if SOL_START in raw and SOL_END in raw:
        s0 = raw.index(SOL_START) + len(SOL_START)
        s1 = raw.index(SOL_END)
        answer = raw[s0:s1].strip()
    elif SOL_START in raw:
        s0 = raw.index(SOL_START) + len(SOL_START)
        answer = raw[s0:].strip()
    elif THINK_END in raw:
        answer = raw[raw.index(THINK_END) + len(THINK_END) :].strip()

    return thought, clean_priest_answer(answer)


class SupraReasoningModel:
    def __init__(self, model_id: str = MODEL_ID, device: str = "auto") -> None:
        self.model_id = model_id
        self.device, self.dtype = resolve_device(device)
        self.tokenizer = load_tokenizer(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=self.dtype,
            device_map=self.device,
        )
        self.model.eval()

    @property
    def torch_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def generate(
        self,
        question: str,
        *,
        max_new_tokens: int = 992,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 25,
    ) -> tuple[str, str]:
        if not question.strip():
            return "", ""

        final = None
        for update in self.generate_stream(
            question,
            knowledge_context=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        ):
            final = update
        return (final or {"thought": "", "answer": ""})["thought"], (final or {"thought": "", "answer": ""})["answer"]

    def generate_stream(
        self,
        question: str,
        *,
        history: list[dict] | None = None,
        knowledge_context: str | None = None,
        memory_context: str | None = None,
        listener_name: str | None = None,
        max_new_tokens: int = 992,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 25,
    ) -> Iterator[dict[str, str]]:
        if not question.strip():
            return

        full_prompt = build_prompt(
            question,
            history,
            knowledge_context,
            memory_context,
            listener_name,
        )
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.torch_device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.torch_device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=False,
        )
        generation_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        accumulated = ""
        for token in streamer:
            accumulated += token
            raw = _clean_generated(accumulated)
            thought, answer = parse_output(raw)
            yield {"thought": thought, "answer": answer, "token": token}

        thread.join()
