MODEL_ID = "SupraLabs/Supra-50M-Reasoning"

THINK_START = "<|begin_of_thought|>"
THINK_END = "<|end_of_thought|>"
SOL_START = "<|begin_of_solution|>"
SOL_END = "<|end_of_solution|>"

from .languages import ENGLISH, pick_intro_greeting, pick_name_prompt, pick_name_welcome

SYSTEM_PROMPT = ENGLISH.system_prompt
INTRO_GREETINGS = ENGLISH.intro_greetings
CHURCH_INTRODUCTION = INTRO_GREETINGS[0]
NAME_PROMPTS = ENGLISH.name_prompts
NAME_WELCOME_TEMPLATES = ENGLISH.name_welcome_templates


def intro_opening_line(intro: str) -> str:
    from .languages import intro_opening_line as _intro_opening_line

    return _intro_opening_line(intro, "en")

PRIEST_VOICE = ENGLISH.default_voice
KOKORO_WARMUP_VOICE = ENGLISH.voices[ENGLISH.default_voice]

SILENCE_PROMPT = ENGLISH.silence_prompt

IDLE_SECONDS_BEFORE_PROMPT = 12
MAX_PROMPT_HISTORY_MESSAGES = 16
