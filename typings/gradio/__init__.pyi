from typing import Any, Literal

Visible = bool | Literal["hidden"]

from . import themes as themes

__all__ = [
    "Blocks",
    "State",
    "HTML",
    "Chatbot",
    "Textbox",
    "Number",
    "Column",
    "Checkbox",
    "Dropdown",
    "Slider",
    "Markdown",
    "Button",
    "skip",
    "update",
    "Timer",
    "themes",
]

class Blocks:
    def __init__(
        self,
        *,
        theme: Any = ...,
        title: str = ...,
        css: str = ...,
        js: str = ...,
        fill_height: bool = ...,
    ) -> None: ...
    def load(self, *args: Any, **kwargs: Any) -> Any: ...
    def launch(self, *args: Any, **kwargs: Any) -> None: ...
    def queue(self, *args: Any, **kwargs: Any) -> Blocks: ...
    def __enter__(self) -> Blocks: ...
    def __exit__(self, *args: Any) -> None: ...

class State:
    def __init__(self, value: Any = ...) -> None: ...

class HTML:
    def __init__(
        self,
        value: str = ...,
        *,
        elem_id: str = ...,
        container: bool = ...,
        visible: Visible = ...,
    ) -> None: ...

class Chatbot:
    def __init__(
        self,
        *,
        label: str = ...,
        height: int = ...,
        type: str = ...,
        show_label: bool = ...,
        elem_id: str = ...,
        visible: Visible = ...,
    ) -> None: ...

class Textbox:
    def __init__(
        self,
        value: str = ...,
        *,
        label: str = ...,
        lines: int = ...,
        max_lines: int = ...,
        interactive: bool = ...,
        elem_id: str = ...,
        visible: Visible = ...,
        show_label: bool = ...,
        placeholder: str = ...,
    ) -> None: ...
    def change(self, *args: Any, **kwargs: Any) -> Any: ...

class Number:
    def __init__(
        self,
        value: float = ...,
        *,
        elem_id: str = ...,
        show_label: bool = ...,
        visible: Visible = ...,
    ) -> None: ...
    def change(self, *args: Any, **kwargs: Any) -> Any: ...

class Column:
    def __init__(self, *, elem_id: str = ...) -> None: ...
    def __enter__(self) -> Column: ...
    def __exit__(self, *args: Any) -> None: ...

class Checkbox:
    def __init__(self, *, value: bool = ..., label: str = ...) -> None: ...

class Dropdown:
    def __init__(
        self,
        *,
        choices: list[Any] = ...,
        value: Any = ...,
        label: str = ...,
    ) -> None: ...
    def change(self, *args: Any, **kwargs: Any) -> Any: ...

class Slider:
    def __init__(
        self,
        minimum: float,
        maximum: float,
        *,
        value: float = ...,
        step: float = ...,
        label: str = ...,
    ) -> None: ...

class Markdown:
    def __init__(self, value: str = ..., *, elem_id: str = ...) -> None: ...

class Button:
    def __init__(self, value: str = ..., *, label: str = ...) -> None: ...
    def click(self, *args: Any, **kwargs: Any) -> Any: ...

def skip() -> dict[str, Any]: ...

def update(**kwargs: Any) -> dict[str, Any]: ...

class Timer:
    def __init__(self, value: float, *, active: bool = ...) -> None: ...
    def tick(self, *args: Any, **kwargs: Any) -> Any: ...
