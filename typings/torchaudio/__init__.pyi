from typing import Any

def load(path: str, *args: Any, **kwargs: Any) -> tuple[Any, int]: ...

class functional:
    @staticmethod
    def resample(
        waveform: Any,
        orig_freq: int,
        new_freq: int,
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...
