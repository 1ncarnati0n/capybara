from __future__ import annotations

import threading
import time


class TranslationCancelled(RuntimeError):
    """Raised when the user requests cancellation from the UI."""


class TranslationControl:
    def __init__(self) -> None:
        self._paused = threading.Event()
        self._cancelled = threading.Event()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def cancel(self) -> None:
        self._cancelled.set()
        self.resume()

    def checkpoint(self) -> None:
        if self.cancelled:
            raise TranslationCancelled("번역 중지 요청됨")
        while self.paused:
            if self.cancelled:
                raise TranslationCancelled("번역 중지 요청됨")
            time.sleep(0.2)
