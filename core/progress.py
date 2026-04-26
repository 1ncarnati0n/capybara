from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field
from typing import Literal

EventKind = Literal["progress", "stats", "log", "fill_fail", "done", "error", "cancelled"]


@dataclass
class ProgressEvent:
    kind: EventKind
    payload: object = None
    ts: float = field(default_factory=time.time)


class ProgressBus:
    """Thread-safe one-way channel from the worker thread to the Streamlit UI."""

    def __init__(self) -> None:
        self._q: queue.Queue[ProgressEvent] = queue.Queue()

    def emit(self, kind: EventKind, payload: object = None) -> None:
        self._q.put(ProgressEvent(kind=kind, payload=payload))

    def drain(self) -> list[ProgressEvent]:
        items: list[ProgressEvent] = []
        while True:
            try:
                items.append(self._q.get_nowait())
            except queue.Empty:
                break
        return items
