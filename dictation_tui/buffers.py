from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Segment:
    id: int
    text: str
    start_ts: float
    end_ts: float
    stable: bool = False
    cleaned: str | None = None
    cleanup_source: str = ""


class RawBuffer:
    def __init__(self):
        self.segments: list[Segment] = []
        self.pending: Segment | None = None
        self._next_id = 0

    def start_pending(self, text: str, ts: float):
        self.pending = Segment(id=self._next_id, text=text, start_ts=ts, end_ts=ts)

    def update_pending(self, text: str, ts: float):
        if self.pending is None:
            self.start_pending(text, ts)
        else:
            self.pending.text = text
            self.pending.end_ts = ts

    def commit_pending(self) -> Segment | None:
        if self.pending is None or not self.pending.text.strip():
            self.pending = None
            return None
        self.pending.stable = True
        seg = self.pending
        self.segments.append(seg)
        self._next_id += 1
        self.pending = None
        return seg

    def get_full_text(self) -> str:
        parts = [s.text for s in self.segments]
        if self.pending and self.pending.text.strip():
            parts.append(self.pending.text)
        return "\n\n".join(parts)

    def clear(self):
        self.segments.clear()
        self.pending = None
        self._next_id = 0


class CleanBuffer:
    def __init__(self):
        self.segments: list[Segment] = []

    def add(self, segment: Segment):
        self.segments.append(segment)

    def get_full_text(self) -> str:
        return "\n\n".join(s.cleaned or s.text for s in reversed(self.segments))

    def clear(self):
        self.segments.clear()
