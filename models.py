from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Message:
    role: str
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
