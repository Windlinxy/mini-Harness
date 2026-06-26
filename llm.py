from __future__ import annotations

import os

from models import Message
from tools import TOOL_SCHEMAS


def chat(model: str, messages: list[Message]) -> Message:
    """Call the LLM with tool definitions. Return one assistant Message."""
    from openai import OpenAI

    client = OpenAI(
        base_url=os.environ["MUSES_BASE_URL"],
        api_key=os.environ["MUSES_API_KEY"],
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[_to_dict(m) for m in messages],
        tools=TOOL_SCHEMAS,
        max_tokens=4096,
    )
    msg = resp.choices[0].message
    return Message(
        role="assistant",
        content=msg.content,
        tool_calls=[
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in (msg.tool_calls or [])
        ],
    )


def _to_dict(m: Message) -> dict:
    d: dict = {"role": m.role}
    if m.content is not None:
        d["content"] = m.content
    if m.tool_calls:
        d["tool_calls"] = m.tool_calls
    if m.tool_call_id:
        d["tool_call_id"] = m.tool_call_id
    return d
