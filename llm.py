from __future__ import annotations

import os
import json
from typing import Iterator

from models import Message
from tools import TOOL_SCHEMAS

def chat_stream(model: str, messages: list[Message]) -> Iterator[dict]:
    """流式调用 LLM,yield 每个 chunk 的增量内容。

    chunk 格式: {"type": "reasoning"|"content"|"tool_call", ...}
    - reasoning: {"type":"reasoning","text":"..."}
    - content:   {"type":"content","text":"..."}
    - tool_call: {"type":"tool_call","index":0,"id":"...","name":"...","arguments":"..."}
    结束时 yield {"type":"done","tool_calls":[...]}
    """
    from openai import OpenAI

    base_url = os.environ.get("MUSES_BASE_URL")
    if not base_url:
        raise RuntimeError("请设置环境变量 MUSES_BASE_URL")

    client = OpenAI(
        base_url=base_url,
        api_key=os.environ["MUSES_API_KEY"],
    )
    stream = client.chat.completions.create(
        model=model,
        messages=[_to_dict(m) for m in messages],
        tools=TOOL_SCHEMAS,
        max_tokens=4096,
        stream=True,
    )

    tool_calls_acc: dict[int, dict] = {}

    for chunk in stream:
        delta = chunk.choices[0].delta

        # 思考内容(DeepSeek 兼容协议)
        rc = getattr(delta, "reasoning_content", None)
        if rc:
            yield {"type": "reasoning", "text": rc}

        # 正文内容
        if delta.content:
            yield {"type": "content", "text": delta.content}

        # 工具调用(流式分片到达,需累积)
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {
                        "id": tc.id or "",
                        "name": "",
                        "arguments": "",
                    }
                if tc.function:
                    if tc.function.name:
                        tool_calls_acc[idx]["name"] += tc.function.name
                    if tc.function.arguments:
                        tool_calls_acc[idx]["arguments"] += tc.function.arguments

    final_tool_calls = [
        {
            "id": tc["id"],
            "type": "function",
            "function": {"name": tc["name"], "arguments": tc["arguments"]},
        }
        for _, tc in sorted(tool_calls_acc.items())
    ]
    yield {"type": "done", "tool_calls": final_tool_calls}


def _to_dict(m: Message) -> dict:
    d: dict = {"role": m.role}
    if m.content is not None:
        d["content"] = m.content
    if m.tool_calls:
        d["tool_calls"] = m.tool_calls
    if m.tool_call_id:
        d["tool_call_id"] = m.tool_call_id
    return d
