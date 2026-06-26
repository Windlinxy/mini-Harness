from __future__ import annotations

import json

from llm import chat
from tools import TOOL_SCHEMAS, dispatch
from models import Message


def run(
    user_input: str,
    *,
    model: str = "DeepSeek/deepseek-v4-pro",
    system_prompt: str = "You are a helpful assistant.",
    max_steps: int = 8,
) -> str:
    """Run the agent loop until the model returns a final text answer."""
    messages: list[Message] = [
        Message("system", system_prompt),
        Message("user", user_input),
    ]

    for _ in range(max_steps):
        assistant = chat(model=model, messages=messages)
        messages.append(assistant)

        # 没有 tool_calls = 模型给出最终回答
        if not assistant.tool_calls:
            return assistant.content or ""

        # 执行所有 tool_calls，把结果作为 tool 消息回喂
        for call in assistant.tool_calls:
            args = json.loads(call["function"]["arguments"] or "{}")
            result = dispatch(call["function"]["name"], args)
            messages.append(
                Message("tool", content=result, tool_call_id=call["id"])
            )

    return "(达到最大步数上限，agent 未完成)"
