from __future__ import annotations

import json
import sys

from llm import chat_stream
from tools import dispatch
from models import Message


def run_step(
    messages: list[Message],
    *,
    model: str = "DeepSeek/deepseek-v4-pro",
) -> bool:
    """执行一步:流式输出思考+正文,若调工具则执行并回喂。
    返回 True 表示 agent 给出最终回答(无工具调用),False 表示还需继续。"""

    content_parts: list[str] = []
    final_tool_calls: list[dict] = []

    reasoning_started = False
    content_started = False

    for chunk in chat_stream(model=model, messages=messages):
        ctype = chunk["type"]

        if ctype == "reasoning":
            if not reasoning_started:
                print("\n" + "=" * 20 + " 思考 " + "=" * 20, flush=True)
                reasoning_started = True
            print(chunk["text"], end="", flush=True)

        elif ctype == "content":
            if not content_started and chunk["text"]:
                print("\n" + "=" * 20 + " 回答 " + "=" * 20, flush=True)
                content_started = True
            print(chunk["text"], end="", flush=True)
            content_parts.append(chunk["text"])

        elif ctype == "done":
            final_tool_calls = chunk["tool_calls"]

    print(flush=True)

    assistant = Message(
        role="assistant",
        content="".join(content_parts) or None,
        tool_calls=final_tool_calls or None,
    )
    messages.append(assistant)

    if not final_tool_calls:
        return True

    for call in final_tool_calls:
        args = json.loads(call["function"]["arguments"] or "{}")
        print(f"\n[执行工具] {call['function']['name']} {args}", flush=True)
        result = dispatch(call["function"]["name"], args)
        print(f"[工具结果] {result[:500]}\n", flush=True)
        messages.append(
            Message("tool", content=result, tool_call_id=call["id"])
        )

    return False


def run(
    user_input: str,
    *,
    model: str = "DeepSeek/deepseek-v4-pro",
    system_prompt: str = "You are a helpful assistant.",
    max_steps: int = 8,
) -> str:
    """单轮:处理一次用户输入,内部循环到最终回答。"""
    messages: list[Message] = [
        Message("system", system_prompt),
        Message("user", user_input),
    ]

    for _ in range(max_steps):
        done = run_step(messages, model=model)
        if done:
            return messages[-1].content or ""

    return "(达到最大步数上限)"
