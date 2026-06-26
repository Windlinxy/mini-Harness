from __future__ import annotations

import subprocess
from typing import Any

# 1. 工具实现：纯函数，输入 dict，输出 str
def run_command(args: dict[str, Any]) -> str:
    return subprocess.run(
        args["cmd"], shell=True, capture_output=True, text=True, timeout=30
    ).stdout


# 2. 工具 schema：OpenAI function-calling 格式
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "在 shell 执行命令并返回 stdout",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    }
]

# 3. 调度表：名字 -> 函数
_REGISTRY = {"run_command": run_command}


def dispatch(name: str, args: dict[str, Any]) -> str:
    fn = _REGISTRY.get(name)
    if not fn:
        return f"未知工具: {name}"
    try:
        return fn(args) or ""
    except Exception as e:
        return f"工具执行出错: {e}"
