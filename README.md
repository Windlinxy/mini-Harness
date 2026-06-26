# 最小 Python harness

一个 agentic harness 骨架：用户输入 → LLM → tool calling → 执行工具 → 回喂 → 循环到最终回答。

## 结构

```
harness/
├── main.py          # 入口，拼装参数调 run()
├── agent_loop.py    # 核心：循环调用 LLM + 工具
├── llm.py           # LLM 接口层：OpenAI chat completions + tool 定义
├── tools.py         # 工具实现 + schema + 调度
├── models.py        # 共享数据结构 (Message)
└── requirements.txt
```

## 数据流

```
user_input
   │
   ▼
agent_loop.run()
   │  组装 messages → llm.chat() → 拿到 assistant Message
   │
   ├── 有 tool_calls? → tools.dispatch() 执行 → 结果作为 tool 消息回喂 → 继续循环
   │
   └── 无 tool_calls?  → 返回最终文本
```

## 运行

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python main.py "列出当前目录的文件"
```

## 如何扩展

- 加工具：在 `tools.py` 写一个函数 + 对应 schema + 注册到 `_REGISTRY`
- 加沙箱：在 `tools.dispatch` 或 `run_command` 里加路径/网络限制
- 加审批：在 `dispatch` 执行前判断是否需要用户确认
- 加流式：`llm.chat` 改用 `stream=True`，逐 chunk 解析
