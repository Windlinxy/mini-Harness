# mini-Harness —— 给小白的详尽说明

这是一个**最小的 AI 助手框架**(英文叫 harness)。它能让你在终端里和大语言模型(LLM)持续对话,模型会**实时流式打印出它的思考过程和回答**,还能**调用工具**(比如执行 shell 命令)来帮你干活。

你不需要懂太多编程就能跑起来。下面把**每个文件、每段代码**都讲清楚。

---

## 一、它到底能干什么?

简单说,它做了三件事:

1. **持续对话**:你在终端输入问题,模型回答,然后继续输入下一轮,上下文不丢(它记得你前面说过什么)。
2. **流式输出**:模型一边思考一边把内容"吐"到屏幕上,你不用等它全部想完才看到。其中"思考"过程和"正式回答"会分开显示。
3. **调用工具**:如果模型觉得需要执行命令(比如"看看当前目录有什么文件"),它会自动调用 `run_command` 工具执行命令,把结果拿回来继续推理,最后给你答案。

整个流程就是:**你说话 → 模型思考 →(可能调工具)→ 模型回答 → 你继续说话……**如此循环。

---

## 二、项目结构一览

```
mini-Harness/
├── main.py          # 程序入口,负责和你交互(读你输入、显示输出)
├── agent_loop.py    # 核心"大脑循环",协调模型思考和工具调用
├── llm.py           # 负责和 LLM(大模型)通信,流式拿回内容
├── tools.py         # 定义"工具"——模型能调用的能力(如执行命令)
├── models.py        # 定义数据结构 Message(一条对话消息长什么样)
├── requirements.txt # 项目依赖清单(要装哪些第三方库)
└── .gitignore       # 告诉 git 哪些文件不要上传
```

它们之间的关系可以这样理解:

```
你 ──输入──▶ main.py ──▶ agent_loop.py ──▶ llm.py ──网络──▶ 大模型
                                    ▲            │
                                    │            ▼ 流式返回思考/回答/工具调用
                                    └── tools.py(执行工具,把结果回喂给模型)
```

---

## 三、逐文件详解

### 1. `models.py` —— 一条消息长什么样

这是最小的文件,定义了整个项目共用的"消息"格式。

```python
from dataclasses import dataclass

@dataclass
class Message:
    role: str                              # 谁说的:system/user/assistant/tool
    content: str | None = None             # 说的内容(文字)
    tool_calls: list[dict] | None = None   # 模型想调用的工具
    tool_call_id: str | None = None        # 工具调用的编号(回喂时配对用)
```

- `@dataclass` 是 Python 的一个装饰器,加上它你就不用手写 `__init__` 了,Python 会自动帮你生成构造函数。你只要写字段名和类型,它就能 `Message(role="user", content="你好")` 这样用。
- `role` 有四种取值,对应对话里的不同角色:
  - `system`:系统设定,告诉模型"你是个助手"(开局给一次)
  - `user`:你(用户)说的话
  - `assistant`:模型说的话
  - `tool`:工具执行完返回的结果(回喂给模型)
- `content` 是文字内容;`tool_calls` 是模型表示"我想调用某个工具"的结构;`tool_call_id` 是工具结果的编号,用来让模型知道这个结果对应哪次调用。
- `str | None = None` 表示"可以是字符串,也可以是 None(空),默认是 None"。这是 Python 3.10+ 的写法。

**为什么单独建这个文件?** 因为 `llm.py` 和 `agent_loop.py` 都要用到 `Message`,放在公共文件里避免互相导入出错(早期我们把它放在 `types.py`,但和 Python 标准库撞名了,所以改叫 `models.py`)。

---

### 2. `tools.py` —— 模型能用什么工具

这个文件定义了"工具":模型能调用的具体能力。目前只有一个工具 `run_command`(执行 shell 命令)。

文件分三部分:

**(1) 工具实现:真正干活的函数**

```python
def run_command(args: dict[str, Any]) -> str:
    return subprocess.run(
        args["cmd"], shell=True, capture_output=True, text=True, timeout=30
    ).stdout
```

- 输入是一个字典 `args`(比如 `{"cmd": "ls"}`),输出是字符串(命令的输出)。
- `subprocess.run` 是 Python 标准库里用来执行外部命令的函数:
  - `shell=True` 表示把命令交给 shell 解释执行(支持管道、通配符等)。
  - `capture_output=True` 表示把 stdout/stderr 抓回来,不直接打到屏幕。
  - `text=True` 表示返回字符串而不是字节。
  - `timeout=30` 表示命令最多跑 30 秒,超时自动杀掉,防止卡死。

**(2) 工具 schema:告诉模型这个工具怎么用**

```python
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
```

这是 [JSON Schema](https://json-schema.org/) 格式,描述工具的名字、用途、参数。**为什么要这个?** 因为模型(大模型)需要知道"有哪些工具可用、每个工具要传什么参数",它才能决定要不要调、怎么调。这个结构会随请求一起发给模型。

**(3) 调度表 + dispatch 函数:按名字找到并执行工具**

```python
_REGISTRY = {"run_command": run_command}

def dispatch(name: str, args: dict[str, Any]) -> str:
    fn = _REGISTRY.get(name)
    if not fn:
        return f"未知工具: {name}"
    try:
        return fn(args) or ""
    except Exception as e:
        return f"工具执行出错: {e}"
```

- `_REGISTRY` 是个字典,把工具名字映射到对应函数。**想加新工具?** 写个函数,加条 schema,再在这里注册一行就行。
- `dispatch` 是统一入口:给它工具名和参数,它查出对应函数并执行;找不到工具或执行出错都返回友好提示,不会让程序崩掉。

---

### 3. `llm.py` —— 和大模型通信

这个文件负责把对话发给大模型,并**流式**地拿回结果。是整个项目里最复杂的一个。

**(1) 创建客户端 + 发起流式请求**

```python
def chat_stream(model: str, messages: list[Message]) -> Iterator[dict]:
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
```

- **`base_url` 和 `api_key` 从环境变量读**,不写死在代码里。这样:① 密钥不会泄露到 GitHub;② 换别的兼容 OpenAI 协议的服务只改环境变量即可,代码不动。`MUSES_BASE_URL` 没设时会直接报错提示,而不是用错误的默认值。
- `client.chat.completions.create(...)` 是 OpenAI SDK 的标准调用方法:
  - `model`:用哪个模型(这里默认 `DeepSeek/deepseek-v4-pro`)。
  - `messages`:对话历史,用 `_to_dict` 把 `Message` 对象转成 API 要的字典格式。
  - `tools=TOOL_SCHEMAS`:把工具定义一起发过去,模型才知道能调什么。
  - `max_tokens=4096`:模型最多生成多少 token(防止无限输出)。
  - `stream=True`:**关键**,开启流式,模型一边生成一边返回小块(chunk),而不是等全部生成完才返回。

**(2) 流式接收并分类**

```python
tool_calls_acc: dict[int, dict] = {}

for chunk in stream:
    delta = chunk.choices[0].delta

    rc = getattr(delta, "reasoning_content", None)
    if rc:
        yield {"type": "reasoning", "text": rc}

    if delta.content:
        yield {"type": "content", "text": delta.content}

    if delta.tool_calls:
        ...
```

- 流式模式下,服务器会不断发来 `chunk`,每个 chunk 只是一小段增量(几个字)。`delta` 就是"这次的增量"。
- 我们把增量分成三类 `yield` 出去(让上层逐个处理):
  - `reasoning`(思考内容):DeepSeek 兼容协议里有个 `reasoning_content` 字段,是模型的"内心独白"。用 `getattr(..., None)` 是为了兼容不返回这个字段的服务,没有就不报错。
  - `content`(正文):模型正式的回答文字。
  - `tool_call`(工具调用):模型要调工具时,调用信息也是分片到达的,需要累积拼起来。
- **为什么要累积?** 流式时一个工具调用的名字、参数可能被拆成好几片到达,不能直接用,得一片片拼。`tool_calls_acc` 就是用来按 `index` 累积每个工具调用的完整信息。

**(3) 收尾:汇总完整 tool_calls**

```python
final_tool_calls = [
    {"id": ..., "type": "function", "function": {"name": ..., "arguments": ...}}
    for _, tc in sorted(tool_calls_acc.items())
]
yield {"type": "done", "tool_calls": final_tool_calls}
```

流结束后,把累积好的完整工具调用列表通过 `done` 事件交出去,上层据此决定要不要执行工具。

**(4) `_to_dict` 辅助函数**

```python
def _to_dict(m: Message) -> dict:
    d: dict = {"role": m.role}
    if m.content is not None:
        d["content"] = m.content
    ...
    return d
```

API 只认字典格式,这个函数把 `Message` 对象转成字典,并按需带上 `content`/`tool_calls`/`tool_call_id` 字段。

---

### 4. `agent_loop.py` —— 核心循环

这是"大脑",协调模型思考和工具调用,决定什么时候结束。

**(1) `run_step`:执行一步**

```python
def run_step(messages: list[Message], *, model: str = ...) -> bool:
```

- 输入是当前所有对话历史 `messages`,`*` 表示后面参数必须用关键字传(`model=...`)。
- 返回 `True` 表示模型给了最终回答(没调工具),`False` 表示调了工具、还要继续循环。

内部逻辑:

```python
for chunk in chat_stream(model=model, messages=messages):
    if ctype == "reasoning":
        ...print(chunk["text"], end="", flush=True)      # 实时打印思考
    elif ctype == "content":
        ...print(chunk["text"], end="", flush=True)      # 实时打印回答
        content_parts.append(chunk["text"])              # 同时存起来
    elif ctype == "done":
        final_tool_calls = chunk["tool_calls"]           # 拿到完整工具调用
```

- `print(..., end="", flush=True)`:不换行、立即刷新到屏幕,实现"打字机"流式效果。
- 思考和回答前会各打印一条分隔线 `==== 思考 ====` / `==== 回答 ====`,让你看清楚哪段是思考、哪段是回答。

```python
if not final_tool_calls:
    return True                    # 没调工具 = 说完了

for call in final_tool_calls:
    args = json.loads(call["function"]["arguments"] or "{}")
    result = dispatch(call["function"]["name"], args)   # 执行工具
    messages.append(Message("tool", content=result, tool_call_id=call["id"]))  # 结果回喂
return False                       # 调了工具,继续循环
```

- 如果模型没调工具,这一步就结束了。否则把每个工具执行,结果作为 `tool` 消息塞回 `messages`,下一轮模型能看到结果继续推理。

**(2) `run`:单轮处理(保留给单次调用用)**

```python
def run(user_input: str, *, model=..., system_prompt=..., max_steps=8) -> str:
    messages = [Message("system", system_prompt), Message("user", user_input)]
    for _ in range(max_steps):
        done = run_step(messages, model=model)
        if done:
            return messages[-1].content or ""
    return "(达到最大步数上限)"
```

- 一次性处理一条用户输入,内部最多循环 `max_steps`(默认 8)次防止死循环。
- 现在 `main.py` 用的是持续对话版,直接调 `run_step`;`run` 留着方便单次调用或测试。

---

### 5. `main.py` —— 入口,和你交互

最外层,负责读你输入、显示输出、维持多轮对话。

```python
def main() -> None:
    messages: list[Message] = [Message("system", SYSTEM_PROMPT)]   # 开局给系统设定
    print("harness 已启动,输入 'exit' 退出。\n")

    while True:
        try:
            user_input = input("你 ▶ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break
        ...
        messages.append(Message("user", user_input))

        for _ in range(8):
            done = run_step(messages)
            if done:
                break
```

- `messages` 在整个 `while` 循环里一直保留,**所以模型记得前面说过的话**(多轮上下文)。
- `input("你 ▶ ")` 在终端显示提示符等你输入。`EOFError`/`KeyboardInterrupt` 是你按 Ctrl+D / Ctrl+C 时的异常,捕获后优雅退出。
- 输入 `exit`/`quit`/`退出` 就结束。
- 每轮最多循环 8 步(`for _ in range(8)`),防止模型一直调工具停不下来。

`if __name__ == "__main__": main()` 是 Python 的惯例:直接运行这个文件时才执行 `main()`,被 import 时不执行。

---

### 6. `requirements.txt` —— 依赖清单

```
openai>=1.0.0
```

只有一行:依赖 `openai` 这个第三方库(版本 ≥ 1.0.0)。安装命令 `pip install -r requirements.txt` 会读这个文件,把里面列的库全装上。**`openai` 库虽然名字叫 openai,但任何兼容 OpenAI 协议的服务都能用**(比如本项目的 DeepSeek 兼容代理),只要改 `base_url`。

---

### 7. `.gitignore` —— 不上传哪些文件

```
.venv/           # 虚拟环境,体积大且每台机器不同
__pycache__/     # Python 编译缓存
*.pyc            # 编译后的字节码文件
.env             # 存密钥的文件,绝对不能上传!
.DS_Store        # macOS 的文件夹元数据,无用
.idea/           # PyCharm 的配置目录,个人本地用
```

这些文件要么是缓存、要么是个人配置、要么含密钥,都不该进 git 仓库。`.gitignore` 列出来后,`git add` 会自动忽略它们。

---

## 四、怎么跑起来

**第一次准备环境:**

```bash
cd 你放项目的目录
python3 -m venv .venv                                  # 创建虚拟环境
.venv/bin/pip install -r requirements.txt              # 装依赖
```

**设置密钥(只做一次,写进 ~/.zshrc 永久生效):**

```bash
echo 'export MUSES_API_KEY="你的key"' >> ~/.zshrc
echo 'export MUSES_BASE_URL="你的_LLM_API_base_url"' >> ~/.zshrc
source ~/.zshrc
```

> 密钥只在环境变量里,不进项目代码,所以推到 GitHub 也不会泄露。

**运行:**

```bash
.venv/bin/python main.py
```

启动后看到 `你 ▶` 提示符就可以输入问题了。输入 `exit` 退出。你会先看到 `==== 思考 ====` 流式打印模型的推理,再看到 `==== 回答 ====` 流式打印正式回答;如果模型调用了工具,还会看到 `[执行工具]` 和 `[工具结果]`。

---

## 五、想自己扩展?

- **加新工具**:在 `tools.py` 写一个函数(输入 dict 输出 str)→ 在 `TOOL_SCHEMAS` 加对应 schema → 在 `_REGISTRY` 注册一行。模型就能自动用上。
- **换模型**:`agent_loop.py` 和 `main.py` 里改 `model` 参数,或调用时传 `model="别的模型名"`。
- **加沙箱限制**:在 `tools.dispatch` 或 `run_command` 里加路径/命令白名单,防止模型乱删文件。
- **加审批**:在 `dispatch` 执行前判断高危操作,弹确认让你批准再跑。

---

## 六、一句话总结

> `main.py` 听你说话 → `agent_loop.py` 协调 → `llm.py` 联网问模型 → `tools.py` 干活 → 结果回喂 → 循环直到回答完。
