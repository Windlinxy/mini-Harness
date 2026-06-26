import sys

from agent_loop import run_step
from models import Message

SYSTEM_PROMPT = "You are a helpful assistant."


def main() -> None:
    messages: list[Message] = [Message("system", SYSTEM_PROMPT)]
    print("harness 已启动,输入 'exit' 退出。\n")

    while True:
        try:
            user_input = input("你 ▶ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "退出"):
            print("再见!")
            break

        messages.append(Message("user", user_input))

        for _ in range(8):
            done = run_step(messages)
            if done:
                break

        print()


if __name__ == "__main__":
    main()
