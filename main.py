import sys

from agent_loop import run


def main() -> None:
    user_input = " ".join(sys.argv[1:]) or "列出当前目录文件"
    answer = run(user_input)
    print(answer)


if __name__ == "__main__":
    main()
