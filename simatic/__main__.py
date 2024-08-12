import sys
from .commands.chat import chat


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'chat':

        sys.argv = [sys.argv[0]] + sys.argv[2:]
        chat()
    else:
        print("Usage: simatic chat [args...]")
        print("For more information, use: simatic chat --help")


if __name__ == "__main__":
    main()
