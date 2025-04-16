# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "ryd-numerov",
# ]
# ///

import sys

import ryd_numerov


def main() -> None:
    print("ryd-numerov version:", ryd_numerov.__version__)
    print("python version:", sys.version)


if __name__ == "__main__":
    main()
