"""Execute a command after joining a resctrl group via its tasks file."""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 3 or args[1] != "--" or not args[0]:
        print(
            "usage: python3 -m measure.exec_in_group TASKS_FILE -- CMD [ARGS...]",
            file=sys.stderr,
        )
        return 2

    tasks_path = pathlib.Path(args[0])
    cmd = args[2:]

    try:
        tasks_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    except OSError as exc:
        print(f"error: failed to write pid to {tasks_path}: {exc}", file=sys.stderr)
        return 1

    try:
        os.execvp(cmd[0], cmd)
    except OSError as exc:
        print(f"error: failed to exec {cmd[0]}: {exc}", file=sys.stderr)
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
