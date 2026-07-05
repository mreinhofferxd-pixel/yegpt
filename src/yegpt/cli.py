"""yegpt: one console entry point that dispatches to the package's per-module CLIs.

`[project.scripts]` wires `yegpt = "yegpt.cli:main"`, so `yegpt <command> [args...]` runs the
matching module's own `main()`. Each subcommand owns its flags and its parser; this dispatcher
only routes to it. That means `yegpt sample --help` shows sample.py's parser (not this one), and
adding a flag to a subcommand never touches this file.

The delegated mains call `argparse`'s `parse_args()` with no argument, which reads `sys.argv`.
So we hand each one a clean `sys.argv` whose prog is `yegpt <command>` and whose remaining args
are only its own, then restore the original afterward.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

from yegpt import __version__, data_prep, dedup, export, export_samples, sample, train, tweet_prep

# Ordered to read roughly as the pipeline runs: build corpus -> train -> sample -> export.
_COMMANDS: dict[str, Callable[[], None]] = {
    "data-prep": data_prep.main,
    "tweet-prep": tweet_prep.main,
    "dedup": dedup.main,
    "train": train.main,
    "sample": sample.main,
    "export": export.main,
    "export-samples": export_samples.main,
}


def _usage() -> str:
    commands = "\n".join(f"  {name}" for name in _COMMANDS)
    return (
        f"usage: yegpt <command> [args...]\n\n"
        f"yeGPT {__version__} - a character-level GPT trained from scratch.\n\n"
        f"commands:\n{commands}\n\n"
        f"Run 'yegpt <command> --help' for command-specific options."
    )


def main(argv: list[str] | None = None) -> None:
    """Route `yegpt <command> ...` to the matching module `main`, else print usage/version.

    `argv` defaults to the process arguments (`sys.argv[1:]`); pass a list to drive it in tests.
    Bare invocation or `-h/--help` prints the command list; `-V/--version` prints the version.
    An unknown command writes usage to stderr and exits non-zero via `SystemExit`.
    """
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in ("-h", "--help"):
        print(_usage())
        return
    if args[0] in ("-V", "--version"):
        print(f"yeGPT {__version__}")
        return

    command = args[0]
    handler = _COMMANDS.get(command)
    if handler is None:
        print(_usage(), file=sys.stderr)
        raise SystemExit(f"yegpt: unknown command {command!r}")

    saved_argv = sys.argv
    sys.argv = [f"yegpt {command}", *args[1:]]
    try:
        handler()
    finally:
        sys.argv = saved_argv


if __name__ == "__main__":  # pragma: no cover
    main()
