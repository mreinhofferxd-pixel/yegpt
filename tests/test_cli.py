"""Tests for the `yegpt` console dispatcher (cli.py).

We drive the router, not the subcommands: each subcommand already has its own tests. So we stub a
handler to confirm the dispatcher (a) routes the right command to it, (b) hands it a clean
`sys.argv` (prog `yegpt <command>`, only its own args), and (c) restores `sys.argv` afterward.
"""

from __future__ import annotations

import sys

import pytest

from yegpt import __version__, cli


def test_no_args_prints_usage_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main([])
    out = capsys.readouterr().out
    assert "usage: yegpt <command>" in out
    # Every registered command is advertised in the usage block.
    for name in cli._COMMANDS:
        assert name in out


def test_version_flag_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_unknown_command_raises_system_exit() -> None:
    with pytest.raises(SystemExit):
        cli.main(["nope"])


def test_dispatch_rewrites_argv_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_handler() -> None:
        seen.extend(sys.argv)

    monkeypatch.setitem(cli._COMMANDS, "sample", fake_handler)
    sentinel = ["something", "else"]
    monkeypatch.setattr(sys, "argv", sentinel)

    cli.main(["sample", "--prompt", "yo", "-n", "10"])

    # The handler saw a clean argv scoped to its own command...
    assert seen == ["yegpt sample", "--prompt", "yo", "-n", "10"]
    # ...and the process argv was restored to exactly what it was before dispatch.
    assert sys.argv is sentinel
