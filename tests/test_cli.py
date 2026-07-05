"""Tests for the `yegpt` console entry point (cli.py): subcommand routing + the streaming demo.

Routing: each subcommand already has its own tests, so we stub a handler to confirm the
dispatcher (a) routes the right command to it, (b) hands it a clean `sys.argv` (prog
`yegpt <command>`, only its own args), and (c) restores `sys.argv` afterward.

Demo (prompt mode): a first argument that is not a known command streams a generation. We build
a tiny untrained GPT saved through the real checkpoint format and assert the EXACT seeded stdout
against `sample.sample_from_checkpoint` with the same knobs - which also pins the demo's default
knobs (temperature 0.9, top-p 0.92, repetition penalty 1.3) and the `--max-chars` ->
`max_new_tokens` mapping. Invalid knobs and a missing checkpoint must exit non-zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from yegpt import __version__, cli
from yegpt.config import TrainConfig
from yegpt.model import GPT
from yegpt.sample import sample_from_checkpoint
from yegpt.tokenizer import CharTokenizer
from yegpt.train import load_checkpoint, save_checkpoint

# Same tiny corpus/config family as test_sample: a newline in the vocab, CPU, dropout off.
_CORPUS = "yeezy taught me\n" * 8
_TOKENIZER = CharTokenizer.from_text(_CORPUS)
_CFG = TrainConfig(
    n_layer=2,
    n_head=2,
    n_embd=16,
    block_size=16,
    dropout=0.0,
    batch_size=8,
    device="cpu",
    seed=0,
).with_vocab_size(_TOKENIZER.vocab_size)


def _save_tiny_checkpoint(path: Path) -> None:
    """Build a tiny untrained GPT and persist it through the real checkpoint format."""
    torch.manual_seed(0)  # deterministic init so a given run is repeatable
    model = GPT(_CFG)
    save_checkpoint(path, model=model, cfg=_CFG, tokenizer=_TOKENIZER, step=0, val_loss=0.0)


@pytest.fixture
def checkpoint_path(tmp_path: Path) -> Path:
    path = tmp_path / "yegpt-ckpt.pt"
    _save_tiny_checkpoint(path)
    return path


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


def test_prompt_mode_streams_exact_seeded_output(
    checkpoint_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Reference: the non-streaming sampler with the demo's DEFAULT knobs and the same seed. The
    # CLI passes no knob flags, so equality proves both the streaming path and the defaults.
    ckpt = load_checkpoint(checkpoint_path)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    expected = sample_from_checkpoint(
        ckpt,
        prompt="ye",
        max_new_tokens=32,
        device=torch.device("cpu"),
        generator=generator,
        temperature=0.9,
        top_p=0.92,
        repetition_penalty=1.3,
    )

    cli.main(["ye", "--checkpoint", str(checkpoint_path), "--seed", "0", "--max-chars", "32"])

    assert capsys.readouterr().out == expected + "\n"


def test_prompt_mode_honors_explicit_knobs(
    checkpoint_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ckpt = load_checkpoint(checkpoint_path)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    expected = sample_from_checkpoint(
        ckpt,
        prompt="taught",
        max_new_tokens=16,
        device=torch.device("cpu"),
        generator=generator,
        temperature=1.1,
        top_k=5,
        top_p=0.8,
        repetition_penalty=1.05,
    )

    cli.main(
        [
            "taught",
            "--checkpoint", str(checkpoint_path),
            "--seed", "7",
            "--max-chars", "16",
            "--temperature", "1.1",
            "--top-k", "5",
            "--top-p", "0.8",
            "--repetition-penalty", "1.05",
        ]
    )

    assert capsys.readouterr().out == expected + "\n"


def test_prompt_mode_invalid_temperature_exits_nonzero(checkpoint_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["ye", "--checkpoint", str(checkpoint_path), "--temperature", "-1.0"])
    assert excinfo.value.code not in (0, None)


def test_prompt_mode_invalid_top_p_exits_nonzero(checkpoint_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["ye", "--checkpoint", str(checkpoint_path), "--top-p", "1.5"])
    assert excinfo.value.code not in (0, None)


def test_prompt_mode_negative_max_chars_exits_nonzero(checkpoint_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["ye", "--checkpoint", str(checkpoint_path), "--max-chars", "-3"])
    assert excinfo.value.code not in (0, None)


def test_prompt_mode_missing_checkpoint_exits_nonzero(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["ye", "--checkpoint", str(tmp_path / "missing.pt")])
    assert excinfo.value.code not in (0, None)


def test_prompt_mode_out_of_vocab_prompt_exits_nonzero(checkpoint_path: Path) -> None:
    # "@" is not in the tiny corpus vocab; the tokenizer's ValueError must become a clean exit.
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["@@@", "--checkpoint", str(checkpoint_path), "--max-chars", "4"])
    assert excinfo.value.code not in (0, None)
