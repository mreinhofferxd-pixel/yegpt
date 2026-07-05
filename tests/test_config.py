"""Tests for TrainConfig (TICKET-02): defaults, validation, immutability, vocab fill."""

import dataclasses
from collections.abc import Callable

import pytest

from yegpt.config import TrainConfig


def test_defaults_are_valid_and_consistent() -> None:
    cfg = TrainConfig()
    assert cfg.n_embd % cfg.n_head == 0
    assert cfg.head_size == cfg.n_embd // cfg.n_head
    assert not cfg.vocab_is_set  # vocab unknown until a corpus is loaded


def test_with_vocab_size_returns_new_populated_config() -> None:
    cfg = TrainConfig()
    filled = cfg.with_vocab_size(73)
    assert filled.vocab_size == 73
    assert filled.vocab_is_set
    assert cfg.vocab_size == 0  # original untouched
    assert filled is not cfg


def test_config_is_frozen() -> None:
    cfg = TrainConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.n_layer = 8  # type: ignore[misc]


@pytest.mark.parametrize(
    "build",
    [
        lambda: TrainConfig(n_embd=130, n_head=4),  # not divisible
        lambda: TrainConfig(n_layer=0),
        lambda: TrainConfig(block_size=-1),
        lambda: TrainConfig(dropout=1.0),
        lambda: TrainConfig(dropout=-0.1),
        lambda: TrainConfig(lr=0.0),
    ],
)
def test_invalid_configs_raise(build: Callable[[], TrainConfig]) -> None:
    with pytest.raises(ValueError):
        build()


def test_with_vocab_size_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        TrainConfig().with_vocab_size(0)
