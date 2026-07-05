"""Tests for CharDataset (TICKET-05): shapes, dtypes, the next-char shift, reproducibility.

All tests build from a small in-memory corpus — none depend on a real data/corpus.txt.
A small block_size/batch_size keeps the required corpus tiny and the tests fast.
"""

from pathlib import Path

import pytest
import torch

from yegpt.config import TrainConfig
from yegpt.dataset import CharDataset, encode_to_tensor, train_val_split
from yegpt.tokenizer import CharTokenizer

# 200 chars over a 10-char vocab: train ~180, val ~20 — both well above block_size + 1.
_CORPUS = "abcdefghij" * 20
_CFG = TrainConfig(block_size=8, batch_size=4, seed=1234)


def _make_dataset(text: str = _CORPUS, cfg: TrainConfig = _CFG) -> CharDataset:
    tok = CharTokenizer.from_text(text)
    return CharDataset.from_text(cfg, tok, text)


def test_encode_to_tensor_is_long_1d() -> None:
    tok = CharTokenizer.from_text("abc")
    t = encode_to_tensor(tok, "cab")
    assert t.dtype == torch.long
    assert t.ndim == 1
    assert t.tolist() == [2, 0, 1]  # sorted vocab -> a=0, b=1, c=2


def test_train_val_split_is_contiguous_and_proportional() -> None:
    data = torch.arange(100, dtype=torch.long)
    train, val = train_val_split(data, train_frac=0.9)
    assert train.numel() == 90
    assert val.numel() == 10
    # Contiguous cut: concatenating the parts reproduces the original sequence.
    assert torch.equal(torch.cat([train, val]), data)


@pytest.mark.parametrize("split", ["train", "val"])
def test_batch_shapes_and_dtype(split: str) -> None:
    ds = _make_dataset()
    x, y = ds.get_batch(split)  # type: ignore[arg-type]  # exercising both Literal values
    assert x.shape == (_CFG.batch_size, _CFG.block_size)
    assert y.shape == (_CFG.batch_size, _CFG.block_size)
    assert x.dtype == torch.long
    assert y.dtype == torch.long
    # Batches stay on CPU; train.py owns the move to the device.
    assert x.device.type == "cpu"
    assert y.device.type == "cpu"


def test_y_is_x_shifted_by_one() -> None:
    ds = _make_dataset()
    x, y = ds.get_batch("train")
    # The seed-independent invariant: y is x advanced by exactly one position.
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_batch_indices_are_in_range() -> None:
    ds = _make_dataset()
    x, y = ds.get_batch("val")
    vocab_size = 10
    assert int(x.min()) >= 0
    assert int(x.max()) < vocab_size
    assert int(y.max()) < vocab_size


def test_same_seed_yields_identical_batches() -> None:
    a = _make_dataset()
    b = _make_dataset()
    # Identical across several consecutive draws, not just the first.
    for _ in range(3):
        xa, ya = a.get_batch("train")
        xb, yb = b.get_batch("train")
        assert torch.equal(xa, xb)
        assert torch.equal(ya, yb)


def test_different_seed_yields_different_batches() -> None:
    a = _make_dataset(cfg=TrainConfig(block_size=8, batch_size=4, seed=1))
    b = _make_dataset(cfg=TrainConfig(block_size=8, batch_size=4, seed=2))
    # Different seeds: the start positions differ, so the batches should differ.
    assert not torch.equal(a.get_batch("train")[0], b.get_batch("train")[0])


def test_split_too_small_raises() -> None:
    # 50 chars -> train 45, val 5; with block_size 8 the val split lacks block_size + 1.
    tok = CharTokenizer.from_text("abcde")
    with pytest.raises(ValueError, match="val split has 5 tokens"):
        CharDataset.from_text(_CFG, tok, "abcde" * 10)


def test_from_corpus_reads_file(tmp_path: Path) -> None:
    path = tmp_path / "corpus.txt"
    path.write_text(_CORPUS, encoding="utf-8")
    tok = CharTokenizer.from_text(_CORPUS)
    ds = CharDataset.from_corpus(_CFG, tok, corpus_path=path)
    x, y = ds.get_batch("train")
    assert x.shape == (_CFG.batch_size, _CFG.block_size)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_train_frac_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="train_frac"):
        train_val_split(torch.arange(10, dtype=torch.long), train_frac=1.0)
