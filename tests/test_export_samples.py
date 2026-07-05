"""Tests for the batch fragment export (backlog Unit 2).

Fast and CPU-only. We build a tiny untrained GPT, save it through the real `train.save_checkpoint`,
and drive both the pure `generate_samples` core and the `export_samples` disk path. The model is
untrained, so the *text* is noise; these tests assert the batch *harness* -- fragment count/length,
vocab-closure, per-fragment variety, batch reproducibility under a seed, and the on-disk file --
which is exactly what this export delivers (output quality is out of scope per SPEC.md §0).
"""

from pathlib import Path

import pytest
import torch

from yegpt.config import TrainConfig
from yegpt.export_samples import export_samples, format_samples, generate_samples
from yegpt.model import GPT
from yegpt.tokenizer import CharTokenizer
from yegpt.train import Checkpoint, load_checkpoint, save_checkpoint

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

_CPU = torch.device("cpu")


def _save_tiny_checkpoint(path: Path) -> None:
    """Build a tiny untrained GPT and persist it through the real checkpoint format."""
    torch.manual_seed(0)  # deterministic init so a given run is repeatable
    model = GPT(_CFG)
    save_checkpoint(path, model=model, cfg=_CFG, tokenizer=_TOKENIZER, step=0, val_loss=0.0)


@pytest.fixture
def checkpoint(tmp_path: Path) -> Checkpoint:
    path = tmp_path / "yegpt-ckpt.pt"
    _save_tiny_checkpoint(path)
    return load_checkpoint(path)


def test_returns_requested_count_of_prompt_prefixed_fragments(checkpoint: Checkpoint) -> None:
    prompt = "ye"
    samples = generate_samples(
        checkpoint, num_samples=5, max_new_tokens=32, device=_CPU, prompt=prompt, seed=0
    )
    assert len(samples) == 5
    vocab = set(checkpoint.vocab)
    for text in samples:
        assert text.startswith(prompt)
        assert len(text) == len(prompt) + 32
        assert all(ch in vocab for ch in text)


def test_same_seed_reproduces_whole_batch(checkpoint: Checkpoint) -> None:
    first = generate_samples(checkpoint, num_samples=4, max_new_tokens=24, device=_CPU, seed=123)
    second = generate_samples(checkpoint, num_samples=4, max_new_tokens=24, device=_CPU, seed=123)
    assert first == second


def test_different_seed_changes_batch(checkpoint: Checkpoint) -> None:
    first = generate_samples(checkpoint, num_samples=4, max_new_tokens=24, device=_CPU, seed=1)
    second = generate_samples(checkpoint, num_samples=4, max_new_tokens=24, device=_CPU, seed=2)
    assert first != second


def test_fragments_within_a_batch_differ(checkpoint: Checkpoint) -> None:
    # Per-fragment seed offset (seed+i) means the fragments are drawn differently, so a seeded
    # batch is not a list of identical strings.
    samples = generate_samples(checkpoint, num_samples=3, max_new_tokens=32, device=_CPU, seed=7)
    assert samples[0] != samples[1]


def test_zero_samples_is_empty(checkpoint: Checkpoint) -> None:
    assert generate_samples(checkpoint, num_samples=0, max_new_tokens=16, device=_CPU) == []


def test_negative_samples_raises(checkpoint: Checkpoint) -> None:
    with pytest.raises(ValueError, match="num_samples"):
        generate_samples(checkpoint, num_samples=-1, max_new_tokens=16, device=_CPU)


def test_format_samples_numbers_blocks() -> None:
    rendered = format_samples(["alpha", "beta"])
    assert rendered == "--- sample 1 ---\nalpha\n\n--- sample 2 ---\nbeta\n"


def test_export_writes_file_and_reports_count(tmp_path: Path) -> None:
    source = tmp_path / "yegpt-ckpt.pt"
    dest = tmp_path / "dist" / "yegpt-samples.txt"  # nested dir must be created by the export
    _save_tiny_checkpoint(source)

    result = export_samples(source, dest, num_samples=3, max_new_tokens=16, seed=0)

    assert dest.exists()
    assert result.num_samples == 3
    assert result.dest_path == dest
    assert result.dest_bytes == dest.stat().st_size
    text = dest.read_text(encoding="utf-8")
    assert "--- sample 1 ---" in text
    assert "--- sample 3 ---" in text


def test_export_disk_path_is_reproducible(tmp_path: Path) -> None:
    source = tmp_path / "yegpt-ckpt.pt"
    _save_tiny_checkpoint(source)

    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    export_samples(source, first, num_samples=3, max_new_tokens=16, seed=42)
    export_samples(source, second, num_samples=3, max_new_tokens=16, seed=42)

    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
