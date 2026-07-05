"""Tests for the batch fragment export (backlog Unit 4).

Fast and CPU-only. We build a tiny untrained GPT, save it through the real `train.save_checkpoint`,
and drive both the pure `generate_samples` core and the `export_samples` disk path. The model is
untrained, so the *text* is noise; these tests assert the export *harness* -- fragment
count/length, vocab-closure, per-fragment variety, batch reproducibility under a seed, the
profanity screen, and the on-disk JSON shape -- which is exactly what this export delivers (output
quality is out of scope per SPEC.md 0).
"""

import json
from pathlib import Path

import pytest
import torch

from yegpt.config import TrainConfig
from yegpt.export_samples import (
    build_document,
    contains_profanity,
    export_samples,
    filter_profanity,
    generate_samples,
)
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


def test_profanity_filter_drops_flagged_fragment() -> None:
    clean = "yeezy taught me the way"
    dirty = "this line has shit in it"
    assert contains_profanity(dirty)
    assert not contains_profanity(clean)
    assert filter_profanity([clean, dirty, clean]) == [clean, clean]


def test_profanity_filter_matches_whole_words_only() -> None:
    # A wordlist term buried in an unrelated word (dickens, cocktail) must not trip the screen.
    assert not contains_profanity("charles dickens wrote it")
    assert not contains_profanity("a cocktail party")


def test_profanity_filter_catches_inflections() -> None:
    # Common inflected forms of a base term are still profanity and must be caught.
    assert contains_profanity("this is fucking wild")
    assert contains_profanity("he shits on it")


def test_build_document_shape() -> None:
    doc = build_document(
        ["frag one", "frag two"],
        model="checkpoints/run3/yegpt-ckpt.pt",
        seed=1234,
        temperature=0.9,
        top_k=None,
        top_p=0.92,
        repetition_penalty=1.3,
        max_new_tokens=200,
        prompt="",
        profanity_filter=True,
    )
    assert doc["samples"] == ["frag one", "frag two"]
    generated_with = doc["generated_with"]
    assert isinstance(generated_with, dict)
    assert generated_with["model"] == "checkpoints/run3/yegpt-ckpt.pt"
    assert generated_with["seed"] == 1234
    knobs = generated_with["knobs"]
    assert isinstance(knobs, dict)
    assert knobs["num_samples"] == 2
    assert knobs["profanity_filter"] is True
    assert knobs["top_p"] == 0.92


def test_export_writes_valid_json_with_documented_shape(tmp_path: Path) -> None:
    source = tmp_path / "yegpt-ckpt.pt"
    dest = tmp_path / "web" / "samples.json"  # nested dir must be created by the export
    _save_tiny_checkpoint(source)

    result = export_samples(
        source, dest, num_samples=3, max_new_tokens=16, seed=0, model_name="tiny-test"
    )

    assert dest.exists()
    assert result.dest_path == dest
    assert result.num_samples == 3
    assert result.dest_bytes == dest.stat().st_size

    doc = json.loads(dest.read_text(encoding="utf-8"))
    assert set(doc) == {"generated_with", "samples"}
    assert doc["generated_with"]["model"] == "tiny-test"
    assert doc["generated_with"]["seed"] == 0
    assert isinstance(doc["samples"], list)
    assert len(doc["samples"]) == 3
    vocab = set(_TOKENIZER.itos)
    for fragment in doc["samples"]:
        assert all(ch in vocab for ch in fragment)


def test_export_json_is_reproducible(tmp_path: Path) -> None:
    source = tmp_path / "yegpt-ckpt.pt"
    _save_tiny_checkpoint(source)

    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    export_samples(source, first, num_samples=3, max_new_tokens=16, seed=42, model_name="m")
    export_samples(source, second, num_samples=3, max_new_tokens=16, seed=42, model_name="m")

    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
