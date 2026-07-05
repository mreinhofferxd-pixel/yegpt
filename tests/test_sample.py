"""Tests for sampling from a checkpoint (TICKET-08).

Fast and CPU-only — no CUDA, no training. We build a tiny GPT, save it through the real
`train.save_checkpoint`, load it back with `load_checkpoint`, and drive the pure
`sample_from_checkpoint` core. The model is untrained, so the *text* is noise; these tests
assert the sampling *harness* is correct — output type/length, vocab-closure, the prompt-prefix
invariant, reproducibility under a seed, and clean error surfacing — which is exactly what
TICKET-08 delivers. (That the output is noise, not lyrics, is the expected scope per SPEC.md §0.)
"""

from pathlib import Path

import pytest
import torch

from yegpt.config import TrainConfig
from yegpt.model import GPT
from yegpt.sample import generate_text, sample_from_checkpoint
from yegpt.tokenizer import CharTokenizer
from yegpt.train import Checkpoint, load_checkpoint, save_checkpoint

# Small vocab that includes a newline, which exercises the empty-prompt "\n" start path.
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


def _seeded(seed: int) -> torch.Generator:
    generator = torch.Generator()  # CPU generator, matching the CPU sampling device
    generator.manual_seed(seed)
    return generator


def test_output_is_str_with_expected_length(checkpoint: Checkpoint) -> None:
    prompt = "yeezy "
    out = sample_from_checkpoint(checkpoint, prompt=prompt, max_new_tokens=40, device=_CPU)
    assert isinstance(out, str)
    assert len(out) == len(prompt) + 40


def test_every_output_char_is_in_vocab(checkpoint: Checkpoint) -> None:
    out = sample_from_checkpoint(checkpoint, prompt="ye", max_new_tokens=64, device=_CPU)
    vocab = set(checkpoint.vocab)
    assert all(ch in vocab for ch in out)


def test_prompt_is_a_prefix_of_output(checkpoint: Checkpoint) -> None:
    prompt = "taught me"
    out = sample_from_checkpoint(checkpoint, prompt=prompt, max_new_tokens=32, device=_CPU)
    assert out.startswith(prompt)


def test_empty_prompt_drops_primer_and_keeps_length(checkpoint: Checkpoint) -> None:
    # No prompt: the synthetic primer is dropped, so length is exactly max_new_tokens.
    out = sample_from_checkpoint(checkpoint, prompt="", max_new_tokens=48, device=_CPU)
    assert len(out) == 48
    assert all(ch in set(checkpoint.vocab) for ch in out)


def test_same_seed_reproduces_output(checkpoint: Checkpoint) -> None:
    out1 = sample_from_checkpoint(
        checkpoint, prompt="ye", max_new_tokens=64, device=_CPU, generator=_seeded(123)
    )
    out2 = sample_from_checkpoint(
        checkpoint, prompt="ye", max_new_tokens=64, device=_CPU, generator=_seeded(123)
    )
    assert out1 == out2


def test_different_seed_changes_output(checkpoint: Checkpoint) -> None:
    out1 = sample_from_checkpoint(
        checkpoint, prompt="ye", max_new_tokens=64, device=_CPU, generator=_seeded(1)
    )
    out2 = sample_from_checkpoint(
        checkpoint, prompt="ye", max_new_tokens=64, device=_CPU, generator=_seeded(2)
    )
    assert out1 != out2


def test_top_k_one_is_deterministic_through_sampler(checkpoint: Checkpoint) -> None:
    # top_k=1 is greedy, so the generator can't matter -- two different seeds must agree. This
    # only holds if sample_from_checkpoint actually threads top_k into model.generate.
    out1 = sample_from_checkpoint(
        checkpoint, prompt="ye", max_new_tokens=40, device=_CPU, generator=_seeded(1), top_k=1
    )
    out2 = sample_from_checkpoint(
        checkpoint, prompt="ye", max_new_tokens=40, device=_CPU, generator=_seeded(2), top_k=1
    )
    assert out1 == out2


def test_tiny_top_p_is_deterministic_through_sampler(checkpoint: Checkpoint) -> None:
    # A near-zero top_p keeps only the argmax each step (greedy), so the generator can't matter --
    # two different seeds must agree. This only holds if sample_from_checkpoint threads top_p in.
    out1 = sample_from_checkpoint(
        checkpoint, prompt="ye", max_new_tokens=40, device=_CPU, generator=_seeded(1), top_p=0.01
    )
    out2 = sample_from_checkpoint(
        checkpoint, prompt="ye", max_new_tokens=40, device=_CPU, generator=_seeded(2), top_p=0.01
    )
    assert out1 == out2


def test_repetition_penalty_threads_through(checkpoint: Checkpoint) -> None:
    # model.generate validates repetition_penalty (> 0), so an invalid value only raises if the
    # arg is actually threaded through the sampler. (On this untrained model the penalty's *effect*
    # is a near-no-op -- logits are ~0 -- so its behaviour is verified deterministically in
    # test_model.py; here we just confirm the wiring.)
    with pytest.raises(ValueError, match="repetition_penalty"):
        sample_from_checkpoint(
            checkpoint, prompt="ye", max_new_tokens=8, device=_CPU, repetition_penalty=0.0
        )


def test_out_of_vocab_prompt_raises(checkpoint: Checkpoint) -> None:
    # 'Q' never appears in the corpus, so encoding the prompt must fail loudly and cleanly.
    with pytest.raises(ValueError, match="not in vocab"):
        sample_from_checkpoint(checkpoint, prompt="Q", max_new_tokens=8, device=_CPU)


def test_generate_text_disk_path_is_reproducible(tmp_path: Path) -> None:
    # Exercises the thin disk wrapper end to end: same seed through the path -> identical text.
    path = tmp_path / "yegpt-ckpt.pt"
    _save_tiny_checkpoint(path)

    out1 = generate_text(path, prompt="ye", max_new_tokens=48, seed=7)
    out2 = generate_text(path, prompt="ye", max_new_tokens=48, seed=7)
    assert out1 == out2
    assert out1.startswith("ye")
    assert len(out1) == len("ye") + 48
