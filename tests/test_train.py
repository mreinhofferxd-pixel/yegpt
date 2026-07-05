"""Tests for the training loop (TICKET-07).

A fast, CPU-only smoke test: a tiny in-memory corpus, a tiny model, a couple hundred cheap
iters. Asserts the loop runs end to end, writes a checkpoint that round-trips back into a
working model, produces finite losses, and that the val loss at the end beats the untrained
baseline (the loop actually learns). Nothing here needs CUDA — `_autocast` is a no-op on CPU.
"""

import math
from pathlib import Path

import pytest
import torch

from yegpt.config import TrainConfig
from yegpt.data_prep import DEFAULT_CORPUS_PATH
from yegpt.model import GPT
from yegpt.tokenizer import CharTokenizer
from yegpt.train import (
    DEFAULT_CHECKPOINT_DIR,
    Checkpoint,
    TrainInvocation,
    load_checkpoint,
    parse_train_args,
    resolve_device,
    train,
)

# Repetitive corpus over a small vocab: learnable in a few steps so the loss visibly drops.
_CORPUS = "yeezy taught me " * 64
# Tiny + cpu, with a higher lr than the 3e-4 default so the drop is unambiguous in ~200 steps.
_CFG = TrainConfig(
    n_layer=2,
    n_head=2,
    n_embd=16,
    block_size=8,
    dropout=0.0,
    batch_size=8,
    lr=5e-3,
    max_iters=200,
    eval_interval=100,
    device="cpu",
    seed=0,
)


def test_train_runs_and_writes_checkpoint(tmp_path: Path) -> None:
    result = train(_CFG, text=_CORPUS, checkpoint_dir=tmp_path, eval_batches=10)
    assert result.checkpoint_path == tmp_path / "yegpt-ckpt.pt"
    assert result.checkpoint_path.exists()


def test_losses_are_finite_and_decrease(tmp_path: Path) -> None:
    result = train(_CFG, text=_CORPUS, checkpoint_dir=tmp_path, eval_batches=10)
    assert len(result.history) >= 2  # at least the step-0 baseline and the final eval
    for point in result.history:
        assert math.isfinite(point.train_loss)
        assert math.isfinite(point.val_loss)
    # The whole point of the loop: the trained model beats its untrained baseline.
    assert result.final_loss < result.start_loss


def test_checkpoint_round_trips_into_a_working_model(tmp_path: Path) -> None:
    result = train(_CFG, text=_CORPUS, checkpoint_dir=tmp_path, eval_batches=10)
    ckpt = load_checkpoint(result.checkpoint_path)
    assert isinstance(ckpt, Checkpoint)

    # The stored config has vocab filled in from the corpus, and the vocab ordering matches.
    tokenizer = CharTokenizer.from_text(_CORPUS)
    assert ckpt.config.vocab_size == tokenizer.vocab_size
    assert ckpt.vocab == tokenizer.itos
    assert ckpt.step == _CFG.max_iters

    # The weights load cleanly into a freshly-built model of that config, which then runs.
    model = GPT(ckpt.config)
    model.load_state_dict(ckpt.model_state)
    logits, _ = model(torch.zeros((1, _CFG.block_size), dtype=torch.long))
    assert logits.shape == (1, _CFG.block_size, ckpt.config.vocab_size)


def test_best_checkpoint_tracks_the_min_val(tmp_path: Path) -> None:
    result = train(_CFG, text=_CORPUS, checkpoint_dir=tmp_path, eval_batches=10)

    # The best snapshot is the minimum val over the whole recorded curve (first-wins on ties,
    # matching the strict-improvement check in the loop and `min`'s first-min behavior).
    best_point = min(result.history, key=lambda point: point.val_loss)
    assert result.best_loss == best_point.val_loss
    assert result.best_step == best_point.step
    assert result.best_loss <= result.final_loss  # best is over all evals, incl. the final one

    # It lands in its own file, distinct from the final checkpoint.
    assert result.best_checkpoint_path == tmp_path / "yegpt-best.pt"
    assert result.best_checkpoint_path != result.checkpoint_path
    assert result.best_checkpoint_path.exists()

    # The best checkpoint round-trips and is stamped with the step/val it was saved at.
    ckpt = load_checkpoint(result.best_checkpoint_path)
    assert ckpt.step == result.best_step
    assert ckpt.val_loss == pytest.approx(result.best_loss)


def test_throughput_metrics_present_finite_and_nonnegative(tmp_path: Path) -> None:
    # TICKET-10.0 instrumentation: the loop now surfaces throughput + peak VRAM so 10.1 can report
    # where the 4080 caps out. Assert the fields exist, are finite/non-negative, and stay internally
    # consistent. This is a CPU run, so it must touch no CUDA at all.
    result = train(_CFG, text=_CORPUS, checkpoint_dir=tmp_path, eval_batches=10)

    assert math.isfinite(result.steps_per_sec)
    assert math.isfinite(result.tokens_per_sec)
    assert result.steps_per_sec >= 0.0
    assert result.tokens_per_sec >= 0.0
    # tokens/sec is just steps/sec scaled by the tokens processed per step.
    assert result.tokens_per_sec == pytest.approx(
        result.steps_per_sec * _CFG.batch_size * _CFG.block_size
    )

    # A CPU run never queries CUDA, so peak VRAM is reported as exactly 0 (keeps the smoke test
    # CUDA-free) and stays a plain int.
    assert isinstance(result.peak_vram_bytes, int)
    assert result.peak_vram_bytes == 0


def test_resolve_device_honors_cuda_availability() -> None:
    # Branches on the box this runs on: cuda when present, a loud error when requested but not.
    if torch.cuda.is_available():
        assert resolve_device(TrainConfig(device="cuda")).type == "cuda"
    else:
        with pytest.raises(RuntimeError, match="CUDA is unavailable"):
            resolve_device(TrainConfig(device="cuda"))
    assert resolve_device(TrainConfig(device="cpu")).type == "cpu"


def test_load_checkpoint_rejects_foreign_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"format": "something-else"}, path)
    with pytest.raises(ValueError, match="not a yegpt-checkpoint"):
        load_checkpoint(path)


def test_parse_train_args_overrides_every_field(tmp_path: Path) -> None:
    # Every flag set to something distinct from the defaults, so a missed mapping shows up.
    inv = parse_train_args(
        [
            "--n-layer", "6",
            "--n-head", "8",
            "--n-embd", "256",
            "--block-size", "256",
            "--dropout", "0.2",
            "--lr", "1e-3",
            "--max-iters", "1234",
            "--eval-interval", "250",
            "--batch-size", "32",
            "--device", "cpu",
            "--seed", "42",
            "--out-dir", str(tmp_path / "run1"),
            "--corpus", str(tmp_path / "mycorpus.txt"),
        ]
    )
    assert isinstance(inv, TrainInvocation)
    cfg = inv.config
    assert (cfg.n_layer, cfg.n_head, cfg.n_embd) == (6, 8, 256)
    assert (cfg.block_size, cfg.dropout, cfg.lr) == (256, 0.2, 1e-3)
    assert (cfg.max_iters, cfg.eval_interval, cfg.batch_size) == (1234, 250, 32)
    assert (cfg.device, cfg.seed) == ("cpu", 42)
    assert inv.checkpoint_dir == tmp_path / "run1"
    assert inv.corpus_path == tmp_path / "mycorpus.txt"
    # vocab_size is not a flag; it stays the sentinel until train() derives it from the corpus.
    assert not cfg.vocab_is_set


def test_parse_train_args_defaults_match_dataclass() -> None:
    # No flags => config equals the dataclass defaults; paths fall back to the canonical ones.
    inv = parse_train_args([])
    assert inv.config == TrainConfig()
    assert inv.checkpoint_dir == DEFAULT_CHECKPOINT_DIR
    assert inv.corpus_path == DEFAULT_CORPUS_PATH
