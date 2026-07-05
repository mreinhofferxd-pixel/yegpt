"""TrainConfig: the one place every knob lives.

Design choices:
- **Frozen + slots.** The config is immutable once built, so it can be passed around by
  dependency injection without anyone mutating shared state mid-run. `vocab_size` is the
  one value not known at authoring time (it depends on the corpus), so it starts at a
  sentinel and is filled with `with_vocab_size(...)`, which returns a *new* frozen config.
- **No globals.** Modules receive a `TrainConfig` argument; nothing reads a module-level
  singleton.
- **Defaults sized to prove the loop fast on a 4080**, not to maximize quality. This is
  the "start smaller (~1M params)" config from the spec — it trains in minutes so the
  pipeline can be validated before scaling toward ~10M (SPEC.md §2).
"""

from dataclasses import dataclass, replace
from typing import Final

import torch

# Sentinel for "vocab not known yet". A real vocab is always >= 1, so 0 is unambiguous.
_VOCAB_UNSET: Final[int] = 0


def _default_device() -> str:
    """Pick CUDA when present (the 4080), else CPU. Override explicitly via the field."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """All hyperparameters and runtime settings for a yeGPT training run.

    Defaults target the RTX 4080 and a *fast first loop* (~0.8M params), not best quality.
    """

    # --- Architecture ---
    n_layer: int = 4
    """Number of transformer blocks stacked in the model."""
    n_head: int = 4
    """Number of attention heads per block. Must divide `n_embd` evenly."""
    n_embd: int = 128
    """Embedding/residual-stream width. Per-head size is `n_embd // n_head`."""
    block_size: int = 128
    """Context length in characters (start 128, target 256 per SPEC.md §2)."""
    dropout: float = 0.1
    """Dropout probability used in attention and the MLP. 0.0 disables it."""

    # --- Optimization ---
    batch_size: int = 64
    """Sequences per training step. Fits the 4080 comfortably at this width."""
    lr: float = 3e-4
    """AdamW learning rate. 3e-4 is the standard nanoGPT-class starting point."""
    max_iters: int = 5000
    """Total optimizer steps. Small enough to finish a first run in minutes."""
    eval_interval: int = 500
    """Run a train/val loss estimate every this many steps."""

    # --- Runtime ---
    vocab_size: int = _VOCAB_UNSET
    """Number of distinct characters; filled at runtime via `with_vocab_size`."""
    device: str = "cuda"
    """Torch device string, e.g. "cuda", "cuda:0", "cpu". Defaults to the 4080."""
    seed: int = 1337
    """Single RNG seed for reproducible batches, init, and sampling."""

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})."
            )
        for name, value in (
            ("n_layer", self.n_layer),
            ("n_head", self.n_head),
            ("n_embd", self.n_embd),
            ("block_size", self.block_size),
            ("batch_size", self.batch_size),
            ("max_iters", self.max_iters),
            ("eval_interval", self.eval_interval),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}.")
        if self.lr <= 0.0:
            raise ValueError(f"lr must be positive, got {self.lr}.")
        # vocab_size may legitimately be the sentinel (pre-corpus) or a real positive value.
        if self.vocab_size < 0:
            raise ValueError(f"vocab_size must be >= 0, got {self.vocab_size}.")

    @property
    def head_size(self) -> int:
        """Per-head channel width."""
        return self.n_embd // self.n_head

    @property
    def vocab_is_set(self) -> bool:
        return self.vocab_size != _VOCAB_UNSET

    def with_vocab_size(self, vocab_size: int) -> "TrainConfig":
        """Return a new config with `vocab_size` populated (the frozen-friendly setter)."""
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}.")
        return replace(self, vocab_size=vocab_size)


def default_device() -> str:
    """Convenience for callers that want runtime device autodetection (cuda/cpu)."""
    return _default_device()
