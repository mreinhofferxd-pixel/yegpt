"""dataset: corpus text -> reproducible, batched (x, y) tensors for next-char training.

The job (SPEC.md §5): encode the corpus to one long int64 tensor, carve a train/val
split, and hand out random `(x, y)` batches where `y` is `x` shifted one character to
the right — i.e. for every position, the target is "the next character".

Design choices:
- **Dependency injection, no globals.** The dataset receives a `TrainConfig` and either
  raw text + a `CharTokenizer` or pre-built split tensors. Nothing here reads a singleton.
- **Batches are built on CPU.** `get_batch` returns CPU tensors and draws indices from a
  CPU `torch.Generator`; `train.py` owns the single `.to(device)` hop. This keeps the
  dataset device-agnostic and sidesteps the extra reproducibility rules CUDA generators
  carry. The cost — one host->device copy per step — is negligible next to the forward pass.
- **Stateful by necessity, so a plain class (not a frozen dataclass).** Each `get_batch`
  advances the generator, so successive batches differ; the *sequence* of batches is fully
  determined by `cfg.seed`. Two datasets built with the same seed and data yield identical
  batch sequences. A frozen dataclass would misrepresent that mutating RNG state.
- **Vectorized index gather.** Start positions are sampled once, broadcast against
  `arange(block_size)`, and used for a single advanced-index gather — no per-row Python
  slicing. The `y = x shifted by one` invariant then falls straight out of the offsets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

import torch
from torch import Tensor

from yegpt.config import TrainConfig
from yegpt.data_prep import DEFAULT_CORPUS_PATH
from yegpt.tokenizer import CharTokenizer

# The only two valid splits; Literal lets the type checker catch typos at call sites.
Split = Literal["train", "val"]

_DEFAULT_TRAIN_FRAC: Final[float] = 0.9


def encode_to_tensor(tokenizer: CharTokenizer, text: str) -> Tensor:
    """Encode `text` to a 1-D `torch.long` (int64) tensor of token ids."""
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)


def train_val_split(
    data: Tensor, train_frac: float = _DEFAULT_TRAIN_FRAC
) -> tuple[Tensor, Tensor]:
    """Split a 1-D token tensor into (train, val) by a contiguous cut at `train_frac`.

    Contiguous (not shuffled) on purpose: the data is a sequence and batches sample
    windows from it, so a single split point preserves local order within each part.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0.0, 1.0), got {train_frac}.")
    cut = int(train_frac * data.numel())
    return data[:cut], data[cut:]


class CharDataset:
    """Holds the encoded train/val splits and serves reproducible `(x, y)` batches."""

    def __init__(self, cfg: TrainConfig, train_data: Tensor, val_data: Tensor) -> None:
        """Validate the splits and seed the batch-sampling RNG from `cfg.seed`.

        Each split must hold at least `block_size + 1` tokens — one full context window
        (`x`) plus the one-position-shifted target window (`y`).
        """
        min_tokens = cfg.block_size + 1
        for name, data in (("train", train_data), ("val", val_data)):
            if data.ndim != 1:
                raise ValueError(
                    f"{name} split must be 1-D, got shape {tuple(data.shape)}."
                )
            if data.dtype != torch.long:
                raise ValueError(
                    f"{name} split must be torch.long, got {data.dtype}."
                )
            if data.numel() < min_tokens:
                raise ValueError(
                    f"{name} split has {data.numel()} tokens; need at least "
                    f"block_size + 1 = {min_tokens}. Use a larger corpus or a smaller "
                    f"block_size (SPEC.md §5)."
                )
        self._cfg: Final = cfg
        self._train: Final = train_data
        self._val: Final = val_data
        # A dedicated CPU generator makes batch sampling reproducible and isolated from
        # the global torch RNG (which model init / dropout will also draw from).
        self._generator: Final = torch.Generator()
        self._generator.manual_seed(cfg.seed)

    @classmethod
    def from_text(
        cls,
        cfg: TrainConfig,
        tokenizer: CharTokenizer,
        text: str,
        train_frac: float = _DEFAULT_TRAIN_FRAC,
    ) -> CharDataset:
        """Build directly from in-memory text (used by tests and any non-file caller)."""
        train, val = train_val_split(encode_to_tensor(tokenizer, text), train_frac)
        return cls(cfg, train, val)

    @classmethod
    def from_corpus(
        cls,
        cfg: TrainConfig,
        tokenizer: CharTokenizer,
        corpus_path: Path = DEFAULT_CORPUS_PATH,
        train_frac: float = _DEFAULT_TRAIN_FRAC,
    ) -> CharDataset:
        """Build from a corpus file on disk (the `train.py` path)."""
        return cls.from_text(
            cfg, tokenizer, corpus_path.read_text(encoding="utf-8"), train_frac
        )

    @property
    def train_tokens(self) -> int:
        return self._train.numel()

    @property
    def val_tokens(self) -> int:
        return self._val.numel()

    def get_batch(self, split: Split) -> tuple[Tensor, Tensor]:
        """Return one `(x, y)` batch of shape `(batch_size, block_size)`, both `torch.long`.

        `x[b, t]` is a context character and `y[b, t]` is the character that follows it,
        so `y` is exactly `x` shifted one step right. Tensors stay on CPU.
        """
        data = self._train if split == "train" else self._val
        block, batch = self._cfg.block_size, self._cfg.batch_size

        # Sample `batch` start positions. The last legal start is `numel - block - 1`
        # (so that `start + block` still has a target), and randint's bound is exclusive.
        starts = torch.randint(
            data.numel() - block, (batch,), generator=self._generator
        )
        # (batch, 1) starts + (1, block) offsets -> (batch, block) absolute indices.
        idx = starts[:, None] + torch.arange(block)[None, :]
        x = data[idx]
        y = data[idx + 1]  # same windows shifted by one => next-char targets
        return x, y

    def __repr__(self) -> str:
        return (
            f"CharDataset(train_tokens={self.train_tokens}, "
            f"val_tokens={self.val_tokens}, block_size={self._cfg.block_size})"
        )
