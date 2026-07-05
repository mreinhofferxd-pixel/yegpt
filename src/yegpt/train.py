"""train: the loop that turns the hand-written GPT into a (styled-gibberish) model.

SPEC.md §5/§6, TICKET-07. This module owns three things nothing else in the project does:
the optimizer + step loop, the bf16 mixed-precision policy, and the on-disk **checkpoint
format** that TICKET-08's `sample.py` reads back to reconstruct the exact model.

Design choices:
- **Pure dependency injection.** Everything is built from a single injected `TrainConfig`:
  tokenizer <- corpus text, dataset <- (cfg, tokenizer, text), model <- cfg. No globals, no
  config singleton. The corpus source, the checkpoint directory, and (via `cfg.max_iters`)
  the run length are all overridable — which is exactly what lets the smoke test drive a
  tiny CPU loop through the same code path a real run uses.
- **The dataset stays on CPU (its design); the model moves once.** Each `(x, y)` batch is
  copied host->device per step — negligible next to the forward/backward — so the dataset
  keeps its own reproducible RNG and stays device-agnostic.
- **bf16 autocast is gated to CUDA.** bf16 autocast on CPU buys nothing and can be slower,
  so the CPU path (the smoke test) runs plain fp32 under a no-op context. No `GradScaler`:
  that exists to rescue tiny *fp16* gradients from underflow; bf16 keeps fp32's exponent
  range, so there is nothing to rescale.
- **Two checkpoints per run: final and best-val.** Every eval that lowers val loss snapshots the
  live weights to `yegpt-best.pt`; the final step always writes `yegpt-ckpt.pt`. Saving only the
  final is wrong for an overfitting run — that snapshot is its *most* memorized state — so the
  loop keeps the best-generalizing weights too, without losing the final that shows the overfit.

Honest scope (SPEC.md §0): this loop only minimizes next-character cross-entropy. Watching
that loss fall is the lesson — the end model is recognizably-Kanye-styled gibberish, not
coherent text.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from torch import Tensor

from yegpt.config import TrainConfig, default_device
from yegpt.data_prep import DEFAULT_CORPUS_PATH
from yegpt.dataset import CharDataset, Split
from yegpt.model import GPT
from yegpt.tokenizer import CharTokenizer

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_DIR: Final[Path] = _REPO_ROOT / "checkpoints"
_CHECKPOINT_NAME: Final[str] = "yegpt-ckpt.pt"
# Alongside the final-step checkpoint, the loop also writes the lowest-val snapshot here. On an
# overfitting run the two diverge sharply — the final weights are the most memorized, the best
# the most general — so keeping both lets you sample either (the TICKET-09 memorization lesson).
_BEST_CHECKPOINT_NAME: Final[str] = "yegpt-best.pt"
# The default file a no-arg run writes and `sample.py` reads back. Public so sample.py has one
# source of truth for the path instead of re-deriving the private filename.
DEFAULT_CHECKPOINT_PATH: Final[Path] = DEFAULT_CHECKPOINT_DIR / _CHECKPOINT_NAME

# Stamped into every payload so a load can reject a foreign / future-format file outright.
_CHECKPOINT_FORMAT: Final[str] = "yegpt-checkpoint-v1"

# Batches to average each train/val loss estimate over. A handful denoises the curve without
# letting eval dominate the step loop; the smoke test overrides it down for speed.
_DEFAULT_EVAL_BATCHES: Final[int] = 50


@dataclass(frozen=True, slots=True)
class LossPoint:
    """One row of the printed loss curve: the two split losses estimated at a given step."""

    step: int
    train_loss: float
    val_loss: float


@dataclass(frozen=True, slots=True)
class TrainResult:
    """What a finished run produced: where the checkpoints landed and how the curve moved.

    Two checkpoints are written: `checkpoint_path` holds the final-step weights and
    `best_checkpoint_path` holds the lowest-val snapshot seen during the run. They diverge
    exactly when the model overfits — the whole point to watch on this corpus.

    The trailing three fields are throughput/footprint instrumentation (TICKET-10.0), measured
    over the optimizer loop so the 10.1 scale ablation can report "where the 4080 caps out" —
    here that's time-per-checkpoint and context length, not param VRAM.
    """

    checkpoint_path: Path  # final-step weights
    best_checkpoint_path: Path  # lowest-val snapshot (often an earlier step than the final)
    start_loss: float  # first val estimate (untrained baseline)
    final_loss: float  # val estimate after the last optimizer step
    best_loss: float  # lowest val estimate seen (matches best_checkpoint_path)
    best_step: int  # step at which best_loss was observed
    history: tuple[LossPoint, ...]
    # Measured over the full optimizer loop, periodic eval included (that eval is part of the
    # real time-per-checkpoint, and 10.1 fixes eval cadence across runs so the comparison holds).
    steps_per_sec: float  # cfg.max_iters / loop wall-clock
    tokens_per_sec: float  # steps_per_sec * batch_size * block_size
    peak_vram_bytes: int  # torch.cuda.max_memory_allocated over the loop; 0 on CPU


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """The on-disk training artifact (the format `train.py` owns).

    Everything `sample.py` (TICKET-08) needs to rebuild the *exact* model: the config it was
    built from, the tokenizer's vocab ordering (ids are meaningless without it), the learned
    weights, and where in the run the snapshot was taken.
    """

    config: TrainConfig
    vocab: tuple[str, ...]
    model_state: dict[str, Tensor]
    step: int
    val_loss: float


def set_seed(cfg: TrainConfig) -> None:
    """Seed the global torch RNG (model init + dropout) and every CUDA device.

    The dataset seeds its *own* batch generator from `cfg.seed`, so we deliberately don't
    touch that here — re-seeding it would just re-derive the identical batch stream.
    """
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)  # a safe no-op when no CUDA device is present


def resolve_device(cfg: TrainConfig) -> torch.device:
    """Honor the explicit `cfg.device`, failing loudly on a CUDA request we can't satisfy.

    `config.default_device()` is the autodetect helper for *building* a config; here we treat
    `cfg.device` as a deliberate choice and refuse to silently downgrade a "cuda" run to CPU —
    a 4080 run quietly crawling on CPU for hours is a worse failure than an upfront error.
    """
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"cfg.device={cfg.device!r} but CUDA is unavailable. Pass device='cpu' "
            f"explicitly (e.g. for the CPU smoke test) or fix the CUDA install. "
            f"Autodetected device would be {default_device()!r}."
        )
    return torch.device(cfg.device)


@contextmanager
def _autocast(device: torch.device) -> Iterator[None]:
    """bf16 mixed precision on CUDA, a no-op everywhere else.

    Wraps only the forward+loss, never backward. Gating on device keeps the CPU path in plain
    fp32 (bf16 autocast on CPU is pointless/slow), which also makes the smoke test a clean
    no-op rather than a slow detour.
    """
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


@torch.no_grad()
def estimate_loss(
    model: GPT, dataset: CharDataset, device: torch.device, eval_batches: int
) -> dict[Split, float]:
    """Average loss over a few batches for both splits, with the model in eval mode.

    `eval()` disables dropout so the estimate reflects the weights rather than a sampled mask;
    we restore `train()` before returning. `forward` types loss as `Tensor | None`, but with
    targets supplied it is always a Tensor — assert to narrow the Optional rather than ignore it.
    """
    model.eval()
    out: dict[Split, float] = {}
    splits: tuple[Split, ...] = ("train", "val")
    for split in splits:
        total = 0.0
        for _ in range(eval_batches):
            x, y = dataset.get_batch(split)
            x, y = x.to(device), y.to(device)
            with _autocast(device):
                _, loss = model(x, y)
            assert loss is not None  # targets given -> loss is a Tensor
            total += float(loss)
        out[split] = total / eval_batches
    model.train()
    return out


def save_checkpoint(
    path: Path,
    *,
    model: GPT,
    cfg: TrainConfig,
    tokenizer: CharTokenizer,
    step: int,
    val_loss: float,
) -> None:
    """Write a `Checkpoint`-shaped payload to `path`, creating its directory.

    The payload is typed `dict[str, object]` on purpose: `torch.save` is the one place we hand
    data across an untyped (pickle) boundary, so we keep the values `object` (never `Any`) and
    let `load_checkpoint` re-narrow each field on the way back in.
    """
    payload: dict[str, object] = {
        "format": _CHECKPOINT_FORMAT,
        "config": cfg,
        "vocab": list(tokenizer.itos),
        "model_state": model.state_dict(),
        "step": step,
        "val_loss": val_loss,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _require_int(value: object, field: str) -> int:
    # bool is an int subclass; reject it so a stray True can't masquerade as a step count.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"checkpoint field {field!r} must be an int, got {value!r}.")
    return value


def _require_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"checkpoint field {field!r} must be a number, got {value!r}.")
    return float(value)


def load_checkpoint(path: Path, map_location: str = "cpu") -> Checkpoint:
    """Read a checkpoint written by `save_checkpoint`, validating it into a typed `Checkpoint`.

    `torch.load` is typed as returning `Any`; we capture it as `object` and narrow every field
    with explicit `isinstance` checks so nothing `Any`-typed escapes this function. We pass
    `weights_only=False` because the payload intentionally holds non-tensor objects (the
    `TrainConfig`, the vocab) — so only ever load checkpoints you produced and trust.
    """
    obj: object = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: not a yegpt checkpoint (expected a dict payload).")
    raw: dict[str, object] = {str(key): value for key, value in obj.items()}
    if raw.get("format") != _CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: not a {_CHECKPOINT_FORMAT} checkpoint.")

    config = raw.get("config")
    if not isinstance(config, TrainConfig):
        raise ValueError(f"{path}: 'config' is not a TrainConfig.")

    vocab_value = raw.get("vocab")
    if not isinstance(vocab_value, list):
        raise ValueError(f"{path}: 'vocab' is missing or not a list.")
    vocab: list[str] = []
    for entry in vocab_value:
        if not isinstance(entry, str) or len(entry) != 1:
            raise ValueError(f"{path}: invalid vocab entry {entry!r}.")
        vocab.append(entry)

    state_value = raw.get("model_state")
    if not isinstance(state_value, dict):
        raise ValueError(f"{path}: 'model_state' is missing or not a dict.")
    model_state: dict[str, Tensor] = {}
    for key, value in state_value.items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            raise ValueError(f"{path}: 'model_state' must map str -> Tensor.")
        model_state[key] = value

    return Checkpoint(
        config=config,
        vocab=tuple(vocab),
        model_state=model_state,
        step=_require_int(raw.get("step"), "step"),
        val_loss=_require_float(raw.get("val_loss"), "val_loss"),
    )


def train(
    cfg: TrainConfig,
    *,
    text: str | None = None,
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    eval_batches: int = _DEFAULT_EVAL_BATCHES,
) -> TrainResult:
    """Run the full training loop from an injected config; return where it landed and the curve.

    Wiring (the one place these are built): tokenizer <- corpus, dataset <- (cfg, tokenizer,
    corpus), model <- cfg. `text` overrides the on-disk corpus (tests / any in-memory caller);
    otherwise the corpus is read from `corpus_path`. `cfg.vocab_size` is (re)derived from the
    tokenizer so the model's head always matches the data it is trained on.
    """
    set_seed(cfg)
    device = resolve_device(cfg)

    corpus = text if text is not None else corpus_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(corpus)
    cfg = cfg.with_vocab_size(tokenizer.vocab_size)

    dataset = CharDataset.from_text(cfg, tokenizer, corpus)
    model = GPT(cfg)
    model.to(device)  # in-place for nn.Module; we keep the GPT-typed handle, not .to()'s return
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    print(
        f"yeGPT: {model.num_parameters():,} parameters | device={device} "
        f"| vocab={cfg.vocab_size} | max_iters={cfg.max_iters}"
    )

    history: list[LossPoint] = []
    best_path = checkpoint_dir / _BEST_CHECKPOINT_NAME
    best_val = float("inf")
    best_step = 0

    def evaluate(step: int) -> LossPoint:
        nonlocal best_val, best_step
        losses = estimate_loss(model, dataset, device, eval_batches)
        point = LossPoint(step=step, train_loss=losses["train"], val_loss=losses["val"])
        history.append(point)
        improved = point.val_loss < best_val
        if improved:
            # No optimizer step has run since this eval, so the live weights are exactly the ones
            # just scored — snapshot them as the new best before training moves past this point.
            best_val = point.val_loss
            best_step = step
            save_checkpoint(
                best_path,
                model=model,
                cfg=cfg,
                tokenizer=tokenizer,
                step=step,
                val_loss=point.val_loss,
            )
        marker = "  <- best" if improved else ""
        print(
            f"step {step:>6} | train {point.train_loss:.4f} | val {point.val_loss:.4f}{marker}"
        )
        return point

    # Instrument the loop (TICKET-10.0): reset CUDA's peak tracker to the current (model +
    # optimizer) baseline so the recorded peak is the loop's own activation/grad high-water mark,
    # and start the wall-clock. The loss path below is untouched — timing only brackets the loop,
    # so 10.1 stays a clean comparison.
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    loop_start = time.perf_counter()
    for step in range(cfg.max_iters):
        if step % cfg.eval_interval == 0:
            evaluate(step)  # step 0 records the untrained baseline
        x, y = dataset.get_batch("train")
        x, y = x.to(device), y.to(device)
        with _autocast(device):
            _, loss = model(x, y)
        assert loss is not None  # targets given -> loss is a Tensor
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    # CUDA kernels are queued asynchronously; sync before stopping the clock or we'd time only the
    # Python-side dispatch, not the GPU work. A no-op on CPU.
    if is_cuda:
        torch.cuda.synchronize(device)
    loop_elapsed = time.perf_counter() - loop_start
    peak_vram_bytes = int(torch.cuda.max_memory_allocated(device)) if is_cuda else 0
    # Guard the degenerate (sub-resolution / zero-iter) case so the metrics stay finite.
    steps_per_sec = cfg.max_iters / loop_elapsed if loop_elapsed > 0.0 else 0.0
    tokens_per_sec = steps_per_sec * cfg.batch_size * cfg.block_size

    # Final eval after the last step so the curve ends on the fully-trained weights.
    final_point = evaluate(cfg.max_iters)

    checkpoint_path = checkpoint_dir / _CHECKPOINT_NAME
    save_checkpoint(
        checkpoint_path,
        model=model,
        cfg=cfg,
        tokenizer=tokenizer,
        step=cfg.max_iters,
        val_loss=final_point.val_loss,
    )
    print(f"saved final checkpoint -> {checkpoint_path} (val {final_point.val_loss:.4f})")
    print(f"best val {best_val:.4f} @ step {best_step} -> {best_path}")
    print(
        f"throughput: {steps_per_sec:,.1f} steps/s | {tokens_per_sec:,.0f} tokens/s "
        f"| peak VRAM {peak_vram_bytes / 1024**3:.2f} GiB ({peak_vram_bytes:,} B) "
        f"| {loop_elapsed:.2f}s over {cfg.max_iters:,} steps"
    )

    return TrainResult(
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_path,
        start_loss=history[0].val_loss,
        final_loss=final_point.val_loss,
        best_loss=best_val,
        best_step=best_step,
        history=tuple(history),
        steps_per_sec=steps_per_sec,
        tokens_per_sec=tokens_per_sec,
        peak_vram_bytes=peak_vram_bytes,
    )


@dataclass(frozen=True, slots=True)
class TrainInvocation:
    """A train() call resolved from CLI args: the config plus the corpus/checkpoint path overrides.

    Splitting arg-parsing (this + `parse_train_args`) from the heavy `train()` call keeps the
    argv -> config mapping unit-testable without spinning up a training run.
    """

    config: TrainConfig
    corpus_path: Path
    checkpoint_dir: Path


def _build_arg_parser() -> argparse.ArgumentParser:
    """CLI mirroring the tunable TrainConfig fields; every default is the dataclass default itself.

    Sourcing defaults from a fresh `TrainConfig()` means the CLI can never drift from the
    dataclass and keeps this a pure *override* layer — we sweep configs via flags, never by
    editing `TrainConfig` (TICKET-09.1). `vocab_size` is intentionally not a flag: `train`
    derives it from the corpus.
    """
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Train yeGPT on a corpus and write a checkpoint (TICKET-07 CLI surface)."
    )
    parser.add_argument(
        "--n-layer", type=int, default=defaults.n_layer,
        help="Number of transformer blocks.",
    )
    parser.add_argument(
        "--n-head", type=int, default=defaults.n_head,
        help="Attention heads per block (must divide --n-embd).",
    )
    parser.add_argument(
        "--n-embd", type=int, default=defaults.n_embd,
        help="Embedding / residual-stream width.",
    )
    parser.add_argument(
        "--block-size", type=int, default=defaults.block_size,
        help="Context length in characters.",
    )
    parser.add_argument(
        "--dropout", type=float, default=defaults.dropout,
        help="Dropout probability in attention and the MLP.",
    )
    parser.add_argument(
        "--lr", type=float, default=defaults.lr,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--max-iters", type=int, default=defaults.max_iters,
        help="Total optimizer steps.",
    )
    parser.add_argument(
        "--eval-interval", type=int, default=defaults.eval_interval,
        help="Steps between train/val loss estimates.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=defaults.batch_size,
        help="Sequences per optimizer step.",
    )
    parser.add_argument(
        "--device", type=str, default=defaults.device,
        help="Torch device, e.g. 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--seed", type=int, default=defaults.seed,
        help="RNG seed for reproducible init, batches, and sampling.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR,
        help="Checkpoint directory (use distinct dirs to keep multiple runs, e.g. before/after).",
    )
    parser.add_argument(
        "--corpus", type=Path, default=DEFAULT_CORPUS_PATH,
        help="Corpus text file to train on.",
    )
    return parser


def parse_train_args(argv: Sequence[str] | None = None) -> TrainInvocation:
    """Parse argv into a TrainInvocation, overriding only the TrainConfig fields that were passed.

    Each argparse attribute is read into a typed local before use, so no `Any` leaks out of the
    untyped Namespace (the pattern from `sample.py:main`).
    """
    args = _build_arg_parser().parse_args(argv)

    n_layer: int = args.n_layer
    n_head: int = args.n_head
    n_embd: int = args.n_embd
    block_size: int = args.block_size
    dropout: float = args.dropout
    lr: float = args.lr
    max_iters: int = args.max_iters
    eval_interval: int = args.eval_interval
    batch_size: int = args.batch_size
    device: str = args.device
    seed: int = args.seed
    out_dir: Path = args.out_dir
    corpus_path: Path = args.corpus

    config = TrainConfig(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        dropout=dropout,
        batch_size=batch_size,
        lr=lr,
        max_iters=max_iters,
        eval_interval=eval_interval,
        device=device,
        seed=seed,
    )
    return TrainInvocation(config=config, corpus_path=corpus_path, checkpoint_dir=out_dir)


def main() -> None:  # pragma: no cover - thin entrypoint; parse_train_args holds the testable logic
    invocation = parse_train_args()
    train(
        invocation.config,
        corpus_path=invocation.corpus_path,
        checkpoint_dir=invocation.checkpoint_dir,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
