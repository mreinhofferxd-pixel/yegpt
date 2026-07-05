"""export: turn a trained checkpoint into a smaller fp16 distributable for the release.

This is a pure orchestration module in the spirit of `sample.py`: it reuses the checkpoint
contract `train.py` owns rather than forking it. The flow is load -> reconstruct -> cast ->
re-save through the *same* `save_checkpoint`, so the exported file is byte-compatible with
`load_checkpoint` and `sample.sample_from_checkpoint`:

    train.load_checkpoint(source)  -> Checkpoint (fp32 weights, loaded on CPU)
    GPT(config) <- config; load_state_dict(model_state); model.half()  (float params -> fp16)
    train.save_checkpoint(dest, model=<fp16 model>, ...)               (same v1 format)

fp16 halves the storage of every float parameter, so the distributable is ~half the size.
It is a lossy weight cast, not a format change: on load, `sample_from_checkpoint` builds an
fp32 GPT and `load_state_dict` upcasts the fp16 tensors straight back into fp32 params.

CPU only: no forward pass runs here, just a dtype cast on the parameters, so nothing touches
CUDA even when the source config's device reads "cuda".
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from yegpt.model import GPT
from yegpt.tokenizer import CharTokenizer
from yegpt.train import load_checkpoint, save_checkpoint

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
# The released model is the settled run3, not train.py's generic output path (which only
# exists right after a fresh training run).
DEFAULT_SOURCE_PATH: Final[Path] = _REPO_ROOT / "checkpoints" / "run3" / "yegpt-ckpt.pt"
# The distributable artifact the GitHub release ships. `dist/` is gitignored, so this is a
# build output, never committed.
DEFAULT_EXPORT_PATH: Final[Path] = _REPO_ROOT / "dist" / "yegpt-small-fp16.pt"


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Where the export wrote and how much the fp16 cast saved on disk."""

    source_path: Path
    dest_path: Path
    source_bytes: int
    dest_bytes: int


def export_fp16(source_path: Path, dest_path: Path) -> ExportResult:
    """Load `source_path`, cast the model weights to fp16, and re-save to `dest_path`.

    The output goes through `train.save_checkpoint` unchanged, so it stays in the exact v1
    format `load_checkpoint`/`sample_from_checkpoint` read back. We reconstruct the model from
    the checkpoint, `load_state_dict` the fp32 weights into it, then `model.half()` casts every
    float parameter to fp16 before the state dict is written. Returns the source/dest paths and
    their on-disk byte sizes (the dest is smaller because fp16 halves each float's storage).
    """
    ckpt = load_checkpoint(source_path)  # loads to CPU; the cast below never needs a device

    tokenizer = CharTokenizer(ckpt.vocab)
    model = GPT(ckpt.config)
    model.load_state_dict(ckpt.model_state)
    model.half()  # in-place cast of the float parameters/buffers to fp16

    save_checkpoint(
        dest_path,
        model=model,
        cfg=ckpt.config,
        tokenizer=tokenizer,
        step=ckpt.step,
        val_loss=ckpt.val_loss,
    )

    return ExportResult(
        source_path=source_path,
        dest_path=dest_path,
        source_bytes=source_path.stat().st_size,
        dest_bytes=dest_path.stat().st_size,
    )


def main() -> None:  # pragma: no cover - thin CLI wrapper; export_fp16 holds the tested logic
    parser = argparse.ArgumentParser(
        description="Export a yeGPT checkpoint to a smaller fp16 distributable."
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_SOURCE_PATH,
        help="Source checkpoint written by train.py.",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_EXPORT_PATH,
        help="Destination path for the fp16 checkpoint.",
    )
    args = parser.parse_args()

    # argparse Namespace attrs are Any; read each into a typed local so no Any leaks downstream.
    checkpoint_path: Path = args.checkpoint
    out_path: Path = args.out

    result = export_fp16(checkpoint_path, out_path)
    saved = result.source_bytes - result.dest_bytes
    print(
        f"exported {result.source_path} ({result.source_bytes:,} B) -> "
        f"{result.dest_path} ({result.dest_bytes:,} B) "
        f"| saved {saved:,} B ({saved / result.source_bytes:.1%})"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
