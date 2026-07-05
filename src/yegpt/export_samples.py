"""export_samples: dump a batch of short generated fragments from a checkpoint (Unit 2).

A companion to `sample.py` for the release: where `sample` streams one long generation to watch
the style, this writes a *set* of short, independent fragments to a file -- the kind of showcase
you paste into a model card or README. It is pure orchestration in the same spirit as `export.py`
and reuses the checkpoint contract rather than forking it:

    train.load_checkpoint(path) -> Checkpoint
    generate_samples(...)       -> [sample.sample_from_checkpoint(...) for each fragment]
    write the numbered fragments to disk

Each fragment is produced by delegating to `sample.sample_from_checkpoint`, so the model
reconstruction, prompt-prefix invariant, and sampling knobs all stay owned by `sample.py`; this
module only loops, seeds per fragment for reproducible-yet-varied output, and formats the result.

Honest scope (SPEC.md §0): a checkpoint from a tiny / under-trained run samples near-noise. These
fragments read as noise -> word-shaped -> recognizably-Kanye-styled gibberish as training
improves; this is the harness for collecting them, not a path to coherent lyrics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch

from yegpt.sample import sample_from_checkpoint
from yegpt.train import DEFAULT_CHECKPOINT_PATH, Checkpoint, load_checkpoint

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
# A build output, like the fp16 export: `dist/` is gitignored, so this file is never committed.
DEFAULT_SAMPLES_PATH: Final[Path] = _REPO_ROOT / "dist" / "yegpt-samples.txt"

# Twelve fragments is enough to show range without becoming a wall of text; each is kept short so
# the batch reads as a set of snippets, not one long dump (that is what `sample.py` is for).
_DEFAULT_NUM_SAMPLES: Final[int] = 12
_DEFAULT_FRAGMENT_TOKENS: Final[int] = 120


@dataclass(frozen=True, slots=True)
class SampleExportResult:
    """Where the fragments were written and how many/how large the file is."""

    dest_path: Path
    num_samples: int
    dest_bytes: int


def generate_samples(
    ckpt: Checkpoint,
    *,
    num_samples: int = _DEFAULT_NUM_SAMPLES,
    max_new_tokens: int = _DEFAULT_FRAGMENT_TOKENS,
    device: torch.device,
    prompt: str = "",
    seed: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
) -> list[str]:
    """Generate `num_samples` independent fragments from an in-memory checkpoint.

    Pure (no disk I/O): the checkpoint is already loaded. Each fragment is one call to
    `sample.sample_from_checkpoint`, so every fragment obeys that function's contract -- it starts
    with `prompt` and has length `len(prompt) + max_new_tokens`. When `seed` is given, fragment
    `i` is drawn with a generator seeded `seed + i`, so the whole batch reproduces exactly yet the
    fragments differ from one another; without a seed, every fragment is drawn unseeded.
    `temperature`, `top_k`, `top_p`, and `repetition_penalty` are threaded straight through to the
    sampler (see `model.generate`).
    """
    if num_samples < 0:
        raise ValueError(f"num_samples must be >= 0, got {num_samples}.")

    samples: list[str] = []
    for offset in range(num_samples):
        generator: torch.Generator | None = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed + offset)  # distinct per fragment, deterministic per batch
        samples.append(
            sample_from_checkpoint(
                ckpt,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                device=device,
                generator=generator,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        )
    return samples


def format_samples(samples: list[str]) -> str:
    """Render fragments as numbered blocks separated by blank lines, with a trailing newline."""
    blocks = [f"--- sample {index} ---\n{text}" for index, text in enumerate(samples, start=1)]
    return "\n\n".join(blocks) + "\n"


def export_samples(
    checkpoint_path: Path,
    dest_path: Path,
    *,
    num_samples: int = _DEFAULT_NUM_SAMPLES,
    max_new_tokens: int = _DEFAULT_FRAGMENT_TOKENS,
    prompt: str = "",
    device: torch.device | None = None,
    seed: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
) -> SampleExportResult:
    """Load a checkpoint off disk, generate the fragments, and write them to `dest_path`.

    Device defaults to CPU for the same reasons as `sample.generate_text`: sampling a handful of
    short fragments is cheap and serial, it keeps callers CUDA-free, and it avoids honoring a
    checkpoint whose `config.device` reads "cuda" on a GPU-less box. Missing parent directories of
    `dest_path` are created. Returns the destination path, the fragment count, and the file size.
    """
    ckpt = load_checkpoint(checkpoint_path)  # loads to CPU; the sampler moves the model as needed
    target_device = device if device is not None else torch.device("cpu")

    samples = generate_samples(
        ckpt,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        device=target_device,
        prompt=prompt,
        seed=seed,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(format_samples(samples), encoding="utf-8")

    return SampleExportResult(
        dest_path=dest_path,
        num_samples=len(samples),
        dest_bytes=dest_path.stat().st_size,
    )


def main() -> None:  # pragma: no cover - thin CLI wrapper; the core above is what the tests drive
    parser = argparse.ArgumentParser(
        description="Export a batch of short sample fragments from a yeGPT checkpoint."
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH,
        help="Source checkpoint written by train.py.",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_SAMPLES_PATH,
        help="Destination text file for the fragments.",
    )
    parser.add_argument(
        "-n", "--num-samples", type=int, default=_DEFAULT_NUM_SAMPLES,
        help="Number of fragments to generate.",
    )
    parser.add_argument(
        "--num-tokens", type=int, default=_DEFAULT_FRAGMENT_TOKENS,
        help="Characters to generate per fragment.",
    )
    parser.add_argument(
        "--prompt", type=str, default="",
        help="Seed text each fragment is conditioned on (default: empty -> primed start).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Base RNG seed; fragment i uses seed+i for reproducible-yet-varied output.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device, e.g. 'cpu' or 'cuda' (default: cpu).",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Softmax temperature (>0): <1 sharpens, >1 flattens (default: 1.0).",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Sample only from the K most likely characters each step (default: full vocab).",
    )
    parser.add_argument(
        "--top-p", type=float, default=None,
        help="Nucleus sampling: keep the top chars summing to P probability (default: off).",
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.0,
        help="Down-weight already-seen chars to break loops (>0, 1.0=off; try ~1.2).",
    )
    args = parser.parse_args()

    # argparse Namespace attrs are Any; read each into a typed local so no Any leaks downstream.
    checkpoint_path: Path = args.checkpoint
    out_path: Path = args.out
    num_samples: int = args.num_samples
    num_tokens: int = args.num_tokens
    prompt: str = args.prompt
    seed: int | None = args.seed
    device_str: str | None = args.device
    temperature: float = args.temperature
    top_k: int | None = args.top_k
    top_p: float | None = args.top_p
    repetition_penalty: float = args.repetition_penalty
    device = torch.device(device_str) if device_str is not None else None

    result = export_samples(
        checkpoint_path,
        out_path,
        num_samples=num_samples,
        max_new_tokens=num_tokens,
        prompt=prompt,
        device=device,
        seed=seed,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    print(
        f"wrote {result.num_samples} fragment(s) to {result.dest_path} "
        f"({result.dest_bytes:,} B)"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
