"""yegpt: one console entry point - the streaming demo plus per-module subcommands.

`[project.scripts]` wires `yegpt = "yegpt.cli:main"`, and the first argument picks the mode:

* `yegpt <command> [args...]` (first arg IS a known subcommand) runs the matching module's own
  `main()`. Each subcommand owns its flags and its parser; this dispatcher only routes to it.
  That means `yegpt sample --help` shows sample.py's parser (not this one), and adding a flag to
  a subcommand never touches this file.
* `yegpt "some prompt" [knobs...]` (first arg is NOT a known subcommand) is the demo: it loads a
  checkpoint, seeds generation with the prompt, and typewriter-streams the output - each
  character printed the moment its token is sampled via `GPT.generate_stream`, not after the
  whole generation finishes (which is why this does not route through `sample.main`). CPU only;
  knob defaults are the recommended sampling settings from the model card.

The delegated mains call `argparse`'s `parse_args()` with no argument, which reads `sys.argv`.
So we hand each one a clean `sys.argv` whose prog is `yegpt <command>` and whose remaining args
are only its own, then restore the original afterward.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Final

import torch

from yegpt import __version__, data_prep, dedup, export, export_samples, sample, train, tweet_prep
from yegpt.model import GPT
from yegpt.tokenizer import CharTokenizer
from yegpt.train import DEFAULT_CHECKPOINT_DIR, Checkpoint, load_checkpoint

# Ordered to read roughly as the pipeline runs: build corpus -> train -> sample -> export.
_COMMANDS: dict[str, Callable[[], None]] = {
    "data-prep": data_prep.main,
    "tweet-prep": tweet_prep.main,
    "dedup": dedup.main,
    "train": train.main,
    "sample": sample.main,
    "export": export.main,
    "export-samples": export_samples.main,
}

# The demo defaults: the released run3 checkpoint read out with the model-card knobs
# (temperature 0.9, top-p 0.92, repetition penalty 1.3) for ~200 characters.
_DEMO_CHECKPOINT_PATH: Final[Path] = DEFAULT_CHECKPOINT_DIR / "run3" / "yegpt-ckpt.pt"
_DEMO_TEMPERATURE: Final[float] = 0.9
_DEMO_TOP_P: Final[float] = 0.92
_DEMO_REPETITION_PENALTY: Final[float] = 1.3
_DEMO_MAX_CHARS: Final[int] = 200

_CPU: Final[torch.device] = torch.device("cpu")


def _usage() -> str:
    commands = "\n".join(f"  {name}" for name in _COMMANDS)
    return (
        f"usage: yegpt <command> [args...]\n"
        f'       yegpt "prompt" [--checkpoint PATH] [--max-chars N] [--seed N] [knobs...]\n\n'
        f"yeGPT {__version__} - a character-level GPT trained from scratch.\n\n"
        f"commands:\n{commands}\n\n"
        f"Anything that is not a command is treated as a prompt and streamed live.\n"
        f"Run 'yegpt <command> --help' for command-specific options."
    )


def _build_prompt_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yegpt",
        description="Stream a generation from a checkpoint, typing each character as sampled.",
    )
    parser.add_argument("prompt", type=str, help="Seed text to condition on.")
    parser.add_argument(
        "--checkpoint", type=Path, default=_DEMO_CHECKPOINT_PATH,
        help="Path to a checkpoint written by train.py (default: the released run3 weights).",
    )
    parser.add_argument(
        "--max-chars", type=int, default=_DEMO_MAX_CHARS,
        help="Number of characters to generate (default: %(default)s).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for reproducible sampling (default: unseeded).",
    )
    parser.add_argument(
        "--temperature", type=float, default=_DEMO_TEMPERATURE,
        help="Softmax temperature (>0): <1 sharpens, >1 flattens (default: %(default)s).",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Sample only from the K most likely characters each step (default: full vocab).",
    )
    parser.add_argument(
        "--top-p", type=float, default=_DEMO_TOP_P,
        help="Nucleus sampling: keep the top chars summing to P probability "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=_DEMO_REPETITION_PENALTY,
        help="Down-weight already-seen chars to break loops (>0, 1.0=off; "
             "default: %(default)s).",
    )
    return parser


def _stream_chars(
    ckpt: Checkpoint,
    *,
    prompt: str,
    max_new_tokens: int,
    seed: int | None,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float,
) -> Iterator[str]:
    """Reconstruct the model from `ckpt` and yield each freshly sampled character.

    Same reconstruction as `sample.sample_from_checkpoint` (train.py owns the format; we only
    rebuild from it) but delivered through `GPT.generate_stream`, so the caller can print each
    character as its token is drawn. CPU always: the demo must never touch the GPU, and moving
    the model to CPU explicitly avoids honoring a checkpoint whose `config.device` reads "cuda".
    An empty prompt is primed with a synthetic start token that never reaches the output.
    """
    tokenizer = CharTokenizer(ckpt.vocab)
    model = GPT(ckpt.config)
    model.load_state_dict(ckpt.model_state)
    model.eval()  # disable dropout so sampling reflects the weights, not a sampled mask
    model.to(_CPU)

    generator: torch.Generator | None = None
    if seed is not None:
        generator = torch.Generator(device=_CPU)
        generator.manual_seed(seed)

    start_ids = tokenizer.encode(prompt) if prompt else [tokenizer.stoi.get("\n", 0)]
    idx = torch.tensor([start_ids], dtype=torch.long, device=_CPU)  # (1, len(start_ids))

    for token in model.generate_stream(
        idx,
        max_new_tokens,
        generator,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    ):
        yield tokenizer.decode([int(token[0, 0])])


def _run_prompt(args: list[str]) -> None:
    """Parse prompt-mode arguments and typewriter-stream the generation to stdout.

    Errors a user can cause - missing/corrupt checkpoint, out-of-vocab prompt characters,
    invalid sampling knobs - surface as `SystemExit` with a message (non-zero exit) instead of a
    traceback. Knob validation lives in `model.generate_stream` and runs when iteration begins,
    so the whole streaming loop sits inside the try.
    """
    parser = _build_prompt_parser()
    ns = parser.parse_args(args)

    # argparse Namespace attrs are Any; read each into a typed local so no Any leaks downstream.
    prompt: str = ns.prompt
    checkpoint_path: Path = ns.checkpoint
    max_chars: int = ns.max_chars
    seed: int | None = ns.seed
    temperature: float = ns.temperature
    top_k: int | None = ns.top_k
    top_p: float | None = ns.top_p
    repetition_penalty: float = ns.repetition_penalty

    if max_chars < 0:
        parser.error(f"--max-chars must be >= 0, got {max_chars}.")

    try:
        ckpt = load_checkpoint(checkpoint_path)
    except (OSError, ValueError) as err:
        raise SystemExit(f"yegpt: {err}") from err

    print(prompt, end="", flush=True)
    try:
        for ch in _stream_chars(
            ckpt,
            prompt=prompt,
            max_new_tokens=max_chars,
            seed=seed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ):
            print(ch, end="", flush=True)
    except ValueError as err:
        raise SystemExit(f"yegpt: {err}") from err
    print()


def main(argv: list[str] | None = None) -> None:
    """Route `yegpt <command> ...` to the matching module `main`, else run the streaming demo.

    `argv` defaults to the process arguments (`sys.argv[1:]`); pass a list to drive it in tests.
    Bare invocation or `-h/--help` prints the command list; `-V/--version` prints the version.
    A first argument that is not a known command is treated as a prompt (see `_run_prompt`).
    """
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in ("-h", "--help"):
        print(_usage())
        return
    if args[0] in ("-V", "--version"):
        print(f"yeGPT {__version__}")
        return

    command = args[0]
    handler = _COMMANDS.get(command)
    if handler is None:
        _run_prompt(args)
        return

    saved_argv = sys.argv
    sys.argv = [f"yegpt {command}", *args[1:]]
    try:
        handler()
    finally:
        sys.argv = saved_argv


if __name__ == "__main__":  # pragma: no cover
    main()
