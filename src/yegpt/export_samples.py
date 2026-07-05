"""export_samples: dump a batch of short generated fragments from a checkpoint (Unit 4).

Companion to `sample.py` for the public launch: where `sample` streams one long generation to
watch the style, this writes a *set* of short, independent fragments to `web/samples.json` -- the
showcase the static web embed replays. It is pure orchestration over the checkpoint contract and
reuses the sampler rather than forking it:

    train.load_checkpoint(path) -> Checkpoint
    generate_samples(...)       -> [sample.sample_from_checkpoint(...) per fragment]
    filter_profanity(...)       -> drop fragments containing built-in wordlist terms
    build_document(...)         -> {"generated_with": {model, seed, knobs}, "samples": [...]}

Model reconstruction, the prompt-prefix invariant, and sampling knobs all stay owned by
`sample.py`; this module only loops, seeds per fragment for reproducible-yet-varied output,
optionally screens profanity, and serialises the batch as JSON.

These fragments are model OUTPUT (parody generation), so `web/samples.json` is safe to commit even
though the raw corpus never is. Honest scope (SPEC.md 0): a small model
samples near-noise, so the fragments read as gibberish; this is the harness for collecting and
shipping them, not a path to coherent lyrics.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch

from yegpt.sample import sample_from_checkpoint
from yegpt.train import DEFAULT_CHECKPOINT_PATH, Checkpoint, load_checkpoint

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
# Committed showcase output (parody fragments), not raw lyrics -- safe to ship.
DEFAULT_SAMPLES_PATH: Final[Path] = _REPO_ROOT / "web" / "samples.json"

# Twelve fragments show range without becoming a wall of text; ~200 chars each keeps each one a
# snippet, not one long dump (that is what `sample.py` is for).
_DEFAULT_NUM_SAMPLES: Final[int] = 12
_DEFAULT_FRAGMENT_TOKENS: Final[int] = 200

# Recommended sampling knobs (MODEL_CARD.md): sharpen a touch, nucleus-clip the tail, and penalise
# repeats so the fragments do not collapse into a single looped character.
_DEFAULT_TEMPERATURE: Final[float] = 0.9
_DEFAULT_TOP_P: Final[float] = 0.92
_DEFAULT_REPETITION_PENALTY: Final[float] = 1.3
# Fixed, documented base seed so the committed artifact reproduces byte-for-byte.
_DEFAULT_SEED: Final[int] = 1234

# Small built-in screen: a fragment containing any of these as a whole word is dropped when the
# profanity filter is on (case-insensitive, word-boundary). A profanity filter has to name the
# words it blocks; this is the entire list.
_PROFANITY_WORDLIST: Final[frozenset[str]] = frozenset(
    {
        "fuck",
        "shit",
        "bitch",
        "cunt",
        "pussy",
        "dick",
        "cock",
        "faggot",
        "nigga",
        "nigger",
    }
)


@dataclass(frozen=True, slots=True)
class SampleExportResult:
    """Where the fragments were written and how many survived the filter / how large the file is."""

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


# Common inflectional endings appended to a base term (fuck -> fucking/fucked/fucks). Matching
# base + optional-suffix + word boundary catches these while sparing clean words that merely start
# with a base term (dickens, cocktail), which a bare prefix match would wrongly flag.
_INFLECTION: Final[str] = r"(?:s|es|ed|ing|in|er|ers|y)?"


def contains_profanity(text: str, wordlist: frozenset[str] = _PROFANITY_WORDLIST) -> bool:
    """True if `text` contains any wordlist term (or a common inflection of it) as a whole word.

    Case-insensitive and word-boundary anchored: `shit` matches `shit`/`shits`/`shitting` but not
    `shirt`, and `dick` does not trip on `dickens`.
    """
    lowered = text.lower()
    return any(
        re.search(rf"\b{re.escape(word)}{_INFLECTION}\b", lowered) is not None for word in wordlist
    )


def filter_profanity(
    samples: list[str], wordlist: frozenset[str] = _PROFANITY_WORDLIST
) -> list[str]:
    """Drop fragments containing a wordlist term; the survivors keep their original order."""
    return [text for text in samples if not contains_profanity(text, wordlist)]


def build_document(
    samples: list[str],
    *,
    model: str,
    seed: int | None,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float,
    max_new_tokens: int,
    prompt: str,
    profanity_filter: bool,
) -> dict[str, object]:
    """Assemble the `{generated_with, samples}` document written to `web/samples.json`.

    `generated_with` records exactly what a reader needs to regenerate the batch: the source model,
    the base seed, and the sampling knobs (including the post-filter fragment count).
    """
    return {
        "generated_with": {
            "model": model,
            "seed": seed,
            "knobs": {
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "max_new_tokens": max_new_tokens,
                "num_samples": len(samples),
                "prompt": prompt,
                "profanity_filter": profanity_filter,
            },
        },
        "samples": samples,
    }


def _render_json(document: dict[str, object]) -> str:
    """Serialise the document as pretty JSON with a trailing newline (stable across runs)."""
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def export_samples(
    checkpoint_path: Path,
    dest_path: Path,
    *,
    num_samples: int = _DEFAULT_NUM_SAMPLES,
    max_new_tokens: int = _DEFAULT_FRAGMENT_TOKENS,
    prompt: str = "",
    device: torch.device | None = None,
    seed: int | None = _DEFAULT_SEED,
    temperature: float = _DEFAULT_TEMPERATURE,
    top_k: int | None = None,
    top_p: float | None = _DEFAULT_TOP_P,
    repetition_penalty: float = _DEFAULT_REPETITION_PENALTY,
    profanity_filter: bool = True,
    model_name: str | None = None,
) -> SampleExportResult:
    """Load a checkpoint off disk, generate the fragments, optionally filter, and write JSON.

    Device defaults to CPU for the same reasons as `sample.generate_text`: sampling a handful of
    short fragments is cheap and serial, it keeps callers CUDA-free, and it avoids honoring a
    checkpoint whose `config.device` reads "cuda" on a GPU-less box. When `profanity_filter` is on
    (the default), fragments hitting the built-in wordlist are dropped before serialisation, so the
    committed count can be below `num_samples`. `model_name` labels the artifact; it defaults to the
    checkpoint path as a POSIX string so no absolute Windows path leaks in. Missing parents of
    `dest_path` are created. Returns the destination, the surviving fragment count, and the size.
    """
    ckpt = load_checkpoint(checkpoint_path)  # loads to CPU; the sampler moves the model as needed
    target_device = device if device is not None else torch.device("cpu")

    generated = generate_samples(
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
    kept = filter_profanity(generated) if profanity_filter else generated

    document = build_document(
        kept,
        model=model_name if model_name is not None else checkpoint_path.as_posix(),
        seed=seed,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        prompt=prompt,
        profanity_filter=profanity_filter,
    )

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(_render_json(document), encoding="utf-8")

    return SampleExportResult(
        dest_path=dest_path,
        num_samples=len(kept),
        dest_bytes=dest_path.stat().st_size,
    )


def main() -> None:  # pragma: no cover - thin CLI wrapper; the core above is what the tests drive
    parser = argparse.ArgumentParser(
        description="Export a batch of short parody fragments from a yeGPT checkpoint to JSON."
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH,
        help="Source checkpoint written by train.py.",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_SAMPLES_PATH,
        help="Destination JSON file for the fragments (default: web/samples.json).",
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
        "--seed", type=int, default=_DEFAULT_SEED,
        help="Base RNG seed; fragment i uses seed+i for reproducible-yet-varied output.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device, e.g. 'cpu' or 'cuda' (default: cpu).",
    )
    parser.add_argument(
        "--temperature", type=float, default=_DEFAULT_TEMPERATURE,
        help="Softmax temperature (>0): <1 sharpens, >1 flattens (default: 0.9).",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Sample only from the K most likely characters each step (default: full vocab).",
    )
    parser.add_argument(
        "--top-p", type=float, default=_DEFAULT_TOP_P,
        help="Nucleus sampling: keep the top chars summing to P probability (default: 0.92).",
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=_DEFAULT_REPETITION_PENALTY,
        help="Down-weight already-seen chars to break loops (>0, 1.0=off; default: 1.3).",
    )
    parser.add_argument(
        "--profanity-filter", action=argparse.BooleanOptionalAction, default=True,
        help="Drop fragments containing built-in profanity (default: on).",
    )
    args = parser.parse_args()

    # argparse Namespace attrs are Any; read each into a typed local so no Any leaks downstream.
    checkpoint_path: Path = args.checkpoint
    out_path: Path = args.out
    num_samples: int = args.num_samples
    num_tokens: int = args.num_tokens
    prompt: str = args.prompt
    seed: int = args.seed
    device_str: str | None = args.device
    temperature: float = args.temperature
    top_k: int | None = args.top_k
    top_p: float | None = args.top_p
    repetition_penalty: float = args.repetition_penalty
    profanity_filter: bool = args.profanity_filter
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
        profanity_filter=profanity_filter,
    )
    print(
        f"wrote {result.num_samples} fragment(s) to {result.dest_path} "
        f"({result.dest_bytes:,} B)"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
