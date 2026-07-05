"""sample: reconstruct a trained GPT from a checkpoint and watch the style emerge.

SPEC.md §5, TICKET-08. This is the *read* side of the checkpoint contract `train.py` owns:
`train.py` writes and validates the on-disk format; `sample.py` only **reconstructs** from it.
We never re-read or re-validate the layout here — `load_checkpoint` already did that — we just
rebuild the exact pieces a forward pass needs:

    Checkpoint.vocab       -> CharTokenizer(vocab)      (the vocab is embedded; no token file)
    Checkpoint.config      -> GPT(config)
    Checkpoint.model_state -> model.load_state_dict(...)
    -> model.eval() -> model.to(device)

Two layers on purpose: a pure `sample_from_checkpoint` core that does **no disk I/O** (the test
drives this), and a thin `generate_text` that loads a checkpoint off disk and delegates.

Sampling knobs (`--temperature`, `--top-k`) live where they belong: in `model.generate`
(model.py / TICKET-06), which reshapes the logits before the multinomial draw. This module only
*surfaces* them as CLI flags and threads them through — it never reshapes logits itself, keeping
the checkpoint-reconstruction contract above the sampling policy.

Honest scope (SPEC.md §0): a checkpoint from a tiny / under-trained run samples near-noise.
That is expected — this script is the harness for watching output go noise -> word-shaped ->
recognizably-Kanye-styled gibberish as training improves, not a path to coherent lyrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import torch

from yegpt.model import GPT
from yegpt.tokenizer import CharTokenizer
from yegpt.train import DEFAULT_CHECKPOINT_PATH, Checkpoint, load_checkpoint

# A few hundred characters is plenty to read the style and stays cheap even on CPU.
_DEFAULT_NUM_TOKENS: Final[int] = 500


def _starting_token(tokenizer: CharTokenizer) -> int:
    """Pick a priming token to begin generation when the prompt is empty.

    The model cannot condition on an empty context, so it needs at least one token to start
    from. A newline is the natural "line start" in this corpus, so use it when the vocab has
    one; otherwise fall back to token 0. This primer is dropped from the returned text — it is
    not part of the prompt the caller asked for.
    """
    return tokenizer.stoi.get("\n", 0)


def sample_from_checkpoint(
    ckpt: Checkpoint,
    *,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    generator: torch.Generator | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
) -> str:
    """Reconstruct the model from `ckpt` and return `prompt` followed by sampled characters.

    Pure (no disk I/O): the checkpoint is already in memory. The result is exactly
    `prompt + <max_new_tokens generated chars>`, so it always has length
    `len(prompt) + max_new_tokens` and always starts with `prompt`.

    An out-of-vocab character in `prompt` surfaces as `CharTokenizer.encode`'s `ValueError`
    rather than crashing opaquely later. `generator`, when given, must live on the same `device`
    (it seeds the multinomial draw); pass a seeded one so a `(checkpoint, prompt, seed)` triple
    reproduces exactly. `temperature`, `top_k`, `top_p`, and `repetition_penalty` are passed
    straight to `model.generate` (which validates and applies them); see its docstring for their
    effect on the sampling distribution.
    """
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}.")

    tokenizer = CharTokenizer(ckpt.vocab)
    model = GPT(ckpt.config)
    model.load_state_dict(ckpt.model_state)
    model.eval()  # disable dropout so sampling reflects the weights, not a sampled mask
    model.to(device)  # in-place for nn.Module; we keep the GPT-typed handle, not .to()'s return

    # A non-empty prompt seeds generation directly (and `encode` validates its chars on the way
    # in). An empty prompt has nothing to condition on, so prime with a synthetic start token
    # that we then drop from the output.
    start_ids = tokenizer.encode(prompt) if prompt else [_starting_token(tokenizer)]
    idx = torch.tensor([start_ids], dtype=torch.long, device=device)  # (1, len(start_ids))

    out = model.generate(  # (1, len(start_ids) + max_new_tokens)
        idx,
        max_new_tokens,
        generator,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    # Decode only the freshly sampled tail and prepend the literal prompt, so the prefix
    # invariant holds structurally and the synthetic primer (if any) never reaches the output.
    generated = out[0]
    new_ids: list[int] = [
        int(generated[len(start_ids) + offset]) for offset in range(max_new_tokens)
    ]
    return prompt + tokenizer.decode(new_ids)


def generate_text(
    checkpoint_path: Path,
    *,
    prompt: str = "",
    max_new_tokens: int = _DEFAULT_NUM_TOKENS,
    device: torch.device | None = None,
    seed: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
) -> str:
    """Load a checkpoint off disk and delegate to `sample_from_checkpoint`.

    Device defaults to CPU: autoregressively sampling a few hundred characters is cheap and
    serial, defaulting to CPU keeps callers (and the test) CUDA-free, and it avoids honoring a
    checkpoint whose `config.device` happens to read "cuda" on a box with no GPU. Pass an
    explicit `device` to override. A `seed` builds a generator on the chosen device so a given
    `(checkpoint, prompt, seed)` reproduces exactly; without one, sampling is unseeded.
    `temperature`, `top_k`, `top_p`, and `repetition_penalty` shape the sampling distribution
    (see `model.generate`).
    """
    ckpt = load_checkpoint(checkpoint_path)  # loads to CPU; we move the model below
    target_device = device if device is not None else torch.device("cpu")

    generator: torch.Generator | None = None
    if seed is not None:
        generator = torch.Generator(device=target_device)
        generator.manual_seed(seed)

    return sample_from_checkpoint(
        ckpt,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        device=target_device,
        generator=generator,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )


def main() -> None:  # pragma: no cover - thin CLI wrapper; the core is what the tests drive
    parser = argparse.ArgumentParser(description="Sample text from a yeGPT checkpoint.")
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH,
        help="Path to the checkpoint written by train.py.",
    )
    parser.add_argument(
        "--prompt", type=str, default="",
        help="Seed text to condition on (default: empty -> primed start).",
    )
    parser.add_argument(
        "-n", "--num-tokens", type=int, default=_DEFAULT_NUM_TOKENS,
        help="Number of characters to generate.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for reproducible sampling (default: unseeded).",
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
    prompt: str = args.prompt
    num_tokens: int = args.num_tokens
    seed: int | None = args.seed
    device_str: str | None = args.device
    temperature: float = args.temperature
    top_k: int | None = args.top_k
    top_p: float | None = args.top_p
    repetition_penalty: float = args.repetition_penalty
    device = torch.device(device_str) if device_str is not None else None

    print(
        generate_text(
            checkpoint_path,
            prompt=prompt,
            max_new_tokens=num_tokens,
            device=device,
            seed=seed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
