"""CharTokenizer: the simplest possible tokenizer — one integer per character.

Why character-level (SPEC.md §2): Kanye text is slang, ad-libs, and chaotic punctuation.
A char vocab (~80-120 symbols) sidesteps all subword/vocab engineering and stays fully
readable — every id maps to exactly one visible character. The cost is longer sequences
and the model spending capacity learning to spell; that's an accepted tradeoff for a
learning build.

The vocab is the **sorted set of distinct characters** in the corpus. Sorting makes the
mapping deterministic (same corpus → same ids), which matters because a checkpoint's
weights are tied to a specific id-for-char assignment. `save`/`load` persist that exact
assignment so `sample.py` decodes with the same vocab `train.py` used.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

_FORMAT_TAG: Final[str] = "yegpt-char-tokenizer-v1"


class CharTokenizer:
    """Bidirectional char <-> int mapping over a fixed, sorted vocabulary."""

    def __init__(self, vocab: Sequence[str]) -> None:
        """`vocab` is the ordered list of single-character symbols (the canonical order).

        Use `from_text` to derive it from a corpus; this constructor validates an
        already-chosen ordering (e.g. one read back from disk).
        """
        for ch in vocab:
            if len(ch) != 1:
                raise ValueError(f"vocab entries must be single characters, got {ch!r}.")
        if len(set(vocab)) != len(vocab):
            raise ValueError("vocab contains duplicate characters.")
        self._itos: Final[tuple[str, ...]] = tuple(vocab)
        self._stoi: Final[dict[str, int]] = {c: i for i, c in enumerate(self._itos)}

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        """Build the vocab as the sorted set of distinct characters in `text`."""
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self._itos)

    @property
    def itos(self) -> tuple[str, ...]:
        """Index -> character, in vocab order."""
        return self._itos

    @property
    def stoi(self) -> dict[str, int]:
        """Character -> index (a copy; the internal map is not exposed for mutation)."""
        return dict(self._stoi)

    def encode(self, text: str) -> list[int]:
        try:
            return [self._stoi[c] for c in text]
        except KeyError as exc:  # a character the corpus never contained
            bad = exc.args[0]
            raise ValueError(
                f"cannot encode {bad!r}: not in vocab (size {self.vocab_size})."
            ) from exc

    def decode(self, ids: Iterable[int]) -> str:
        out: list[str] = []
        for i in ids:
            if not 0 <= i < self.vocab_size:
                raise ValueError(f"id {i} out of range for vocab size {self.vocab_size}.")
            out.append(self._itos[i])
        return "".join(out)

    def save(self, path: Path) -> None:
        """Persist the exact vocab ordering as JSON."""
        payload = {"format": _FORMAT_TAG, "vocab": list(self._itos)}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CharTokenizer:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("format") != _FORMAT_TAG:
            raise ValueError(f"{path} is not a {_FORMAT_TAG} file.")
        vocab_obj = raw.get("vocab")
        if not isinstance(vocab_obj, list):
            raise ValueError("tokenizer file has no 'vocab' list.")
        vocab: list[str] = []
        for item in vocab_obj:
            if not isinstance(item, str) or len(item) != 1:
                raise ValueError(f"invalid vocab entry: {item!r}.")
            vocab.append(item)
        return cls(vocab)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CharTokenizer):
            return NotImplemented
        return self._itos == other._itos

    def __hash__(self) -> int:
        return hash(self._itos)

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"
