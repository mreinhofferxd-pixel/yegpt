"""Tests for CharTokenizer (TICKET-04): round-trip, sorted vocab, errors, persistence."""

from pathlib import Path

import pytest

from yegpt.tokenizer import CharTokenizer

_SAMPLES = [
    "I miss the old Kanye",
    "yeezy\ntaught\tme",
    "good morning, look at the valedictorian",
    "“quotes” — dashes… 100%",
    "",
]


@pytest.mark.parametrize("s", _SAMPLES)
def test_round_trip(s: str) -> None:
    tok = CharTokenizer.from_text("".join(_SAMPLES))
    assert tok.decode(tok.encode(s)) == s


def test_vocab_is_sorted_and_deduped() -> None:
    tok = CharTokenizer.from_text("bca abc")
    assert tok.itos == (" ", "a", "b", "c")
    assert tok.vocab_size == 4


def test_stoi_itos_are_inverse() -> None:
    tok = CharTokenizer.from_text("the quick brown fox")
    for ch, idx in tok.stoi.items():
        assert tok.itos[idx] == ch
    for idx, ch in enumerate(tok.itos):
        assert tok.stoi[ch] == idx


def test_encode_unknown_char_raises() -> None:
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError, match="not in vocab"):
        tok.encode("z")


def test_decode_out_of_range_raises() -> None:
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError, match="out of range"):
        tok.decode([0, 99])


def test_decode_accepts_arbitrary_int_iterable() -> None:
    tok = CharTokenizer.from_text("abc")
    assert tok.decode(i for i in [0, 1, 2]) == "abc"


def test_constructor_rejects_bad_vocab() -> None:
    with pytest.raises(ValueError, match="single characters"):
        CharTokenizer(["ab", "c"])
    with pytest.raises(ValueError, match="duplicate"):
        CharTokenizer(["a", "a"])


def test_save_load_round_trip(tmp_path: Path) -> None:
    tok = CharTokenizer.from_text("“Ye” said\n100%")
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = CharTokenizer.load(path)
    assert loaded == tok
    assert loaded.itos == tok.itos
    # Same id assignment survives the round trip (critical for checkpoint compatibility).
    assert loaded.encode("Ye") == tok.encode("Ye")


def test_load_rejects_foreign_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"format": "something-else", "vocab": ["a"]}', encoding="utf-8")
    with pytest.raises(ValueError, match="not a yegpt-char-tokenizer"):
        CharTokenizer.load(path)
