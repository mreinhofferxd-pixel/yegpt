"""Tests for data_prep (TICKET-03): normalization steps, idempotency, the build + gate."""

from pathlib import Path

from yegpt.data_prep import (
    MIN_CORPUS_BYTES,
    build_corpus,
    format_report,
    normalize_text,
)


def test_section_tags_are_dropped() -> None:
    # A leading tag vanishes entirely; a tag between content lines leaves a paragraph break.
    raw = "[Verse 1]\nI woke up\n[Chorus: Kanye West]\nlate again\n"
    assert normalize_text(raw) == "I woke up\n\nlate again"


def test_urls_and_handles_removed() -> None:
    raw = "shoutout https://genius.com/x to @kanyewest and @ye_2024 lol"
    assert normalize_text(raw) == "shoutout to and lol"


def test_email_at_is_preserved() -> None:
    # The @ inside an email is preceded by a word char, so it is not a handle.
    assert normalize_text("mail me a@b.com") == "mail me a@b.com"


def test_unicode_quotes_and_dashes_folded() -> None:
    raw = "“I’m the greatest” — no… wait–"
    assert normalize_text(raw) == '"I\'m the greatest" - no... wait-'


def test_genius_artifacts_removed() -> None:
    raw = "last barYou might also like and the end5Embed"
    assert normalize_text(raw) == "last bar and the end"


def test_blank_lines_collapsed_and_trimmed() -> None:
    raw = "\n\n\nline one\n\n\n\nline two\n\n\n"
    assert normalize_text(raw) == "line one\n\nline two"


def test_crlf_normalized() -> None:
    assert normalize_text("a\r\nb\rc") == "a\nb\nc"


def test_normalize_is_idempotent() -> None:
    raw = (
        "[Intro]\r\n“YO”   @ye said\n\n\n"
        "check http://t.co/abc out…\n\n\nEND7Embed\n"
    )
    once = normalize_text(raw)
    assert normalize_text(once) == once


def test_build_corpus_concatenates_sorted_and_writes(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "b_tweets.txt").write_text("@ye yo\n", encoding="utf-8")
    (raw_dir / "a_lyrics.txt").write_text("[Verse]\nbars here\n", encoding="utf-8")
    out = tmp_path / "corpus.txt"

    report = build_corpus(raw_dir, out)

    # Sorted order => a_lyrics before b_tweets; section tag and handle stripped.
    assert out.read_text(encoding="utf-8") == "bars here\n\nyo\n"
    assert report.char_count == len("bars here\n\nyo\n")
    assert report.byte_count == report.char_count
    assert [p.name for p in report.source_files] == ["a_lyrics.txt", "b_tweets.txt"]


def test_build_corpus_is_idempotent(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "x.txt").write_text("[Hook]\nname one\n\n\nname two5Embed\n", encoding="utf-8")
    out = tmp_path / "corpus.txt"

    first = build_corpus(raw_dir, out).char_count
    first_bytes = out.read_bytes()
    second = build_corpus(raw_dir, out).char_count
    assert out.read_bytes() == first_bytes
    assert first == second


def test_under_threshold_flag_and_warning(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "x.txt").write_text("tiny corpus\n", encoding="utf-8")
    out = tmp_path / "corpus.txt"

    report = build_corpus(raw_dir, out)
    assert report.byte_count < MIN_CORPUS_BYTES
    assert report.under_threshold
    assert "WARNING" in format_report(report)


def test_empty_raw_dir_does_not_crash(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out = tmp_path / "corpus.txt"

    report = build_corpus(raw_dir, out)
    assert report.char_count == 0
    assert report.source_files == ()
    assert out.read_text(encoding="utf-8") == ""
    assert "No .txt files found" in format_report(report)
