"""Tests for dedup (TICKET-09.0): the two passes that finalize the corpus.

Pass 1 (`dedup_text`): exact whole-stanza dedup — long duplicate blocks removed (first wins),
short repeated blocks (choruses = style signal) survive.

Pass 2 (`dedup_line_runs`): distant duplicate runs of content lines removed regardless of how
each source segmented them into stanzas, while *local* repeats (choruses, under the distance
gate) survive. Tests pass a small `min_distance_lines` so distant cases stay compact.
"""

from pathlib import Path

from yegpt.dedup import (
    CorpusDedupReport,
    dedup_corpus,
    dedup_line_runs,
    dedup_text,
    format_line_run_report,
    format_report,
)


def _long_block() -> str:
    """A single multi-line stanza well over the 200-char default threshold (no blank line)."""
    return "\n".join(f"line {i}: a long verse with plenty of real content" for i in range(8))


# --- Pass 1: whole-stanza dedup -------------------------------------------------------------


def test_long_duplicate_block_removed_once() -> None:
    block = _long_block()
    assert len(block) >= 200
    # The same long verse appears twice, with short distinct blocks around/between it.
    text = f"intro\n\n{block}\n\nmiddle\n\n{block}\n\noutro\n"
    out, report = dedup_text(text)
    assert out.count(block) == 1
    assert report.removed_blocks == 1
    # The short surrounding blocks are all preserved.
    assert "intro" in out and "middle" in out and "outro" in out


def test_short_repeated_block_survives() -> None:
    chorus = "we don't care\nwe don't care"  # a repeated hook, well under the threshold
    assert len(chorus) < 200
    text = f"{chorus}\n\nverse one here\n\n{chorus}\n"
    out, report = dedup_text(text)
    assert out.count(chorus) == 2  # both copies of the chorus kept
    assert report.removed_blocks == 0


def test_threshold_is_inclusive_of_min_block_chars() -> None:
    block = "x" * 50
    text = f"{block}\n\n{block}\n"
    # 50 >= 51 is False -> both kept; 50 >= 50 is True -> the second is dropped.
    _, kept = dedup_text(text, min_block_chars=51)
    _, dropped = dedup_text(text, min_block_chars=50)
    assert kept.removed_blocks == 0
    assert dropped.removed_blocks == 1


def test_block_dedup_is_idempotent() -> None:
    block = _long_block()
    text = f"{block}\n\nbridge\n\n{block}\n"
    once, _ = dedup_text(text)
    twice, report = dedup_text(once)
    assert twice == once
    assert report.removed_blocks == 0


def test_block_report_counts_are_correct() -> None:
    block = _long_block()
    text = f"{block}\n\n{block}\n\n{block}\n"  # the same long block three times
    out, report = dedup_text(text)
    assert report.total_blocks == 3
    assert report.removed_blocks == 2
    assert report.kept_blocks == 1
    assert report.chars_before == len(text)
    assert report.chars_after == len(out)
    assert report.chars_removed == len(text) - len(out)
    assert "2 duplicate block(s) removed" in format_report(report)


# --- Pass 2: line-run dedup -----------------------------------------------------------------


def _filler(n: int) -> str:
    """n distinct single-line stanzas to put real distance between duplicate runs."""
    return "\n\n".join(f"unique filler line {i}" for i in range(n))


def test_line_run_distant_duplicate_removed() -> None:
    verse = "alpha\nbravo\ncharlie\ndelta"  # 4 content lines
    text = f"{verse}\n\n{_filler(6)}\n\n{verse}\n"
    out, report = dedup_line_runs(text, min_run_lines=4, min_distance_lines=3)
    assert out.count(verse) == 1
    assert report.runs_removed == 1
    assert report.lines_removed == 4
    assert report.chars_removed == report.chars_before - report.chars_after
    assert "1 distant duplicate run(s) removed" in format_line_run_report(report)


def test_line_run_local_repeat_survives() -> None:
    chorus = "hook one\nhook two\nhook three\nhook four"  # >= min_run_lines, but local
    text = f"{chorus}\n\nbridge line\n\n{chorus}\n"
    out, report = dedup_line_runs(text, min_run_lines=4, min_distance_lines=500)
    assert out.count(chorus) == 2  # under the distance gate -> kept as style signal
    assert report.runs_removed == 0


def test_line_run_short_run_survives_even_when_distant() -> None:
    short = "ay\nyeah\nuh"  # 3 content lines, below min_run_lines=4
    text = f"{short}\n\n{_filler(6)}\n\n{short}\n"
    out, report = dedup_line_runs(text, min_run_lines=4, min_distance_lines=3)
    assert out.count(short) == 2
    assert report.runs_removed == 0


def test_line_run_matches_across_different_stanza_segmentation() -> None:
    # Same four lines, but the second copy is broken into two stanzas (a blank line inserted) —
    # exactly the cross-file re-segmentation the whole-stanza pass cannot catch.
    original = "line a\nline b\nline c\nline d"
    resegmented = "line a\nline b\n\nline c\nline d"
    text = f"{original}\n\n{_filler(6)}\n\n{resegmented}\n"
    _, report = dedup_line_runs(text, min_run_lines=4, min_distance_lines=3)
    assert report.runs_removed == 1
    assert report.lines_removed == 4


def test_line_run_is_idempotent() -> None:
    verse = "one\ntwo\nthree\nfour\nfive"
    text = f"{verse}\n\n{_filler(6)}\n\n{verse}\n"
    once, _ = dedup_line_runs(text, min_run_lines=4, min_distance_lines=3)
    twice, report = dedup_line_runs(once, min_run_lines=4, min_distance_lines=3)
    assert twice == once
    assert report.runs_removed == 0


# --- The two-pass corpus orchestrator -------------------------------------------------------


def test_dedup_corpus_runs_both_passes_in_place(tmp_path: Path) -> None:
    block = _long_block()
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(f"{block}\n\n{block}\n", encoding="utf-8")  # an exact whole-stanza dup
    report = dedup_corpus(corpus, corpus)  # default is in-place; pass it explicitly here
    assert isinstance(report, CorpusDedupReport)
    assert report.block.removed_blocks == 1  # pass 1 caught the exact stanza dup
    assert corpus.read_text(encoding="utf-8").count(block) == 1
    # The full two-pass pipeline is idempotent.
    first = corpus.read_text(encoding="utf-8")
    dedup_corpus(corpus, corpus)
    assert corpus.read_text(encoding="utf-8") == first


def test_dedup_corpus_line_run_pass_catches_resegmented(tmp_path: Path) -> None:
    block = _long_block()  # 8 identical content lines, > 200 chars
    resegmented = block.replace("\n", "\n\n", 1)  # split the first line into its own stanza
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(f"{block}\n\n{_filler(6)}\n\n{resegmented}\n", encoding="utf-8")
    report = dedup_corpus(corpus, corpus, min_run_lines=4, min_distance_lines=3)
    # Pass 1 can't match (the copy is segmented differently); pass 2 does, via content lines.
    assert report.block.removed_blocks == 0
    assert report.line_run.runs_removed == 1
    assert corpus.read_text(encoding="utf-8").count(block) == 1
