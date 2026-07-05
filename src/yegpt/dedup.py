r"""dedup: drop cross-file duplicate passages from the normalized corpus.

Why this exists (TICKET-09.0): ``kanye_verses.txt`` and ``kanye_lyrics.txt`` share ~36% of
their songs, and ``data_prep`` concatenates raw sources *without* dedup. A verse present in
both files would (a) inflate the corpus and (b) leak across the 90/10 train/val split
``dataset`` makes — the same lines landing in both train and val makes val loss look
artificially good and *hides the memorization this project exists to watch*. Removing the
duplication makes the val signal honest. It is the one quality lever left on a corpus that is
staying under the ~1MB gate (consciously waived for this run).

Pipeline placement (decision — standalone module, documented two-step):

    data/raw/*.txt --data_prep--> data/corpus.txt --dedup--> data/corpus.txt (deduped)

``dedup`` is its own module rather than a ``data_prep --dedup`` flag because dedup is a
*corpus-level* operation (it compares passages across the whole concatenated text) while
``data_prep``'s job is generic per-file normalize+concat. Keeping them single-purpose costs
only a documented two-step run order. By default ``dedup`` rewrites ``corpus.txt`` in place
(a gitignored, regenerable artifact), so ``train`` reads the deduped corpus from its usual
default path with no extra flag.

Two passes, two granularities (measured against the real corpus — see TICKET-09.0):

1. ``dedup_text`` — exact whole-**stanza** dedup. Blocks (blank-line-separated stanzas) at or
   above ``min_block_chars`` are kept once, first wins; shorter blocks are always kept. A
   *byte-identical* full stanza is copy-paste bloat, not artful repetition (artful repeats
   carry ad-lib variation), so removing it is safe even when the copies are near each other.

2. ``dedup_line_runs`` — exact **line-run** dedup that survives re-segmentation. The two lyric
   scrapes break the *same* songs into *different* stanzas (363 vs 915 blocks for ~the same
   material), so pass 1 alone misses most overlap. This pass matches on the sequence of
   non-blank **content lines** (ignoring where blank lines fall) and drops a run of
   ``>= min_run_lines`` consecutive lines when an identical run occurred at least
   ``min_distance_lines`` earlier. The **distance gate** is the key: intra-song chorus repeats
   are *local* (same song, well under the gate) and survive as style signal, while cross-file
   duplicate verses are *distant* (different files, thousands of lines apart) and are exactly
   the duplicates that can straddle the contiguous 90/10 split and corrupt val.

Both passes match exact strings (whitespace was normalized upstream); near-duplicates that
differ by a word are left alone, the conservative choice. Both are idempotent: after one pass
no qualifying duplicate remains, so a second pass changes nothing.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from yegpt.data_prep import DEFAULT_CORPUS_PATH

# Stanzas at or above this many characters are treated as whole-verse content that should
# appear once; below it, repetition is intra-song style (choruses/ad-libs) and is preserved.
_DEFAULT_MIN_BLOCK_CHARS: Final[int] = 200

# Line-run pass: a duplicate must be this many consecutive content lines (filters out short
# coincidental matches) AND recur at least this many content lines later (so only distant,
# cross-file/cross-song duplicates are dropped; local chorus repeats are spared). The corpus
# splits cleanly at this distance: every measured cross-file dup is >1000 lines away, every
# local repeat <300, so 500 sits in the gap (TICKET-09.0).
_DEFAULT_MIN_RUN_LINES: Final[int] = 4
_DEFAULT_MIN_DISTANCE_LINES: Final[int] = 500

# In a data_prep-normalized corpus, blocks are separated by exactly one blank line.
_BLOCK_SEPARATOR: Final[str] = "\n\n"
# Re-collapse blank-line runs created when the line-run pass deletes lines mid-text.
_BLANK_RUN_RE: Final[re.Pattern[str]] = re.compile(r"\n{3,}")


@dataclass(frozen=True, slots=True)
class DedupReport:
    """What the whole-stanza pass did: how many stanza blocks and characters it dropped."""

    total_blocks: int
    removed_blocks: int
    chars_before: int
    chars_after: int
    bytes_before: int
    bytes_after: int
    min_block_chars: int

    @property
    def kept_blocks(self) -> int:
        return self.total_blocks - self.removed_blocks

    @property
    def chars_removed(self) -> int:
        return self.chars_before - self.chars_after

    @property
    def bytes_removed(self) -> int:
        return self.bytes_before - self.bytes_after

    @property
    def size_mb_after(self) -> float:
        """Decimal megabytes of the deduped corpus (bytes / 1e6), matching data_prep wording."""
        return self.bytes_after / 1_000_000


@dataclass(frozen=True, slots=True)
class LineRunReport:
    """What the line-run pass did: how many distant duplicate runs/lines it dropped."""

    runs_removed: int
    lines_removed: int
    chars_before: int
    chars_after: int
    bytes_before: int
    bytes_after: int
    min_run_lines: int
    min_distance_lines: int

    @property
    def chars_removed(self) -> int:
        return self.chars_before - self.chars_after

    @property
    def bytes_removed(self) -> int:
        return self.bytes_before - self.bytes_after

    @property
    def size_mb_after(self) -> float:
        return self.bytes_after / 1_000_000


@dataclass(frozen=True, slots=True)
class CorpusDedupReport:
    """The end-to-end result of dedup_corpus: the two passes plus the final size."""

    block: DedupReport
    line_run: LineRunReport

    @property
    def chars_before(self) -> int:
        return self.block.chars_before

    @property
    def chars_after(self) -> int:
        return self.line_run.chars_after

    @property
    def chars_removed(self) -> int:
        return self.chars_before - self.chars_after

    @property
    def size_mb_after(self) -> float:
        return self.line_run.size_mb_after


def _restore_trailing_newline(body: str, *, had_trailing: bool) -> str:
    """Re-attach a single trailing newline (the corpus convention) iff the input had one."""
    return body + "\n" if had_trailing and body else body


def dedup_text(
    text: str, *, min_block_chars: int = _DEFAULT_MIN_BLOCK_CHARS
) -> tuple[str, DedupReport]:
    """Drop exact-duplicate stanza blocks ``>= min_block_chars``; keep short repeats. Pure.

    Splits ``text`` on blank lines into blocks, walks them in order keeping the first
    occurrence of each long block and dropping later exact copies, and rejoins. Short blocks
    (chorus/ad-lib repetition = style signal) are always kept. A single trailing newline is
    preserved (the data_prep corpus convention), so the transform is idempotent.
    """
    if min_block_chars <= 0:
        raise ValueError(f"min_block_chars must be positive, got {min_block_chars}.")

    had_trailing_newline = text.endswith("\n")
    body = text.rstrip("\n")
    blocks = body.split(_BLOCK_SEPARATOR) if body else []

    seen: set[str] = set()
    kept: list[str] = []
    removed = 0
    for block in blocks:
        if len(block) >= min_block_chars:
            if block in seen:
                removed += 1
                continue
            seen.add(block)
        kept.append(block)

    out = _restore_trailing_newline(_BLOCK_SEPARATOR.join(kept), had_trailing=had_trailing_newline)

    report = DedupReport(
        total_blocks=len(blocks),
        removed_blocks=removed,
        chars_before=len(text),
        chars_after=len(out),
        bytes_before=len(text.encode("utf-8")),
        bytes_after=len(out.encode("utf-8")),
        min_block_chars=min_block_chars,
    )
    return out, report


def _line_run_drop_flags(
    content: list[str], min_run_lines: int, min_distance_lines: int
) -> list[bool]:
    """Mark which content lines belong to a distant duplicate run (>= min_run_lines long).

    Greedy left-to-right with first-occurrence-wins. For each position, look up the window of
    the next ``min_run_lines`` lines among earlier *kept* windows; for any earlier occurrence
    at least ``min_distance_lines`` back, extend the match forward against the actual earlier
    lines to get the full (maximal) run length. If that run is long enough, drop it whole and
    skip past it; otherwise keep the line and index its window. Extending against the real
    earlier lines (not just window membership) is what drops the run's tail too, leaving no
    orphan fragment.
    """
    run = min_run_lines
    total = len(content)
    drop = [False] * total
    # window (run consecutive lines) -> positions in `emitted` where it begins.
    window_starts: dict[tuple[str, ...], list[int]] = {}
    emitted: list[str] = []  # the kept content lines, in order
    emitted_origin: list[int] = []  # emitted_origin[p] = original index of emitted[p]

    pos = 0
    while pos < total:
        match_len = 0
        if pos + run <= total:
            window = tuple(content[pos : pos + run])
            for start in window_starts.get(window, ()):
                if pos - emitted_origin[start] < min_distance_lines:
                    continue  # a local repeat (e.g. a chorus) — leave it as style signal
                length = 0
                while (
                    pos + length < total
                    and start + length < len(emitted)
                    and content[pos + length] == emitted[start + length]
                ):
                    length += 1
                if length > match_len:
                    match_len = length
                    if pos + match_len >= total:
                        break
        if match_len >= run:
            for offset in range(match_len):
                drop[pos + offset] = True
            pos += match_len
            continue
        emitted.append(content[pos])
        emitted_origin.append(pos)
        if len(emitted) >= run:
            window_starts.setdefault(tuple(emitted[-run:]), []).append(len(emitted) - run)
        pos += 1

    return drop


def dedup_line_runs(
    text: str,
    *,
    min_run_lines: int = _DEFAULT_MIN_RUN_LINES,
    min_distance_lines: int = _DEFAULT_MIN_DISTANCE_LINES,
) -> tuple[str, LineRunReport]:
    """Drop distant duplicate runs of content lines (cross-file verses); keep local repeats. Pure.

    Matches on the sequence of non-blank content lines so duplicates are caught regardless of
    how each source split them into stanzas. Drops runs of ``>= min_run_lines`` consecutive
    lines whose identical copy occurred ``>= min_distance_lines`` earlier; local repeats
    (choruses) fall under the distance gate and survive. Blank-line runs left behind are
    re-collapsed and a single trailing newline restored, so the transform is idempotent.
    """
    if min_run_lines <= 0:
        raise ValueError(f"min_run_lines must be positive, got {min_run_lines}.")
    if min_distance_lines <= 0:
        raise ValueError(f"min_distance_lines must be positive, got {min_distance_lines}.")

    had_trailing_newline = text.endswith("\n")
    lines = text.split("\n")
    content_positions = [i for i, line in enumerate(lines) if line.strip()]
    content = [lines[i] for i in content_positions]

    drop = _line_run_drop_flags(content, min_run_lines, min_distance_lines)
    dropped_positions = {content_positions[k] for k in range(len(content)) if drop[k]}
    kept_lines = [line for i, line in enumerate(lines) if i not in dropped_positions]

    body = _BLANK_RUN_RE.sub(_BLOCK_SEPARATOR, "\n".join(kept_lines)).strip()
    out = _restore_trailing_newline(body, had_trailing=had_trailing_newline)

    lines_removed = sum(drop)
    runs_removed = sum(1 for k in range(len(drop)) if drop[k] and (k == 0 or not drop[k - 1]))

    report = LineRunReport(
        runs_removed=runs_removed,
        lines_removed=lines_removed,
        chars_before=len(text),
        chars_after=len(out),
        bytes_before=len(text.encode("utf-8")),
        bytes_after=len(out.encode("utf-8")),
        min_run_lines=min_run_lines,
        min_distance_lines=min_distance_lines,
    )
    return out, report


def dedup_corpus(
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    out_path: Path = DEFAULT_CORPUS_PATH,
    *,
    min_block_chars: int = _DEFAULT_MIN_BLOCK_CHARS,
    min_run_lines: int = _DEFAULT_MIN_RUN_LINES,
    min_distance_lines: int = _DEFAULT_MIN_DISTANCE_LINES,
) -> CorpusDedupReport:
    """Read ``corpus_path``, run both dedup passes, write ``out_path``, return the report.

    Whole-stanza pass first (removes byte-identical stanzas), then the line-run pass (removes
    distant duplicate passages that survived re-segmentation). Defaults to rewriting
    ``data/corpus.txt`` in place (the documented ``data_prep`` -> ``dedup`` step); the full
    read-then-write makes the in-place case safe, and the pipeline is idempotent.
    """
    text = corpus_path.read_text(encoding="utf-8")
    after_blocks, block_report = dedup_text(text, min_block_chars=min_block_chars)
    after_runs, line_run_report = dedup_line_runs(
        after_blocks, min_run_lines=min_run_lines, min_distance_lines=min_distance_lines
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(after_runs, encoding="utf-8")
    return CorpusDedupReport(block=block_report, line_run=line_run_report)


def format_report(report: DedupReport) -> str:
    """Human-readable summary of the whole-stanza pass."""
    return "\n".join(
        (
            f"Stanza dedup (blocks >= {report.min_block_chars} chars):",
            f"  blocks     : {report.total_blocks:,} -> {report.kept_blocks:,} "
            f"({report.removed_blocks:,} duplicate block(s) removed)",
            f"  characters : {report.chars_before:,} -> {report.chars_after:,} "
            f"({report.chars_removed:,} removed)",
        )
    )


def format_line_run_report(report: LineRunReport) -> str:
    """Human-readable summary of the line-run pass."""
    return "\n".join(
        (
            f"Line-run dedup (runs >= {report.min_run_lines} lines, "
            f"distance >= {report.min_distance_lines} lines):",
            f"  runs       : {report.runs_removed:,} distant duplicate run(s) removed "
            f"({report.lines_removed:,} lines)",
            f"  characters : {report.chars_before:,} -> {report.chars_after:,} "
            f"({report.chars_removed:,} removed)",
        )
    )


def format_corpus_report(report: CorpusDedupReport) -> str:
    """Human-readable summary of the full two-pass corpus dedup."""
    return "\n".join(
        (
            format_report(report.block),
            format_line_run_report(report.line_run),
            f"Final corpus: {report.chars_after:,} chars, {report.size_mb_after:.2f} MB "
            f"({report.chars_removed:,} chars removed total)",
        )
    )


def main() -> None:  # pragma: no cover - thin CLI wrapper; the core is what the tests drive
    parser = argparse.ArgumentParser(
        description="Drop cross-file duplicate passages from a data_prep corpus.txt."
    )
    parser.add_argument(
        "--corpus", type=Path, default=DEFAULT_CORPUS_PATH,
        help="Corpus to dedup (default: data/corpus.txt).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Where to write the deduped corpus (default: in place over --corpus).",
    )
    parser.add_argument(
        "--min-block-chars", type=int, default=_DEFAULT_MIN_BLOCK_CHARS,
        help="Min block length in chars for the whole-stanza pass (default: 200).",
    )
    parser.add_argument(
        "--min-run-lines", type=int, default=_DEFAULT_MIN_RUN_LINES,
        help="Min consecutive content lines for the line-run pass (default: 4).",
    )
    parser.add_argument(
        "--min-distance-lines", type=int, default=_DEFAULT_MIN_DISTANCE_LINES,
        help="Min recurrence distance (content lines) for the line-run pass (default: 500).",
    )
    args = parser.parse_args()

    # argparse Namespace attrs are Any; read each into a typed local so no Any leaks downstream.
    corpus_path: Path = args.corpus
    out_arg: Path | None = args.out
    min_block_chars: int = args.min_block_chars
    min_run_lines: int = args.min_run_lines
    min_distance_lines: int = args.min_distance_lines
    out_path: Path = out_arg if out_arg is not None else corpus_path

    report = dedup_corpus(
        corpus_path,
        out_path,
        min_block_chars=min_block_chars,
        min_run_lines=min_run_lines,
        min_distance_lines=min_distance_lines,
    )
    print(f"Read  {corpus_path}")
    print(f"Wrote {out_path}")
    print(format_corpus_report(report))


if __name__ == "__main__":  # pragma: no cover
    main()
