"""tweet_prep: turn a one-column CSV of Kanye's tweets into a clean ``data/raw/*.txt``.

A *source adapter*, deliberately separate from ``data_prep``. ``data_prep``'s contract is
generic: "normalize and concatenate the ``.txt`` files in ``data/raw/``." A tweet CSV needs
tweet-specific knowledge that has no business in that generic normalizer — retweets (someone
else's voice), @reply-only posts (no content once the handle is stripped), emoji (a char model
would spend an embedding row on each), HTML entities, and exact reposts. So this module does
the tweet-shaped extraction and emits a plain ``.txt`` that ``data_prep`` then ingests like any
other source:

    KanyeTweets.csv --tweet_prep--> data/raw/kanye_tweets.txt --data_prep--> corpus.txt

Per-tweet cleaning (order matters):

1. ``html.unescape`` — ``&amp;``/``&lt;`` back to real characters.
2. drop retweets — a leading ``RT @user`` is a quote of someone else, not his voice.
3. ``normalize_text`` (reused from ``data_prep``) — fold unicode quotes/dashes so contractions
   survive as ASCII ``'``, strip URLs and @handles, collapse whitespace.
4. drop emoji / other-script symbols — keep printable ASCII + Latin letters (so ``café`` and
   ``niño`` live; 🔥 and CJK do not), because each exotic glyph is a near-useless singleton
   vocab row at this scale.
5. flatten to one line — one tweet per line; drop it if it emptied out (e.g. was pure @mentions).

Exact-duplicate tweets are then removed (archives repost), preserving first-seen order.

**Single-voice gate is NOT enforced here.** This adapter assumes the CSV is already *his*
tweets — one text column, no per-row author. It cannot verify authorship, so a multi-user
"tweets mentioning Kanye" scrape would pass straight through and poison the corpus. Keeping the
input single-voice is the author's responsibility upstream (SPEC.md §4 data contract).
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from yegpt.data_prep import DEFAULT_RAW_DIR, normalize_text

# The cleaned tweets land here so a subsequent ``data_prep`` run picks them up automatically
# alongside the other raw ``.txt`` sources.
DEFAULT_OUT_PATH: Final[Path] = DEFAULT_RAW_DIR / "kanye_tweets.txt"

# Tweet text fields go by a few names across archives; fall back to the last column otherwise.
_TEXT_COLUMN_NAMES: Final[tuple[str, ...]] = ("tweet", "text", "content", "full_text")

# Some archives store very large rows (threads, expanded URLs); lift csv's default field cap.
_FIELD_SIZE_LIMIT: Final[int] = 10**7

_RT_RE: Final[re.Pattern[str]] = re.compile(r"^\s*RT\s+@", re.IGNORECASE)
_WS_RE: Final[re.Pattern[str]] = re.compile(r"\s+")


def _pick_text_column(header: list[str]) -> int:
    """Index of the tweet-text column: a known name (case-insensitive), else the last column."""
    if not header:
        raise ValueError("CSV has no header row; cannot locate the tweet text column.")
    lowered = [h.strip().lower() for h in header]
    for name in _TEXT_COLUMN_NAMES:
        if name in lowered:
            return lowered.index(name)
    return len(header) - 1


def _keep_text_chars(text: str) -> str:
    """Drop emoji / non-Latin symbols, keeping printable ASCII, newlines, and Latin letters.

    A char tokenizer pays one embedding row per distinct character, so a stray 🔥 or CJK glyph
    that appears a handful of times is pure overhead. Latin letters are kept so accented
    loanwords/names (``café``, ``niño``) stay intact; everything else above ASCII is discarded.
    """
    kept: list[str] = []
    for ch in text:
        code = ord(ch)
        if ch == "\n" or 0x20 <= code < 0x7F:
            kept.append(ch)
        elif unicodedata.name(ch, "").startswith("LATIN"):
            kept.append(ch)
    return "".join(kept)


def clean_tweet(raw: str) -> str:
    """Clean one raw tweet to a single line, or return ``""`` if it carries no usable content.

    Empty results (retweets, pure-@mention replies, link-only posts) signal "drop this row".
    """
    if _RT_RE.match(raw):
        return ""
    text = html.unescape(raw)
    text = normalize_text(text)  # fold unicode, strip URLs/@handles; reuse the one normalizer
    text = _keep_text_chars(text)
    return _WS_RE.sub(" ", text).strip()  # flatten any internal newlines to one line


@dataclass(frozen=True, slots=True)
class TweetReport:
    """What an extraction produced: how many rows came in, what was dropped, what was written."""

    source_path: Path
    total_rows: int
    retweets_dropped: int
    empty_dropped: int
    duplicates_dropped: int
    kept: int
    char_count: int
    byte_count: int
    out_path: Path

    @property
    def size_mb(self) -> float:
        """Decimal megabytes (bytes / 1e6), matching the data_prep ~1MB gate wording."""
        return self.byte_count / 1_000_000


def _read_text_column(csv_path: Path) -> list[str]:
    """Return the raw text cell of every data row, picking the tweet-text column from the header."""
    csv.field_size_limit(_FIELD_SIZE_LIMIT)
    # utf-8-sig transparently strips a leading BOM (these exports often carry one).
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return []
        column = _pick_text_column(header)
        # Tolerate short/ragged rows rather than indexing past their end.
        return [row[column] for row in reader if len(row) > column]


def extract_tweets(
    csv_path: Path, out_path: Path = DEFAULT_OUT_PATH
) -> TweetReport:
    """Read a tweet CSV, clean + dedupe the text column, write a ``.txt``, return a report.

    Output is one cleaned tweet per line with a trailing newline (empty if nothing survived),
    ready for ``data_prep`` to normalize and concatenate with the other raw sources.
    """
    raw_rows = _read_text_column(csv_path)

    retweets = 0
    empty = 0
    duplicates = 0
    seen: set[str] = set()
    kept: list[str] = []
    for raw in raw_rows:
        if _RT_RE.match(raw):
            retweets += 1
            continue
        text = clean_tweet(raw)
        if not text:
            empty += 1
            continue
        if text in seen:
            duplicates += 1
            continue
        seen.add(text)
        kept.append(text)

    corpus = "\n".join(kept)
    corpus = corpus + "\n" if corpus else corpus
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(corpus, encoding="utf-8")

    return TweetReport(
        source_path=csv_path,
        total_rows=len(raw_rows),
        retweets_dropped=retweets,
        empty_dropped=empty,
        duplicates_dropped=duplicates,
        kept=len(kept),
        char_count=len(corpus),
        byte_count=len(corpus.encode("utf-8")),
        out_path=out_path,
    )


def format_report(report: TweetReport) -> str:
    """Human-readable summary of an extraction."""
    return "\n".join(
        (
            f"Read {report.total_rows:,} rows from {report.source_path.name}",
            f"  dropped : {report.retweets_dropped:,} retweets, "
            f"{report.empty_dropped:,} empty/@-only, {report.duplicates_dropped:,} duplicate",
            f"  kept    : {report.kept:,} tweets",
            f"Wrote {report.out_path}",
            f"  characters : {report.char_count:,}",
            f"  size       : {report.size_mb:.2f} MB ({report.byte_count:,} bytes)",
        )
    )


def main() -> None:  # pragma: no cover - thin CLI wrapper; the core is what the tests drive
    parser = argparse.ArgumentParser(
        description="Clean a one-column Kanye-tweets CSV into a data/raw/*.txt for data_prep."
    )
    parser.add_argument("csv_path", type=Path, help="Path to the tweets CSV.")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT_PATH,
        help="Where to write the cleaned .txt (default: data/raw/kanye_tweets.txt).",
    )
    args = parser.parse_args()
    csv_path: Path = args.csv_path
    out_path: Path = args.out
    print(format_report(extract_tweets(csv_path, out_path)))


if __name__ == "__main__":  # pragma: no cover
    main()
