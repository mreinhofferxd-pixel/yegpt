"""Tests for tweet_prep: the tweet-CSV source adapter.

Covers the per-tweet cleaning rules (RT/handle/URL/emoji/entity/contraction) and the
end-to-end extract: column detection, BOM handling, drop accounting, dedupe, and the
one-tweet-per-line output that data_prep then ingests.
"""

import csv
from pathlib import Path

from yegpt.tweet_prep import clean_tweet, extract_tweets, format_report


def test_retweets_are_dropped() -> None:
    assert clean_tweet("RT @someone: this is not his voice") == ""


def test_urls_and_handles_stripped_via_normalize() -> None:
    # Delegated to data_prep.normalize_text, but assert the adapter actually applies it.
    assert clean_tweet("yo @ye check http://t.co/abc out") == "yo check out"


def test_reply_only_tweet_empties_out() -> None:
    # Nothing but @mentions -> no content -> dropped.
    assert clean_tweet("@kim @north @saint") == ""


def test_html_entities_unescaped() -> None:
    assert clean_tweet("me &amp; you &lt;3") == "me & you <3"


def test_curly_apostrophe_folded_so_contraction_survives() -> None:
    # The fold must happen before symbol-stripping, or "I'm" would lose its apostrophe.
    assert clean_tweet("I’m the greatest") == "I'm the greatest"


def test_emoji_and_other_scripts_dropped_but_latin_kept() -> None:
    # Emoji + CJK vanish; an accented Latin loanword stays intact.
    assert clean_tweet("fire \U0001f525 中 cup of café") == "fire cup of café"


def test_internal_newlines_flatten_to_one_line() -> None:
    assert clean_tweet("line one\nline two") == "line one line two"


def _write_csv(path: Path, header: list[str], rows: list[list[str]], *, bom: bool = False) -> None:
    encoding = "utf-8-sig" if bom else "utf-8"
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def test_extract_counts_dedupes_and_writes(tmp_path: Path) -> None:
    csv_path = tmp_path / "tweets.csv"
    _write_csv(
        csv_path,
        ["", "Tweet"],  # leading unnamed index column, like the real export
        [
            ["0", "First real tweet"],
            ["1", "RT @x: a retweet"],          # dropped: retweet
            ["2", "@onlyhandles @here"],         # dropped: empties out
            ["3", "First real tweet"],           # dropped: duplicate
            ["4", "Second tweet \U0001f3a4"],    # emoji stripped, kept
        ],
        bom=True,
    )
    out_path = tmp_path / "raw" / "kanye_tweets.txt"

    report = extract_tweets(csv_path, out_path)

    assert report.total_rows == 5
    assert report.retweets_dropped == 1
    assert report.empty_dropped == 1
    assert report.duplicates_dropped == 1
    assert report.kept == 2
    # One tweet per line, trailing newline; the picked column is "Tweet", not the index.
    assert out_path.read_text(encoding="utf-8") == "First real tweet\nSecond tweet\n"
    assert report.char_count == len("First real tweet\nSecond tweet\n")
    assert report.byte_count == report.char_count
    assert "kept    : 2 tweets" in format_report(report)


def test_extract_picks_last_column_when_no_known_name(tmp_path: Path) -> None:
    csv_path = tmp_path / "weird.csv"
    _write_csv(csv_path, ["id", "body"], [["0", "hello world"]])
    out_path = tmp_path / "out.txt"

    report = extract_tweets(csv_path, out_path)

    assert report.kept == 1
    assert out_path.read_text(encoding="utf-8") == "hello world\n"


def test_extract_empty_csv_writes_empty(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    _write_csv(csv_path, ["", "Tweet"], [])
    out_path = tmp_path / "out.txt"

    report = extract_tweets(csv_path, out_path)

    assert report.kept == 0
    assert report.char_count == 0
    assert out_path.read_text(encoding="utf-8") == ""
