"""data_prep: turn whatever lands in ``data/raw/*.txt`` into a single ``data/corpus.txt``.

Per SPEC.md §4 this is a *gate*, not just a converter. It normalizes the raw text,
concatenates it, and reports size — printing a loud warning under ~1MB, because below
that a from-scratch char model memorizes instead of generalizing.

Normalization is a pipeline of small, individually-testable steps (`normalize_text`):
newline canonicalization → unicode quote/dash folding → per-line scrubbing (URLs,
@handles, section tags like ``[Verse 1]``, known Genius scrape artifacts) → whitespace
and blank-line collapse. It is **idempotent**: ``normalize_text(normalize_text(x)) ==
normalize_text(x)`` and re-running the whole build on unchanged input yields byte-identical
output.

What is deliberately NOT stripped: arbitrary freeform song titles. There is no reliable
char-level signal that separates a title line from a real lyric line, so auto-detecting
them would delete real content. Section tags, URLs, handles, and the common Genius
artifacts are the reliably-removable metadata; freeform titles are left to the author.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# Project-default locations (overridable via the functions' arguments — no globals leak).
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR: Final[Path] = _REPO_ROOT / "data" / "raw"
DEFAULT_CORPUS_PATH: Final[Path] = _REPO_ROOT / "data" / "corpus.txt"

# SPEC.md §4 gate: warn below ~1MB of UTF-8 text.
MIN_CORPUS_BYTES: Final[int] = 1_000_000

# Unicode characters that should fold to a plain ASCII equivalent so the char vocab stays
# small and consistent (curly quotes, the dash family, ellipsis, exotic spaces, invisibles).
_CHAR_MAP: Final[dict[str, str]] = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",  # single quotes
    "“": '"', "”": '"', "„": '"', "‟": '"',  # double quotes
    "´": "'",                                                # acute accent as apostrophe
    "–": "-", "—": "-", "―": "-", "−": "-",  # en/em/bar/minus dashes
    "…": "...",                                              # ellipsis
    " ": " ", " ": " ", " ": " ", " ": " ",  # non-breaking/thin spaces
    "﻿": "", "​": "", "‌": "", "‍": "", "�": "",  # invisibles/replacement
}
_CHAR_TABLE: Final[dict[int, str]] = {ord(k): v for k, v in _CHAR_MAP.items()}

_URL_RE: Final[re.Pattern[str]] = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
# An @handle, but not the @ inside an email (which is preceded by a word char).
_HANDLE_RE: Final[re.Pattern[str]] = re.compile(r"(?<!\w)@\w+")
# A whole line that is only a bracketed section tag, e.g. "[Verse 1]", "[Chorus: Kanye]".
_SECTION_TAG_RE: Final[re.Pattern[str]] = re.compile(r"^\[[^\]]*\]$")
# Genius scrape artifacts: trailing "...word5Embed", and the injected "You might also like".
_EMBED_RE: Final[re.Pattern[str]] = re.compile(r"\d*Embed\s*$")
_GENIUS_INLINE_RE: Final[re.Pattern[str]] = re.compile(r"You might also like")
_MULTISPACE_RE: Final[re.Pattern[str]] = re.compile(r"[ \t]+")
_BLANKLINES_RE: Final[re.Pattern[str]] = re.compile(r"\n{3,}")


def _scrub_line(line: str) -> str:
    """Apply per-line removals. Returns "" for lines that were pure metadata."""
    line = _URL_RE.sub("", line)
    line = _HANDLE_RE.sub("", line)
    line = _GENIUS_INLINE_RE.sub("", line)
    line = _EMBED_RE.sub("", line)
    line = _MULTISPACE_RE.sub(" ", line).strip()
    if _SECTION_TAG_RE.match(line):
        return ""
    return line


def normalize_text(raw: str) -> str:
    """Full normalization pipeline. Pure and idempotent.

    Returns text with no leading/trailing whitespace and at most one blank line between
    blocks. Does not append a trailing newline (the caller decides how blocks are joined).
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(_CHAR_TABLE)
    scrubbed = "\n".join(_scrub_line(line) for line in text.split("\n"))
    scrubbed = _BLANKLINES_RE.sub("\n\n", scrubbed)
    return scrubbed.strip()


@dataclass(frozen=True, slots=True)
class CorpusReport:
    """Outcome of a corpus build: what went in, how big it came out, and the gate verdict."""

    source_files: tuple[Path, ...]
    char_count: int
    byte_count: int
    out_path: Path

    @property
    def size_mb(self) -> float:
        """Decimal megabytes (bytes / 1e6), matching the ~1MB gate wording."""
        return self.byte_count / 1_000_000

    @property
    def under_threshold(self) -> bool:
        return self.byte_count < MIN_CORPUS_BYTES


def find_raw_files(raw_dir: Path) -> tuple[Path, ...]:
    """All ``*.txt`` under ``raw_dir``, sorted for deterministic concatenation order."""
    return tuple(sorted(raw_dir.glob("*.txt")))


def _read_text(path: Path) -> str:
    # Robust to stray bytes; the replacement char is folded away in normalization.
    return path.read_text(encoding="utf-8", errors="replace")


def build_corpus(
    raw_dir: Path = DEFAULT_RAW_DIR,
    out_path: Path = DEFAULT_CORPUS_PATH,
) -> CorpusReport:
    """Normalize every raw ``.txt``, concatenate, write ``corpus.txt``, return a report.

    Idempotent: same inputs produce byte-identical output.
    """
    sources = find_raw_files(raw_dir)
    blocks = [normalize_text(_read_text(p)) for p in sources]
    blocks = [b for b in blocks if b]  # drop files that normalized to nothing
    corpus = "\n\n".join(blocks)
    corpus = corpus + "\n" if corpus else corpus

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(corpus, encoding="utf-8")

    return CorpusReport(
        source_files=sources,
        char_count=len(corpus),
        byte_count=len(corpus.encode("utf-8")),
        out_path=out_path,
    )


def format_report(report: CorpusReport) -> str:
    """Human-readable summary, including the sub-1MB gate warning when triggered."""
    lines: list[str] = []
    if report.source_files:
        lines.append(f"Read {len(report.source_files)} source file(s):")
        lines.extend(f"  - {p.name}" for p in report.source_files)
    else:
        lines.append("No .txt files found in the raw directory.")
    lines.append(f"Wrote {report.out_path}")
    lines.append(f"  characters : {report.char_count:,}")
    lines.append(f"  size       : {report.size_mb:.2f} MB ({report.byte_count:,} bytes)")
    if report.under_threshold:
        lines.append("")
        lines.append(
            f"WARNING: corpus is under ~1MB ({report.size_mb:.2f} MB). A from-scratch "
            "char model will MEMORIZE this rather than learn the style. Add more "
            "transcripts/tweets before a real training run (SPEC.md section 4)."
        )
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - thin CLI wrapper
    report = build_corpus()
    print(format_report(report))
    if report.under_threshold:
        sys.exit(0)  # a warning, not an error: the author may be mid-collection


if __name__ == "__main__":  # pragma: no cover
    main()
