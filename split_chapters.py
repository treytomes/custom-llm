# split_chapters.py
"""
split_chapters.py — Split Gutenberg novels into individual chapter files

Reads preprocessed novel text files (boilerplate already stripped by
prep_corpus.py) and splits them into one file per chapter or diary entry.
Output files are named {book_stem}_ch001.txt, {book_stem}_ch002.txt, etc.

Supports multiple chapter header formats:
  - Standard:     CHAPTER I, CHAPTER 1, PART TWO, BOOK III
  - Numeric only: 1, 2, 3 (A Little Princess)
  - Roman only:   I, II, III (Anne of Avonlea)
  - Numbered:     1. Title (some Gutenberg formats)
  - Preface
  - Diary entries: Saturday, 20 June, 1942 (Anne Frank)
  - Letter headers: My dear Mr. Washington (Carver letters)

Per-book pattern overrides are keyed on substrings of the filename stem.
Add new entries to BOOK_PATTERNS to handle additional formats.

Usage:
    python split_chapters.py
    python split_chapters.py --input_dir ./corpus_novels --output_dir ./chapters
    python split_chapters.py --book diary_of_a_young_girl.txt  (test one book)
    python split_chapters.py --upload --bucket bitnet-training-456088019014-us-east-1-an
"""

import argparse
import re
import boto3
from pathlib import Path


# ── Chapter patterns ───────────────────────────────────────────────────────────

CHAPTER_PATTERN = re.compile(
    r'\n+'
    r'('
    r'(?:\s*CHAPTER|PART|BOOK)\s+(?:[IVXLCDM]+|\d+|ONE|TWO|THREE|FOUR|FIVE|'
    r'SIX|SEVEN|EIGHT|NINE|TEN|THE\s+FIRST|THE\s+SECOND|THE\s+THIRD|'
    r'THE\s+FOURTH|THE\s+FIFTH)\.?\s*[^\n]*'
    r'|'
    r'^\s*PREFACE\s*$'
    r'|'
    r'^\d+\..*$'
    r'|'
    r'^\s*\d{1,3}\s*$'
    r'|'
    r'^\s*(?:I+|X[XCVI]+|XL|XC|L?X{1,3}|IX|IV|VI{0,3}|II{1,2})\s*\.?\s*$'
    r')',
    re.IGNORECASE | re.MULTILINE
)

# Anne Frank's diary — entries headed by long-form dates
# Handles both:
#   "Saturday, 20 June, 1942"  (comma before year)
#   "Sunday, 2 August 1942"    (no comma before year)
DIARY_PATTERN = re.compile(
    r'\n+'
    r'('
    r'(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)'
    r'\s*,\s*\d{1,2}\s+'
    r'(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)'
    r'(?:,\s*|\s+)\d{4}'
    r')',
    re.IGNORECASE | re.MULTILINE
)

# Carver letters — entries headed by salutations or letter-style dates
# e.g. "My dear Mr. Washington" or "Dear friend" or "January 5, 1922"
LETTER_PATTERN = re.compile(
    r'\n+'
    r'('
    r'(?:My\s+dear|Dear)\s+[A-Z][^\n]{2,60}'
    r'|'
    r'(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)'
    r'\s+\d{1,2},?\s+\d{4}'
    r')',
    re.IGNORECASE | re.MULTILINE
)


# ── Per-book pattern overrides ─────────────────────────────────────────────────
#
# Keys are substrings matched against the lowercase filename stem.
# First matching key wins.
# Add new entries here as you encounter books with unusual formatting.

BOOK_PATTERNS = {
    "diary":      DIARY_PATTERN,    # Anne Frank: The Diary of a Young Girl
    "anne_frank": DIARY_PATTERN,
    "frank":      DIARY_PATTERN,
    "carver":     LETTER_PATTERN,   # George Washington Carver letters
}


def get_pattern_for_book(input_path: Path) -> tuple[re.Pattern, str]:
    """
    Return the appropriate split pattern and its name for a given book file.
    """
    stem_lower = input_path.stem.lower()
    for key, pattern in BOOK_PATTERNS.items():
        if key in stem_lower:
            return pattern, f"override:{key}"
    return CHAPTER_PATTERN, "default"


# ── Single book splitter ───────────────────────────────────────────────────────

def split_book(
    input_path: Path,
    output_dir: Path,
    pattern: re.Pattern = None,
    min_words: int = 100,
) -> int:
    """
    Split a single book file into section files.

    Args:
        input_path: path to the preprocessed book .txt file
        output_dir: directory to write section files into
        pattern:    regex pattern to split on; if None, auto-detected
                    from filename via get_pattern_for_book()
        min_words:  skip sections shorter than this (TOC artifacts etc.)

    Returns:
        number of sections written
    """
    text = input_path.read_text(encoding="utf-8", errors="replace")
    stem = input_path.stem

    if pattern is None:
        pattern, pattern_name = get_pattern_for_book(input_path)
        print(f"  Pattern: {pattern_name}")
    
    parts = pattern.split(text)

    # parts layout after split():
    #   parts[0]       — pre-section material (title page, preface etc.)
    #   parts[1]       — first header
    #   parts[2]       — first content
    #   parts[3]       — second header
    #   parts[4]       — second content
    #   ...

    output_dir.mkdir(parents=True, exist_ok=True)

    section_num      = 0
    sections_written = 0
    i = 1

    while i < len(parts) - 1:
        header  = parts[i].strip() if parts[i] is not None else ""
        content = parts[i + 1].strip() if (i + 1 < len(parts) and parts[i + 1] is not None) else ""
        section_num += 1

        # Skip None groups from non-matching pattern alternatives
        if not header and not content:
            i += 2
            continue

        word_count = len(content.split())

        if word_count < min_words:
            print(f"  ch{section_num:03d} — skipping "
                  f"({word_count} words, below minimum) [{header[:40]}]")
            i += 2
            continue

        out_path = output_dir / f"{stem}_ch{section_num:03d}.txt"
        out_path.write_text(
            f"{header}\n\n{content}",
            encoding="utf-8"
        )

        print(f"  ch{section_num:03d} — {word_count:,} words  [{header[:50]}]")
        sections_written += 1
        i += 2

    # If no section boundaries found, write whole book as single chunk
    if sections_written == 0:
        word_count = len(text.split())
        print(f"  No section boundaries found — "
              f"writing as single file ({word_count:,} words)")
        out_path = output_dir / f"{stem}_ch001.txt"
        out_path.write_text(text, encoding="utf-8")
        sections_written = 1

    return sections_written


# ── S3 upload ──────────────────────────────────────────────────────────────────

def upload_chapters_to_s3(
    chapters_dir: Path,
    bucket: str,
    s3_prefix: str = "chapters/",
) -> int:
    """Upload all chapter files to S3."""
    s3       = boto3.client("s3")
    files    = sorted(chapters_dir.glob("*.txt"))
    uploaded = 0

    print(f"\nUploading {len(files)} files to s3://{bucket}/{s3_prefix}")

    for fpath in files:
        key = s3_prefix + fpath.name
        s3.upload_file(str(fpath), bucket, key)
        print(f"  Uploaded: {key}")
        uploaded += 1

    print(f"\nUpload complete: {uploaded} files")
    return uploaded


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Split Gutenberg novels into chapter/entry files"
    )
    parser.add_argument("--input_dir",  type=str, default="./corpus_novels",
                        help="Directory containing preprocessed .txt files")
    parser.add_argument("--output_dir", type=str, default="./chapters",
                        help="Directory to write split files into")
    parser.add_argument("--min_words",  type=int, default=100,
                        help="Minimum word count to keep a section (default 100)")
    parser.add_argument("--upload",     action="store_true",
                        help="Upload chapter files to S3 after splitting")
    parser.add_argument("--bucket",     type=str,
                        default="bitnet-training-456088019014-us-east-1-an",
                        help="S3 bucket to upload to (requires --upload)")
    parser.add_argument("--s3_prefix",  type=str, default="chapters/",
                        help="S3 key prefix for uploaded files")
    parser.add_argument("--book",       type=str, default=None,
                        help="Process only this specific filename (for testing)")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return

    # Single book mode for testing
    if args.book:
        book_path = input_dir / args.book
        if not book_path.exists():
            print(f"File not found: {book_path}")
            return
        books = [book_path]
    else:
        books = sorted(input_dir.glob("*.txt"))

    if not books:
        print(f"No .txt files found in {input_dir}")
        return

    print(f"Found {len(books)} book(s) to split\n")

    total_sections = 0

    for book in books:
        print(f"{'─' * 60}")
        print(f"Splitting: {book.name}")
        n = split_book(book, output_dir, min_words=args.min_words)
        total_sections += n
        print(f"  → {n} sections written\n")

    print(f"{'═' * 60}")
    print(f"Total sections written : {total_sections}")
    print(f"Output directory       : {output_dir}")

    if args.upload:
        upload_chapters_to_s3(output_dir, args.bucket, args.s3_prefix)
    else:
        print(f"\nTo upload to S3 when ready:")
        print(f"  python split_chapters.py --upload --bucket {args.bucket}")


if __name__ == "__main__":
    main()
