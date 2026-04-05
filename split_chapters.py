# split_chapters.py
"""
split_chapters.py — Split Gutenberg novels into individual chapter files

Reads preprocessed novel text files (boilerplate already stripped by
prep_corpus.py) and splits them into one file per chapter. Output files
are named {book_stem}_ch001.txt, {book_stem}_ch002.txt, etc.

These chapter files are uploaded to S3 and consumed by the SageMaker
transformation job (transform_corpus.py).

Usage:
    python split_chapters.py
    python split_chapters.py --input_dir ./corpus_novels --output_dir ./chapters
    python split_chapters.py --upload --bucket bitnet-training-456088019014-us-east-1-an
"""

import argparse
import re
import boto3
from pathlib import Path


# ── Chapter boundary detection ─────────────────────────────────────────────────
#
# Matches common Gutenberg chapter header formats:
#   CHAPTER I
#   CHAPTER 1
#   Chapter One
#   CHAPTER THE FIRST
#   Part I / Part 1
#
# The pattern splits on the header itself so we can reconstruct it
# as the first line of each chapter file.

CHAPTER_PATTERN = re.compile(
    r'\n((?:CHAPTER|PART|BOOK)\s+(?:[IVXLCDM]+|\d+|ONE|TWO|THREE|FOUR|FIVE|'
    r'SIX|SEVEN|EIGHT|NINE|TEN|THE\s+FIRST|THE\s+SECOND|THE\s+THIRD|'
    r'THE\s+FOURTH|THE\s+FIFTH)[^\n]*)',
    re.IGNORECASE
)


# ── Single book splitter ───────────────────────────────────────────────────────

def split_book(input_path: Path, output_dir: Path, min_words: int = 100) -> int:
    """
    Split a single book file into chapter files.

    Args:
        input_path: path to the preprocessed book .txt file
        output_dir: directory to write chapter files into
        min_words:  skip chapters shorter than this (TOC artifacts etc.)

    Returns:
        number of chapters written
    """
    text = input_path.read_text(encoding="utf-8", errors="replace")
    stem = input_path.stem

    parts = CHAPTER_PATTERN.split(text)

    # parts[0] is pre-chapter material (title page, preface etc.)
    # then alternating: header, content, header, content, ...

    output_dir.mkdir(parents=True, exist_ok=True)

    chapter_num     = 0
    chapters_written = 0
    i = 1

    while i < len(parts) - 1:
        header  = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        chapter_num += 1

        word_count = len(content.split())

        if word_count < min_words:
            print(f"  ch{chapter_num:03d} — skipping ({word_count} words, below minimum)")
            i += 2
            continue

        out_path = output_dir / f"{stem}_ch{chapter_num:03d}.txt"
        out_path.write_text(
            f"{header}\n\n{content}",
            encoding="utf-8"
        )

        print(f"  ch{chapter_num:03d} — {word_count:,} words → {out_path.name}")
        chapters_written += 1
        i += 2

    # If no chapter boundaries were found, write the whole book as one chunk
    if chapters_written == 0:
        word_count = len(text.split())
        print(f"  No chapter boundaries found — writing as single file")
        out_path = output_dir / f"{stem}_ch001.txt"
        out_path.write_text(text, encoding="utf-8")
        chapters_written = 1

    return chapters_written


# ── S3 upload ──────────────────────────────────────────────────────────────────

def upload_chapters_to_s3(
    chapters_dir: Path,
    bucket: str,
    s3_prefix: str = "chapters/",
) -> int:
    """
    Upload all chapter files to S3.

    Args:
        chapters_dir: local directory containing chapter files
        bucket:       S3 bucket name
        s3_prefix:    key prefix in the bucket

    Returns:
        number of files uploaded
    """
    s3     = boto3.client("s3")
    files  = sorted(chapters_dir.glob("*.txt"))
    uploaded = 0

    print(f"\nUploading {len(files)} chapter files to s3://{bucket}/{s3_prefix}")

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
        description="Split Gutenberg novels into chapter files"
    )
    parser.add_argument("--input_dir",  type=str, default="./corpus_novels",
                        help="Directory containing preprocessed novel .txt files")
    parser.add_argument("--output_dir", type=str, default="./chapters",
                        help="Directory to write chapter files into")
    parser.add_argument("--min_words",  type=int, default=100,
                        help="Minimum word count to keep a chapter (default 100)")
    parser.add_argument("--upload",     action="store_true",
                        help="Upload chapter files to S3 after splitting")
    parser.add_argument("--bucket",     type=str,
                        default="bitnet-training-456088019014-us-east-1-an",
                        help="S3 bucket to upload to (requires --upload)")
    parser.add_argument("--s3_prefix",  type=str, default="chapters/",
                        help="S3 key prefix for uploaded files")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return

    books = sorted(input_dir.glob("*.txt"))
    if not books:
        print(f"No .txt files found in {input_dir}")
        return

    print(f"Found {len(books)} book(s) to split\n")

    total_chapters = 0

    for book in books:
        print(f"Splitting: {book.name}")
        n = split_book(book, output_dir, min_words=args.min_words)
        total_chapters += n
        print(f"  → {n} chapters written\n")

    print(f"Total chapters written: {total_chapters}")
    print(f"Output directory: {output_dir}")

    if args.upload:
        upload_chapters_to_s3(output_dir, args.bucket, args.s3_prefix)
    else:
        print(f"\nTo upload to S3:")
        print(f"  python split_chapters.py --upload --bucket {args.bucket}")


if __name__ == "__main__":
    main()
