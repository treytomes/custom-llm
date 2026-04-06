# merge_corpus.py
"""
merge_corpus.py — Merge local and S3 corpus_transformed outputs

Syncs S3 results down, then merges with local results into a single
output directory. Never overwrites existing files — skips any filename
that already exists in the destination.

Usage
-----
python merge_corpus.py \
    --local_dir  ./corpus_transformed_local \
    --s3_uri     s3://bitnet-training-456088019014-us-east-1-an/corpus_transformed/ \
    --output_dir ./corpus_merged
"""

import argparse
import shutil
import subprocess
from pathlib import Path


def sync_from_s3(s3_uri: str, dest_dir: Path):
    print(f"Syncing from {s3_uri} → {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["aws", "s3", "sync", s3_uri, str(dest_dir), "--no-progress"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  S3 sync error: {result.stderr}")
    else:
        print(f"  Done\n")


def merge_dirs(sources: list[Path], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    copied  = 0
    skipped = 0

    for source_dir in sources:
        files = sorted(source_dir.glob("*.txt"))
        print(f"Merging {source_dir.name}: {len(files)} files")

        for src in files:
            dest = output_dir / src.name
            if dest.exists():
                skipped += 1
            else:
                shutil.copy2(src, dest)
                copied += 1

    print(f"\nResult: {copied} copied, {skipped} skipped (duplicates)")
    print(f"Total files in {output_dir}: {len(list(output_dir.glob('*.txt')))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir",  default="../data/corpus/dialogue")
    parser.add_argument("--s3_uri",     default="s3://bitnet-training-456088019014-us-east-1-an/corpus_transformed/")
    parser.add_argument("--output_dir", default="../data/corpus/dialogue_merged")
    parser.add_argument("--skip_s3_sync", action="store_true",
                        help="Skip S3 download if you already have it locally")
    args = parser.parse_args()

    local_dir  = Path(args.local_dir)
    s3_local   = Path("./corpus_transformed_s3")
    output_dir = Path(args.output_dir)

    if not args.skip_s3_sync:
        sync_from_s3(args.s3_uri, s3_local)
    else:
        print("Skipping S3 sync (--skip_s3_sync set)\n")

    sources = [d for d in [local_dir, s3_local] if d.exists()]
    if not sources:
        print("No source directories found.")
        return

    merge_dirs(sources, output_dir)


if __name__ == "__main__":
    main()