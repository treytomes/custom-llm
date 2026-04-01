"""
sync_corpus_to_s3.py

Deletes the contents of the S3 corpus folder and uploads the local corpus directory.

Usage:
python sync_corpus_to_s3.py
"""

import boto3
from pathlib import Path

BUCKET = "bitnet-training-456088019014-us-east-1-an"
S3_PREFIX = "corpus/"
LOCAL_CORPUS = Path("./corpus")

def clear_s3_prefix(s3, bucket, prefix):
    print(f"Clearing s3://{bucket}/{prefix}")

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    keys = []
    for page in pages:
        for obj in page.get("Contents", []):
            keys.append({"Key": obj["Key"]})

            if not keys:
                print("No existing files found.")
                return

            for i in range(0, len(keys), 1000):
                batch = keys[i:i+1000]
                s3.delete_objects(
                    Bucket=bucket,
                    Delete={"Objects": batch}
                )

    print(f"Deleted {len(keys)} objects.")


def upload_corpus(s3, bucket, prefix, local_dir):
    files = list(local_dir.rglob("*"))

    upload_count = 0

    for path in files:
        if not path.is_file():
            continue

        rel = path.relative_to(local_dir)
        s3_key = f"{prefix}{rel.as_posix()}"

        print(f"Uploading {path} → s3://{bucket}/{s3_key}")

        s3.upload_file(
            str(path),
            bucket,
            s3_key
        )

        upload_count += 1

    print(f"Uploaded {upload_count} files.")


def main():
    if not LOCAL_CORPUS.exists():
        raise RuntimeError("Local corpus directory does not exist")

    s3 = boto3.client("s3")
    clear_s3_prefix(s3, BUCKET, S3_PREFIX)
    upload_corpus(s3, BUCKET, S3_PREFIX, LOCAL_CORPUS)
    print("Corpus sync complete.")


if __name__ == "__main__":
    main()