import boto3
import os
from dotenv import load_dotenv
from pathlib import Path

import config


S3_BUCKET_NAME_KEY = "S3_BUCKET_NAME"
s3 = boto3.client("s3")


def download_file(bucketName, key):
    local_path = config.DATA_DIR / key
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists():
        print(f"Skipping existing file: {local_path}")
        return

    print(f"Downloading s3://{bucketName}/{key} -> {local_path}")
    s3.download_file(bucketName, key, str(local_path))


def restore(bucketName:str=None):
    if bucketName is None:
        bucketName = os.getenv(S3_BUCKET_NAME_KEY)
    print(f"Uploading to bucket: {bucketName}")

    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucketName):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            download_file(bucketName, key)

    print("Restore complete.")


if __name__ == '__main__':
    load_dotenv()
    restore()
