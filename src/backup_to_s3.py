import boto3
import os
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from pathlib import Path

import config


S3_BUCKET_NAME_KEY = "S3_BUCKET_NAME"
s3 = boto3.client("s3")


def object_exists(bucketName, key):
    try:
        s3.head_object(Bucket=bucketName, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def upload_file(bucketName, path):
    key = path.relative_to(config.DATA_DIR).as_posix()

    if object_exists(bucketName, key):
        print(f"Skipping existing object: s3://{bucketName}/{key}")
        return

    print(f"Uploading {path} -> s3://{bucketName}/{key}")
    s3.upload_file(str(path), bucketName, key)


def backup(bucketName:str=None):
    if bucketName is None:
        bucketName = os.getenv(S3_BUCKET_NAME_KEY)
    print(f"Uploading to bucket: {bucketName}")

    for path in config.DATA_DIR.rglob("*"):
        if path.is_file():
            upload_file(bucketName, path)

    print("Backup complete.")


if __name__ == "__main__":
    load_dotenv()
    backup()