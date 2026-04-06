# launch_transform_remote.py
"""
launch_transform_remote.py — Launch corpus transformation job on SageMaker

Orchestrates transform_corpus.py on a SageMaker CPU instance. The job reads
chapter files from S3, transforms each into Trey/Scout conversations using
Mistral Large 3 via Azure, and writes results back to S3.

NOTE: No GPU is needed. Inference is handled by Azure. SageMaker here
provides overnight orchestration, logging, and S3 integration only.

Data Flow
---------

S3: chapters/               ← split chapter files (from split_chapters.py --upload)
S3: voice/scout_voice.txt   ← Scout voice reference
        │
        ▼
SageMaker job (ml.m5.xlarge — CPU only, Azure handles inference)
        │
        ▼
S3: corpus_transformed/     ← one .txt file per conversation per chapter

Usage
-----

1. Split novels and upload chapters:
   python split_chapters.py --upload

2. Upload scout_voice.txt:
   aws s3 cp ./corpus/scout_voice.txt s3://<bucket>/voice/scout_voice.txt

3. Set Azure credentials in AWS Secrets Manager (one-time setup):
   aws secretsmanager create-secret --name scout/azure_mistral \\
       --secret-string '{"endpoint":"...","key":"..."}'

4. Launch:
   python launch_transform_remote.py

5. Monitor:
   Open SageMaker console → Training jobs → view logs in CloudWatch

6. Download and review output:
   aws s3 sync s3://<bucket>/corpus_transformed/ ./corpus_transformed/
"""

import boto3
import json
import sys
from pathlib import Path

import sagemaker
from sagemaker.pytorch import PyTorch


# ── Configuration ──────────────────────────────────────────────────────────────

BUCKET      = "bitnet-training-456088019014-us-east-1-an"
SECRET_NAME = "scout/azure_mistral"   # Stores endpoint + API key


# ── Session ────────────────────────────────────────────────────────────────────

session = sagemaker.Session()
role    = sagemaker.get_execution_role()
s3      = boto3.client("s3")
secrets = boto3.client("secretsmanager")


# ── Helpers ────────────────────────────────────────────────────────────────────

def s3_prefix_exists(bucket: str, prefix: str) -> bool:
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return response.get("KeyCount", 0) > 0


def count_s3_objects(bucket: str, prefix: str) -> int:
    paginator = s3.get_paginator("list_objects_v2")
    total = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        total += page.get("KeyCount", 0)
    return total


def get_azure_credentials() -> tuple[str, str]:
    """Retrieve Azure endpoint and key from Secrets Manager."""
    try:
        response = secrets.get_secret_value(SecretId=SECRET_NAME)
        secret   = json.loads(response["SecretString"])
        return secret["endpoint"], secret["key"]
    except Exception as e:
        print(f"  ✗  Could not retrieve Azure credentials from Secrets Manager.")
        print(f"     Secret name: {SECRET_NAME}")
        print(f"     Error: {e}")
        print(f"\n     Run this once to store your credentials:")
        print(f"       aws secretsmanager create-secret --name {SECRET_NAME} \\")
        print(f"           --secret-string '{{\"endpoint\":\"...\",\"key\":\"...\"}}'\n")
        return None, None


# ── Preflight checks ───────────────────────────────────────────────────────────

def preflight_checks(endpoint: str) -> bool:
    print("Preflight checks\n")

    all_ok = True

    # S3 inputs
    checks = [
        (
            "chapters/",
            "Chapter files missing.\n"
            "     Run: python split_chapters.py --upload"
        ),
        (
            "voice/scout_voice.txt",
            "scout_voice.txt missing.\n"
            f"     Run: aws s3 cp ./corpus/scout_voice.txt s3://{BUCKET}/voice/scout_voice.txt"
        ),
    ]

    for prefix, fix_message in checks:
        exists = s3_prefix_exists(BUCKET, prefix)
        mark   = "✓" if exists else "✗"
        print(f"  {mark}  s3://{BUCKET}/{prefix}")
        if not exists:
            print(f"     {fix_message}")
            all_ok = False

    # Azure credentials
    cred_ok = endpoint is not None
    mark    = "✓" if cred_ok else "✗"
    print(f"  {mark}  Azure credentials ({SECRET_NAME})")
    if not cred_ok:
        all_ok = False

    # Progress report
    if all_ok:
        chapter_count     = count_s3_objects(BUCKET, "chapters/")
        transformed_count = count_s3_objects(BUCKET, "corpus_transformed/")
        print(f"\n  Chapters available:    {chapter_count}")
        print(f"  Already transformed:   {transformed_count}")
        if transformed_count > 0:
            print(f"  Resuming from chapter: {transformed_count + 1} (existing files will be skipped)")

    print()
    return all_ok


# ── Upload scout_voice.txt if present locally ──────────────────────────────────

def upload_voice_file():
    local_path = Path("./corpus/scout_voice.txt")
    s3_key     = "voice/scout_voice.txt"

    if not local_path.exists():
        return

    if s3_prefix_exists(BUCKET, s3_key):
        print(f"scout_voice.txt already in S3 — skipping upload\n")
        return

    print(f"Uploading {local_path} → s3://{BUCKET}/{s3_key}")
    s3.upload_file(str(local_path), BUCKET, s3_key)
    print("  Done\n")


# ── Estimator ──────────────────────────────────────────────────────────────────

def build_estimator(endpoint: str, api_key: str) -> PyTorch:
    """
    CPU instance only — inference is handled by Azure, not SageMaker.
    ml.m5.xlarge is sufficient and much cheaper than a GPU instance.
    """
    return PyTorch(
        entry_point="transform_corpus.py",
        source_dir=".",

        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",       # CPU only — no GPU needed

        framework_version="2.1",
        py_version="py310",

        environment={
            "AZURE_MISTRAL_ENDPOINT": endpoint,
            "AZURE_MISTRAL_KEY":      api_key,
        },

        hyperparameters={
            "chapters_dir": "/opt/ml/input/data/chapters",
            "output_dir":   "/opt/ml/output/data",
            "voice_file":   "/opt/ml/input/data/voice/scout_voice.txt",
            "upload_s3":    f"s3://{BUCKET}/corpus_transformed/",
            "temperature":  0.7,
        },
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("═" * 60)
    print("  Scout Corpus Transformation")
    print("  Novels → Trey/Scout conversations")
    print("  Inference: Mistral Large 3 via Azure")
    print("═" * 60)
    print()

    # Retrieve Azure credentials first — needed for preflight
    endpoint, api_key = get_azure_credentials()

    upload_voice_file()

    if not preflight_checks(endpoint):
        print("Preflight checks failed. Resolve the issues above and retry.")
        sys.exit(1)

    estimator = build_estimator(endpoint, api_key)

    input_channels = {
        "chapters": f"s3://{BUCKET}/chapters/",
        "voice":    f"s3://{BUCKET}/voice/",
    }

    print("Launching SageMaker job...\n")
    estimator.fit(input_channels, wait=False)    # Non-blocking — returns immediately

    job_name = estimator.latest_training_job.name

    print("═" * 60)
    print(f"  Job launched: {job_name}")
    print(f"\n  Monitor progress:")
    print(f"    AWS Console → SageMaker → Training jobs → {job_name}")
    print(f"    CloudWatch  → Log groups → /aws/sagemaker/TrainingJobs")
    print(f"\n  When complete, download results:")
    print(f"    aws s3 sync s3://{BUCKET}/corpus_transformed/ ./corpus_transformed/")
    print(f"\n  If quality is good, launch training:")
    print(f"    python launch_training_remote.py")
    print("═" * 60)


if __name__ == "__main__":
    main()