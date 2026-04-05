# launch_transform_remote.py
"""
launch_transform_remote.py — Launch corpus transformation job on SageMaker

Runs transform_corpus.py on a SageMaker GPU instance. The job reads
chapter files from S3, transforms each one into a Trey/Scout conversation
using Mistral-7B-Instruct downloaded from HuggingFace, and writes results
back to S3.

Data Flow
---------

S3: chapters/          ← split chapter files (from split_chapters.py --upload)
S3: voice/             ← scout_voice.txt
        │
        ▼
SageMaker transform job (ml.g5.2xlarge, 1x A10G GPU)
        │
        ▼
S3: corpus_transformed/ ← one transformed .txt file per chapter

Usage
-----

1. Split novels and upload chapters:
   python split_chapters.py --upload

2. Upload scout_voice.txt:
   aws s3 cp ./corpus/scout_voice.txt s3://<bucket>/voice/scout_voice.txt

3. Launch:
   python launch_transform_remote.py

4. Download and review output:
   aws s3 sync s3://<bucket>/corpus_transformed/ ./corpus_transformed/
"""

import boto3
from pathlib import Path

import sagemaker
from sagemaker.pytorch import PyTorch


# ── Configuration ──────────────────────────────────────────────────────────────

BUCKET   = "bitnet-training-456088019014-us-east-1-an"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"


# ── Session ────────────────────────────────────────────────────────────────────

session = sagemaker.Session()
role    = sagemaker.get_execution_role()
s3      = boto3.client("s3")


# ── Preflight checks ───────────────────────────────────────────────────────────

def s3_prefix_exists(bucket: str, prefix: str) -> bool:
    """Return True if at least one object exists under the given S3 prefix."""
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return response.get("KeyCount", 0) > 0


def preflight_checks() -> bool:
    """Verify required S3 inputs exist before launching the job."""
    print("Preflight checks\n")

    checks = [
        (
            "chapters/",
            "Chapter files missing.\n"
            "  Run: python split_chapters.py --upload"
        ),
        (
            "voice/scout_voice.txt",
            "scout_voice.txt missing.\n"
            "  Run: aws s3 cp ./corpus/scout_voice.txt "
            f"s3://{BUCKET}/voice/scout_voice.txt"
        ),
    ]

    all_ok = True
    for prefix, fix_message in checks:
        exists = s3_prefix_exists(BUCKET, prefix)
        mark   = "✓" if exists else "✗"
        print(f"  {mark}  s3://{BUCKET}/{prefix}")
        if not exists:
            print(f"     {fix_message}")
            all_ok = False

    print()
    return all_ok


# ── Upload scout_voice.txt if present locally ──────────────────────────────────

def upload_voice_file():
    """Upload scout_voice.txt to S3 if it exists locally and isn't there yet."""
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

def build_estimator() -> PyTorch:
    return PyTorch(
        entry_point="transform_corpus.py",
        source_dir=".",

        role=role,
        instance_count=1,

        # ml.g5.2xlarge: 1x A10G (24GB VRAM)
        # Sufficient for Mistral-7B in fp16 (~14GB)
        instance_type="ml.g5.2xlarge",

        framework_version="2.1",
        py_version="py310",

        hyperparameters={
            "model_id":          MODEL_ID,
            "output_s3":         f"s3://{BUCKET}/corpus_transformed/",
            "max_new_tokens":    800,
            "temperature":       0.7,
            "min_chapter_words": 100,
        },
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("═" * 60)
    print("  Corpus Transformation Job")
    print("  Novels → Trey/Scout conversations")
    print(f"  Model: {MODEL_ID}")
    print("═" * 60)
    print()

    # Upload voice file if available locally
    upload_voice_file()

    # Verify required inputs exist in S3
    if not preflight_checks():
        print("Preflight checks failed. Resolve the issues above and retry.")
        return

    estimator = build_estimator()

    input_channels = {
        "chapters": f"s3://{BUCKET}/chapters/",
        "voice":    f"s3://{BUCKET}/voice/",
    }

    print("Launching SageMaker job...\n")
    estimator.fit(input_channels)

    print("\n" + "═" * 60)
    print("  Job complete.")
    print(f"\n  Transformed chapters:")
    print(f"    s3://{BUCKET}/corpus_transformed/")
    print(f"\n  Download and review:")
    print(f"    aws s3 sync s3://{BUCKET}/corpus_transformed/ ./corpus_transformed/")
    print(f"\n  If quality is good, add to corpus and train:")
    print(f"    cp ./corpus_transformed/*.txt ./corpus/")
    print(f"    python launch_training_remote.py")
    print("═" * 60)


if __name__ == "__main__":
    main()
