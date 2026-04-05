# launch_transform_remote.py
"""
launch_transform_remote.py — Launch corpus transformation job on SageMaker

Runs transform_corpus.py on a SageMaker GPU instance. The job reads
chapter files from S3, transforms each one into Scout's first-person
reflective voice using a local Mistral model, and writes the results
back to S3.

HuggingFace model caching:
    The Mistral model is cached to S3 after the first download so
    subsequent jobs skip the HuggingFace download entirely. This saves
    several minutes per job and avoids rate limiting.

Data Flow
---------

S3: chapters/          ← split chapter files (from split_chapters.py)
S3: voice/             ← scout_voice.txt
S3: model_cache/       ← cached HuggingFace model weights (auto-populated)
        │
        ▼
SageMaker transform job (ml.g5.2xlarge)
        │
        ▼
S3: corpus_transformed/ ← one transformed .txt file per chapter

Usage
-----

1. Split and upload chapters:
   python split_chapters.py --upload

2. Upload scout_voice.txt:
   aws s3 cp ./corpus/scout_voice.txt s3://<bucket>/voice/scout_voice.txt

3. Launch transformation:
   python launch_transform_remote.py

4. After job completes, download transformed corpus:
   aws s3 sync s3://<bucket>/corpus_transformed/ ./corpus_transformed/
"""

import boto3
import os
from pathlib import Path

import sagemaker
from sagemaker.pytorch import PyTorch


# ── Configuration ──────────────────────────────────────────────────────────────

BUCKET        = "bitnet-training-456088019014-us-east-1-an"
MODEL_ID      = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_CACHE_PREFIX = "model_cache/mistral-7b-instruct-v0.3/"


# ── Session ────────────────────────────────────────────────────────────────────

session = sagemaker.Session()
role    = sagemaker.get_execution_role()
s3      = boto3.client("s3")


# ── Preflight checks ───────────────────────────────────────────────────────────

def check_s3_prefix_exists(bucket: str, prefix: str) -> bool:
    """Return True if at least one object exists under the given S3 prefix."""
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return response.get("KeyCount", 0) > 0


def preflight_checks():
    """Verify required S3 inputs exist before launching the job."""
    print("Running preflight checks...\n")

    checks = [
        ("chapters/",       "Chapter files (run split_chapters.py --upload)"),
        ("voice/",          "scout_voice.txt (run: aws s3 cp ./corpus/scout_voice.txt "
                            f"s3://{BUCKET}/voice/scout_voice.txt)"),
    ]

    all_ok = True

    for prefix, description in checks:
        exists = check_s3_prefix_exists(BUCKET, prefix)
        status = "✓" if exists else "✗"
        print(f"  {status}  s3://{BUCKET}/{prefix}")
        if not exists:
            print(f"     Missing: {description}")
            all_ok = False

    # Model cache is optional — job will populate it on first run
    model_cached = check_s3_prefix_exists(BUCKET, MODEL_CACHE_PREFIX)
    cache_status = "✓ (cached)" if model_cached else "○ (will download and cache)"
    print(f"  {cache_status}  s3://{BUCKET}/{MODEL_CACHE_PREFIX}")

    print()
    return all_ok


# ── Upload scout_voice.txt if present locally ──────────────────────────────────

def upload_voice_file():
    """Upload scout_voice.txt to S3 if it exists locally."""
    local_path = Path("./corpus/scout_voice.txt")

    if not local_path.exists():
        return

    s3_key = "voice/scout_voice.txt"
    print(f"Uploading {local_path} → s3://{BUCKET}/{s3_key}")
    s3.upload_file(str(local_path), BUCKET, s3_key)
    print("  Done\n")


# ── Estimator ──────────────────────────────────────────────────────────────────

def build_estimator():
    return PyTorch(
        entry_point="transform_corpus.py",
        source_dir=".",

        role=role,
        instance_count=1,

        # g5.2xlarge: 1x A10G (24GB VRAM), sufficient for Mistral 7B in fp16
        instance_type="ml.g5.2xlarge",

        framework_version="2.1",
        py_version="py310",

        # No checkpoint sync needed — outputs go directly to S3
        # via the training job output mechanism

        hyperparameters={
            # HuggingFace model to use for transformation
            "model_id":          MODEL_ID,

            # S3 path for model cache — job checks here before downloading
            # from HuggingFace. Populated automatically after first run.
            "model_cache_s3":    f"s3://{BUCKET}/{MODEL_CACHE_PREFIX}",

            # S3 output location for transformed chapters
            "output_s3":         f"s3://{BUCKET}/corpus_transformed/",

            # Transformation parameters
            "max_new_tokens":    800,
            "temperature":       0.7,

            # Minimum chapter word count — shorter chapters are skipped
            "min_chapter_words": 100,
        },

        # Environment variables available inside the container
        environment={
            # Tells HuggingFace to use the SageMaker input channel
            # as the model cache directory when available
            "TRANSFORMERS_CACHE": "/opt/ml/input/data/model_cache",
            "HF_HOME":            "/opt/ml/input/data/model_cache",
        },
    )


# ── Data channels ──────────────────────────────────────────────────────────────

def build_input_channels(model_cached: bool) -> dict:
    """
    Build SageMaker input channel configuration.

    If the model is already cached in S3, it is provided as an input
    channel so the container can load it without downloading from
    HuggingFace. On the first run the model_cache channel is omitted
    and the job downloads and then caches the model.
    """
    channels = {
        "chapters": f"s3://{BUCKET}/chapters/",
        "voice":    f"s3://{BUCKET}/voice/",
    }

    if model_cached:
        print("Model cache found in S3 — mounting as input channel")
        channels["model_cache"] = f"s3://{BUCKET}/{MODEL_CACHE_PREFIX}"
    else:
        print("No model cache found — job will download from HuggingFace")
        print(f"Model will be cached to s3://{BUCKET}/{MODEL_CACHE_PREFIX} after first run")

    return channels


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("═" * 60)
    print("  Corpus Transformation Job")
    print("  Novels → Scout's first-person reflective voice")
    print("═" * 60)
    print()

    # Upload scout_voice.txt if available locally
    upload_voice_file()

    # Verify inputs
    if not preflight_checks():
        print("Preflight checks failed. Please resolve the issues above.")
        return

    model_cached = check_s3_prefix_exists(BUCKET, MODEL_CACHE_PREFIX)

    estimator = build_estimator()
    channels  = build_input_channels(model_cached)

    print("Launching SageMaker transformation job...\n")

    estimator.fit(channels)

    print("\n" + "═" * 60)
    print("  Transformation job complete.")
    print(f"  Transformed chapters: s3://{BUCKET}/corpus_transformed/")
    print()
    print("  Next steps:")
    print("  1. Review sample outputs:")
    print(f"     aws s3 sync s3://{BUCKET}/corpus_transformed/ ./corpus_transformed/")
    print("  2. If quality is good, add to training corpus:")
    print("     cp ./corpus_transformed/*.txt ./corpus/")
    print("  3. Launch fresh training run:")
    print("     python launch_training_remote.py")
    print("═" * 60)


if __name__ == "__main__":
    main()
