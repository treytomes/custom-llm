"""
launch_dpo_remote.py

Launch a DPO fine-tuning job on AWS SageMaker.

Data Flow
---------

Local DPO pairs
      │
      ▼
Upload to S3: s3://<bucket>/dpo_data/pairs.jsonl
      │
      ▼
SageMaker downloads to:
/opt/ml/input/data/dpo
      │
      ▼
fine_tune.py
      │
      ▼
Checkpoint saved to:
/opt/ml/checkpoints/dpo_latest.pt
      │
      ▼
Synchronized to:
s3://<bucket>/checkpoints/dpo/

Usage
-----

1. Upload your pairs file:
   aws s3 cp ./dpo_data/pairs.jsonl s3://<bucket>/dpo_data/pairs.jsonl

2. Run this script:
   python launch_dpo_remote.py
"""

import boto3
from pathlib import Path
import sagemaker
from sagemaker.pytorch import PyTorch


# ── Session ────────────────────────────────────────────────────────────────────
session = sagemaker.Session()
role    = sagemaker.get_execution_role()

bucket  = "bitnet-training-456088019014-us-east-1-an"


# ── Upload pairs to S3 ─────────────────────────────────────────────────────────
pairs_local = Path("./dpo_data/pairs.jsonl")
pairs_s3_key = "dpo_data/pairs.jsonl"

if pairs_local.exists():
    print(f"Uploading {pairs_local} to s3://{bucket}/{pairs_s3_key}")
    s3 = boto3.client("s3")
    s3.upload_file(str(pairs_local), bucket, pairs_s3_key)
    print("Upload complete")
else:
    raise FileNotFoundError(
        f"Pairs file not found: {pairs_local}\n"
        f"Run dpo.py first to generate it."
    )


# ── Upload base checkpoint to S3 ───────────────────────────────────────────────
# The DPO job needs the Phase 1 checkpoint to fine-tune from.
# If it is already in S3 from training, you can skip this block.
checkpoint_local = Path("./checkpoints/latest.pt")
checkpoint_s3_key = "checkpoints/latest.pt"

if checkpoint_local.exists():
    print(f"Uploading {checkpoint_local} to s3://{bucket}/{checkpoint_s3_key}")
    s3.upload_file(str(checkpoint_local), bucket, checkpoint_s3_key)
    print("Checkpoint upload complete")


# ── Estimator ──────────────────────────────────────────────────────────────────
estimator = PyTorch(
    entry_point="fine_tune.py",
    source_dir=".",

    role=role,
    instance_count=1,
    instance_type="ml.g5.2xlarge",

    framework_version="2.1",
    py_version="py310",

    checkpoint_s3_uri=f"s3://{bucket}/checkpoints/dpo/",
    checkpoint_local_path="/opt/ml/checkpoints",

    hyperparameters={
        "pairs":      "/opt/ml/input/data/dpo/pairs.jsonl",
        "checkpoint": "/opt/ml/input/data/checkpoint/latest.pt",
        "output":     "/opt/ml/checkpoints",
        "steps":      50,
        "lr":         1e-7,
        "beta":       0.05,
        "log-every":  10,
    }
)


# ── Launch ─────────────────────────────────────────────────────────────────────
estimator.fit({
    "dpo":        f"s3://{bucket}/dpo_data/",
    "checkpoint": f"s3://{bucket}/checkpoints/",
})