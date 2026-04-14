"""
launch_training_remote.py

Launch a training job on AWS SageMaker using the PyTorch training container.

This script packages the current repository, uploads it to SageMaker, and
executes the training pipeline on a managed GPU instance.

The SageMaker container runs:

    launch_training_local.py

inside the training environment. That script handles:

    • tokenizing the training corpus
    • launching the training loop
    • saving checkpoints
    • uploading checkpoints to S3

Data Flow
---------

Local corpus
      │
      ▼
sync_corpus_to_s3.sh
      │
      ▼
S3: s3://<bucket>/corpus/
      │
      ▼
SageMaker downloads to:
/opt/ml/input/data/train
      │
      ▼
launch_training_local.py
      │
      ▼
training/train.py
      │
      ▼
Checkpoints saved to:
/opt/ml/checkpoints
      │
      ▼
Automatically synchronized to:
s3://<bucket>/checkpoints/

Usage
-----

python launch_training_remote.py

Requirements
------------

• AWS credentials configured
• IAM role with SageMaker + S3 permissions
• Corpus uploaded to S3

"""

import argparse
import datetime
import os
import sagemaker
import sys
from dotenv import load_dotenv
# from sagemaker.pytorch import PyTorch
from sagemaker.pytorch.model import PyTorchModel

import config

# -------------------------------------------------------------
# Constants
# -------------------------------------------------------------

INSTANCE_TYPE = "ml.g5.2xlarge"

# Alternative CPU-only instance for debugging
# instance_type="ml.r5.xlarge",

INSTANCE_COUNT = 1

# Approximate on-demand pricing.
# Reference: https://calculator.aws/#/createCalculator/SageMaker
INSTANCE_PRICE_PER_HOUR = {
    "ml.g5.2xlarge": 1.51,
    "ml.r5.xlarge": 0.30,
}

TARGET_COST_USD = 10.0


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def max_runtime_seconds(target_cost, instance_type, instance_count):
    price = INSTANCE_PRICE_PER_HOUR[instance_type]
    hours = target_cost / (price * instance_count)
    return int(hours * 3600)


def build_training_job_name(project, environment):
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"{project}-{environment}-{timestamp}"


# -------------------------------------------------------------
# Load environment
# -------------------------------------------------------------

load_dotenv()

PROJECT = os.getenv("PROJECT")
OWNER = os.getenv("OWNER")
ENVIRONMENT = os.getenv("ENVIRONMENT")

ROLE_ARN = os.getenv("SAGEMAKER_EXECUTION_ROLE_ARN")
BUCKET = os.getenv("S3_BUCKET_NAME")

# Establish a connection to the SageMaker service.
session = sagemaker.Session()

# -------------------------------------------------------------
# S3 paths
# -------------------------------------------------------------

training_data = f"s3://{BUCKET}/corpus/"
checkpoint_s3 = f"s3://{BUCKET}/checkpoints/"

# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--estimate", action="store_true",
                    help="Estimate runtime instead of launching job")
args = parser.parse_args()

# -------------------------------------------------------------
# Runtime budget
# -------------------------------------------------------------

max_runtime = max_runtime_seconds(
    TARGET_COST_USD,
    INSTANCE_TYPE,
    INSTANCE_COUNT,
)

hours = max_runtime / 3600

if args.estimate:
    print("")
    print("SageMaker Training Cost Estimate")
    print("------------------------------------------------")
    print(f"Instance type        : {INSTANCE_TYPE}")
    print(f"Instance count       : {INSTANCE_COUNT}")
    print(f"Instance price/hr    : ${INSTANCE_PRICE_PER_HOUR[INSTANCE_TYPE]:.2f}")
    print(f"Target cost          : ${TARGET_COST_USD:.2f}")
    print(f"Max runtime          : {max_runtime} seconds ({hours:.2f} hours)")

    # Optional step estimate (requires throughput guess)
    SECONDS_PER_LOG = 18400 # from the local CPU run
    ASSUMED_STEPS_PER_SEC = config.LOG_INTERVAL / SECONDS_PER_LOG 

    est_steps = int(max_runtime * ASSUMED_STEPS_PER_SEC)

    print("")
    print("Approximate Training Capacity")
    print("------------------------------------------------")
    print(f"Assumed steps/sec    : {ASSUMED_STEPS_PER_SEC}")
    print(f"Estimated steps      : {est_steps:,}")

    sys.exit(0)
    
print(f"Target cost: ${TARGET_COST_USD}")
print(f"Max runtime: {max_runtime} seconds")

# -------------------------------------------------------------
# Estimator
# -------------------------------------------------------------

estimator = PyTorch(
    entry_point="main.py",
    source_dir=".",
    role=ROLE_ARN,
    instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    framework_version="2.1",
    py_version="py310",
    checkpoint_s3_uri=checkpoint_s3,
    checkpoint_local_path="/opt/ml/checkpoints",
    max_run=max_runtime,
    hyperparameters={
        "data_file": "/opt/ml/input/data/train/corpus.pt",
        "output_dir": "/opt/ml/checkpoints",
        "s3_bucket": BUCKET,
    },
    tags=[
        {"Key": "project", "Value": PROJECT},
        {"Key": "owner", "Value": OWNER},
        {"Key": "environment", "Value": ENVIRONMENT},
    ],
)

# -------------------------------------------------------------
# Launch training
# -------------------------------------------------------------

job_name = build_training_job_name(PROJECT, ENVIRONMENT)

print(f"Launching training job: {job_name}")

estimator.fit(
    {"train": training_data},
    job_name=job_name
)