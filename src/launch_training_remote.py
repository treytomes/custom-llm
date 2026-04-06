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

import sagemaker
from sagemaker.pytorch import PyTorch


# -------------------------------------------------------------
# Initialize SageMaker session
# -------------------------------------------------------------

# Establish a connection to the SageMaker service.
session = sagemaker.Session()

# Retrieve the IAM execution role attached to the environment.
# This role must allow:
#   - reading training data from S3
#   - writing checkpoints to S3
#   - launching SageMaker training jobs
role = sagemaker.get_execution_role()


# -------------------------------------------------------------
# S3 Configuration
# -------------------------------------------------------------

# Primary S3 bucket used for training resources
bucket = "bitnet-training-456088019014-us-east-1-an"

# Location of the raw training corpus
# This directory should contain .txt files.
training_data = f"s3://{bucket}/corpus/"

# Location where model checkpoints are stored
checkpoint_s3 = f"s3://{bucket}/checkpoints/"


# -------------------------------------------------------------
# Configure the SageMaker PyTorch Estimator
# -------------------------------------------------------------

estimator = PyTorch(

    # Script executed inside the SageMaker training container.
    # This launches the local training pipeline so both
    # local and remote training share the same code path.
    entry_point="launch_training_local.py",

    # Directory containing the training source code.
    # The entire project directory will be uploaded to SageMaker.
    source_dir=".",

    # IAM role used by the training container.
    role=role,

    # Number of instances used for distributed training.
    # Set to >1 for multi-node training.
    instance_count=1,

    # Type of compute instance used for training.
    # ml.g5.2xlarge includes:
    #   • 1 NVIDIA A10G GPU
    #   • 8 vCPUs
    #   • 32 GB RAM
    #
    # This is sufficient for training ~50M parameter models.
    instance_type="ml.g5.2xlarge",

    # Alternative CPU-only instance for debugging
    # instance_type="ml.r5.xlarge",

    # PyTorch framework version used by the SageMaker container.
    framework_version="2.1",

    # Python runtime version used inside the container.
    py_version="py310",

    # ---------------------------------------------------------
    # Checkpoint configuration
    # ---------------------------------------------------------

    # S3 location where checkpoints will be stored.
    checkpoint_s3_uri=checkpoint_s3,

    # Local directory inside the SageMaker container where
    # checkpoints are written during training.
    checkpoint_local_path="/opt/ml/checkpoints",

    # ---------------------------------------------------------
    # Hyperparameters passed to the training script
    # ---------------------------------------------------------

    hyperparameters={

        # Location where SageMaker downloads the training data.
        # The dataset from `training_data` will appear here.
        "data_dir": "/opt/ml/input/data/train",

        # Directory where the training script should write
        # model checkpoints.
        "output_dir": "/opt/ml/checkpoints",

        # S3 bucket used by the training script when uploading
        # checkpoints such as "latest.pt".
        "s3_bucket": bucket,
    }
)


# -------------------------------------------------------------
# Launch training job
# -------------------------------------------------------------

# Start the SageMaker training job.
# SageMaker will:
#   1. Upload the project source code
#   2. Provision the training instance
#   3. Download the dataset from S3
#   4. Execute launch_training_local.py
#   5. Stream logs to the console
estimator.fit({
    "train": training_data
})
