# launch_training.py

import sagemaker
from sagemaker.pytorch import PyTorch


# SageMaker session
session = sagemaker.Session()
role = sagemaker.get_execution_role()

# S3 locations
bucket = "bitnet-training-456088019014-us-east-1-an"

training_data = f"s3://{bucket}/corpus/"
checkpoint_s3 = f"s3://{bucket}/checkpoints/"


estimator = PyTorch(
    entry_point="train.py",
    source_dir="./training",

    role=role,
    instance_count=1,
    # instance_type="ml.r5.xlarge",   # training instance
    instance_type="ml.g5.2xlarge",

    framework_version="2.1",
    py_version="py310",

    checkpoint_s3_uri="s3://bitnet-training-456088019014-us-east-1-an/checkpoints/",
    checkpoint_local_path="/opt/ml/checkpoints",

    hyperparameters={
        "data_dir": "/opt/ml/input/data/train",
        "output_dir": "/opt/ml/checkpoints",
    }
)

# Launch training
estimator.fit({
    "train": training_data
})
