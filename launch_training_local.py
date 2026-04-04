"""
launch_training_local.py

Launch the local training pipeline for the custom LLM project.

This script prepares the training environment and then calls the core
training routine defined in `training/train.py`.

It performs the following steps:

    1. Determine local paths for the corpus, tokenized data, and checkpoints
    2. Load the tokenizer used for training
    3. Tokenize the corpus if the tokenized dataset does not already exist
    4. Construct training arguments
    5. Start the training loop

The same script is used both for:

    • Local development training
    • SageMaker remote training

When executed inside a SageMaker container, the `--data_dir`,
`--output_dir`, and `--s3_bucket` arguments are automatically passed
by the training job configuration.

Usage
-----

Local training:

    python launch_training_local.py

SageMaker training (arguments passed automatically):

    python launch_training_local.py \
        --data_dir /opt/ml/input/data/train \
        --output_dir /opt/ml/checkpoints \
        --s3_bucket <bucket-name>

Directory Layout
----------------

project/
    corpus/            # Raw text corpus
    data/              # Tokenized dataset
    checkpoints/       # Model checkpoints
    training/          # Training modules
    launch_training_local.py

"""

from pathlib import Path
import argparse

from training.data import tokenize_corpus
from training.train import main as train_main
from transformers import AutoTokenizer


# -------------------------------------------------------------
# Project Paths
# -------------------------------------------------------------

# Root directory of the repository
ROOT = Path(__file__).parent

# Default location of raw training text files
DEFAULT_CORPUS = ROOT / "corpus"

# Directory used to store tokenized datasets
DEFAULT_DATA = ROOT / "data"

# Directory used for model checkpoints
DEFAULT_CHECKPOINTS = ROOT / "checkpoints"

TRAINING_STEPS = 150000


# -------------------------------------------------------------
# Main Training Launcher
# -------------------------------------------------------------

def main():

    # ---------------------------------------------------------
    # Command Line Arguments
    # ---------------------------------------------------------

    parser = argparse.ArgumentParser()

    # Location of the training corpus.
    # On SageMaker this is automatically set to:
    #   /opt/ml/input/data/train
    parser.add_argument("--data_dir", default=None)

    # Directory where checkpoints should be written.
    # On SageMaker this is usually:
    #   /opt/ml/checkpoints
    parser.add_argument("--output_dir", default=None)

    # Optional S3 bucket used for uploading checkpoints.
    # This is primarily used in SageMaker training jobs.
    parser.add_argument("--s3_bucket", default=None)

    args = parser.parse_args()


    # ---------------------------------------------------------
    # Resolve Paths
    # ---------------------------------------------------------

    # Training corpus directory
    corpus_dir = Path(args.data_dir) if args.data_dir else DEFAULT_CORPUS

    # Checkpoint directory
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_CHECKPOINTS

    # Tokenized dataset directory
    data_dir = DEFAULT_DATA

    # Ensure required directories exist
    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)


    # ---------------------------------------------------------
    # Tokenized Dataset
    # ---------------------------------------------------------

    # Path to the serialized token tensor used for training
    token_file = data_dir / "corpus.pt"


    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------

    # HuggingFace tokenizer used for encoding text
    tokenizer_name = "mistralai/Mistral-7B-v0.1"

    print(f"Loading tokenizer: {tokenizer_name}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load the tokenizer from the pretrained model repository,
    # which provides tokenization rules and vocabulary needed
    # to convert text into token IDs for model training [1].


    # ---------------------------------------------------------
    # Tokenize Corpus (if necessary)
    # ---------------------------------------------------------

    if not token_file.exists():
        print("Tokenizing corpus...")

        tokenize_corpus(
            input_dir=corpus_dir,
            tokenizer=tokenizer,
            output_path=token_file,
        )

    # The tokenized dataset is stored as a serialized tensor
    # so it can be loaded quickly during training.


    # ---------------------------------------------------------
    # Launch Training
    # ---------------------------------------------------------

    print("Starting training...")

    # Build the argument namespace expected by training/train.py
    train_args = argparse.Namespace(

        # Corpus directory containing raw text files
        corpus_dir=str(corpus_dir),

        # Tokenized dataset file
        data_path=str(token_file),

        # Checkpoint output directory
        out_dir=str(output_dir),

        # Tokenizer used for training
        tokenizer=tokenizer_name,

        # Model / training parameters
        seq_len=512,          # Context window
        batch_size=8,         # Training batch size
        steps=TRAINING_STEPS, # Total training steps

        # Transformer architecture
        dim=512,              # Hidden dimension
        layers=12,            # Number of transformer blocks
        heads=8,              # Attention heads

        # Optimizer settings
        lr=3e-4,

        # Logging / checkpointing
        log_interval=50,
        save_interval=1000,

        # Optional S3 bucket for checkpoint uploads
        s3_bucket=args.s3_bucket,
    )

    # Start the training loop
    train_main(train_args)


# -------------------------------------------------------------
# Script Entry Point
# -------------------------------------------------------------

if __name__ == "__main__":
    main()