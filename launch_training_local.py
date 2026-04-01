"""
launch_training_local.py

Runs the local training pipeline without subprocesses.

Pipeline:

0. Generate sample corpus (optional)
1. Tokenize corpus
2. Train model

Expected project structure:

project/
    corpus/
    data/
    checkpoints/
    training/
        data.py
        train.py
"""

from pathlib import Path
import argparse

from training.data import make_sample_corpus, tokenize_corpus
from training.train import main as train_main
from transformers import AutoTokenizer

ROOT = Path(__file__).parent

CORPUS_DIR = ROOT / "corpus"
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
TOKEN_FILE = DATA_DIR / "corpus.pt"

def main():
    DATA_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    # # Step 0 — Generate sample corpus if none exists
    # if not any(CORPUS_DIR.glob("*.txt")):
    #     print("No corpus detected — generating sample corpus.")
    #     make_sample_corpus(CORPUS_DIR)

    # Step 1 — Load tokenizer
    tokenizer_name = "mistralai/Mistral-7B-v0.1"
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Step 2 — Tokenize corpus
    print("Tokenizing corpus...")
    tokenize_corpus(
        input_dir=CORPUS_DIR,
        tokenizer=tokenizer,
        output_path=TOKEN_FILE,
    )

    # Step 3 — Launch training
    print("Starting training...")
    args = argparse.Namespace(
        corpus_dir=str(CORPUS_DIR),
        data_path=str(TOKEN_FILE),
        out_dir=str(CHECKPOINT_DIR),
        tokenizer=tokenizer_name,

        # seq_len=512,
        # batch_size=8,
        # steps=1000,

        seq_len = 32,
        batch_size = 2,
        steps = 200,

        dim=512,
        layers=12,
        heads=8,
        lr=3e-4,
        log_interval=50,
        save_interval=100,
    )

    train_main(args)

if __name__ == "__main__":
    main()