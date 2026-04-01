"""
train.py — Train a small GPT-style language model

Features
- Automatic corpus tokenization when needed
- Uses data.py dataset + dataloader
- Simple GPT training loop
- Checkpointing
"""

import os
import time
from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import AutoTokenizer

from .config import DEFAULT_CONFIG, update_vocab_size
from .model import GPT
from .data import (
    tokenize_corpus,
    load_token_tensor,
    build_dataloader,
)


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def newest_file_mtime(directory):
    newest = 0
    for p in Path(directory).rglob("*"):
        if p.is_file():
            newest = max(newest, p.stat().st_mtime)
    return newest


def corpus_needs_tokenization(corpus_dir, token_file):
    corpus_dir = Path(corpus_dir)
    token_file = Path(token_file)

    if not token_file.exists():
        return True

    corpus_time = newest_file_mtime(corpus_dir)
    token_time = token_file.stat().st_mtime

    return corpus_time > token_time


# -------------------------------------------------------------
# Training
# -------------------------------------------------------------

def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    corpus_dir = Path(args.corpus_dir)
    token_file = Path(args.data_path)

    token_file.parent.mkdir(parents=True, exist_ok=True)

    cfg = DEFAULT_CONFIG.copy()
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    cfg = update_vocab_size(cfg, tokenizer)

    # ---------------------------------------------------------
    # Tokenize if needed
    # ---------------------------------------------------------

    if corpus_needs_tokenization(corpus_dir, token_file):
        print("Corpus changed or tokens missing — tokenizing...")
        tokenize_corpus(
            input_dir=corpus_dir,
            tokenizer=tokenizer,
            output_path=token_file,
        )
    else:
        print("Tokenized corpus is up to date.")

    # ---------------------------------------------------------
    # Load tokens
    # ---------------------------------------------------------

    print(f"Loading corpus: {token_file}")
    tokens = load_token_tensor(token_file)

    print(f"Corpus size: {len(tokens):,} tokens ({len(tokens)/1e6:.2f}M)")

    loader = build_dataloader(
        tokens,
        block_size=cfg["block_size"],
        batch_size=cfg["batch_size"],
        shuffle=True,
    )

    data_iter = iter(loader)

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = GPT(
        vocab_size=cfg["vocab_size"],
        dim=cfg["dim"],
        layers=cfg["layers"],
        heads=cfg["heads"],
        max_seq=cfg["block_size"],
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=0.1)
    scheduler = CosineAnnealingLR(optimizer, cfg["max_steps"])

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------

    step = 0
    start_time = time.time()

    while step < args.steps:

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)

        logits = model(x)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"step {step:6d} | "
                f"loss {loss.item():.4f} | "
                f"{elapsed:.1f}s"
            )

        if step % args.save_interval == 0 and step > 0:
            ckpt = Path(args.out_dir) / f"model_{step}.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"Saved checkpoint: {ckpt}")

        step += 1


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--corpus_dir", default="../corpus")
    parser.add_argument("--data_path", default="../data/corpus.pt")
    parser.add_argument("--out_dir", default="../checkpoints")

    parser.add_argument("--tokenizer", default="mistralai/Mistral-7B-v0.1")

    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=1000)

    args = parser.parse_args()

    main(args)