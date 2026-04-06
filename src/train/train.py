"""
train.py — Train a small GPT-style language model (Scout)

Features
────────────────────────────────────────────────────
• Standard AutoTokenizer — no vocabulary patching required
• Linear warmup → cosine decay LR schedule
• Scheduler state saved and restored in checkpoints
• config.py values as authoritative defaults
• Gradient accumulation for effective large batch sizes
• Validation loss logged against held-out token file if present
• SageMaker integration with S3 checkpoint sync
"""

import boto3
import time
from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from transformers import AutoTokenizer

from .config import DEFAULT_CONFIG, update_vocab_size
from .model import GPT
from .data import tokenize_corpus, load_token_tensor, build_dataloader


# ─────────────────────────────────────────────────────────────
# SageMaker helpers
# ─────────────────────────────────────────────────────────────

def running_in_sagemaker() -> bool:
    return Path("/opt/ml").exists()


def detect_sagemaker_paths(args):
    if running_in_sagemaker():
        print("SageMaker environment detected")
        if args.corpus_dir is None:
            args.corpus_dir = "/opt/ml/input/data/train"
        if args.data_path is None:
            args.data_path = "/opt/ml/input/data/train/corpus.pt"
        if args.out_dir is None:
            args.out_dir = "/opt/ml/checkpoints"
    return args


def upload_checkpoint(s3, bucket, prefix, file_path):
    key = f"{prefix}/{Path(file_path).name}"
    print(f"  Uploading {file_path} → s3://{bucket}/{key}")
    s3.upload_file(str(file_path), bucket, key)


# ─────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────

def build_scheduler(
    optimizer,
    total_steps:  int,
    warmup_steps: int,
    min_lr:       float,
):
    """
    Linear warmup for warmup_steps, then cosine decay to min_lr.
    """
    warmup = LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps)
    )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=min_lr,
    )

    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


# ─────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────

def save_checkpoint(
    out_dir, step, model, optimizer, scheduler, cfg,
    s3=None, s3_bucket=None, s3_prefix="checkpoints",
):
    out_dir     = Path(out_dir)
    ckpt_path   = out_dir / f"model_{step}.pt"
    latest_path = out_dir / "latest.pt"

    checkpoint = {
        "step":      step,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config":    cfg,
    }

    torch.save(checkpoint, ckpt_path)
    torch.save(checkpoint, latest_path)
    print(f"  Saved checkpoint: {ckpt_path}")

    if s3 and s3_bucket:
        upload_checkpoint(s3, s3_bucket, s3_prefix, ckpt_path)
        upload_checkpoint(s3, s3_bucket, s3_prefix, latest_path)


def try_resume_checkpoint(model, optimizer, scheduler, checkpoint_dir, device):
    latest = Path(checkpoint_dir) / "latest.pt"

    if not latest.exists():
        print("No checkpoint found — starting fresh.")
        return 0

    print(f"Resuming from checkpoint: {latest}")
    ckpt = torch.load(latest, map_location=device)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        print("  Warning: no scheduler state in checkpoint — scheduler reset.")

    step = ckpt.get("step", 0)
    print(f"  Resumed from step {step}")
    return step


# ─────────────────────────────────────────────────────────────
# Corpus helpers
# ─────────────────────────────────────────────────────────────

def newest_file_mtime(directory):
    newest = 0
    for p in Path(directory).rglob("*"):
        if p.is_file():
            newest = max(newest, p.stat().st_mtime)
    return newest


def corpus_needs_tokenization(corpus_dir, token_file):
    token_file = Path(token_file)
    if not token_file.exists():
        return True
    return newest_file_mtime(corpus_dir) > token_file.stat().st_mtime


# ─────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def validation_loss(model, val_tokens, cfg, device, num_batches=20):
    """
    Estimate loss on held-out validation tokens.

    Place tokenized held-out conversations in data/corpus_val.pt
    to enable. Logs every log_interval * 10 steps during training.
    """
    model.eval()

    loader = build_dataloader(
        val_tokens,
        block_size=cfg["block_size"],
        batch_size=cfg["batch_size"],
        shuffle=False,
    )

    total_loss = 0.0
    count      = 0

    for batch in loader:
        if count >= num_batches:
            break

        x      = batch["input_ids"].to(device)
        y      = batch["labels"].to(device)
        logits = model(x)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )

        total_loss += loss.item()
        count      += 1

    model.train()
    return total_loss / max(count, 1)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main(args):
    args   = detect_sagemaker_paths(args)
    use_s3 = running_in_sagemaker() and args.s3_bucket is not None
    s3     = boto3.client("s3") if use_s3 else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Paths ────────────────────────────────────────────────
    corpus_dir     = Path(args.corpus_dir)
    token_file     = Path(args.data_path)
    val_token_file = token_file.parent / "corpus_val.pt"

    token_file.parent.mkdir(parents=True, exist_ok=True)

    # ── Config ───────────────────────────────────────────────
    cfg = DEFAULT_CONFIG.copy()

    # Launcher overrides take precedence over config defaults
    cfg["batch_size"]    = getattr(args, "batch_size",    cfg["batch_size"])
    cfg["block_size"]    = getattr(args, "seq_len",       cfg["block_size"])
    cfg["max_steps"]     = getattr(args, "steps",         cfg["max_steps"])
    cfg["learning_rate"] = getattr(args, "lr",            cfg["learning_rate"])
    cfg["warmup_steps"]  = getattr(args, "warmup_steps",  cfg["warmup_steps"])
    cfg["min_lr"]        = getattr(args, "min_lr",        cfg.get("min_lr", 3e-5))

    # ── Tokenizer ────────────────────────────────────────────
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    cfg = update_vocab_size(cfg, tokenizer)

    # ── Tokenize corpus ──────────────────────────────────────
    if corpus_needs_tokenization(corpus_dir, token_file):
        print("Corpus changed or tokens missing — tokenizing...")
        tokenize_corpus(
            input_dir=corpus_dir,
            tokenizer=tokenizer,
            output_path=token_file,
        )
    else:
        print("Tokenized corpus is up to date.")

    # ── Load tokens ──────────────────────────────────────────
    print(f"Loading corpus: {token_file}")
    tokens = load_token_tensor(token_file)
    print(f"Corpus: {len(tokens):,} tokens ({len(tokens)/1e6:.2f}M)")

    val_tokens = None
    if val_token_file.exists():
        val_tokens = load_token_tensor(val_token_file)
        print(f"Validation corpus: {len(val_tokens):,} tokens")
    else:
        print("No validation corpus found — skipping validation loss.")
        print(f"  To enable: tokenize held-out conversations → {val_token_file}")

    # ── Dataloader ───────────────────────────────────────────
    loader    = build_dataloader(
        tokens,
        block_size=cfg["block_size"],
        batch_size=cfg["batch_size"],
        shuffle=True,
    )
    data_iter = iter(loader)

    # ── Model ────────────────────────────────────────────────
    model = GPT(
        vocab_size=cfg["vocab_size"],
        dim=cfg["dim"],
        layers=cfg["layers"],
        heads=cfg["heads"],
        max_seq=cfg["block_size"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # ── Optimizer ────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    # ── Scheduler ────────────────────────────────────────────
    scheduler = build_scheduler(
        optimizer,
        total_steps=cfg["max_steps"],
        warmup_steps=cfg["warmup_steps"],
        min_lr=cfg["min_lr"],
    )

    # ── Resume ───────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_step = try_resume_checkpoint(model, optimizer, scheduler, out_dir, device)
    step       = start_step

    # ── Gradient accumulation ────────────────────────────────
    accum_steps = getattr(args, "accum_steps", 1)
    if accum_steps > 1:
        print(
            f"Gradient accumulation: {accum_steps} steps "
            f"(effective batch size: {cfg['batch_size'] * accum_steps})"
        )

    # ── Training loop ────────────────────────────────────────
    print(f"\nTraining from step {start_step} → {cfg['max_steps']}\n")

    model.train()
    optimizer.zero_grad()

    start_time = time.time()
    accum_loss = 0.0

    while step < cfg["max_steps"]:

        # Fetch next batch, cycling the dataloader as needed
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch     = next(data_iter)

        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)

        # Forward + loss
        logits = model(x)
        loss   = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )

        # Scale for gradient accumulation
        (loss / accum_steps).backward()
        accum_loss += loss.item()

        # Optimizer step every accum_steps
        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Logging
        if step % args.log_interval == 0:
            elapsed    = time.time() - start_time
            current_lr = scheduler.get_last_lr()[0]
            avg_loss   = accum_loss / max(step - start_step + 1, 1)

            val_str = ""
            if (val_tokens is not None
                    and step > 0
                    and step % (args.log_interval * 10) == 0):
                v_loss  = validation_loss(model, val_tokens, cfg, device)
                val_str = f" | val {v_loss:.4f}"

            print(
                f"step {step:6d} | "
                f"loss {loss.item():.4f} | "
                f"avg {avg_loss:.4f} | "
                f"lr {current_lr:.2e}"
                f"{val_str} | "
                f"{elapsed:.0f}s"
            )

        # Checkpoint
        if step % args.save_interval == 0 and step > start_step:
            save_checkpoint(
                out_dir, step,
                model, optimizer, scheduler, cfg,
                s3=s3, s3_bucket=args.s3_bucket,
            )

        step += 1

    # Final checkpoint
    print("\nTraining complete.")
    save_checkpoint(
        out_dir, step,
        model, optimizer, scheduler, cfg,
        s3=s3, s3_bucket=args.s3_bucket,
    )


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--corpus_dir",    default="../corpus")
    parser.add_argument("--data_path",     default="../data/corpus.pt")
    parser.add_argument("--out_dir",       default="../checkpoints")
    parser.add_argument("--tokenizer",     default="mistralai/Mistral-7B-v0.1")

    parser.add_argument("--steps",         type=int,   default=DEFAULT_CONFIG["max_steps"])
    parser.add_argument("--batch_size",    type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--seq_len",       type=int,   default=DEFAULT_CONFIG["block_size"])
    parser.add_argument("--lr",            type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--warmup_steps",  type=int,   default=DEFAULT_CONFIG["warmup_steps"])
    parser.add_argument("--min_lr",        type=float, default=3e-5)
    parser.add_argument("--accum_steps",   type=int,   default=1)

    parser.add_argument("--log_interval",  type=int,   default=DEFAULT_CONFIG["log_interval"])
    parser.add_argument("--save_interval", type=int,   default=DEFAULT_CONFIG["save_interval"])

    parser.add_argument("--s3_bucket",     default=None)

    args = parser.parse_args()
    main(args)
