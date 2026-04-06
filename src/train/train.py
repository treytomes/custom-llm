"""
train.py — Training engine for Scout

This module contains the core training loop but does not expose a CLI.
The application entry point (main.py) is responsible for launching it
and handling user-facing logging / dashboard output.
"""

import logging
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from transformers import AutoTokenizer

import config
from .model import GPT
from .data import load_token_tensor, build_dataloader


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────

def build_scheduler(optimizer, total_steps, warmup_steps, min_lr):
    warmup = LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps),
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

def save_checkpoint(out_dir, step, model, optimizer, scheduler, cfg):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    logger.info("Saved checkpoint: %s", ckpt_path)


def try_resume_checkpoint(model, optimizer, scheduler, checkpoint_dir, device):

    latest = Path(checkpoint_dir) / "latest.pt"

    if not latest.exists():
        logger.info("No checkpoint found — starting fresh.")
        return 0

    logger.info("Resuming from checkpoint: %d", latest)

    ckpt = torch.load(latest, map_location=device)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        logger.info("Warning: no scheduler state in checkpoint — scheduler reset.")

    step = ckpt.get("step", 0)

    logger.info("Resumed from step %d", step)

    return step


# ─────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def validation_loss(model, val_tokens, device, num_batches=20):

    model.eval()

    loader = build_dataloader(
        val_tokens,
        block_size=config.BLOCK_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    total_loss = 0.0
    count = 0

    for batch in loader:

        if count >= num_batches:
            break

        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits = model(x)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )

        total_loss += loss.item()
        count += 1

    model.train()

    return total_loss / max(count, 1)


# ─────────────────────────────────────────────────────────────
# Training entry point
# ─────────────────────────────────────────────────────────────

def run_training():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    token_file = Path(config.DATA_PATH)
    checkpoint_dir = Path(config.CHECKPOINT_DIR)

    val_token_file = token_file.parent / "corpus_val.pt"

    # ── Load tokens ──────────────────────────────────────────

    logger.info("Loading corpus: %s", token_file)

    tokens, vocab_size = load_token_tensor(token_file)

    logger.info("Corpus: %d tokens (%fM)", len(tokens), len(tokens)/1e6)

    val_tokens = None

    if val_token_file.exists():
        val_tokens, val_vocab_size = load_token_tensor(val_token_file)
        logger.info("Validation corpus: %d tokens (%d vocab size)", len(val_tokens), val_vocab_size)
    else:
        logger.info("No validation corpus found — skipping validation loss.")

    # ── Tokenizer ────────────────────────────────────────────

    logger.info("Loading tokenizer: %s", config.TOKENIZER_NAME)

    if vocab_size is None:
        tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
        vocab_size = tokenizer.vocab_size

    # ── DataLoader ───────────────────────────────────────────

    loader = build_dataloader(
        tokens,
        block_size=config.BLOCK_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )

    data_iter = iter(loader)

    # ── Model ────────────────────────────────────────────────

    model = GPT(
        vocab_size=vocab_size,
        dim=config.MODEL_DIM,
        layers=config.MODEL_LAYERS,
        heads=config.MODEL_HEADS,
        max_seq=config.BLOCK_SIZE,
    ).to(device)
    model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())

    logger.info("Model parameters: %d (%fM)", n_params, n_params/1e6)

    # ── Optimizer ────────────────────────────────────────────

    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    # ── Scheduler ────────────────────────────────────────────

    scheduler = build_scheduler(
        optimizer,
        total_steps=config.MAX_STEPS,
        warmup_steps=config.WARMUP_STEPS,
        min_lr=config.MIN_LR,
    )

    # ── Resume ───────────────────────────────────────────────

    start_step = try_resume_checkpoint(
        model,
        optimizer,
        scheduler,
        checkpoint_dir,
        device,
    )

    step = start_step

    logger.info("Training from step %d → %d", start_step, config.MAX_STEPS)

    model.train()

    optimizer.zero_grad()

    start_time = time.time()

    accum_loss = 0.0

    # ── Training loop ────────────────────────────────────────

    last_log_time = start_time
    while step < config.MAX_STEPS:

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits = model(x)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        accum_loss += loss.item()

        # Logging

        if step % config.LOG_INTERVAL == 0:
            now = time.time()
            elapsed = now - start_time
            step_time = now - last_log_time
            last_log_time = now

            tokens_per_step = config.BLOCK_SIZE * config.BATCH_SIZE
            tokens_per_sec = tokens_per_step / max(step_time, 1e-6)

            remaining_steps = config.MAX_STEPS - step
            eta_seconds = remaining_steps * step_time

            current_lr = scheduler.get_last_lr()[0]

            avg_loss = accum_loss / max(step - start_step + 1, 1)

            val_loss = None
            if (
                val_tokens is not None
                and step > 0
                and step % (config.LOG_INTERVAL * 10) == 0
            ):
                val_loss = validation_loss(model, val_tokens, device)

            yield {
                "step": step,
                "loss": loss.item(),
                "avg_loss": avg_loss,
                "lr": current_lr,
                "val_loss": val_loss,
                "elapsed": elapsed,
                "tokens_per_sec": tokens_per_sec,
                "eta": eta_seconds,
            }

        # Checkpoint

        if step % config.SAVE_INTERVAL == 0 and step > start_step:

            save_checkpoint(
                checkpoint_dir,
                step,
                model,
                optimizer,
                scheduler,
                {
                    "vocab_size": vocab_size,
                    "dim": config.MODEL_DIM,
                    "layers": config.MODEL_LAYERS,
                    "heads": config.MODEL_HEADS,
                    "block_size": config.BLOCK_SIZE,
                },
            )

        step += 1

    logger.info("Training complete.")

    save_checkpoint(
        checkpoint_dir,
        step,
        model,
        optimizer,
        scheduler,
        {},
    )