"""
train/dream_train.py

Lightweight supervised fine‑tuning on Scout dream files.
Dreams are treated as plain causal LM training data.

Input:
    directory of dream text files

Output:
    updated checkpoint
"""

import logging
import shutil
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer

from model.loader import init_model, load_checkpoint
import config

logger = logging.getLogger(__name__)


def load_recent_dreams(dream_dir: Path, window: int = 5):
    dream_files = sorted(
        dream_dir.glob("*.txt"),
        key=lambda p: p.stat().st_mtime
    )

    if not dream_files:
        raise ValueError("No dream files found.")

    recent = dream_files[-window:]

    texts = []
    for p in recent:
        try:
            texts.append(p.read_text(encoding="utf-8"))
        except:
            continue

    if not texts:
        raise ValueError("Dream files could not be read.")

    return "\n\n".join(texts)


def build_chunks(tokenizer, text, block_size, overlap=2):
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    turns = [
        l for l in lines
        if l.startswith(f"[{config.MODEL_NAME}]") or l.startswith("[Inner]")
    ]

    # Build Scout–Inner exchanges
    exchanges = []
    i = 0
    while i < len(turns) - 1:
        if turns[i].startswith(f"[{config.MODEL_NAME}]") and turns[i+1].startswith("[Inner]"):
            exchanges.append(turns[i] + "\n" + turns[i+1])
            i += 2
        else:
            exchanges.append(turns[i])
            i += 1

    chunks = []
    i = 0

    while i < len(exchanges):
        current = []
        tokens_used = 0
        j = i

        while j < len(exchanges):
            ex = exchanges[j]
            ex_tokens = tokenizer.encode(ex + "\n", add_special_tokens=False)

            if tokens_used + len(ex_tokens) > block_size:
                break

            current.append(ex)
            tokens_used += len(ex_tokens)
            j += 1

        if current:
            chunk_text = "\n".join(current)
            chunk_tokens = tokenizer.encode(chunk_text, add_special_tokens=False)
            chunks.append(torch.tensor(chunk_tokens, dtype=torch.long))

        if j == len(exchanges):
            break

        i = max(j - overlap, i + 1)

    return chunks


def compute_loss(model, input_ids):
    logits = model(input_ids)

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
    )


def run_dream_training(
    dream_dir: Path,
    checkpoint_path: Path,
    steps: int = 300,
    lr: float = 5e-6,
):
    from train.train import save_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("═" * 60)
    logger.info("Dream training — %s", config.MODEL_NAME)
    logger.info("Dreams     : %s", dream_dir)
    logger.info("Steps      : %d", steps)
    logger.info("LR         : %.2e", lr)
    logger.info("Device     : %s", device)
    logger.info("═" * 60)

    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)

    text = load_recent_dreams(dream_dir)

    chunks = build_chunks(
        tokenizer,
        text,
        config.BLOCK_SIZE,
    )

    if not chunks:
        raise ValueError("No training chunks generated from dream text.")

    logger.info("Training chunks: %d", len(chunks))

    model = init_model(tokenizer.vocab_size, device)

    checkpoint, state = load_checkpoint(
        checkpoint_path,
        model,
        device,
    )

    base_step = checkpoint.get("step", 0)
    global_step = base_step

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=steps,
    )

    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if "scheduler" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler"])
        except Exception:
            logger.warning("Scheduler state incompatible — resetting.")

    model.train()

    total_loss = 0.0
    t0 = time.time()

    for step in range(steps):

        batch = chunks[step % len(chunks)].unsqueeze(0).to(device)

        loss = compute_loss(model, batch)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        global_step += 1
        total_loss += loss.item()

        if (step + 1) % 20 == 0:

            avg = total_loss / 20
            elapsed = time.time() - t0

            logger.info(
                "step %d | loss %.4f | %.1fs",
                global_step,
                avg,
                elapsed,
            )

            total_loss = 0.0
            t0 = time.time()

    checkpoint_dir = checkpoint_path.parent

    logger.info("Saving checkpoint at step %d", global_step)

    save_checkpoint(
        out_dir=checkpoint_dir,
        step=global_step,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg={
            "vocab_size": tokenizer.vocab_size,
            "dim": config.MODEL_DIM,
            "layers": config.MODEL_LAYERS,
            "heads": config.MODEL_HEADS,
            "block_size": config.BLOCK_SIZE,
        },
    )

    logger.info("Dream training complete")

    return checkpoint_dir / "latest.pt"