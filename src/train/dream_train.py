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

import config
from ai_client.tokenizer import load_tokenizer
from model.loader import 

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


def build_chunks(
    tokenizer: TokenizersBackend | SentencePieceBackend,
    text: str,
    block_size: int,
    overlap: int = 3
) -> list[torch.Tensor]:
    """
    Convert a dream transcript into tokenized training chunks.

    Dream transcripts contain reflective dialogue between the model's
    outward voice and its internal reasoning voice. Each line must begin
    with a bracketed speaker tag such as:

        [MODEL_NAME] outward reflection
        [Inner] internal reasoning

    Unlike the previous implementation, this function does NOT assume
    that lines occur in fixed pairs. Instead it treats each labeled line
    as an independent conversational turn and packs turns sequentially
    into chunks that fit within the model's context window.

    This approach is more robust for self‑generated dreams where the
    model may produce long runs of `[Inner]` reflections without
    alternating speakers.

    Parameters
    ----------
    tokenizer
        HuggingFace tokenizer used to convert text into token IDs.
    text
        Raw dream transcript containing labeled speaker turns.
    block_size
        Maximum number of tokens allowed in a training chunk.
    overlap
        Number of turns shared between adjacent chunks to preserve
        conversational continuity across chunk boundaries.

    Returns
    -------
    list[torch.Tensor]
        A list of tensors containing token IDs. Each tensor represents
        one training chunk suitable for causal language‑model training.
    """

    # ----------------------------------------------------------
    # Step 1: Extract valid speaker turns.
    #
    # Only lines beginning with the model tag or "[Inner]" are
    # considered part of the dream dialogue.
    # ----------------------------------------------------------
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    turns = [
        l for l in lines
        if l.startswith(f"[{config.MODEL_NAME}]") or l.startswith("[Inner]")
    ]

    # ----------------------------------------------------------
    # Step 2: Pack turns into token chunks.
    #
    # Turns are added sequentially until the token budget for
    # the chunk would be exceeded.
    # ----------------------------------------------------------
    chunks = []
    i = 0

    while i < len(turns):
        current_turns = []
        tokens_used = 0
        j = i

        while j < len(turns):

            turn = turns[j]

            # Estimate token count if this turn were added
            turn_tokens = tokenizer.encode(
                turn + "\n",
                add_special_tokens=False
            )

            if tokens_used + len(turn_tokens) > block_size:
                break

            current_turns.append(turn)
            tokens_used += len(turn_tokens)
            j += 1

        # Convert accumulated turns into token IDs
        if current_turns:
            chunk_text = "\n".join(current_turns)

            token_ids = tokenizer.encode(
                chunk_text,
                add_special_tokens=False
            )

            chunks.append(torch.tensor(token_ids, dtype=torch.long))

        # If we've consumed all turns we are finished
        if j == len(turns):
            break

        # ------------------------------------------------------
        # Step 3: Advance the window with overlap.
        #
        # Instead of jumping directly to j, step backward by
        # `overlap` turns so adjacent chunks share context.
        # ------------------------------------------------------
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

    tokenizer = load_tokenizer()

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