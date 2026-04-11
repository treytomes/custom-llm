"""
Clean a chat log using a teacher model and run short SFT training.

Pipeline
────────────────────────────────
1. Load JSONL chat log
2. Format transcript
3. Teacher labels exchanges KEEP or DROP
4. Teacher repairs weak responses
5. Build cleaned transcript
6. Run short SFT training pass
"""

import datetime
import json
import logging
import os
import re
import shutil
import time
import torch
from pathlib import Path

import ai_client.azure
import config
from ai_client.tokenizer import load_tokenizer
from model.loader import init_model, load_checkpoint
from train.dream_train import build_chunks, compute_loss

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────
# LOAD JSONL LOG
# ───────────────────────────────────────────────────────────

def get_next_chat_filename(base_dir):
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.datetime.now(datetime.UTC)
    y = today.year
    m = today.month
    d = today.day

    pattern = re.compile(
        rf"chat_{y}_{m}_{d}_conversation_(\d+)\.txt"
    )

    max_index = 0
    for f in base_dir.glob(f"chat_{y}_{m}_{d}_conversation_*.txt"):
        match = pattern.match(f.name)
        if match:
            idx = int(match.group(1))
            max_index = max(max_index, idx)

    next_index = max_index + 1
    filename = f"chat_{y}_{m}_{d}_conversation_{next_index}.txt"
    return base_dir / filename


def load_chat_pairs(path):
    pairs = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except:
                continue

            prompt = r.get("prompt", "").strip()
            response = r.get("response", "").strip()

            if prompt and response:
                pairs.append((prompt, response))

    return pairs


# ───────────────────────────────────────────────────────────
# FORMAT TRANSCRIPT
# ───────────────────────────────────────────────────────────

def build_transcript(pairs):
    lines = []
    for prompt, response in pairs:
        lines.append(f"[{config.USER_NAME}] {prompt}")
        lines.append(f"[{config.MODEL_NAME}] {response}")
    return "\n".join(lines)


# ───────────────────────────────────────────────────────────
# TEACHER PROMPT
# ───────────────────────────────────────────────────────────

CLEANUP_PROMPT = f"""
You are reviewing conversation logs from a conversational AI named {config.MODEL_NAME}.

Your task is to prepare the conversation for training.

Each exchange contains:

[{config.USER_NAME}] user message

[{config.MODEL_NAME}] model reply

For each exchange decide:

KEEP
or
DROP

Rules
────────────────
KEEP if:
• the response is coherent
• it relates to the prompt
• it reflects the model's thoughtful voice

DROP if:
• the response is incomplete
• the response does not answer the prompt
• the response is confused or nonsensical

If an exchange is KEEP but the response contains sentence fragments or bad grammar,
rewrite the response so it becomes a complete and natural answer
while preserving the original meaning.

STRICT OUTPUT FORMAT
────────────────

Exchange 1: KEEP
[{config.USER_NAME}] ...
[{config.MODEL_NAME}] ...

Exchange 2: DROP

Exchange 3: KEEP
[{config.USER_NAME}] ...
[{config.MODEL_NAME}] ...

Do not invent new topics.
Do not change the prompt.
Output only the labeled exchanges.
"""


# ───────────────────────────────────────────────────────────
# TEACHER CLEANUP
# ───────────────────────────────────────────────────────────

def teacher_cleanup(client, transcript):
    return azure.generate(client, transcript, 0.2, 5000)


# ───────────────────────────────────────────────────────────
# PARSE CLEANED OUTPUT
# ───────────────────────────────────────────────────────────

def parse_cleaned_output(text):
    cleaned_lines = []
    keep = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("Exchange"):
            keep = "KEEP" in line
            continue

        if keep and (
            line.startswith(f"[{config.USER_NAME}]")
            or line.startswith(f"[{config.MODEL_NAME}]")
        ):
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


# ───────────────────────────────────────────────────────────
# VALIDATION
# ───────────────────────────────────────────────────────────

def validate_cleaned_chat(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    user_turns = sum(
        1 for l in lines if l.startswith(f"[{config.USER_NAME}]")
    )
    scout_turns = sum(
        1 for l in lines if l.startswith(f"[{config.MODEL_NAME}]")
    )

    if user_turns < 1 or scout_turns < 1:
        return False

    return True


# ───────────────────────────────────────────────────────────
# SAVE CLEANED TRANSCRIPT
# ───────────────────────────────────────────────────────────

def save_cleaned_chat(text, output_dir, chat_filename="cleaned_chat.txt"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / chat_filename
    path.write_text(text, encoding="utf-8")
    return path


# ───────────────────────────────────────────────────────────
# MAIN PIPELINE
# ───────────────────────────────────────────────────────────

def run_chat_training(
    chat_path: Path,
    checkpoint_path: Path,
    steps: int = 120,
    lr: float = 1e-6,
):
    from train.train import save_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("═" * 60)
    logger.info("Chat training — %s", config.MODEL_NAME)
    logger.info("Chat file  : %s", chat_path)
    logger.info("Steps      : %d", steps)
    logger.info("LR         : %.2e", lr)
    logger.info("Device     : %s", device)
    logger.info("═" * 60)

    tokenizer = load_tokenizer()

    if not chat_path.exists():
        raise FileNotFoundError(f"Chat transcript not found: {chat_path}")

    text = chat_path.read_text(encoding="utf-8")

    chunks = build_chunks(
        tokenizer,
        text,
        config.BLOCK_SIZE,
    )

    if not chunks:
        raise ValueError("No training chunks generated from chat transcript.")

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

    logger.info("Chat training complete")
    return checkpoint_dir / "latest.pt"


def run_chat_cleanup_and_training(
    chat_log_path,
    endpoint,
    api_key,
    checkpoint_path
):
    client = azure.build_client(endpoint, api_key)

    pairs = load_chat_pairs(chat_log_path)

    if not pairs:
        logger.info("No valid chat pairs found.")
        return

    transcript = build_transcript(pairs)

    logger.info("Running teacher cleanup...")

    teacher_output = teacher_cleanup(client, transcript)

    cleaned = parse_cleaned_output(teacher_output)

    # ── Show transcript to user ─────────────────────────────

    logger.info("──── Cleaned Transcript ────")
    logger.info(cleaned)
    logger.info("────────────────────────────")

    if not validate_cleaned_chat(cleaned):
        logger.info("Cleaned chat failed validation.")
        return

    chat_dir = Path("../data/chat_transcripts")
    chat_path = get_next_chat_filename(chat_dir)
    chat_path.write_text(cleaned, encoding="utf-8")
    logger.info(f"Cleaned transcript saved → {chat_path}")

    # ── Ask user for approval ───────────────────────────────

    decision = input("Accept this transcript for training? (y/n): ").strip().lower()

    if decision not in ("y", "yes"):
        logger.info("Transcript rejected. Training aborted.")
        try:
            chat_path.unlink()
        except FileNotFoundError:
            pass
        return

    logger.info("Transcript accepted.")
    logger.info("Starting chat SFT training...")

    run_chat_training(
        chat_path=chat_path,
        checkpoint_path=checkpoint_path,
        steps=120,
        lr=1e-6,
    )

    logger.info("Chat training complete.")
