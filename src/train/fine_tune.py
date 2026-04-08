"""
train/fine_tune.py

DPO fine-tuning from preference pairs.
"""

import copy
import json
import time
import logging
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model.model import GPT
from model.loader import (
    init_model,
    load_checkpoint,
)
import config

logger = logging.getLogger(__name__)


# ── Log probability computation ──────────────────────────────

def compute_log_prob(
    model: GPT,
    input_ids: torch.Tensor,
    response_start: int,
) -> torch.Tensor:

    logits = model(input_ids)

    shift_logits = logits[0, :-1, :]
    shift_labels = input_ids[0, 1:]

    resp_logits = shift_logits[response_start - 1:]
    resp_labels = shift_labels[response_start - 1:]

    if resp_labels.shape[0] == 0:
        return torch.tensor(0.0, requires_grad=True, device=input_ids.device)

    log_probs = F.log_softmax(resp_logits, dim=-1)
    token_log_probs = log_probs.gather(1, resp_labels.unsqueeze(1)).squeeze(1)

    return token_log_probs.mean()


# ── DPO loss ─────────────────────────────────────────────────

def dpo_loss(
    log_prob_chosen,
    log_prob_rejected,
    ref_log_prob_chosen,
    ref_log_prob_rejected,
    beta: float = 0.1,
):

    chosen_ratio   = log_prob_chosen   - ref_log_prob_chosen
    rejected_ratio = log_prob_rejected - ref_log_prob_rejected

    return -F.logsigmoid(beta * (chosen_ratio - rejected_ratio))


# ── Public training entrypoint ───────────────────────────────

def run_dpo_fine_tune(
    pairs_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    steps: int = 200,
    lr: float = 1e-6,
    beta: float = 0.1,
    log_every: int = 10,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("═" * 60)
    logger.info("DPO Fine-tuning — %s", config.MODEL_NAME)
    logger.info("Pairs      : %s", pairs_path)
    logger.info("Checkpoint : %s", checkpoint_path)
    logger.info("Steps      : %d", steps)
    logger.info("Beta       : %.3f", beta)
    logger.info("LR         : %.2e", lr)
    logger.info("Device     : %s", device)
    logger.info("═" * 60)

    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

    pairs = []

    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    if not pairs:
        raise ValueError("No preference pairs found.")

    logger.info("Loaded %d preference pairs", len(pairs))

    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Loading checkpoint: %s", checkpoint_path)

    model = init_model(tokenizer.vocab_size, device)
    ref_model = init_model(tokenizer.vocab_size, device)

    checkpoint, state = load_checkpoint(checkpoint_path, model, device)

    ref_model.load_state_dict(copy.deepcopy(state))

    for p in ref_model.parameters():
        p.requires_grad = False

    ref_model.eval()

    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    logger.info("Model     : %.1fM parameters", total_params)
    logger.info("Reference : frozen")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    base_step = checkpoint.get("step", 0)

    logger.info("Starting DPO training")

    total_loss = 0.0
    t0 = time.time()
    pairs_cycle = 0
    skipped = 0

    model.train()

    for step in range(steps):
        if step % len(pairs) == 0:
            random.shuffle(pairs)
        
        pair = pairs[pairs_cycle % len(pairs)]
        pairs_cycle += 1

        prompt   = pair["prompt"]
        chosen   = pair["chosen"]
        rejected = pair["rejected"]

        history = pair.get("history", [])

        if history:
            context = "\n\n".join(history) + f"\n\n[{config.USER_NAME}] {prompt}"
        else:
            context = f"[{config.USER_NAME}] {prompt}"

        prompt_text   = context + f"\n\n[{config.MODEL_NAME}]"
        chosen_text   = prompt_text + chosen
        rejected_text = prompt_text + rejected

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        response_start = len(prompt_ids)

        chosen_ids = tokenizer.encode(
            chosen_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)

        rejected_ids = tokenizer.encode(
            rejected_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)

        if chosen_ids.shape[1] > config.BLOCK_SIZE or \
           rejected_ids.shape[1] > config.BLOCK_SIZE:

            skipped += 1
            continue

        if response_start >= chosen_ids.shape[1] or \
           response_start >= rejected_ids.shape[1]:

            skipped += 1
            continue

        with torch.no_grad():

            ref_lp_chosen = compute_log_prob(
                ref_model,
                chosen_ids,
                response_start,
            )

            ref_lp_rejected = compute_log_prob(
                ref_model,
                rejected_ids,
                response_start,
            )

        lp_chosen = compute_log_prob(
            model,
            chosen_ids,
            response_start,
        )

        lp_rejected = compute_log_prob(
            model,
            rejected_ids,
            response_start,
        )

        loss = dpo_loss(
            lp_chosen,
            lp_rejected,
            ref_lp_chosen,
            ref_lp_rejected,
            beta=beta,
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % log_every == 0:

            avg_loss = total_loss / log_every
            elapsed  = time.time() - t0

            logger.info(
                "step %d | loss %.4f | chosen_lp %.3f | rejected_lp %.3f | %.1fs",
                step + 1,
                avg_loss,
                lp_chosen.item(),
                lp_rejected.item(),
                elapsed,
            )

            total_loss = 0.0
            t0 = time.time()

    if skipped:
        logger.info("Skipped %d pairs (context overflow)", skipped)

    save_path = output_dir / "dpo_latest.pt"

    torch.save({
        "step": base_step,
        "dpo_steps": steps,
        "model": model.state_dict(),
    }, save_path)

    logger.info("DPO training complete")
    logger.info("Checkpoint saved: %s", save_path)

    return save_path