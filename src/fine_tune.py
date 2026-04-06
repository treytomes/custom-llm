# fine_tune.py
"""
fine_tune.py — DPO fine-tuning from preference pairs

Implements Direct Preference Optimization on top of a Phase 1 checkpoint.
Given (prompt, chosen, rejected) pairs, updates the model to assign higher
probability to chosen responses and lower probability to rejected ones.

DPO loss (with reference model):
    chosen_ratio   = log_prob_chosen   - ref_log_prob_chosen
    rejected_ratio = log_prob_rejected - ref_log_prob_rejected
    L = -log(sigmoid(beta * (chosen_ratio - rejected_ratio)))

The reference model is a frozen copy of the base checkpoint. Computing
the loss relative to the reference prevents the model from drifting
arbitrarily — it must improve on chosen relative to where it started,
not just in absolute terms.

Where beta controls preference enforcement strength:
  - Low  (0.05): gentle nudge, preserves base model character
  - High (0.5):  strong enforcement, more aggressive shaping

At 50M parameters, use a conservative learning rate (1e-6 to 5e-6) and
short runs with frequent evaluation to avoid overwriting what was built
during Phase 1.

Usage:
    python fine_tune.py --pairs ./dpo_data/pairs.jsonl
    python fine_tune.py --pairs ./dpo_data/pairs.jsonl --steps 50 --lr 2e-6
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from train.model import GPT
from train.config import DEFAULT_CONFIG


USER_NAME  = "Trey"
MODEL_NAME = "Scout"


# ── Log probability computation ────────────────────────────────────────────────

def compute_log_prob(
    model: GPT,
    input_ids: torch.Tensor,
    response_start: int,
) -> torch.Tensor:
    """
    Compute the mean log probability of response tokens only.

    Only the response portion is scored — not the prompt — because we
    want to measure how likely the model is to produce this response
    given this prompt.

    Args:
        model:          GPT model (training or reference)
        input_ids:      (1, seq_len) tokenized prompt + response
        response_start: token index where the response begins

    Returns:
        scalar tensor: mean log prob of response tokens
    """
    # model returns raw logits: (1, seq_len, vocab_size)
    logits = model(input_ids)

    # Shift: logits[i] predicts token[i+1]
    shift_logits = logits[0, :-1, :]    # (seq_len-1, vocab_size)
    shift_labels = input_ids[0, 1:]     # (seq_len-1,)

    # Score only the response tokens
    resp_logits = shift_logits[response_start - 1:]
    resp_labels = shift_labels[response_start - 1:]

    if resp_labels.shape[0] == 0:
        return torch.tensor(0.0, requires_grad=True, device=input_ids.device)

    log_probs       = F.log_softmax(resp_logits, dim=-1)
    token_log_probs = log_probs.gather(1, resp_labels.unsqueeze(1)).squeeze(1)

    return token_log_probs.mean()


# ── DPO loss ───────────────────────────────────────────────────────────────────

def dpo_loss(
    log_prob_chosen: torch.Tensor,
    log_prob_rejected: torch.Tensor,
    ref_log_prob_chosen: torch.Tensor,
    ref_log_prob_rejected: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Direct Preference Optimization loss with reference model.

    Computes the loss relative to a frozen reference model rather than
    absolute log probabilities. This prevents arbitrary drift — the
    training model must improve on chosen relative to where it started.

    Args:
        log_prob_chosen:       training model log prob of chosen response
        log_prob_rejected:     training model log prob of rejected response
        ref_log_prob_chosen:   reference model log prob of chosen response
        ref_log_prob_rejected: reference model log prob of rejected response
        beta:                  preference enforcement strength
    """
    chosen_ratio   = log_prob_chosen   - ref_log_prob_chosen
    rejected_ratio = log_prob_rejected - ref_log_prob_rejected
    return -F.logsigmoid(beta * (chosen_ratio - rejected_ratio))


# ── Fine-tuning loop ───────────────────────────────────────────────────────────

def fine_tune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'═' * 60}")
    print(f"  DPO Fine-tuning — {MODEL_NAME}")
    print(f"{'═' * 60}")
    print(f"  Pairs      : {args.pairs}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Steps      : {args.steps}")
    print(f"  Beta       : {args.beta}")
    print(f"  LR         : {args.lr}")
    print(f"  Device     : {device}")
    print(f"{'═' * 60}\n")

    # ── Load pairs ─────────────────────────────────────────────────────────────
    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

    pairs = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    if not pairs:
        raise ValueError("No pairs found.")

    print(f"Loaded {len(pairs)} preference pairs")

    # ── Load tokenizer and config ──────────────────────────────────────────────
    cfg = DEFAULT_CONFIG.copy()
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    cfg["vocab_size"] = tokenizer.vocab_size

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt

    # ── Build training model ───────────────────────────────────────────────────
    model = GPT(
        vocab_size=cfg["vocab_size"],
        dim=cfg["dim"],
        layers=cfg["layers"],
        heads=cfg["heads"],
        max_seq=cfg["block_size"],
    ).to(device)
    model.load_state_dict(state)

    # ── Build frozen reference model ───────────────────────────────────────────
    # The reference model is a frozen copy of the base checkpoint.
    # It provides a stable baseline for the DPO loss so the training
    # model is rewarded for improving relative to where it started,
    # not just for assigning high probability to chosen in absolute terms.
    ref_model = GPT(
        vocab_size=cfg["vocab_size"],
        dim=cfg["dim"],
        layers=cfg["layers"],
        heads=cfg["heads"],
        max_seq=cfg["block_size"],
    ).to(device)
    ref_model.load_state_dict(copy.deepcopy(state))

    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model       : {total_params:.1f}M parameters")
    print(f"Ref model   : frozen ({total_params:.1f}M parameters, no grad)")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    # ── Output directory ───────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_step = ckpt.get("step", 0)

    # ── Training loop ──────────────────────────────────────────────────────────
    model.train()
    print(f"\nStarting DPO fine-tuning...\n")

    total_loss  = 0.0
    t0          = time.time()
    pairs_cycle = 0
    skipped     = 0

    for step in range(args.steps):
        pair = pairs[pairs_cycle % len(pairs)]
        pairs_cycle += 1

        prompt   = pair["prompt"]
        chosen   = pair["chosen"]
        rejected = pair["rejected"]

        # ── Reconstruct conversational context ────────────────────────────────
        history = pair.get("history", [])
        if history:
            context = "\n\n".join(history) + f"\n\n[{USER_NAME}] {prompt}"
        else:
            context = f"[{USER_NAME}] {prompt}"

        # Prime with Scout's name — response_start lands after this
        prompt_text   = context + f"\n\n[{MODEL_NAME}]"
        chosen_text   = prompt_text + chosen
        rejected_text = prompt_text + rejected

        # ── Tokenize ──────────────────────────────────────────────────────────
        prompt_ids     = tokenizer.encode(prompt_text, add_special_tokens=False)
        response_start = len(prompt_ids)

        chosen_ids = tokenizer.encode(
            chosen_text, return_tensors="pt", add_special_tokens=False
        ).to(device)

        rejected_ids = tokenizer.encode(
            rejected_text, return_tensors="pt", add_special_tokens=False
        ).to(device)

        # ── Skip invalid sequences ────────────────────────────────────────────
        if chosen_ids.shape[1] > cfg["block_size"] or \
           rejected_ids.shape[1] > cfg["block_size"]:
            skipped += 1
            continue

        if response_start >= chosen_ids.shape[1] or \
           response_start >= rejected_ids.shape[1]:
            skipped += 1
            continue

        # ── Reference log probs (no gradient) ─────────────────────────────────
        with torch.no_grad():
            ref_log_prob_chosen   = compute_log_prob(
                ref_model, chosen_ids,   response_start
            )
            ref_log_prob_rejected = compute_log_prob(
                ref_model, rejected_ids, response_start
            )

        # ── Training model log probs (with gradient) ───────────────────────────
        log_prob_chosen   = compute_log_prob(model, chosen_ids,   response_start)
        log_prob_rejected = compute_log_prob(model, rejected_ids, response_start)

        # ── DPO loss ───────────────────────────────────────────────────────────
        loss = dpo_loss(
            log_prob_chosen,
            log_prob_rejected,
            ref_log_prob_chosen,
            ref_log_prob_rejected,
            beta=args.beta,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        # ── Logging ───────────────────────────────────────────────────────────
        if (step + 1) % args.log_every == 0:
            avg_loss = total_loss / args.log_every
            elapsed  = time.time() - t0
            print(
                f"step {step+1:>5} | "
                f"loss {avg_loss:.4f} | "
                f"chosen_lp {log_prob_chosen.item():>7.3f} | "
                f"rejected_lp {log_prob_rejected.item():>7.3f} | "
                f"ref_chosen_lp {ref_log_prob_chosen.item():>7.3f} | "
                f"{elapsed:.1f}s"
            )
            total_loss = 0.0
            t0 = time.time()

    # ── Summary ────────────────────────────────────────────────────────────────
    if skipped:
        print(f"\nSkipped {skipped} pairs (exceeded context window or empty response)")

    # ── Save ───────────────────────────────────────────────────────────────────
    save_path = output_dir / "dpo_latest.pt"
    torch.save({
        "step":      base_step,
        "dpo_steps": args.steps,
        "model":     model.state_dict(),
        "config":    cfg,
    }, save_path)

    print(f"\n{'═' * 60}")
    print(f"  DPO complete.")
    print(f"  Checkpoint : {save_path}")
    print(f"\n  Next steps:")
    print(f"    Test : python infer.py --checkpoint {save_path}")
    print(f"    More : run another interact.py session and accumulate pairs")
    print(f"{'═' * 60}")


# ── Entry ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DPO fine-tuning from preference pairs"
    )
    parser.add_argument("--pairs",      type=str, required=True,
                        help="Path to pairs.jsonl from dpo.py")
    parser.add_argument("--checkpoint", type=str,
                        default="./checkpoints/latest.pt",
                        help="Base Phase 1 checkpoint to fine-tune from")
    parser.add_argument("--output",     type=str,
                        default="./checkpoints/dpo/",
                        help="Directory to save the fine-tuned checkpoint")
    parser.add_argument("--steps",      type=int, default=50,
                        help="DPO training steps (start small: 50-100)")
    parser.add_argument("--lr",         type=float, default=2e-6,
                        help="Learning rate — conservative for 50M model")
    parser.add_argument("--beta",       type=float, default=0.1,
                        help="Preference enforcement strength (0.05-0.5)")
    parser.add_argument("--log-every",  type=int, default=10,
                        help="Log every N steps")
    args = parser.parse_args()

    fine_tune(args)
