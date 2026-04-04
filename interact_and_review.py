"""
interact_and_review.py — Single-turn DPO data collection

Generates multiple candidate responses for a prompt, lets the user rank
them immediately, and logs the annotated result for dpo.py.

Usage:
    python interact_and_review.py
    python interact_and_review.py --samples 4
"""

import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoTokenizer

from training.model import GPT
from training.config import DEFAULT_CONFIG
from infer import sample_next

TOKENIZER_NAME = "mistralai/Mistral-7B-v0.1"

MAX_NEW_TOKENS = 60
REP_PENALTY = 1.3

USER_NAME = "Trey"
MODEL_NAME = "Scout"


class SessionLogger:

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.turn = 0

        print(f"Logging to: {self.log_path}")

    def log(self, entry: dict):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_model(checkpoint_path, device):

    cfg = DEFAULT_CONFIG.copy()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    cfg["vocab_size"] = tokenizer.vocab_size

    model = GPT(
        vocab_size=cfg["vocab_size"],
        dim=cfg["dim"],
        layers=cfg["layers"],
        heads=cfg["heads"],
        max_seq=cfg["block_size"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state)

    model.eval()

    return model, tokenizer, cfg


def generate(model, tokenizer, cfg, prompt, device):

    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)

    eos_id = tokenizer.eos_token_id

    for _ in range(MAX_NEW_TOKENS):

        tokens = tokens[:, -cfg["block_size"]:]

        with torch.no_grad():
            logits = model(tokens)

        logits = logits[:, -1, :]

        next_token = sample_next(
            logits,
            generated_tokens=tokens[0],
            rep_penalty=REP_PENALTY,
        )

        tokens = torch.cat([tokens, next_token], dim=1)

        if eos_id is not None and next_token.item() == eos_id:
            break

    return tokenizer.decode(tokens[0], skip_special_tokens=True)


def choose_index(prompt, max_i):

    while True:

        raw = input(prompt).strip()

        if raw == "":
            return None

        if raw.isdigit():

            i = int(raw)

            if 0 <= i < max_i:
                return i

        print(f"Enter a number between 0 and {max_i-1}")


def run(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print("Loading model...")

    model, tokenizer, cfg = load_model(args.checkpoint, device)

    print("Model ready\n")

    logger = SessionLogger(Path(args.log))

    while True:

        user_input = input(f"[{USER_NAME}] > ").strip()

        if not user_input:
            continue

        if user_input == ":quit":
            break

        # Build prompt
        user_input = user_input.replace("\\n", "\n")
        context = f"[{USER_NAME}] {user_input}\n\n[{MODEL_NAME}]"

        candidates = []

        print()

        t0 = time.time()

        for i in range(args.samples):

            out = generate(model, tokenizer, cfg, context, device)

            response = out[len(context):].strip() if out.startswith(context) else out.strip()

            candidates.append(response)

            print(f"\n--- Candidate {i} ---")
            print(f"[{MODEL_NAME}] {response}")

        elapsed = time.time() - t0

        print(f"\n[{args.samples} candidates | {elapsed:.1f}s]\n")

        best = choose_index("Best candidate #: ", args.samples)
        worst = choose_index("Worst candidate #: ", args.samples)

        print("\nOptional: write a better response.")
        correction = input("Correction (Enter to skip): ").strip()

        notes = input("Notes (optional): ").strip()

        entry = {
            "session_id": logger.session_id,
            "turn": logger.turn + 1,
            "timestamp": datetime.now().isoformat(),

            "prompt": user_input,

            "candidates": candidates,

            "chosen": candidates[best] if best is not None else None,
            "rejected": candidates[worst] if worst is not None else None,

            "correction": correction or None,
            "notes": notes or None,
        }

        logger.turn += 1

        logger.log(entry)

        print()

    print(f"\nSession saved to {args.log}")


if __name__ == "__main__":

    default_log = f"./sessions/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    parser = argparse.ArgumentParser()

    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--log", type=str, default=default_log)
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/latest.pt")

    args = parser.parse_args()

    run(args)