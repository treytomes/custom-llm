"""
interact.py — Interactive session logger for DPO data collection

Generates multiple candidate responses per prompt and logs them to JSONL.
Each entry stores the full text of all candidates, ready for review.py.

Usage:
    python interact.py
    python interact.py --samples 4 --log ./sessions/my_session.jsonl
    python interact.py --checkpoint ./checkpoints/step_080000.pt
"""

import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoTokenizer

from train.model import GPT
from train.config import DEFAULT_CONFIG
from src.chat.infer import sample_next

from config import *


# ── Session Logger ─────────────────────────────────────────────────────────────

class SessionLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.turn = 0
        print(f"Logging to: {self.log_path}")


    def log(self, prompt: str, candidates: list[str], history: list[str] = None):
        self.turn += 1
        entry = {
            "session_id":  self.session_id,
            "turn":        self.turn,
            "timestamp":   datetime.now().isoformat(),
            "prompt":      prompt,
            "history":     history or [],
            "candidates":  candidates,
            "chosen":      None,
            "rejected":    None,
            "correction":  None,
            "notes":       None,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Model Loading ──────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
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


# ── Generation ─────────────────────────────────────────────────────────────────


def format_prompt(speaker: str, text: str) -> str:
    return f"[{speaker}] {text}"


def generate(model, tokenizer, cfg, prompt: str, device: torch.device) -> str:
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


# ── Interactive Session ────────────────────────────────────────────────────────

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading model...")
    model, tokenizer, cfg = load_model(args.checkpoint, device)
    print("Model ready")
    print("Commands: :quit to exit\n")

    logger = SessionLogger(Path(args.log))
    conversation_history = []

    while True:
        raw_input = input(f"[{USER_NAME}] > ").strip()
        if not raw_input:
            continue
        if raw_input == ":quit":
            break

        # Format the user's turn
        user_turn = f"[{USER_NAME}] {raw_input}"
        conversation_history.append(user_turn)

        # Build the full prompt from conversation history,
        # ending with Scout's name to prime the response
        context = "\n\n".join(conversation_history) + f"\n\n[{MODEL_NAME}]"

        candidates = []
        t0 = time.time()

        for i in range(args.samples):
            out = generate(model, tokenizer, cfg, context, device)
            # Strip the context from the output
            response = out[len(context):].strip() if out.startswith(context) else out.strip()
            candidates.append(response)
            print(f"\n--- Candidate {i+1} ---\n[{MODEL_NAME}] {response}")

        elapsed = time.time() - t0
        print(f"\n[{args.samples} candidates | {elapsed:.1f}s]")

        logger.log(raw_input, candidates, conversation_history.copy())

        # Ask which candidate to add to history, or allow manual entry
        if args.samples > 1:
            choice = input(f"\nAdd which candidate to history? (0-{args.samples-1}, Enter to skip): ").strip()
            if choice.isdigit() and 0 <= int(choice) < args.samples:
                chosen_response = candidates[int(choice)]
                scout_turn = f"[{MODEL_NAME}] {chosen_response}"
                conversation_history.append(scout_turn)
        else:
            scout_turn = f"[{MODEL_NAME}] {candidates[0]}"
            conversation_history.append(scout_turn)

        print()

    print(f"\nSession saved to {args.log}")


# ── Entry ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    default_log = f"./sessions/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples",    type=int, default=4,
                        help="Number of candidate responses per prompt")
    parser.add_argument("--log",        type=str, default=default_log,
                        help="Path to session log file")
    parser.add_argument("--checkpoint", type=str,
                        default="./checkpoints/latest.pt",
                        help="Model checkpoint to load")
    args = parser.parse_args()
    run(args)
