# infer.py

import torch
from transformers import AutoTokenizer

from training.model import GPT
from training.config import DEFAULT_CONFIG

from config import *


# Switch this line to test DPO output:
# ACTIVE_CHECKPOINT = CHECKPOINT_PATH
ACTIVE_CHECKPOINT = DPO_CHECKPOINT_PATH


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

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle both raw state_dict and full checkpoint format
    if "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint

    model.load_state_dict(state)
    model.eval()

    return model, tokenizer, cfg


def sample_next(logits, generated_tokens=None, rep_penalty=1.0):
    """
    Sample the next token from logits.

    Args:
        logits:           Raw logits tensor of shape (1, vocab_size)
        generated_tokens: 1-D tensor of already-generated token ids,
                          used to penalise repetition. Pass None to disable.
        rep_penalty:      Repetition penalty factor. Values > 1.0 reduce the
                          probability of tokens that have already appeared.
                          1.0 means no penalty.
    """
    logits = logits.clone()

    # ── Repetition penalty ────────────────────────────────────────────
    if generated_tokens is not None and rep_penalty != 1.0:
        for tok_id in set(generated_tokens.tolist()):
            logits[0, tok_id] /= rep_penalty

    # ── Temperature ───────────────────────────────────────────────────
    logits = logits / TEMPERATURE

    # ── Top-k filtering ───────────────────────────────────────────────
    if TOP_K is not None:
        v, _ = torch.topk(logits, TOP_K)
        logits[logits < v[:, [-1]]] = -float("inf")

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate(model, tokenizer, cfg, prompt, device):
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    eos_id = tokenizer.eos_token_id

    for _ in range(MAX_NEW_TOKENS):

        # Trim to context window
        tokens = tokens[:, -cfg["block_size"]:]

        with torch.no_grad():
            logits = model(tokens)

        # Only the last token's logits matter for next-token prediction
        logits = logits[:, -1, :]

        # Pass the current sequence (flattened to 1-D) for repetition penalty
        next_token = sample_next(
            logits,
            generated_tokens=tokens[0],
            rep_penalty=REP_PENALTY,
        )

        tokens = torch.cat([tokens, next_token], dim=1)

        # Stop cleanly at end-of-sequence
        if eos_id is not None and next_token.item() == eos_id:
            break

    return tokenizer.decode(tokens[0], skip_special_tokens=True)


def format_prompt(text: str) -> str:
    return f"[{USER_NAME}] {text}\n\n[{MODEL_NAME}] "


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Loading model...")
    model, tokenizer, cfg = load_model(ACTIVE_CHECKPOINT, device)
    print("Model loaded\n")

    while True:
        prompt = input("Prompt > ").strip()
        if not prompt:
            continue
        prompt = format_prompt(prompt)
        output = generate(model, tokenizer, cfg, prompt, device)
        print(f"\n--- Output ---\n\n{output}\n")


if __name__ == "__main__":
    main()
