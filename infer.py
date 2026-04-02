# test_checkpoint.py

import torch
from transformers import AutoTokenizer

from training.model import GPT
from training.config import DEFAULT_CONFIG


CHECKPOINT_PATH = "./checkpoints/latest.pt"
TOKENIZER_NAME = "mistralai/Mistral-7B-v0.1"

MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 40


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

    # handle both raw state_dict and full checkpoint format
    if "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint

    model.load_state_dict(state)
    model.eval()

    return model, tokenizer, cfg


def sample_next(logits):
    logits = logits / TEMPERATURE
    if TOP_K is not None:
        v, _ = torch.topk(logits, TOP_K)
        logits[logits < v[:, [-1]]] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate(model, tokenizer, cfg, prompt, device):
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(MAX_NEW_TOKENS):
        tokens = tokens[:, -cfg["block_size"] :]
        with torch.no_grad():
            logits = model(tokens)
        logits = logits[:, -1, :]
        next_token = sample_next(logits)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokenizer.decode(tokens[0])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model...")
    model, tokenizer, cfg = load_model(CHECKPOINT_PATH, device)
    print("Model loaded")
    while True:
        prompt = input("\nPrompt > ")
        if not prompt.strip():
            continue
        output = generate(model, tokenizer, cfg, prompt, device)
        print("\n--- Output ---\n")
        print(output)


if __name__ == "__main__":
    main()