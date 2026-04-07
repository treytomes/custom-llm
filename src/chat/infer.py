import torch
from transformers import AutoTokenizer
from rich.console import Console

from train.model import GPT
import config

console = Console()


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    vocab_size = tokenizer.vocab_size

    model = GPT(
        vocab_size=vocab_size,
        dim=config.MODEL_DIM,
        layers=config.MODEL_LAYERS,
        heads=config.MODEL_HEADS,
        max_seq=config.BLOCK_SIZE,
    ).to(device)

    state = checkpoint["model"] if "model" in checkpoint else checkpoint

    # Fix compiled-model checkpoints
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    model.load_state_dict(state)

    model.eval()

    return model, tokenizer


# ── Sampling ──────────────────────────────────────────────────────────────────

def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text))


def sample_next(logits, generated_tokens=None, rep_penalty=1.0):
    logits = logits.clone()

    if generated_tokens is not None and rep_penalty != 1.0:
        for tok_id in set(generated_tokens.tolist()):
            logits[0, tok_id] /= rep_penalty

    logits = logits / config.TEMPERATURE

    if config.TOP_K is not None:
        v, _ = torch.topk(logits, config.TOP_K)
        logits[logits < v[:, [-1]]] = -float("inf")

    probs = torch.softmax(logits, dim=-1)

    return torch.multinomial(probs, num_samples=1)


# ── Generation ────────────────────────────────────────────────────────────────

def generate(model, tokenizer, prompt, device):
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    eos_id = tokenizer.eos_token_id

    for _ in range(config.MAX_NEW_TOKENS):
        tokens = tokens[:, -config.BLOCK_SIZE:]
        with torch.no_grad():
            logits = model(tokens)

        logits = logits[:, -1, :]

        next_token = sample_next(
            logits,
            generated_tokens=tokens[0],
            rep_penalty=config.REP_PENALTY,
        )

        tokens = torch.cat([tokens, next_token], dim=1)

        if eos_id is not None and next_token.item() == eos_id:
            break

    return tokenizer.decode(tokens[0], skip_special_tokens=True)


def stream_generate(model, tokenizer, prompt, device):
    """
    Stream text generation while preserving tokenizer spacing.
    Yields progressively longer decoded strings so the caller
    can print only the new portion each step.
    """

    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    eos_id = tokenizer.eos_token_id
    generated_ids = []

    for _ in range(config.MAX_NEW_TOKENS):
        tokens = tokens[:, -config.BLOCK_SIZE:]

        with torch.no_grad():
            logits = model(tokens)

        logits = logits[:, -1, :]

        next_token = sample_next(
            logits,
            generated_tokens=tokens[0],
            rep_penalty=config.REP_PENALTY,
        )

        tokens = torch.cat([tokens, next_token], dim=1)

        tok_id = next_token.item()
        generated_ids.append(tok_id)

        # Stop cleanly on EOS
        if eos_id is not None and tok_id == eos_id:
            break

        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if text.endswith('['):
            text = text[:-1]
            if len(text) > 0:
                yield text
            break

        yield text
