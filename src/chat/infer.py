import torch
from rich.console import Console

import config

console = Console()


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

    generated_ids = []
    buffer_ids = []

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

        if eos_id is not None and tok_id == eos_id:
            break

        buffer_ids.append(tok_id)

        preview = tokenizer.decode(buffer_ids, skip_special_tokens=True).strip()

        # Detect baton pass like "Trey]" or "Mariam]"
        if preview.endswith("]"):
            return "<silence>"

        # Detect new speaker tag
        if "[" in preview:
            return "<silence>"

        # Once buffer reaches 3 tokens, release it
        if len(buffer_ids) >= 3:
            generated_ids.extend(buffer_ids)
            buffer_ids.clear()

    # Flush remaining buffer
    generated_ids.extend(buffer_ids)

    if not generated_ids:
        return "<silence>"

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return text.strip()


def stream_generate(model, tokenizer, prompt, device):
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    eos_id = tokenizer.eos_token_id

    generated_ids = []
    buffer_ids = []

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

        if eos_id is not None and tok_id == eos_id:
            break

        buffer_ids.append(tok_id)

        preview = tokenizer.decode(buffer_ids, skip_special_tokens=True).strip()

        # Baton pass detection
        if preview.endswith("]"):
            if not generated_ids:
                yield "<silence>"
            return

        # Speaker tag detection
        if "[" in preview:
            if not generated_ids:
                yield "<silence>"
            return

        # Release buffer once safe
        if len(buffer_ids) >= 3:
            generated_ids.extend(buffer_ids)
            buffer_ids.clear()

            text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            yield text

    generated_ids.extend(buffer_ids)

    if not generated_ids:
        yield "<silence>"
        return

    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    yield text