import json
import re
import torch
from datetime import datetime
from transformers import AutoTokenizer
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.live import Live

from train.model import GPT
import config

console = Console()


# ── Chat logging ─────────────────────────────────────────────────────────────

def log_chat(prompt, response):

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user": config.USER_NAME,
        "model": config.MODEL_NAME,
        "prompt": prompt,
        "response": response,
    }

    with open(config.LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


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


# ── Prompt formatting ─────────────────────────────────────────────────────────

def format_prompt(text: str) -> str:
    return f"[{config.USER_NAME}] {text}\n\n[{config.MODEL_NAME}] "


# ── Chat REPL (called from main.py) ───────────────────────────────────────────

def run_chat_repl():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]{config.MODEL_NAME} Interactive Chat[/bold cyan]\n"
            f"Device: [yellow]{device}[/yellow]",
            border_style="cyan",
        )
    )

    console.print("[dim]Type your prompt and press Enter. Ctrl+C to exit.[/dim]\n")

    console.print("[yellow]Loading model...[/yellow]")

    model, tokenizer = load_model(config.CHECKPOINT_PATH, device)

    console.print("[green]Model loaded.[/green]\n")

    try:

        while True:

            user_text = Prompt.ask(f"[bold blue]{config.USER_NAME}[/bold blue]")

            if not user_text.strip():
                continue

            prompt = format_prompt(user_text)

            spinner = Spinner("dots", text="Generating...")

            response_chunks = []
            printed = ""

            with Live(spinner, console=console, refresh_per_second=10):

                for text in stream_generate(model, tokenizer, prompt, device):

                    delta = text[len(printed):]

                    if delta:
                        console.print(delta, end="", soft_wrap=True)

                    printed = text

                    response_chunks.append(delta)

            console.print()

            output = "".join(response_chunks)

            console.print(
                Panel(
                    output,
                    title=f"[bold green]{config.MODEL_NAME}[/bold green]",
                    border_style="green",
                )
            )

            console.print()

            log_chat(user_text, output)

    except KeyboardInterrupt:

        console.print("\n[bold red]Exiting chat.[/bold red]\n")