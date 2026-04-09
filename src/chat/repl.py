"""
chat/repl.py

REPL management for chat sessions.
"""

import os
import torch
from pathlib import Path
from rich.console import (
    Console,
    Group,
)
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.live import Live

import config
from .infer import (
    count_tokens,
    stream_generate,
)
from .logging import (
    build_log_path,
    log_chat,
    log_turn,
)
from corpus.dream_sequence import run_dream
from model.loader import load_model
from train.dream_train import run_dream_training
from train.chat_train import run_chat_cleanup_and_training


console = Console()


# ── Initialization ─────────────────────────────────────────────────────────


def load_model_state():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print("[yellow]Loading model...[/yellow]")
    model, tokenizer = load_model(config.CHECKPOINT_PATH, device)
    console.print("[green]Model loaded.[/green]\n")
    return model, tokenizer, device


def initialize(title):
    """
    Shared initialization.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]{config.MODEL_NAME} {title}[/bold cyan]\n"
            f"Device: [yellow]{device}[/yellow]",
            border_style="cyan",
        )
    )
    console.print("[dim]Type your prompt and press Enter. Ctrl+C to exit.[/dim]\n")
    return load_model_state()


# ── Prompt formatting ─────────────────────────────────────────────────────────


def format_prompt(text: str) -> str:
    return f"[{config.USER_NAME}] {text}\n[{config.MODEL_NAME}]"


def prompt_user() -> str:
    user_text = Prompt.ask(f"[bold blue]{config.USER_NAME}[/bold blue]")
    return user_text.strip()


# ── Output formatting ─────────────────────────────────────────────────────────


def stream_display(model, tokenizer, prompt, device):
    spinner = Spinner("dots", text="Generating...")
    response_chunks = []
    printed = ""
    panel_text = ""

    with Live(
        Panel(
            Group(spinner, panel_text),
            title=f"[bold green]{config.MODEL_NAME}[/bold green]",
            border_style="green",
        ),
        console=console,
        refresh_per_second=12,
    ) as live:
        for text in stream_generate(model, tokenizer, prompt, device):
            delta = text[len(printed):]
            printed = text

            if delta:
                response_chunks.append(delta)
                panel_text += delta

            live.update(
                Panel(
                    Group(spinner, panel_text.strip()),
                    title=f"[bold green]{config.MODEL_NAME}[/bold green]",
                    border_style="green",
                )
            )

        # Final render without spinner
        live.update(
            Panel(
                panel_text.strip(),
                title=f"[bold green]{config.MODEL_NAME}[/bold green]",
                border_style="green",
            )
        )
    console.print()
    output = "".join(response_chunks)
    return output


# ── Chat REPL (called from main.py) ───────────────────────────────────────────


def run_chat_repl_one_shot():
    model, tokenizer, device = initialize("One-Shot Interactive Chat")

    try:
        while True:
            user_text = prompt_user()
            if not user_text:
                continue

            prompt = format_prompt(user_text)
            output = stream_display(model, tokenizer, prompt, device)
            log_chat(user_text, output)

    except KeyboardInterrupt:
        console.print("\n[bold red]Exiting chat.[/bold red]\n")


def generate_dpo_pair(user_text, last_prompt, last_response):
    correction = user_text[len(":correction"):].strip()

    if not correction:
        console.print("[yellow]Correction text missing.[/yellow]")
        return

    if last_prompt is None or last_response is None:
        console.print("[yellow]No previous exchange to correct.[/yellow]")
        return

    entry = {
        "prompt": last_prompt,
        "chosen": correction,
        "rejected": last_response,
        "notes": None,
        "session": "",
        "turn": 0,
    }

    with open(config.DPO_PAIRS_PATH, "a", encoding="utf-8") as f:
        import json
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    console.print("[green]DPO correction saved.[/green]\n")


def begin_dreaming(log_file: str):
    console.print(
        "\n[bold yellow]Dreaming about the events of the day...[/bold yellow]\n"
    )

    dream_path = run_dream(
        chat_log_path=log_file,
        endpoint=os.environ.get("AZURE_MISTRAL_ENDPOINT"),
        api_key=os.environ.get("AZURE_MISTRAL_KEY"),
        voice_file=config.VOICE_FILE,
        output_dir="../data/dreams",
    )

    if dream_path is not None:
        run_dream_training(
            Path("../data/dreams"),
            config.CHECKPOINT_PATH
        )

        console.print("[dim]Cleaning chat log for training...[/dim]\n")

        run_chat_cleanup_and_training(
            chat_log_path=log_file,
            endpoint=os.environ.get("AZURE_MISTRAL_ENDPOINT"),
            api_key=os.environ.get("AZURE_MISTRAL_KEY"),
            checkpoint_path=config.CHECKPOINT_PATH,
        )

    console.print(
        "\n[bold yellow]It's time to wake up.[/bold yellow]\n"
    )


def run_chat_repl():
    model, tokenizer, device = initialize("Conversational Interactive Chat")

    conversation_history = []
    conversation_tokens = 0

    log_file = build_log_path()
    console.print(f"[dim]Logging conversation → {log_file.name}[/dim]\n")

    dpo_pairs_path = config.DPO_PAIRS_PATH
    dpo_pairs_path.parent.mkdir(parents=True, exist_ok=True)

    last_prompt = None
    last_response = None

    try:
        while True:
            user_text = prompt_user()
            if not user_text:
                continue

            # ── DPO correction command ─────────────────────

            if user_text.startswith(":correction"):
                generate_dpo_pair(user_text, last_prompt, last_response)
                continue

            # ── Normal conversation turn ───────────────────

            user_turn = f"[{config.USER_NAME}] {user_text}\n"

            prompt = "".join(conversation_history) + f"[{config.MODEL_NAME}]"

            conversation_history.append(user_turn)

            output = stream_display(model, tokenizer, prompt, device)

            scout_turn = f"[{config.MODEL_NAME}] {output}\n"

            conversation_history.append(scout_turn)

            log_turn(log_file, user_text, output)

            # Track last exchange for corrections
            last_prompt = "".join(conversation_history[:-1])
            last_response = output

            # ── Context accounting ─────────────────────────

            full_text = "".join(conversation_history)

            conversation_tokens = count_tokens(tokenizer, full_text)

            console.print(
                f"[dim]Context: {conversation_tokens}/{config.BLOCK_SIZE} tokens[/dim]"
            )

            # ── Context overflow handling ──────────────────

            if conversation_tokens >= config.BLOCK_SIZE:
                console.print(
                    "\n[bold yellow]Context window full.[/bold yellow]\n"
                )

                begin_dreaming(log_file)
                
                console.print(
                    "\n[bold green]Starting new conversation.[/bold green]\n"
                )

                model, tokenizer, device = load_model_state()

                conversation_history = []
                conversation_tokens = 0

                log_file = build_log_path()

                console.print(
                    f"[dim]New log file → {log_file.name}[/dim]\n"
                )

    except KeyboardInterrupt:
        console.print("\n[bold red]Exiting chat.[/bold red]\n")
