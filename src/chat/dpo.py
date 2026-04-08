"""
chat/dpo.py

Interactive DPO data collection.
"""

import time
import torch
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

import config
from .infer import load_model, generate
from .logging import build_log_path

console = Console()


def choose_index(prompt, max_i):
    while True:
        raw = Prompt.ask(prompt).strip()

        if raw == "":
            return None

        if raw.isdigit():
            i = int(raw)
            if 0 <= i < max_i:
                return i

        console.print(f"[red]Enter a number between 0 and {max_i-1}[/red]")


def run_dpo_repl():
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]{config.MODEL_NAME} DPO Collection[/bold cyan]\n"
            f"Device: [yellow]{device}[/yellow]",
            border_style="cyan",
        )
    )

    console.print("[dim]Ctrl+C to exit[/dim]\n")

    model, tokenizer = load_model(config.CHECKPOINT_PATH, device)
    log_file = build_log_path(prefix="dpo")
    console.print(f"[dim]Logging → {log_file.name}[/dim]\n")

    turn = 0
    while True:
        user_text = Prompt.ask(f"[bold blue]{config.USER_NAME}[/bold blue]").strip()
        if not user_text:
            continue

        user_text = user_text.replace("\\n", "\n")
        prompt = f"[{config.USER_NAME}] {user_text}\n[{config.MODEL_NAME}]"
        console.print()

        candidates = []
        t0 = time.time()

        temperature_sweep = [0.55, 0.65, 0.75, 0.85]
        topk_sweep = [20, 40, 60, 80]
        rep_penalty_sweep = [1.05, 1.1, 1.15, 1.2]

        for i in range(config.DPO_SAMPLE_COUNT):
            torch.manual_seed(time.time_ns() % 2**32)

            # Select sampling parameters
            config.TEMPERATURE = temperature_sweep[i % len(temperature_sweep)]
            config.TOP_K = topk_sweep[i % len(topk_sweep)]
            config.REP_PENALTY = rep_penalty_sweep[i % len(rep_penalty_sweep)]

            out = generate(model, tokenizer, prompt, device)

            response = (
                out[len(prompt):].strip()
                if out.startswith(prompt)
                else out.strip()
            )

            candidates.append(response)

            console.print(
                Panel(
                    f"[dim]T={config.TEMPERATURE}  K={config.TOP_K}[/dim]\n\n{response}",
                    title=f"Candidate {i}",
                    border_style="yellow",
                )
            )

        elapsed = time.time() - t0

        console.print(f"[dim]{config.DPO_SAMPLE_COUNT} candidates | {elapsed:.1f}s[/dim]\n")

        best = choose_index("Best candidate #", config.DPO_SAMPLE_COUNT)
        worst = choose_index("Worst candidate #", config.DPO_SAMPLE_COUNT)

        correction = Prompt.ask(
            "Better response (optional)",
            default="",
        ).strip()

        notes = Prompt.ask(
            "Notes (optional)",
            default="",
        ).strip()

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "prompt": user_text,
            "candidates": candidates,
            "chosen": candidates[best] if best is not None else None,
            "rejected": candidates[worst] if worst is not None else None,
            "correction": correction or None,
            "notes": notes or None,
        }

        with open(log_file, "a", encoding="utf-8") as f:
            import json
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        turn += 1
        console.print()
