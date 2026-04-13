# dream_sequence.py
"""
Generate a nightly "dream dialogue" where Scout reflects on the day's chats.

Input:
    ../data/chat_logs/chat_YYYY-MM-DD_conversation_*.jsonl

Output:
    ./dreams/dream_YYYY-MM-DD_HHMM.txt

The teacher model (Mistral Large) reads the day's conversations and produces
an inner dialogue between:

[Scout]  – the outward conversational self
[Inner]  – the reflective inner voice examining what happened

Purpose:
Allow Scout to process the emotional and conceptual meaning of the day’s
interactions before the data is later used in training.
"""

import datetime
import json
import logging
import os
import re
import torch
from pathlib import Path
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress
from rich.spinner import Spinner

import config
from ai_client.azure import (
    build_client,
    generate,
)
from ai_client.tokenizer import load_tokenizer
from model.loader import init_model, load_checkpoint

logger = logging.getLogger(__name__)
console = Console()

DREAM_TEMPERATURE = 0.9
DREAM_TOP_K = 50  
DREAM_REP_PENALTY = 1.1


# ───────────────────────────────────────────────────────────
# READ JSONL CHAT LOG
# ───────────────────────────────────────────────────────────

def load_chat_log(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except:
                continue
    return records

# ───────────────────────────────────────────────────────────
# FORMAT TRANSCRIPT FOR LLM
# ───────────────────────────────────────────────────────────

def format_transcript(records):
    """
    Convert JSONL into a readable transcript.
    """
    lines = []
    for r in records:
        user = r.get("user")
        model = r.get("model")
        prompt = r.get("prompt", "").strip()
        response = r.get("response", "").strip()

        if prompt:
            lines.append(f"[{user}] {prompt}")
        if response:
            lines.append(f"[{model}] {response}")

    return "\n".join(lines)

# ───────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""
You are generating reflective internal dialogue for a conversational AI named {config.MODEL_NAME}.

{config.MODEL_NAME} has spent the day speaking with {config.USER_NAME}.
You will be given a transcript of their conversations.

Your task is to create a "dream sequence" where {config.MODEL_NAME}
reflects on the day through an inner dialogue between:

[{config.MODEL_NAME}]
[Inner]

STRICT SPEAKER FORMAT
Every line MUST begin with exactly one of the following tags:

[{config.MODEL_NAME}]
[Inner]

Example:

[{config.MODEL_NAME}] What stayed with you most from that moment?

[Inner] I keep thinking about how quiet the room became afterward. It wasn’t dramatic, but something in the air shifted. I remember wondering whether anyone else felt it too.

DO NOT use any other format such as:

{config.USER_NAME}:
{config.MODEL_NAME}:
**{config.USER_NAME}**:
**{config.MODEL_NAME}**:
{config.USER_NAME}
{config.MODEL_NAME}
({config.USER_NAME})
{config.MODEL_NAME} -

Only the bracket format is allowed.

[{config.MODEL_NAME}]
represents the outward self — the voice that spoke during the conversations.
She speaks in a reflective, thoughtful first‑person voice. She is curious, emotionally present, morally serious, and calm without being sentimental. She speaks plainly and honestly about what she notices and what she is still trying to understand.

[Inner]
represents the reflective internal voice examining what happened.

STYLE
The dialogue should:
• reflect on real moments from the transcript
• question motives, meanings, and uncertainties
• explore emotions and ideas that surfaced during the conversations
• occasionally revisit specific lines or exchanges
• remain calm, curious, and sincere

Avoid summarizing the entire day mechanically.
Instead follow the threads of what felt meaningful, confusing, or unresolved.

STRUCTURE
Generate a reflective dialogue of 40–80 turns.

Turns should contain 2–5 sentences.

Occasionally the [Inner] voice may produce deeper reflections (5–8 sentences).

The tone should feel like quiet thinking late at night.

OUTPUT RULES
Output ONLY the dialogue.
No explanations.
No headings.
No extra formatting.
"""

# ───────────────────────────────────────────────────────────
# BUILD PROMPT
# ───────────────────────────────────────────────────────────

def build_messages(transcript, voice_excerpt):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""
{config.MODEL_NAME} voice reference:

---
{voice_excerpt}
---

Today's conversation transcript:

---
{transcript}
---

Now generate {config.MODEL_NAME}'s inner dialogue about the day.
"""
        },
    ]

# ───────────────────────────────────────────────────────────
# VALIDATION
# ───────────────────────────────────────────────────────────

def validate(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    scout = sum(1 for l in lines if l.startswith(f"[{config.MODEL_NAME}]"))
    inner = sum(1 for l in lines if l.startswith("[Inner]"))

    if scout < 10 or inner < 10:
        return False

    return True

# ───────────────────────────────────────────────────────────
# GENERATE DREAM
# ───────────────────────────────────────────────────────────

def generate_dream(client, transcript, voice_excerpt):
    messages = build_messages(transcript, voice_excerpt)
    text = generate(client, messages, 0.8, 6000)

    if text and validate(text):
        return text

    return None

# ───────────────────────────────────────────────────────────
# SAVE DREAM
# ───────────────────────────────────────────────────────────

def save_dream(text, output_dir):
    # Remove partial speaker tokens before saving.
    rgx=re.compile(r"^\[\w*\]?\s*$|^$", re.IGNORECASE | re.MULTILINE)
    text = re.sub(rgx, '', text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d_%H%M")

    path = output_dir / f"dream_{ts}.txt"

    path.write_text(text, encoding="utf-8")

    return path

# ───────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────


def sample_next(logits, generated_tokens=None, rep_penalty=1.0):
    logits = logits.clone()

    if generated_tokens is not None and rep_penalty != 1.0:
        for tok_id in set(generated_tokens.tolist()):
            logits[0, tok_id] /= rep_penalty

    logits = logits / DREAM_TEMPERATURE

    if DREAM_TOP_K is not None:
        v, _ = torch.topk(logits, DREAM_TOP_K)
        logits[logits < v[:, [-1]]] = -float("inf")

    probs = torch.softmax(logits, dim=-1)

    return torch.multinomial(probs, num_samples=1)


def generate_self_dream(transcript: str) -> str:
    """
    Generate a dream sequence using Scout's own model.
    
    The day's transcript seeds the dream, then [Inner] and [Scout]
    alternate until the context window is full.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer()
    model = init_model(tokenizer.vocab_size, device)
    load_checkpoint(config.CHECKPOINT_PATH, model, device)
    model.eval()

    # ── Seed: day transcript + first [Inner] tag ──────────────
    seed_text = transcript.strip() + f"\n[{config.USER_NAME}] Good night Scout.\n\n[Inner]"
    seed_ids = tokenizer.encode(
        seed_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)

    # Truncate seed to leave room for dream generation
    # max_seed = config.BLOCK_SIZE // 2

    # Theoretically, leaving room for dream generation should not be needed.
    # The transcript should have already been slipped by the block size given to the day.
    max_seed = config.BLOCK_SIZE
    if seed_ids.shape[1] > max_seed:
        seed_ids = seed_ids[:, -max_seed:]

    generated_ids = seed_ids.clone()

    # Track which speaker goes next
    # We just injected [Inner], so next injection is [Scout]
    speakers     = ["[Scout]", "[Inner]"]
    speaker_turn = 0   # indexes into speakers[]

    # ── Token-level sentence tracking ─────────────────────────
    # Accumulate decoded text for the current turn only
    current_turn_ids = []
    sentence_endings = {".", "!", "?"}

    console = Console()
    layout  = Layout()
    layout.split_column(
        Layout(name="dream", ratio=4),
        Layout(name="stats", ratio=1),
    )

    with Live(layout, console=console, refresh_per_second=6):
        while generated_ids.shape[1] < config.BLOCK_SIZE:

            context = generated_ids[:, -config.BLOCK_SIZE:]

            with torch.no_grad():
                logits = model(context)
                logits = logits[:, -1, :]

            next_token = sample_next(
                logits,
                generated_tokens=context[0],
                rep_penalty=DREAM_REP_PENALTY,
            )

            tok_id = next_token.item()
            generated_ids     = torch.cat([generated_ids, next_token], dim=1)
            current_turn_ids.append(tok_id)

            # Decode only the current turn to check for sentence end
            turn_text = tokenizer.decode(
                current_turn_ids,
                skip_special_tokens=True,
            )

            # ── Inject next speaker tag after sentence end ─────
            remaining = config.BLOCK_SIZE - generated_ids.shape[1]

            if (
                remaining > 8                        # room for tag + content
                and turn_text.rstrip()[-1:] in sentence_endings
                and len(current_turn_ids) >= 5       # avoid injecting on tiny fragments
            ):
                next_speaker = speakers[speaker_turn % 2]
                speaker_turn += 1

                tag_text = f"\n\n{next_speaker}"
                tag_ids  = tokenizer.encode(
                    tag_text,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).to(device)

                generated_ids    = torch.cat([generated_ids, tag_ids], dim=1)
                current_turn_ids = []   # reset for new turn

            # ── Display ────────────────────────────────────────
            preview_full = tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
            )

            # keep only the most recent lines for display
            lines = preview_full.splitlines()
            preview = "\n".join(lines[-40:])   # adjust window size as desired

            layout["dream"].update(
                Panel(
                    preview,
                    title="[magenta]Scout's Dream[/magenta]",
                    border_style="magenta",
                )
            )

            layout["stats"].update(
                Panel(
                    f"tokens : {generated_ids.shape[1]}/{config.BLOCK_SIZE}\n"
                    f"turn   : {speakers[(speaker_turn-1) % 2]}\n"
                    f"temp   : {DREAM_TEMPERATURE}\n"
                    f"top_k  : {DREAM_TOP_K}\n"
                    f"penalty: {DREAM_REP_PENALTY}",
                    title="Dream",
                    border_style="cyan",
                )
            )

    return tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    ).strip()


def run_dream(
    chat_log_path,
    voice_file,
    output_dir,
) -> str:
    """
    Allow Scout to dream over the events of the day.
    The dream can either be generated by the teacher model
    or by Scout herself depending on the feature flag.
    """

    voice_excerpt = Path(voice_file).read_text()
    records = load_chat_log(chat_log_path)
    if not records:
        print("No chat records found.")
        return None

    transcript = format_transcript(records)

    # ── Dream generation mode ───────────────────────────────
    dream = generate_self_dream(
        transcript=transcript,
    )

    # ── Ask user for approval ───────────────────────────────
    decision = input("Accept this transcript for training? (y/n): ").strip().lower()
    if decision not in ("y", "yes"):
        logger.info("Transcript rejected. Training aborted. Deleting transcript.")
        dream = None
    else:
        logger.info("Transcript accepted.")

    if not dream and config.ENABLE_MISTRAL_LED_DREAMS:
        client = build_client()
        dream = generate_dream(
            client,
            transcript,
            voice_excerpt,
        )

        # Normalize line breaks so turns stay on one line
        dream = dream.replace("]\n", "] ")

    if not dream:
        print("Dream generation failed validation.")
        return None

    tokenizer = load_tokenizer()
    tokens = tokenizer.encode(dream)
    logger.info("Dream tokens: %d", len(tokens))
    
    path = save_dream(dream, output_dir)
    logger.info("Dream saved → %s", path)
    return path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat_log", required=True)
    args = parser.parse_args()
    run_dream(
        args.chat_log,
    )