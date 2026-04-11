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
import os
from pathlib import Path
from openai import AzureOpenAI

import config


# ───────────────────────────────────────────────────────────
# CLIENT
# ───────────────────────────────────────────────────────────

def build_client(endpoint, api_key):
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2024-05-01-preview",
    )

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

    response = client.chat.completions.create(
        model=os.getenv("AZURE_MODEL_ID"),
        messages=messages,
        temperature=0.8,
        max_tokens=6000,
    )

    text = response.choices[0].message.content

    if text and validate(text):
        return text

    return None

# ───────────────────────────────────────────────────────────
# SAVE DREAM
# ───────────────────────────────────────────────────────────

def save_dream(text, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d_%H%M")

    path = output_dir / f"dream_{ts}.txt"

    path.write_text(text, encoding="utf-8")

    return path

# ───────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────

def run_dream(
    chat_log_path,
    endpoint,
    api_key,
    voice_file,
    output_dir,
) -> str:
    """
    Allow Scout to dream over the events of the day.
    """
    voice_excerpt = Path(voice_file).read_text()
    client = build_client(endpoint, api_key)

    records = load_chat_log(chat_log_path)

    if not records:
        print("No chat records found.")
        return None

    transcript = format_transcript(records)

    dream = generate_dream(client, transcript, voice_excerpt)

    if not dream:
        print("Dream generation failed validation.")
        return None

    dream = dream.replace("]\n", "] ")

    path = save_dream(dream, output_dir)
    print(f"Dream saved → {path}")
    return path


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--chat_log", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--api_key", required=True)

    args = parser.parse_args()

    run_dream(
        args.chat_log,
        args.endpoint,
        args.api_key
    )