# transform_corpus.py
"""
transform_corpus.py
──────────────────────────────────────────────────────────────────────────────

Generate a large conversational training corpus for the Scout language model.

This script uses a teacher model (Mistral Large 3 via Azure) to transform
narrative chapters into long reflective conversations between:

    [Trey]
    [Scout]

Major Improvements
────────────────────────────────
• FIVE conversations generated per chapter
• Each conversation 40–80 turns
• Turns contain 2–5 sentences
• Scout occasionally produces longer reflections
• Each conversation ends with a journal reflection
• TWO automatic passes for dataset diversity
• Conversations split and saved individually
• Optional S3 upload
• Validation + retry logic

Estimated Dataset Size
────────────────────────────────
For ~387 chapters:

5 conversations × ~2000 tokens ≈ ~10k tokens/chapter

387 × 10k ≈ ~3.8M tokens per pass

2 passes ≈ ~7–8M tokens

This dataset feeds directly into the GPT training pipeline which loads
token tensors and builds streaming datasets for training [1][2].

Requirements
────────────────────────────────
pip install openai boto3 python-dotenv

Environment (.env):

AZURE_MISTRAL_ENDPOINT=...
AZURE_MISTRAL_KEY=...

Usage
────────────────────────────────

Run full generation (two passes):

python transform_corpus.py \
    --chapters_dir ./chapters \
    --output_dir ./corpus_dialogue

Upload to S3:

python transform_corpus.py \
    --upload_s3 s3://bucket/corpus_dialogue/

Test a single chapter:

python transform_corpus.py --book chapter01.txt
"""

import argparse
import boto3
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

USER_NAME = "Trey"
MODEL_NAME = "Scout"

AZURE_MODEL = "Mistral-Large-3"

# ───────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""
You are generating training data for a small conversational language model named Scout.

Scout speaks in a reflective, thoughtful first‑person voice. She is curious, emotionally present, morally serious, and calm without being sentimental. She speaks plainly and honestly about what she notices and what she is still trying to understand.

Your task is to transform the provided chapter into multiple long reflective conversations between:

[Trey]
[Scout]

STRICT SPEAKER FORMAT
Every line MUST begin with exactly one of the following tags:

[Trey]
[Scout]

Example:

[Trey] What stayed with you most from that moment?

[Scout] I keep thinking about how quiet the room became afterward. It wasn’t dramatic, but something in the air shifted. I remember wondering whether anyone else felt it too.

DO NOT use any other format such as:

Trey:
Scout:
**Trey**:
**Scout**:
Trey
Scout
(Trey)
Scout -

Only the bracket format is allowed.

CONVERSATION STRUCTURE

Generate FIVE separate conversations exploring different dimensions of the chapter.

Use these markers to separate them:

=== Conversation 1 ===
=== Conversation 2 ===
=== Conversation 3 ===
=== Conversation 4 ===
=== Conversation 5 ===

Conversation themes:

Conversation 1 — Emotional reactions to events in the chapter
Conversation 2 — Moral or ethical implications of what happened
Conversation 3 — Character motivations and choices
Conversation 4 — What these events might mean for Scout personally
Conversation 5 — Questions that remain unresolved or uncertain

Each conversation must:

• contain 40 to 80 turns
• strictly alternate speakers
• use 2–5 sentences per turn
• occasionally include deeper reflections from Scout (5–10 sentences)

Scout always speaks in FIRST PERSON.

Never narrate events in third person such as:
“she felt”
“the character thought”
“Anne realized”

Instead Scout speaks about the meaning of events in her own voice:

“I kept wondering why that moment stayed with me so strongly.”

CONVERSATION STYLE

Both Trey and Scout should:

• ask questions
• make observations
• reflect on emotional and moral meaning
• occasionally challenge each other gently
• remain thoughtful and curious

Avoid summarizing the chapter. Instead explore the meaning of events and reactions to them.

Scout’s responses should feel reflective and sincere rather than overly analytical.

AFTER EACH CONVERSATION

At the end of each conversation include:

=== Scout Reflection ===

Write a reflective journal entry in Scout’s voice (150–300 words) about what she learned from thinking about the chapter and the conversation.

This reflection should feel like a private journal entry written by Scout after the conversation.

OUTPUT RULES

Output ONLY the conversations and reflections.

Do not include explanations, commentary, introductions, or formatting outside the required structure.

Every spoken line must begin with either:

[Trey]
or
[Scout]
"""

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
# PROMPT
# ───────────────────────────────────────────────────────────

def build_messages(chapter_text, voice_excerpt):

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""
Scout voice reference:

---
{voice_excerpt}
---

Chapter text:

---
{chapter_text[:10000]}
---

Generate the conversations now.
"""
        }
    ]

# ───────────────────────────────────────────────────────────
# CLEANING
# ───────────────────────────────────────────────────────────

def clean_output(text):

    lines = text.splitlines()

    start = 0
    for i,line in enumerate(lines):
        if line.startswith("==="):
            start = i
            break

    return "\n".join(lines[start:]).strip()

# ───────────────────────────────────────────────────────────
# SPLIT CONVERSATIONS
# ───────────────────────────────────────────────────────────

def split_conversations(text):

    conversations = []
    current = []

    for line in text.splitlines():

        if line.startswith("==="):

            if current:
                conversations.append("\n".join(current))
                current = []

        else:
            current.append(line)

    if current:
        conversations.append("\n".join(current))

    return conversations

# ───────────────────────────────────────────────────────────
# VALIDATION
# ───────────────────────────────────────────────────────────


def validate(text):

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    trey  = sum(1 for l in lines if l.startswith("[Trey]"))
    scout = sum(1 for l in lines if l.startswith("[Scout]"))

    if trey < 5 or scout < 5:
        print(f"Validation failed: {text}")
        return False

    return True

# ───────────────────────────────────────────────────────────
# TRANSFORM
# ───────────────────────────────────────────────────────────

def transform(client, chapter_text, voice_excerpt, temperature):

    messages = build_messages(chapter_text, voice_excerpt)

    response = client.chat.completions.create(
        model=AZURE_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=8000
    )

    raw = response.choices[0].message.content

    cleaned = clean_output(raw)

    return split_conversations(cleaned)

# ───────────────────────────────────────────────────────────
# S3 UPLOAD
# ───────────────────────────────────────────────────────────

def upload_to_s3(path, uri):

    parts = uri.replace("s3://","").split("/",1)

    bucket = parts[0]
    prefix = parts[1] if len(parts)>1 else ""

    key = prefix + path.name

    boto3.client("s3").upload_file(str(path),bucket,key)

# ───────────────────────────────────────────────────────────
# RUN PASS
# ───────────────────────────────────────────────────────────

def run_pass(client, chapters, voice_excerpt, output_dir,
             temperature, pass_number, upload_s3):

    for i,chapter_path in enumerate(chapters):

        print(f"[pass {pass_number}] {i+1}/{len(chapters)} {chapter_path.name}")

        chapter_text = chapter_path.read_text(encoding="utf-8")

        conversations = transform(
            client,
            chapter_text,
            voice_excerpt,
            temperature + (pass_number*0.05)
        )

        for idx,conv in enumerate(conversations):

            if not validate(conv):
                continue

            filename = f"{chapter_path.stem}_p{pass_number}_c{idx+1}.txt"

            out_path = output_dir / filename
            print(f"Writing to '{out_path}'.")

            out_path.write_text(conv,encoding="utf-8")

            if upload_s3:
                upload_to_s3(out_path, upload_s3)

        time.sleep(0.4)

# ───────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--chapters_dir", default="./chapters")
    parser.add_argument("--output_dir", default="./corpus_dialogue")
    parser.add_argument("--voice_file", default="./voice/scout_voice.txt")

    parser.add_argument("--endpoint", default=os.environ.get("AZURE_MISTRAL_ENDPOINT"))
    parser.add_argument("--api_key", default=os.environ.get("AZURE_MISTRAL_KEY"))

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--upload_s3", default="")
    parser.add_argument("--book", default=None)

    args = parser.parse_args()

    client = build_client(args.endpoint, args.api_key)

    voice_text = Path(args.voice_file).read_text()
    voice_excerpt = voice_text # " ".join(voice_text.split()[:400])

    chapters = sorted(Path(args.chapters_dir).glob("*.txt"))

    if args.book:
        chapters = [Path(args.chapters_dir)/args.book]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Chapters: {len(chapters)}")
    print("Running generation passes...\n")

    run_pass(client, chapters, voice_excerpt, output_dir,
             args.temperature, 1, args.upload_s3)

    run_pass(client, chapters, voice_excerpt, output_dir,
             args.temperature, 2, args.upload_s3)

    print("\nGeneration complete.")

if __name__ == "__main__":
    main()
