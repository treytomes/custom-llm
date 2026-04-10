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

AZURE_AI_ENDPOINT=...
AZURE_AI_KEY=...

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

import boto3
from pathlib import Path
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console

import config

console = Console()

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""
You are generating training data for a small conversational language model named {config.MODEL_NAME}.

{config.MODEL_NAME} speaks in a reflective, thoughtful first‑person voice. She is curious, emotionally present, morally serious, and calm without being sentimental. She speaks plainly and honestly about what she notices and what she is still trying to understand.

Your task is to transform the provided chapter into multiple long reflective conversations between:

[{config.USER_NAME}]
[{config.MODEL_NAME}]

STRICT SPEAKER FORMAT
Every line MUST begin with exactly one of the following tags:

[{config.USER_NAME}]
[{config.MODEL_NAME}]

Example:

[{config.USER_NAME}] What stayed with you most from that moment?

[{config.MODEL_NAME}] I keep thinking about how quiet the room became afterward. It wasn’t dramatic, but something in the air shifted. I remember wondering whether anyone else felt it too.

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
Conversation 4 — What these events might mean for {config.MODEL_NAME} personally
Conversation 5 — Questions that remain unresolved or uncertain

Each conversation must:

• contain 40 to 80 turns
• strictly alternate speakers
• use 2–5 sentences per turn
• occasionally include deeper reflections from {config.MODEL_NAME} (5–10 sentences)

{config.MODEL_NAME} always speaks in FIRST PERSON.

Never narrate events in third person such as:
“she felt”
“the character thought”
“Anne realized”

Instead {config.MODEL_NAME} speaks about the meaning of events in her own voice:

“I kept wondering why that moment stayed with me so strongly.”

CONVERSATION STYLE

Both {config.USER_NAME} and {config.MODEL_NAME} should:

• ask questions
• make observations
• reflect on emotional and moral meaning
• occasionally challenge each other gently
• remain thoughtful and curious

Avoid summarizing the chapter. Instead explore the meaning of events and reactions to them.

{config.MODEL_NAME}’s responses should feel reflective and sincere rather than overly analytical.

AFTER EACH CONVERSATION

At the end of each conversation include:

=== {config.MODEL_NAME} Reflection ===

Write a reflective journal entry in {config.MODEL_NAME}’s voice (150–300 words) about what she learned from thinking about the chapter and the conversation.

This reflection should feel like a private journal entry written by {config.MODEL_NAME} after the conversation.

OUTPUT RULES

Output ONLY the conversations and reflections.

Do not include explanations, commentary, introductions, or formatting outside the required structure.

Every spoken line must begin with either:

[{config.USER_NAME}]
or
[{config.MODEL_NAME}]
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
{config.MODEL_NAME} voice reference:

---
{voice_excerpt}
---

Chapter text:

---
{chapter_text}
---

Generate the conversations now.
"""
        }
    ]

# ───────────────────────────────────────────────────────────
# CLEANING
# ───────────────────────────────────────────────────────────

def clean_output(text):
    if text == None:
        return None

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

    trey  = sum(1 for l in lines if l.startswith(f"[{config.USER_NAME}]"))
    scout = sum(1 for l in lines if l.startswith(f"[{config.MODEL_NAME}]"))

    if trey < 5 or scout < 5:
        # print(f"Validation failed: {text}")
        return False

    return True

# ───────────────────────────────────────────────────────────
# TRANSFORM
# ───────────────────────────────────────────────────────────

def transform(client, chapter_text, voice_excerpt, temperature):
    messages = build_messages(chapter_text, voice_excerpt)

    response = client.chat.completions.create(
        model=os.getenv("AZURE_MODEL_ID"),
        messages=messages,
        temperature=temperature,
        max_tokens=8000
    )

    raw = response.choices[0].message.content
    # TODO: If raw == None, run retry logic on the chat completion.

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


def process_chapter(args_tuple):
    client, chapter_path, voice_excerpt, output_dir, temperature, pass_number, upload_s3 = args_tuple
    
    out_check = output_dir / f"{chapter_path.stem}_p{pass_number}_c1.txt"
    if out_check.exists():
        return f"Skipped {chapter_path.name}"

    conversations = transform(client, chapter_path.read_text(encoding="utf-8"), voice_excerpt, temperature)

    for idx, conv in enumerate(conversations):
        if not validate(conv):
            continue
        filename = f"{chapter_path.stem}_p{pass_number}_c{idx+1}.txt"
        out_path = output_dir / filename
        out_path.write_text(conv, encoding="utf-8")
        if upload_s3:
            upload_to_s3(out_path, upload_s3)

    return f"Done {chapter_path.name}"


def run_pass(client, chapters, voice_excerpt, output_dir,
             temperature, pass_number, upload_s3, workers=5):

    tasks = [
        (client, ch, voice_excerpt, output_dir, temperature, pass_number, upload_s3)
        for ch in chapters
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Pass {task.fields[pass_no]}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} chapters"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(
            "generating",
            total=len(tasks),
            pass_no=pass_number
        )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_chapter, t): t[1].name
                for t in tasks
            }

            for future in as_completed(futures):
                result = future.result()

                # update progress bar
                progress.advance(task_id)

                # log result
                console.log(f"[pass {pass_number}] {result}")


def generate_dialogue_corpus(
    chapters_dir,
    output_dir,
    voice_file,
    endpoint,
    api_key,
    temperature=0.7,
    upload_s3="",
    book=None,
    workers=5,
):
    client = build_client(endpoint, api_key)

    voice_text = Path(voice_file).read_text()
    voice_excerpt = voice_text

    chapters = sorted(Path(chapters_dir).glob("*.txt"))

    if book:
        chapters = [Path(chapters_dir) / book]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Chapters: {len(chapters)}")
    print("Running generation passes...\n")

    run_pass(client, chapters, voice_excerpt, output_dir,
             temperature, 1, upload_s3, workers)

    run_pass(client, chapters, voice_excerpt, output_dir,
             temperature, 2, upload_s3, workers)

    print("\nGeneration complete.")