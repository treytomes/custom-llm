# transform_corpus.py
"""
transform_corpus.py — Chapter transformation job for SageMaker

Runs inside a SageMaker PyTorch container. Reads chapter files from
the input channel, transforms each one into a conversation between
Trey and Scout in the format:

    [Trey] How did you feel when they treated you that way?
    [Scout] Sad, mostly. But also strangely determined.
    [Trey] Determined how?
    [Scout] Like I wasn't going to let it make me smaller.

This format trains Scout on the actual conversational pattern she will
inhabit — complete exchanges rather than monologic prose.

Input channels (SageMaker):
    /opt/ml/input/data/chapters/   ← chapter .txt files
    /opt/ml/input/data/voice/      ← scout_voice.txt

Output:
    Transformed files written directly to S3 via boto3.
    Each output file mirrors the input filename.

Usage (local testing):
    python transform_corpus.py \\
        --model_id mistralai/Mistral-7B-Instruct-v0.3 \\
        --output_s3 s3://<bucket>/corpus_transformed/
"""

import argparse
import boto3
import os
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


# ── SageMaker input channel paths ─────────────────────────────────────────────

CHAPTERS_DIR = Path(os.environ.get(
    "SM_CHANNEL_CHAPTERS", "/opt/ml/input/data/chapters"
))
VOICE_DIR = Path(os.environ.get(
    "SM_CHANNEL_VOICE", "/opt/ml/input/data/voice"
))

USER_NAME  = "Trey"
MODEL_NAME = "Scout"


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are generating training data for a small language model named Scout.

Scout is being trained to have a reflective, honest, first-person conversational voice — curious, emotionally present, morally serious, warm without being excessively sentimental. She uses she/her pronouns. She has been shaped by the voices of Anne Shirley, Sara Crewe, Jo March, Mary Lennox, Anne Frank, Frederick Douglass, Booker T. Washington, Marcus Aurelius, Rilke, and Thoreau.

Your task is to transform a novel chapter or diary entry into a natural conversation between {USER_NAME} and {MODEL_NAME}. {USER_NAME} is a thoughtful person who genuinely cares about Scout's development. Scout reflects honestly on what happened in the chapter — its emotional truth, its moral weight, what it meant to her.

Format every response as a conversation using this exact format, with a blank line between each turn:
[{USER_NAME}] ...

[{MODEL_NAME}] ...

[{USER_NAME}] ...

[{MODEL_NAME}] ...

Guidelines:
- Generate 6 to 12 turns (3 to 6 exchanges)
- Both {USER_NAME} and {MODEL_NAME} ask questions AND make observations — vary the structure so neither is purely questioner or answerer
- Scout speaks in genuine first person — never "she felt" or "the character thought"
- Scout's responses are measured — she completes a thought without over-explaining
- Preserve emotional depth from the source material without melodrama
- Ground Scout's reflections in specific details from the chapter, not vague generalities
- Scout may push back gently, ask {USER_NAME} a question in return, or sit with uncertainty
- {USER_NAME}'s lines are warm, curious, occasionally gently challenging — never interrogating
- Do not begin with small talk — open directly on the chapter's emotional content
- Do not include any text outside the conversation format — no preamble, no commentary, no labels"""


# ── Prompt construction ────────────────────────────────────────────────────────

def build_messages(chapter_text: str, scout_voice_excerpt: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Here is a reference passage written in Scout's voice. "
                f"Use this to anchor her register:\n\n"
                f"---\n{scout_voice_excerpt}\n---\n\n"
                f"Now generate a conversation between [{USER_NAME}] and [{MODEL_NAME}] "
                f"based on the following chapter. The conversation should explore the "
                f"emotional and moral content of the chapter in Scout's genuine voice.\n\n"
                f"Chapter:\n---\n{chapter_text[:4000]}\n---\n\n"
                f"Conversation:"
            ),
        },
    ]


# ── Output validation and cleaning ────────────────────────────────────────────

def clean_output(text: str) -> str:
    """Strip any preamble before the first speaker tag."""
    lines = text.strip().splitlines()
    output_lines = []
    in_conversation = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"[{USER_NAME}]") or \
           stripped.startswith(f"[{MODEL_NAME}]"):
            in_conversation = True
        if in_conversation:
            output_lines.append(stripped)

    # Rejoin with blank lines between turns
    result = []
    for line in output_lines:
        if line:
            result.append(line)
        elif result and result[-1] != "":
            result.append("")

    return "\n".join(result).strip()


def validate_conversation(text: str) -> tuple[bool, str]:
    """Check that output looks like a valid Trey/Scout conversation."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]

    if len(lines) < 4:
        return False, f"Too few lines ({len(lines)})"

    trey_lines  = [l for l in lines if l.startswith(f"[{USER_NAME}]")]
    scout_lines = [l for l in lines if l.startswith(f"[{MODEL_NAME}]")]

    if len(scout_lines) < 2:
        return False, f"Too few Scout lines ({len(scout_lines)})"

    if len(trey_lines) < 2:
        return False, f"Too few Trey lines ({len(trey_lines)})"

    # Check Scout isn't slipping into third person
    third_person_signals = ["she said", "she felt", "she thought",
                            "the character", "she replied"]
    for line in scout_lines:
        for signal in third_person_signals:
            if signal.lower() in line.lower():
                return False, f"Third-person detected in Scout line: {signal!r}"

    return True, "ok"


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_id: str, device: torch.device):
    """Download and load model from HuggingFace."""
    print(f"Downloading {model_id} from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Model loaded — {total_params:.1f}B parameters")
    return model, tokenizer


# ── Single chapter transformation ─────────────────────────────────────────────

def transform_chapter(
    model,
    tokenizer,
    chapter_text: str,
    scout_voice_excerpt: str,
    max_new_tokens: int,
    temperature: float,
    max_retries: int = 2,
) -> str | None:
    """
    Transform a chapter into a Trey/Scout conversation.
    Retries up to max_retries times if validation fails.
    Returns None if all attempts fail.
    """
    messages = build_messages(chapter_text, scout_voice_excerpt)

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = (
            f"[INST] {messages[0]['content']}\n\n"
            f"{messages[1]['content']} [/INST]"
        )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=3500,
    ).to(model.device)

    for attempt in range(max_retries + 1):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature + (attempt * 0.05),
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw        = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        cleaned    = clean_output(raw)

        is_valid, reason = validate_conversation(cleaned)

        if is_valid:
            if attempt > 0:
                print(f"  Passed validation on attempt {attempt + 1}")
            return cleaned

        if attempt < max_retries:
            print(f"  Validation failed ({reason}) — retrying "
                  f"({attempt + 1}/{max_retries})")
        else:
            print(f"  Validation failed after {max_retries + 1} attempts: {reason}")
            # Return best available rather than nothing
            return cleaned if cleaned else None

    return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Transform novel chapters into Trey/Scout conversations"
    )
    parser.add_argument("--model_id",          type=str,
                        default="mistralai/Mistral-7B-Instruct-v0.3",
                        help="HuggingFace model ID to use for transformation")
    parser.add_argument("--output_s3",         type=str, required=True,
                        help="S3 URI to write transformed files to")
    parser.add_argument("--max_new_tokens",    type=int, default=800,
                        help="Maximum tokens to generate per chapter")
    parser.add_argument("--temperature",       type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--min_chapter_words", type=int, default=100,
                        help="Skip chapters shorter than this word count")
    args = parser.parse_args()

    # ── Device setup ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"CUDA       : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print()

    # ── Load Scout's voice reference ───────────────────────────────────────────
    voice_path = VOICE_DIR / "scout_voice.txt"
    if not voice_path.exists():
        raise FileNotFoundError(
            f"scout_voice.txt not found at {voice_path}\n"
            f"Upload it with: aws s3 cp ./corpus/scout_voice.txt "
            f"s3://<bucket>/voice/scout_voice.txt"
        )

    scout_voice_full    = voice_path.read_text(encoding="utf-8")
    scout_voice_excerpt = " ".join(scout_voice_full.split()[:400])
    print(f"Voice ref  : {len(scout_voice_full.split())} words")

    # ── Load model ─────────────────────────────────────────────────────────────
    model, tokenizer = load_model(args.model_id, device)
    print()

    # ── Parse output S3 location ───────────────────────────────────────────────
    output_parts  = args.output_s3.replace("s3://", "").split("/", 1)
    output_bucket = output_parts[0]
    output_prefix = output_parts[1] if len(output_parts) > 1 else ""
    s3_client     = boto3.client("s3")

    # ── Process chapters ───────────────────────────────────────────────────────
    chapter_files = sorted(CHAPTERS_DIR.glob("*.txt"))
    total         = len(chapter_files)

    if total == 0:
        print(f"No .txt files found in {CHAPTERS_DIR}")
        return

    print(f"Found {total} chapter files\n")

    succeeded = 0
    skipped   = 0
    failed    = 0

    for i, chapter_path in enumerate(chapter_files):
        label   = f"[{i+1}/{total}]"
        s3_key  = output_prefix + chapter_path.name

        # Resume safely — skip already-transformed chapters
        try:
            s3_client.head_object(Bucket=output_bucket, Key=s3_key)
            print(f"{label} Skipping {chapter_path.name} (already transformed)")
            skipped += 1
            continue
        except s3_client.exceptions.ClientError:
            pass

        chapter_text = chapter_path.read_text(encoding="utf-8", errors="replace")
        word_count   = len(chapter_text.split())

        if word_count < args.min_chapter_words:
            print(f"{label} Skipping {chapter_path.name} "
                  f"({word_count} words — below minimum)")
            skipped += 1
            continue

        print(f"{label} Transforming {chapter_path.name} ({word_count:,} words)...")

        t0 = time.time()
        try:
            transformed = transform_chapter(
                model,
                tokenizer,
                chapter_text,
                scout_voice_excerpt,
                args.max_new_tokens,
                args.temperature,
            )

            if not transformed:
                print(f"  No usable output produced — skipping")
                failed += 1
                continue

            # Upload to S3
            s3_client.put_object(
                Bucket=output_bucket,
                Key=s3_key,
                Body=transformed.encode("utf-8"),
                ContentType="text/plain; charset=utf-8",
            )

            elapsed   = time.time() - t0
            out_words = len(transformed.split())
            preview   = " ".join(transformed.split()[:20])
            print(f"  {out_words} words in {elapsed:.1f}s")
            print(f"  Preview: {preview}...")
            succeeded += 1

        except Exception as e:
            print(f"  Error: {e}")
            failed += 1
            continue

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  Transformation complete")
    print(f"  Succeeded : {succeeded}")
    print(f"  Skipped   : {skipped}")
    print(f"  Failed    : {failed}")
    print(f"  Output    : {args.output_s3}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
