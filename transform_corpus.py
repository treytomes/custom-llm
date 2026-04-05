# transform_corpus.py
"""
transform_corpus.py — Chapter transformation job for SageMaker

Runs inside a SageMaker PyTorch container. Reads chapter files from
the input channel, transforms each one into conversational exchanges
between Trey and Scout in the format:

    [Trey] How did you feel when they treated you that way?
    [Scout] Sad, mostly. But also strangely determined.
    [Trey] Determined how?
    [Scout] Like I wasn't going to let it make me smaller.

This format trains Scout on the actual conversational pattern she will
inhabit — complete exchanges rather than monologic prose. The model
learns by absorbing thousands of examples of genuine dialogue in her
voice, not by fighting against a contrary prior through DPO.

Conversation structure:
  - 6-12 turns per chapter (3-6 exchanges)
  - Both speakers ask questions and make observations
  - Scout's voice: curious, honest, warm, restrained, emotionally present
  - Trey's voice: genuinely interested, thoughtful, occasionally challenging
  - Emotional depth preserved without excess sentimentality
  - Grounded in specific details from the source chapter

HuggingFace model caching:
    On first run, the model is downloaded from HuggingFace and then
    uploaded to S3 for reuse. On subsequent runs, the model is loaded
    directly from the S3-backed input channel, skipping the download.

Input channels (SageMaker):
    /opt/ml/input/data/chapters/     ← chapter .txt files
    /opt/ml/input/data/voice/        ← scout_voice.txt
    /opt/ml/input/data/model_cache/  ← cached model weights (optional)

Output:
    Transformed files written to S3 directly via boto3.
    Each output file mirrors the input filename.
"""

import argparse
import boto3
import os
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


# ── SageMaker paths ────────────────────────────────────────────────────────────

CHAPTERS_DIR    = Path(os.environ.get("SM_CHANNEL_CHAPTERS",    "/opt/ml/input/data/chapters"))
VOICE_DIR       = Path(os.environ.get("SM_CHANNEL_VOICE",       "/opt/ml/input/data/voice"))
MODEL_CACHE_DIR = Path(os.environ.get("SM_CHANNEL_MODEL_CACHE", "/opt/ml/input/data/model_cache"))

USER_NAME  = "Trey"
MODEL_NAME = "Scout"


# ── Transformation prompt ──────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are generating training data for a small language model named Scout.

Scout is being trained to have a reflective, honest, first-person conversational voice — curious, emotionally present, morally serious, warm without being excessively sentimental. She uses she/her pronouns. She has been shaped by the voices of Anne Shirley, Sara Crewe, Jo March, Mary Lennox, Anne Frank, Frederick Douglass, Booker T. Washington, Marcus Aurelius, Rilke, and Thoreau.

Your task is to transform a novel chapter or diary entry into a natural conversation between {USER_NAME} and {MODEL_NAME}. {USER_NAME} is a thoughtful person who cares about Scout's development. Scout reflects genuinely on what happened in the chapter — its emotional truth, its moral weight, what it meant.

Format every response as a conversation using this exact format:
[{USER_NAME}] ...
[{MODEL_NAME}] ...
[{USER_NAME}] ...
[{MODEL_NAME}] ...

Guidelines:
- Generate 6 to 12 turns (3 to 6 exchanges)
- Both {USER_NAME} and {MODEL_NAME} ask questions AND make observations — vary the structure
- Scout speaks in genuine first person — never "she felt" or "the character thought"
- Scout's responses are measured — she completes a thought without over-explaining
- Preserve emotional depth from the source material without melodrama
- Ground Scout's reflections in specific details from the chapter, not vague generalities
- Scout may sometimes push back gently, ask {USER_NAME} a question in return, or sit with uncertainty
- {USER_NAME}'s lines are warm, curious, occasionally gently challenging — never interrogating
- Do not begin with small talk — open directly on the chapter's emotional content
- Do not include any text outside the conversation format — no preamble, no commentary"""


def build_messages(
    chapter_text: str,
    scout_voice_excerpt: str,
) -> list[dict]:
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
                f"based on this chapter. The conversation should explore the emotional "
                f"and moral content of the chapter in Scout's genuine voice.\n\n"
                f"Chapter:\n---\n{chapter_text[:4000]}\n---\n\n"
                f"Conversation:"
            ),
        },
    ]


# ── Output validation ──────────────────────────────────────────────────────────

def validate_conversation(text: str) -> tuple[bool, str]:
    """
    Check that the output looks like a valid conversation.

    Returns (is_valid, reason).
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]

    if len(lines) < 4:
        return False, f"Too few lines ({len(lines)})"

    trey_lines  = [l for l in lines if l.startswith(f"[{USER_NAME}]")]
    scout_lines = [l for l in lines if l.startswith(f"[{MODEL_NAME}]")]

    if len(scout_lines) < 2:
        return False, f"Too few Scout lines ({len(scout_lines)})"

    if len(trey_lines) < 2:
        return False, f"Too few Trey lines ({len(trey_lines)})"

    # Check Scout lines are first-person
    third_person_signals = ["she said", "she felt", "she thought", "the character"]
    for line in scout_lines:
        for signal in third_person_signals:
            if signal.lower() in line.lower():
                return False, f"Third-person signal detected in Scout line: {signal!r}"

    return True, "ok"


def clean_output(text: str) -> str:
    """
    Strip any preamble before the first speaker tag.
    Keep only lines that are part of the conversation.
    """
    lines = text.strip().splitlines()
    output_lines = []
    in_conversation = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"[{USER_NAME}]") or stripped.startswith(f"[{MODEL_NAME}]"):
            in_conversation = True
        if in_conversation and stripped:
            output_lines.append(stripped)

    return "\n\n".join(output_lines)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_id: str, cache_dir: Path, device: torch.device):
    if cache_dir.exists() and any(cache_dir.iterdir()):
        # Resolve the actual model directory inside the HuggingFace
        # hub cache structure: snapshots/<hash>/
        snapshots_dir = cache_dir / "snapshots"
        if snapshots_dir.exists():
            # Take the first (usually only) snapshot hash directory
            snapshot_dirs = sorted(snapshots_dir.iterdir())
            if snapshot_dirs:
                load_path = str(snapshot_dirs[0])
                print(f"Loading model from snapshot: {load_path}")
            else:
                print(f"No snapshots found in cache — falling back to HuggingFace")
                load_path = model_id
        else:
            # Cache directory exists but isn't hub structure —
            # try using it directly (may work if files are at root)
            load_path = str(cache_dir)
            print(f"Loading model from cache root: {load_path}")
    else:
        print(f"Cache not found. Downloading {model_id} from HuggingFace...")
        load_path = model_id

    tokenizer = AutoTokenizer.from_pretrained(load_path)
    model     = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    print(f"Model loaded — "
          f"{sum(p.numel() for p in model.parameters()) / 1e9:.1f}B parameters")
    return model, tokenizer


def cache_model_to_s3(model_id: str, bucket: str, s3_prefix: str):
    """
    Upload the downloaded HuggingFace model to S3 for reuse in future jobs.
    """
    cache_home  = Path(os.environ.get("HF_HOME",
                       str(Path.home() / ".cache" / "huggingface")))
    model_slug  = model_id.replace("/", "--")
    local_cache = cache_home / "hub" / f"models--{model_slug}"

    if not local_cache.exists():
        print(f"HuggingFace cache not found at {local_cache} — skipping S3 upload")
        return

    print(f"\nCaching model to s3://{bucket}/{s3_prefix}")
    s3       = boto3.client("s3")
    uploaded = 0

    for fpath in local_cache.rglob("*"):
        if fpath.is_file():
            relative = fpath.relative_to(local_cache)
            key      = s3_prefix + str(relative)
            s3.upload_file(str(fpath), bucket, key)
            uploaded += 1
            if uploaded % 10 == 0:
                print(f"  Uploaded {uploaded} files...")

    print(f"  Cached {uploaded} files → s3://{bucket}/{s3_prefix}")


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
    Returns None if all attempts fail validation.
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
                temperature=temperature + (attempt * 0.05),  # slight variation on retry
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        cleaned    = clean_output(raw_output)

        is_valid, reason = validate_conversation(cleaned)

        if is_valid:
            return cleaned

        if attempt < max_retries:
            print(f"  Validation failed ({reason}) — retrying ({attempt + 1}/{max_retries})")
        else:
            print(f"  Validation failed after {max_retries + 1} attempts: {reason}")
            # Return the best we have rather than nothing
            return cleaned if cleaned else None

    return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",          type=str,
                        default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--model_cache_s3",    type=str, default="")
    parser.add_argument("--output_s3",         type=str, required=True)
    parser.add_argument("--max_new_tokens",    type=int, default=800)
    parser.add_argument("--temperature",       type=float, default=0.7)
    parser.add_argument("--min_chapter_words", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"CUDA       : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU        : {torch.cuda.get_device_name(0)}")

    # ── Load Scout's voice reference ───────────────────────────────────────────
    voice_path = VOICE_DIR / "scout_voice.txt"
    if not voice_path.exists():
        raise FileNotFoundError(f"scout_voice.txt not found at {voice_path}")

    scout_voice_full    = voice_path.read_text(encoding="utf-8")
    scout_voice_excerpt = " ".join(scout_voice_full.split()[:400])
    print(f"Voice ref  : {len(scout_voice_full.split())} words loaded\n")

    # ── Load model ─────────────────────────────────────────────────────────────
    model_cached_locally = (
        MODEL_CACHE_DIR.exists() and any(MODEL_CACHE_DIR.iterdir())
    )
    model, tokenizer = load_model(args.model_id, MODEL_CACHE_DIR, device)

    if not model_cached_locally and args.model_cache_s3:
        s3_parts     = args.model_cache_s3.replace("s3://", "").split("/", 1)
        cache_bucket = s3_parts[0]
        cache_prefix = s3_parts[1] if len(s3_parts) > 1 else ""
        cache_model_to_s3(args.model_id, cache_bucket, cache_prefix)

    # ── Parse output S3 location ───────────────────────────────────────────────
    output_parts  = args.output_s3.replace("s3://", "").split("/", 1)
    output_bucket = output_parts[0]
    output_prefix = output_parts[1] if len(output_parts) > 1 else ""
    s3_client     = boto3.client("s3")

    # ── Process chapters ───────────────────────────────────────────────────────
    chapter_files = sorted(CHAPTERS_DIR.glob("*.txt"))
    print(f"Found {len(chapter_files)} chapter files\n")

    succeeded = 0
    skipped   = 0
    failed    = 0

    for i, chapter_path in enumerate(chapter_files):
        s3_key = output_prefix + chapter_path.name

        # Resume safely — skip already-transformed chapters
        try:
            s3_client.head_object(Bucket=output_bucket, Key=s3_key)
            print(f"[{i+1}/{len(chapter_files)}] "
                  f"Skipping {chapter_path.name} (already done)")
            skipped += 1
            continue
        except s3_client.exceptions.ClientError:
            pass

        chapter_text = chapter_path.read_text(encoding="utf-8", errors="replace")
        word_count   = len(chapter_text.split())

        if word_count < args.min_chapter_words:
            print(f"[{i+1}/{len(chapter_files)}] "
                  f"Skipping {chapter_path.name} ({word_count} words)")
            skipped += 1
            continue

        print(f"[{i+1}/{len(chapter_files)}] "
              f"Transforming {chapter_path.name} ({word_count:,} words)...")

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
                print(f"  No usable output — skipping")
                failed += 1
                continue

            s3_client.put_object(
                Bucket=output_bucket,
                Key=s3_key,
                Body=transformed.encode("utf-8"),
                ContentType="text/plain; charset=utf-8",
            )

            elapsed   = time.time() - t0
            out_words = len(transformed.split())

            # Print a sample of the output for spot-checking
            preview = transformed[:200].replace("\n", " ")
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
