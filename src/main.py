# main.py

import logging
import os
import typer

from pathlib import Path
from rich.logging import RichHandler
from transformers import AutoTokenizer
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv

import config
from chat.repl import run_chat_repl
from train.data import (
    corpus_needs_tokenization,
    load_token_tensor,
    tokenize_corpus,
)
from train.train import run_training
from corpus.transform_corpus import generate_dialogue_corpus

load_dotenv()

app = typer.Typer()

# ---------------------------------------------------------
# logging setup

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger(config.LOGGER_NAME)


# ---------------------------------------------------------
# reporting helpers

def generate_training_plan(
    token_file: Path,
    target_tokens: int | None = None,
):
    """
    Analyze the tokenized corpus and estimate training workload.
    """

    seq_len = config.BLOCK_SIZE
    batch_size = config.BATCH_SIZE

    tokens, vocab_size = load_token_tensor(token_file)

    total_tokens = len(tokens)
    samples = max(0, total_tokens - seq_len - 1)

    tokens_per_step = seq_len * batch_size
    steps_per_epoch = samples // batch_size

    file_size_mb = token_file.stat().st_size / (1024 * 1024)

    logger.info("")
    logger.info("Training Planner")
    logger.info("------------------------------------------------")
    logger.info("Token file            : %s", token_file)
    logger.info("File size             : %.2f MB", file_size_mb)
    logger.info("Total tokens          : %s", f"{total_tokens:,}")
    logger.info("Sequence length       : %d", seq_len)
    logger.info("Batch size            : %d", batch_size)
    logger.info("Vocab size            : %s", vocab_size)
    logger.info("Tokens / step         : %s", f"{tokens_per_step:,}")
    logger.info("Training samples      : %s", f"{samples:,}")
    logger.info("Steps / epoch         : %s", f"{steps_per_epoch:,}")

    if target_tokens:
        steps_required = target_tokens // tokens_per_step
        epochs_required = steps_required / max(steps_per_epoch, 1)

        logger.info("")
        logger.info("Target Training Budget")
        logger.info("------------------------------------------------")
        logger.info("Target tokens         : %s", f"{target_tokens:,}")
        logger.info("Steps required        : %s", f"{steps_required:,}")
        logger.info("Epochs over dataset   : %.2f", epochs_required)


# ---------------------------------------------------------
# data helpers

def make_sample_corpus(output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = {
        "essay.txt": """
Curiosity is the engine of discovery. When a person asks a question
sincerely, they create a space in which new understanding can appear.
The question itself may be simple, but its implications often unfold
across many layers of thought.
""",

        "dialogue.txt": """
User: What makes a system intelligent?
Assistant: Intelligence is not only the ability to answer questions,
but the ability to recognize which questions matter.
User: So curiosity is part of intelligence?
Assistant: Often the most important part.
""",

        "technical.txt": """
Language models operate by predicting tokens in sequence.
During training, the model receives a context window and learns to
estimate probability distributions for the next token.
Scaling the dataset generally improves generalization.
""",
    }

    for name, text in samples.items():
        (output_dir / name).write_text(text.strip() * 200, encoding="utf-8")

    logger.info("Sample corpus written to %s", output_dir)


# ---------------------------------------------------------
# rendering helpers

def format_eta(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)

    if h > 0:
        return f"{h}h{m:02d}m"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def render_training_dashboard(stats):
    table = Table(title=f"{config.MODEL_NAME} Training")

    table.add_column("Step", justify="right")
    table.add_column("Loss")
    table.add_column("Avg")
    table.add_column("LR")
    table.add_column("Val")
    table.add_column("Tok/s")
    table.add_column("ETA")
    table.add_column("Elapsed")

    val = "-"
    if stats["val_loss"] is not None:
        val = f"{stats['val_loss']:.4f}"

    table.add_row(
        f"{stats['step']}",
        f"{stats['loss']:.4f}",
        f"{stats['avg_loss']:.4f}",
        f"{stats['lr']:.2e}",
        val,
        f"{stats['tokens_per_sec']:,.0f}",
        format_eta(stats["eta"]),
        f"{stats['elapsed']:.0f}s",
    )

    return Panel(table)


# ---------------------------------------------------------
# commands

@app.command()
def corpus_make_sample():
    """
    Generate the sample corpus.
    """

    corpus_dir = Path(config.CORPUS_DIR)
    make_sample_corpus(corpus_dir)


@app.command()
def corpus_prepare():
    """
    Prepare tokenized training corpus.
    """

    corpus_dir = Path(config.CORPUS_DIR)
    output_dir = Path(config.OUTPUT_DIR)
    token_file = output_dir / config.CORPUS_TOKEN_FILE

    logger.info("Loading tokenizer: %s", config.TOKENIZER_NAME)
    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)

    if corpus_needs_tokenization(corpus_dir, token_file):
        logger.info("Tokenizing corpus...")
        tokenize_corpus(
            input_dir=corpus_dir,
            tokenizer=tok,
            output_path=token_file,
        )
    else:
        logger.info("Token file already up to date")
    
    generate_training_plan(token_file)


@app.command()
def corpus_report():
    """
    Display statistics about the tokenized corpus.
    """

    token_file = Path(config.OUTPUT_DIR) / config.CORPUS_TOKEN_FILE

    if not token_file.exists():
        logger.error("Token file not found: %s", token_file)
        raise typer.Exit(1)

    logger.info("Loading token tensor: %s", token_file)

    generate_training_plan(token_file)


@app.command()
def train():
    """
    Launch model training.
    """

    token_file = Path(config.DATA_PATH)

    if not token_file.exists():
        logger.error("Token file missing. Run corpus_prepare first.")
        raise typer.Exit(1)

    logger.info("Starting training run...\n")

    from train.train import run_training

    with Live(refresh_per_second=4) as live:
        for stats in run_training():
            live.update(render_training_dashboard(stats))


@app.command()
def chat():
    """
    Start interactive chat with the trained model.
    """
    run_chat_repl()


@app.command()
def corpus_generate_dialogue(
    temperature: float = 0.7,
    upload_s3: str = "",
    book: str = None,
):
    """
    Generate conversational training data using the teacher model.
    """

    generate_dialogue_corpus(
        chapters_dir=config.CHAPTERS_DIR,
        output_dir=config.DIALOGUE_OUTPUT_DIR,
        voice_file=config.VOICE_FILE,
        endpoint=os.environ.get("AZURE_MISTRAL_ENDPOINT"),
        api_key=os.environ.get("AZURE_MISTRAL_KEY"),
        temperature=temperature,
        upload_s3=upload_s3,
        book=book,
    )


# ---------------------------------------------------------

if __name__ == "__main__":
    app()