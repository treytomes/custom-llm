"""
data.py — Dataset preparation and loading

Library module for tokenization and dataset construction.
No CLI, no direct console output. Logging is used so that
the application entry point controls presentation.
"""

import logging
import random
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

import config

logger = logging.getLogger(__name__)


class StreamingTextDataset(Dataset):
    """
    Streaming dataset over a flat token tensor.
    """

    def __init__(self, token_ids: torch.Tensor, block_size: int = 512):
        self.tokens = token_ids
        self.block_size = block_size
        self.n_samples = len(token_ids) - block_size - 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx
        end = idx + self.block_size + 1
        chunk = self.tokens[start:end]

        input_ids = chunk[:-1].long()
        labels = chunk[1:].long()

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


# -------------------------------------------------------------

def tokenize_corpus(
    input_dir: str | Path,
    tokenizer: PreTrainedTokenizer,
    output_path: str | Path,
    extensions: tuple = (".txt",),
    shuffle: bool = True,
):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_dir = Path(input_dir)
    output_path = Path(output_path)

    files = sorted(
        f for f in input_dir.rglob("*")
        if f.suffix.lower() in extensions and f.is_file()
    )

    if not files:
        raise ValueError(f"No {extensions} files found in {input_dir}")

    logger.info("Found %d files to tokenize", len(files))

    if shuffle:
        random.shuffle(files)

    all_ids = []
    total_chars = 0

    for i, fpath in enumerate(files):
        text = fpath.read_text(encoding="utf-8", errors="replace").strip()

        if not text:
            continue

        total_chars += len(text)

        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)

        if tokenizer.eos_token_id is not None:
            all_ids.append(tokenizer.eos_token_id)

        if (i + 1) % 10 == 0 or (i + 1) == len(files):
            logger.info(
                "[%d/%d] %s tokens so far",
                i + 1,
                len(files),
                f"{len(all_ids):,}",
            )

    token_tensor = torch.tensor(all_ids, dtype=torch.long)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "tokens": token_tensor,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_name": tokenizer.name_or_path,
    }

    torch.save(artifact, output_path)

    logger.info("Tokenization complete")
    logger.info("Files processed : %d", len(files))
    logger.info("Characters      : %s", f"{total_chars:,}")
    logger.info("Tokens          : %s", f"{len(all_ids):,}")
    logger.info("Compression     : %.2f chars/token", total_chars / len(all_ids))
    logger.info("Saved to        : %s", output_path)

    return token_tensor


# -------------------------------------------------------------

def load_token_tensor(path: str | Path):
    obj = torch.load(path, weights_only=True)

    if isinstance(obj, dict):
        return obj["tokens"], obj.get("vocab_size")
    else:
        return obj, None

# -------------------------------------------------------------

def build_dataloader(
    token_tensor: torch.Tensor,
    block_size: int = config.BLOCK_SIZE,
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = True,
    num_workers: int = config.NUM_WORKERS,
):
    dataset = StreamingTextDataset(token_tensor, block_size)

    logger.info(
        "Dataset: %s samples of %d tokens each",
        f"{len(dataset):,}",
        block_size,
    )

    # TODO: pin_memory=<is CUDA available?>, basically, this flag only works if acceleration is detected.

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


# -------------------------------------------------------------

def newest_file_mtime(directory):
    newest = 0

    for p in Path(directory).rglob("*"):
        if p.is_file():
            newest = max(newest, p.stat().st_mtime)

    return newest


def corpus_needs_tokenization(corpus_dir, token_file):
    token_file = Path(token_file)

    if not token_file.exists():
        return True

    return newest_file_mtime(corpus_dir) > token_file.stat().st_mtime