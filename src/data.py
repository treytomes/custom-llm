"""
data.py — Dataset preparation and loading

Handles tokenization and packaging of raw text into training batches.
Designed to work with plain .txt files so you can feed it anything —
books, essays, conversations, domain-specific text — without needing
a specific dataset format.

Usage:
    # Prepare a tokenized dataset from raw text files
    python data.py --input_dir ./corpus --output_dir ./data/prepared

    # Or import and use in training:
    from data import TextDataset, build_dataloader
"""

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer


class StreamingTextDataset(Dataset):
    """
    Streaming dataset over a flat token tensor.

    Instead of precomputing sliding windows, we simply index
    contiguous token chunks on demand.

    This avoids dataset construction overhead and improves
    training throughput.
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

    files = sorted([
        f for f in input_dir.rglob("*")
        if f.suffix.lower() in extensions and f.is_file()
    ])

    if not files:
        raise ValueError(f"No {extensions} files found in {input_dir}")

    print(f"Found {len(files)} files to tokenize")

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

        # EOS between documents
        if tokenizer.eos_token_id is not None:
            all_ids.append(tokenizer.eos_token_id)  # teaches document boundaries [1]

        if (i + 1) % 10 == 0 or (i + 1) == len(files):
            print(f"[{i+1}/{len(files)}] {len(all_ids):,} tokens so far...")

    token_tensor = torch.tensor(all_ids, dtype=torch.long)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(token_tensor, output_path)

    print("\nTokenization complete:")
    print(f"Files processed : {len(files)}")
    print(f"Characters      : {total_chars:,}")
    print(f"Tokens          : {len(all_ids):,}")
    print(f"Compression     : {total_chars/len(all_ids):.2f} chars/token")
    print(f"Saved to        : {output_path}")

    return token_tensor


def load_token_tensor(path: str | Path) -> torch.Tensor:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Token file not found: {path}\n"
            f"Run: python data.py --input_dir ./corpus --output_dir ./data"
        )

    return torch.load(path, weights_only=True)


def build_dataloader(
    token_tensor: torch.Tensor,
    block_size: int = 512,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
):

    dataset = StreamingTextDataset(token_tensor, block_size)

    print(f"Dataset: {len(dataset):,} samples of {block_size} tokens each")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def make_sample_corpus(output_dir: str | Path):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        ("sample1.txt", "Curiosity drives discovery."),
        ("sample2.txt", "Reasoning improves understanding."),
        ("sample3.txt", "Honesty strengthens communication."),
    ]

    for filename, content in samples:
        (output_dir / filename).write_text(content, encoding="utf-8")

    print(f"Sample corpus written to {output_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prepare training data")

    parser.add_argument("--input_dir", type=str, default="./corpus")
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
    )
    parser.add_argument("--make_sample", action="store_true")

    args = parser.parse_args()

    if args.make_sample:

        make_sample_corpus(args.input_dir)

    else:

        from transformers import AutoTokenizer

        print(f"Loading tokenizer: {args.tokenizer}")

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

        tokenize_corpus(
            input_dir=args.input_dir,
            tokenizer=tokenizer,
            output_path=Path(args.output_dir) / "corpus.pt",
        )