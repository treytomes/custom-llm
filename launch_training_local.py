from pathlib import Path
import argparse

from training.data import tokenize_corpus
from training.train import main as train_main
from transformers import AutoTokenizer


ROOT = Path(__file__).parent

DEFAULT_CORPUS = ROOT / "corpus"
DEFAULT_DATA = ROOT / "data"
DEFAULT_CHECKPOINTS = ROOT / "checkpoints"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--s3_bucket", default=None)

    args = parser.parse_args()

    corpus_dir = Path(args.data_dir) if args.data_dir else DEFAULT_CORPUS
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_CHECKPOINTS
    data_dir = DEFAULT_DATA

    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    token_file = data_dir / "corpus.pt"

    tokenizer_name = "mistralai/Mistral-7B-v0.1"

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if not token_file.exists():
        print("Tokenizing corpus...")
        tokenize_corpus(
            input_dir=corpus_dir,
            tokenizer=tokenizer,
            output_path=token_file,
        )

    print("Starting training...")

    train_args = argparse.Namespace(
        corpus_dir=str(corpus_dir),
        data_path=str(token_file),
        out_dir=str(output_dir),
        tokenizer=tokenizer_name,
        seq_len=512,
        batch_size=8,
        steps=40000,
        dim=512,
        layers=12,
        heads=8,
        lr=3e-4,
        log_interval=50,
        save_interval=1000,
        s3_bucket=args.s3_bucket,
    )

    train_main(train_args)


if __name__ == "__main__":
    main()