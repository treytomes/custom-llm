"""
config.py — Central configuration for training and model architecture
"""

DEFAULT_CONFIG = {

    # ── Paths ─────────────────────────────────────────────
    "corpus_dir": "./corpus",
    "data_path": "./data/corpus.pt",
    "output_dir": "./checkpoints",

    # ── Tokenizer ─────────────────────────────────────────
    "tokenizer": "mistralai/Mistral-7B-v0.1",

    # ── Model architecture (~50M parameters) ──────────────
    "dim": 512,
    "layers": 12,
    "heads": 8,

    # Context window
    "block_size": 512,

    # Will be filled after tokenizer loads
    "vocab_size": None,

    # ── Training hyperparameters ──────────────────────────
    "batch_size": 8,
    "max_steps": 20000,
    "learning_rate": 3e-4,

    # Scheduler
    "warmup_steps": 100,

    # ── Logging / checkpoints ─────────────────────────────
    "log_interval": 50,
    "save_interval": 1000,
}

def update_vocab_size(cfg, tokenizer):
    """
    Ensure model vocab matches tokenizer vocab.
    """
    cfg["vocab_size"] = tokenizer.vocab_size
    return cfg