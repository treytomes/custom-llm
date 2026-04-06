# config.py

import os
from pathlib import Path

TOKENIZER_NAME = "mistralai/Mistral-7B-v0.1"
USER_NAME      = "Trey"
MODEL_NAME     = "Scout"

#
# Training parameters
#

# The current max steps is 150,932 based on the corpus size.

MAX_STEPS      = 1000
LEARNING_RATE  = 3e-4
WARMUP_STEPS   = 100
MIN_LR         = 3e-5

LOG_INTERVAL   = 20
SAVE_INTERVAL  = 50
NUM_WORKERS    = os.cpu_count() // 2

#
# Inference Parameters
#

# Kept short intentionally — Scout's coherent window at 50M is ~3 sentences.
# Increase as coherence improves with scale.
MAX_NEW_TOKENS = 40

TEMPERATURE    = 0.8
TOP_K          = 40
REP_PENALTY    = 1.3   # 1.0 = disabled; 1.2–1.5 is a reasonable range

#
# Training Parameters
#

# Data paths

# Checkpoints
# Update CHECKPOINT_PATH locally in infer.py when testing DPO output
CHECKPOINT_DIR      = "../data/checkpoints"
CHECKPOINT_PATH     = Path(CHECKPOINT_DIR) / "latest.pt"
DPO_CHECKPOINT_PATH = Path(CHECKPOINT_DIR) / "dpo/dpo_latest.pt"

LOGGER_NAME         = "llm"
OUTPUT_DIR          = "../data/corpus"
CORPUS_DIR          = Path(OUTPUT_DIR) / "dialogue"
CORPUS_TOKEN_FILE   = "corpus.pt"

DATA_PATH           = Path(OUTPUT_DIR) / CORPUS_TOKEN_FILE

# ── Training hyperparameters ──────────────────────────

# Context window size in tokens.
BLOCK_SIZE = 128

BATCH_SIZE = 8

# Model architecture (~50M parameters)
MODEL_DIM   = 512
MODEL_LAYERS = 12
MODEL_HEADS  = 8
