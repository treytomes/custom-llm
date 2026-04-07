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

MAX_STEPS      = 30000
LEARNING_RATE  = 3e-4
WARMUP_STEPS   = 100
MIN_LR         = 3e-5

LOG_INTERVAL   = 20
SAVE_INTERVAL  = 50
NUM_WORKERS    = os.cpu_count() // 2

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
VOICE_FILE          = "../data/voice/scout_voice.txt"
CHAPTERS_DIR        = "../data/corpus/chapters"
DIALOGUE_OUTPUT_DIR = "../data/corpus/dialogue"


DATA_PATH           = Path(OUTPUT_DIR) / CORPUS_TOKEN_FILE

# ── Training hyperparameters ──────────────────────────

# Context window size in tokens.
BLOCK_SIZE = 128

BATCH_SIZE = 8
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

MAX_STEPS      = 30000
LEARNING_RATE  = 3e-4
WARMUP_STEPS   = 100
MIN_LR         = 3e-5

LOG_INTERVAL   = 20
SAVE_INTERVAL  = 50
NUM_WORKERS    = os.cpu_count() // 2

#
# Data paths
#

# Checkpoints
# Update CHECKPOINT_PATH locally in infer.py when testing DPO output
CHECKPOINT_DIR      = "../data/checkpoints"
CHECKPOINT_PATH     = Path(CHECKPOINT_DIR) / "latest.pt"
DPO_CHECKPOINT_PATH = Path(CHECKPOINT_DIR) / "dpo/dpo_latest.pt"

LOGGER_NAME         = "llm"
OUTPUT_DIR          = "../data/corpus"
CORPUS_DIR          = Path(OUTPUT_DIR) / "dialogue"
CORPUS_TOKEN_FILE   = "corpus.pt"
VOICE_FILE          = "../data/voice/scout_voice.txt"
CHAPTERS_DIR        = "../data/corpus/chapters"
DIALOGUE_OUTPUT_DIR = "../data/corpus/dialogue"

DATA_PATH           = Path(OUTPUT_DIR) / CORPUS_TOKEN_FILE

#
# Chat Logging
#

LOG_DIR = Path("../chat_logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "chat.jsonl"

#
# Generation parameters (how the model speaks)
#

# This limits how many tokens the model is allowed to generate in a single response.
# 
# Kept short intentionally — Scout's coherent window at 50M is ~3 sentences.
# Increase as coherence improves with scale.
MAX_NEW_TOKENS = 80

# Temperature controls randomness in sampling.
# Lower values → safer, repetitive, predictable.
# Higher values → more creative but more errors.
# Typical range:
# | Temperature | Behavior |
# |---|---|
# | 0.2–0.4 | deterministic |
# | 0.5–0.7 | balanced |
# | 0.8–1.0 | creative |
# | >1.0 | chaotic |
TEMPERATURE    = 0.6   # Try raising this later in the training cycle.

# Top‑K sampling restricts token choices to the K most probable tokens.
# Example:
# 
# If the vocabulary has 32k tokens but TOP_K = 40, the model only samples from the 40 most likely next tokens.
# 
# This reduces:
# * nonsense outputs
# * rare-token glitches
# * degenerate sampling loops
TOP_K          = 40

# This penalizes tokens that already appeared earlier in the response.
# Typical values:
# | Value | Effect |
# |---|---|
# | 1.0 | off |
# | 1.1 | mild |
# | 1.2–1.5 | strong |
REP_PENALTY    = 1.1

#
# Context and Training Parameters
#

# This is the maximum context window the model can see at once.
# 
# During training the model predicts the next token given the previous 128 tokens.
# 
# Effects:
# * smaller → faster training
# * larger → better reasoning and memory
BLOCK_SIZE = 128

# This is how many training sequences are processed per optimization step.
# i.e. the tokens / step throughput
BATCH_SIZE = 8

#
# Model architecture (~50M parameters)
#

# This is the size of the embedding vector for each token.
# Larger dimensions allow richer representations but increase compute.
# | Model size | Dim |
# |---|---|
# | Small | 256–512 |
# | Medium | 768–1024 |
# | Large | 2048+ |
MODEL_DIM   = 512

# This is the number of transformer blocks stacked together.
# 
# Each layer performs:
# * attention
# * feedforward transformation
# * residual mixing
# 
# More layers → deeper reasoning.
MODEL_LAYERS = 12

# This is the number of attention heads per layer.
# 
# Each head attends to different relationships.
# 
# Example:
# * head 1 → grammar
# * head 2 → topic continuity
# * head 3 → punctuation
# * etc.
MODEL_HEADS  = 8

#
# Inference Parameters
#

# Kept short intentionally — Scout's coherent window at 50M is ~3 sentences.
# Increase as coherence improves with scale.
MAX_NEW_TOKENS = BLOCK_SIZE
TEMPERATURE    = 0.8   # Try raising this later in the training cycle.
TOP_K          = 40
REP_PENALTY    = 1.1   # 1.0 = disabled; 1.2–1.5 is a reasonable range
