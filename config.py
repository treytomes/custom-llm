# config.py

TOKENIZER_NAME = "mistralai/Mistral-7B-v0.1"
USER_NAME      = "Trey"
MODEL_NAME     = "Scout"

# Inference parameters

# Kept short intentionally — Scout's coherent window at 50M is ~3 sentences.
# Increase as coherence improves with scale.
MAX_NEW_TOKENS = 40

TEMPERATURE    = 0.8
TOP_K          = 40
REP_PENALTY    = 1.3   # 1.0 = disabled; 1.2–1.5 is a reasonable range

# Checkpoints
# Update CHECKPOINT_PATH locally in infer.py when testing DPO output
CHECKPOINT_PATH     = "./checkpoints/latest.pt"
DPO_CHECKPOINT_PATH = "./checkpoints/dpo/dpo_latest.pt"