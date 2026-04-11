# ai_clients/tokenizer.py

import logging
from pathlib import Path
from transformers import AutoTokenizer

import config


logger = logging.getLogger(__name__)


def load_tokenizer():
    logger.info("Loading tokenizer: %s", config.TOKENIZER_NAME)
    try:
        # Try loading strictly from local cache
        return AutoTokenizer.from_pretrained(
            config.TOKENIZER_NAME,
            cache_dir=config.HUGGINGFACE_CACHE_PATH,
            local_files_only=True,
        )
    except Exception:
        # If not cached yet, download and store in cache
        tokenizer = AutoTokenizer.from_pretrained(
            config.TOKENIZER_NAME,
            cache_dir=config.HUGGINGFACE_CACHE_PATH,
        )
        return tokenizer