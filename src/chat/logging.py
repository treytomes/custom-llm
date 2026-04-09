"""
chat/infer.py

Chat logging helpers.
"""

import json
import datetime
from pathlib import Path

import config


def get_now():
    return datetime.datetime.now(datetime.UTC)


def get_timestamp():
    return get_now().isoformat()


def log_chat(prompt, response):
    entry = {
        "timestamp": get_timestamp(),
        "user": config.USER_NAME,
        "model": config.MODEL_NAME,
        "prompt": prompt,
        "response": response,
    }

    with open(config.LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def build_log_path(prefix="chat"):
    today = get_now().strftime("%Y-%m-%d")

    idx = 1
    while True:
        path = config.LOG_DIR / f"{prefix}_{today}_conversation_{idx}.jsonl"
        if not path.exists():
            return path
        idx += 1


def log_turn(log_file, prompt, response):
    entry = {
        "timestamp": get_timestamp(),
        "user": config.USER_NAME,
        "model": config.MODEL_NAME,
        "prompt": prompt,
        "response": response,
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

