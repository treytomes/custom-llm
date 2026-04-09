#!/usr/bin/env python3

"""
generate_chat_corpus.py

Convert chat logs into a format that can be inserted into the pre-training conversational corpus.
"""

import json
from pathlib import Path

import config


def convert_log_to_transcript(jsonl_path: Path, output_path: Path):
    lines = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            user = record.get("user", "User")
            model = record.get("model", "Model")
            prompt = (record.get("prompt") or "").strip()
            response = (record.get("response") or "").strip()

            if prompt:
                lines.append(f"[{user}] {prompt}")

            if response and response != "<silence>":
                lines.append(f"[{model}] {response}")

    if not lines:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines))
        f.write("\n")


def generate_chat_corpus():
    files = sorted(config.LOG_DIR.glob("chat*.jsonl"))

    if not files:
        print("No chat log files found.")
        return

    for jsonl_file in files:
        out_name = jsonl_file.stem + ".txt"
        out_path = Path(config.DIALOGUE_OUTPUT_DIR) / out_name

        convert_log_to_transcript(jsonl_file, out_path)
        print(f"Converted {jsonl_file.name} → {out_path.name}")
