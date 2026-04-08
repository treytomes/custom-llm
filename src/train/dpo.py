"""
train/dpo.py

Build DPO training pairs from annotated chat session logs.

Converts interactive review logs into preference pairs suitable
for Direct Preference Optimization training.

chosen   = best response (or correction if provided)
rejected = worst response
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Load session logs ─────────────────────────────────────────

def load_all_sessions(logs_dir: Path) -> list[dict]:

    all_entries = []

    # Only load DPO session files
    files = sorted(logs_dir.glob("dpo_*.jsonl"))

    if not files:
        raise FileNotFoundError(
            f"No DPO .jsonl files found in {logs_dir} (expected prefix 'dpo_')"
        )

    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:

            for line in f:
                line = line.strip()

                if line:
                    all_entries.append(json.loads(line))

    logger.info(
        "Loaded %d DPO entries from %d session files",
        len(all_entries),
        len(files),
    )

    return all_entries


# ── Build preference pairs ─────────────────────────────────────

def build_preference_pairs(
    entries: list[dict],
    prefer_corrections: bool = True,
) -> list[dict]:
    """
    Build DPO preference pairs from annotated entries.

    chosen is determined by (in priority order):
      1. Hand-written correction (if present and prefer_corrections=True)
      2. Selected best candidate (entry["chosen"])

    rejected is entry["rejected"] — the selected worst candidate.

    Both chosen and rejected must be non-empty strings for a pair to be built.
    """
    pairs = []

    skipped_no_chosen = 0
    skipped_no_rejected = 0
    skipped_same = 0

    for entry in entries:

        prompt = entry.get("prompt", "").strip()

        if not prompt:
            continue

        correction = entry.get("correction", "")
        chosen_raw = entry.get("chosen", "")

        # Determine chosen response

        if prefer_corrections and isinstance(correction, str) and correction.strip():
            chosen = correction.strip()

        elif isinstance(chosen_raw, str) and chosen_raw.strip():
            chosen = chosen_raw.strip()

        else:
            skipped_no_chosen += 1
            continue

        # Determine rejected response

        rejected_raw = entry.get("rejected", "")

        if not isinstance(rejected_raw, str) or not rejected_raw.strip():
            skipped_no_rejected += 1
            continue

        rejected = rejected_raw.strip()

        if chosen == rejected:
            skipped_same += 1
            continue

        pair = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "notes": entry.get("notes", ""),
            "session": entry.get("session_id", ""),
            "turn": entry.get("turn", 0),
        }
        pairs.append(pair)

    logger.info("Pair construction summary:")
    logger.info("  Total entries         : %d", len(entries))
    logger.info("  Pairs built           : %d", len(pairs))
    logger.info("  Skipped (no chosen)   : %d", skipped_no_chosen)
    logger.info("  Skipped (no rejected) : %d", skipped_no_rejected)
    logger.info("  Skipped (identical)   : %d", skipped_same)

    return pairs


# ── Save pairs ─────────────────────────────────────────────────

def save_pairs(pairs: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info("Saved %d pairs to %s", len(pairs), output_path)


# ── Debug helper ───────────────────────────────────────────────

def print_sample(pairs: list[dict], n: int = 3):

    logger.info("Sample pairs (first %d)", min(n, len(pairs)))

    for i, pair in enumerate(pairs[:n]):

        logger.info(
            "[%d] Prompt: %.80s",
            i + 1,
            pair["prompt"],
        )

        logger.info(
            "     Chosen: %.100s",
            pair["chosen"],
        )

        logger.info(
            "     Rejected: %.100s",
            pair["rejected"],
        )


# ── Public pipeline entrypoint ─────────────────────────────────

def build_dpo_dataset(
    logs_dir: Path,
    output_path: Path,
    prefer_corrections: bool = True,
):
    entries = load_all_sessions(logs_dir)

    pairs = build_preference_pairs(
        entries,
        prefer_corrections=prefer_corrections,
    )

    if not pairs:
        logger.info("No pairs built from logs")
        return

    save_pairs(pairs, output_path)
    print_sample(pairs)