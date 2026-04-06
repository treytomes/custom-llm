# dpo.py
"""
dpo.py — Build DPO training pairs from reviewed session logs

Reads annotated session logs from review.py and constructs preference
pairs for Direct Preference Optimization:

    chosen   = the best response (or your hand-written correction)
    rejected = the worst response

Usage:
    python dpo.py --logs ./sessions/ --output ./dpo_data/pairs.jsonl
"""

import argparse
import json
from pathlib import Path


def load_all_sessions(logs_dir: Path) -> list[dict]:
    all_entries = []
    files = sorted(logs_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {logs_dir}")

    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_entries.append(json.loads(line))

    print(f"Loaded {len(all_entries)} entries from {len(files)} session files")
    return all_entries


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
    skipped_no_chosen   = 0
    skipped_no_rejected = 0
    skipped_same        = 0

    for entry in entries:
        prompt = entry.get("prompt", "").strip()
        if not prompt:
            continue

        # Determine chosen text
        correction = entry.get("correction", "")
        chosen_raw = entry.get("chosen", "")

        if prefer_corrections and isinstance(correction, str) and correction.strip():
            chosen = correction.strip()
        elif isinstance(chosen_raw, str) and chosen_raw.strip():
            chosen = chosen_raw.strip()
        else:
            skipped_no_chosen += 1
            continue

        # Determine rejected text
        rejected_raw = entry.get("rejected", "")
        if not isinstance(rejected_raw, str) or not rejected_raw.strip():
            skipped_no_rejected += 1
            continue
        rejected = rejected_raw.strip()

        # Don't create pairs where chosen and rejected are identical
        if chosen == rejected:
            skipped_same += 1
            continue

        pair = {
            "prompt":   prompt,
            "chosen":   chosen,
            "rejected": rejected,
            "notes":    entry.get("notes", ""),
            "session":  entry.get("session_id", ""),
            "turn":     entry.get("turn", 0),
        }
        pairs.append(pair)

    print(f"\nPair construction summary:")
    print(f"  Total entries          : {len(entries)}")
    print(f"  Pairs built            : {len(pairs)}")
    print(f"  Skipped (no chosen)    : {skipped_no_chosen}")
    print(f"  Skipped (no rejected)  : {skipped_no_rejected}")
    print(f"  Skipped (identical)    : {skipped_same}")

    return pairs


def save_pairs(pairs: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(pairs)} pairs to: {output_path}")


def print_sample(pairs: list[dict], n: int = 3):
    print(f"\n── Sample pairs (first {min(n, len(pairs))}) ──")
    for i, pair in enumerate(pairs[:n]):
        print(f"\n[{i+1}] Prompt  : {pair['prompt'][:80]}")
        print(f"     Chosen  : {pair['chosen'][:100]}")
        print(f"     Rejected: {pair['rejected'][:100]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build DPO training pairs from reviewed session logs"
    )
    parser.add_argument("--logs",   type=str, default="./sessions/",
                        help="Directory containing .jsonl session logs")
    parser.add_argument("--output", type=str,
                        default="./dpo_data/pairs.jsonl",
                        help="Output path for preference pairs JSONL")
    parser.add_argument("--no-prefer-corrections", action="store_true",
                        help="Use selected best candidate even when a "
                             "hand-written correction is available")
    args = parser.parse_args()

    entries = load_all_sessions(Path(args.logs))
    pairs   = build_preference_pairs(
        entries,
        prefer_corrections=not args.no_prefer_corrections,
    )

    if not pairs:
        print("\nNo pairs built. Make sure you have:")
        print("  1. Run interact.py to generate session logs")
        print("  2. Run review.py to select best/worst candidates")
        print("  3. At least some entries with both chosen and rejected filled in")
    else:
        save_pairs(pairs, Path(args.output))
        print_sample(pairs)
        print(f"\nNext step: python fine_tune.py "
              f"--pairs {args.output} "
              f"--checkpoint ./checkpoints/latest.pt")
