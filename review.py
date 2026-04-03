"""
review.py — Review multi-candidate session logs for DPO training

Reads session logs from interact.py and lets you select the best and
worst responses. Stores full response text (not indices) so dpo.py
can consume entries directly.

Usage:
    python review.py --log ./sessions/session_20240101_120000.jsonl
    python review.py --log ./sessions/session_20240101_120000.jsonl --unreviewed-only
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path


WRAP = 72


def wrap(text: str) -> str:
    return textwrap.fill(
        text, WRAP,
        initial_indent="  ",
        subsequent_indent="  "
    )


def load_entries(path: Path) -> list[dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def save_entries(path: Path, entries: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def show_candidates(entry: dict):
    candidates = entry.get("candidates", [])
    for i, c in enumerate(candidates):
        print(f"\n{'─' * 60}")
        print(f"  Candidate {i}")
        print(f"{'─' * 60}")
        print(wrap(c or "(empty)"))


def choose_index(prompt: str, max_i: int) -> int | None:
    while True:
        try:
            raw = input(prompt).strip()
            if raw == "":
                return None
            i = int(raw)
            if 0 <= i < max_i:
                return i
            print(f"  Enter a number between 0 and {max_i - 1}")
        except ValueError:
            print("  Enter a number.")


def review_entry(entry: dict, idx: int, total: int) -> dict:
    print(f"\n{'═' * 70}")
    print(f"  Entry {idx + 1} / {total}  —  Session {entry.get('session_id', '?')}  Turn {entry.get('turn', '?')}")
    print(f"{'═' * 70}")

    print("\nPrompt:\n")
    print(wrap(entry["prompt"]))

    candidates = entry.get("candidates", [])
    show_candidates(entry)

    n = len(candidates)
    if n == 0:
        print("  No candidates to review.")
        return entry

    print()
    chosen_idx   = choose_index("Best candidate #  (Enter to skip): ", n)
    rejected_idx = choose_index("Worst candidate # (Enter to skip): ", n)

    # Store full text, not indices — this is what dpo.py expects
    if chosen_idx is not None:
        entry["chosen"] = candidates[chosen_idx]

    if rejected_idx is not None:
        entry["rejected"] = candidates[rejected_idx]

    print("\nOptional: write an improved response (Enter to skip).")
    correction = input("Correction: ").strip()
    if correction:
        entry["correction"] = correction

    notes = input("Notes (optional): ").strip()
    if notes:
        entry["notes"] = notes

    return entry


def run_review(args):
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        sys.exit(1)

    entries = load_entries(log_path)

    if args.unreviewed_only:
        targets = [
            (i, e) for i, e in enumerate(entries)
            if e.get("chosen") is None
        ]
    else:
        targets = list(enumerate(entries))

    print(f"\nLoaded {len(entries)} entries")
    print(f"Reviewing {len(targets)} entries\n")

    if not targets:
        print("Nothing to review.")
        return

    reviewed = 0
    for i, (idx, entry) in enumerate(targets):
        entries[idx] = review_entry(entry, i, len(targets))
        # Save after every entry so progress isn't lost
        save_entries(log_path, entries)
        reviewed += 1

        cont = input("\nContinue? (Enter = yes, q = quit): ").strip().lower()
        if cont == "q":
            break

    print(f"\n{'═' * 60}")
    print(f"  Review complete: {reviewed} entries reviewed")
    print(f"{'═' * 60}")
    print(f"\nNext step:")
    print(f"  python dpo.py --logs ./sessions/ --output ./dpo_data/pairs.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",              type=str, required=True,
                        help="Path to session JSONL file to review")
    parser.add_argument("--unreviewed-only",  action="store_true",
                        help="Only show entries not yet reviewed")
    args = parser.parse_args()
    run_review(args)
