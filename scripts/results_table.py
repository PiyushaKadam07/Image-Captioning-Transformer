"""Collect *_metrics.json files from runs/ into a Markdown table for the README.

Usage:
    python scripts/results_table.py runs
"""
import json
import sys
from pathlib import Path

COLS = ["BLEU-1", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    rows = []
    for path in sorted(root.rglob("*_metrics.json")):
        m = json.loads(path.read_text())
        decode = "greedy" if m["beam_size"] == 1 else f"beam {m['beam_size']}"
        cells = [f"{m['scores'][c]:.3f}" if c in m["scores"] else "n/a" for c in COLS]
        rows.append(f"| {m['encoder']} | {decode} | {m['split']} ({m['images']}) | " + " | ".join(cells) + " |")
    print("| Encoder | Decoding | Split (images) | " + " | ".join(COLS) + " |")
    print("|" + "---|" * (3 + len(COLS)))
    print("\n".join(rows))


if __name__ == "__main__":
    main()
