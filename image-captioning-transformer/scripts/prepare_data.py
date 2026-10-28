"""Convert the Karpathy COCO split file (dataset_coco.json) into train/val/test JSONL files.

Each output line: {"image_id": int, "image": "<folder>/<file>.jpg", "captions": [...]}.
The "restval" images are folded into training, following standard practice.

Usage:
    python scripts/prepare_data.py --karpathy path/to/dataset_coco.json --out data
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--karpathy", required=True, help="Path to dataset_coco.json")
    ap.add_argument("--out", default="data", help="Output folder for the JSONL files")
    args = ap.parse_args()

    images = json.load(open(args.karpathy))["images"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {s: open(out / f"{s}.jsonl", "w") for s in ("train", "val", "test")}
    counts = Counter()
    for img in images:
        split = "train" if img["split"] == "restval" else img["split"]
        record = {
            "image_id": img.get("cocoid", img.get("imgid")),
            "image": f"{img['filepath']}/{img['filename']}",
            "captions": [s["raw"] for s in img["sentences"]],
        }
        files[split].write(json.dumps(record) + "\n")
        counts[split] += 1
    for f in files.values():
        f.close()
    print("Images per split:", dict(counts))


if __name__ == "__main__":
    main()
