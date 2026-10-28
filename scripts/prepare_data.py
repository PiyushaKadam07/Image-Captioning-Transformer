"""Build train/val/test JSONL files for MS COCO 2014.

Two sources are supported:

1. The Karpathy split file (preferred, matches published papers):
       python scripts/prepare_data.py --karpathy path/to/dataset_coco.json --out data

2. The official COCO caption annotations, when the Karpathy file is not available:
       python scripts/prepare_data.py --coco-train captions_train2014.json \
           --coco-val captions_val2014.json --out data
   This follows the same recipe as the Karpathy split (5,000 val + 5,000 test images drawn from
   val2014, everything else used for training) with a fixed random seed. The image counts match,
   but the exact images differ, so scores are close to, not identical with, the Karpathy protocol.

Each output line: {"image_id": int, "image": "<folder>/<file>.jpg", "captions": [...]}.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def from_karpathy(path):
    for img in json.load(open(path))["images"]:
        split = "train" if img["split"] == "restval" else img["split"]
        yield split, {
            "image_id": img.get("cocoid", img.get("imgid")),
            "image": f"{img['filepath']}/{img['filename']}",
            "captions": [s["raw"] for s in img["sentences"]],
        }


def _load_coco(path, folder):
    data = json.load(open(path))
    caps = defaultdict(list)
    for ann in data["annotations"]:
        caps[ann["image_id"]].append(ann["caption"].strip())
    return [{"image_id": im["id"], "image": f"{folder}/{im['file_name']}", "captions": caps[im["id"]]}
            for im in sorted(data["images"], key=lambda im: im["id"]) if caps[im["id"]]]


def from_coco(train_path, val_path, n_val=5000, n_test=5000, seed=123):
    train = _load_coco(train_path, "train2014")
    val = _load_coco(val_path, "val2014")
    random.Random(seed).shuffle(val)
    for r in val[:n_val]:
        yield "val", r
    for r in val[n_val:n_val + n_test]:
        yield "test", r
    for r in train + val[n_val + n_test:]:
        yield "train", r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--karpathy", help="Path to dataset_coco.json")
    ap.add_argument("--coco-train", help="Path to captions_train2014.json")
    ap.add_argument("--coco-val", help="Path to captions_val2014.json")
    ap.add_argument("--out", default="data", help="Output folder for the JSONL files")
    args = ap.parse_args()

    if args.karpathy:
        rows = from_karpathy(args.karpathy)
    elif args.coco_train and args.coco_val:
        rows = from_coco(args.coco_train, args.coco_val)
    else:
        ap.error("Give either --karpathy, or both --coco-train and --coco-val.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {s: open(out / f"{s}.jsonl", "w") for s in ("train", "val", "test")}
    counts = Counter()
    for split, record in rows:
        files[split].write(json.dumps(record) + "\n")
        counts[split] += 1
    for f in files.values():
        f.close()
    print("Images per split:", dict(counts))


if __name__ == "__main__":
    main()
