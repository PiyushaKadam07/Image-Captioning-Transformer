"""Score a trained checkpoint on the test split with the COCO caption metrics.

Usage:
    python scripts/evaluate.py --checkpoint runs/clip/best.pt
    python scripts/evaluate.py --checkpoint runs/clip/best.pt --beam-size 1   # greedy, for the ablation table
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from captioner.data import read_jsonl  # noqa: E402
from captioner.utils import get_tokenizer, load_checkpoint  # noqa: E402
from scripts.train import evaluate_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--beam-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="Score only N images")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_checkpoint(args.checkpoint, device)
    tok = get_tokenizer(cfg)
    beam = args.beam_size or cfg["decode"]["beam_size"]
    scores, hyps = evaluate_split(model, tok, read_jsonl(cfg["data"][args.split]), cfg, device,
                                  limit=args.limit, beam_size=beam)

    name = f"{args.split}_beam{beam}"
    out = Path(args.checkpoint).parent
    json.dump({"encoder": cfg["model"]["encoder"], "split": args.split, "beam_size": beam,
               "images": len(hyps), "scores": scores}, open(out / f"{name}_metrics.json", "w"), indent=2)
    json.dump([{"image_id": k, "caption": v} for k, v in hyps.items()],
              open(out / f"{name}_captions.json", "w"), indent=2)
    print(f"{cfg['model']['encoder']} | {args.split} | beam {beam} | {len(hyps)} images")
    for k, v in scores.items():
        print(f"  {k:8s} {v:.3f}")


if __name__ == "__main__":
    main()
