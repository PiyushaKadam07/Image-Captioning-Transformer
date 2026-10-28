"""Caption one image, and optionally save a word-by-word attention map.

Usage:
    python scripts/caption.py --checkpoint runs/clip/best.pt --image photo.jpg --attention attention.png
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from captioner.data import build_transform  # noqa: E402
from captioner.decoding import generate  # noqa: E402
from captioner.utils import get_tokenizer, load_checkpoint, plot_attention  # noqa: E402


def caption_image(model, tok, cfg, image: Image.Image, device="cpu", beam_size=None):
    tf = build_transform(model.encoder.mean, model.encoder.std, cfg["data"]["image_size"])
    feats = model.encode(tf(image.convert("RGB")).unsqueeze(0).to(device))
    d = cfg["decode"]
    (toks,), (attn,) = generate(model, feats, tok.eos_token_id, tok.eos_token_id,
                                beam_size=beam_size or d["beam_size"], max_len=d["max_len"],
                                length_penalty=d["length_penalty"], no_repeat_ngram=d["no_repeat_ngram"],
                                return_attention=True)
    words = [tok.decode([t]).strip() for t in toks]
    return tok.decode(toks).strip(), words, attn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--attention", default=None, help="Where to save the attention figure (PNG)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_checkpoint(args.checkpoint, device)
    tok = get_tokenizer(cfg)
    image = Image.open(args.image)
    with torch.no_grad():
        caption, words, attn = caption_image(model, tok, cfg, image, device)
    print(caption)
    if args.attention:
        plot_attention(image, words, attn, args.attention)
        print(f"Attention map saved to {args.attention}")


if __name__ == "__main__":
    main()
