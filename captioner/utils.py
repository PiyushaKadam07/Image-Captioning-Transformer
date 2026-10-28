"""Shared helpers: tokenizer, checkpoints, seeding, attention heatmaps."""
from __future__ import annotations

import math
import random

import numpy as np
import torch

from .config import load_config
from .model import Captioner


class ByteTokenizer:
    """Tiny offline tokenizer (one token per byte). Used only by the test suite."""

    eos_token_id = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="ignore")


def get_tokenizer(cfg: dict):
    if cfg["model"].get("tokenizer") == "bytes":
        return ByteTokenizer()
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(cfg["model"]["decoder_name"])
    tok.pad_token = tok.eos_token  # GPT-2 has no pad token; padded positions are masked out
    return tok


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, model: Captioner, cfg: dict, extra: dict | None = None) -> None:
    torch.save({"model": model.state_dict(), "config": cfg, **(extra or {})}, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = load_config(overrides=ckpt["config"])
    # All weights come from the checkpoint, so only the architecture configs are fetched.
    model = Captioner(load_config(overrides={**cfg, "model": {**cfg["model"], "load_weights": False}}))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg


def attention_to_grid(attn_row: torch.Tensor) -> np.ndarray:
    """(regions,) attention weights -> square grid normalised to [0, 1]."""
    side = int(math.isqrt(attn_row.numel()))
    grid = attn_row[: side * side].reshape(side, side).numpy()
    grid = grid - grid.min()
    return grid / (grid.max() + 1e-8)


def plot_attention(image, words: list[str], attn: torch.Tensor, path: str, max_words: int = 12) -> None:
    """Save a figure showing which image regions each generated word attended to."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    words = words[:max_words]
    cols = 4
    rows = math.ceil((len(words) + 1) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_1d(axes).ravel()
    # Same resize + centre crop as the model input, so heatmaps line up with what the model saw.
    img = image.convert("RGB")
    scale = 224 / min(img.size)
    img = img.resize((max(224, round(img.width * scale)), max(224, round(img.height * scale))))
    left, top = (img.width - 224) // 2, (img.height - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    axes[0].imshow(img)
    axes[0].set_title("input")
    for i, word in enumerate(words, start=1):
        heat = Image.fromarray(np.uint8(attention_to_grid(attn[i - 1]) * 255)).resize((224, 224), Image.BICUBIC)
        axes[i].imshow(img)
        axes[i].imshow(np.asarray(heat), cmap="jet", alpha=0.45)
        axes[i].set_title(word)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
