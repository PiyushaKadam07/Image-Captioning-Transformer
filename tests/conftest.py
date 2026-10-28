import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLORS = {"red": (220, 30, 30), "blue": (30, 60, 220)}


def tiny_overrides(encoder: str, tmp: Path | None = None) -> dict:
    """A randomly initialised, CPU-sized model that needs no downloads."""
    cfg = {
        "model": {"encoder": encoder, "pretrained": False, "tokenizer": "bytes",
                  "decoder_overrides": {"n_layer": 2, "n_head": 2, "n_embd": 64, "vocab_size": 260,
                                        "n_positions": 64, "bos_token_id": 256, "eos_token_id": 256}},
        "decode": {"beam_size": 2, "max_len": 16, "no_repeat_ngram": 0},
    }
    if tmp is not None:
        cfg["data"] = {"train": str(tmp / "train.jsonl"), "val": str(tmp / "val.jsonl"),
                       "test": str(tmp / "test.jsonl"), "image_root": str(tmp / "images"),
                       "num_workers": 0, "max_tokens": 20}
    return cfg


@pytest.fixture
def toy_dataset(tmp_path):
    """Solid red and blue images captioned 'red' and 'blue'."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    records = []
    for i in range(8):
        color = "red" if i % 2 == 0 else "blue"
        arr = np.zeros((64, 64, 3), dtype=np.uint8) + np.array(COLORS[color], dtype=np.uint8)
        name = f"img{i}.png"
        Image.fromarray(arr).save(img_dir / name)
        records.append({"image_id": i, "image": name, "captions": [color]})
    for split in ("train", "val", "test"):
        with open(tmp_path / f"{split}.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    return tmp_path
