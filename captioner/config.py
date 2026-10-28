"""Configuration loading with sensible defaults."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "model": {
        "encoder": "resnet50",            # "resnet50" or "clip"
        "clip_name": "openai/clip-vit-base-patch32",
        "decoder_name": "gpt2",
        "pretrained": True,               # False builds randomly initialised models (tests only)
        "load_weights": True,             # False when weights will be restored from a checkpoint
        "decoder_overrides": {},          # e.g. {"n_layer": 2} for tiny test models
        "encoder_dropout": 0.1,
    },
    "data": {
        "train": "data/train.jsonl",
        "val": "data/val.jsonl",
        "test": "data/test.jsonl",
        "image_root": "data/images",
        "image_size": 224,
        "max_tokens": 30,
        "num_workers": 4,
    },
    "train": {
        "epochs": 15,
        "all_captions": False,            # False: one random caption per image per epoch
        "batch_size": 64,
        "grad_accum": 1,
        "lr_decoder": 1.0e-4,
        "lr_encoder": 1.0e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "label_smoothing": 0.1,
        "grad_clip": 1.0,
        "amp": True,
        "freeze_encoder_epochs": 4,       # encoder frozen at first, then partly fine-tuned
        "unfreeze_last_blocks": 1,
        "val_images": 1000,               # images used for per-epoch validation CIDEr
        "max_steps": None,                # cap optimizer steps (smoke tests)
        "seed": 42,
        "output_dir": "runs/resnet50",
    },
    "decode": {
        "beam_size": 3,
        "max_len": 30,
        "length_penalty": 1.0,
        "no_repeat_ngram": 3,
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    return _merge(cfg, overrides or {})
