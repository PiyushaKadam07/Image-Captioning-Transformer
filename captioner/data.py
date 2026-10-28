"""Datasets, image transforms and caption tokenization."""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_transform(mean, std, size: int = 224, train: bool = False):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def clean_caption(text: str) -> str:
    return " ".join(text.strip().lower().rstrip(".").split())


def encode_caption(tokenizer, text: str, max_tokens: int) -> list[int]:
    """[BOS] caption [EOS]; GPT-2 uses its EOS token as BOS."""
    ids = tokenizer.encode(" " + clean_caption(text))[: max_tokens - 2]
    return [tokenizer.eos_token_id] + ids + [tokenizer.eos_token_id]


class CaptionTrainDataset(Dataset):
    """Training examples.

    all_captions=True:  one example per (image, caption) pair (5x more steps per epoch).
    all_captions=False: one example per image, with a random reference caption each time it is
                        drawn. Every caption is still seen over several epochs, at a fifth of the
                        compute per epoch, which suits free GPU quotas.
    """

    def __init__(self, records, image_root, tokenizer, transform, max_tokens=30, all_captions=False):
        if all_captions:
            self.items = [(r["image"], [c]) for r in records for c in r["captions"]]
        else:
            self.items = [(r["image"], r["captions"]) for r in records]
        self.root, self.tok, self.tf, self.max_tokens = Path(image_root), tokenizer, transform, max_tokens

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        image, captions = self.items[i]
        caption = random.choice(captions)
        img = Image.open(self.root / image).convert("RGB")
        return self.tf(img), encode_caption(self.tok, caption, self.max_tokens)


class CaptionEvalDataset(Dataset):
    """One example per image (all reference captions kept for scoring)."""

    def __init__(self, records, image_root, transform, limit: int | None = None, seed: int = 0):
        records = list(records)
        if limit and limit < len(records):
            records = random.Random(seed).sample(records, limit)
        self.records, self.root, self.tf = records, Path(image_root), transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = Image.open(self.root / r["image"]).convert("RGB")
        return self.tf(img), r["image_id"]


def collate_train(batch, pad_id: int):
    images = torch.stack([b[0] for b in batch])
    longest = max(len(b[1]) for b in batch)
    input_ids = torch.full((len(batch), longest - 1), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), longest - 1), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), longest - 1), dtype=torch.long)
    for i, (_, ids) in enumerate(batch):
        n = len(ids) - 1
        input_ids[i, :n] = torch.tensor(ids[:-1])
        labels[i, :n] = torch.tensor(ids[1:])  # teacher forcing: predict the next token
        mask[i, :n] = 1
    return images, input_ids, mask, labels


def collate_eval(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]
