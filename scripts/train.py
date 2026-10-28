"""Train the captioner. Picks the best epoch by validation CIDEr.

Usage:
    python scripts/train.py --config configs/clip.yaml
    python scripts/train.py --config configs/resnet50.yaml --subset 2000 --epochs 1   # quick check
"""
import argparse
import csv
import json
import math
import sys
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from captioner.config import load_config  # noqa: E402
from captioner.data import (CaptionEvalDataset, CaptionTrainDataset, build_transform,  # noqa: E402
                            collate_eval, collate_train, read_jsonl)
from captioner.decoding import generate  # noqa: E402
from captioner.metrics import score_captions  # noqa: E402
from captioner.model import Captioner  # noqa: E402
from captioner.utils import get_tokenizer, save_checkpoint, set_seed  # noqa: E402


def evaluate_split(model, tok, records, cfg, device, limit=None, beam_size=1):
    """Caption images and score them against their references."""
    tf = build_transform(model.encoder.mean, model.encoder.std, cfg["data"]["image_size"])
    ds = CaptionEvalDataset(records, cfg["data"]["image_root"], tf, limit=limit)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=cfg["data"]["num_workers"],
                    collate_fn=collate_eval)
    dcfg = cfg["decode"]
    hyps = {}
    model.eval()
    with torch.no_grad():
        for images, ids in dl:
            feats = model.encode(images.to(device))
            outs = generate(model, feats, tok.eos_token_id, tok.eos_token_id, beam_size=beam_size,
                            max_len=dcfg["max_len"], length_penalty=dcfg["length_penalty"],
                            no_repeat_ngram=dcfg["no_repeat_ngram"])
            for image_id, toks in zip(ids, outs):
                hyps[image_id] = tok.decode(toks).strip()
    refs = {r["image_id"]: r["captions"] for r in ds.records}
    return score_captions(refs, hyps), hyps


def param_groups(model, cfg):
    t = cfg["train"]
    enc = [p for p in model.encoder.parameters()]
    rest = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    return [
        {"params": rest, "lr": t["lr_decoder"]},
        {"params": enc, "lr": t["lr_encoder"]},
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--subset", type=int, default=None, help="Use only N training captions (quick test)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    overrides = {"train": {}}
    if args.epochs:
        overrides["train"]["epochs"] = args.epochs
    if args.output_dir:
        overrides["train"]["output_dir"] = args.output_dir
    cfg = load_config(args.config, overrides)
    t, d = cfg["train"], cfg["data"]
    set_seed(t["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(t["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(out_dir / "config.json", "w"), indent=2)

    tok = get_tokenizer(cfg)
    model = Captioner(cfg).to(device)
    model.encoder.freeze()

    train_records = read_jsonl(d["train"])
    val_records = read_jsonl(d["val"])
    train_ds = CaptionTrainDataset(train_records, d["image_root"], tok,
                                   build_transform(model.encoder.mean, model.encoder.std, d["image_size"], train=True),
                                   d["max_tokens"], all_captions=t["all_captions"])
    if args.subset:
        train_ds = Subset(train_ds, range(min(args.subset, len(train_ds))))
    train_dl = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, num_workers=d["num_workers"],
                          collate_fn=partial(collate_train, pad_id=tok.eos_token_id), drop_last=False,
                          pin_memory=device == "cuda")

    steps_per_epoch = math.ceil(len(train_dl) / t["grad_accum"])
    total_steps = steps_per_epoch * t["epochs"]
    if t["max_steps"]:
        total_steps = min(total_steps, t["max_steps"])
    optimizer = torch.optim.AdamW(param_groups(model, cfg), weight_decay=t["weight_decay"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * t["warmup_ratio"]), total_steps)
    use_amp = t["amp"] and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log_path = out_dir / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_CIDEr", "val_BLEU-4", "seconds"])

    best_cider, step = -1.0, 0
    for epoch in range(1, t["epochs"] + 1):
        if epoch == t["freeze_encoder_epochs"] + 1 and t["unfreeze_last_blocks"] > 0:
            model.encoder.unfreeze_last(t["unfreeze_last_blocks"])
            print(f"Epoch {epoch}: fine-tuning the last {t['unfreeze_last_blocks']} encoder block(s)")
        model.train()
        # Keep pretrained BatchNorm statistics fixed (frozen layers must not drift).
        for m in model.encoder.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
        if not any(p.requires_grad for p in model.encoder.parameters()):
            model.encoder.eval()
        start, running, n = time.time(), 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for i, (images, input_ids, mask, labels) in enumerate(train_dl, 1):
            images, input_ids = images.to(device, non_blocking=True), input_ids.to(device)
            mask, labels = mask.to(device), labels.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                loss = model(images, input_ids, mask, labels, t["label_smoothing"]) / t["grad_accum"]
            scaler.scale(loss).backward()
            running += loss.item() * t["grad_accum"]
            n += 1
            if i % t["grad_accum"] == 0 or i == len(train_dl):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 100 == 0:
                    print(f"epoch {epoch} step {step}/{total_steps} loss {running / n:.3f}", flush=True)
                if t["max_steps"] and step >= t["max_steps"]:
                    break

        scores, _ = evaluate_split(model, tok, val_records, cfg, device, limit=t["val_images"])
        elapsed = time.time() - start
        print(f"Epoch {epoch}: train loss {running / max(n, 1):.3f} | val CIDEr {scores['CIDEr']:.3f} "
              f"| val BLEU-4 {scores['BLEU-4']:.3f} | {elapsed / 60:.1f} min", flush=True)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, round(running / max(n, 1), 4), round(scores["CIDEr"], 4),
                                    round(scores["BLEU-4"], 4), round(elapsed)])
        save_checkpoint(out_dir / "last.pt", model, cfg, {"epoch": epoch, "val": scores})
        if scores["CIDEr"] > best_cider:
            best_cider = scores["CIDEr"]
            save_checkpoint(out_dir / "best.pt", model, cfg, {"epoch": epoch, "val": scores})
            print(f"  new best checkpoint (val CIDEr {best_cider:.3f})")
        if t["max_steps"] and step >= t["max_steps"]:
            break
    print(f"Done. Best val CIDEr {best_cider:.3f}. Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
