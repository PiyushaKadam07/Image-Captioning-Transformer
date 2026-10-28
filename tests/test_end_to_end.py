"""Train a tiny model on red/blue images, then evaluate and caption with the real scripts."""
import json
import subprocess
import sys

import yaml
from conftest import ROOT, tiny_overrides


def _run(args):
    result = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_train_evaluate_caption(toy_dataset):
    cfg = tiny_overrides("clip", toy_dataset)
    cfg["train"] = {"epochs": 40, "batch_size": 8, "lr_decoder": 3e-3, "lr_encoder": 1e-3,
                    "warmup_ratio": 0.0, "label_smoothing": 0.0, "freeze_encoder_epochs": 20,
                    "unfreeze_last_blocks": 1, "val_images": 8, "amp": False,
                    "output_dir": str(toy_dataset / "run")}
    cfg_path = toy_dataset / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    log = _run(["scripts/train.py", "--config", str(cfg_path)])
    assert "new best checkpoint" in log
    ckpt = toy_dataset / "run" / "best.pt"
    assert ckpt.exists()

    out = _run(["scripts/evaluate.py", "--checkpoint", str(ckpt), "--beam-size", "2"])
    metrics = json.loads((toy_dataset / "run" / "test_beam2_metrics.json").read_text())
    assert metrics["images"] == 8
    assert metrics["scores"]["BLEU-1"] > 0.99, out  # the model has learned red vs blue

    out = _run(["scripts/caption.py", "--checkpoint", str(ckpt), "--image",
                str(toy_dataset / "images" / "img1.png"), "--attention", str(toy_dataset / "attn.png")])
    assert out.splitlines()[0].strip() == "blue"
    assert (toy_dataset / "attn.png").exists()
