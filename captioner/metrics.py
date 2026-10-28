"""COCO caption metrics (BLEU, METEOR, ROUGE-L, CIDEr) via pycocoevalcap."""
from __future__ import annotations

import re
import shutil

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge


def _simple_tokenize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _tokenize(refs: dict, hyps: dict, use_java: bool):
    if use_java:
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        tok = PTBTokenizer()
        refs = tok.tokenize({k: [{"caption": c} for c in v] for k, v in refs.items()})
        hyps = tok.tokenize({k: [{"caption": v}] for k, v in hyps.items()})
        return refs, hyps
    return ({k: [_simple_tokenize(c) for c in v] for k, v in refs.items()},
            {k: [_simple_tokenize(v)] for k, v in hyps.items()})


def score_captions(refs: dict, hyps: dict, meteor: bool = True) -> dict:
    """refs: {image_id: [ref captions]}, hyps: {image_id: caption}."""
    ids = [i for i in hyps if i in refs]
    refs, hyps = {i: refs[i] for i in ids}, {i: hyps[i] for i in ids}
    java = shutil.which("java") is not None
    refs, hyps = _tokenize(refs, hyps, java)
    results = {}
    bleu, _ = Bleu(4).compute_score(refs, hyps)
    for n, value in enumerate(bleu, 1):
        results[f"BLEU-{n}"] = value
    results["ROUGE-L"], _ = Rouge().compute_score(refs, hyps)
    results["CIDEr"], _ = Cider().compute_score(refs, hyps)
    if meteor and java:
        from pycocoevalcap.meteor.meteor import Meteor
        results["METEOR"], _ = Meteor().compute_score(refs, hyps)
    return {k: float(v) for k, v in results.items()}
