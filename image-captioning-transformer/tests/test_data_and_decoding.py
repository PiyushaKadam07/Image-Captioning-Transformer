import torch
from conftest import tiny_overrides

from captioner.config import load_config
from captioner.data import collate_train
from captioner.decoding import _banned_tokens, generate
from captioner.metrics import score_captions
from captioner.model import Captioner


def test_collate_shifts_labels_for_teacher_forcing():
    images = torch.zeros(2, 3, 8, 8)
    _, input_ids, mask, labels = collate_train([(images[0], [9, 1, 2, 9]), (images[1], [9, 3, 9])], pad_id=9)
    assert input_ids.tolist() == [[9, 1, 2], [9, 3, 9]]
    assert labels.tolist() == [[1, 2, 9], [3, 9, -100]]
    assert mask.tolist() == [[1, 1, 1], [1, 1, 0]]


def test_no_repeat_ngram_ban():
    assert _banned_tokens([1, 2, 3, 1, 2], 3) == {3}
    assert _banned_tokens([1, 2], 3) == set()


def test_generate_shapes_and_attention():
    model = Captioner(load_config(overrides=tiny_overrides("clip"))).eval()
    feats = model.encode(torch.randn(3, 3, 224, 224))
    outs, attn = generate(model, feats, 256, 256, beam_size=3, max_len=6, return_attention=True)
    assert len(outs) == 3 and len(attn) == 3
    for toks, a in zip(outs, attn):
        assert 256 not in toks and len(toks) <= 6
        assert a.shape == (len(toks), 49)
        if len(toks):
            assert torch.allclose(a.sum(-1), torch.ones(len(toks)), atol=1e-4)


def test_metrics_reward_exact_matches():
    refs = {1: ["a dog runs on grass", "a dog running"], 2: ["a red car parked", "a parked red car"]}
    good = score_captions(refs, {1: "a dog runs on grass", 2: "a red car parked"}, meteor=False)
    bad = score_captions(refs, {1: "a red car parked", 2: "a dog runs on grass"}, meteor=False)
    assert good["CIDEr"] > bad["CIDEr"] and good["BLEU-1"] > bad["BLEU-1"]
