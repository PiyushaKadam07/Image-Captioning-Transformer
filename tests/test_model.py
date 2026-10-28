import pytest
import torch
from conftest import tiny_overrides

from captioner.config import load_config
from captioner.data import collate_train
from captioner.model import Captioner


@pytest.mark.parametrize("encoder,regions", [("resnet50", 49), ("clip", 49)])
def test_forward_and_backward(encoder, regions):
    model = Captioner(load_config(overrides=tiny_overrides(encoder)))
    images = torch.randn(2, 3, 224, 224)
    feats = model.encode(images)
    assert feats.shape == (2, regions, 64)  # projected into the decoder's width
    batch = collate_train([(images[0], [256, 5, 6, 7, 256]), (images[1], [256, 8, 256])], pad_id=256)
    loss = model(*batch, label_smoothing=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.proj[0].weight.grad is not None


def test_unfreeze_last_blocks_only():
    model = Captioner(load_config(overrides=tiny_overrides("resnet50")))
    model.encoder.unfreeze_last(1)
    trainable = {n.split(".")[1] for n, p in model.encoder.named_parameters() if p.requires_grad}
    assert trainable == {"7"}  # layer4 is the 8th child of the ResNet body
