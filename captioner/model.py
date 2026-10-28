"""ResNet-50 / CLIP image encoder + GPT-2 decoder with cross-attention."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionConfig, CLIPVisionModel, GPT2Config, GPT2LMHeadModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ImageEncoder(nn.Module):
    """Returns a grid of region features, shape (batch, regions, dim)."""

    def __init__(self, kind: str = "resnet50", clip_name: str = "openai/clip-vit-base-patch32",
                 pretrained: bool = True, load_weights: bool = True):
        super().__init__()
        self.kind = kind
        if kind == "resnet50":
            import torchvision
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if (pretrained and load_weights) else None
            backbone = torchvision.models.resnet50(weights=weights)
            # Drop the pooling and classification head: keep the 7x7 spatial feature map.
            self.body = nn.Sequential(*list(backbone.children())[:-2])
            self.blocks = [backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]
            self.out_dim = 2048
            self.mean, self.std = IMAGENET_MEAN, IMAGENET_STD
        elif kind == "clip":
            if pretrained and load_weights:
                self.clip = CLIPVisionModel.from_pretrained(clip_name)
            elif pretrained:  # same architecture, weights restored later from a checkpoint
                self.clip = CLIPVisionModel(CLIPVisionConfig.from_pretrained(clip_name))
            else:
                self.clip = CLIPVisionModel(CLIPVisionConfig(
                    hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                    num_attention_heads=2, image_size=224, patch_size=32))
            self.blocks = list(self._vision().encoder.layers)
            self.out_dim = self.clip.config.hidden_size
            self.mean, self.std = CLIP_MEAN, CLIP_STD
        else:
            raise ValueError(f"Unknown encoder '{kind}'. Use 'resnet50' or 'clip'.")

    def _vision(self):
        # Older Transformers versions wrap the layers in `.vision_model`; newer ones do not.
        return getattr(self.clip, "vision_model", self.clip)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.kind == "resnet50":
            fmap = self.body(images)                       # (B, 2048, 7, 7)
            return fmap.flatten(2).transpose(1, 2)         # (B, 49, 2048)
        hidden = self.clip(pixel_values=images).last_hidden_state  # (B, 1 + 49, D)
        return hidden[:, 1:, :]                            # patch tokens only, so attention maps align with the image grid

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_last(self, n_blocks: int) -> None:
        self.freeze()
        for block in self.blocks[-n_blocks:] if n_blocks > 0 else []:
            for p in block.parameters():
                p.requires_grad = True
        if self.kind == "clip":  # keep the final layer norm trainable with the last blocks
            for p in self._vision().post_layernorm.parameters():
                p.requires_grad = n_blocks > 0


class Captioner(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        mcfg = cfg["model"]
        load_weights = mcfg.get("load_weights", True)
        self.encoder = ImageEncoder(mcfg["encoder"], mcfg["clip_name"], mcfg["pretrained"], load_weights)
        if mcfg["pretrained"]:
            dec_cfg = GPT2Config.from_pretrained(mcfg["decoder_name"], add_cross_attention=True,
                                                 **mcfg["decoder_overrides"])
            if load_weights:
                self.decoder = GPT2LMHeadModel.from_pretrained(
                    mcfg["decoder_name"], config=dec_cfg, attn_implementation="eager")
            else:
                self.decoder = GPT2LMHeadModel._from_config(dec_cfg, attn_implementation="eager")
        else:
            dec_cfg = GPT2Config(add_cross_attention=True, **mcfg["decoder_overrides"])
            self.decoder = GPT2LMHeadModel._from_config(dec_cfg, attn_implementation="eager")
        hidden = self.decoder.config.n_embd
        # Learned bridge from image feature space into GPT-2's embedding space.
        self.proj = nn.Sequential(
            nn.Linear(self.encoder.out_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Dropout(mcfg["encoder_dropout"]),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.proj(self.encoder(images))

    def decode_logits(self, input_ids, feats, attention_mask=None, output_attentions=False):
        return self.decoder(input_ids=input_ids, attention_mask=attention_mask,
                            encoder_hidden_states=feats, output_attentions=output_attentions,
                            use_cache=False)

    def forward(self, images, input_ids, attention_mask, labels, label_smoothing: float = 0.0):
        feats = self.encode(images)
        logits = self.decode_logits(input_ids, feats, attention_mask).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                               ignore_index=-100, label_smoothing=label_smoothing)
        return loss
