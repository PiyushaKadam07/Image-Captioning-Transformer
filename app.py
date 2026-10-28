"""Gradio demo: upload an image, get a caption and word-by-word attention maps.

Run locally:   CHECKPOINT=runs/clip/best.pt python app.py
On Hugging Face Spaces, upload best.pt next to this file (or set CHECKPOINT).
"""
import os
import tempfile

import gradio as gr
import torch

from captioner.utils import get_tokenizer, load_checkpoint, plot_attention
from scripts.caption import caption_image

CHECKPOINT = os.environ.get("CHECKPOINT", "best.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model, cfg = load_checkpoint(CHECKPOINT, DEVICE)
tok = get_tokenizer(cfg)


def run(image, beam_size):
    if image is None:
        return "Upload an image to get a caption.", None
    with torch.no_grad():
        caption, words, attn = caption_image(model, tok, cfg, image, DEVICE, int(beam_size))
    path = os.path.join(tempfile.mkdtemp(), "attention.png")
    plot_attention(image, words, attn, path)
    return caption, path


demo = gr.Interface(
    fn=run,
    inputs=[gr.Image(type="pil", label="Image"),
            gr.Slider(1, 5, value=cfg["decode"]["beam_size"], step=1, label="Beam size")],
    outputs=[gr.Textbox(label="Caption"), gr.Image(label="Where the model looked for each word")],
    title="Image Captioning with a Vision Encoder + GPT-2",
    description=f"Encoder: {cfg['model']['encoder']}. Each panel shows the image regions the decoder "
                "attended to while generating that word.",
)

if __name__ == "__main__":
    demo.launch()
