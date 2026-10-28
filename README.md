# Image Captioning with a Vision Encoder and GPT-2

Generates natural-language captions for images by pairing a pretrained image encoder (**ResNet-50** or **CLIP ViT-B/32**) with a **GPT-2** decoder that attends to image regions through cross-attention. It shows, word by word, which part of the image the model looked at.

This is a rebuilt and extended version of our Kent State University course project *Image Captioning Using Transformer* (see [Credits](#credits)). The original ResNet-50 + GPT-2 model reached **0.395 CIDEr** on MS COCO after 6 epochs. This version keeps that model as a baseline and adds a CLIP encoder, a standard benchmark split, a stronger training recipe and a demo.

> **Status:** code complete and tested on synthetic data. Full MS COCO results will be added after training (see [Results](#results)).

---

## What's new compared to the original project

| | Original project | This version |
|---|---|---|
| Image encoder | ResNet-50 | ResNet-50 **and** CLIP ViT-B/32 (switch in the config) |
| Data split | Custom 80/20 split | **Karpathy split** (113K train / 5K val / 5K test), the standard used by published results; a same-recipe split can be built from the official caption files when the Karpathy file isn't available |
| Encoder training | Frozen | Frozen first, then the last block(s) fine-tuned at a lower learning rate |
| Training recipe | AdamW, scheduler, gradient clipping | Adds cosine schedule with warmup, label smoothing and mixed precision |
| Model selection | Training loss | Best **validation CIDEr** |
| Decoding | Beam search | Beam search with length penalty and **n-gram blocking** (stops repeats like "a vase of flowers … a vase of flowers") |
| Interpretability | Attention maps (not shown in report) | Word-by-word attention heatmaps aligned to the image grid |
| Demo | None | Gradio web app |
| Tests | None | Unit tests and an end-to-end training test |

## How it works

```
image ──► encoder (ResNet-50: 7×7 grid  |  CLIP: 7×7 patch grid) ──► 49 region features
                                                                          │
                                         Linear + LayerNorm projection ◄──┘
                                                                          │
caption so far ──► GPT-2 (self-attention + cross-attention to regions) ──► next word
```

1. The encoder turns a 224×224 image into a 7×7 grid of region features.
2. A learned projection maps them into GPT-2's 768-dimensional space.
3. GPT-2's cross-attention layers attend over the 49 regions while predicting each next word. They're trained with teacher forcing and cross-entropy.
4. At inference, beam search builds the caption, and the cross-attention weights become the heatmaps.

## Results

*To be filled in after training on MS COCO (Karpathy test split, 5,000 images).* `python scripts/results_table.py runs` prints this table from the evaluation outputs.

| Encoder | Decoding | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|---|
| ResNet-50 | greedy | | | | |
| ResNet-50 | beam 3 | | | | |
| CLIP ViT-B/32 | greedy | | | | |
| CLIP ViT-B/32 | beam 3 | | | | |
| *Original course project (ResNet-50, custom split)* | *beam* | *0.31* | *0.27* | *0.54* | *0.395* |

## Project structure

```
captioner/
  config.py      default settings, merged with a YAML config
  model.py       ResNet-50 / CLIP encoders, projection, GPT-2 decoder with cross-attention
  data.py        datasets, image transforms, caption tokenization, batching
  decoding.py    greedy and beam search with n-gram blocking; attention extraction
  metrics.py     BLEU, METEOR, ROUGE-L, CIDEr (official COCO caption toolkit)
  utils.py       tokenizer, checkpoints, attention heatmaps
scripts/
  prepare_data.py   Karpathy split → train/val/test JSONL
  train.py          training loop; keeps the best checkpoint by validation CIDEr
  evaluate.py       scores a checkpoint on the test split
  caption.py        captions one image and saves an attention figure
  results_table.py  builds the results table from evaluation outputs
configs/            resnet50.yaml (baseline), clip.yaml (upgrade)
notebooks/          train_on_kaggle.ipynb, a ready-to-run free-GPU training notebook
app.py              Gradio demo
tests/              unit tests and an end-to-end train → evaluate → caption test
```

## Getting started

### 1. Install
```bash
git clone https://github.com/PiyushaKadam07/Image-Captioning-Transformer.git
cd Image-Captioning-Transformer
pip install -r requirements.txt
```
METEOR and the official COCO tokenizer need Java. Without it, the other metrics still run.

### 2. Get the data
- **MS COCO 2014** images (`train2014/`, `val2014/`) from [cocodataset.org](https://cocodataset.org/#download)
- The **Karpathy split** file `dataset_coco.json` (preferred), or the official `captions_train2014.json` and `captions_val2014.json`

```bash
python scripts/prepare_data.py --karpathy path/to/dataset_coco.json --out data
# or, without the Karpathy file (same recipe: 5K val + 5K test from val2014, fixed seed):
python scripts/prepare_data.py --coco-train captions_train2014.json --coco-val captions_val2014.json --out data
ln -s /path/to/coco2014 data/images      # the folder containing train2014/ and val2014/
```

### 3. Train
A GPU is needed for full training. The easiest free option is the Kaggle notebook in `notebooks/train_on_kaggle.ipynb`.
```bash
python scripts/train.py --config configs/resnet50.yaml                      # baseline
python scripts/train.py --config configs/clip.yaml                          # upgrade
python scripts/train.py --config configs/clip.yaml --subset 5000 --epochs 1  # quick check
```
Each epoch uses one random reference caption per image, so every caption is still seen across epochs at a fifth of the compute. Set `train.all_captions: true` to use all five per epoch.

### 4. Evaluate
```bash
python scripts/evaluate.py --checkpoint runs/clip/best.pt                 # beam search
python scripts/evaluate.py --checkpoint runs/clip/best.pt --beam-size 1   # greedy
python scripts/results_table.py runs
```

### 5. Caption your own image
```bash
python scripts/caption.py --checkpoint runs/clip/best.pt --image photo.jpg --attention attention.png
CHECKPOINT=runs/clip/best.pt python app.py      # web demo
```

## Tests
```bash
pytest tests
```
The tests use a tiny randomly initialized model and a byte-level tokenizer, so they run offline on a CPU. The end-to-end test trains on red and blue images, checks the model learns to caption them correctly, then runs the evaluation and captioning scripts. It takes a few minutes.

## Limitations
- Captions are limited to what MS COCO describes: everyday objects and scenes, in English.
- COCO captions carry cultural and gender biases, which a model trained on them can repeat.
- Training uses cross-entropy only. Self-critical sequence training (optimizing CIDEr directly, Rennie et al. 2017) is a natural next step.

## Credits
The original course project (Kent State University) was built by:
- **Piyusha Kadam**: MS COCO data pipeline (preprocessing, augmentation, image-caption alignment), GPT-2 tokenization and padding pipeline, ResNet-50 feature-extractor integration, literature review
- **Varun Nagarkar**: training loop, hyperparameter tuning, evaluation scripts, attention visualization
- **Suwei Gao**: GPT-2 cross-attention decoder, decoding strategies, ablation studies

This rebuilt version was implemented by Piyusha Kadam.

Built on [PyTorch](https://pytorch.org/), [Hugging Face Transformers](https://huggingface.co/docs/transformers) (GPT-2, CLIP), [torchvision](https://pytorch.org/vision/) (ResNet-50) and [pycocoevalcap](https://github.com/salaniz/pycocoevalcap). Dataset: [MS COCO](https://cocodataset.org/).

## License
MIT
