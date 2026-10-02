# Mini Transformer

A tiny next-word language model built from scratch in PyTorch and trained on a corpus of YouTube video transcripts. Each stage of the pipeline lives in its own small module, so every step can be read and tested on its own.

```
corpus → tokenize → build vocabulary → encode → create_training_data
       → dataloader → TinyLanguageModel → logits → CrossEntropyLoss
       → backpropagation → optimizer
```

## Project layout

```
mini-transformer/
├── src/mini_transformer/
│   ├── cleaning.py      # transcript snippets → one text string
│   ├── tokenizer.py     # lowercase + split words/punctuation
│   ├── vocabulary.py    # build_vocabulary (stoi/itos with <PAD>, <UNK>), encode, decode
│   ├── dataset.py       # create_training_data: sliding windows, target shifted by 1
│   ├── dataloader.py    # create_dataloader: wraps inputs/targets in a PyTorch DataLoader
│   ├── embeddings.py    # TokenEmbedding, learnable PositionalEmbedding
│   ├── attention.py     # single-head causal SelfAttention
│   ├── model.py         # TinyLanguageModel
│   ├── loss.py          # calculate_loss (CrossEntropyLoss over flattened logits)
│   ├── training.py      # simple full-batch train loop (used by scripts/train.py)
│   └── evaluation.py    # loss, perplexity, top-1 / top-k accuracy
├── scripts/
│   ├── download_transcripts.py  # data/raw/videos.txt → data/raw/transcripts/*.txt
│   ├── process_transcript.py    # normalise whitespace → data/processed/*.txt
│   ├── build_corpus.py          # join processed transcripts → data/processed/corpus.txt
│   ├── train.py                 # toy run on two hard-coded sentences
│   ├── train_corpus.py          # full training on the corpus (GPU, val split, checkpoints)
│   ├── plot_history.py          # plot train/val loss & accuracy
│   └── evaluate.py              # load checkpoint, report metrics, show predictions, generate text
├── data/            # raw + processed transcripts (contents not committed)
└── checkpoints/     # tiny_lm.pt, history.json, history.png (not committed)
```

## Model

```
input_ids (B, T)
  ├─ TokenEmbedding       vocab → 64
  └─ PositionalEmbedding  positions 0..T-1 → 64 (learned)
        ↓ add
  SelfAttention (1 head, 64 → 64, causal mask, scaled by √64)
        ↓
  Linear 64 → vocab   →   logits (B, T, vocab)
```

With the current corpus (vocabulary of 8,727 words, context of 32 tokens) the model has about **1.14M parameters**. About 99% of them are in the token embedding and the output layer; the attention itself is about 12k.

## Setup

Needs [uv](https://docs.astral.sh/uv/) and Python 3.10+. `pyproject.toml` installs PyTorch from the CUDA 12.6 index on Windows and Linux, so an NVIDIA GPU is used automatically. Run every command from the `mini-transformer/` folder.

```bash
cd mini-transformer
uv sync
```

## 1. Build the corpus

The transcripts aren't committed. To rebuild them, put one YouTube URL per line in `data/raw/videos.txt`, then run:

```bash
uv run python scripts/download_transcripts.py   # → data/raw/transcripts/
uv run python scripts/process_transcript.py     # → data/processed/
uv run python scripts/build_corpus.py           # → data/processed/corpus.txt
```

## 2. Train

```bash
uv run python scripts/train_corpus.py
```

Settings are at the top of the script: sequence length 32, embedding 64, batch size 64, 10 epochs, Adam with lr 1e-3. The last 10% of the corpus is held out as a validation set and never trained on. After every epoch the script writes:
- `checkpoints/tiny_lm.pt`: weights, optimizer state, vocabulary and config
- `checkpoints/history.json`: train/val loss and accuracy per epoch

## 3. Plot

```bash
uv run python scripts/plot_history.py            # opens a window and saves checkpoints/history.png
uv run python scripts/plot_history.py --no-show  # save only
```

## 4. Evaluate

```bash
uv run python scripts/evaluate.py
uv run python scripts/evaluate.py --prompt "the best way to learn coding is" --num-tokens 40 --temperature 0.8
uv run python scripts/evaluate.py --help
```

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint` | `checkpoints/tiny_lm.pt` | Model to load |
| `--corpus` | `data/processed/corpus.txt` | Used to rebuild the same validation split |
| `--examples` | `5` | Number of example next-word predictions to show |
| `--prompt` | `"today we are going to talk about"` | Starting text for generation |
| `--num-tokens` | `30` | Tokens to generate |
| `--temperature` | `0.8` | Lower is safer and more repetitive; higher is more varied |

## Results

10 epochs on a GTX 1660 Ti, using about 738k tokens (664k train / 74k validation):

| Metric (validation) | Model | Baseline |
|---|---|---|
| Loss | 3.41 | – |
| Perplexity | 30.1 | 8,727 (uniform random guess) |
| Top-1 accuracy | 30.9% | 4.2% (always predict ".") |
| Top-5 accuracy | 55.2% | – |

Validation loss was still falling at epoch 10, so the model was not yet overfitting. The train–val gap grew from 0.12 to 0.39.

Sample generation (temperature 0.8):

> today we are going to talk about the other people who are also in a sandbox somewhere here . um requires and you know a little bit of ego than that i am i generated . um

The model picks up local phrasing and the transcript style ("um", "you know"), but it can't hold a thought together.

## Limitations & next steps

This is deliberately a *partial* transformer block. Improvements, roughly in order of impact:
- residual connections around attention
- a feed-forward (MLP) sublayer
- LayerNorm
- multiple attention heads
- stacking several blocks
- a subword tokenizer (BPE) instead of word-level

The [`../Transformer`](../Transformer) notebook implements most of these.
