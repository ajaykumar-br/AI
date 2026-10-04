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
│   ├── attention.py     # one causal self-attention head
│   ├── multi_head_attention.py  # MultiHeadAttention: several SelfAttention heads + output projection
│   ├── layer_norm.py    # LayerNorm written from scratch
│   ├── feed_forward.py  # FeedForward: Linear → GELU → Linear
│   ├── block.py         # TransformerBlock: residual + LayerNorm around multi-head attention and feed-forward
│   ├── model.py         # TinyLanguageModel (current architecture)
│   ├── attention_only_model.py  # the first model, kept so its checkpoint still loads
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
└── checkpoints/     # one .pt, _history.json and _history.png per model (not committed)
```

## Model

```
input_ids (B, T)
  ├─ TokenEmbedding       vocab → 64
  └─ PositionalEmbedding  positions 0..T-1 → 64 (learned)
        ↓ add
  TransformerBlock × 2 (pre-LayerNorm)
  │   x = x + MultiHeadAttention(LayerNorm(x))   4 heads × 16 dims, causal mask
  │   x = x + FeedForward(LayerNorm(x))          64 → 256 → 64, GELU
        ↓
  LayerNorm
        ↓
  Linear 64 → vocab   →   logits (B, T, vocab)
```

With the current corpus (vocabulary of 8,727 words, context of 32 tokens) the model has about **1.23M parameters**. About 91% of them are in the token embedding and the output layer; each transformer block is about 50k (attention 17k, feed-forward 33k).

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

Settings are at the top of the script: sequence length 32, embedding 64, 4 heads, 2 blocks, batch size 64, 10 epochs, Adam with lr 1e-3. The last 10% of the corpus is held out as a validation set and never trained on. After every epoch the script writes:
- `checkpoints/tiny_lm_2block_4head.pt`: weights, optimizer state, vocabulary and config
- `checkpoints/tiny_lm_2block_4head_history.json`: train/val loss and accuracy per epoch

The file name comes from `num_layers` and `num_heads`, so changing them starts a new file instead of overwriting the last model. The first, attention-only model is kept as `checkpoints/tiny_lm_attention_only.pt`.

## 3. Plot

```bash
uv run python scripts/plot_history.py            # opens a window and saves checkpoints/tiny_lm_2block_4head_history.png
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
| `--checkpoint` | `checkpoints/tiny_lm_2block_4head.pt` | Model to load; use `checkpoints/tiny_lm_attention_only.pt` for the first model |
| `--corpus` | `data/processed/corpus.txt` | Used to rebuild the same validation split |
| `--examples` | `5` | Number of example next-word predictions to show |
| `--prompt` | `"today we are going to talk about"` | Starting text for generation |
| `--num-tokens` | `30` | Tokens to generate |
| `--temperature` | `0.8` | Lower is safer and more repetitive; higher is more varied |
| `--seed` | none | Fix the random seed so generation repeats; leave unset for different text each run |

## Results

These results are from the earlier attention-only model (no residuals, feed-forward or LayerNorm). 10 epochs on a GTX 1660 Ti, using about 738k tokens (664k train / 74k validation):

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

The model is two small transformer blocks trained on under a million words. Improvements, roughly in order of impact:
- more training data
- dropout, before trying more blocks
- a subword tokenizer (BPE) instead of word-level

The [`../Transformer`](../Transformer) notebook implements most of these.
