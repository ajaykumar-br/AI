# Mini Transformer

A tiny next-word language model built from scratch in PyTorch and trained on a corpus of YouTube video transcripts. Each stage of the pipeline lives in its own small module, so every step can be read and tested on its own.

```
corpus → tokenize (word-level or BPE) → token IDs → create_training_data
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
│   ├── bpe.py           # BPETokenizer: byte-level BPE (train, encode, decode), no <UNK>
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

Diagrams, a plain-language walkthrough and the parameter breakdown: [TinyLanguageModel, 2 blocks × 4 heads](https://claude.ai/artifact/WGhRjf88Mm3ezjf4thVsVR).

```
input_ids (B, T)
  ├─ TokenEmbedding       vocab → 64
  └─ PositionalEmbedding  positions 0..T-1 → 64 (learned)
        ↓ add
  TransformerBlock × 2 (pre-LayerNorm)
  │   x = x + MultiHeadAttention(LayerNorm(x))   4 heads × 16 dims, causal mask
  │   x = x + FeedForward(LayerNorm(x))          64 → 256 → 64, GELU
  │   optional dropout on embeddings, attention weights and each sublayer output (training only)
        ↓
  LayerNorm
        ↓
  Linear 64 → vocab   →   logits (B, T, vocab)
```

With the word-level tokenizer (vocabulary of 8,727 words, context of 32 tokens) the model has about **1.23M parameters**. About 91% of them are in the token embedding and the output layer; each transformer block is about 50k (attention 17k, feed-forward 33k).

With BPE (vocabulary of 2,048, context of 40 tokens) the same architecture has about **367k parameters**, because the token embedding and output layer shrink to a quarter of their size.

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

Settings are at the top of the script: BPE tokenizer with vocab size 2048, sequence length 40, embedding 64, 4 heads, 2 blocks, no dropout, batch size 64, 15 epochs, Adam with lr 1e-3. The last 10% of the corpus is held out as a validation set and never trained on. The script writes:
- `checkpoints/tiny_lm_2block_4head_bpe2048.pt`: weights, optimizer state, tokenizer and config, saved only when validation loss reaches a new best, so it always holds the best epoch (stored as `epoch` and `val_loss` in the checkpoint)
- `checkpoints/tiny_lm_2block_4head_bpe2048_history.json`: train/val loss and accuracy, updated every epoch

`tokenizer_type` picks the tokenizer:
- `"bpe"`: learns byte-level BPE merges from the lowercased corpus at the start of every run (about 40 s). The checkpoint stores `"tokenizer": "bpe"` and the learned `merges`. BPE gives about 1.18 tokens per word, so a context of 40 tokens covers about as much text as 32 words.
- `"word"`: the original word-level tokenizer. The checkpoint stores `"tokenizer": "word"`, `stoi` and `itos`.

The file name comes from `num_layers`, `num_heads`, the tokenizer and `dropout`, so changing them starts a new file instead of overwriting the last model. Earlier models are kept as `checkpoints/tiny_lm_2block_4head.pt` (word-level, no dropout), `checkpoints/tiny_lm_2block_4head_dropout0.1.pt` (word-level, dropout 0.1) and `checkpoints/tiny_lm_attention_only.pt` (the first, attention-only model).

With dropout above 0, the training loss and accuracy printed for each epoch are measured with dropout on, so they look worse than the model really is. Validation is measured with dropout off.

## 3. Plot

```bash
uv run python scripts/plot_history.py            # opens a window and saves checkpoints/tiny_lm_2block_4head_history.png
uv run python scripts/plot_history.py --history checkpoints/tiny_lm_2block_4head_dropout0.1_history.json
uv run python scripts/plot_history.py --no-show  # save only
```

## 4. Evaluate

```bash
uv run python scripts/evaluate.py
uv run python scripts/evaluate.py --checkpoint checkpoints/tiny_lm_2block_4head_bpe2048.pt
uv run python scripts/evaluate.py --prompt "the best way to learn coding is" --num-tokens 40 --temperature 0.8
uv run python scripts/evaluate.py --help
```

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint` | `checkpoints/tiny_lm_2block_4head.pt` | Model to load, e.g. `checkpoints/tiny_lm_2block_4head_bpe2048.pt`, `checkpoints/tiny_lm_2block_4head_dropout0.1.pt` or `checkpoints/tiny_lm_attention_only.pt` |
| `--corpus` | `data/processed/corpus.txt` | Used to rebuild the same validation split |
| `--examples` | `5` | Number of example next-token predictions to show |
| `--prompt` | `"today we are going to talk about"` | Starting text for generation |
| `--num-tokens` | `30` | Tokens to generate |
| `--temperature` | `0.8` | Lower is safer and more repetitive; higher is more varied |
| `--seed` | none | Fix the random seed so generation repeats; leave unset for different text each run |

The tokenizer is read from the checkpoint (word-level if the checkpoint has no `tokenizer` key), so old and new checkpoints both work.

Per-token loss and accuracy can't be compared across tokenizers, because a BPE token is often only part of a word. `evaluate.py` also prints **loss per word** and **word perplexity**: the total validation loss divided by the number of words, as the word-level tokenizer splits them. For word-level checkpoints these equal the per-token numbers. Use them to compare BPE and word-level models.

## Results

Validation results, 10 epochs each, word-level tokenizer:

| Model | Loss (= per word) | Word perplexity | Top-1 | Top-5 |
|---|---|---|---|---|
| Attention only | 3.406 | 30.1 | 30.9% | 55.2% |
| 2 blocks × 4 heads | **2.752** | **15.7** | **43.3%** | **65.4%** |
| 2 blocks × 4 heads, dropout 0.1 | 3.077 | 21.7 | 35.6% | 60.1% |

Dropout shrank the train–val loss gap from 0.58 to 0.08 but made validation worse. The model wasn't overfitting yet, so dropout only slowed learning at this size. A BPE model has to beat a word perplexity of 15.7 to improve on word-level.

### Attention-only model

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
- train the BPE model and compare word perplexity with the word-level model
- spend the parameters BPE frees up on a bigger model (e.g. embedding 128, more blocks), then turn dropout back on

The [`../Transformer`](../Transformer) notebook implements most of these.
