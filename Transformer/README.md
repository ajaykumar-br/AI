# Transformer (from scratch, notebook)

A decoder-style transformer language model written from scratch in a single Jupyter notebook ([`transformer.ipynb`](transformer.ipynb)). It's trained on a synthetic corpus of simple English sentences to predict the next token.

## What's in the notebook

1. **Synthetic corpus:** 10,000 sentences built from templates (subjects × verbs × objects × adjectives × locations), e.g. *"the child is old and eats the car."*
2. **Tokenizer & vocabulary:** lowercase, split off punctuation, plus special tokens `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`.
3. **Training data:** one sentence per sequence, target shifted by one token, padded to length 32.
4. **Model components, each written by hand:**
   - token embedding (`nn.Embedding`)
   - sinusoidal positional encoding (sin on even dimensions, cos on odd)
   - `SelfAttention`: Q/K/V projections, scaled dot-product
   - `MultiHeadAttention`: several `SelfAttention` heads, concatenated and passed through `W_O`
   - `FeedForward`: Linear → GELU → Linear
   - `TransformerBlock`: attention + residual + LayerNorm, then feed-forward + residual + LayerNorm (post-LN)
   - `MiniTransformer`: embeddings → N blocks → final LayerNorm → Linear to vocabulary logits
5. **Training:** `CrossEntropyLoss(ignore_index=<PAD>)`, Adam, accuracy measured on non-padding positions, and loss/accuracy plots.

## Configuration

| Setting | Value |
|---|---|
| Embedding dim (`d_model`) | 128 |
| Attention heads | 4 (head dim 32) |
| Transformer layers | 4 |
| Feed-forward hidden dim | 512 |
| Max sequence length | 32 |
| Optimiser | Adam, lr 1e-3 |
| Training steps | 20, full batch of all 10,000 sentences per step |

## Running

```bash
cd Transformer
jupyter notebook transformer.ipynb
```

Any environment with `torch` and `matplotlib` works. For example, from this folder:

```bash
uv run --with torch --with matplotlib --with jupyter jupyter notebook transformer.ipynb
```

## Results so far

After 20 full-batch steps the training loss is **1.53** and token accuracy is **51%**, both still improving. The notebook notes that the plan is to raise this to about 1,000 steps once everything checks out.

## Known issues / TODO

- **The causal mask isn't applied yet.** `SelfAttention.forward` builds `causal_mask` but never uses it (the attention scores are recomputed without `masked_fill`), and the padding mask is passed in but unused too. Every position can therefore see future tokens, so the accuracy above overstates real next-token prediction. Applying `scores.masked_fill(causal_mask == 0, float('-inf'))` before the softmax fixes it.
- **Shape handling assumes unbatched input.** Inside attention, `sequence_length = x.shape[0]` reads the batch size, not the sequence length, for batched input of shape `(B, T, d)`; it should be `x.shape[1]`. The same applies to the positional-encoding slice in `MiniTransformer.forward`, which uses `input_ids.shape[0]`.
- No validation split or text generation yet.

The simpler, script-based [`../mini-transformer`](../mini-transformer) project already has the masking right and trains on real text. Merging this notebook's multi-head blocks into it is the natural next step.
