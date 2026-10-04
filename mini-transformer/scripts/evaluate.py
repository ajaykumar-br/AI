"""
Load the trained TinyLanguageModel and evaluate it on the held-out
validation split (the last val_fraction of the corpus).

Usage:

    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --prompt "the future of ai is" --num-tokens 40
    uv run python scripts/evaluate.py --seed 0   # repeatable generation
    uv run python scripts/evaluate.py --checkpoint checkpoints/tiny_lm_attention_only.pt
    uv run python scripts/evaluate.py --checkpoint checkpoints/tiny_lm_2block_4head_bpe2048.pt
"""

import argparse
import math
from collections import Counter
from pathlib import Path

import torch

from mini_transformer.tokenizer import tokenize
from mini_transformer.vocabulary import encode, decode
from mini_transformer.bpe import BPETokenizer
from mini_transformer.dataset import create_training_data
from mini_transformer.dataloader import create_dataloader
from mini_transformer.model import TinyLanguageModel
from mini_transformer.attention_only_model import AttentionOnlyLanguageModel
from mini_transformer.evaluation import evaluate


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint["config"]

    # The first model (tiny_lm_attention_only.pt) saved head_dim
    # instead of num_layers / num_heads.
    if "num_layers" in config:
        model_class = TinyLanguageModel
    else:
        model_class = AttentionOnlyLanguageModel

    model = model_class(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


class WordTokenizer:
    """
    The original word-level tokenizer, with the same methods as BPETokenizer.
    """

    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(stoi)

    def encode(self, text):
        return encode(tokenize(text), self.stoi)

    def decode(self, token_ids):
        return " ".join(decode(token_ids, self.itos))

    def token_strings(self, token_ids):
        return decode(token_ids, self.itos)


class LowercaseBPETokenizer(BPETokenizer):
    """
    BPE was trained on lowercased text, so lowercase here too
    (the word-level tokenize() already does this).
    """

    def encode(self, text):
        return super().encode(text.lower())


def load_tokenizer(checkpoint):
    # Checkpoints from before BPE have no "tokenizer" key: they are word-level.
    tokenizer_type = checkpoint.get("tokenizer", "word")

    if tokenizer_type == "word":
        return tokenizer_type, WordTokenizer(checkpoint["stoi"], checkpoint["itos"])

    if tokenizer_type == "bpe":
        return tokenizer_type, LowercaseBPETokenizer(checkpoint["merges"])

    raise ValueError(f"unknown tokenizer in checkpoint: {tokenizer_type!r}")


@torch.no_grad()
def show_predictions(model, input_ids, target_ids, tokenizer, device, num_examples, top_k=5):
    """
    For a few validation windows, show the last context tokens, the true
    next token, and the model's top-k guesses.
    """

    indices = torch.linspace(0, len(input_ids) - 1, num_examples).long()

    for i in indices:
        x = input_ids[i].unsqueeze(0).to(device)
        logits = model(x)[0, -1]
        probs = torch.softmax(logits, dim=-1)
        top = probs.topk(top_k)

        context = tokenizer.decode(input_ids[i, -12:].tolist())
        actual = tokenizer.token_strings([target_ids[i, -1].item()])[0]
        guesses = ", ".join(
            f"{token!r} ({p.item():.0%})"
            for p, token in zip(top.values, tokenizer.token_strings(top.indices.tolist()))
        )
        mark = "[correct]" if top.indices[0].item() == target_ids[i, -1].item() else "[wrong]"

        print(f"\n  ...{context}")
        print(f"     actual: {actual!r}   {mark}")
        print(f"     model : {guesses}")


@torch.no_grad()
def generate(model, prompt, tokenizer, sequence_length, num_tokens, temperature, device):
    token_ids = tokenizer.encode(prompt)

    for _ in range(num_tokens):
        context = torch.tensor([token_ids[-sequence_length:]], dtype=torch.long, device=device)
        logits = model(context)[0, -1] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        token_ids.append(next_id)

    return tokenizer.decode(token_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tiny_lm_2block_4head.pt"))
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/corpus.txt"))
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--prompt", type=str, default="today we are going to talk about")
    parser.add_argument("--num-tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=None,
                        help="Fix the RNG for repeatable generation")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # Load model + tokenizer
    # --------------------------------------------------------

    model, checkpoint = load_model(args.checkpoint, device)
    tokenizer_type, tokenizer = load_tokenizer(checkpoint)
    config = checkpoint["config"]
    sequence_length = config["sequence_length"]
    val_fraction = checkpoint.get("val_fraction")

    print("checkpoint:", args.checkpoint, "| epoch:", checkpoint["epoch"], "| device:", device)
    print("tokenizer:", tokenizer_type, "| vocab size:", f"{tokenizer.vocab_size:,}")
    print("config:", config)
    print("parameters:", f"{sum(p.numel() for p in model.parameters()):,}")

    # --------------------------------------------------------
    # Rebuild the same validation split used in training
    # --------------------------------------------------------

    token_ids = tokenizer.encode(args.corpus.read_text(encoding="utf-8"))

    if val_fraction is None:
        print("\nWARNING: checkpoint has no val_fraction - evaluating on the full corpus,"
              " which the model was trained on.")
        split_index = 0
    else:
        split_index = int(len(token_ids) * (1 - val_fraction))

    train_token_ids = token_ids[:split_index]
    val_token_ids = token_ids[split_index:]

    val_input_ids, val_target_ids = create_training_data(val_token_ids, sequence_length)
    val_loader = create_dataloader(val_input_ids, val_target_ids, batch_size=512, shuffle=False)
    print("val tokens:", len(val_token_ids), "| val sequences:", len(val_loader.dataset))

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = evaluate(model, val_loader, device)

    # Per-token loss can't be compared across tokenizers: a BPE token is
    # often part of a word. Spread the total loss over the number of words
    # (as the word-level tokenizer splits them) to get a fair number.
    # For word-level checkpoints this is the same as the per-token loss.
    val_words = len(tokenize(tokenizer.decode(val_token_ids)))
    loss_per_word = metrics["loss"] * len(val_token_ids) / val_words

    # Baselines: random guessing, and always predicting the most common training token
    most_common_id, _ = Counter(train_token_ids or token_ids).most_common(1)[0]
    most_common_token = tokenizer.token_strings([most_common_id])[0]
    baseline_accuracy = sum(t == most_common_id for t in val_token_ids[1:]) / (len(val_token_ids) - 1)

    print("\n=== Validation metrics (per token) ===")
    print(f"  loss            : {metrics['loss']:.4f}")
    print(f"  perplexity      : {metrics['perplexity']:.2f}")
    print(f"  top-1 accuracy  : {metrics['accuracy']:.2%}")
    print(f"  top-5 accuracy  : {metrics['top_5_accuracy']:.2%}")

    print("\n=== Validation metrics (per word, comparable across tokenizers) ===")
    print(f"  tokens per word : {len(val_token_ids) / val_words:.3f}")
    print(f"  loss per word   : {loss_per_word:.4f}")
    print(f"  word perplexity : {math.exp(loss_per_word):.2f}")

    print("\n=== Baselines ===")
    print(f"  random guess perplexity          : {tokenizer.vocab_size:,}")
    print(f"  always {most_common_token!r} accuracy : {baseline_accuracy:.2%}")

    # --------------------------------------------------------
    # Example predictions
    # --------------------------------------------------------

    print("\n=== Example next-token predictions (validation text) ===")
    show_predictions(
        model,
        torch.as_tensor(val_input_ids),
        torch.as_tensor(val_target_ids),
        tokenizer, device, args.examples
    )

    # --------------------------------------------------------
    # Text generation
    # --------------------------------------------------------

    print(f"\n=== Generated text (temperature {args.temperature}) ===")
    print(" ", generate(
        model, args.prompt, tokenizer, sequence_length,
        args.num_tokens, args.temperature, device
    ))
