"""
Load the trained TinyLanguageModel and evaluate it on the held-out
validation split (the last val_fraction of the corpus).

Usage:

    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --prompt "the future of ai is" --num-tokens 40
"""

import argparse
from collections import Counter
from pathlib import Path

import torch

from mini_transformer.tokenizer import tokenize
from mini_transformer.vocabulary import encode, decode
from mini_transformer.dataset import create_training_data
from mini_transformer.dataloader import create_dataloader
from mini_transformer.model import TinyLanguageModel
from mini_transformer.evaluation import evaluate


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = TinyLanguageModel(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


@torch.no_grad()
def show_predictions(model, input_ids, target_ids, itos, device, num_examples, top_k=5):
    """
    For a few validation windows, show the last context words, the true
    next word, and the model's top-k guesses.
    """

    indices = torch.linspace(0, len(input_ids) - 1, num_examples).long()

    for i in indices:
        x = input_ids[i].unsqueeze(0).to(device)
        logits = model(x)[0, -1]
        probs = torch.softmax(logits, dim=-1)
        top = probs.topk(top_k)

        context = " ".join(decode(input_ids[i, -10:].tolist(), itos))
        actual = itos[target_ids[i, -1].item()]
        guesses = ", ".join(
            f"{itos[idx.item()]} ({p.item():.0%})"
            for p, idx in zip(top.values, top.indices)
        )
        mark = "[correct]" if top.indices[0].item() == target_ids[i, -1].item() else "[wrong]"

        print(f"\n  ...{context}")
        print(f"     actual: {actual!r}   {mark}")
        print(f"     model : {guesses}")


@torch.no_grad()
def generate(model, prompt, stoi, itos, sequence_length, num_tokens, temperature, device):
    token_ids = encode(tokenize(prompt), stoi)

    for _ in range(num_tokens):
        context = torch.tensor([token_ids[-sequence_length:]], dtype=torch.long, device=device)
        logits = model(context)[0, -1] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        token_ids.append(next_id)

    return " ".join(decode(token_ids, itos))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tiny_lm.pt"))
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/corpus.txt"))
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--prompt", type=str, default="today we are going to talk about")
    parser.add_argument("--num-tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # Load model + vocabulary
    # --------------------------------------------------------

    model, checkpoint = load_model(args.checkpoint, device)
    stoi, itos = checkpoint["stoi"], checkpoint["itos"]
    config = checkpoint["config"]
    sequence_length = config["sequence_length"]
    val_fraction = checkpoint.get("val_fraction")

    print("checkpoint:", args.checkpoint, "| epoch:", checkpoint["epoch"], "| device:", device)
    print("config:", config)
    print("parameters:", f"{sum(p.numel() for p in model.parameters()):,}")

    # --------------------------------------------------------
    # Rebuild the same validation split used in training
    # --------------------------------------------------------

    tokens = tokenize(args.corpus.read_text(encoding="utf-8"))
    token_ids = encode(tokens, stoi)

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

    # Baselines: random guessing, and always predicting the most common training word
    most_common_id, _ = Counter(train_token_ids or token_ids).most_common(1)[0]
    baseline_accuracy = sum(t == most_common_id for t in val_token_ids[1:]) / (len(val_token_ids) - 1)

    print("\n=== Validation metrics ===")
    print(f"  loss            : {metrics['loss']:.4f}")
    print(f"  perplexity      : {metrics['perplexity']:.2f}")
    print(f"  top-1 accuracy  : {metrics['accuracy']:.2%}")
    print(f"  top-5 accuracy  : {metrics['top_5_accuracy']:.2%}")

    print("\n=== Baselines ===")
    print(f"  random guess perplexity          : {len(stoi):,}")
    print(f"  always '{itos[most_common_id]}' accuracy : {baseline_accuracy:.2%}")

    # --------------------------------------------------------
    # Example predictions
    # --------------------------------------------------------

    print("\n=== Example next-word predictions (validation text) ===")
    show_predictions(
        model,
        torch.as_tensor(val_input_ids),
        torch.as_tensor(val_target_ids),
        itos, device, args.examples
    )

    # --------------------------------------------------------
    # Text generation
    # --------------------------------------------------------

    print(f"\n=== Generated text (temperature {args.temperature}) ===")
    print(" ", generate(
        model, args.prompt, stoi, itos, sequence_length,
        args.num_tokens, args.temperature, device
    ))
