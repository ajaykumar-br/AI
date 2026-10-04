import json
from pathlib import Path

import torch

from mini_transformer.tokenizer import tokenize
from mini_transformer.vocabulary import build_vocabulary, encode, decode
from mini_transformer.dataset import create_training_data
from mini_transformer.dataloader import create_dataloader
from mini_transformer.model import TinyLanguageModel
from mini_transformer.loss import calculate_loss
from mini_transformer.evaluation import count_correct, evaluate

if __name__ == "__main__":
    corpus_path = Path("data/processed/corpus.txt")
    sequence_length = 32
    embedding_dim = 64
    head_dim = 64
    batch_size = 64
    epochs = 10
    val_fraction = 0.1
    save_path = Path("checkpoints/tiny_lm.pt")
    history_path = Path("checkpoints/history.json")

    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # --------------------------------------------------------
    # corpus -> tokenize -> tokens
    # --------------------------------------------------------

    text = corpus_path.read_text(encoding="utf-8")
    tokens = tokenize(text)
    print("tokens:", len(tokens))
    print("first tokens:", tokens[:10])

    # --------------------------------------------------------
    # build vocabulary
    # --------------------------------------------------------

    stoi, itos = build_vocabulary(tokens)
    print("vocab size:", len(stoi))

    # --------------------------------------------------------
    # encode -> token IDs
    # --------------------------------------------------------

    token_ids = encode(tokens, stoi)
    print("first token IDs:", token_ids[:10])
    print("decoded:", decode(token_ids[:10], itos))

    # --------------------------------------------------------
    # train / validation split
    # (the last 10% of the corpus is never trained on)
    # --------------------------------------------------------

    split_index = int(len(token_ids) * (1 - val_fraction))
    train_token_ids = token_ids[:split_index]
    val_token_ids = token_ids[split_index:]
    print("train tokens:", len(train_token_ids), "| val tokens:", len(val_token_ids))

    # --------------------------------------------------------
    # create_training_data -> input_ids + target_ids
    # --------------------------------------------------------

    train_input_ids, train_target_ids = create_training_data(train_token_ids, sequence_length)
    val_input_ids, val_target_ids = create_training_data(val_token_ids, sequence_length)

    # --------------------------------------------------------
    # dataloader -> mini-batches
    # --------------------------------------------------------

    loader = create_dataloader(train_input_ids, train_target_ids, batch_size)
    val_loader = create_dataloader(val_input_ids, val_target_ids, batch_size=512, shuffle=False)
    print("train sequences:", len(loader.dataset), "| batches per epoch:", len(loader))
    print("val sequences:", len(val_loader.dataset))

    x, y = next(iter(loader))
    print("batch input shape:", x.shape)
    print("batch target shape:", y.shape)
    print("input  :", " ".join(decode(x[0, :8].tolist(), itos)))
    print("target :", " ".join(decode(y[0, :8].tolist(), itos)))

    # --------------------------------------------------------
    # TinyLanguageModel -> logits
    # --------------------------------------------------------

    model = TinyLanguageModel(
        vocab_size=len(stoi),
        embedding_dim=embedding_dim,
        head_dim=head_dim,
        sequence_length=sequence_length
    ).to(device)
    print("logits shape:", model(x.to(device)).shape)

    # --------------------------------------------------------
    # CrossEntropyLoss -> backpropagation -> optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        for step, (x, y) in enumerate(loader, start=1):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = calculate_loss(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += count_correct(logits.detach(), y)
            total_tokens += y.numel()

            if step % 1000 == 0:
                print(f"epoch {epoch} / step {step} / {len(loader)} / loss: {loss.item():.4f}")

        train_loss = total_loss / len(loader)
        train_accuracy = total_correct / total_tokens
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"epoch {epoch} done"
            f" / train loss: {train_loss:.4f} / train acc: {train_accuracy:.2%}"
            f" / val loss: {val_metrics['loss']:.4f} / val acc: {val_metrics['accuracy']:.2%}"
        )

        # --------------------------------------------------------
        # Save loss / accuracy history (overwritten every epoch)
        # --------------------------------------------------------

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        })

        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        # --------------------------------------------------------
        # Save model + vocabulary (overwritten every epoch)
        # --------------------------------------------------------

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "stoi": stoi,
                "itos": itos,
                "config": {
                    "vocab_size": len(stoi),
                    "embedding_dim": embedding_dim,
                    "head_dim": head_dim,
                    "sequence_length": sequence_length,
                },
                "val_fraction": val_fraction,
                "epoch": epoch,
            },
            save_path
        )
        print("saved model to:", save_path)
