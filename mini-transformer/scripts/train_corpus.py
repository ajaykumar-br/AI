import json
from pathlib import Path

import torch

from mini_transformer.tokenizer import tokenize
from mini_transformer.vocabulary import build_vocabulary, encode, decode
from mini_transformer.bpe import BPETokenizer
from mini_transformer.dataset import create_training_data
from mini_transformer.dataloader import create_dataloader
from mini_transformer.model import TinyLanguageModel
from mini_transformer.loss import calculate_loss
from mini_transformer.evaluation import count_correct, evaluate

if __name__ == "__main__":
    corpus_path = Path("data/processed/corpus.txt")
    tokenizer_type = "bpe"   # "word" or "bpe"
    bpe_vocab_size = 2048    # 256 bytes + 1792 learned merges
    sequence_length = 40
    embedding_dim = 64
    num_heads = 4
    num_layers = 2
    dropout = 0.0
    ff_hidden_dim = 4 * embedding_dim
    batch_size = 64
    epochs = 15
    val_fraction = 0.1
    # The name comes from the settings, so a run with different
    # settings never overwrites an earlier model.
    run_name = f"tiny_lm_{num_layers}block_{num_heads}head"
    if tokenizer_type == "bpe":
        run_name += f"_bpe{bpe_vocab_size}"
    if dropout > 0:
        run_name += f"_dropout{dropout}"
    save_path = Path(f"checkpoints/{run_name}.pt")
    history_path = Path(f"checkpoints/{run_name}_history.json")

    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    text = corpus_path.read_text(encoding="utf-8")

    if tokenizer_type == "word":

        # --------------------------------------------------------
        # corpus -> tokenize -> tokens -> vocabulary -> token IDs
        # --------------------------------------------------------

        tokens = tokenize(text)
        print("tokens:", len(tokens))
        print("first tokens:", tokens[:10])

        stoi, itos = build_vocabulary(tokens)
        vocab_size = len(stoi)
        token_ids = encode(tokens, stoi)

        # IDs -> list of token strings, for printing
        def token_strings(ids):
            return decode(ids, itos)

        tokenizer_state = {"stoi": stoi, "itos": itos}

    elif tokenizer_type == "bpe":

        # --------------------------------------------------------
        # corpus -> learn BPE merges -> token IDs
        # (lowercased, like the word-level tokenizer, so the only
        # thing that changes is how words are split)
        # --------------------------------------------------------

        text = text.lower()
        bpe = BPETokenizer()
        print(f"learning {bpe_vocab_size - 256} BPE merges...")
        bpe.train(text, bpe_vocab_size, verbose=True)

        vocab_size = bpe.vocab_size
        token_ids = bpe.encode(text)
        token_strings = bpe.token_strings

        tokenizer_state = {"merges": bpe.merges}

    else:
        raise ValueError(f"unknown tokenizer_type: {tokenizer_type!r}")

    print("tokenizer:", tokenizer_type, "| vocab size:", vocab_size)
    print("token IDs:", len(token_ids))
    print("first token IDs:", token_ids[:10])
    print("decoded:", token_strings(token_ids[:10]))

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
    print("input  :", token_strings(x[0, :8].tolist()))
    print("target :", token_strings(y[0, :8].tolist()))

    # --------------------------------------------------------
    # TinyLanguageModel -> logits
    # --------------------------------------------------------

    model = TinyLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        sequence_length=sequence_length,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_hidden_dim=ff_hidden_dim,
        dropout=dropout
    ).to(device)
    print("logits shape:", model(x.to(device)).shape)

    # --------------------------------------------------------
    # CrossEntropyLoss -> backpropagation -> optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    best_val_loss = float("inf")

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
        # Save model + tokenizer, only when val loss improves
        # (so an overfitting epoch never replaces a better model)
        # --------------------------------------------------------

        if val_metrics["loss"] >= best_val_loss:
            print(f"val loss did not improve (best: {best_val_loss:.4f}), model not saved")
            continue

        best_val_loss = val_metrics["loss"]

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                # word: "stoi" + "itos" | bpe: "merges"
                "tokenizer": tokenizer_type,
                **tokenizer_state,
                "config": {
                    "vocab_size": vocab_size,
                    "embedding_dim": embedding_dim,
                    "num_heads": num_heads,
                    "num_layers": num_layers,
                    "ff_hidden_dim": ff_hidden_dim,
                    "dropout": dropout,
                    "sequence_length": sequence_length,
                },
                "val_fraction": val_fraction,
                "epoch": epoch,
                "val_loss": best_val_loss,
            },
            save_path
        )
        print(f"val loss improved to {best_val_loss:.4f}, saved model to:", save_path)
