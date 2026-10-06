"""Train a ViT from scratch on CIFAR-10.   Run:  uv run python scripts/train.py"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from vit.config import TrainConfig, ViTConfig
from vit.data import get_cifar10_loaders
from vit.model import VisionTransformer
from vit.training import build_optimizer, evaluate, train_epoch, warmup_cosine


def main():
    tcfg = TrainConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=tcfg.epochs)
    p.add_argument("--batch-size", type=int, default=tcfg.batch_size)
    p.add_argument("--lr", type=float, default=tcfg.lr)
    p.add_argument("--num-workers", type=int, default=tcfg.num_workers)
    args = p.parse_args()
    tcfg.epochs, tcfg.batch_size, tcfg.lr, tcfg.num_workers = args.epochs, args.batch_size, args.lr, args.num_workers

    torch.manual_seed(tcfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(tcfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_cifar10_loaders(tcfg.data_dir, tcfg.batch_size, tcfg.num_workers)

    model = VisionTransformer(ViTConfig()).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=tcfg.label_smoothing)
    optimizer = build_optimizer(model, tcfg.lr, tcfg.weight_decay)
    total_steps = tcfg.epochs * len(train_loader)
    scheduler = warmup_cosine(optimizer, tcfg.warmup_epochs * len(train_loader), total_steps)

    hist = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in range(tcfg.epochs):
        tl, ta = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        vl, va = evaluate(model, test_loader, criterion, device)
        for k, v in zip(hist, (tl, ta, vl, va)):
            hist[k].append(v)
        print(f"Epoch {epoch+1:02d}/{tcfg.epochs}  train {tl:.3f}/{ta:.2f}%  test {vl:.3f}/{va:.2f}%  gap {ta-va:.2f}%")

    torch.save(model.state_dict(), out_dir / "vit_cifar10.pth")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["test_loss"], label="test")
    ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend(); ax[0].grid()
    ax[1].plot(hist["train_acc"], label="train"); ax[1].plot(hist["test_acc"], label="test")
    ax[1].set_title("Accuracy (%)"); ax[1].set_xlabel("Epoch"); ax[1].legend(); ax[1].grid()
    plt.tight_layout(); plt.savefig(out_dir / "vit_training_curves.png", dpi=100)
    print(f"Final test accuracy: {hist['test_acc'][-1]:.2f}%")


if __name__ == "__main__":
    main()
