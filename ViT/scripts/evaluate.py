"""Evaluate a trained ViT on the CIFAR-10 test set.

Run:  uv run python scripts/evaluate.py
      uv run python scripts/evaluate.py --checkpoint outputs/vit_cifar10.pth --out-dir outputs

Prints overall accuracy, per-class accuracy and the most-confused class pairs, and saves
  <out-dir>/confusion_matrix.png
  <out-dir>/misclassified.png   (most confident mistakes)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from vit.config import TrainConfig, ViTConfig
from vit.data import CIFAR_MEAN, CIFAR_STD
from vit.model import VisionTransformer


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run the model over a loader. Returns (labels, preds, confidences, summed_loss), all on CPU.

    labels, preds, confidences: shape (num_samples,).  confidence = softmax probability of the prediction.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")   # plain CE, no label smoothing, for evaluation
    all_labels, all_preds, all_conf = [], [], []
    total_loss = 0.0
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        logits = model(data)                                   # (B, num_classes)
        total_loss += criterion(logits, labels).item()
        probs = logits.softmax(dim=1)                          # (B, num_classes)
        conf, preds = probs.max(dim=1)                         # (B,), (B,)
        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())
        all_conf.append(conf.cpu())
    return torch.cat(all_labels), torch.cat(all_preds), torch.cat(all_conf), total_loss


def confusion_matrix(labels, preds, num_classes):
    """cm[true, predicted] = number of samples."""
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    idx = labels * num_classes + preds
    cm += torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm


def print_report(cm, class_names, top_k=5):
    per_class = cm.diag().float() / cm.sum(dim=1).float() * 100
    print("\nPer-class accuracy:")
    for i in torch.argsort(per_class, descending=True).tolist():
        print(f"  {class_names[i]:<12s} {per_class[i]:6.2f}%")

    off = cm.clone()
    off.fill_diagonal_(0)
    flat = torch.argsort(off.flatten(), descending=True)[:top_k]
    print(f"\nMost confused pairs (true -> predicted):")
    for f in flat.tolist():
        t, p = divmod(f, cm.shape[0])
        if off[t, p] > 0:
            print(f"  {class_names[t]:<12s} -> {class_names[p]:<12s} {off[t, p].item():4d} images")


def plot_confusion(cm, class_names, path):
    # Rows normalised to percentages so classes with different counts are comparable.
    norm = cm.float() / cm.sum(dim=1, keepdim=True).float() * 100
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (% of each true class)")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            v = norm[i, j].item()
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="white" if v > 50 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_misclassified(dataset, labels, preds, conf, class_names, path, n=16):
    """Grid of the n wrong predictions the model was most confident about."""
    wrong = (labels != preds).nonzero(as_tuple=True)[0]
    if len(wrong) == 0:
        return
    order = wrong[torch.argsort(conf[wrong], descending=True)][:n]
    mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(3, 1, 1)
    cols = 4
    rows = (len(order) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.7 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, order.tolist()):
        img, _ = dataset[i]                                    # (3, 32, 32), normalised
        img = (img * std + mean).clamp(0, 1).permute(1, 2, 0)  # (32, 32, 3) for imshow
        ax.imshow(img)
        ax.set_title(f"true: {class_names[labels[i]]}\npred: {class_names[preds[i]]} ({conf[i]:.0%})", fontsize=8)
    fig.suptitle("Most confident mistakes", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    tcfg = TrainConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(Path(tcfg.out_dir) / "vit_cifar10.pth"))
    p.add_argument("--data-dir", default=tcfg.data_dir)
    p.add_argument("--out-dir", default=tcfg.out_dir)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=tcfg.num_workers)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    test_ds = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=test_tf)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    class_names = test_ds.classes

    cfg = ViTConfig()
    model = VisionTransformer(cfg).to(device)
    # The checkpoint only fits a model built with the same ViTConfig it was trained with.
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded {args.checkpoint}")

    labels, preds, conf, total_loss = collect_predictions(model, loader, device)
    n = len(labels)
    acc = (labels == preds).float().mean().item() * 100
    print(f"\nTest accuracy: {acc:.2f}%  ({int((labels == preds).sum())}/{n})")
    print(f"Test cross-entropy (no label smoothing): {total_loss / n:.4f}")

    cm = confusion_matrix(labels, preds, cfg.num_classes)
    print_report(cm, class_names)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion(cm, class_names, out_dir / "confusion_matrix.png")
    plot_misclassified(test_ds, labels, preds, conf, class_names, out_dir / "misclassified.png")
    print(f"\nSaved plots to {out_dir}/")


if __name__ == "__main__":
    main()