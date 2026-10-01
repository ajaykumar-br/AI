"""
Plot train / validation loss and accuracy recorded by train_corpus.py.

Usage:

    uv run python scripts/plot_history.py
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

TRAIN_COLOR = "#2a78d6"   # blue
VAL_COLOR = "#eb6834"     # orange
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID = "#e6e5e1"


def style_axis(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=12, pad=10)
    ax.set_xlabel("epoch", color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_MUTED, length=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def plot_series(ax, epochs, values, color, label, fmt):
    ax.plot(
        epochs, values,
        color=color, linewidth=2, marker="o", markersize=6,
        markeredgecolor=SURFACE, markeredgewidth=1.5, label=label
    )
    # direct label on the last point
    ax.annotate(
        f"{label} {fmt(values[-1])}",
        xy=(epochs[-1], values[-1]),
        xytext=(8, 0), textcoords="offset points",
        va="center", color=TEXT_SECONDARY, fontsize=9
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, default=Path("checkpoints/history.json"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/history.png"))
    parser.add_argument("--no-show", action="store_true", help="only save the PNG")
    args = parser.parse_args()

    history = json.loads(args.history.read_text(encoding="utf-8"))
    epochs = [h["epoch"] for h in history]

    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=SURFACE)

    # Loss
    plot_series(loss_ax, epochs, [h["train_loss"] for h in history],
                TRAIN_COLOR, "train", lambda v: f"{v:.3f}")
    plot_series(loss_ax, epochs, [h["val_loss"] for h in history],
                VAL_COLOR, "val", lambda v: f"{v:.3f}")
    style_axis(loss_ax, "Cross-entropy loss", "loss")

    # Accuracy
    plot_series(acc_ax, epochs, [h["train_accuracy"] * 100 for h in history],
                TRAIN_COLOR, "train", lambda v: f"{v:.1f}%")
    plot_series(acc_ax, epochs, [h["val_accuracy"] * 100 for h in history],
                VAL_COLOR, "val", lambda v: f"{v:.1f}%")
    style_axis(acc_ax, "Next-token accuracy (top-1)", "accuracy (%)")

    for ax in (loss_ax, acc_ax):
        ax.set_xticks(epochs)
        ax.set_xlim(epochs[0] - 0.3, epochs[-1] + 1.6)  # room for end labels
        ax.legend(frameon=False, labelcolor=TEXT_SECONDARY, loc="best")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, facecolor=SURFACE)
    print("saved plot to:", args.output)

    if not args.no_show:
        plt.show()
