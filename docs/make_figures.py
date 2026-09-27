"""
Generates the figures used in the README from the datasets and saved runs.

Usage (from the repo root, with src on PYTHONPATH):
    python docs/make_figures.py \
        --cifar-best run_<timestamp> --mnist-best run_<timestamp> \
        --curve "Baseline=run_<timestamp>" --curve "BatchNorm=run_<timestamp>" ...

Images are written to docs/images/.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from myproject.config import RUNS_DIR
from myproject.data_scripts.load_data import load_MNIST, load_CIFAR10
from myproject.model.activations import ConvolutionalLayer, ReLU
from myproject.utils.io import build_NeuralNetwork

OUT_DIR = Path(__file__).resolve().parent / "images"

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
# Categorical slots in fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": TEXT_PRIMARY,
    "axes.labelcolor": TEXT_SECONDARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "font.size": 10,
})


def save(fig, name):
    OUT_DIR.mkdir(parents = True, exist_ok = True)
    fig.savefig(OUT_DIR / name, dpi = 150, bbox_inches = "tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR / name}")


def first_per_class(targets, per_class):
    targets = torch.as_tensor(targets)
    return [torch.nonzero(targets == c).flatten()[:per_class].tolist() for c in range(10)]


def dataset_samples(images, targets, names, filename, cmap = None, per_class = 2):
    """Grid with per_class examples of each class, one class per column."""
    idx = first_per_class(targets, per_class)
    fig, axes = plt.subplots(per_class, 10, figsize = (10, 1.0 * per_class + 0.3))
    for c in range(10):
        for r in range(per_class):
            ax = axes[r, c]
            ax.imshow(images[idx[c][r]], cmap = cmap, interpolation = "nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[0, c].set_title(names[c], fontsize = 9, color = TEXT_SECONDARY)
    fig.subplots_adjust(wspace = 0.04, hspace = 0.04)
    save(fig, filename)


def normalize(t):
    t = t - t.min()
    return t / t.max().clamp_min(1e-8)


def first_conv_filters(model, filename, title):
    """First-layer kernels, RGB when the input has 3 channels, each min-max normalized."""
    conv = next(layer for layer in model.layers if isinstance(layer, ConvolutionalLayer))
    w = conv.weights.data.cpu()
    n = w.shape[0]
    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize = (cols * 0.9, rows * 0.9 + 0.4))
    for i, ax in enumerate(axes.flat):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if i >= n:
            continue
        k = normalize(w[i])
        if k.shape[0] == 3:
            ax.imshow(k.permute(1, 2, 0).numpy(), interpolation = "nearest")
        else:
            ax.imshow(k[0].numpy(), cmap = "gray", interpolation = "nearest")
    fig.suptitle(title, color = TEXT_PRIMARY)
    fig.subplots_adjust(wspace = 0.1, hspace = 0.1)
    save(fig, filename)


def feature_maps(model, image, label, filename, num_maps = 8):
    """
    Passes one image through the trained net (eval mode) and shows the
    strongest activation maps after the first and second conv blocks.
    """
    model.eval()
    x = image.unsqueeze(0)
    conv_seen = 0
    captured = {}
    out = x
    # Capture after the ReLU that follows conv 1 and conv 3 (after the first pool)
    for layer in model.layers:
        out = layer.forward(out)
        if isinstance(layer, ConvolutionalLayer):
            conv_seen += 1
        if isinstance(layer, ReLU) and conv_seen in (1, 3) and conv_seen not in captured:
            captured[conv_seen] = out[0].detach().cpu()
        if len(captured) == 2:
            break
    captured = [captured[1], captured[3]]

    fig, axes = plt.subplots(2, num_maps + 1, figsize = (num_maps + 1, 2.4))
    for r, maps in enumerate(captured):
        ax = axes[r, 0]
        if r == 0:
            ax.imshow(image.permute(1, 2, 0).cpu().numpy(), interpolation = "nearest")
            ax.set_title(f"input: {label}", fontsize = 8, color = TEXT_SECONDARY)
        ax.axis("off")
        strongest = maps.flatten(1).mean(1).argsort(descending = True)[:num_maps]
        for j, ch in enumerate(strongest.tolist()):
            a = axes[r, j + 1]
            a.imshow(maps[ch].numpy(), cmap = "magma", interpolation = "nearest")
            a.axis("off")
        axes[r, 1].set_title(["conv 1 (32×32)", "conv 3 (16×16)"][r], fontsize = 8,
                             color = TEXT_SECONDARY, loc = "left")
    fig.subplots_adjust(wspace = 0.05, hspace = 0.25)
    save(fig, filename)


def read_accuracy(run):
    path = RUNS_DIR / run / "metrics" / "train_metrics.jsonl"
    with open(path) as f:
        return [json.loads(line)["accuracy"] * 100 for line in f]


def accuracy_curves(curves, filename):
    """Test accuracy per epoch for several runs, direct-labeled at the line end."""
    fig, ax = plt.subplots(figsize = (8, 4.2))
    ends = []
    for i, (label, run) in enumerate(curves):
        acc = read_accuracy(run)
        epochs = range(1, len(acc) + 1)
        color = SERIES[i]
        ax.plot(epochs, acc, color = color, linewidth = 2, label = f"{label} (best {max(acc):.1f}%)")
        ends.append([len(acc), acc[-1], label, color])

    # Spread end labels vertically so they don't overlap
    ends.sort(key = lambda e: e[1])
    min_gap = 1.6
    for k in range(1, len(ends)):
        ends[k][1] = max(ends[k][1], ends[k - 1][1] + min_gap)
    for x, y, label, color in ends:
        ax.plot([x + 0.4], [y], marker = "o", markersize = 4, color = color)
        ax.annotate(label, (x + 1.0, y), va = "center", fontsize = 9, color = TEXT_PRIMARY)

    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("CIFAR-10 test accuracy per epoch", loc = "left", color = TEXT_PRIMARY)
    ax.grid(axis = "y", color = GRID, linewidth = 0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    max_epoch = max(e[0] for e in ends)
    ax.set_xlim(0, max_epoch + 12)
    ax.legend(loc = "lower right", frameon = False, fontsize = 8, labelcolor = TEXT_SECONDARY)
    save(fig, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cifar-best", required = True)
    parser.add_argument("--mnist-best", required = True)
    parser.add_argument("--curve", action = "append", default = [], help = "Label=run_dir")
    args = parser.parse_args()

    mnist_train, _ = load_MNIST()
    dataset_samples(mnist_train.data, mnist_train.targets, [str(d) for d in range(10)],
                    "mnist_samples.png", cmap = "gray")

    cifar_train, cifar_test = load_CIFAR10()
    dataset_samples(cifar_train.data, cifar_train.targets, CIFAR_CLASSES, "cifar10_samples.png")

    cifar_model = build_NeuralNetwork(args.cifar_best)
    first_conv_filters(cifar_model, "cifar10_filters.png", "CIFAR-10: learned first-layer filters (3×3 RGB)")

    mnist_model = build_NeuralNetwork(args.mnist_best)
    first_conv_filters(mnist_model, "mnist_filters.png", "MNIST: learned first-layer filters (3×3)")

    # Same test image each time: the first ship
    idx = cifar_test.targets.index(8)
    image = torch.tensor(cifar_test.data[idx], dtype = torch.float32).permute(2, 0, 1) / 255.0
    feature_maps(cifar_model, image, CIFAR_CLASSES[8], "cifar10_feature_maps.png")

    if args.curve:
        accuracy_curves([c.split("=", 1) for c in args.curve], "cifar10_accuracy.png")


if __name__ == "__main__":
    main()
