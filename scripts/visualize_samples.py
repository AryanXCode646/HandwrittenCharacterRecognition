"""Visualize representative MNIST samples and export a PNG grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from torchvision import datasets  # noqa: E402

from handwritten_recognition.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Save representative MNIST sample grid.")
    parser.add_argument("--config", default="configs/config.json", help="Path to config JSON.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = datasets.MNIST(root=cfg["dataset"]["data_dir"], train=True, download=False)
    out_dir = Path(cfg["evaluation"]["figures_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 5, figsize=(10, 4.2))
    seen: dict[int, int] = {}
    for img, label in ds:
        if label not in seen:
            seen[label] = 1
            ax = axes.flat[label]
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Digit: {label}")
            ax.axis("off")
        if len(seen) == 10:
            break
    plt.tight_layout()
    out_path = out_dir / "representative_samples.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    # Also save one raw sample image for quick CLI prediction demo
    sample_img, sample_label = ds[0]
    sample_path = Path("data/sample_digit.png")
    sample_img.save(sample_path)
    print(
        f"Saved sample grid to '{out_path}' and sample digit ({sample_label}) to '{sample_path}'."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
