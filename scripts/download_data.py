"""Download the MNIST dataset via torchvision."""

from __future__ import annotations

import argparse
import sys

from handwritten_recognition.config import load_config
from handwritten_recognition.data import download_mnist


def main() -> int:
    parser = argparse.ArgumentParser(description="Download MNIST dataset.")
    parser.add_argument("--config", default="configs/config.json", help="Path to config JSON.")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        root, n_train, n_test = download_mnist(cfg["dataset"]["data_dir"])
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"MNIST ready at '{root}' (train={n_train}, test={n_test}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
