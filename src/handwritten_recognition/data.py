"""MNIST dataset loading and deterministic train/val/test splitting."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets

from handwritten_recognition.preprocessing import build_mnist_transform


def download_mnist(data_dir: str | Path = "data") -> tuple[Path, int, int]:
    """Download MNIST train and test splits via torchvision.

    Returns:
        Tuple of (data_dir_path, train_count, test_count).
    """
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    transform = build_mnist_transform()
    try:
        train_ds = datasets.MNIST(root=str(root), train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root=str(root), train=False, download=True, transform=transform)
    except Exception as exc:
        raise RuntimeError(f"Failed to download MNIST dataset to '{root}': {exc}") from exc
    return root, len(train_ds), len(test_ds)


def create_dataloaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    val_fraction: float = 0.1,
    random_state: int = 42,
    download: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create deterministic train, validation, and test DataLoaders for MNIST."""
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be between 0 and 0.5.")

    root = Path(data_dir)
    transform = build_mnist_transform()
    try:
        full_train = datasets.MNIST(
            root=str(root), train=True, download=download, transform=transform
        )
        test_ds = datasets.MNIST(
            root=str(root), train=False, download=download, transform=transform
        )
    except RuntimeError as exc:
        raise FileNotFoundError(
            f"MNIST dataset not found in '{root}'. Run `python scripts/download_data.py` first."
        ) from exc

    n_total = len(full_train)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(random_state)
    train_subset: Subset
    val_subset: Subset
    train_subset, val_subset = random_split(full_train, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader
