"""Training workflow for the MNIST Handwritten Digit Recognition CNN."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from handwritten_recognition.config import load_config
from handwritten_recognition.data import create_dataloaders
from handwritten_recognition.model import DigitCNN


def set_deterministic_seed(seed: int = 42) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one training or evaluation epoch and return (avg_loss, accuracy)."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            preds = logits.argmax(dim=1)
            total_correct += int((preds == labels).sum().item())
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("DataLoader yielded zero samples.")
    return total_loss / total_samples, total_correct / total_samples


def train_cnn(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 3,
    random_state: int = 42,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train DigitCNN with validation tracking and early-stopping checkpointing."""
    set_deterministic_seed(random_state)
    target_device = device or torch.device("cpu")
    model = DigitCNN(num_classes=10).to(target_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, target_device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, target_device, optimizer=None)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "train_acc": round(train_acc, 5),
                "val_loss": round(val_loss, 5),
                "val_acc": round(val_acc, 5),
            }
        )
        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "model": model,
        "state_dict": best_state_dict or model.state_dict(),
        "history": history,
        "best_val_loss": float(best_val_loss),
        "metadata": {
            "dataset": "MNIST",
            "epochs_ran": len(history),
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "random_state": random_state,
            "torch_version": str(torch.__version__),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }


def save_checkpoint(
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Save model weights and plain metadata safely for weights_only loading."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": state_dict, "metadata": metadata}, path)
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for training the MNIST CNN."""
    parser = argparse.ArgumentParser(description="Train MNIST DigitCNN.")
    parser.add_argument("--config", default="configs/config.json", help="Path to config JSON.")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        train_loader, val_loader, _ = create_dataloaders(
            data_dir=cfg["dataset"]["data_dir"],
            batch_size=cfg["training"]["batch_size"],
            val_fraction=cfg["dataset"]["val_fraction"],
            random_state=cfg["dataset"]["random_state"],
            download=False,
        )
        result = train_cnn(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=cfg["training"]["epochs"],
            learning_rate=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
            patience=cfg["training"]["patience"],
            random_state=cfg["training"]["random_state"],
        )
        ckpt_path = save_checkpoint(
            result["state_dict"],
            result["metadata"],
            cfg["training"]["model_output_path"],
        )
        hist_path = Path(cfg["training"]["history_output_path"])
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        hist_path.write_text(json.dumps(result["history"], indent=2), encoding="utf-8")
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved model checkpoint to: {ckpt_path}")
    print(f"Saved training history to: {hist_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
