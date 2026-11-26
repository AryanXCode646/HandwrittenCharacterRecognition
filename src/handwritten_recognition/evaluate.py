"""Evaluation, metric computation, and plotting for MNIST DigitCNN."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from handwritten_recognition.config import load_config  # noqa: E402
from handwritten_recognition.data import create_dataloaders  # noqa: E402
from handwritten_recognition.model import DigitCNN  # noqa: E402


def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute overall and per-class classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    return {
        "accuracy": acc,
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_w),
        "recall_weighted": float(rec_w),
        "f1_weighted": float(f1_w),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def _plot_training_curves(history: list[dict[str, float]], out_dir: Path) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, [r["train_loss"] for r in history], marker="o", label="Train Loss")
    axes[0].plot(epochs, [r["val_loss"] for r in history], marker="s", label="Val Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()

    axes[1].plot(epochs, [r["train_acc"] for r in history], marker="o", label="Train Accuracy")
    axes[1].plot(epochs, [r["val_acc"] for r in history], marker="s", label="Val Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close()


def _plot_confusion_matrix(cm: np.ndarray, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted Digit")
    ax.set_ylabel("True Digit")
    ax.set_title("MNIST Test Set Confusion Matrix")
    thresh = cm.max() / 2.0
    for i in range(10):
        for j in range(10):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()


def _plot_sample_predictions(
    images: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for idx, ax in enumerate(axes.flat):
        if idx >= len(images):
            ax.axis("off")
            continue
        ax.imshow(images[idx].squeeze(), cmap="gray")
        color = "green" if y_true[idx] == y_pred[idx] else "red"
        ax.set_title(f"True: {y_true[idx]} | Pred: {y_pred[idx]}", color=color, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "sample_predictions.png", dpi=150)
    plt.close()


def evaluate_model(
    model_path: str | Path,
    data_dir: str | Path,
    batch_size: int = 128,
    val_fraction: float = 0.1,
    random_state: int = 42,
    history_path: str | Path = "reports/training_history.json",
    figures_dir: str | Path = "reports/figures",
    report_path: str | Path = "reports/model_evaluation.md",
    metrics_json_path: str | Path = "reports/metrics.json",
) -> dict[str, Any]:
    """Evaluate trained DigitCNN on the held-out MNIST test split."""
    ckpt = Path(model_path)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at '{ckpt}'. "
            "Run `python -m handwritten_recognition.train` first."
        )

    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    model = DigitCNN(num_classes=10)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    _, _, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_fraction=val_fraction,
        random_state=random_state,
        download=False,
    )

    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    sample_images: np.ndarray | None = None
    sample_targets: np.ndarray | None = None
    sample_preds: np.ndarray | None = None

    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images)
            preds = logits.argmax(dim=1).numpy()
            targets = labels.numpy()
            all_preds.append(preds)
            all_targets.append(targets)
            if sample_images is None:
                sample_images = images[:10].numpy()
                sample_targets = targets[:10]
                sample_preds = preds[:10]

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    metrics = compute_multiclass_metrics(y_true, y_pred)

    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_confusion_matrix(np.array(metrics["confusion_matrix"]), out_dir)
    if sample_images is not None and sample_targets is not None and sample_preds is not None:
        _plot_sample_predictions(sample_images, sample_targets, sample_preds, out_dir)

    hist_file = Path(history_path)
    history: list[dict[str, float]] = []
    if hist_file.exists():
        history = json.loads(hist_file.read_text(encoding="utf-8"))
        _plot_training_curves(history, out_dir)

    metrics_out = Path(metrics_json_path)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report_lines = [
        "# Handwritten Digit Recognition (MNIST CNN) — Evaluation Report",
        "",
        f"- **Test Accuracy**: {metrics['accuracy']:.4f}",
        f"- **Macro Precision**: {metrics['precision_macro']:.4f}",
        f"- **Macro Recall**: {metrics['recall_macro']:.4f}",
        f"- **Macro F1-Score**: {metrics['f1_macro']:.4f}",
        f"- **Weighted F1-Score**: {metrics['f1_weighted']:.4f}",
        f"- **Test Samples**: {len(y_true)}",
        "",
        "## Per-Class Classification Report",
        "",
        "```text",
        metrics["classification_report"].strip(),
        "```",
    ]
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for evaluating DigitCNN."""
    parser = argparse.ArgumentParser(description="Evaluate MNIST DigitCNN.")
    parser.add_argument("--config", default="configs/config.json", help="Path to config JSON.")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        metrics = evaluate_model(
            model_path=cfg["training"]["model_output_path"],
            data_dir=cfg["dataset"]["data_dir"],
            batch_size=cfg["training"]["batch_size"],
            val_fraction=cfg["dataset"]["val_fraction"],
            random_state=cfg["dataset"]["random_state"],
            history_path=cfg["training"]["history_output_path"],
            figures_dir=cfg["evaluation"]["figures_dir"],
            report_path=cfg["evaluation"]["report_path"],
            metrics_json_path=cfg["evaluation"]["metrics_json_path"],
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Test Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['f1_macro']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
