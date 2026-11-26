"""Unit tests for MNIST CNN preprocessing, model, training, evaluation, and inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from handwritten_recognition.evaluate import compute_multiclass_metrics
from handwritten_recognition.model import DigitCNN
from handwritten_recognition.predict import load_trained_model, predict_digit
from handwritten_recognition.preprocessing import preprocess_image
from handwritten_recognition.train import save_checkpoint, train_cnn


def _make_synthetic_loaders(
    n_samples: int = 32, batch_size: int = 8
) -> tuple[DataLoader, DataLoader]:
    rng = torch.Generator().manual_seed(42)
    x_train = torch.randn(n_samples, 1, 28, 28, generator=rng)
    y_train = torch.randint(0, 10, (n_samples,), generator=rng)
    x_val = torch.randn(16, 1, 28, 28, generator=rng)
    y_val = torch.randint(0, 10, (16,), generator=rng)
    return (
        DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False),
    )


def test_preprocess_image_shapes_and_auto_invert(tmp_path: Path) -> None:
    arr = np.full((64, 64, 3), 255, dtype=np.uint8)
    arr[20:44, 28:36] = 0  # dark stroke on white background
    img_path = tmp_path / "digit.png"
    Image.fromarray(arr).save(img_path)

    tensor = preprocess_image(img_path, auto_invert=True)
    assert tensor.shape == (1, 1, 28, 28)
    assert tensor.dtype == torch.float32


def test_preprocess_rejects_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        preprocess_image(tmp_path / "does_not_exist.png")
    with pytest.raises(ValueError):
        preprocess_image(np.zeros((10,), dtype=np.uint8))


def test_model_forward_and_shape_validation() -> None:
    model = DigitCNN(num_classes=10)
    out = model(torch.randn(4, 1, 28, 28))
    assert out.shape == (4, 10)
    with pytest.raises(ValueError):
        model(torch.randn(4, 3, 28, 28))


def test_train_save_reload_and_predict(tmp_path: Path) -> None:
    train_loader, val_loader = _make_synthetic_loaders()
    res1 = train_cnn(train_loader, val_loader, epochs=2, random_state=42)
    res2 = train_cnn(train_loader, val_loader, epochs=2, random_state=42)
    assert res1["history"][0]["train_loss"] == res2["history"][0]["train_loss"]

    ckpt_path = save_checkpoint(res1["state_dict"], res1["metadata"], tmp_path / "cnn.pt")
    loaded = load_trained_model(ckpt_path)

    dummy_img = Image.fromarray(np.zeros((28, 28), dtype=np.uint8))
    pred = predict_digit(dummy_img, loaded, auto_invert=False)
    assert 0 <= pred["predicted_digit"] <= 9
    assert pytest.approx(1.0, abs=1e-2) == sum(pred["probabilities"].values())


def test_compute_multiclass_metrics() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2])
    metrics = compute_multiclass_metrics(y_true, y_pred)
    assert metrics["accuracy"] == pytest.approx(5 / 6)
    assert len(metrics["confusion_matrix"]) == 3
