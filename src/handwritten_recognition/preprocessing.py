"""Image preprocessing and normalization utilities for MNIST."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
IMAGE_SIZE = (28, 28)


def build_mnist_transform() -> transforms.Compose:
    """Return standard MNIST tensor conversion and normalization transform."""
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )


def preprocess_image(
    image_input: str | Path | Image.Image | np.ndarray,
    auto_invert: bool = True,
) -> torch.Tensor:
    """Preprocess an arbitrary image into a normalized (1, 1, 28, 28) tensor.

    Args:
        image_input: File path, PIL Image, or numpy array (H, W) / (H, W, C).
        auto_invert: Invert image if background appears light (mean > 127),
            matching MNIST white-on-black convention.

    Returns:
        Float tensor of shape (1, 1, 28, 28).

    Raises:
        FileNotFoundError: If a path is provided and does not exist.
        ValueError: If the image input type or shape is invalid.
    """
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Input image not found at '{path}'.")
        pil_img = Image.open(path)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    elif isinstance(image_input, np.ndarray):
        arr = image_input
        if arr.ndim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D image array, got shape {arr.shape}.")
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr)
    else:
        raise ValueError("Unsupported image input type. Provide a path, PIL.Image, or ndarray.")

    gray = pil_img.convert("L")
    if auto_invert and float(np.asarray(gray).mean()) > 127.0:
        gray = ImageOps.invert(gray)

    tensor = build_mnist_transform()(gray)
    return tensor.unsqueeze(0)
