"""Single-image inference interface for MNIST DigitCNN."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from handwritten_recognition.model import DigitCNN
from handwritten_recognition.preprocessing import preprocess_image


def load_trained_model(model_path: str | Path = "models/mnist_cnn.pt") -> DigitCNN:
    """Load DigitCNN weights safely using weights_only=True."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at '{path}'. "
            "Run `python -m handwritten_recognition.train` first."
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Invalid checkpoint structure in '{path}'.")
    model = DigitCNN(num_classes=10)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def predict_digit(
    image_input: str | Path | Image.Image | np.ndarray,
    model: DigitCNN,
    auto_invert: bool = True,
) -> dict[str, Any]:
    """Predict handwritten digit (0-9) and class probabilities for one image."""
    tensor = preprocess_image(image_input, auto_invert=auto_invert)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=-1).squeeze(0).numpy()

    predicted_digit = int(np.argmax(probs))
    return {
        "predicted_digit": predicted_digit,
        "confidence": round(float(probs[predicted_digit]), 4),
        "probabilities": {str(i): round(float(probs[i]), 4) for i in range(10)},
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for predicting a handwritten digit from an image file."""
    parser = argparse.ArgumentParser(description="Predict handwritten digit from image.")
    parser.add_argument("--model", default="models/mnist_cnn.pt", help="Path to model checkpoint.")
    parser.add_argument("--image", required=True, help="Path to input digit image.")
    parser.add_argument(
        "--no-auto-invert",
        action="store_true",
        help="Disable automatic background inversion for white-on-black images.",
    )
    args = parser.parse_args(argv)

    try:
        model = load_trained_model(args.model)
        result = predict_digit(args.image, model, auto_invert=not args.no_auto_invert)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
