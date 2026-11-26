"""CPU-friendly Convolutional Neural Network for handwritten digit recognition."""

from __future__ import annotations

import torch
import torch.nn as nn


class DigitCNN(nn.Module):
    """Compact CNN with Conv, BatchNorm, ReLU, MaxPool, Dropout, and Linear layers."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw class logits of shape (N, num_classes)."""
        if x.ndim != 4 or x.shape[1:] != (1, 28, 28):
            raise ValueError(f"Expected input shape (N, 1, 28, 28), got {tuple(x.shape)}.")
        x = self.features(x)
        return self.classifier(x)
