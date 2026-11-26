# Dataset Documentation — MNIST Handwritten Digits

- **Name**: MNIST (Modified National Institute of Standards and Technology)
- **Source**: Yann LeCun, Corinna Cortes, Christopher J.C. Burges (`http://yann.lecun.com/exdb/mnist/`) via `torchvision.datasets.MNIST`
- **Size**: 60,000 training images + 10,000 test images (28×28 grayscale, 10 classes: digits `0`–`9`)
- **Split**: 54,000 train / 6,000 validation (deterministic split, `seed=42`) / 10,000 official test set

## Setup

Download automatically via:

```bash
python scripts/download_data.py
```
