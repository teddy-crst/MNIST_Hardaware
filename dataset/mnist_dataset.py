from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import flatten
from torch.utils.data import Dataset, Subset, random_split
from torchvision import datasets, transforms

from plots.misc import save_or_show
from utils.settings import settings


def generate_mnist(
    portion: float = 1,
    image_size: Sequence[int] = (28, 28),
    root: Path | str | None = None,
    seed: int = 42,
    train_ratio: float = 0.7,
    test_ratio: float = 0.15,
) -> tuple[Dataset, Dataset, Dataset]:
    """Download and split the MNIST dataset.

    The function keeps the splitting logic in one place, makes the storage
    location configurable, and ensures reproducible splits by using a manual
    seed.
    """

    if portion <= 0 or portion > 1:
        raise ValueError("portion must be in the interval (0, 1]")

    if train_ratio <= 0 or test_ratio < 0 or (train_ratio + test_ratio) >= 1:
        raise ValueError("train_ratio and test_ratio must be positive and sum to less than 1")

    root_path = Path(root) if root else settings.dataset_dir
    root_path.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(flatten)
    ])

    dataset = datasets.MNIST(root=str(root_path), download=True, transform=transform)

    generator = torch.Generator().manual_seed(seed)
    subset_length = int(np.floor(portion * len(dataset)))
    subset_indices = torch.randperm(len(dataset), generator=generator)[:subset_length]
    dataset = Subset(dataset, subset_indices.tolist())

    train_size = int(train_ratio * len(dataset))
    test_size = int(test_ratio * len(dataset))
    val_size = len(dataset) - train_size - test_size
    trainset, testset, validationset = random_split(dataset, [train_size, test_size, val_size], generator=generator)

    return trainset, testset, validationset


def _reshape_flattened(image: torch.Tensor) -> torch.Tensor:
    return image.view(28, 28)


def _get_samples(dataset: Dataset, limit: int, generator: torch.Generator | None = None) -> Iterable[tuple[torch.Tensor, int]]:
    generator = generator or torch.Generator().manual_seed(settings.seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    for idx in indices:
        yield dataset[idx]


def preview_mnist_samples(dataset: Dataset, rows: int = 2, cols: int = 5, save_path: Path | None = None) -> None:
    """Render a compact grid of sample MNIST digits.

    The function respects the global visualization settings and can write the
    preview image to disk when ``settings.save_images`` is enabled.
    """

    total = rows * cols
    samples = list(_get_samples(dataset, total))
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes = axes.flatten()

    for ax, (image, label) in zip(axes, samples):
        ax.imshow(_reshape_flattened(image), cmap='gray')
        ax.set_title(f"Label: {label}")
        ax.axis('off')

    for ax in axes[len(samples):]:
        ax.axis('off')

    default_save_path = Path(save_path) if save_path else settings.dataset_dir / "mnist_preview.png"
    save_or_show(fig, default_save_path)