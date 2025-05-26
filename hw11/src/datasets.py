from typing import Callable
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import (
    ToTensor,
    Normalize,
    Compose,
    RandomHorizontalFlip,
    RandomCrop,
    ColorJitter,
    RandomRotation,
)

from config import DATA_ROOT


def get_transform(train: bool) -> Callable:
    """Return a transform for the CIFAR datasets. When train is True, include data augmentation.
    When train is False, only include normalization.
    """
    if train:
        return Compose(
            [
                RandomHorizontalFlip(p=0.5),
                RandomCrop(32, padding=4),
                ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                RandomRotation(degrees=10),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    else:
        return Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def get_datasets(dataset_name: str) -> tuple[Dataset, Dataset]:
    """Return training and validation datasets for the named dataset."""
    if dataset_name == "cifar10":
        ds_train = CIFAR10(
            root=DATA_ROOT / "cifar10", train=True, transform=get_transform(True), download=True
        )
        ds_val = CIFAR10(root=DATA_ROOT / "cifar10", train=False, transform=get_transform(False))
    elif dataset_name == "cifar100":
        ds_train = CIFAR100(
            root=DATA_ROOT / "cifar100", train=True, transform=get_transform(True), download=True
        )
        ds_val = CIFAR100(root=DATA_ROOT / "cifar100", train=False, transform=get_transform(False))
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    return ds_train, ds_val


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str.lower, default="cifar10", choices=["cifar10", "cifar100"]
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    ds_train, ds_val = get_datasets(args.dataset)
    print(f"Training Dataset: {args.dataset}")
    print(f"Size: {len(ds_train)}")
    print(f"Classes: {ds_train.classes if args.dataset == 'cifar10' else '100 classes'}")
    print(f"\nValidation Dataset Size: {len(ds_val)}")
