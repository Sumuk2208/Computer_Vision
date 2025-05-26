from pathlib import Path

import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose

from config import DATA_ROOT


def dataset_info(dataset: Dataset) -> str:
    num_samples = len(dataset)  # Number of samples is the length of the dataset
    # For classification datasets, we can get the number of classes from the classes attribute
    num_classes = len(dataset.classes) if hasattr(dataset, "classes") else "unknown"
    # Get shape and dtype from the first sample
    sample = dataset[0][0] if isinstance(dataset[0], (tuple, list)) else dataset[0]
    im_shape = tuple(sample.shape)  # Shape of the image tensor
    im_dtype = sample.dtype  # Data type of the image tensor
    return (
        f"Dataset with {num_samples} samples, "
        f"{num_classes} classes, image shape torch.Size({list(im_shape)}) "
        f"and dtype {im_dtype}."
    )


def dataloader_info(dataloader: DataLoader):
    num_batches = len(dataloader)  # Number of batches is the length of the dataloader
    num_samples = len(dataloader.dataset)  # Total samples from the underlying dataset
    batch_size = dataloader.batch_size  # Batch size from the dataloader
    # Get shape from the first batch
    first_batch = next(iter(dataloader))
    batch_data = first_batch[0] if isinstance(first_batch, (tuple, list)) else first_batch
    batch_shape = tuple(batch_data.shape)  # Shape of a batch (batch_size, channels, height, width)
    return (
        f"DataLoader with {num_samples} total samples "
        f"split across {num_batches} batches of size {batch_size}. "
        f"Batch shape is torch.Size({list(batch_shape)})."
    )


def get_dataloader(
    data_root: str | Path, dataset_name: str, batch_size: int, train: bool, **kwargs
) -> DataLoader:
    data_root = Path(data_root)
    if dataset_name == "mnist":
        transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
        ds = torchvision.datasets.MNIST(root=data_root / "mnist", train=train, transform=transform)
    elif dataset_name == "cifar10":
        transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        ds = torchvision.datasets.CIFAR10(
            root=data_root / "cifar10", train=train, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    return DataLoader(ds, batch_size=batch_size, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str.lower, default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    args = parser.parse_args()

    dl = get_dataloader(args.data_root, args.dataset, args.batch_size, train=True)
    ds = dl.dataset
    print(dataset_info(ds))
    print(dataloader_info(dl))
