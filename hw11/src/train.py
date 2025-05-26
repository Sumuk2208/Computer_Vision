import datetime
from pathlib import Path
from typing import Optional, Callable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange, tqdm

from datasets import get_datasets
from models import get_model
from early_stopping import EarlyStopper


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum()
        return correct / len(labels)


def train_single_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_data: DataLoader,
    loss_fn: Callable,
    step: int,
    device: torch.device,
    writer: Optional[SummaryWriter] = None,
) -> int:
    model.train()
    for im, label in tqdm(train_data, desc="training loop", position=0, leave=False):
        im, label = im.to(device), label.to(device)

        # Forward pass
        out = model(im)
        loss = loss_fn(out, label)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if writer is not None and step % 100 == 0:
            writer.add_scalar("loss/train", loss.item(), step)
            writer.add_scalar("acc/train", accuracy(out, label).item(), step)

        step += 1

    return step


def evaluate(
    model: nn.Module,
    val_data: DataLoader,
    criterion: Callable,
    step: int,
    device: torch.device,
    writer: Optional[SummaryWriter] = None,
) -> tuple[float, float]:
    """Evaluate the model on the validation data. Returns the accuracy and loss."""
    model.eval()
    with torch.no_grad():
        total_loss, total_acc = torch.zeros((), device=device), torch.zeros((), device=device)
        total_samples = 0

        for im, label in tqdm(val_data, desc="validation loop", position=1, leave=False):
            im, label = im.to(device), label.to(device)
            out = model(im)
            total_loss += criterion(out, label) * len(label)
            total_acc += accuracy(out, label) * len(label)
            total_samples += len(label)
        val_loss = total_loss / total_samples
        val_acc = total_acc / total_samples
        if writer is not None:
            writer.add_scalar("loss/val", val_loss, step)
            writer.add_scalar("acc/val", val_acc, step)
    return val_acc.item(), val_loss.item()


def train(
    model: nn.Module,
    train_data: Dataset,
    val_data: Dataset,
    num_epochs: int,
    batch_size: int,
    lr: float,
    lr_decay_amt: float,
    lr_decay_every: int,
    stopper_patience: int,
    device: torch.device,
    log_dir: Path,
):
    model.to(device)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(val_data, batch_size=500, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize components
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Changed to Adam
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_every, gamma=lr_decay_amt)
    early_stopper = EarlyStopper(patience=stopper_patience)

    start_epoch, step, history = 0, 0, []
    best_val_loss = float("inf")

    def _save(_file, _ep, _model, _optimizer, _scheduler, _early_stopper, _history):
        checkpoint = {
            "epoch": _ep,
            "name": _model.__class__.__name__,
            "history": _history,
            "model_state_dict": _model.state_dict(),
            "optimizer_state_dict": _optimizer.state_dict(),
            "scheduler_state_dict": _scheduler.state_dict(),
            "early_stopper_state_dict": _early_stopper.state_dict(),
            "best_val_loss": best_val_loss,
        }
        torch.save(checkpoint, _file)

    writer = SummaryWriter(log_dir / "tensorboard")

    # Load checkpoint if exists
    ckpt_latest = log_dir / "checkpoint_latest.pt"
    ckpt_best = log_dir / "checkpoint_best.pt"
    if ckpt_latest.exists():
        checkpoint = torch.load(ckpt_latest, map_location=device)
        history = checkpoint["history"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        early_stopper.load_state_dict(checkpoint["early_stopper_state_dict"])
        best_val_loss = checkpoint["best_val_loss"]
        start_epoch = checkpoint["epoch"] + 1
        step = start_epoch * len(train_loader)

    for _ep in trange(start_epoch, num_epochs, desc="Training Epochs"):
        # Training
        step = train_single_epoch(model, optimizer, train_loader, loss_fn, step, device, writer)

        # Validation
        val_acc, val_loss = evaluate(model, val_loader, loss_fn, step, device, writer)

        # Update scheduler
        scheduler.step()

        # Early stopping check
        if early_stopper.update(val_loss):
            print(f"Early stopping at epoch {_ep}")
            break

        # Save checkpoints
        history.append({"val_acc": val_acc, "val_loss": val_loss})
        _save(ckpt_latest, _ep, model, optimizer, scheduler, early_stopper, history)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save(ckpt_best, _ep, model, optimizer, scheduler, early_stopper, history)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str.upper, default="VGG11", choices=["VGG11", "VGG13", "VGG16", "VGG19"]
    )
    parser.add_argument("--log-dir", type=Path, default=Path("../checkpoints"))
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--device", type=torch.device, default=None)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-decay-amt", type=float, default=1.0)
    parser.add_argument("--lr-decay-every", type=int, default=float("inf"))
    parser.add_argument("--stopper-patience", type=int, default=float("inf"))
    args = parser.parse_args()

    if args.device is None:
        args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    hyperparameter_names = [
        "model",
        "num_epochs",
        "lr",
        "batch_size",
        "lr_decay_amt",
        "lr_decay_every",
        "stopper_patience",
    ]

    train_kwargs = {param: getattr(args, param) for param in hyperparameter_names}
    run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.log_dir = args.log_dir / run_name
    args.log_dir.mkdir(parents=True, exist_ok=True)

    train_data, val_data = get_datasets("cifar10")
    train_kwargs["model"] = get_model(args.model)

    train(
        **train_kwargs,
        train_data=train_data,
        val_data=val_data,
        device=args.device,
        log_dir=args.log_dir,
    )
