from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from models import get_model


def count_parameters(model):
    """Count the number of parameters in a model. Every tensor where requires_grad is True is
    counted.
    """
    # NOTE: this is a one-liner in the answer key. No need to over-complicate it!
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_metric_vs_num_parameters(checkpoints: list[Path], metric: str = "val_acc"):
    plt.figure(figsize=(5, 5))

    models_params = {}
    models_values = {}
    for checkpoint in checkpoints:
        checkpoint = torch.load(checkpoint, map_location="cpu")

        model = get_model(checkpoint["name"])

        key = checkpoint["name"]
        models_params[key] = count_parameters(model)
        models_values[key] = models_values.get(key, []) + [checkpoint["history"][-1][metric]]

    x = list(models_params.values())
    means = [np.mean(values) for values in models_values.values()]
    stderrs = [np.std(values) / np.sqrt(len(values)) for values in models_values.values()]
    plt.errorbar(x, y=means, yerr=stderrs, marker=".")
    for name, (x_value, y_value) in zip(models_params.keys(), zip(x, means)):
        plt.text(x_value, y_value, name)
    plt.xlabel("Num Parameters")
    plt.ylabel(metric)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("logs_dir", type=Path, default=Path("logs"))
    parser.add_argument("--metric", type=str, default="val_acc")
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("images"))
    args = parser.parse_args()

    checkpoints = list(args.logs_dir.glob("**/checkpoint_best.pt"))
    if not checkpoints:
        print(f"No checkpoints found in {args.logs_dir.resolve()}.")
    else:
        plot_metric_vs_num_parameters(checkpoints, args.metric)

    if not args.out_dir.exists():
        args.out_dir.mkdir(parents=True)

    plt.savefig(args.out_dir / f"{args.metric}_vs_num_parameters.png")
    plt.show()
