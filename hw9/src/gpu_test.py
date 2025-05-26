import torch
import time
from tqdm import trange


print("GPU available?", torch.cuda.is_available())


def create_and_multiply_big_matrices(device="cpu"):
    x, y = torch.randn(5000, 5000, device=device), torch.randn(5000, 5000, device=device)
    z = x @ y
    return z


def main():
    devices = ["cpu", "cuda"]
    for device in devices:
        start = time.time()
        for _ in trange(100, desc=device):
            create_and_multiply_big_matrices(device)
        elapsed = time.time() - start

        print(f"Average time on {device}: {elapsed / 100}s per loop")


if __name__ == "__main__":
    main()
