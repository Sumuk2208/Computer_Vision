import cv2 as cv
import numpy as np
import torch


def cv_to_torch(image: cv.Mat) -> torch.Tensor:
    return torch.from_numpy(image / 255.0).float().permute(2, 0, 1).unsqueeze(0)


def torch_to_cv(image: torch.Tensor) -> list[cv.Mat]:
    return [np.clip(np.round(im.permute(1, 2, 0).numpy()), 0, 255).astype(np.uint8) for im in image]


def augment(points: torch.Tensor, dim=0) -> torch.Tensor:
    ones_shape = list(points.shape)
    ones_shape[dim] = 1
    return torch.concatenate([points, torch.ones(ones_shape)], dim)
