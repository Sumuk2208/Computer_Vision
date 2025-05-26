import numpy as np


def uint8_to_float32(image: np.ndarray) -> np.ndarray:
    """Convert an image from uint8 format (ranging in [0,255]) to float format (ranging in [0,1])."""
    return image.astype(np.float32) / 255.0


def float32_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an image from float format (ranging in [0,1]) to uint8 format (ranging in [0,255])."""
    return np.clip(np.round(image * 255), 0, 255).astype(np.uint8)


def ceil_16(value: int | float) -> int:
    """Round the given value up to the next multiple of 16"""
    return np.ceil(value / 16).astype(int) * 16


def fix16_to_float32(values: np.ndarray[np.int16], fractional=4) -> np.ndarray[np.float32]:
    return values / 2**fractional


def augment(points: np.ndarray, axis=0) -> np.ndarray:
    ones_shape = list(points.shape)
    ones_shape[axis] = 1
    return np.concatenate([points, np.ones(ones_shape)], axis)
