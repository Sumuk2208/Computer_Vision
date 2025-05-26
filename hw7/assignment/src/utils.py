import numpy as np
import cv2 as cv


def non_maximal_suppression(image: np.ndarray, block_size: int) -> np.ndarray:
    kernel = np.ones((block_size, block_size), np.uint8)
    return np.where(cv.dilate(image, kernel) == image, image, 0)


def uint8_to_float32(image: np.ndarray) -> np.ndarray:
    """Convert an image from uint8 format (ranging in [0,255]) to float format (ranging in [0,1])."""
    return image.astype(np.float32) / 255.0


def float32_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an image from float format (ranging in [0,1]) to uint8 format (ranging in [0,255])."""
    return np.clip(np.round(image * 255), 0, 255).astype(np.uint8)
