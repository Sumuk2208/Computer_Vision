import numpy as np
from pathlib import Path


def uint8_to_float(image: np.ndarray) -> np.ndarray:
    """Converts a uint8 image to a float image in the range [0.0, 1.0]."""
    if image.dtype.kind == "f":
        return image
    elif image.dtype == np.uint8:
        return image / 255.0
    else:
        raise TypeError(f"Unsupported image type: {image.dtype}")


def float_to_uint8(image: np.ndarray) -> np.ndarray:
    """Converts a float image in the range [0.0, 1.0] to a uint8 image."""
    if image.dtype == np.uint8:
        return image
    if image.dtype.kind == "f":
        return np.clip(np.round(image * 255), 0, 255).astype(np.uint8)
    else:
        raise TypeError(f"Unsupported image type: {image.dtype}")


def text_to_array(file: Path) -> np.ndarray:
    """Read a text file containing a matrix of numbers and return it as a numpy array. The file
    should contain one row per line, with the numbers separated by whitespace. Returned array will
    always have ndim=2 even if the file contains a row or column vector.

    :param file: The file to read
    """
    with open(file) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return np.array([[float(x) for x in line.split()] for line in lines])
