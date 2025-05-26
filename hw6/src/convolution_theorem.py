import argparse
from pathlib import Path

import cv2 as cv
import numpy as np

from utils import uint8_to_float, text_to_array


def conv2D(image: np.ndarray, h: np.ndarray, **kwargs) -> np.ndarray:
    """Apply a *convolution* operation rather than a *correlation* operation. Using the fact that
    convolution is equivalent to correlation with a flipped kernel, and cv.filter2D implements
    correlation.

    :param image: The input image
    :param h: The kernel
    :param kwargs: Additional arguments to cv.filter2D
    :return: The result of convolving the image with the kernel
    """
    return cv.filter2D(image, -1, cv.flip(h, -1), **kwargs)


def convolution_theorem(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """Replicate the behavior of conv2D with borderType=cv.BORDER_CONSTANT, but do it in the
    frequency domain using the convolution theorem.
    """
    # img_shape = image.shape[:2]
    # filter_shape = filter.shape[:2]
    #
    # # Compute the padded shape (same as zero-padding but without explicitly padding)
    # padded_shape = (img_shape[0] + filter_shape[0] - 1, img_shape[1] + filter_shape[1] - 1)
    #
    # # Compute 2D DFT of image and filter (with implicit zero-padding using 's' argument)
    # image_f = np.fft.rfft2(image, s=padded_shape, axes=(0, 1))
    # filter_f = np.fft.rfft2(filter, s=padded_shape, axes=(0, 1))
    # if filter_f.ndim == 2:  # If filter is single-channel
    #     filter_f = np.expand_dims(filter_f, axis=-1)
    # # Element-wise multiplication in the frequency domain
    # conv_f = image_f * filter_f
    #
    # # Compute the inverse 2D DFT to get the result in spatial domain
    # convolved = np.fft.ifft2(conv_f, axes=(0, 1)).real
    #
    # # Crop to match the original image size
    # start_x, start_y = filter_shape[0] // 2, filter_shape[1] // 2
    # result = convolved[start_x : start_x + img_shape[0], start_y : start_y + img_shape[1]]
    #
    # return result
    img_shape = image.shape[:2]
    filter_shape = filter.shape[:2]

    # Compute the padded shape
    padded_shape = (img_shape[0] + filter_shape[0] - 1, img_shape[1] + filter_shape[1] - 1)

    # Convert to float32 to reduce memory usage
    image = image.astype(np.float32)
    filter = filter.astype(np.float32)

    # Ensure filter has the same number of channels as the image
    if image.ndim == 2:  # Grayscale image
        image = image[..., np.newaxis]  # Add an extra dimension for channels
    if filter.ndim == 2:  # if the filter is 2D, expand it to 3D to match image channels
        filter = filter[..., np.newaxis]

    # Broadcast filter across all channels
    filter = np.repeat(filter, image.shape[2], axis=-1)

    # Compute the 2D FFT for the entire image (all channels at once) and the filter
    image_f = np.fft.fft2(image, s=padded_shape, axes=(0, 1))
    filter_f = np.fft.fft2(filter, s=padded_shape, axes=(0, 1))

    # Element-wise multiplication in the frequency domain (no need for explicit loops)
    conv_f = image_f * filter_f

    # Compute the inverse 2D FFT to get the result in the spatial domain
    convolved = np.fft.ifft2(conv_f, axes=(0, 1)).real

    # Crop to match the original image size
    start_x, start_y = filter_shape[0] // 2, filter_shape[1] // 2
    result = convolved[start_x : start_x + img_shape[0], start_y : start_y + img_shape[1]]

    # If the image was grayscale, remove the extra dimension
    if result.ndim == 3 and result.shape[2] == 1:
        result = result[..., 0]

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("filter", type=Path)
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image {args.image} does not exist")
    if not args.filter.exists():
        raise FileNotFoundError(f"Filter {args.filter} does not exist")

    image = uint8_to_float(cv.imread(str(args.image)))
    filter = uint8_to_float(text_to_array(args.filter))

    out1 = conv2D(image, filter, borderType=cv.BORDER_CONSTANT)
    out2 = convolution_theorem(image, filter)

    assert np.allclose(out1, out2, atol=1e-6), "Results do not match"
