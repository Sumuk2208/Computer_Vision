import argparse
from pathlib import Path

import cv2 as cv
import numpy as np

from utils import uint8_to_float

# RECORD VALUES YOU CHOOSE AS SENSIBLE DEFAULTS HERE
LOW_FREQ_CUTOFF_DEFAULT = 0.05
HIGH_FREQ_CUTOFF_DEFAULT = 0.07


def frequency_coordinates(img_shape: tuple) -> (np.ndarray, np.ndarray):
    """Get the frequency coordinates for an image of a given size. In other words, the outputs
    are arrays f_x and f_y such that f_y[i, j] is the vertical frequency and f_x[i, j] is the
    horizontal frequency of the sinusoid denoted by im_f[i, j] where im_f is the fourier
    transform of an image of size img_shape.

    For more information, see np.fft.fftfreq.
    """
    fx_1d = np.fft.fftfreq(img_shape[1])
    fy_1d = np.fft.fftfreq(img_shape[0])
    return np.meshgrid(fx_1d, fy_1d)


def gaussian_low_pass_filter(img_shape: tuple, cutoff_freq: float) -> np.ndarray:
    """Create a gaussian low pass filter with a given cutoff frequency to be applied
    multiplicatively in the Fourier domain. The cutoff_freq defines the point at which the mask
    has a value of 0.5."""
    fx, fy = frequency_coordinates(img_shape)
    spatial_frequency = np.sqrt(fx**2 + fy**2)
    sigma = cutoff_freq / np.sqrt(2 * np.log(2))
    mask = np.exp(-1 / 2 * (spatial_frequency / sigma) ** 2)
    return np.atleast_3d(mask)


def low_pass(img: np.ndarray, cutoff_freq: float) -> np.ndarray:
    """Applies a low pass filter to the image by multiplying the fourier transform of the image by
    a mask.

    :param img: the image to filter
    :param cutoff_freq: the cutoff frequency (in units of cycles per pixel)
    :return: the filtered image
    """
    img_f = np.fft.fft2(img, axes=(0, 1))
    mask = gaussian_low_pass_filter(img.shape[:2], cutoff_freq)
    img_f = img_f * mask
    return np.fft.ifft2(img_f, axes=(0, 1)).real


def high_pass(img: np.ndarray, cutoff_freq: float) -> np.ndarray:
    """Applies a high pass filter to the image by multiplying the fourier transform of the image by
    a mask.

    :param img: the image to filter
    :param cutoff_freq: the cutoff frequency (in units of cycles per pixel)
    :return: the filtered image
    """
    return img - low_pass(img, cutoff_freq)


def hybrid(img_low: np.ndarray, img_high: np.ndarray, low_cutoff: float, high_cutoff: float):
    """Creates a hybrid image from a low pass filtered version of img_low and a high pass
    filtered version of img_high. All images should be in float [0.0, 1.0] format

    :param img_low: the image to low pass filter and use for the low frequencies
    :param img_high: the image to high pass filter and use for the high frequencies
    :param low_cutoff: the cutoff frequency for the low pass filter (in units of cycles per pixel)
    :param high_cutoff: the cutoff frequency for the high pass filter (in units of cycles per pixel)
    :return: the hybrid image
    """
    img_low_pass = low_pass(img_low, low_cutoff)
    img_high_pass = high_pass(img_high, high_cutoff)
    return np.clip(img_low_pass + img_high_pass, 0, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image1", type=Path)
    parser.add_argument("image2", type=Path)
    parser.add_argument("--low-cutoff", type=float, default=LOW_FREQ_CUTOFF_DEFAULT)
    parser.add_argument("--high-cutoff", type=float, default=HIGH_FREQ_CUTOFF_DEFAULT)
    args = parser.parse_args()

    if not args.image1.exists():
        print(f"Image {args.image1} not found")
        exit(1)

    if not args.image2.exists():
        print(f"Image {args.image2} not found")
        exit(1)

    img1 = uint8_to_float(cv.imread(str(args.image1)))
    # img1_low_pass = low_pass(img1, args.low_cutoff)
    # cv.imwrite(
    #     f"images/{args.image1.stem}_low_pass.png", float_to_uint8(img1_low_pass)
    # )
    #
    img2 = uint8_to_float(cv.imread(str(args.image2)))
    # img2_high_pass = high_pass(img2, args.high_cutoff)
    # cv.imwrite(
    #     f"images/{args.image2.stem}_high_pass.png",
    #     float_to_uint8(img2_high_pass + 0.5),
    # )
    #
    # img_hybrid = img1_low_pass + img2_high_pass
    # cv.imwrite(
    #     f"images/{args.image1.stem}_{args.image2.stem}_hybrid.png",
    #     float_to_uint8(img_hybrid),
    # )

    # hybrid_image = hybrid(img1, img2, args.low_cutoff, args.high_cutoff)
    # cv.imwrite(
    #     f"images/{args.image1.stem}_{args.image2.stem}_hybrid.png",
    #     float_to_uint8(hybrid_image),
    # )
