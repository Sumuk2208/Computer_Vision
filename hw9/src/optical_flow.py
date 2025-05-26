from pathlib import Path

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from utils import (
    gaussian_pyramid,
    visualize_flow_hsv,
    visualize_flow_quiver,
    uint8_to_float32,
    sliding_window_view,
)


def solve_optical_flow_constraint_equation_lucas_kanade(
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    grad_t: np.ndarray,
    window_size: int,
    alpha: float = 1e-3,
):
    """Solve the optical flow constraint equation using the least squares method and a local window
    around each pixel (Lucas-Kanade method)

    Args:
        grad_x: Spatial gradient in the x direction, as a (h, w) array.
        grad_y: Spatial gradient in the y direction, as a (h, w) array.
        grad_t: Temporal gradient, as a (h, w) array.
        window_size: size of 'sliding window' used to introduce more equations per pixel (Lucas-
            Kanade method)
        alpha: Regularization parameter for ridge regression.

    Returns:
        uv: The estimated optical flow, as a (h, w, 2) array.
    """
    h, w = grad_x.shape[:2]

    # Slice a window x window region around each pixel. Result is shape (h, w, window**2)
    grad_x_windows = sliding_window_view(grad_x, window_size, mode="reflect")
    grad_y_windows = sliding_window_view(grad_y, window_size, mode="reflect")
    grad_t_windows = sliding_window_view(grad_t, window_size, mode="reflect")

    batchA = np.stack(
        [
            grad_x_windows.reshape(h * w, window_size**2),
            grad_y_windows.reshape(h * w, window_size**2),
        ],
        axis=-1,
    )
    batchB = -grad_t_windows.reshape(h * w, window_size**2)

    batchATA = np.einsum("bwx,bwy->bxy", batchA, batchA)
    batchATA += alpha * np.eye(2)[None, :, :]
    batchATB = np.einsum("bwx,bw->bx", batchA, batchB)

    uv = np.array([np.linalg.solve(a, b) for a, b in zip(batchATA, batchATB)])

    return uv.reshape(h, w, 2)


def warp_by_vector_field(image: np.ndarray, vectors_uv: np.ndarray):
    """Warp an image using the given vector field. Usage is like this:

        approx_frame[t+1] = warp_by_vector_field(frame[t], flow_uv[t])

    Args:
        - image: The image to be warped, as a (h, w) array.
        - vectors_uv: The vector field to warp by, as a (h, w, 2) array.

    Returns:
        - warped_image: The warped image, as a (h, w) array.
    """
    h, w = image.shape[:2]

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x - vectors_uv[..., 0]).astype(np.float32)
    map_y = (grid_y - vectors_uv[..., 1]).astype(np.float32)
    warped_image = cv.remap(
        image, map_x, map_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT
    )
    return warped_image


def calculate_gradients(image1: np.ndarray, image2: np.ndarray):
    """Get I_x, I_y, I_t gradients.

    Each must have interpretable units, e.g. change in brightness per pixel for the spatial
    gradients and change in brightness per frame for the temporal gradient.

    Args:
        image1: The first image, as a (h, w) array.
        image2: The second image, as a (h, w) array.

    Returns:
        gradient_x: The spatial gradient in the x direction, as a (h, w) array.
        gradient_y: The spatial gradient in the y direction, as a (h, w) array.
        gradient_t: The temporal gradient, as a (h, w) array.
    """
    grad_x1 = cv.Sobel(image1, cv.CV_32F, 1, 0, ksize=3, scale=1 / 8)
    grad_x2 = cv.Sobel(image2, cv.CV_32F, 1, 0, ksize=3, scale=1 / 8)
    gradient_x = (grad_x1 + grad_x2) / 2

    grad_y1 = cv.Sobel(image1, cv.CV_32F, 0, 1, ksize=3, scale=1 / 8)
    grad_y2 = cv.Sobel(image2, cv.CV_32F, 0, 1, ksize=3, scale=1 / 8)
    gradient_y = (grad_y1 + grad_y2) / 2
    tent_kernel = np.array(
        [[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]], dtype=np.float32
    )
    smoothed_1 = cv.filter2D(image1, -1, tent_kernel)
    smoothed_2 = cv.filter2D(image2, -1, tent_kernel)

    gradient_t = smoothed_2 - smoothed_1
    return gradient_x, gradient_y, gradient_t


def coarse_to_fine_optical_flow(
    image1: np.ndarray,
    image2: np.ndarray,
    levels: int,
    window_size: int,
    alpha: float = 1e-3,
):
    assert image1.ndim == 2 and image2.ndim == 2, "Color images not supported"
    original_size = image1.shape[:2]
    pyramid1 = gaussian_pyramid(image1, levels, resize=True)
    pyramid2 = gaussian_pyramid(image2, levels, resize=True)

    # Initialize flow to zeros at the coarsest level
    flow_uv = np.zeros((*pyramid1[-1].shape, 2), dtype=np.float32)

    for level in reversed(range(levels)):
        img1 = pyramid1[level]
        img2 = pyramid2[level]

        # Upsample flow from previous level (if not coarsest)
        if level < levels - 1:
            h, w = img1.shape
            flow_uv = cv.pyrUp(flow_uv)
            flow_uv = cv.resize(flow_uv, (w, h), interpolation=cv.INTER_LINEAR)
            flow_uv *= 2.0  # account for resolution scaling

        # Warp image1 using current flow estimate
        warped_img1 = warp_by_vector_field(img1, flow_uv)

        # Compute gradients between warped image1 and img2
        grad_x, grad_y, grad_t = calculate_gradients(warped_img1, img2)

        # Estimate residual flow using LK
        residual_flow = solve_optical_flow_constraint_equation_lucas_kanade(
            grad_x, grad_y, grad_t, window_size, alpha
        )

        # Update flow with residual
        flow_uv += residual_flow

    # Rescale flow to match original image resolution
    scale = np.array(original_size) / np.array(flow_uv.shape[:2])
    flow_uv = scale * cv.resize(flow_uv, original_size[::-1], interpolation=cv.INTER_LINEAR)

    return flow_uv


def calculate_optical_flow_pyr_lk_cross_check(
    image1: np.ndarray,
    image2: np.ndarray,
    levels: int,
    window_size: int,
    alpha: float,
    goodness_threshold: float,
):
    forward_flow = coarse_to_fine_optical_flow(image1, image2, levels, window_size, alpha)
    reverse_flow = coarse_to_fine_optical_flow(image2, image1, levels, window_size, alpha)
    error = np.linalg.norm(forward_flow + reverse_flow, axis=-1)
    good_mask = error < goodness_threshold
    avgflow = (forward_flow - reverse_flow) / 2
    flow = np.full_like(forward_flow, np.nan)
    flow[good_mask] = avgflow[good_mask]
    return flow


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Two-frame optical flow")
    parser.add_argument("image1", help="First image", type=Path)
    parser.add_argument("image2", help="Second image", type=Path)
    parser.add_argument("--levels", help="Number of pyramid levels", type=int, default=5)
    parser.add_argument("--window_size", help="Window size", type=int, default=7)
    parser.add_argument("--alpha", help="Regularization parameter", type=float, default=1e-3)
    parser.add_argument(
        "--goodness-threshold",
        help="Mismatch threshold for forward/reverse flow to be 'good'",
        type=float,
        default=2.0,
    )
    args = parser.parse_args()

    # Read the images
    args.image1 = uint8_to_float32(cv.imread(str(args.image1), cv.IMREAD_GRAYSCALE))
    args.image2 = uint8_to_float32(cv.imread(str(args.image2), cv.IMREAD_GRAYSCALE))

    assert args.image1.shape == args.image2.shape, "Images must have the same shape"

    flow = calculate_optical_flow_pyr_lk_cross_check(
        args.image1,
        args.image2,
        args.levels,
        args.window_size,
        args.alpha,
        args.goodness_threshold,
    )

    plt.figure(figsize=(args.image1.shape[1] / 100, args.image1.shape[0] / 100), dpi=300)
    visualize_flow_quiver(flow, spacing=10)
    plt.show()

    plt.figure(figsize=(args.image1.shape[1] / 100, args.image1.shape[0] / 100), dpi=300)
    plt.imshow(visualize_flow_hsv(flow, max_magnitude=2**args.levels))
    plt.show()
