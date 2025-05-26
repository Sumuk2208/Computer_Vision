from pathlib import Path

import cv2 as cv
import numpy as np
from scene3d import plot_3d_points
from utils import ceil_16, fix16_to_float32


def disparity_to_3d(
    disparities: np.ndarray,
    focal_length: float,
    baseline_mm: float,
    dmin: float = 0.0,
) -> np.ndarray:
    """
    Convert a disparity map to 3D coordinates.

    Disparity formula is Z = f * B / (d + dmin), where:
    - Z is the true depth of the pixel in mm,
    - f is the focal length of the camera in pixels,
    - B is the baseline distance between cameras in mm,
    - d is the disparity value in pixels,
    - dmin is a value added to each disparity value (e.g. to account for image cropping).

    Any values where d=0 are ignored.

    Args:
        image (np.ndarray): (h, w, 3) RGB image.
        disparities (np.ndarray): (h, w, 1) disparity map.
        focal_length (float): Focal length of the camera in units of pixels.
        baseline_mm (float): Baseline x distance between cameras in millimeters.
        dmin (float): value added to each disparity value (e.g. to account for image cropping).

    Returns:
        - (N, 3) array of 3D points in (x, y, z) format, all in mm units.
    """
    h, w = disparities.shape
    mask = disparities > 0  # Ignore pixels with zero disparity

    # Compute the Z coordinate
    Z = np.zeros_like(disparities, dtype=np.float32)
    Z[mask] = (focal_length * baseline_mm) / (disparities[mask] + dmin)

    # Create meshgrid for pixel coordinates
    x_coords = np.arange(w) - w // 2
    y_coords = np.arange(h) - h // 2
    X, Y = np.meshgrid(x_coords, y_coords)

    # Compute real-world X and Y coordinates
    X = X * Z / focal_length
    Y = Y * Z / focal_length

    # Stack into (N, 3) format
    return np.column_stack((X[mask], Y[mask], Z[mask]))


def my_sad_disparity_map(
    img1: np.ndarray,
    img2: np.ndarray,
    window_size: int,
    max_disparity: int,
) -> np.ndarray:
    """Compute a disparity value for each pixel in img1 using the SAD metric.

    :param img1: left image
    :param img2: right image
    :param window_size: size of the window used in template matching
    :param max_disparity: maximum disparity value to search for
    :return: disparity map of same width and height as img1, in float32.
    """
    img1, img2 = np.atleast_3d(img1) / 255.0, np.atleast_3d(img2) / 255.0
    assert img1.shape == img2.shape, "Images must have the same shape."

    h, w, c = img1.shape
    disparities = np.arange(max_disparity)

    # Initialize SAD volume
    sad_volume = np.zeros((h, w, max_disparity), dtype=np.float32)

    # Compute absolute differences for each disparity
    for d in disparities:
        # Shift the right image by disparity d
        shifted_img2 = np.roll(img2, shift=d, axis=1)
        shifted_img2[:, :d] = 0  # Set the leftmost columns to 0 (no information)

        # Compute absolute difference
        abs_diff = np.abs(img1 - shifted_img2)

        # Sum across color channels
        abs_diff = np.sum(abs_diff, axis=2)

        # Apply box filter to compute SAD (sum over window)
        kernel = np.ones((window_size, window_size), dtype=np.float32)
        sad = cv.filter2D(abs_diff, -1, kernel, borderType=cv.BORDER_REPLICATE)

        sad_volume[..., d] = sad

    # Find disparity with minimum SAD for each pixel
    disparity_map = np.argmin(sad_volume, axis=2).astype(np.float32)

    return disparity_map


def my_improved_disparity_map(
    img1: np.ndarray,
    img2: np.ndarray,
    window_size: int = 9,
    max_disparity: int = None,
    min_disparity: int = 0,
    threshold: float = 1.5,
    use_ncc: bool = False,
    multi_scale: bool = True,
) -> np.ndarray:
    """Compute an improved disparity map using multiple optimization techniques.

    :param img1: left image
    :param img2: right image
    :param window_size: size of the window used in template matching
    :param max_disparity: maximum disparity value to search for
    :param min_disparity: minimum disparity value to search for
    :param threshold: threshold for invalid matches (relative to mean cost)
    :param use_ncc: whether to use NCC instead of SAD
    :param multi_scale: whether to use multi-scale processing
    :return: disparity map of same width and height as img1, in float32
    """
    # Convert images to float and normalize
    img1, img2 = np.atleast_3d(img1) / 255.0, np.atleast_3d(img2) / 255.0
    assert img1.shape == img2.shape, "Images must have the same shape."

    h, w, c = img1.shape

    # Set default max disparity if not provided
    if max_disparity is None:
        max_disparity = w // 8

    if multi_scale:
        return multi_scale_disparity(
            img1, img2, window_size, max_disparity, min_disparity, threshold, use_ncc
        )

    # Compute matching cost volume
    if use_ncc:
        cost_volume = compute_ncc_volume(img1, img2, window_size, max_disparity, min_disparity)
        # For NCC, we want maximum (best correlation) rather than minimum
        disparity_map = np.argmax(cost_volume, axis=2).astype(np.float32) + min_disparity
        best_scores = np.max(cost_volume, axis=2)
    else:
        cost_volume = compute_sad_volume(img1, img2, window_size, max_disparity, min_disparity)
        disparity_map = np.argmin(cost_volume, axis=2).astype(np.float32) + min_disparity
        best_scores = np.min(cost_volume, axis=2)

    # Sub-pixel refinement
    disparity_map = subpixel_refinement(cost_volume, disparity_map, min_disparity, use_ncc)

    # Invalid disparity detection
    mean_cost = np.mean(best_scores)
    invalid_mask = (
        (best_scores > threshold * mean_cost)
        if not use_ncc
        else (best_scores < -threshold * mean_cost)
    )
    disparity_map[invalid_mask] = 0

    return disparity_map


def compute_sad_volume(img1, img2, window_size, max_disparity, min_disparity):
    """Compute SAD cost volume"""
    h, w, c = img1.shape
    disparities = np.arange(min_disparity, max_disparity)
    sad_volume = np.zeros((h, w, len(disparities)), dtype=np.float32)

    kernel = np.ones((window_size, window_size), dtype=np.float32) / (window_size * window_size)

    for i, d in enumerate(disparities):
        shifted_img2 = np.roll(img2, shift=d, axis=1)
        shifted_img2[:, :d] = 0

        abs_diff = np.sum(np.abs(img1 - shifted_img2), axis=2)
        sad = cv.filter2D(abs_diff, -1, kernel, borderType=cv.BORDER_REPLICATE)
        sad_volume[..., i] = sad

    return sad_volume


def compute_ncc_volume(img1, img2, window_size, max_disparity, min_disparity):
    """Compute NCC cost volume"""
    h, w, c = img1.shape
    disparities = np.arange(min_disparity, max_disparity)
    ncc_volume = np.zeros((h, w, len(disparities)), dtype=np.float32)

    half_win = window_size // 2
    img1_pad = cv.copyMakeBorder(img1, half_win, half_win, half_win, half_win, cv.BORDER_REFLECT)
    img2_pad = cv.copyMakeBorder(img2, half_win, half_win, half_win, half_win, cv.BORDER_REFLECT)

    for i, d in enumerate(disparities):
        shifted_img2 = np.roll(img2_pad, shift=d, axis=1)
        shifted_img2[:, : d + half_win] = 0

        for y in range(half_win, h + half_win):
            for x in range(half_win, w + half_win):
                window1 = img1_pad[y - half_win : y + half_win + 1, x - half_win : x + half_win + 1]
                window2 = shifted_img2[
                    y - half_win : y + half_win + 1, x - half_win : x + half_win + 1
                ]

                if np.all(window1 == 0) or np.all(window2 == 0):
                    ncc_volume[y - half_win, x - half_win, i] = -1
                    continue

                mean1 = np.mean(window1)
                mean2 = np.mean(window2)
                std1 = np.std(window1)
                std2 = np.std(window2)

                if std1 < 1e-6 or std2 < 1e-6:
                    ncc_volume[y - half_win, x - half_win, i] = -1
                else:
                    ncc = np.sum((window1 - mean1) * (window2 - mean2)) / (
                        window_size**2 * std1 * std2
                    )
                    ncc_volume[y - half_win, x - half_win, i] = ncc

    return ncc_volume


def subpixel_refinement(cost_volume, disparity_map, min_disparity, use_ncc):
    """Quadratic interpolation for sub-pixel refinement"""
    h, w = disparity_map.shape
    refined_map = np.zeros_like(disparity_map)

    for y in range(h):
        for x in range(w):
            d = int(disparity_map[y, x] - min_disparity)
            if d <= 0 or d >= cost_volume.shape[2] - 1:
                refined_map[y, x] = disparity_map[y, x]
                continue

            c0 = cost_volume[y, x, d - 1]
            c1 = cost_volume[y, x, d]
            c2 = cost_volume[y, x, d + 1]

            if use_ncc:
                # For NCC, we have a maximum rather than minimum
                offset = (c0 - c2) / (2 * (c0 - 2 * c1 + c2)) if (c0 - 2 * c1 + c2) != 0 else 0
            else:
                # For SAD, we have a minimum
                offset = (c0 - c2) / (2 * (c0 - 2 * c1 + c2)) if (c0 - 2 * c1 + c2) != 0 else 0

            if abs(offset) <= 1.0:
                refined_map[y, x] = disparity_map[y, x] + offset
            else:
                refined_map[y, x] = disparity_map[y, x]

    return refined_map


def multi_scale_disparity(
    img1, img2, window_size, max_disparity, min_disparity, threshold, use_ncc, levels=3
):
    """Multi-scale disparity computation"""
    # Create pyramid
    pyramid1 = [img1]
    pyramid2 = [img2]
    for _ in range(1, levels):
        pyramid1.append(cv.pyrDown(pyramid1[-1]))
        pyramid2.append(cv.pyrDown(pyramid2[-1]))

    # Compute disparity from coarsest to finest
    disparity = np.zeros(pyramid1[-1].shape[:2], dtype=np.float32)

    for i in reversed(range(levels)):
        current_max = max(4, max_disparity // (2**i))
        current_min = min_disparity // (2**i) if i > 0 else min_disparity
        current_window = max(3, window_size // (2**i))

        if i != levels - 1:
            # Upscale previous disparity and multiply by 2
            disparity = cv.resize(
                disparity,
                (pyramid1[i].shape[1], pyramid1[i].shape[0]),
                interpolation=cv.INTER_LINEAR,
            )
            disparity = disparity * 2

        # Refine disparity at current level
        disparity = my_improved_disparity_map(
            pyramid1[i],
            pyramid2[i],
            window_size=current_window,
            max_disparity=current_max,
            min_disparity=current_min,
            threshold=threshold,
            use_ncc=use_ncc,
            multi_scale=False,
        )

    return disparity


def my_leaderboard_disparity_map(img1: np.ndarray, img2: np.ndarray):
    """Optimized version for best performance on the leaderboard"""
    return my_improved_disparity_map(
        img1,
        img2,
        window_size=7,
        max_disparity=None,
        min_disparity=0,
        threshold=1.2,
        use_ncc=False,
        multi_scale=True,
    )


def main(scene_folder: Path, baseline_mm: float, window_size: int, scale: float):
    # Load images
    img1 = cv.imread(str(scene_folder / "view1.png"), cv.IMREAD_COLOR)
    img2 = cv.imread(str(scene_folder / "view5.png"), cv.IMREAD_COLOR)

    if (scene_folder / "disp1.png").exists():
        true_disparities = cv.imread(str(scene_folder / "disp1.png"), cv.IMREAD_GRAYSCALE)
        with open(scene_folder / "dmin.txt") as f:
            dmin = int(f.readline())

        # Per the Middlebury middlebury-stereo data docs, disparity maps are stored relative to
        # full-resolution images and need to be scaled. But, values where true_disparities is zero
        # are considered unknown and should remain at zero.
        true_disparities[true_disparities > 0] = true_disparities[true_disparities > 0] * scale

        pl = plot_3d_points(
            disparity_to_3d(
                true_disparities,
                focal_length=3740 * scale,
                baseline_mm=baseline_mm,
                dmin=dmin * scale,
            ),
            colors=img1[true_disparities > 0, :].reshape(-1, 3)[:, ::-1],
        )
        pl.show(title="True depth from Middlebury dataset")

    # Heuristic: set max disparity to the smallest multiple of 16 that is larger than 1/8th the
    # image width. Note that StereoSGBM requires this to be a multiple of 16.
    max_disparity = ceil_16(img1.shape[1] / 8)

    # Compute disparity (OpenCV).
    cv_stereo_matcher = cv.StereoSGBM.create(
        minDisparity=0,
        numDisparities=max_disparity,
        mode=cv.STEREO_SGBM_MODE_HH,
        blockSize=window_size,
    )
    cv_disparities = fix16_to_float32(cv_stereo_matcher.compute(img1, img2), fractional=4)

    pl = plot_3d_points(
        disparity_to_3d(
            cv_disparities,
            focal_length=3740 * scale,
            baseline_mm=baseline_mm,
            dmin=dmin * scale,
        ),
        colors=img1[cv_disparities > 0, :].reshape(-1, 3)[:, ::-1],
    )
    pl.show(title="Depth inferred from OpenCV SGBM")

    # Compute disparity (custom implementation of SAD)
    my_disparities = my_sad_disparity_map(
        img1, img2, max_disparity=max_disparity, window_size=window_size
    )

    pl = plot_3d_points(
        disparity_to_3d(
            my_disparities,
            focal_length=3740 * scale,
            baseline_mm=baseline_mm,
            dmin=dmin * scale,
        ),
        colors=img1[my_disparities > 0, :].reshape(-1, 3)[:, ::-1],
    )
    pl.show(title="Depth inferred from SAD")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scene_folder",
        type=Path,
        help="Path to the scene folder which must at least contain view1.png, view5.png, and "
        "disp1.png.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=7,
        help="Size of the SAD box filter.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1 / 3,
        help="Scale factor for GT disparity. "
        "Defaults to 1/3, given from the Middlebury dataset docs.",
    )
    parser.add_argument(
        "--baseline-mm",
        type=float,
        default=160,
        help="Baseline distance between cameras for view1.png and view5.png. "
        "Defaults to 160, given from the Middlebury dataset docs.",
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=3740,
        help="Focal length of the camera in units of pixels. "
        "Defaults to 3740, given from the Middlebury dataset docs.",
    )
    args = parser.parse_args()
    main(args.scene_folder, args.baseline_mm, args.window_size, args.scale)
