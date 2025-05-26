from typing import Optional

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def uint8_to_float32(image: np.ndarray) -> np.ndarray:
    """Convert an image from uint8 format (ranging in [0,255]) to float format (ranging in [0,1])."""
    return image.astype(np.float32) / 255.0


def float32_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an image from float format (ranging in [0,1]) to uint8 format (ranging in [0,255])."""
    return np.clip(np.round(image * 255), 0, 255).astype(np.uint8)


def resize_for_pyramid(image: np.ndarray, levels: int) -> np.ndarray:
    """Resize an image so that its dimensions are divisible by 2^(levels - 1)."""
    height, width = image.shape[:2]
    divisible_by = 2 ** (levels - 1)
    new_height = int(np.ceil(height / divisible_by) * divisible_by)
    new_width = int(np.ceil(width / divisible_by) * divisible_by)
    return cv.resize(image, (new_width, new_height), interpolation=cv.INTER_LINEAR)


def gaussian_pyramid(image: np.ndarray, levels: int, resize: bool = True) -> list[np.ndarray]:
    """Create a Gaussian pyramid of an image with length `levels`."""
    if resize:
        image = resize_for_pyramid(image, levels)
    pyr = [image]
    for k in range(levels - 1):
        pyr.append(cv.pyrDown(pyr[k]))
    return pyr


def visualize_flow_hsv(flow_uv: np.ndarray, max_magnitude: Optional[float] = None):
    nan_mask = np.any(np.isnan(flow_uv), axis=2)
    flow_uv[nan_mask] = 0
    magnitude = np.linalg.norm(flow_uv, axis=2)
    if max_magnitude is None:
        max_magnitude = np.max(magnitude)
    angle = np.arctan2(flow_uv[..., 1], flow_uv[..., 0])
    hsv = np.zeros(flow_uv.shape[:2] + (3,), dtype=np.uint8)
    hsv[..., 0] = (angle + np.pi) * 180 / np.pi / 2
    hsv[..., 1] = np.clip(magnitude / max_magnitude * 255, 0, 255).astype(np.uint8)
    hsv[..., 2] = 255
    hsv[nan_mask, :] = 0
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)


def flow_hsv_key_image(size=100):
    u, v = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    flow_uv = np.dstack((u, v))
    im = visualize_flow_hsv(flow_uv, max_magnitude=1)
    im[np.linalg.norm(flow_uv, axis=2) > 1] = 0
    return im


def visualize_flow_quiver(flow_uv: np.ndarray, spacing: int = 10, ax=None):
    ax = ax or plt.gca()

    y, x = np.indices(flow_uv.shape[:2])

    y_subset = y[spacing // 2 :: spacing, spacing // 2 :: spacing]
    x_subset = x[spacing // 2 :: spacing, spacing // 2 :: spacing]
    u_subset = flow_uv[spacing // 2 :: spacing, spacing // 2 :: spacing, 0]
    v_subset = flow_uv[spacing // 2 :: spacing, spacing // 2 :: spacing, 1]

    ax.quiver(
        x_subset,
        y_subset,
        u_subset,
        v_subset,
        color="k",
        angles="xy",
        scale_units="xy",
        scale=1 / 2,
    )

    ax.axis("equal")
    ax.invert_yaxis()

    return ax


def sliding_window_view(image: np.ndarray, window_size: int, **pad_kwargs) -> np.ndarray:
    """Given a (h, w) single-plane image, return a (h,w,window_size**2) 'view' into the image
    where each (i,j) element of the view is a window_size x window_size window around the (i,j)
    element of the original image.
    """
    h, w = image.shape[:2]
    half_window = window_size // 2
    padded_image = np.pad(image, pad_width=half_window, **pad_kwargs)
    return np.lib.stride_tricks.sliding_window_view(
        padded_image, (window_size, window_size), axis=(0, 1)
    ).reshape(h, w, window_size**2)


def read_flo_file(file_path: str):
    """Read an optical flow file in .flo format. Docs:

    / ".flo" file format used for optical flow evaluation
    //
    // Stores 2-band float image for horizontal (u) and vertical (v) flow components.
    // Floats are stored in little-endian order.
    // A flow value is considered "unknown" if either |u| or |v| is greater than 1e9.
    //
    //  bytes  contents
    //
    //  0-3     tag: "PIEH" in ASCII, which in little endian happens to be the float 202021.25
    //          (just a sanity check that floats are represented correctly)
    //  4-7     width as an integer
    //  8-11    height as an integer
    //  12-end  data (width*height*2*4 bytes total)
    //          the float values for u and v, interleaved, in row order, i.e.,
    //          u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...
    //
    """
    with open(file_path, "rb") as f:
        # Read the magic number
        magic = np.fromfile(f, np.float32, count=1)[0]
        if magic != 202021.25:
            raise ValueError("Invalid .flo file format")

        # Read the dimensions
        width = np.fromfile(f, np.int32, count=1)[0]
        height = np.fromfile(f, np.int32, count=1)[0]

        # Read the flow data
        flow_data = np.fromfile(f, np.float32, count=2 * width * height)

        # Reshape to height x width x 2 (x and y components)
        flow = flow_data.reshape((height, width, 2))
        flow[flow > 1e9] = np.nan

    return flow
