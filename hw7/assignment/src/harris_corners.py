import cv2 as cv
import numpy as np
from utils import uint8_to_float32, non_maximal_suppression


def my_harris_corners(image: np.ndarray, block_size: int, k: float) -> np.ndarray:
    """Compute the Harris response function for an image."""
    image = uint8_to_float32(image)  # Normalize image to [0,1]

    # Compute image gradients using Sobel filters
    Ix = cv.Sobel(image, cv.CV_32F, 1, 0, ksize=3)
    Iy = cv.Sobel(image, cv.CV_32F, 0, 1, ksize=3)

    # Compute second-moment matrix components
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Apply box filter to compute sums over the neighborhood
    Sxx = cv.boxFilter(Ixx, -1, (block_size, block_size))
    Syy = cv.boxFilter(Iyy, -1, (block_size, block_size))
    Sxy = cv.boxFilter(Ixy, -1, (block_size, block_size))

    # Compute Harris response
    det_M = (Sxx * Syy) - (Sxy * Sxy)
    trace_M = Sxx + Syy
    R = det_M - k * (trace_M**2)

    return R


def locate_corners(
    harris: np.ndarray, threshold: float, nms_block_size: int = 5
) -> list[tuple[int, int]]:
    """Find corners by applying thresholding and non-maximal suppression."""
    # Apply threshold
    corners = np.argwhere(harris > threshold)

    # Apply non-maximal suppression
    suppressed = non_maximal_suppression(harris, nms_block_size)

    # Get final corner points after suppression
    final_corners = [(x, y) for y, x in corners if suppressed[y, x]]

    return final_corners


def main(image_filename: str, blur_size=5, block_size=5, k=0.04):
    image = cv.imread(image_filename, cv.IMREAD_GRAYSCALE)
    blurred = cv.GaussianBlur(image, (blur_size, blur_size), 0)

    opencv_harris = cv.cornerHarris(
        blurred,
        blockSize=block_size,
        ksize=3,
        k=k,
    )
    my_harris = my_harris_corners(blurred, block_size, k)

    # Normalize both so we don't have to worry about scale
    opencv_harris = opencv_harris / opencv_harris.max()
    my_harris = my_harris / my_harris.max()

    # Create a window with a slider for the threshold
    window_name = "Harris Corners (OpenCV left / Custom right)"
    cv.namedWindow(window_name)

    def draw(threshold):
        opencv_xy = locate_corners(opencv_harris, threshold)
        my_xy = locate_corners(my_harris, threshold)

        opencv_display = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        my_display = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        for x, y in opencv_xy:
            cv.circle(opencv_display, (x, y), 3, (0, 0, 255), -1)
        for x, y in my_xy:
            cv.circle(my_display, (x, y), 3, (0, 0, 255), -1)

        cv.imshow(window_name, np.hstack([opencv_display, my_display]))

    def on_slider_update(slider_value):
        threshold = slider_value / 100
        draw(threshold)

    cv.createTrackbar("Threshold", window_name, 0, 100, on_slider_update)
    cv.setTrackbarPos("Threshold", window_name, 50)

    while cv.waitKey(1) != ord("q"):
        pass
    cv.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="The image to process")
    parser.add_argument(
        "--blur-size", type=int, default=5, help="The size of the Gaussian blur kernel"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=5,
        help="The size of the neighborhood to consider for each pixel",
    )
    parser.add_argument(
        "-k",
        type=float,
        default=0.04,
        help="A constant used to tune the sensitivity of the corner detector",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        parser.error(f"The file {args.image} does not exist")
    main(args.image, args.blur_size, args.block_size, args.k)
