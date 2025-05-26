import cv2 as cv
import numpy as np


class HoughCircleDetector:
    def __init__(
        self,
        image_shape: tuple[int, int],
        radius: float,
        soft_vote_sigma: float = 1.0,
    ):
        h, w = image_shape

        # We'll set the accumulator array resolution to 1 pixel per circle center, so accumulator
        # dimensions are the same as the image dimensions.

        # Create a grid of parameters (x and y centers of circles)
        self.center_x = np.arange(w)
        self.center_y = np.arange(h)
        self.radius = radius

        # Precompute array of votes, which will be the impulse-response of the accumulator to each
        # edge.
        ir_radius = int(np.ceil(radius + 3 * soft_vote_sigma))
        ir_x = np.linspace(-ir_radius, ir_radius, 2 * ir_radius + 1)
        ir_y = np.linspace(-ir_radius, ir_radius, 2 * ir_radius + 1)
        ir_r = np.sqrt(ir_x[None, :] ** 2 + ir_y[:, None] ** 2)
        self._accumulator_impulse_response = np.exp(
            -((ir_r - radius) ** 2) / soft_vote_sigma**2 / 2
        )

        # Initialize self.accumulator to be a 2D array of zeros with the same shape as the
        # parameter space
        self.accumulator = np.zeros(shape=(h, w), dtype=float)

    def clear(self):
        self.accumulator = np.zeros_like(self.accumulator)

    def add_edges(self, edges: np.ndarray):
        self.accumulator += cv.filter2D(
            edges.astype(np.float32), -1, self._accumulator_impulse_response
        )

    def non_maximal_suppression(self, nms_diameter: int):
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (nms_diameter, nms_diameter))
        local_max = cv.dilate(self.accumulator, kernel)
        self.accumulator[self.accumulator < local_max] = 0

    def get_circles(self, threshold: float, nms_radius: float) -> np.ndarray:
        self.non_maximal_suppression(int(np.ceil(2 * nms_radius)))
        max_votes = self.accumulator.max()
        candidates = np.argwhere(self.accumulator >= threshold * max_votes)
        return np.fliplr(candidates)  # Convert (row, col) -> (x, y)


def main(
    image: np.ndarray,
    canny_blur: int,
    canny_thresholds: tuple[float, float],
    accumulator_threshold: float,
    nms_radius: float,
    radius: float,
    soft_vote_sigma: float,
) -> np.ndarray:
    annotated_image = image.copy()

    # Convert to grayscale.
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image before running Canny edge detection.
    image = cv.GaussianBlur(image, (canny_blur, canny_blur), 0)

    # Run Canny edge detection.
    edges = cv.Canny(image, *canny_thresholds)

    # Create a HoughCircleDetector object.
    hough = HoughCircleDetector(image.shape[:2], radius, soft_vote_sigma)

    # Add the edge points to the HoughCircleDetector.
    hough.add_edges(edges)

    # Get the circles from the HoughCircleDetector.
    circles = hough.get_circles(accumulator_threshold, nms_radius)

    # Draw the circles on the original image.
    for cx, cy in circles:
        cv.circle(
            annotated_image, (int(cx), int(cy)), int(radius + 0.5), (0, 0, 255), 2, cv.LINE_AA
        )

    return annotated_image
