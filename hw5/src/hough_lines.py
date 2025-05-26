import cv2 as cv
import numpy as np
from typing import Optional


class HoughLineDetector:
    def __init__(
        self,
        image_shape: tuple[int, int],
        min_angle: float = 0,
        max_angle: float = np.pi,
        angle_spacing: float = np.pi / 180,
        offset_spacing: float = 2.0,
    ):
        h, w = image_shape

        # We'll use the center of the image as our "origin" for the coordinate system for lines.
        self.origin_xy = np.array([w / 2, h / 2])

        # Largest possible offset is the distance from the origin to the corner of the image.
        max_offset = np.sqrt(h**2 + w**2) / 2

        num_offsets = int(np.ceil(2 * max_offset / (offset_spacing or 2)))
        num_angles = int(np.ceil(np.pi / angle_spacing))

        # Create a coordinate system of offsets (rho) and angles (theta) for the parameter space.
        self.offsets = np.linspace(-max_offset, max_offset, num_offsets)
        # We don't want to include the same angle twice (e.g. both 0 and pi denote vertical lines),
        # so we'll create num_angles+1 values and exclude the last one.
        self.angles = np.linspace(0, np.pi, num_angles + 1)[:num_angles]

        self._min_angle = min_angle
        self._max_angle = max_angle

        # Precompute a 2xnum_angles array of cosines and sines of the angles.
        self._cos_sin = np.stack([np.cos(self.angles), np.sin(self.angles)], axis=0)

        # Initialize self.accumulator to be a 2D array of zeros with the same shape as the
        # parameter space. The value in accumulator[i,j] represents the 'votes' for a line with
        # rho = offsets[i] and theta = angles[j].
        self.accumulator = np.zeros(shape=(len(self.offsets), len(self.angles)), dtype=float)

    def clear(self):
        self.accumulator = np.zeros(shape=(len(self.offsets), len(self.angles)), dtype=float)

    def add_edges(self, edges: np.ndarray):
        """Increment the accumulator for all lines passing through edge points."""
        xy = np.argwhere(edges)[:, ::-1]  # Get edge coordinates (x, y)
        xy = xy - self.origin_xy  # Adjust to the image center

        # Compute rho values for each (x, y) at all angles
        rhos = xy @ self._cos_sin  # Shape: (num_points, num_angles)

        # Convert rhos to index positions
        rho_indices = (rhos - self.offsets[0]) / (self.offsets[1] - self.offsets[0])

        # Clip rho indices to be within the valid range [0, len(self.offsets) - 1]
        rho_indices = np.clip(rho_indices, 0, len(self.offsets) - 1).astype(int)

        # Vectorized accumulation of votes
        valid_indices = (rho_indices >= 0) & (rho_indices < len(self.offsets))
        valid_rho_indices, valid_theta_indices = (
            rho_indices[valid_indices],
            np.where(valid_indices)[1],
        )
        # Increment the accumulator at the valid (rho, theta) pairs
        np.add.at(self.accumulator, (valid_rho_indices, valid_theta_indices), 1)

    def non_maximal_suppression(self, angle_range: float, offset_range: float) -> np.ndarray:
        """Do non-maximal suppression on the accumulator array. This means that for each cell in
        the accumulator which represents some line, compare it to other 'neighboring' lines. Two
        lines are considered 'neighbors' if their difference in angle is < angle_range and their
        difference in offset is < offset_range.

        Note that this requires a bit of care where 'angles' wrap around. To handle this,
        we'll use a temporary wrap-around padding on the accumulator.
        """
        h, w = self.accumulator.shape

        angle_spacing = self.angles[1] - self.angles[0]
        offset_spacing = self.offsets[1] - self.offsets[0]
        angle_window = int(np.ceil(angle_range / angle_spacing))
        offset_window = int(np.ceil(offset_range / offset_spacing))

        angle_mask = np.logical_and(self._min_angle <= self.angles, self.angles <= self._max_angle)
        acc = self.accumulator * angle_mask[None, :]

        # Wrap the accumulator around the angle axis and flip the offset axis to account for the
        # fact that rho values at theta=0 and theta=pi are the same distance but opposite signs.
        wrapped_acc = np.hstack(
            [
                acc[::-1, -angle_window:],
                acc,
                acc[::-1, :angle_window],
            ]
        )

        # Use cv.dilate to run NMS on the wrapped accumulator.
        kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE, (2 * offset_window + 1, 2 * angle_window + 1)
        )

        is_local_max = cv.dilate(wrapped_acc, kernel) == wrapped_acc
        return acc * is_local_max[:, angle_window : angle_window + w]

    def get_lines(
        self, threshold: float, nms_angle_range: float, nms_offset_range: float
    ) -> np.ndarray:
        """Return (rho, theta) pairs that have enough votes and are local maxima."""
        votes = self.non_maximal_suppression(nms_angle_range, nms_offset_range)

        # Normalize votes
        max_votes = votes.max()
        if max_votes > 0:
            votes /= max_votes

        # Extract (rho, theta) pairs where votes exceed threshold
        line_indices = np.argwhere(votes > threshold)
        return np.column_stack((self.offsets[line_indices[:, 0]], self.angles[line_indices[:, 1]]))

    def line_to_p(self, offset, angle):
        """Convert a line (rho, theta) to the point (x, y) in the image coordinate system on the
        line closest to the origin.
        """
        return np.array([np.cos(angle), np.sin(angle)]) * offset + self.origin_xy

    def line_to_xy_endpoints(self, offset, angle, length: Optional[float] = None):
        """Convert a line (offset, angle) to a pair of points (x1, y1), (x2, y2) which are the
        endpoints of the line.
        """
        if length is None:
            length = float(self.offsets.max() - self.offsets.min())
        p = self.line_to_p(offset, angle)
        v = np.array([np.sin(angle), -np.cos(angle)])
        return p + v * length / 2, p - v * length / 2


def main(
    image: np.ndarray,
    canny_blur: int,
    canny_thresholds: tuple[float, float],
    min_angle: float,
    max_angle: float,
    angle_spacing: float,
    offset_spacing: float,
    accumulator_threshold: float,
    nms_angle_range: float,
    nms_offset_range: float,
) -> np.ndarray:
    annotated_image = image.copy()

    # Convert to grayscale.
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image before running Canny edge detection.
    image = cv.GaussianBlur(image, (canny_blur, canny_blur), 0)

    # Run Canny edge detection.
    edges = cv.Canny(image, *canny_thresholds)

    # Create a HoughLineDetector object.
    hough = HoughLineDetector(image.shape[:2], min_angle, max_angle, angle_spacing, offset_spacing)

    # Iterate over the edges and add each edge to the HoughLineDetector.
    hough.add_edges(edges)

    # Get the lines from the HoughLineDetector.
    lines = hough.get_lines(accumulator_threshold, nms_angle_range, nms_offset_range)

    # Draw the lines on the original image.
    for offset, angle in lines:
        p1, p2 = hough.line_to_xy_endpoints(offset, angle)
        cv.line(
            annotated_image,
            tuple(p1.astype(int)),
            tuple(p2.astype(int)),
            (0, 0, 255),
            2,
            cv.LINE_AA,
        )

    return annotated_image
