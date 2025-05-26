import numpy as np
import cv2 as cv
import os
from typing import Optional, NamedTuple
from enum import Enum


class Size(Enum):
    LARGE = "large"
    SMALL = "small"

    def __str__(self):
        return self.value


class Color(Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    CYAN = "cyan"
    BLUE = "blue"
    MAGENTA = "magenta"

    def __str__(self):
        return self.value


class Shape(Enum):
    CIRCLE = "circle"
    WEDGE = "wedge"
    RECTANGLE = "rectangle"
    CROSS = "cross"

    def __str__(self):
        return self.value


class ShapeInfo(NamedTuple):
    centroid_xy: tuple[float, float]
    area: float
    color: Color
    shape: Shape


COLOR_TO_HUE_LOOKUP = dict(zip(Color, [0, 30, 60, 90, 120, 150]))
MIN_AREA = 10
HUE_TOLERANCE = 10


def otsu_threshold(counts: np.ndarray, bins: Optional[np.ndarray] = None) -> float:
    """Given a histogram (a numpy array where counts[i] is # of values where x=bins[i]) return
    the threshold that minimizes intra-class variance using Otsu's method. If 'bins' is provided,
    then the x-coordinate of counts[i] is set by bins[i]. Otherwise, it is assumed that the
    x-coordinates are 0, 1, 2, ..., len(counts)-1. If provided, 'bins' must be sorted in
    ascending order and have the same length as 'counts'.

    Note: For didactic purposes, this function uses numpy only and does not rely on OpenCV.
    """
    if bins is not None:
        if not len(counts) == len(bins):
            raise ValueError("bins must have the same length as counts")
        if not np.all(bins[:-1] <= bins[1:]):
            raise ValueError("bins must be sorted in ascending order")
    else:
        bins = np.arange(len(counts))

    def variance_helper(bins_: np.ndarray, counts_: np.ndarray) -> float:
        n = np.sum(counts_)
        if n == 0:
            return 0
        mu = np.dot(bins_, counts_) / n
        return np.dot(counts_, (bins_ - mu) ** 2) / n

    # Test possible thresholds, which are all midpoints between adjacent bins
    possible_thresholds = (bins[:-1] + bins[1:]) / 2
    lowest_variance, best_threshold = float("inf"), 0
    for i, th in enumerate(possible_thresholds):
        variance_left = variance_helper(bins[: i + 1], counts[: i + 1])
        variance_right = variance_helper(bins[i + 1 :], counts[i + 1 :])
        total_variance = variance_left + variance_right
        if total_variance < lowest_variance:
            lowest_variance, best_threshold = total_variance, th
    return best_threshold


def roundedness(moments: dict[str, float]) -> float:
    """Given the moments of a shape, return the roundedness of the shape.

    Note: Uses central moments 'mu20', 'mu02', and 'mu11' to compute covariance and
    calculate the ratio of standard deviations along principal axes of the shape.
    """
    # Build the covariance matrix
    covariance = np.array([[moments["mu20"], moments["mu11"]], [moments["mu11"], moments["mu02"]]])

    # Calculate eigenvalues (variances) of the covariance matrix
    eigenvalues = np.linalg.eigvalsh(covariance)

    # The roundedness is the ratio of the smallest to the largest standard deviation
    stdevs = np.sqrt(eigenvalues)

    # Return the ratio of the smallest standard deviation to the largest
    return min(stdevs) / max(stdevs)


def threshold_on_hue(image: np.ndarray, color: Color, hue_tolerance: int = 10) -> np.ndarray:
    """The job of this function is to convert the image to binary form, where the shapes of a
    particular color are 1 and the background is 0. By 'color' we mean that the hue of a pixel is
    equal to the hue named by the given color string, plus or minus hue_tolerance. This is done
    by thresholding the image, and applying some morphological operations to clean up the result.
    """

    # Convert to HSV
    bgr_image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    hsv = cv.cvtColor(bgr_image, cv.COLOR_RGB2HSV)

    # Choose a saturation threshold using Otsu's method so that we select only the most saturated
    # pixels as ones.
    saturation_hist = cv.calcHist([hsv], [1], None, [256], [0, 256]).astype(np.int32).ravel()
    saturation_lo, saturation_hi = otsu_threshold(saturation_hist), 255

    # Any value
    value_lo, value_hi = 0, 255

    # The hue range will be set by the color plus or minus hue_tolerance. Because hues wrap around
    # at 0 and 180, it's possible that we try to keep a range like (175 ± 10), which really means
    # both ranges (165 to 180) and (0 to 5). To handle this 'wrapping around' behavior, shift all
    # the hue values so that the reference hue is in the middle of the hue range (90), and all hues
    # are in (0, 180). Then we can do a single thresholding operation on 90 ± hue_tolerance.
    reference_hue = COLOR_TO_HUE_LOOKUP[color]
    hsv[:, :, 0] = ((hsv[:, :, 0].astype(int) - reference_hue + 90) % 180).astype(np.uint8)

    # Do thresholding.
    binary = cv.inRange(
        hsv,
        lowerb=(90 - hue_tolerance, saturation_lo, value_lo),
        upperb=(90 + hue_tolerance, saturation_hi, value_hi),
    )

    # Apply morphological operations to clean up the binary image
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (6, 6))
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    # cv.imshow(f"binary image of color {color}", binary)
    # cv.waitKey(0)
    return binary


def is_shape_symmetric(
    binary: np.ndarray, centroid_xy: tuple[float, float], threshold: float, rotation: float = 0.0
) -> bool:
    """Given a binary image, return True if the shape is symmetric about its centroid when rotated
    by 'rotation' degrees, and False otherwise. Symmetry is determined by calculating intersection
    over union of (1) the original shape with (2) the rotated shape.
    """
    rot = cv.getRotationMatrix2D(centroid_xy, rotation, 1)
    flipped = cv.warpAffine(binary, rot, binary.shape)
    intersection = np.sum(binary & flipped)
    union = np.sum(binary | flipped)
    return intersection / union > threshold


def identify_single_shape(binary: np.ndarray) -> Shape:
    """Given a binary image that contains a single shape, return a string describing the shape, i.e.
    one of the SHAPE_OPTIONS shapes at the top of the file."""
    # Compute the moments of the shape and find its center of mass (AKA centroid)
    moments = cv.moments(binary)
    centroid_xy = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])

    # First, we can distinguish between (rectangles and wedges) vs (circles and crosses) by looking
    # at the roundedness of the shape. If the roundedness is high, it's a circle or cross. If it's
    # low, it's a rectangle or wedge.
    if roundedness(moments) < 0.5:
        # If roundedness is low, it's a rectangle or wedge. We can distinguish between these two
        # by checking if the shape is symmetric about its centroid through 180-degree rotation. If
        # it is, it's a rectangle. Otherwise, it's a wedge.
        if is_shape_symmetric(binary, centroid_xy, threshold=0.6, rotation=180):
            return Shape.RECTANGLE
        else:
            return Shape.WEDGE
    else:
        # If roundedness is high, it's a circle or cross. Crosses are symmetric for rotations of
        # 90 degrees, but we can distinguish them from circles by checking if the shape is symmetric
        # through 45-degree rotation. If it is, it's a circle. Otherwise, it's a cross.
        if is_shape_symmetric(binary, centroid_xy, threshold=0.6, rotation=45):
            return Shape.CIRCLE
        else:
            return Shape.CROSS


def find_shapes(
    image: np.ndarray,
    size: Size,
    color: Color,
    shape: Shape,
) -> np.ndarray:
    """Find all locations (centroids) in the image where there is a shape of the specified
    size, color, and shape type. Return the (x,y) locations of these centroids as a numpy array
    of shape (N, 2) where N is the number of shapes found.
    """
    # Create binary images for all shapes of any color, indexed by color
    shapes_by_color = {}
    for c in Color:
        shapes_by_color[c] = threshold_on_hue(image, c, hue_tolerance=HUE_TOLERANCE)

    # Collect all ShapeInfo objects for all shapes of all colors
    shape_info: list[ShapeInfo] = []
    for c, binary_image in shapes_by_color.items():
        # Find all connected components
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(binary_image)

        for i in range(1, num_labels):
            # Make a new temporary binary image with just the current shape
            shape_i_only = np.zeros_like(binary_image)
            shape_i_only[labels == i] = 255

            shape_info.append(
                ShapeInfo(
                    centroid_xy=tuple(centroids[i]),
                    area=stats[i, cv.CC_STAT_AREA].item(),
                    color=c,
                    shape=identify_single_shape(shape_i_only),
                )
            )

    # Filter out shapes that are not the type we're looking for
    shape_info = [s for s in shape_info if s.shape == shape]

    # Find a threshold that distinguishes 'small' from 'large' shapes
    areas, counts = np.unique([s.area for s in shape_info], return_counts=True)
    area_threshold = otsu_threshold(counts, bins=areas)

    # Filter out shapes that are the wrong color
    shape_info = [s for s in shape_info if s.color == color]

    # Filter out shapes that are the wrong size
    if size == Size.SMALL:
        shape_info = [s for s in shape_info if s.area < area_threshold]
    else:
        shape_info = [s for s in shape_info if s.area >= area_threshold]

    # Return the centroids of the remaining shapes
    return np.array([s.centroid_xy for s in shape_info])


def annotate_locations(image: np.ndarray, locs_xy: np.ndarray) -> np.ndarray:
    """Annotate the locations on the image by drawing circles on each (x,y) location"""
    annotated = image.copy()
    black, white, symbol_size = (0, 0, 0), (255, 255, 255), 8
    for x, y in locs_xy:
        x, y = int(x), int(y)
        cv.circle(annotated, (x, y), symbol_size, white, -1, cv.LINE_AA)
        cv.line(annotated, (x - symbol_size, y), (x + symbol_size, y), black, 1, cv.LINE_AA)
        cv.line(annotated, (x, y - symbol_size), (x, y + symbol_size), black, 1, cv.LINE_AA)
        cv.circle(annotated, (x, y), symbol_size, black, 1, cv.LINE_AA)
    return annotated


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument(
        "size",
        help="Return large or small shapes?",
        type=Size,
        choices=list(Size),
    )
    parser.add_argument(
        "color",
        help="Return shapes of a specific color?",
        type=Color,
        choices=list(Color),
    )
    parser.add_argument(
        "shape",
        help="Return shapes of a specific type?",
        type=Shape,
        choices=list(Shape),
    )
    parser.add_argument("--min-area", type=int, default=10, help="Minimum area of a shape")
    parser.add_argument("--hue-tolerance", type=int, default=10, help="Hue tolerance")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"File not found: {args.image}")

    # Load the image
    im = cv.imread(args.image)

    # Find the shapes
    locations = find_shapes(
        im,
        args.size,
        args.color,
        args.shape,
    )
    description = f"Located {len(locations)} {args.size} {args.color} {args.shape}s"

    # Annotate the locations on the image and display it
    cv.imshow(description, annotate_locations(im, locations))
    cv.waitKey(0)
    cv.destroyAllWindows()

# def main():
#     image="shapes.png"
#     sizes = ['large', 'small']
#     colors = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan']
#     shapes = ['cross', 'circle', 'rectangle', 'wedge']
#     for size in sizes:
#         for color in colors:
#             for shape in shapes:
#                 im = cv.imread(image)
#                 locations = find_shapes(im, Size[size.upper()], Color[color.upper()], Shape[shape.upper()])
#
#                 description = f"Located {len(locations)} {size} {color} {shape}s"
#
#                 # Annotate the locations on the image and display it
#                 cv.imshow(description, annotate_locations(im, locations))
#                 cv.waitKey(0)
# if __name__=='__main__':
#     main()
