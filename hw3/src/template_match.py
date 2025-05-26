from bounding_boxes import bbox_xywh_iou, BBoxType
import cv2 as cv
import numpy as np
import os

DetectionType = tuple[float, BBoxType]


def score_map_to_bounding_boxes(
    score_map: np.ndarray, template_hw: tuple[int, int], image_scale: float = 1.0
) -> list[DetectionType]:
    """Generates a list of 'Detections' from the score map. A 'Detection' is a tuple containing a
    score and a bounding box in (x, y, w, h) format.

    Args:
        score_map (np.ndarray): The thresholded score map.
        template_hw (tuple[int, int]): Size of the original template as (h, w).
        image_scale (float): The scale of this score map relative to the original image.

    Returns:
        list[DetectionType]: A list of detections, where each detection is a tuple containing the
            bounding box (x, y, w, h) and the corresponding score.
    """
    h, w = template_hw
    detections = []
    # Loop over all non-zero elements in the score map
    for y, x in np.argwhere(score_map > 0):
        score = score_map[y, x]
        bbox = (x / image_scale, y / image_scale, w / image_scale, h / image_scale)
        detections.append((score, bbox))
    return detections


def non_maximal_suppression(
    detections: list[DetectionType], iou_threshold: float
) -> list[DetectionType]:
    """Filter out non-maximal detections. Each detection suppresses its lower-scoring neighbors.
    Bounding boxes are 'neighbors' if their IOU is greater than the given iou_threshold.

    WARNING: naive implementation of this function takes quadratic time in the length of the input
    list. The slow implementation will be considered 'correct' for grading purposes, but you may
    need to be clever about how you debug if it's a bottleneck.

    Args:
        detections (list[DetectionType]): A list of above-threshold detections to consider.
        iou_threshold (float): The IoU threshold defining when bounding boxes are 'neighbors'.

    Returns:
        list[DetectionType]: A subset of the input detections where each is guaranteed to be a local
            maximum score among its neighbors.
    """
    detections = sorted(
        detections, key=lambda x: x[0], reverse=True
    )  # Sort by score (highest first)
    keep = []

    while detections:
        best = detections.pop(0)  # Pick the highest-scoring detection
        keep.append(best)
        detections = [
            d for d in detections if bbox_xywh_iou(best[1], d[1]) < iou_threshold
        ]  # Remove overlapping detections

    return keep


def find_objects_by_template_matching(
    image: np.ndarray,
    template: np.ndarray,
    threshold: float,
    iou_threshold: float,
    scale_factor: float = 0.9,
    levels: int = 1,
) -> list[BBoxType]:
    """Finds objects in an image by template matching.a

    Args:
        image (np.ndarray): The input image.
        template (np.ndarray): The template to match.
        threshold (float): Threshold for template matching scores.
        iou_threshold (float): IoU threshold for non-maximal suppression.
        scale_factor (float): If levels > 1, the downscaling factor for each level. That is, the 0th
            level will use the original image, the 1st level will have width*scale_factor and
            height*scale_factor, and so on.
        levels (int): Number of levels (image sizes) to use when doing multi-scale detection.
            Default of 1 means just use the original image.

    Returns:
        list[BBoxType]: A list of bounding boxes for detected objects in the (x, y, w, h) format.
    """

    # Step 1: Multi-scale template matching. Get a score map for every location and scale.
    score_pyramid = []

    for lvl in range(levels):
        scaled_image = cv.resize(image, None, fx=scale_factor**lvl, fy=scale_factor**lvl)

        result = cv.matchTemplate(scaled_image, template, cv.TM_CCOEFF_NORMED)
        score_pyramid.append(result)

    # Step 2: Normalize all scores to [0, 1] and threshold.
    min_score = min([np.min(score) for score in score_pyramid])
    max_score = max([np.max(score) for score in score_pyramid])
    for lvl in range(levels):
        # USE THIS LINE IF YOU CHOOSE TM_SQDIFF OR TM_SQDIFF_NORMED IN cv.matchTemplate
        normalized_score = (max_score - score_pyramid[lvl]) / (max_score - min_score)
        # USE THIS LINE IF YOU CHOOSE ANY OTHER METHOD
        normalized_score = (score_pyramid[lvl] - min_score) / (max_score - min_score)
        score_pyramid[lvl] = np.where(normalized_score < threshold, 0, normalized_score)

    # Step 3: Convert to list of detections at each scale
    detections = []
    for lvl in range(levels):
        detections.extend(
            score_map_to_bounding_boxes(
                score_pyramid[lvl], template.shape[:2], image_scale=scale_factor**lvl
            )
        )

    # Final step: Apply NMS based on IoU across scales
    detections = non_maximal_suppression(detections, iou_threshold)

    return [bbox for score, bbox in detections]


def visualize_matches(scene: np.ndarray, matches: list[BBoxType]):
    """Highlight bounding boxes in the scene by drawing rectangles over them."""
    count = len(matches)
    for x, y, w, h in matches:
        cv.rectangle(scene, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)

    # Add text in the bottom left corner by using x=10 and y=the height of the scene - 20 pixels
    cv.putText(
        scene,
        f"Found {count} matches",
        (10, scene.shape[0] - 20),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    cv.imshow("Matches", scene)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to the scene", required=True)
    parser.add_argument("--template", help="Path to the template", required=True)
    parser.add_argument("--threshold", help="Minimum score threshold", type=float, required=True)
    parser.add_argument(
        "--nms-threshold",
        help="IOU threshold for non-maximal suppression",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=0.5,
        help="Downscaling factor for each level when doing multi-scale detection.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=1,
        help="Number of levels (image sizes) to use when doing multi-scale detection. "
        "Default is 1 (just the original image)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    if not os.path.exists(args.template):
        raise FileNotFoundError(f"Image not found: {args.template}")
    print(args)
    scene = cv.imread(args.image)
    object = cv.imread(args.template)
    bboxes = find_objects_by_template_matching(
        scene,
        object,
        args.threshold,
        args.nms_threshold,
        args.scale_factor,
        args.levels,
    )

    visualize_matches(scene, bboxes)
