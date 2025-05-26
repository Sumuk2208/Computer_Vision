from typing import Union

Number = Union[int, float]
BBoxType = tuple[Number, Number, Number, Number]


def bbox_xyxy_to_xywh(bbox: BBoxType) -> BBoxType:
    """Converts a bounding box from (x1, y1, x2, y2) format to (x, y, w, h) format."""
    x1, y1, x2, y2 = bbox
    return x1, y1, x2 - x1, y2 - y1


def bbox_xywh_to_xyxy(bbox: BBoxType) -> BBoxType:
    """Converts a bounding box from (x, y, w, h) format to (x1, y1, x2, y2) format."""
    x, y, w, h = bbox
    return x, y, x + w, y + h


def bbox_xywh_iou(bbox_a: BBoxType, bbox_b: BBoxType) -> float:
    """Calculates the Intersection over Union (IoU) of two bounding boxes. The bounding boxes are
    given in (x, y, w, h) format.
    """
    ax1, ay1, ax2, ay2 = bbox_xywh_to_xyxy(bbox_a)
    bx1, by1, bx2, by2 = bbox_xywh_to_xyxy(bbox_b)

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a, area_b = (ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1)

    return inter_area / (area_a + area_b - inter_area) if area_a + area_b - inter_area > 0 else 0.0
