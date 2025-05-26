from typing import NamedTuple

import cv2 as cv
import dlib
import numpy as np


# A bounding box from the cv.CascadeClassifier is a tuple of 4 integers: (x, y, w, h) where (x,
# y) is the top-left corner of the bounding box, and (w, h) are the width and height of the
# bounding box.
class BoundingBox(NamedTuple):
    x: int
    y: int
    w: int
    h: int


def ensure_grayscale(img: np.ndarray) -> np.ndarray:
    img = np.atleast_3d(img)
    if img.shape[2] == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    elif img.shape[2] == 4:
        img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
    elif img.shape[2] == 1:
        pass
    else:
        raise ValueError(f"Image has unexpected number of channels: {img.shape[2]}")
    return img


def opencv_bbox_to_dlib_rect(BoundingBox: BoundingBox) -> dlib.rectangle:
    return dlib.rectangle(
        left=BoundingBox[0],
        top=BoundingBox[1],
        right=BoundingBox[0] + BoundingBox[2],
        bottom=BoundingBox[1] + BoundingBox[3],
    )


def dlib_rect_to_opencv_bbox(rect: dlib.rectangle) -> BoundingBox:
    return BoundingBox(rect.left(), rect.top(), rect.width(), rect.height())
