from pathlib import Path

import cv2 as cv
import numpy as np

from utils import BoundingBox, ensure_grayscale


class FaceDetector:
    def __init__(self, xml_model_path: Path):
        if not xml_model_path.exists():
            raise FileNotFoundError(f"Model file not found: {xml_model_path}")

        self.model = cv.CascadeClassifier()
        self.model.load(str(xml_model_path))

    def detect_faces(self, img: np.ndarray) -> list[BoundingBox]:
        # Ensure the image is grayscale
        gray = ensure_grayscale(img)

        # Improve contrast for better detection
        gray = cv.equalizeHist(gray)

        # Detect faces using the cascade model
        detections = self.model.detectMultiScale(gray, minNeighbors=6)

        # Convert the detections to a list of BoundingBox objects
        return [BoundingBox(x, y, w, h) for (x, y, w, h) in detections]
