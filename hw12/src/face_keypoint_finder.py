import bz2
import os
import shutil

import dlib
import numpy as np
import requests

from utils import BoundingBox, ensure_grayscale


class FaceKeypointFinder:
    DLIB_PREDICTOR_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    DEFAULT_KEYPOINT_MODEL_PATH = "shape_predictor_68_face_landmarks.dat"

    def __init__(self, keypoint_model_path: str = "download"):
        self.face_detect = dlib.get_frontal_face_detector()

        if keypoint_model_path == "download":
            if not os.path.exists(FaceKeypointFinder.DEFAULT_KEYPOINT_MODEL_PATH):
                self.download_dlib_predictor()
            self.dlib_model = dlib.shape_predictor(FaceKeypointFinder.DEFAULT_KEYPOINT_MODEL_PATH)
        else:
            self.dlib_model = dlib.shape_predictor(keypoint_model_path)

    @staticmethod
    def download_dlib_predictor():
        # Download the Dlib predictor model
        response = requests.get(FaceKeypointFinder.DLIB_PREDICTOR_URL)
        if response.status_code != 200:
            raise ValueError("Failed to download the Dlib predictor model")
        with open("shape_predictor_68_face_landmarks.dat.bz2", "wb") as f:
            f.write(response.content)
        with open(FaceKeypointFinder.DEFAULT_KEYPOINT_MODEL_PATH, "wb") as f:
            with bz2.open("shape_predictor_68_face_landmarks.dat.bz2", "rb") as bz2_f:
                shutil.copyfileobj(bz2_f, f)

    def detect_keypoints(self, img: np.ndarray, rect: BoundingBox) -> list[tuple[int, ...]]:
        # Ensure grayscale (dlib expects it)
        gray = ensure_grayscale(img)

        # Convert BoundingBox to dlib.rectangle
        dlib_rect = dlib.rectangle(rect.x, rect.y, rect.x + rect.w, rect.y + rect.h)

        # Detect landmarks
        shape = self.dlib_model(gray, dlib_rect)

        # Convert to list of (x, y) tuples
        return [(part.x, part.y) for part in shape.parts()]
