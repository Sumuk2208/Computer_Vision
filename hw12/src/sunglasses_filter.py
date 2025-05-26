from pathlib import Path

import cv2 as cv
import numpy as np

from face_detector import FaceDetector
from face_keypoint_finder import FaceKeypointFinder


class SunglassesFilter:
    def __init__(
        self,
        sunglasses_image_path: str,
        xml_model_path: Path,
        keypoint_model_path: str = "download",
    ):
        self.face_detector = FaceDetector(xml_model_path=xml_model_path)
        self.face_keypoint_finder = FaceKeypointFinder(keypoint_model_path=keypoint_model_path)

        # Read the sunglasses image with alpha channel for transparency
        self.sunglasses = cv.imread(sunglasses_image_path, cv.IMREAD_UNCHANGED)

    def detect_and_draw(self, img: np.ndarray) -> np.ndarray:
        # Make a copy so we don't modify the original image
        img_out = img.copy()

        # Step 1: Detect faces
        faces = self.face_detector.detect_faces(img_out)

        for face in faces:
            # Step 2: Find keypoints
            keypoints = self.face_keypoint_finder.detect_keypoints(img_out, face)

            if len(keypoints) != 68:
                continue  # Unexpected, but just in case

            # Step 3: Identify key landmarks
            left_eye = keypoints[36]  # Left eye corner
            right_eye = keypoints[45]  # Right eye corner

            # Calculate center point between eyes
            eyes_center = (
                (left_eye[0] + right_eye[0]) // 2,
                (left_eye[1] + right_eye[1]) // 2,
            )

            # Calculate sunglasses width based on distance between eyes
            eye_width = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
            sunglass_width = int(eye_width * 2.0)  # a bit wider than just the eyes

            # Resize sunglasses to match face width
            scale_factor = sunglass_width / self.sunglasses.shape[1]
            sunglass_height = int(self.sunglasses.shape[0] * scale_factor)
            resized_sunglasses = cv.resize(
                self.sunglasses, (sunglass_width, sunglass_height), interpolation=cv.INTER_AREA
            )

            # Step 4: Determine top-left corner to place sunglasses
            x1 = int(eyes_center[0] - sunglass_width / 2)
            y1 = int(eyes_center[1] - sunglass_height / 2)

            # Ensure the placement is inside the image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_out.shape[1], x1 + resized_sunglasses.shape[1])
            y2 = min(img_out.shape[0], y1 + resized_sunglasses.shape[0])

            # Adjust sunglasses if going out of bounds
            resized_sunglasses = resized_sunglasses[: y2 - y1, : x2 - x1]

            # Step 5: Composite sunglasses onto the face
            sunglass_rgb = resized_sunglasses[:, :, :3]
            sunglass_alpha = resized_sunglasses[:, :, 3] / 255.0  # Normalize alpha mask

            # Region of interest on the output image
            roi = img_out[y1:y2, x1:x2]

            # Blend sunglasses into the ROI
            for c in range(3):  # For R, G, B channels
                roi[:, :, c] = (
                    sunglass_alpha * sunglass_rgb[:, :, c] + (1 - sunglass_alpha) * roi[:, :, c]
                )

            # Write the blended roi back into the output image
            img_out[y1:y2, x1:x2] = roi

        return img_out
