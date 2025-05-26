import argparse
import time
from pathlib import Path

import cv2 as cv

from sunglasses_filter import SunglassesFilter


def run_single_image(img_path: Path, model: SunglassesFilter, output_path: Path):
    img = cv.imread(str(img_path))
    img_out = model.detect_and_draw(img)
    cv.imwrite(str(output_path), img_out)


def run_live(model: SunglassesFilter, camera: int = 0, resize: float = 0.5):
    cam = cv.VideoCapture(camera)
    last_frame_time = time.time()
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        frame = cv.resize(frame, dsize=None, fx=resize, fy=resize)
        frame = model.detect_and_draw(frame)

        # Calculate FPS and add as an overlay on the frame
        current_time = time.time()
        fps, last_frame_time = 1 / (current_time - last_frame_time), current_time
        cv.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv.LINE_AA,
        )

        cv.imshow("SunglassesFilter (press Q to quit)", frame)
        if cv.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        help="Path to input image or 'live' to use camera 0 live.",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save output image in when not running in 'live' mode",
        default=Path("output_images"),
    )

    parser.add_argument(
        "--detector-xml-file",
        type=Path,
        help="Path to model XML file for cv.CascadeClassifier.",
        default="haarcascade_frontalface_alt.xml",
    )

    parser.add_argument(
        "--dlib-model-file",
        type=str,
        help="Path to keypoint detector configuration file for DLIB face detector",
        default="shape_predictor_68_face_landmarks.dat",
    )

    parser.add_argument(
        "--sunglasses-image-path",
        type=Path,
        default=Path("../images/sunglasses.png"),
        help="Path to png file with sunglasses.",
    )

    args = parser.parse_args()

    model = SunglassesFilter(
        args.sunglasses_image_path, args.detector_xml_file, args.dlib_model_file
    )

    if args.image == "live":
        run_live(model)
    else:
        args.image = Path(args.image)
        if not args.image.exists():
            raise FileNotFoundError(f"Image file not found: {args.image}")
        if not args.output_dir.exists():
            args.output_dir.mkdir(parents=True)
        run_single_image(
            args.image, model, args.output_dir / f"{args.image.stem}_sunglasses_out.jpg"
        )
