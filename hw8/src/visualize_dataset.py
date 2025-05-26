from pathlib import Path

import numpy as np
import cv2 as cv
import pyvista as pv
from scene3d import wireframe_camera, plot_image_in_space


def extract_camera_params(filename: str, target_img: str):
    with open(filename, "r") as f:
        lines = f.readlines()[1:]  # skip the first header line

    for line in lines:
        parts = line.strip().split()
        if parts[0] == target_img:
            if len(parts) != 1 + 9 + 9 + 3:
                raise ValueError("Line format is invalid. Expected 22 values after image name.")

            k_vals = list(map(float, parts[1:10]))
            r_vals = list(map(float, parts[10:19]))
            t_vals = list(map(float, parts[19:22]))

            K = np.array(k_vals).reshape((3, 3))
            R = np.array(r_vals).reshape((3, 3))
            T = np.array(t_vals).reshape((3, 1))

            return K, R, T

    raise ValueError(f"Image '{target_img}' not found in the file.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("temple_data", type=Path)
    args = parser.parse_args()

    if not args.temple_data.exists():
        raise FileNotFoundError("temple data not found at", args.temple_data)

    pl = pv.Plotter()
    for im_file in args.temple_data.glob("*.png"):
        K, R, T = extract_camera_params(
            filename=args.temple_data / "templeR_par.txt", target_img=im_file.name
        )
        im = cv.imread(str(im_file))
        cv.putText(im, im_file.name, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        plot_image_in_space(im, calibration_matrix=K, r=R, t=T, pl=pl, scale=0.1)
        pl.add_mesh(
            wireframe_camera((R, T), K, image_hw=im.shape[:2], scale=0.1),
            color="black",
        )

    # Temple dataset README speficies this set of coordinates as the bounding box around the
    # actual temple object in the scene.
    pl.add_mesh(pv.Box(bounds=[-0.023121, 0.078626, -0.038009, 0.121636, -0.091940, -0.017395]))
    pl.show()
