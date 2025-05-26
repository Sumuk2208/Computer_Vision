from typing import Optional

import cv2 as cv
import numpy as np
import pyvista as pv

from utils import augment


def plot_3d_points(
    points_3d: np.ndarray,
    colors: np.ndarray,
    pl: Optional[pv.Plotter] = None,
) -> pv.Plotter:
    """
    Plot a 3D point cloud using PyVista.

    Args:
        points_3d (np.ndarray): (N, 3) array of 3D coordinates.
        colors (np.ndarray): (N, 3) array of RGB colors.
        pl (pyvista.Plotter): Optional axes to add to.
    """
    # Normalize colors to [0, 1] range
    colors = colors / 255.0 if colors.max() > 1 else colors

    # Create PyVista point cloud
    cloud = pv.PolyData(points_3d)
    cloud["rgb"] = colors

    # Plot
    pl = pl or pv.Plotter()
    pl.add_mesh(cloud, scalars=colors, rgb=True, point_size=2)

    # Set view direction pointing down the optical axis (-z) with the top of the image (-y) in the
    # 'up' direction.
    pl.view_vector([0, 0, -1], viewup=[0, -1, 0])

    return pl


def plot_image_in_space(
    image, calibration_matrix, r, t, scale=1.0, pl: Optional[pv.Plotter] = None
) -> pv.Plotter:
    y_s, x_s = np.indices(image.shape[:2]).astype(np.float32)
    xy_c = cv.undistortPoints(
        np.column_stack([x_s.flatten(), y_s.flatten()]), calibration_matrix, None, None
    )
    xyz_c = augment(xy_c.squeeze(), axis=1) * scale
    xy_w = (xyz_c - t.reshape(1, 3)) @ r
    colors = image.reshape(-1, 3)[:, ::-1]
    return plot_3d_points(xy_w, colors, pl)


def wireframe_camera(camera_pose, calibration_matrix, image_hw, scale=1.0) -> pv.PolyData:
    r, t = camera_pose
    h, w = image_hw[:2]

    # corners of the image
    frame_c = cv.undistortPoints(
        np.array(
            [
                [0, 0],
                [0, h],
                [w, h],
                [w, 0],
            ],
            dtype=np.float32,
        ),
        calibration_matrix,
        None,
        None,
    )
    frame_c = augment(frame_c.squeeze(), axis=1)
    frame_c = np.concatenate([[[0, 0, 0]], frame_c], axis=0) * scale
    frame_world = (frame_c - t.reshape(1, 3)) @ r

    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    lines = []
    for e in edges:
        lines.extend([2, e[0], e[1]])
    return pv.PolyData(frame_world, lines=np.array(lines))
