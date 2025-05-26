from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np
import pyvista as pv

from scene3d import wireframe_camera, plot_image_in_space


def match_points(frame1: cv.Mat, frame2: cv.Mat, dist_threshold: Optional[float] = None):
    """
    Match points between two images using SIFT feature matching.

    Args:
        frame1: the first image
        frame2: the second image
        dist_threshold: if set, only matches below this distance are considered valid

    Returns:
        pts1: the 2D points in the first image
        pts2: the 2D points in the second image
    """
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    matcher = cv.BFMatcher(normType=cv.NORM_L2, crossCheck=True)

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    matches = matcher.match(des1, des2)

    if dist_threshold is not None:
        matches = [m for m in matches if m.distance < dist_threshold]

    pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.array([kp2[m.trainIdx].pt for m in matches])

    return pts1, pts2


def estimate_essential_matrix(pts1: np.ndarray, pts2: np.ndarray, calibration_matrix: np.ndarray):
    """Estimate the essential matrix and filter matches.

    Args:
        pts1, pts2: Matched points (image coordinates)
        calibration_matrix: Camera calibration matrix

    Returns:
        essential_mat: The essential matrix
        pts1_filtered, pts2_filtered: Filtered point matches (inliers only)
    """
    essential_mat, mask = cv.findEssentialMat(
        pts1, pts2, calibration_matrix, method=cv.RANSAC, prob=0.999, threshold=1.0
    )

    # Filter points to keep only inliers
    pts1_filtered = pts1[mask.ravel() == 1]
    pts2_filtered = pts2[mask.ravel() == 1]

    return essential_mat, pts1_filtered, pts2_filtered


def recover_pose_from_essential(
    essential_mat: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, scale=1.0
):
    """Recover camera pose from essential matrix and set correct scale.

    Args:
        essential_mat: Essential matrix
        pts1, pts2: Matched points (normalized coordinates)
        scale: Known distance between cameras

    Returns:
        R: Rotation matrix
        t: Translation vector (with correct scale)
        pts1, pts2: Filtered point matches (inliers only)
    """
    _, R, t, mask = cv.recoverPose(essential_mat, pts1, pts2)

    # Scale the translation vector to match the known distance between cameras
    t = t.ravel() * scale

    # Filter points to keep only inliers
    pts1 = pts1[mask.ravel() > 0]
    pts2 = pts2[mask.ravel() > 0]

    return R, t, pts1, pts2


def triangulate_points(pts1: np.ndarray, pts2: np.ndarray, R: np.ndarray, t: np.ndarray):
    """Triangulate 3D points from 2D correspondences and camera poses.

    Args:
        pts1, pts2: Matched points (normalized coordinates)
        R: Rotation matrix between cameras
        t: Translation vector between cameras

    Returns:
        pts3d: Triangulated 3D points as a Nx3 array
    """
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # First camera projection matrix
    P2 = np.hstack((R, t.reshape(-1, 1)))  # Second camera projection matrix

    # Convert to homogeneous coordinates
    pts4d_homogeneous = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)

    # Convert to 3D by dividing by the last coordinate
    pts3d = (pts4d_homogeneous[:3] / pts4d_homogeneous[3]).T
    return pts3d


def infer_camera_pose_and_triangulate(
    image1, image2, calibration_matrix, known_distance_between_cameras=1.0
):
    """Infer the camera extrinsics (R, t) for camera2 relative to camera1 and triangulate 3D points.

    This function implements the following pipeline:
    1. Find matching features between the images
    2. Estimate the essential matrix, which encodes the relative camera pose
    3. Decompose the essential matrix to recover the rotation and translation
    4. Triangulate the 3D points using the camera poses

    Args:
        image1: the first image
        image2: the second image
        calibration_matrix: the camera intrinsics matrix K (3x3)
        known_distance_between_cameras: the known distance between cameras (for proper scaling)

    Returns:
        R: the rotation matrix, size (3, 3)
        t: the translation vector, size (3,)
        pts3d: triangulated 3D points
    """
    # Step 1: Match features between images
    pts1, pts2 = match_points(image1, image2)

    # Step 2: Estimate essential matrix and filter matches
    essential_mat, pts1, pts2 = estimate_essential_matrix(pts1, pts2, calibration_matrix)

    # Step 2.5: Normalize points because cv.recoverPose and cv.triangulatePoints expect
    # normalized points as inputs.
    pts1 = cv.undistortPoints(pts1, calibration_matrix, None).squeeze()
    pts2 = cv.undistortPoints(pts2, calibration_matrix, None).squeeze()

    # Step 3: Recover pose (R,t) from essential matrix
    R, t, pts1, pts2 = recover_pose_from_essential(
        essential_mat, pts1, pts2, known_distance_between_cameras
    )

    # Step 4: Triangulate 3D points using the recovered pose
    pts3d = triangulate_points(pts1, pts2, R, t)

    return R, t, pts3d


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image1", type=Path)
    parser.add_argument("image2", type=Path)
    parser.add_argument("--true-displacement", type=float, default=0.03)
    args = parser.parse_args()

    if not args.image1.exists():
        raise FileNotFoundError(args.image1)

    if not args.image2.exists():
        raise FileNotFoundError(args.image2)

    img1 = cv.imread(str(args.image1))
    img2 = cv.imread(str(args.image2))

    # Calibration matrix hard-coded, comes from the metadata files included in the Middlebury temple
    # dataset. Hard-coding it here because it's the same for all images in the dataset.
    calibration_matrix = np.array(
        [
            [1520.4, 0.0, 302.320],
            [0.0, 1525.9, 246.870],
            [0.0, 0.0, 1.0],
        ]
    )

    r, t, pts3d = infer_camera_pose_and_triangulate(
        img1, img2, calibration_matrix, known_distance_between_cameras=args.true_displacement
    )

    # Visualize results

    pl = pv.Plotter()
    pl.add_mesh(pv.PolyData(pts3d), color="red", point_size=5)
    pl.add_mesh(
        wireframe_camera(
            camera_pose=(np.eye(3), np.zeros(3)),
            calibration_matrix=calibration_matrix,
            image_hw=img1.shape,
            scale=0.1,
        ),
        color="black",
    )
    plot_image_in_space(img1, calibration_matrix, np.eye(3), np.zeros(3), scale=0.1, pl=pl)
    pl.add_mesh(
        wireframe_camera(
            camera_pose=(r, t),
            calibration_matrix=calibration_matrix,
            image_hw=img2.shape,
            scale=0.1,
        ),
        color="red",
    )
    plot_image_in_space(img1, calibration_matrix, r, t, scale=0.1, pl=pl)
    pl.add_mesh(
        wireframe_camera(
            camera_pose=(r, t),
            calibration_matrix=calibration_matrix,
            image_hw=img2.shape,
            scale=0.1,
        ),
        color="green",
    )
    pl.show()
