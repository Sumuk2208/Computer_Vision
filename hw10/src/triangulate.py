from collections import defaultdict
from typing import Optional, Callable

import cv2 as cv
import numpy as np
import numpy.typing as npt
import pyvista as pv
import torch
from matplotlib import pyplot as plt

from utils import augment


class Camera(object):
    def __init__(
        self,
        calibration_matrix: Optional[npt.ArrayLike | torch.Tensor] = None,
        pose_rotation: Optional[npt.ArrayLike | torch.Tensor] = None,
        pose_translation: Optional[npt.ArrayLike | torch.Tensor] = None,
    ):
        if calibration_matrix is None:
            calibration_matrix = torch.eye(3, dtype=torch.float32)
        else:
            calibration_matrix = torch.tensor(calibration_matrix, dtype=torch.float32)
        if pose_rotation is None:
            pose_rotation = torch.eye(3, dtype=torch.float32)
        else:
            pose_rotation = torch.tensor(pose_rotation, dtype=torch.float32)
        if pose_translation is None:
            pose_translation = torch.zeros(3, dtype=torch.float32)
        else:
            pose_translation = torch.tensor(pose_translation, dtype=torch.float32)

        self.calibration_matrix = calibration_matrix
        self.pose_rotation = pose_rotation
        self.pose_translation = pose_translation

    def perpective_projection(self, points3d_world: torch.Tensor) -> torch.Tensor:
        """Given Nx3 points in world coordinates, return Nx2 points in image coordinates. All
        operations are done with torch so that the projection is differentiable.
        """
        # Transform points from world to camera coordinates
        xyz_c = (points3d_world - self.pose_translation[None, :]) @ self.pose_rotation

        # Project points onto image plane (perspective division)
        xy_c = xyz_c[:, :2] / xyz_c[:, 2].unsqueeze(-1)

        # Transform points from camera to sensor coordinates
        xy_s = (self.calibration_matrix @ augment(xy_c, dim=1).T).T

        return xy_s[:, :2]

    def sensor2camera(self, xy_s: torch.Tensor) -> torch.Tensor:
        """Given Nx2 points in sensor coordinates, return Nx3 points in camera coordinates with Z=1."""
        return torch.linalg.solve(self.calibration_matrix, augment(xy_s, dim=1).T).T

    def camera2world(self, xyz_c: torch.Tensor) -> torch.Tensor:
        """Given Nx3 points in camera coordinates, return Nx3 points in world coordinates."""
        return (xyz_c - self.pose_translation[None, :]) @ self.pose_rotation

    @torch.no_grad()
    def pv_wireframe(self, image_shape: tuple[int, ...], scale: float = 1.0) -> pv.PolyData:
        """Utility/debugging function for viewing this camera in space using PyVista."""
        h, w = image_shape[:2]

        # corners of the image in sensor coordinates
        frame_s = torch.tensor([[0, 0], [0, h], [w, h], [w, 0]], dtype=torch.float32)
        # corners of the image in camera coordinates
        frame_c = self.sensor2camera(frame_s)
        # corners of the image in camera coordinates, scaled, with a point at the optical center
        frame_c = torch.cat([torch.zeros(1, 3), frame_c], dim=0)
        # wireframe points in world coordinates
        frame_world = self.camera2world(frame_c * scale)

        edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
        lines = []
        for e in edges:
            lines.extend([2, e[0], e[1]])
        return pv.PolyData(frame_world.numpy(), lines=np.array(lines))

    @torch.no_grad()
    def pv_image_in_space(self, image: np.ndarray, scale: float = 1.0) -> pv.PolyData:
        y_s, x_s = torch.meshgrid(
            [torch.arange(image.shape[0]), torch.arange(image.shape[1])], indexing="ij"
        )
        xyz_c = self.sensor2camera(torch.stack((x_s.flatten(), y_s.flatten()), dim=1))
        xyz_w = self.camera2world(xyz_c * scale)
        colors = image.reshape(-1, 3)[:, ::-1]

        cloud = pv.PolyData(xyz_w.numpy())
        cloud["rgb"] = colors
        return cloud

    @torch.no_grad()
    def render_projected_points(
        self, image: np.ndarray, xyz_w: torch.Tensor, color: tuple[int, int, int], radius: int = 1
    ) -> np.ndarray:
        h, w = image.shape[:2]
        camera_z = (xyz_w @ self.pose_rotation.T + self.pose_translation)[:, 2]
        valid = camera_z > 0
        pts2d = self.perpective_projection(xyz_w[valid, :])
        for x, y in pts2d:
            if 0 <= x < w and 0 <= y < h:
                cv.circle(image, (int(x), int(y)), radius, color, thickness=-1)
        return image


def reprojection_error(p: torch.Tensor, xy: torch.Tensor, cam: Camera):
    """Calculate reprojection error of a set of 3D points for a given set of 2D points and camera.

    Reprojection error is the 2D Euclidean distance between a projected 3D point and its
        corresponding 2D point.

    Args:
        p: Size (N, 3) 3D points in world coordinates.
        xy: Size (N, 2) 2D points in image.
        cam: Camera object.

    Returns:
        Size (N,) reprojection error for each point.
    """
    projected = cam.perpective_projection(p)
    return torch.linalg.norm(projected - xy, dim=1)


def clip_gradient_norm(params: torch.Tensor, max_grad_norm: float) -> None:
    """Clip gradient norm of parameters to a maximum value.

    Given a collection of N parameters, this function computes the L2 norm of the parameters'
    gradients and then rescales the gradients so that the L2 norm is at most max_grad_norm. For
    example, of `params` is a 4x3 tensor and `params.grad` is equal to

        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
        ]

    then we treat this as a collection of 4 vectors in R^3 whose L2 norms are [3.74, 8.77, 13.93, 19.10].
    Let's say max_grad is set to 10.0. Then we would rescale the gradients only for the 3rd and 4th
    parameters, so that their L2 norms are 10.0. The first two parameters would be unchanged. The
    params.grad attribute will then be modified in-place to be

        [
            [1, 2, 3],
            [4, 5, 6],
            [5.03, 5.74, 6.46],
            [5.23, 5.76, 6.28]
        ]

    This function modifies `params.grad` in-place and has no return value.
    """
    if params.grad is None:
        return

    grad_norm = torch.linalg.norm(params.grad, dim=1)
    scale = max_grad_norm / (grad_norm + 1e-8)
    scale = torch.clamp(scale, max=1.0)
    params.grad *= scale.unsqueeze(1)
    # Your code here. Defaulting to a no-op so you can run it. ≈3 Lines in the AnswerKey.


def triangulate_by_gradient_descent(
    init_pts3d: torch.Tensor,
    pts0: torch.Tensor,
    cam0: Camera,
    pts1: torch.Tensor,
    cam1: Camera,
    num_iters: int = 1000,
    step_size: float = 0.1,
    max_grad_norm: float = 1.0,
    callback: Optional[Callable[[int, torch.Tensor], None]] = None,
):
    """Triangulate 3D points from 2D correspondences using gradient descent."""
    pts3d = init_pts3d.clone()
    pts3d.requires_grad = True

    opt = torch.optim.SGD([pts3d], lr=step_size)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.99)

    history = defaultdict(list)
    for itr in range(num_iters):
        # Zero gradients at start of iteration
        opt.zero_grad()

        err0 = reprojection_error(pts3d, pts0, cam0)
        err1 = reprojection_error(pts3d, pts1, cam1)

        # Sum all errors for the loss
        loss = err0.sum() + err1.sum()
        loss.backward()

        # Clip gradients before optimization step
        clip_gradient_norm(pts3d, max_grad_norm)

        # Update parameters
        opt.step()
        # Update learning rate after parameter update
        scheduler.step()

        history["err0"].append(err0.mean().item())
        history["err1"].append(err1.mean().item())

        if callback is not None:
            callback(itr, pts3d.detach())

    return pts3d.detach(), history


def main(n_points: int, image_size: int = 200, animate: bool = False):
    # True 3d points: a bunch of random 3D points on a half-sphere
    points3d = torch.randn(n_points, 3)
    points3d = points3d / torch.linalg.norm(points3d, dim=-1, keepdim=True)
    points3d += torch.tensor([0.0, 0.0, 5.0])

    calibration = torch.tensor(
        [
            [image_size, 0, image_size / 2],
            [0, image_size, image_size / 2],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )

    camera0 = Camera(
        calibration_matrix=calibration,
        pose_rotation=cv.Rodrigues(np.array([0, +np.pi / 16, 0]))[0],
        pose_translation=[-1, 0, 0],
    )
    camera1 = Camera(
        calibration_matrix=calibration,
        pose_rotation=cv.Rodrigues(np.array([0, -np.pi / 16, 0]))[0],
        pose_translation=[+1, 0, 0],
    )

    ####################################
    # Set up visualization / animation #
    ####################################

    pl, xy0, xy1, recovered_points, plot_callback = _set_up_pyvista_scene(
        camera0, camera1, points3d, animate, image_size
    )

    #####################
    # Run Triangulation #
    #####################

    initial_guess = torch.zeros_like(points3d) + torch.tensor([0.0, 0.0, 5.0])
    recovered_pts3d, history = triangulate_by_gradient_descent(
        initial_guess,
        xy0,
        camera0,
        xy1,
        camera1,
        num_iters=301,
        callback=plot_callback,
        max_grad_norm=1.0,
    )
    pl.show()

    ####################
    # Plot loss curves #
    ####################

    plt.figure()
    plt.plot(history["err0"], label="Camera 0 reprojection errors")
    plt.plot(history["err1"], label="Camera 1 reprojection errors")
    plt.xlabel("Iteration")
    plt.ylabel("Reprojection error")
    plt.yscale("log")
    plt.legend()
    plt.show()


def _set_up_pyvista_scene(
    camera0: Camera,
    camera1: Camera,
    points3d: torch.Tensor,
    animate: bool = False,
    image_size: int = 200,
):
    xy0 = camera0.perpective_projection(points3d)
    xy1 = camera1.perpective_projection(points3d)
    pl = pv.Plotter()
    recovered_points = pv.PolyData(torch.zeros_like(points3d).numpy(), lines=None)
    pl.add_mesh(recovered_points, color="red", point_size=8, render_points_as_spheres=True)
    pl.add_mesh(
        pv.PolyData(points3d.numpy()), color="black", point_size=5, render_points_as_spheres=True
    )
    pl.add_mesh(camera0.pv_wireframe(image_shape=(image_size, image_size), scale=0.5), color="blue")
    pl.add_mesh(
        camera0.pv_image_in_space(
            camera0.render_projected_points(
                np.zeros((image_size, image_size, 3), dtype=np.uint8),
                points3d,
                (255, 0, 0),
                radius=1,
            ),
            scale=0.5,
        ),
        rgb=True,
        point_size=2,
    )
    pl.add_mesh(
        camera1.pv_wireframe(image_shape=(image_size, image_size), scale=0.5), color="green"
    )
    pl.add_mesh(
        camera1.pv_image_in_space(
            camera1.render_projected_points(
                np.zeros((image_size, image_size, 3), dtype=np.uint8),
                points3d,
                (0, 255, 0),
                radius=1,
            ),
            scale=0.5,
        ),
        rgb=True,
        point_size=2,
    )
    pl.view_vector([1, -1, -1], viewup=[0, -1, 0])
    pl.add_axes()
    if animate:
        pl.show(interactive_update=True, window_size=(1200, 900))

    def plot_callback(itr, pts3d):
        # Update data in animation_holder
        recovered_points.points = pts3d.detach().numpy()

        # Update iteration text
        pl.add_text(f"Iteration {itr}", position="upper_left", font_size=16, name="iter_label")

        if animate:
            # Framerate sleep
            pl.update(stime=30)

            if itr in [0, 100, 200, 300]:
                pl.save_graphic("examples/triangulation_" + str(itr) + ".eps")

    return pl, xy0, xy1, recovered_points, plot_callback


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Triangulate 3D points from 2D correspondences.")
    parser.add_argument("--n-points", type=int, default=1000, help="Number of points to use.")
    parser.add_argument("--animate", action="store_true", help="Enable animation.")
    args = parser.parse_args()

    main(args.n_points, animate=args.animate)
