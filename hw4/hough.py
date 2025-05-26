import numpy as np
import cv2
import matplotlib.pyplot as plt

width, height = 500, 500
edges = np.zeros((height, width), dtype=np.uint8)
points = []


def hough_transform():
    accumulator = np.zeros((180, width), dtype=np.uint8)
    thetas = np.deg2rad(np.arange(0, 180))

    for x, y in points:
        for theta_idx, theta in enumerate(thetas):
            rho = int(x * np.cos(theta) + y * np.sin(theta)) + width // 2
            if 0 <= rho < width:
                accumulator[theta_idx, rho] += 1

    plt.imshow(accumulator, cmap='hot', aspect='auto', extent=[-width // 2, width // 2, 180, 0])
    plt.xlabel("Rho")
    plt.ylabel("Theta (degrees)")
    plt.title("Hough Transform")
    plt.colorbar(label="Votes")
    plt.show()


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        edges[y, x] = 255
        cv2.imshow("Select Edge Points", edges)
        hough_transform()


cv2.imshow("Select Edge Points", edges)
cv2.setMouseCallback("Select Edge Points", mouse_callback)
cv2.waitKey(0)
cv2.destroyAllWindows()
