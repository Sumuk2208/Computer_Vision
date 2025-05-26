import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

# You must adjust these values as part of the assignment. You need to figure out what sensible low
# and high thresholds are for this autofocus algorithm to work well.
_DEFAULT_FREQ_BAND = (0.05, 0.2)


def get_mask(sz: tuple[int, int], loc_xy: tuple[float, float], radius: float) -> np.ndarray:
    """Create a Gaussian mask for an image of the given size, centered at the given location,
    with the given radius.

    :param sz: The (width, height) of the image.
    :param loc_xy: The location of the center of the circular mask.
    :param radius: The radius of the circular mask.
    """
    # Create 2D arrays of x and y coordinates
    x, y = np.meshgrid(np.arange(sz[0]), np.arange(sz[1]))  # Note: sz[1] = width, sz[0] = height
    (
        center_x,
        center_y,
    ) = loc_xy
    stdev = radius / 3  # Standard deviation of the Gaussian

    # Compute the squared distance from each pixel to the center (x, y) in image coordinates
    dist = (x - center_x) ** 2 + (y - center_y) ** 2

    # Apply the Gaussian function
    mask = np.exp(-0.5 * (dist / stdev**2))

    # Apply a circular mask (values outside the circle will be 0)
    mask[dist > radius**2] = 0

    return mask


def calculate_power_spectrum(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the power spectrum of the given image, i.e. the power (squared magnitude) and
    spatial frequency (distance from the origin in the frequency domain) of each point in the
    frequency domain.

    :param image: The image to calculate the power spectrum of.
    :return: A tuple of (spatial frequency, power), each a 2D array.
    """
    # Calculate the 2D Fourier transform of the image
    fourier = np.fft.fft2(image)
    power = np.abs(fourier) ** 2

    # Calculate frequencies corresponding to each point in the frequency domain
    freq_x = np.fft.fftfreq(image.shape[1])
    freq_y = np.fft.fftfreq(image.shape[0])
    freq = np.sqrt(freq_x[np.newaxis, :] ** 2 + freq_y[:, np.newaxis] ** 2)

    return freq, power


def average_power_in_band(image: np.ndarray, band: tuple[float, float]) -> float:
    """Calculate the average power in the given frequency band of the power spectrum of the given
    image. The frequency band should be given in the form (f_min, f_max), where each f is freqency
    measured in cycles per pixel (min = 0 is the DC component, max = 0.5 is the Nyquist frequency).

    :param image: The image to calculate the power spectrum of.
    :param band: The frequency band to calculate the average power of.
    :return: The average power in the given frequency band.
    """
    freq, power = calculate_power_spectrum(image)

    # Create a mask to select only the desired frequency range
    mask = (freq >= band[0]) & (freq <= band[1])

    # Compute the average power in the selected frequency band
    avg_power = np.mean(power[mask])

    return avg_power


def select_best_focus(
    grays: list[np.ndarray],
    loc_xy: tuple[float, float],
    radius: float,
    freq_band: tuple[float, float],
) -> int:
    """Select the best-focused image from the given list of images. The best-focused image is the
    one with the highest average power in the given frequency band within the circular region
    centered at loc_xy with the given radius.

    :param grays: A list of grayscale images to choose from. All must have the same dimensions.
    :param loc_xy: The location of the center of the circular region.
    :param radius: The radius of the circular region.
    :param freq_band: The frequency band to calculate the average power in.
    :return: The index of the best-focused image.
    """

    best_focus_idx = -1
    highest_avg_power = -1

    for idx, gray in enumerate(grays):
        # Create the mask for the region of interest
        (x, y) = gray.shape
        mask = get_mask((y, x), loc_xy, radius)

        # Apply the mask to the grayscale image to focus on the desired region
        masked_image = gray * mask

        # Calculate the average power in the given frequency band for the masked image
        avg_power = average_power_in_band(masked_image, freq_band)

        # Update the best focus if this image has higher power in the desired band
        if avg_power > highest_avg_power:
            highest_avg_power = avg_power
            best_focus_idx = idx

    return best_focus_idx


def plot_power_spectrum(
    image: np.ndarray,
    bins: int = 200,
    plot_points: bool = False,
    **kwargs,
) -> None:
    """Plot the power spectral density (average power vs frequency) of the given image. Uses a
    log-log scale.

    :param image: The image to plot the power spectrum of.
    :param bins: The number of frequency bins to use.
    :param plot_points: Whether to plot the individual points in the power spectrum. If False, plots
        the average instead.
    :param kwargs: Additional keyword arguments to pass to the plot function.
    """
    # Calculate the power spectrum of the image
    freq_2d, power_2d = calculate_power_spectrum(image)

    if plot_points:
        plt.plot(freq_2d, power_2d, marker=".", linestyle="none", **kwargs)
    else:
        # Calculate spectral density by averaging power in each frequency bin
        freq_bins = np.linspace(0, np.max(freq_2d), bins + 1)
        power_avg = np.zeros(freq_bins.shape)
        for i in range(len(freq_bins) - 1):
            mask = (freq_2d >= freq_bins[i]) & (freq_2d < freq_bins[i + 1])
            power_avg[i] = np.mean(power_2d[mask])

        # Plot the power spectrum
        plt.plot(freq_bins, power_avg, **kwargs)

    plt.xscale("log")
    plt.yscale("log")


def interactive_autofocus(images: list[np.ndarray], freq_band: tuple[float, float]):
    """Interactively focus the given images using the power spectral density. The images should
    be a stack of images taken from a single vantage point while moving the focus of a lens.
    Allows the user to specify a radius with a slider and click on an (x, y) point to focus on it.

    :param images: A list of images at different focus points.
    """

    h, w = images[0].shape[:2]
    x, y, radius = w // 2, h // 2, min(h, w) // 8

    grays = [cv.cvtColor(im, cv.COLOR_BGR2GRAY) for im in images]

    def update():
        nonlocal x, y, radius
        t = time.time()
        idx = select_best_focus(grays, (x, y), radius, freq_band)
        # for fmin, fmax in [(0.05, 0.2), (0.1, 0.3), (0.2, 0.4)]:
        #     idx2=select_best_focus(grays,(2508,965),699,(fmin,fmax))
        #     print(f"fsdfds{idx2}")
        test_cases = [
            ((2112, 1477), 250),
            ((2508, 965), 699),
            ((2508, 965), 134),
            ((1033, 1909), 185),
        ]
        for i, j in test_cases:
            print(select_best_focus(grays, (i[0], i[1]), j, freq_band))
        print(f"Selected focus {idx + 1} in {time.time() - t:.3f} seconds.")
        display = images[idx].copy()
        cv.circle(display, (x, y), radius, (0, 255, 0), 1, cv.LINE_AA)
        cv.putText(
            display,
            f"Focus: {idx + 1}",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv.LINE_AA,
        )

        cv.imshow(window_name, display)

    def on_mouse(event, x_, y_, flags, param):
        nonlocal x, y
        if event == cv.EVENT_LBUTTONUP:
            x, y = x_, y_

            update()

    def on_trackbar(x):
        nonlocal radius
        radius = x
        update()

    window_name = "Autofocus"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.setMouseCallback(window_name, on_mouse)
    cv.createTrackbar("Radius", window_name, radius, min(w // 2, h // 2), on_trackbar)

    while cv.waitKey(1) != ord("q"):
        pass


def test_get_mask():
    # Test 1: sz = (200, 100), loc_xy = (150, 20), radius = 15
    sz = (200, 100)
    loc_xy = (150, 20)
    radius = 15
    mask = get_mask(sz, loc_xy, radius)
    cv.imshow("s", mask)
    cv.waitKey(0)
    print(f"Test 1 mask shape: {mask.shape}")

    # Test 2: sz = (200, 100), loc_xy = (50, 50), radius = 30
    sz = (200, 100)
    loc_xy = (50, 50)
    radius = 30
    mask = get_mask(sz, loc_xy, radius)
    cv.imshow("s", mask)
    cv.waitKey(0)
    print(f"Test 2 mask shape: {mask.shape}")

    # Test 3: sz = (400, 500), loc_xy = (200, 250), radius = 100
    sz = (400, 500)
    loc_xy = (200, 250)
    radius = 100
    mask = get_mask(sz, loc_xy, radius)
    cv.imshow("s", mask)
    cv.waitKey(0)
    print(f"Test 3 mask shape: {mask.shape}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("image_files", nargs="+", help="The images to focus.", type=Path)
    parser.add_argument(
        "--fmin",
        type=float,
        default=_DEFAULT_FREQ_BAND[0],
        help="The minimum of the frequency band (units of cycles per pixel).",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=_DEFAULT_FREQ_BAND[1],
        help="The maximum of the frequency band (units of cycles per pixel).",
    )
    args = parser.parse_args()

    if args.fmin < 0 or args.fmin >= args.fmax or args.fmax > 0.5:
        raise ValueError("Invalid frequency band (must be 0 <= fmin < fmax <= 0.5).")

    images = []
    for im in args.image_files:
        if not im.exists():
            raise FileNotFoundError(f"Image file {im} does not exist.")
        images.append(cv.imread(str(im)))
        if images[-1] is None:
            raise ValueError(f"Could not read image file {im}.")
    interactive_autofocus(images, (args.fmin, args.fmax))
    test_get_mask()
