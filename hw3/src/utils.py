import numpy as np
import warnings


def split_separable_filter(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given a separable filter, return the two 1D filters that make it up. If the given filter
    is not separable, this function returns the best approximation of it and displays a warning.

    Returns: (vertical_part, horizontal_part) where each is a 1D filter. The vertical part is a
    column vector and the horizontal part is a row vector.
    """
    if h.ndim != 2:
        raise ValueError("Filter must be 2D.")
    u, s, vh = np.linalg.svd(h)
    sum_of_singular_values = np.sum(s[1:])
    if sum_of_singular_values >= 1e-6:
        warnings.warn(
            "Filter is not separable within floating point precision. Using best approximation.",
            stacklevel=2,
        )
    vertical_part = u[:, :1] * np.sqrt(s[:1])
    horizontal_part = vh[:1, :] * np.sqrt(s[:1])
    return vertical_part, horizontal_part


def is_separable(h: np.ndarray, tolerance: float = 1e-6) -> bool:
    """Return whether a given 2D filter is separable within a given tolerance."""
    if h.ndim != 2:
        raise ValueError("Filter must be 2D (1D filters should be given as Mx1 or 1xM explicitly).")
    if h.shape[0] == 1 or h.shape[1] == 1:
        # There is no reason to separate 1D filters
        return False
    u, s, vh = np.linalg.svd(h)
    sum_of_singular_values = np.sum(s[1:])
    return sum_of_singular_values < tolerance
