import numpy as np

cfg = {
    "lines": {
        "house.jpg": {
            "horizontal_lines": {
                "canny_blur": 5,
                "canny_thresholds": (50, 150),
                "min_angle": np.pi / 2 - 0.1,
                "max_angle": np.pi / 2 + 0.1,
                "angle_spacing": np.pi / 360,
                "offset_spacing": 2.0,
                "accumulator_threshold": 0.57,
                "nms_angle_range": np.pi / 45,
                "nms_offset_range": 15.0,
            },
        },
    },
    "circles": {
        "coins.jpg": {
            "pennies": {
                "canny_blur": 9,
                "canny_thresholds": (60, 170),
                "accumulator_threshold": 0.7,
                "nms_radius": 18,
                "radius": 60.0,
                "soft_vote_sigma": 1.2,
            },
            "nickels": {
                "canny_blur": 7,
                "canny_thresholds": (63, 150),
                "accumulator_threshold": 0.76,
                "nms_radius": 28,
                "radius": 63.0,
                "soft_vote_sigma": 1.2,
            },
            "dimes": {
                "canny_blur": 9,  # A good default, but feel free to increase if noise is high
                "canny_thresholds": (80, 160),  # Slightly higher for cleaner edge detection
                "accumulator_threshold": 0.85,  # Slightly more selective to reduce false positives
                "nms_radius": 18,  # Fine-tuned for well-spaced coins
                "radius": 52.0,  # Keep as is, typical dime size
                "soft_vote_sigma": 1.6,
            },
            "quarters": {
                "canny_blur": 5,
                "canny_thresholds": (80, 170),  # Stronger edges needed for larger coins
                "accumulator_threshold": 0.76,  # Slightly more lenient for larger coins
                "nms_radius": 18,  # Fine-tuned for typical coin separation
                "radius": 70.0,  # Standard quarter size
                "soft_vote_sigma": 1.3,  # Moderate smoothing for clearer edges
            },
        },
    },
}
