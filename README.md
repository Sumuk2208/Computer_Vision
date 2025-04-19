# Getting Started with Image Processing

An introduction to the foundational tools and concepts required for the course on image processing. The primary objective was to familiarize ourselves with the development environment, image handling libraries, and submission protocols.

## Overview

- Setting up a development environment using Python 3.10+
- Performing basic image loading and manipulation using NumPy and OpenCV
- Understanding and applying vectorized operations for efficiency
- Generating processed outputs to match provided examples
- Practicing proper code styling and submission workflow (linting, formatting, zipping)

## Image Processing Tasks

Implemented functions in `src/image_handling.py` to perform:

- **Image Cropping**: Handling both in-bound and out-of-bound cropping coordinates
- **Scaling**: Downscaling images using NumPy
- **Transformations**:
  - Horizontal mirroring
  - 90-degree counterclockwise rotation
- **Color Manipulations**:
  - Swapping red and blue channels
  - Scaling saturation by a factor
  - Converting to grayscale
  - Isolating individual blue, green, and red channels
- **Image Tiling**: Creating a 2x2 grid of the original image using BGR order

