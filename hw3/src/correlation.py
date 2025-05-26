import numpy as np
from utils import split_separable_filter, is_separable


def my_correlation(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    image = np.atleast_3d(image).astype(np.float32)
    img_h, img_w, channels = image.shape
    ker_h, ker_w = kernel.shape
    pad_h, pad_w = ker_h // 2, ker_w // 2

    kernel = kernel.astype(np.float32)

    output = np.zeros_like(image, dtype=np.float32)

    if is_separable(kernel):
        vertical, horizontal = split_separable_filter(kernel)
        temp_out = np.zeros_like(image, dtype=np.float32)

        for c in range(channels):
            for i in range(img_h):
                for j in range(img_w):
                    left, right = max(0, j - pad_w), min(img_w - 1, j + pad_w)
                    region = image[i, left: right + 1, c]

                    if j - pad_w < 0:
                        left_pad = np.full(abs(j - pad_w), image[i, 0, c])
                        region = np.concatenate([left_pad, region])
                    if j + pad_w >= img_w:
                        right_pad = np.full(j + pad_w - img_w + 1, image[i, -1, c])
                        region = np.concatenate([region, right_pad])

                    temp_out[i, j, c] = np.dot(region.ravel(), horizontal.ravel())

            for j in range(img_w):
                for i in range(img_h):
                    top, bottom = max(0, i - pad_h), min(img_h - 1, i + pad_h)
                    region = temp_out[top: bottom + 1, j, c]

                    if i - pad_h < 0:
                        top_pad = np.full(abs(i - pad_h), temp_out[0, j, c])
                        region = np.concatenate([top_pad, region])
                    if i + pad_h >= img_h:
                        bottom_pad = np.full(i + pad_h - img_h + 1, temp_out[-1, j, c])
                        region = np.concatenate([region, bottom_pad])

                    output[i, j, c] = np.dot(region.ravel(), vertical.ravel())

    else:
        for c in range(channels):
            for i in range(img_h):
                for j in range(img_w):
                    i_start, i_end = max(0, i - pad_h), min(img_h, i + pad_h + 1)
                    j_start, j_end = max(0, j - pad_w), min(img_w, j + pad_w + 1)

                    region = image[i_start:i_end, j_start:j_end, c]

                    if i - pad_h < 0:
                        top_pad = np.tile(image[0, j_start:j_end, c], (abs(i - pad_h), 1))
                        region = np.vstack([top_pad, region])
                    if i + pad_h >= img_h:
                        bottom_pad = np.tile(
                            image[-1, j_start:j_end, c], (i + pad_h - img_h + 1, 1)
                        )
                        region = np.vstack([region, bottom_pad])
                    if j - pad_w < 0:
                        left_pad = np.tile(region[:, 0].reshape(-1, 1), (1, abs(j - pad_w)))
                        region = np.hstack([left_pad, region])
                    if j + pad_w >= img_w:
                        right_pad = np.tile(
                            region[:, -1].reshape(-1, 1), (1, j + pad_w - img_w + 1)
                        )
                        region = np.hstack([region, right_pad])

                    output[i, j, c] = np.sum(region * kernel)

    # Apply rounding and clipping to match OpenCV behavior
    output = np.round(output)  # Round after the sum to avoid multiple rounding errors
    if np.issubdtype(image.dtype, np.integer):
        output = np.clip(output, np.iinfo(image.dtype).min, np.iinfo(image.dtype).max)

    output = output.astype(image.dtype)

    return output if image.shape[-1] > 1 else output.squeeze(-1)
