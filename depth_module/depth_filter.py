from __future__ import annotations

import cv2
import numpy as np


def inverse_depth_discontinuity_mask(
    depth: np.ndarray,
    *,
    percentile: float,
    fixed_threshold: float | None,
    dilation: int,
    eps: float = 1e-6,
) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > eps)
    if not valid.any():
        return np.zeros(depth.shape, dtype=bool)

    inverse_depth = np.zeros(depth.shape, dtype=np.float32)
    inverse_depth[valid] = 1.0 / np.maximum(depth[valid], eps)
    gx = cv2.Sobel(inverse_depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(inverse_depth, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    candidates = gradient[valid & np.isfinite(gradient) & (gradient > 0)]
    if candidates.size == 0:
        return np.zeros(depth.shape, dtype=bool)

    threshold = fixed_threshold if fixed_threshold is not None else float(np.percentile(candidates, percentile))
    if threshold <= 0:
        return np.zeros(depth.shape, dtype=bool)

    mask = valid & (gradient >= threshold)
    if dilation > 1 and mask.any():
        kernel = np.ones((dilation, dilation), dtype=np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        mask &= valid
    return mask
