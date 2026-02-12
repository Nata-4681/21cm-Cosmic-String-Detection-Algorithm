from __future__ import annotations

import numpy as np


def _neighbors_4(y: int, x: int, H: int, W: int):
    if y > 0: yield (y - 1, x)
    if y < H - 1: yield (y + 1, x)
    if x > 0: yield (y, x - 1)
    if x < W - 1: yield (y, x + 1)


def largest_connected_component(binary: np.ndarray) -> np.ndarray:
    """
    Returns a boolean mask of the largest 4-connected component in `binary`.
    If no True pixels exist, returns all-False mask.
    """
    H, W = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    best_coords = []

    ys, xs = np.where(binary)
    for y0, x0 in zip(ys, xs):
        if visited[y0, x0]:
            continue

        # BFS/DFS
        stack = [(y0, x0)]
        visited[y0, x0] = True
        coords = []

        while stack:
            y, x = stack.pop()
            coords.append((y, x))
            for yn, xn in _neighbors_4(y, x, H, W):
                if binary[yn, xn] and not visited[yn, xn]:
                    visited[yn, xn] = True
                    stack.append((yn, xn))

        if len(coords) > len(best_coords):
            best_coords = coords

    out = np.zeros_like(binary, dtype=bool)
    for y, x in best_coords:
        out[y, x] = True
    return out


def detect_from_probmap(
    prob: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 50,
) -> tuple[bool, np.ndarray]:
    """
    Threshold -> keep largest connected component -> apply min_area rule.
    Returns (detected?, blob_mask)
    """
    binary = prob >= threshold
    blob = largest_connected_component(binary)
    detected = int(blob.sum()) >= int(min_area)
    if not detected:
        blob[:] = False
    return detected, blob


def centroid_from_mask(mask: np.ndarray) -> tuple[float, float]:
    """
    Returns (cx, cy) in pixel coordinates (float).
    If empty mask, returns (nan, nan).
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return float("nan"), float("nan")
    cx = xs.mean()
    cy = ys.mean()
    return float(cx), float(cy)


def estimate_snr_from_blob(
    image: np.ndarray,
    blob: np.ndarray,
) -> float:
    """
    Simple SNR estimate:
      A_hat = mean(image in blob) - mean(image outside blob)
      sigma_hat = std(image outside blob)
      SNR_hat = A_hat / sigma_hat
    """
    if blob.sum() == 0:
        return 0.0
    sig_vals = image[blob]
    bg_vals = image[~blob]
    if bg_vals.size < 10:
        return 0.0
    A_hat = float(sig_vals.mean() - bg_vals.mean())
    sigma_hat = float(bg_vals.std() + 1e-12)
    return A_hat / sigma_hat


def iou(pred: np.ndarray, true: np.ndarray) -> float:
    inter = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)
