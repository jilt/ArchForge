import cv2
import numpy as np
from typing import List, Tuple, Optional

Line = Tuple[Tuple[float, float], Tuple[float, float]]


def line_intersection(l1: Line, l2: Line) -> Optional[Tuple[float, float]]:
    """
    Compute intersection of two 2D lines (if not parallel).
    Each line is ((x1,y1),(x2,y2)).
    Returns (x,y) or None if parallel / degenerate.
    """
    (x1, y1), (x2, y2) = l1
    (x3, y3), (x4, y4) = l2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-8:
        return None

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) -
          (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) -
          (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return float(px), float(py)


def estimate_vanishing_point(lines: List[Line],
                             image_shape: Tuple[int, int, int]
                             ) -> Optional[Tuple[float, float]]:
    """
    Estimate a single vanishing point from a set of lines (e.g., verticals).

    Strategy:
      - Compute all pairwise intersections.
      - Take the median (robust to outliers).

    Returns:
        (vp_x, vp_y) in image coordinates, or None if not enough lines.
    """
    if len(lines) < 2:
        return None

    H, W, _ = image_shape

    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            pt = line_intersection(lines[i], lines[j])
            if pt is None:
                continue
            x, y = pt
            # Filter insane intersections far outside the image
            if not (-W < x < 2 * W and -H < y < 2 * H):
                continue
            intersections.append((x, y))

    if len(intersections) < 2:
        return None

    xs = np.array([p[0] for p in intersections])
    ys = np.array([p[1] for p in intersections])

    vp_x = float(np.median(xs))
    vp_y = float(np.median(ys))
    return vp_x, vp_y


def compute_homography_to_straighten_verticals(
    vp: Tuple[float, float],
    image_shape: Tuple[int, int, int],
    target_distance_px: float = 1e6
) -> np.ndarray:
    """
    Compute a 3x3 homography that moves the vertical vanishing point
    to infinity (i.e., makes vertical lines parallel).

    We approximate this by mapping the VP to a point very far away
    along the vertical direction.

    Args:
        vp: (vp_x, vp_y) in image coordinates.
        image_shape: (H, W, C).
        target_distance_px: how far to push the VP (large number).

    Returns:
        3x3 homography matrix (float64).
    """
    H_img, W_img, _ = image_shape
    vp_x, vp_y = vp

    # Source: image corners
    src = np.array([
        [0, 0],
        [W_img, 0],
        [W_img, H_img],
        [0, H_img],
    ], dtype=np.float32)

    # Simple heuristic warp: shift top edge based on VP
    def shift_x(x: float, y: float) -> float:
        if abs(vp_y - y) < 1e-6:
            return x
        t = (0 - y) / (vp_y - y)
        x_on_vp_line = x + t * (vp_x - x)
        return x + (vp_x - x_on_vp_line) * 0.5

    dst = np.array([
        [shift_x(0, 0), 0],
        [shift_x(W_img, 0), 0],
        [W_img, H_img],
        [0, H_img],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src, dst, method=0)  # 0 = regular DLT
    return H