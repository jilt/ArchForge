import cv2
import numpy as np
from typing import List, Tuple

Line = Tuple[Tuple[float, float], Tuple[float, float]]  # ((x1,y1),(x2,y2))


def detect_lines(image: np.ndarray,
                 canny_low: int = 50,
                 canny_high: int = 150,
                 min_line_length: int = 40,
                 max_line_gap: int = 10) -> List[Line]:
    """
    Detect line segments in an image using Canny + Probabilistic Hough.

    Args:
        image: H×W×3 uint8 (RGB or BGR; will convert to gray).
        canny_low/high: Canny thresholds.
        min_line_length: minimum line length in pixels.
        max_line_gap: max gap to link line segments.

    Returns:
        List of line segments as ((x1,y1),(x2,y2)).
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3, L2gradient=False)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if lines is None:
        return []

    result: List[Line] = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        result.append(((float(x1), float(y1)), (float(x2), float(y2))))
    return result


def cluster_lines(lines: List[Line],
                  angle_thresh_deg: float = 15.0
                  ) -> Tuple[List[Line], List[Line]]:
    """
    Split lines into 'vertical-like' and 'horizontal-like' sets.

    Uses angle w.r.t. horizontal axis.

    Returns:
        (vertical_lines, horizontal_lines)
    """
    vertical = []
    horizontal = []

    thresh_rad = np.deg2rad(angle_thresh_deg)

    for (x1, y1), (x2, y2) in lines:
        dx = x2 - x1
        dy = y2 - y1
        length = np.hypot(dx, dy)
        if length < 1e-3:
            continue

        # Angle from horizontal
        angle = np.arctan2(dy, dx)  # -pi..pi
        abs_angle = np.abs(angle)

        # Normalize to 0..pi/2
        if abs_angle > np.pi / 2:
            abs_angle = np.pi - abs_angle

        # Near horizontal: angle ~ 0
        # Near vertical: angle ~ pi/2
        if abs_angle < thresh_rad:
            horizontal.append(((x1, y1), (x2, y2)))
        elif (np.pi / 2 - abs_angle) < thresh_rad:
            vertical.append(((x1, y1), (x2, y2)))
        else:
            # Ignore strongly diagonal lines for simplicity
            continue

    return vertical, horizontal