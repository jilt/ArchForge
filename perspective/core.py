"""Conservative vertical-perspective correction for ArchForge.
Derived from the camera-rotation approach in metamountain/Perspective-Correction.
This adapter intentionally performs roll + pitch correction only: no yaw, no
facade squaring, no inpainting, and no fixed-canvas black borders.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Settings:
    detect_max_edge: int = 1600
    min_line_length_frac: float = 0.035
    vertical_window_deg: float = 32.0
    horizontal_window_deg: float = 32.0
    angular_softness: float = 0.35
    border_margin_px: int = 3
    ransac_iters: int = 800
    inlier_threshold_deg: float = 1.6
    min_vertical_lines: int = 4
    n_hypotheses: int = 4
    min_vp_distance_frac: float = 1.0
    seed: int = 20260831
    default_focal_35mm: float = 28.0
    min_confidence: float = 0.40
    max_pitch_deg: float = 30.0
    max_roll_deg: float = 12.0
    min_correction_deg: float = 0.15
    crop_min_coverage: float = 0.55
    interpolation: int = cv2.INTER_LANCZOS4


def _norm_vp(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([0.0, 1.0, 0.0])
    v = v / n
    return -v if v[np.argmax(np.abs(v))] < 0 else v


def _lines(seg):
    p0 = np.column_stack([seg[:, 0], seg[:, 1], np.ones(len(seg))])
    p1 = np.column_stack([seg[:, 2], seg[:, 3], np.ones(len(seg))])
    line = np.cross(p0, p1)
    n = np.linalg.norm(line[:, :2], axis=1)
    n[n < 1e-12] = 1e-12
    return line / n[:, None]


def _directions(seg):
    d = np.column_stack([seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]])
    n = np.linalg.norm(d, axis=1)
    n[n < 1e-12] = 1e-12
    return d / n[:, None]


def _lengths(seg):
    return np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])


def _prepare(bgr, settings):
    h0, w0 = bgr.shape[:2]
    scale = min(1.0, settings.detect_max_edge / max(w0, h0))
    if scale < 1.0:
        small = cv2.resize(bgr, (max(1, round(w0 * scale)), max(1, round(h0 * scale))), interpolation=cv2.INTER_AREA)
    else:
        small = bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    try:
        detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV, _scale=0.8, _sigma_scale=0.6, _quant=2.0, _ang_th=22.5, _density_th=0.7, _n_bins=1024)
    except Exception:
        detector = cv2.createLineSegmentDetector()
    result = detector.detect(gray)
    raw = result[0] if isinstance(result, tuple) else result
    if raw is None or len(raw) == 0:
        return None, scale
    seg = np.asarray(raw, dtype=float).reshape(-1, 4)
    length = _lengths(seg)
    min_len = max(8.0, settings.min_line_length_frac * min(w, h))
    seg = seg[length >= min_len]
    if len(seg) == 0:
        return None, scale
    mid = np.column_stack([(seg[:, 0] + seg[:, 2]) * .5, (seg[:, 1] + seg[:, 3]) * .5])
    span_x, span_y = np.abs(seg[:, 2] - seg[:, 0]), np.abs(seg[:, 3] - seg[:, 1])
    margin = settings.border_margin_px
    near_x = (mid[:, 0] < margin) | (mid[:, 0] > w - margin)
    near_y = (mid[:, 1] < margin) | (mid[:, 1] > h - margin)
    seg = seg[~((near_x & (span_x < margin * 2)) | (near_y & (span_y < margin * 2)))]
    if len(seg) == 0:
        return None, scale
    d = _directions(seg)
    angle_to_vertical = np.abs(np.arctan2(d[:, 0], d[:, 1]))
    angle_to_vertical = np.minimum(angle_to_vertical, math.pi - angle_to_vertical)
    vwin = math.radians(settings.vertical_window_deg)
    vmask = angle_to_vertical <= vwin
    seg = seg[vmask]
    if len(seg) == 0:
        return None, scale
    d = _directions(seg)
    mid = np.column_stack([(seg[:, 0] + seg[:, 2]) * .5, (seg[:, 1] + seg[:, 3]) * .5])
    length = _lengths(seg)
    a = np.abs(np.arctan2(d[:, 0], d[:, 1])); a = np.minimum(a, math.pi - a)
    x = np.clip(a / max(vwin, 1e-6), 0.0, 1.0)
    weight = length * np.exp(-(x * x) / (2 * settings.angular_softness * settings.angular_softness))
    return (seg, _lines(seg), mid, d, weight, w, h), scale


def _angular_residual(vp, mid, direction):
    bearing = vp[:2][None, :] - vp[2] * mid
    n = np.linalg.norm(bearing, axis=1)
    n[n < 1e-12] = 1e-12
    cross = np.abs(direction[:, 0] * bearing[:, 1] - direction[:, 1] * bearing[:, 0]) / n
    return np.arcsin(np.clip(cross, 0.0, 1.0))


def _plausible_vertical(vp, w, h, settings):
    cx, cy = w * .5, h * .5
    if abs(vp[2]) < 1e-9:
        dx, dy = vp[0], vp[1]
    else:
        dx, dy = vp[0] / vp[2] - cx, vp[1] / vp[2] - cy
        if math.hypot(dx, dy) < settings.min_vp_distance_frac * max(w, h):
            return False
    return abs(math.atan2(abs(dx), abs(dy))) <= math.radians(settings.vertical_window_deg + 8.0)


def _score(vp, mid, direction, weight, threshold):
    residual = _angular_residual(vp, mid, direction)
    inliers = residual <= threshold
    if not inliers.any():
        return inliers, 0.0
    soft = 1.0 - (residual[inliers] / threshold) ** 2
    return inliers, float(np.sum(weight[inliers] * (.35 + .65 * soft)))


def _refine_vp(lines, mid, weight, vp, iterations=6):
    v = _norm_vp(vp)
    for _ in range(iterations):
        bearing = v[:2][None, :] - v[2] * mid
        d2 = np.sum(bearing * bearing, axis=1)
        d2[d2 < 1e-12] = 1e-12
        matrix = (lines * (weight / d2)[:, None]).T @ lines
        try:
            _, vecs = np.linalg.eigh(matrix)
        except np.linalg.LinAlgError:
            break
        nxt = _norm_vp(vecs[:, 0])
        if np.linalg.norm(nxt - v) < 1e-12:
            break
        v = nxt
    return v


def _parallel_hypothesis(direction, weight, threshold, mid):
    d = direction * np.sign(direction[:, 1])[:, None]
    mean = (d * weight[:, None]).sum(axis=0)
    n = np.linalg.norm(mean)
    if n < 1e-12:
        return None
    vp = _norm_vp(np.array([mean[0] / n, mean[1] / n, 0.0]))
    return vp, _score(vp, mid, direction, weight, threshold)


def _find_vertical_vp(lines, mid, direction, weight, w, h, settings):
    n = len(lines)
    if n < settings.min_vertical_lines:
        return None
    threshold = math.radians(settings.inlier_threshold_deg)
    rng = np.random.default_rng(settings.seed)
    probs = weight / weight.sum() if weight.sum() > 0 else np.full(n, 1.0 / n)
    best = []
    for _ in range(settings.ransac_iters):
        i, j = rng.choice(n, 2, replace=False, p=probs)
        ai = abs(math.atan2(direction[i,0], direction[i,1])); aj = abs(math.atan2(direction[j,0], direction[j,1]))
        if abs(ai - aj) < math.radians(.6):
            continue
        vp = _norm_vp(np.cross(lines[i], lines[j]))
        if not _plausible_vertical(vp, w, h, settings):
            continue
        inliers, score = _score(vp, mid, direction, weight, threshold)
        if score <= 0:
            continue
        if any(abs(float(vp @ old[1])) > math.cos(math.radians(2.0)) for old in best):
            continue
        best.append((score, vp, inliers))
        best.sort(key=lambda x: -x[0])
        best = best[:settings.n_hypotheses]
    parallel = _parallel_hypothesis(direction, weight, threshold, mid)
    if parallel is not None:
        vp, (inliers, score) = parallel
        best.append((score, vp, inliers))
    if not best:
        return None
    best.sort(key=lambda x: -x[0])
    for _, vp, inliers in best:
        if not inliers.any():
            continue
        refined = _refine_vp(lines[inliers], mid[inliers], weight[inliers], vp)
        if _plausible_vertical(refined, w, h, settings):
            vp = refined
        inliers, score = _score(vp, mid, direction, weight, threshold)
        share = float(weight[inliers].sum() / max(weight.sum(), 1e-9))
        return vp, inliers, share
    return None


def _intrinsics(f, cx, cy):
    return np.array([[f,0.,cx],[0.,f,cy],[0.,0.,1.]])


def _focal_px(w, h, focal35=28.0):
    return float(focal35) * math.hypot(w, h) / math.hypot(36., 24.)


def _estimate_roll_pitch(vp, w, h, focal):
    K = _intrinsics(focal, w * .5, h * .5)
    up = np.linalg.inv(K) @ vp
    n = np.linalg.norm(up)
    if n < 1e-12:
        return None
    up = up / n
    if float(up @ np.array([0., -1., 0.])) < 0:
        up = -up
    roll = float(math.atan2(up[0], -up[1]))
    pitch = float(math.atan2(up[2], math.hypot(up[0], up[1])))
    return roll, pitch


def _confidence(inliers, weight, mid, w):
    count = int(inliers.sum())
    if count < 2:
        return 0.0
    share = float(weight[inliers].sum() / max(weight.sum(), 1e-9))
    c_share = min(1.0, share / .55)
    c_count = min(1.0, (count - 1) / 6.0)
    spread = float(mid[inliers, 0].max() - mid[inliers, 0].min()) / max(w, 1)
    c_spread = min(1.0, spread / .45)
    # Without EXIF/horizon evidence in a Comfy tensor, use a conservative focal term.
    return float(np.clip(c_share * c_count * c_spread * .60, 0.0, 1.0))


def _rot_x(a):
    c,s = math.cos(a), math.sin(a)
    return np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])


def _rot_z(a):
    c,s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def _homography(w, h, focal, roll, pitch):
    K = _intrinsics(focal, w * .5, h * .5)
    # Reference convention: correction_rotation(roll,pitch,0) = Rx(pitch) @ Rz(-roll)
    R = _rot_x(pitch) @ _rot_z(-roll)
    H = K @ R @ np.linalg.inv(K)
    return H / H[2,2]


def _apply_h(H, pts):
    p = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    z = p[:,2:3]
    z[np.abs(z) < 1e-12] = 1e-12
    return p[:,:2] / z


def _inside(quad, pts):
    sign = None
    for i in range(len(quad)):
        a,b = quad[i], quad[(i+1)%len(quad)]
        edge = b-a
        cross = edge[0]*(pts[:,1]-a[1]) - edge[1]*(pts[:,0]-a[0])
        values = np.sign(cross); values = values[values != 0]
        if len(values) == 0:
            continue
        if sign is None: sign = values[0]
        if np.any(values != sign): return False
    return True


def _inscribed_rect(quad, aspect, centre, iterations=40):
    lo, hi = 0.0, float(np.max(np.abs(quad-centre))*2.0+1.0)
    for _ in range(iterations):
        hw = (lo+hi)*.5; hh = hw/aspect
        pts = centre + np.array([[-hw,-hh],[hw,-hh],[hw,hh],[-hw,hh]])
        if _inside(quad, pts): lo = hw
        else: hi = hw
    hh = lo/aspect
    return np.array([centre[0]-lo, centre[1]-hh, centre[0]+lo, centre[1]+hh])


def _plan_crop(H, w, h, min_coverage):
    corners = np.array([[0.,0.],[float(w),0.],[float(w),float(h)],[0.,float(h)]])
    quad = _apply_h(H, corners)
    centre = _apply_h(H, np.array([[w*.5,h*.5]]))[0]
    rect = _inscribed_rect(quad, w / h, centre)
    rw, rh = rect[2]-rect[0], rect[3]-rect[1]
    quad_area = .5 * abs(float(np.dot(quad[:,0],np.roll(quad[:,1],-1))-np.dot(quad[:,1],np.roll(quad[:,0],-1))))
    coverage = rw*rh/max(quad_area,1e-9)
    if rw < 8 or rh < 8 or coverage < min_coverage:
        return None
    T = np.array([[1.,0.,-rect[0]],[0.,1.,-rect[1]],[0.,0.,1.]])
    return T @ H, max(1,int(round(rw))), max(1,int(round(rh)))


def compute_perspective_correction(image: np.ndarray, strength: float = 1.0, auto_crop: bool = True) -> np.ndarray:
    """Correct roll and vertical keystone in an RGB uint8 image.

    On uncertain geometry, unsafe corrections, or invalid crop plans, returns
    the unmodified input. Strength blends camera angles before warping.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an HxWx3 uint8 RGB image.")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return image
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    settings = Settings()
    prepared, scale = _prepare(bgr, settings)
    if prepared is None:
        return image
    seg, lines, mid, direction, weight, sw, sh = prepared
    found = _find_vertical_vp(lines, mid, direction, weight, sw, sh, settings)
    if found is None:
        return image
    vp, inliers, _ = found
    confidence = _confidence(inliers, weight, mid, sw)
    if confidence < settings.min_confidence:
        return image
    h, w = bgr.shape[:2]
    focal_small = _focal_px(sw, sh, settings.default_focal_35mm)
    focal = focal_small / scale
    # Full-resolution VP coordinates: homogeneous scaling preserves geometry.
    S = np.array([[scale,0.,0.],[0.,scale,0.],[0.,0.,1.]])
    vp_full = np.linalg.inv(S) @ vp
    roll_pitch = _estimate_roll_pitch(vp_full, w, h, focal)
    if roll_pitch is None:
        return image
    roll, pitch = roll_pitch
    roll *= strength; pitch *= strength
    if abs(math.degrees(roll)) > settings.max_roll_deg or abs(math.degrees(pitch)) > settings.max_pitch_deg:
        return image
    if math.degrees(math.hypot(roll, pitch)) < settings.min_correction_deg:
        return image
    H = _homography(w, h, focal, roll, pitch)
    plan = _plan_crop(H, w, h, settings.crop_min_coverage) if auto_crop else None
    if plan is None:
        # Edge replication is allowed only for a modest expansion. A crop plan
        # failure means the correction is unsafe; preserve the source instead.
        return image
    H_total, out_w, out_h = plan
    out = cv2.warpPerspective(bgr, H_total, (out_w, out_h), flags=settings.interpolation, borderMode=cv2.BORDER_REPLICATE)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
