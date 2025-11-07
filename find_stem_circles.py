# ================================================================
# circle_detector.py — Slice-wise ring detection + tracking for stems
# ---------------------------------------------------------------
# Purpose
#   Detect stem cross-sections in LiDAR slice_*.ply files, track them across
#   height, and export per-slice detections + a stems summary (with DBH).
#
# What it does
#   • Builds a fixed global raster frame over all slices.
#   • Rasterizes each slice to an occupancy heatmap.
#   • Detects rings (Hough + RANSAC) with global gates, then recovers misses
#     via relaxed ROI detection around track predictions.
#   • Tracks centers/diameters by nearest-neighbour association with gates.
#   • Exports CSVs and debug PNG overlays; computes per-track DBH (median
#     over 1.3–1.7 m, fallback to median of all obs).
#
# Inputs
#   • slices_dir: folder containing ASCII PLY slices named like slice_<h>m.ply
#
# Outputs (to out_dir)
#   • slice_XXXX.csv                      — per-slice detections
#   • slice_XXXX_debug.png                — visualization overlays
#   • stems_summary.csv                   — track_id, n_obs, median_DBH_m, mean (x,y)
#
# Configure
#   • Set parameters via the Params dataclass (pixel sizes, gates, thresholds).
#
# Usage (programmatic)
#   from pathlib import Path
#   run(Params(slices_dir=Path(".../slices"), out_dir=Path(".../circle_detections"), pixel_size_m=0.02))
#
# Dependencies
#   Python 3.9+, numpy, opencv-python, dataclasses
# 
# Author 
#   Ethan Glynn [University of Sydney]
# ================================================================

import os
import re
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2
import math

# ---------- Config ----------
@dataclass
class Params:
    # Input/Output
    slices_dir: Path                # folder with slice_*.ply files
    out_dir: Path                   # folder for CSV + debug images

    # Rasterization
    pixel_size_m: float = 0.02      # metres per pixel in XY (adjust to your density)
    padding_m: float = 0.5          # extra border around point extents

    # Preprocessing
    gaussian_sigma_px: int = 3      # blur for denoising before threshold
    min_blob_area_px: int = 150     # small specks are removed
    morph_open_px: int = 3          # radius of opening to separate touching blobs

    # Detection constraints (metres)
    min_diam_m: float = 0.26        # 36cm min diameter, based on ground truth
    max_diam_m: float = 0.6        # 53cm max diameter, based on ground truth
    min_fill_ratio = 0.45       # instead of 0.35
    min_circ_compactness = 0.65 # instead of 0.55
    # File pattern
    slice_regex: str = r"slice_(?P<h>[0-9]+\.[0-9]+)m\.ply"

@dataclass
class Params(Params):
    """
    Extended detector/tracker parameters.

    NOTE: This class intentionally inherits from an earlier `Params`
    declaration that defines core I/O and rasterization fields
    (slices_dir, out_dir, pixel_size_m, etc.). We're layering
    tracking/ROI and global gate settings here.
    """

    # --- Tracking / ROI gates ---
    max_match_dist_m: float = 0.35         # nearest-neighbour association radius (m)
    max_diam_jump_frac: float = 0.28       # allowed fractional diameter jump between slices
    roi_radius_m: float = 0.55             # ROI radius for relaxed re-detection (m)

    # --- Relaxed (ROI / tracking) thresholds ---
    relaxed_grad_align_tol_deg: float = 38.0
    relaxed_min_ringness: float = 0.30
    relaxed_min_arc_deg: float = 110.0     # allow shorter arcs during ROI recovery

    # --- Global (full-slice) thresholds ---
    global_grad_align_deg: float = 20.0    # edge/gradient alignment tolerance (deg)
    global_min_ringness: float = 0.50
    global_min_arc_deg: float = 130.0      # arc completeness for global detect

    # --- Inlier requirements for RANSAC ring fit ---
    global_min_inliers_px: int = 30        # used for full-slice/global detection
    relaxed_min_inliers_px: int = 18       # used for ROI/bridge/perimeter detection

@dataclass
class Track:
    """
    Per-stem track accumulated across slice heights.

    Attributes:
        id:         Unique track identifier.
        centers_m:  Sequence of (x, y) world coordinates per observation.
        diam_m:     Measured diameter (m) per observation.
        heights_m:  Slice heights (m) corresponding to each observation.
        missed:     Consecutive-slice miss counter.
    """
    id: int
    centers_m: List[Tuple[float, float]]   # (x, y) world coords
    diam_m: List[float]                    # diameter per observation
    heights_m: List[float]                 # slice height for each observation
    missed: int = 0                        # consecutive misses

    @property
    def center(self) -> Tuple[float, float]:
        """Return the latest (x, y) center in world coordinates."""
        return self.centers_m[-1]

    @property
    def dbh_m(self) -> float:
        """
        Median diameter near breast height (1.3–1.7 m), falling back to the
        overall median if no observations land in that window.
        """
        vals = [d for d, h in zip(self.diam_m, self.heights_m) if 1.3 <= h <= 1.7]
        vals = vals or self.diam_m
        return float(np.median(vals))

# ---------- I/O ----------
def load_ply_xyz_ascii(filepath: Path) -> np.ndarray:
    with open(filepath, "r") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Invalid PLY: no end_header")
            if line.strip() == "end_header":
                break
        data = np.loadtxt(f, dtype=np.float32)
    if data.ndim == 1:
        if data.size < 3:
            return np.empty((0, 3), dtype=np.float32)
        data = data[None, :]
    return data[:, :3]  # x, y, z

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ---------- Geometry helpers ----------

def world_to_image(points_xy: np.ndarray, pixel_size: float, padding_m: float):
    """
    Convert world (x, y) coordinates to image pixel coordinates,
    while computing image size and transform parameters.
    """
    x = points_xy[:, 0]
    y = points_xy[:, 1]

    # Compute padded bounds
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xmin -= padding_m
    ymin -= padding_m
    xmax += padding_m
    ymax += padding_m

    # Compute image dimensions in pixels
    width_m = max(xmax - xmin, pixel_size)
    height_m = max(ymax - ymin, pixel_size)
    W = int(np.ceil(width_m / pixel_size))
    H = int(np.ceil(height_m / pixel_size))

    # Map world → pixel coordinates
    px = (x - xmin) / pixel_size
    py = (ymax - y) / pixel_size  # invert Y axis (image origin top-left)
    pix = np.stack([px, py], axis=1).astype(np.float32)

    tf = {"xmin": xmin, "ymax": ymax, "pixel_size": pixel_size}
    return (H, W), tf, pix


def image_to_world(cx_px: float, cy_px: float, tf) -> tuple[float, float]:
    """
    Convert image pixel coordinates → world coordinates using the stored transform.
    """
    x = tf["xmin"] + cx_px * tf["pixel_size"]
    y = tf["ymax"] - cy_px * tf["pixel_size"]
    return float(x), float(y)


def world_point_to_pixel(tf, x: float, y: float) -> Tuple[float, float]:
    """
    Convert world (x, y) → fractional pixel coordinates (px, py).
    """
    px = (x - tf["xmin"]) / tf["pixel_size"]
    py = (tf["ymax"] - y) / tf["pixel_size"]
    return float(px), float(py)


def crop_roi(img: np.ndarray, cx_px: float, cy_px: float, r_px: float):
    """
    Crop a square ROI centered at (cx_px, cy_px) with radius r_px (pixels).
    Returns the cropped region and its top-left corner (x0, y0).
    """
    H, W = img.shape[:2]

    x0 = int(max(0, np.floor(cx_px - r_px)))
    y0 = int(max(0, np.floor(cy_px - r_px)))
    x1 = int(min(W, np.ceil(cx_px + r_px)))
    y1 = int(min(H, np.ceil(cy_px + r_px)))

    crop = img[y0:y1, x0:x1].copy()
    return crop, (x0, y0)


def nms_world(dets, min_dist_m=0.15, diam_tol_frac=0.20):
    """
    Apply non-maximum suppression (NMS) to remove duplicate detections
    that are close in position and diameter, keeping higher ringness scores.
    """
    dets = sorted(dets, key=lambda d: d.get("ringness", 0.0), reverse=True)
    kept = []

    for d in dets:
        keep = True
        for k in kept:
            dx = d["center_x_m"] - k["center_x_m"]
            dy = d["center_y_m"] - k["center_y_m"]
            dist = (dx * dx + dy * dy) ** 0.5

            if dist < min_dist_m:
                diam_diff = abs(d["diameter_m"] - k["diameter_m"])
                diam_thresh = diam_tol_frac * max(d["diameter_m"], k["diameter_m"])
                if diam_diff <= diam_thresh:
                    keep = False
                    break

        if keep:
            kept.append(d)

    return kept

def bridge_missing_slices(tracks, heights_sorted, cache_occ_tf, params):
    """
    Attempt to recover detections in missing slice heights by interpolating
    each track's position/diameter between consecutive observations and running
    a relaxed ROI detector around the interpolated expectation.

    Args (structures are intentionally loose to avoid changing behavior):
        tracks: list of Track-like objects with attributes:
                - heights_m: list[float]
                - centers_m: list[(x, y)]
                - diam_m:    list[float]
        heights_sorted: sorted iterable of all slice heights present
        cache_occ_tf: dict-like mapping height -> (occupancy_image, transform)
        params: object with attributes used below (pixel_size_m, roi_radius_m, etc.)
    """
    for t in tracks:
        # Sort this track's observations by height
        obs = sorted(zip(t.heights_m, t.centers_m, t.diam_m))

        # Walk adjacent observations (h1 -> h2), filling any internal gaps
        for (h1, (x1, y1), d1), (h2, (x2, y2), d2) in zip(obs, obs[1:]):
            # Heights strictly between h1 and h2 that the track missed
            gaps = [h for h in heights_sorted if h1 < h < h2 and h not in t.heights_m]

            for h in gaps:
                # Pull cached occupancy + transform for that slice height
                occ, tf = cache_occ_tf[h]

                # Linear interpolation factor (guard tiny div.)
                a = (h - h1) / max(h2 - h1, 1e-6)

                # Expected center (ex, ey) and diameter ed at missing height
                ex = (1 - a) * x1 + a * x2
                ey = (1 - a) * y1 + a * y2
                ed = (1 - a) * d1 + a * d2

                # Run relaxed ROI ring detection around the expectation
                best = detect_rings_relaxed_roi(
                    occ, tf, params.pixel_size_m,
                    center_xy_m=(ex, ey),
                    roi_radius_m=min(params.roi_radius_m, 0.30),
                    min_diam_m=ed * (1.0 - params.max_diam_jump_frac),
                    max_diam_m=ed * (1.0 + params.max_diam_jump_frac),
                    grad_tol_deg=params.relaxed_grad_align_tol_deg,
                    min_ringness=params.relaxed_min_ringness,
                    expected_center_xy_m=(ex, ey),
                    expected_diam_m=ed,
                    center_gate_m=min(params.max_match_dist_m, 0.20),
                    diam_gate_frac=min(params.max_diam_jump_frac, 0.12),
                    top_k=1
                )

                # If we got a plausible recovery, associate it into the track set
                if best:
                    associate_and_update(tracks, best, h, params)

# ---------- Detection pipeline ----------

def rasterize(points_xy: np.ndarray, img_shape, pix: np.ndarray) -> np.ndarray:
    """
    Plot points into a binary raster image (255 at occupied pixels).

    Args:
        points_xy: (N,2) world XY (unused here; kept for API symmetry).
        img_shape: (H, W) image height/width.
        pix:       (N,2) fractional pixel coordinates for each point.

    Returns:
        uint8 image of shape (H, W) with values {0, 255}.
    """
    H, W = img_shape
    img = np.zeros((H, W), dtype=np.uint8)

    ij = np.rint(pix).astype(np.int32)
    ij[:, 0] = np.clip(ij[:, 0], 0, W - 1)  # x
    ij[:, 1] = np.clip(ij[:, 1], 0, H - 1)  # y

    img[ij[:, 1], ij[:, 0]] = 255
    return img


def rasterize_heatmap(points_xy: np.ndarray, img_shape, pix: np.ndarray) -> np.ndarray:
    """
    Accumulate points into a smoothed heatmap normalized to [0, 255].

    Args:
        points_xy: (N,2) world XY (unused here; kept for API symmetry).
        img_shape: (H, W) image height/width.
        pix:       (N,2) fractional pixel coordinates for each point.

    Returns:
        uint8 heatmap of shape (H, W) in [0, 255].
    """
    H, W = img_shape
    heat = np.zeros((H, W), dtype=np.float32)

    ij = np.rint(pix).astype(np.int32)
    x = np.clip(ij[:, 0], 0, W - 1)
    y = np.clip(ij[:, 1], 0, H - 1)

    # Atomic add per (y, x) occurrence
    np.add.at(heat, (y, x), 1.0)

    # Smooth, then normalize to 0..255 if nonzero
    heat = cv2.GaussianBlur(heat, (0, 0), 1.2)
    if heat.max() > 0:
        heat = (255.0 * (heat / heat.max())).astype(np.uint8)
    else:
        heat = heat.astype(np.uint8)

    return heat


def preprocess(mask: np.ndarray, sigma: int, min_area: int, open_r: int) -> np.ndarray:
    """
    Clean up a grayscale/accumulation mask into a binary foreground mask.

    Steps:
      1) Optional Gaussian blur with sigma (derived odd kernel size).
      2) Threshold to binary (mask > 1).
      3) Optional morphological opening (ellipse radius=open_r).
      4) Remove small components (< min_area).

    Args:
        mask:      uint8/float image.
        sigma:     Gaussian sigma; if 0, skip smoothing.
        min_area:  minimum connected-component area to keep (px).
        open_r:    radius for morphological opening; if 0, skip.

    Returns:
        Binary uint8 mask (0/255) of same shape.
    """
    # 1) Optional blur (compute odd kernel size from sigma)
    if sigma > 0:
        k = max(1, int(2 * round(3 * sigma) + 1))
        mask = cv2.GaussianBlur(mask, (k, k), sigmaX=sigma, sigmaY=sigma)

    # 2) Binary threshold
    _, bw = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    # 3) Optional morphological opening
    if open_r > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * open_r + 1, 2 * open_r + 1)
        )
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)

    # 4) Remove small blobs
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    cleaned = np.zeros_like(bw)
    for i in range(1, num):  # skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def edgeify(occ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute edge map and per-pixel gradient direction from an occupancy/heatmap image.

    Args:
        occ: uint8/float image; larger values = denser points.

    Returns:
        edges:   uint8 Canny edge map.
        grad_dir: float32 array of same shape as occ; gradient angle (radians) via arctan2(gy, gx).
    """
    # Light smoothing reduces speckle before edge/gradient ops
    blurred = cv2.GaussianBlur(occ, (5, 5), 1.0)

    # Binary edges
    edges = cv2.Canny(blurred, threshold1=20, threshold2=60, L2gradient=True)

    # Gradient components and direction (in radians)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad_dir = np.arctan2(gy, gx)

    return edges, grad_dir


def contour_compactness(contour: np.ndarray) -> float:
    """
    Return the circular compactness 4πA / P² for a contour.
    Perfect circle → 1.0; smaller values are less circular.

    Args:
        contour: Nx1x2 contour array from OpenCV.

    Returns:
        Compactness score in [0, 1] (0 if perimeter is degenerate).
    """
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    if peri <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (peri * peri))


def detect_circles(
    mask: np.ndarray,
    pixel_size: float,
    min_diam_m: float,
    max_diam_m: float,
    min_fill_ratio: float,
    min_compactness: float
):
    """
    Detect circular blobs in a binary mask using contour geometry + simple gates.

    Steps per contour:
      • Fit min enclosing circle → (cx, cy, r_px)
      • Compute blob fill ratio (blob_area / circle_area)
      • Compute compactness (4πA / P²)
      • Gate on physical diameter, fill ratio, and compactness
      • Keep best via simple NMS in pixel space

    Args:
        mask:            binary mask (uint8) where blobs > 0.
        pixel_size:      metres per pixel.
        min_diam_m:      minimum allowed physical diameter (m).
        max_diam_m:      maximum allowed physical diameter (m).
        min_fill_ratio:  minimum blob fill ratio inside its enclosing circle.
        min_compactness: minimum circular compactness.

    Returns:
        List of dicts, each with:
          { "cx_px", "cy_px", "r_px", "radius_m", "diameter_m",
            "fill_ratio", "compactness" }
    """
    dets = []

    # Find external contours only
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        if len(c) < 5:
            continue

        # Minimal enclosing circle in pixels
        (cx_px, cy_px), r_px = cv2.minEnclosingCircle(c)
        circle_area_px = np.pi * (r_px ** 2)
        blob_area_px = cv2.contourArea(c)

        # Avoid divide-by-zero and tiny circles
        if circle_area_px <= 1.0:
            continue

        # Geometry/appearance metrics
        fill_ratio = float(blob_area_px / circle_area_px)
        compact = contour_compactness(c)

        # Convert to metres and gate by physical size
        r_m = r_px * pixel_size
        d_m = 2.0 * r_m
        if d_m < min_diam_m or d_m > max_diam_m:
            continue

        # Gate by shape/occupancy
        if fill_ratio < min_fill_ratio:
            continue
        if compact < min_compactness:
            continue

        dets.append({
            "cx_px": cx_px, "cy_px": cy_px, "r_px": r_px,
            "radius_m": r_m, "diameter_m": d_m,
            "fill_ratio": fill_ratio, "compactness": compact
        })

    # De-duplicate overlapping circles using pixel-space NMS
    dets = nms_circles(dets, pixel_thresh=6.0)
    return dets

rng = np.random.default_rng(1234)


def hough_ring_candidates(
    edges: np.ndarray,
    pixel_size: float,
    min_diam_m: float,
    max_diam_m: float,
    minDist_px: float = 12.0,
):
    """
    Propose circular candidates via OpenCV HoughCircles, bounded by physical diameter.

    Args:
        edges:       uint8 Canny edge image (H×W).
        pixel_size:  metres per pixel.
        min_diam_m:  minimum allowed circle diameter (m).
        max_diam_m:  maximum allowed circle diameter (m).
        minDist_px:  minimum center distance between circles in pixels.

    Returns:
        List of (cx_px, cy_px, r_px) tuples in pixel units.
    """
    H, W = edges.shape

    # Convert diameter bounds (m) → radius bounds (px)
    minR = max(3, int((min_diam_m / pixel_size) / 2.0))
    maxR = int((max_diam_m / pixel_size) / 2.0)
    if maxR <= minR:
        return []

    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=minDist_px,
        param1=120,
        param2=18,
        minRadius=minR,
        maxRadius=maxR,
    )

    out = []
    if circles is not None:
        for c in circles[0]:
            x, y, r = float(c[0]), float(c[1]), float(c[2])
            # Keep only circles whose centers lie within image bounds
            if 0 <= x < W and 0 <= y < H:
                out.append((x, y, r))
    return out


def ransac_circles_from_edges(
    edges: np.ndarray,
    pixel_size: float,
    min_diam_m: float,
    max_diam_m: float,
    trials: int = 1800,
    inlier_tol_px: float = 2.8,
    min_arc_deg: float = 130.0,
    subsample: int = 9000,
    min_inliers_px: int = 30,  # NEW threshold already in your code
):
    """
    Fit circles to edge points via 3-point RANSAC + coverage gating.

    Pipeline:
      • Sample 3 points → compute circle (center, radius)
      • Radius gate within [minR, maxR] (px) derived from physical limits
      • Inlier set = points within inlier_tol_px of radius
      • Require >= min_inliers_px
      • Compute angular coverage (degrees) and require >= min_arc_deg
      • Score by inliers × (coverage/360)
      • Greedy NMS over (cx, cy, r) to de-duplicate

    Args:
        edges:           uint8 Canny edge image (H×W).
        pixel_size:      metres per pixel (unused directly but kept for symmetry).
        min_diam_m:      minimum diameter in metres.
        max_diam_m:      maximum diameter in metres.
        trials:          RANSAC iterations.
        inlier_tol_px:   radial tolerance for inlier test (px).
        min_arc_deg:     minimum angular coverage required (deg).
        subsample:       limit on sampled edge points for speed.
        min_inliers_px:  minimum number of inliers to accept a model.

    Returns:
        List of tuples (cx_px, cy_px, r_px, score).
    """
    # Collect edge coordinates (x=cols, y=rows) in pixel space
    ys, xs = np.nonzero(edges)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    if pts.shape[0] == 0:
        return []

    # Optional subsampling for speed
    if pts.shape[0] > subsample:
        idx = rng.choice(pts.shape[0], size=subsample, replace=False)
        pts = pts[idx]

    # Diameter (m) → radius (px) bounds
    minR = max(3, int((min_diam_m / pixel_size) / 2.0))
    maxR = int((max_diam_m / pixel_size) / 2.0)

    best = []
    for _ in range(trials):
        # Sample 3 distinct points
        i, j, k = rng.choice(len(pts), size=3, replace=False)
        p1, p2, p3 = pts[i], pts[j], pts[k]

        # Solve circle from 3 points via linear system
        A = 2 * np.array(
            [[p2[0] - p1[0], p2[1] - p1[1]],
             [p3[0] - p1[0], p3[1] - p1[1]]],
            dtype=np.float32,
        )
        b = np.array(
            [[p2[0] ** 2 - p1[0] ** 2 + p2[1] ** 2 - p1[1] ** 2],
             [p3[0] ** 2 - p1[0] ** 2 + p3[1] ** 2 - p1[1] ** 2]],
            dtype=np.float32,
        )

        # Discard degenerate triplets
        det = np.linalg.det(A)
        if abs(det) < 1e-6:
            continue

        center = np.linalg.solve(A, b).ravel()
        cx, cy = center[0], center[1]
        r = np.hypot(p1[0] - cx, p1[1] - cy)

        # Radius gate
        if not (minR <= r <= maxR):
            continue

        # Inlier test: distance to circumference within tolerance
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        resid = np.abs(d - r)
        inliers = resid <= inlier_tol_px
        n_in = int(inliers.sum())
        if n_in < min_inliers_px:
            continue

        # Angular coverage over inliers
        angles = np.degrees(np.arctan2(pts[inliers, 1] - cy, pts[inliers, 0] - cx))
        ang_sorted = np.sort((angles + 360) % 360)
        diffs = np.diff(np.r_[ang_sorted, ang_sorted[0] + 360])
        coverage = 360.0 - float(diffs.max())
        if coverage < min_arc_deg:
            continue

        # Score favors many inliers with broad coverage
        score = n_in * (coverage / 360.0)
        best.append((cx, cy, r, score))

    # Sort by score descending
    best.sort(key=lambda t: t[3], reverse=True)

    # Simple NMS in parameter space to prune near-duplicates
    kept = []
    taken = np.zeros(len(best), dtype=bool)
    for a_idx, a in enumerate(best):
        if taken[a_idx]:
            continue
        kept.append(a)
        for b_idx in range(a_idx + 1, len(best)):
            if taken[b_idx]:
                continue
            b = best[b_idx]
            # Close centers and similar radii → suppress b
            if math.hypot(a[0] - b[0], a[1] - b[1]) < 6.0 and abs(a[2] - b[2]) < 3.0:
                taken[b_idx] = True

    return kept

def score_ringness(
    edges: np.ndarray,
    grad_dir: np.ndarray,
    cx: float,
    cy: float,
    r_px: float,
    radial_tol_px: float = 2.5,
    align_tol_deg: float = 30.0,
):
    """
    Estimate how "ring-like" a candidate circle is at (cx, cy, r_px).

    We look at edge pixels that fall within a thin radial band around r_px,
    compare their gradient direction to the expected radial direction, and
    return the fraction that align within a tolerance.

    Args:
        edges:         uint8 Canny edge image.
        grad_dir:      per-pixel gradient direction (radians) from Sobel.
        cx, cy:        candidate circle center in pixels.
        r_px:          candidate radius in pixels.
        radial_tol_px: thickness of the radial band around r_px.
        align_tol_deg: angle tolerance (degrees) for alignment.

    Returns:
        A float in [0, 1]: fraction of band pixels whose gradient aligns
        with the circle’s radial direction within align_tol_deg.
    """
    ys, xs = np.nonzero(edges)
    dx = xs - cx
    dy = ys - cy

    # Distance of each edge pixel from the candidate circle
    dist = np.hypot(dx, dy)
    band = np.abs(dist - r_px) <= radial_tol_px
    if not np.any(band):
        return 0.0

    # Expected radial direction for band pixels
    radial_theta = np.arctan2(dy[band], dx[band])

    # Observed gradient direction at those pixels
    g = grad_dir[ys[band], xs[band]]

    # Smallest absolute angular difference (wrap-aware)
    delta = np.abs(np.unwrap(radial_theta) - np.unwrap(g))
    delta = np.minimum(delta, 2 * np.pi - delta)

    # Count aligned pixels within the tolerance
    ok = (np.degrees(delta) <= align_tol_deg).sum()
    return float(ok) / float(np.count_nonzero(band))


def detect_rings(
    occ: np.ndarray,
    pixel_size: float,
    min_diam_m: float,
    max_diam_m: float,
    grad_align_tol_deg: float = 35.0,
    min_ringness: float = 0.35,
    min_arc_deg: float = 130.0,
    min_inliers_px: int = 30,
):
    """
    Detect circular rings in a rasterized occupancy map.

    Strategy:
      1) Build edges + gradient directions (edgeify).
      2) Get circle candidates from:
         - Hough transform (fast, coarse).
         - RANSAC over edges (robust; enforces min_arc_deg & inliers).
      3) Merge duplicate candidates in (cx, cy, r) space.
      4) Score each candidate with ringness and filter by min_ringness.
      5) Greedy NMS to prune near-duplicates.
      6) Return dicts with pixel + metric units.

    Args:
        occ:               uint8 occupancy/heatmap image.
        pixel_size:        metres per pixel (used to convert px → metres).
        min_diam_m:        minimum admissible diameter in metres.
        max_diam_m:        maximum admissible diameter in metres.
        grad_align_tol_deg:alignment tolerance for ringness scoring.
        min_ringness:      minimum ringness score to keep a candidate.
        min_arc_deg:       required angular coverage for RANSAC circles.
        min_inliers_px:    minimum inliers for RANSAC acceptance.

    Returns:
        List[dict]: each with keys:
          - "cx_px","cy_px","r_px"
          - "radius_m","diameter_m"
          - "ringness"
    """
    # Edges and gradient directions
    edges, grad_dir = edgeify(occ)

    # Candidate circles from two sources
    cand_a = hough_ring_candidates(edges, pixel_size, min_diam_m, max_diam_m)
    cand_b = ransac_circles_from_edges(
        edges,
        pixel_size,
        min_diam_m,
        max_diam_m,
        min_arc_deg=min_arc_deg,
        min_inliers_px=min_inliers_px,
    )

    # Unify candidate tuples to (cx, cy, r)
    cand = [(cx, cy, r) for (cx, cy, r) in cand_a] + [
        (cx, cy, r) for (cx, cy, r, _) in cand_b
    ]
    if not cand:
        return []

    # Merge close duplicates in parameter space
    merged = []
    used = np.zeros(len(cand), dtype=bool)
    for i, (cx, cy, r) in enumerate(cand):
        if used[i]:
            continue
        group = [(cx, cy, r)]
        for j in range(i + 1, len(cand)):
            if used[j]:
                continue
            cx2, cy2, r2 = cand[j]
            if math.hypot(cx - cx2, cy - cy2) < 5.0 and abs(r - r2) < 3.0:
                used[j] = True
                group.append((cx2, cy2, r2))
        g = np.array(group)
        merged.append(tuple(np.mean(g, axis=0)))
    cand = merged

    # Score via ringness and apply threshold
    dets = []
    for (cx, cy, r) in cand:
        ringness = score_ringness(
            edges,
            grad_dir,
            cx,
            cy,
            r,
            radial_tol_px=2.5,
            align_tol_deg=grad_align_tol_deg,
        )
        if ringness < min_ringness:
            continue
        dets.append((cx, cy, r, ringness))

    # Sort candidates by ringness (desc.)
    dets.sort(key=lambda t: t[3], reverse=True)

    # Greedy NMS over (cx, cy, r)
    final = []
    taken = np.zeros(len(dets), dtype=bool)
    for i, a in enumerate(dets):
        if taken[i]:
            continue
        final.append(a)
        for j in range(i + 1, len(dets)):
            if taken[j]:
                continue
            b = dets[j]
            if math.hypot(a[0] - b[0], a[1] - b[1]) < 6.0 and abs(a[2] - b[2]) < 3.0:
                taken[j] = True

    # Pack outputs with both pixel and metric values
    out = []
    for (cx, cy, r, ringness) in final:
        out.append(
            {
                "cx_px": cx,
                "cy_px": cy,
                "r_px": r,
                "radius_m": r * pixel_size,
                "diameter_m": 2.0 * r * pixel_size,
                "ringness": ringness,
            }
        )
    return out

def detect_rings_relaxed_roi(
    occ: np.ndarray,
    tf,
    pixel_size_m: float,
    center_xy_m: tuple[float, float],
    roi_radius_m: float,
    min_diam_m: float,
    max_diam_m: float,
    grad_tol_deg: float,
    min_ringness: float,
    expected_center_xy_m: tuple[float, float],
    expected_diam_m: float,
    center_gate_m: float = 0.20,
    diam_gate_frac: float = 0.15,
    top_k: int = 1,
    min_arc_deg: float = 130.0,
    min_inliers_px: int = 18,
):
    """
    Detect rings inside a local ROI centered near a predicted stem center.

    Steps:
      • Convert expected world center to pixel coords; crop a square ROI.
      • Run ring detection in the patch with relaxed thresholds.
      • Map detections back to global pixels → world meters.
      • Gate by center distance and diameter error relative to expectations.
      • Score by (ringness × proximity × diameter-consistency) and keep top_k.

    Args:
        occ:                Occupancy/heatmap image (uint8).
        tf:                 World↔image transform dict with keys {"xmin","ymax","pixel_size"}.
        pixel_size_m:       Metres per pixel.
        center_xy_m:        Approx world (x, y) to center the ROI on.
        roi_radius_m:       ROI radius in metres (converted to pixels internally).
        min_diam_m:         Minimum diameter allowed (m).
        max_diam_m:         Maximum diameter allowed (m).
        grad_tol_deg:       Gradient/ radial alignment tolerance for ringness (deg).
        min_ringness:       Minimum ringness to accept.
        expected_center_xy_m: Expected world (x, y) for gating.
        expected_diam_m:    Expected diameter for gating (m).
        center_gate_m:      Max allowed center deviation from expectation (m).
        diam_gate_frac:     Max allowed relative diameter error (fraction).
        top_k:              Number of top-scoring detections to return.
        min_arc_deg:        Minimum arc coverage for RANSAC (deg).
        min_inliers_px:     Minimum inliers for RANSAC acceptance (px).

    Returns:
        List[dict]: up to top_k detection dicts, sorted by descending "score".
    """
    # ROI crop in pixels
    cx_px, cy_px = world_point_to_pixel(tf, *center_xy_m)
    r_px = roi_radius_m / pixel_size_m
    patch, (x0, y0) = crop_roi(occ, cx_px, cy_px, r_px)

    # Detect within the patch using relaxed gates
    dets = detect_rings(
        patch,
        pixel_size=pixel_size_m,
        min_diam_m=min_diam_m,
        max_diam_m=max_diam_m,
        grad_align_tol_deg=grad_tol_deg,
        min_ringness=min_ringness,
        min_arc_deg=min_arc_deg,
        min_inliers_px=min_inliers_px,   # keep logic identical
    )

    kept = []
    ex, ey = expected_center_xy_m

    for d in dets:
        # Map patch-relative pixel coords back to full image pixels
        d["cx_px"] += x0
        d["cy_px"] += y0

        # Convert to world coordinates
        cx_m, cy_m = image_to_world(d["cx_px"], d["cy_px"], tf)
        d["center_x_m"] = cx_m
        d["center_y_m"] = cy_m

        # Center-distance gate
        dist = ((cx_m - ex) ** 2 + (cy_m - ey) ** 2) ** 0.5
        if dist > center_gate_m:
            continue

        # Diameter-consistency gate (relative error)
        diam_err = abs(d["diameter_m"] - expected_diam_m) / max(expected_diam_m, 1e-6)
        if diam_err > diam_gate_frac:
            continue

        # Composite score: ringness × proximity × diameter consistency
        score = d.get("ringness", 0.0) * (1.0 - dist / center_gate_m) * (1.0 - diam_err / diam_gate_frac)
        d["score"] = float(score)
        kept.append(d)

    kept.sort(key=lambda x: x["score"], reverse=True)
    return kept[:top_k]


def nms_circles(dets, pixel_thresh: float = 6.0):
    """
    Non-maximum suppression on circle detections in pixel space.

    Keeps the highest-score circle among detections whose centers are
    within 'pixel_thresh' of each other.

    Args:
        dets:          List of dicts with at least {"cx_px","cy_px","fill_ratio","compactness"}.
        pixel_thresh:  Max distance (pixels) between centers to consider a duplicate.

    Returns:
        List[dict]: pruned detections.
    """
    if not dets:
        return dets

    # Pre-score by shape quality (unchanged logic)
    for d in dets:
        d["score"] = d["fill_ratio"] * d["compactness"]

    dets = sorted(dets, key=lambda x: x["score"], reverse=True)

    kept = []
    taken = np.zeros(len(dets), dtype=bool)

    for i, a in enumerate(dets):
        if taken[i]:
            continue
        kept.append(a)

        # Suppress nearby lower-score centers
        for j in range(i + 1, len(dets)):
            if taken[j]:
                continue
            b = dets[j]
            dx = a["cx_px"] - b["cx_px"]
            dy = a["cy_px"] - b["cy_px"]
            if (dx * dx + dy * dy) ** 0.5 <= pixel_thresh:
                taken[j] = True

    return kept


def draw_debug(
    points_xy: np.ndarray,
    img_shape,
    pix: np.ndarray,
    dets,
    tf,
    out_png: Path
):
    """
    Render a debug overlay showing:
      • all rasterized points (gray),
      • detected circles (green outline),
      • detected centers (yellow dot),
      • label with diameter (cm) and ringness (if available).

    Args:
        points_xy:  (N,2) world XY points for the slice (unused here beyond 'pix').
        img_shape:  (H, W) of the target canvas.
        pix:        (N,2) pixel coords corresponding to points_xy.
        dets:       Iterable of detection dicts with keys:
                      "cx_px", "cy_px", "r_px", "diameter_m", optional "ringness".
        tf:         Transform dict (not used here, kept for signature consistency).
        out_png:    Output path for the rendered PNG.
    """
    H, W = img_shape
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # plot points as light-gray dots
    dots = np.rint(pix).astype(int)
    dots[:, 0] = np.clip(dots[:, 0], 0, W - 1)
    dots[:, 1] = np.clip(dots[:, 1], 0, H - 1)
    canvas[dots[:, 1], dots[:, 0]] = (180, 180, 180)

    # draw detections
    for d in dets:
        center = (int(round(d["cx_px"])), int(round(d["cy_px"])))
        cv2.circle(canvas, center, int(round(d["r_px"])), (0, 220, 0), 2)   # ring outline
        cv2.circle(canvas, center, 2, (0, 255, 255), -1)                    # center dot

        rn = d.get("ringness")
        if rn is not None:
            label = f"{d['diameter_m']*100:.0f} cm | r{rn:.2f}"
        else:
            label = f"{d['diameter_m']*100:.0f} cm"

        y = max(15, center[1] - 8)
        cv2.putText(
            canvas,
            label,
            (max(0, center[0] - 20), y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_png), canvas)

def per_slice_diam_bounds(
    height_m: float,
    dbh_min_m: float,
    dbh_max_m: float,
    bh_m: float = 1.5,
    taper_per_m: float = 0.15
) -> tuple[float, float]:
    """
    Height-aware diameter bounds from DBH limits using a simple linear taper model.

    Args:
        height_m:   Current slice height (m). If None, returns DBH bounds unchanged.
        dbh_min_m:  Min diameter at breast height (m).
        dbh_max_m:  Max diameter at breast height (m).
        bh_m:       Breast height reference (m), default 1.5 m.
        taper_per_m:Linear taper factor per vertical meter (fraction per m).

    Returns:
        (min_diam_m, max_diam_m) at the given height.
    """
    if height_m is None:
        return dbh_min_m, dbh_max_m

    # Positive if height below BH (→ larger diam), negative above BH (→ smaller diam)
    factor = 1.0 + taper_per_m * (bh_m - height_m)
    return dbh_min_m * factor, dbh_max_m * factor


def associate_and_update(
    tracks: List[Track],
    dets: List[dict],
    height_m: float,
    params: Params
):
    """
    Greedy nearest-neighbor association with diameter-jump gating, then track creation.

    Phase 1: For each existing track, pick the nearest unused detection (within distance
             gate) and accept it if diameter change is within allowed jump.
    Phase 2: For remaining detections, try to attach to any nearby track that passes
             the diameter-jump gate; otherwise start a new track.

    Args:
        tracks:   Existing track list (mutated in place).
        dets:     Current-slice detections (must contain center_x_m, center_y_m, diameter_m).
        height_m: Current slice height.
        params:   Params with gates: max_match_dist_m, max_diam_jump_frac.
    """
    used = [False] * len(dets)

    # --- Phase 1: one best match per track (mutual exclusion via 'used') ---
    for t in tracks:
        best_j, best_d = None, 1e9
        tx, ty = t.center

        # find nearest unused detection
        for j, d in enumerate(dets):
            if used[j]:
                continue
            dx = d["center_x_m"] - tx
            dy = d["center_y_m"] - ty
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_d:
                best_d = dist
                best_j = j

        # accept if within distance and diameter jump gates
        if best_j is not None and best_d <= params.max_match_dist_m:
            pred = t.diam_m[-1]
            if abs(dets[best_j]["diameter_m"] - pred) <= params.max_diam_jump_frac * pred:
                d = dets[best_j]
                t.centers_m.append((d["center_x_m"], d["center_y_m"]))
                t.diam_m.append(d["diameter_m"])
                t.heights_m.append(height_m)
                t.missed = 0
                used[best_j] = True
            else:
                t.missed += 1
        else:
            t.missed += 1

    # --- Phase 2: attach leftovers or spawn new tracks ---
    next_id = (max([tr.id for tr in tracks]) + 1) if tracks else 1

    for j, d in enumerate(dets):
        if used[j]:
            continue

        attached = False
        for t in tracks:
            tx, ty = t.center
            dist = ((d["center_x_m"] - tx) ** 2 + (d["center_y_m"] - ty) ** 2) ** 0.5

            if dist <= params.max_match_dist_m:
                pred = t.diam_m[-1]
                if abs(d["diameter_m"] - pred) <= params.max_diam_jump_frac * pred:
                    t.centers_m.append((d["center_x_m"], d["center_y_m"]))
                    t.diam_m.append(d["diameter_m"])
                    t.heights_m.append(height_m)
                    t.missed = 0
                    attached = True
                    break

        if attached:
            continue

        # start a new track
        tracks.append(
            Track(
                id=next_id,
                centers_m=[(d["center_x_m"], d["center_y_m"])],
                diam_m=[d["diameter_m"]],
                heights_m=[height_m],
                missed=0,
            )
        )
        next_id += 1


# ---------- Main-like helpers ----------
def parse_slice_height(name: str, pattern: str) -> float | None:
    m = re.match(pattern, name)
    if not m:
        return None
    return float(m.group("h"))

def process_slice(
    ply_path: Path,
    params: Params,
    tracks: list[Track],
    cache_occ_tf: dict,
    allow_global: bool = True,
):
    """
    Process a single slice_*.ply file:
      1) Load points, rasterize to a global heatmap.
      2) Run global ring detection with per-slice diameter bounds.
      3) Map detections to world coords, NMS in world, associate to tracks.
      4) For tracks that missed this slice, try a relaxed ROI recovery.
      5) Draw a debug overlay and return per-detection rows.

    Notes:
      - Does not alter detection logic or thresholds.
      - Uses GLOBAL_TF computed elsewhere (global raster frame).
    """
    # ---- Load and short-circuit on empty ----
    pts = load_ply_xyz_ascii(ply_path)
    if pts.size == 0:
        return []

    # ---- XY, global raster & occupancy ----
    xy = pts[:, :2]
    img_shape = (GLOBAL_TF["H"], GLOBAL_TF["W"])
    _, pix = world_to_image_fixed(xy, GLOBAL_TF)
    occ = rasterize_heatmap_fixed(xy, GLOBAL_TF)

    # ---- Per-slice bounds from height ----
    h = parse_slice_height(ply_path.name, params.slice_regex)
    minD, maxD = per_slice_diam_bounds(
        h,
        dbh_min_m=0.3592,
        dbh_max_m=0.5342,
        bh_m=1.5,
        taper_per_m=0.15,
    )
    cache_occ_tf[h] = (occ, GLOBAL_TF)

    # ---- Global ring detection (full slice) ----
    dets = detect_rings(
        occ,
        pixel_size=GLOBAL_TF["pixel_size"],
        min_diam_m=minD,
        max_diam_m=maxD,
        grad_align_tol_deg=params.global_grad_align_deg,
        min_ringness=params.global_min_ringness,
        min_arc_deg=params.global_min_arc_deg,
        min_inliers_px=params.global_min_inliers_px,  # <— NEW
    )

    # ---- Pixels → world coordinates ----
    for d in dets:
        cx_m, cy_m = image_to_world(d["cx_px"], d["cy_px"], GLOBAL_TF)
        d["center_x_m"] = cx_m
        d["center_y_m"] = cy_m

    # ---- NMS in world space + associate to tracks ----
    dets = nms_world(dets, min_dist_m=0.15, diam_tol_frac=0.20)
    associate_and_update(tracks, dets, h, params)

    # ---- Try relaxed ROI recovery for tracks that missed this height ----
    extra = []
    for t in tracks:
        # If this track already got an obs at height h, skip
        if t.heights_m[-1] == h:
            continue

        pred_d = t.diam_m[-1]
        dmin = max(minD, pred_d * (1.0 - params.max_diam_jump_frac))
        dmax = min(maxD, pred_d * (1.0 + params.max_diam_jump_frac))

        best = detect_rings_relaxed_roi(
            occ,
            GLOBAL_TF,
            GLOBAL_TF["pixel_size"],
            center_xy_m=t.center,
            roi_radius_m=params.roi_radius_m,
            min_diam_m=dmin,
            max_diam_m=dmax,
            grad_tol_deg=params.relaxed_grad_align_tol_deg,
            min_ringness=params.relaxed_min_ringness,
            expected_center_xy_m=t.center,
            expected_diam_m=pred_d,
            center_gate_m=params.max_match_dist_m,
            diam_gate_frac=params.max_diam_jump_frac,
            top_k=1,
            min_arc_deg=params.relaxed_min_arc_deg,
            min_inliers_px=params.relaxed_min_inliers_px,  # <— NEW
        )

        if best:
            best = nms_world(best, min_dist_m=0.15, diam_tol_frac=0.20)
            associate_and_update(tracks, best, h, params)
            extra.extend(best)

    # Include any ROI-recovered dets in the debug overlay
    dets.extend(extra)

    # ---- Debug render ----
    draw_debug(
        xy,
        img_shape,
        pix,
        dets,
        GLOBAL_TF,
        params.out_dir / (ply_path.stem + "_debug.png"),
    )

    # ---- Row output ----
    results = []
    for d in dets:
        results.append({
            "center_x_m": d["center_x_m"],
            "center_y_m": d["center_y_m"],
            "radius_m": d["radius_m"],
            "diameter_m": d["diameter_m"],
            "ringness": d.get("ringness"),
        })
    return results


# --- fixed global frame helpers ---
PIXEL_SIZE_M = 0.02
PADDING_M = 0.5


def compute_global_tf_from_slices(
    slices_dir: Path,
    pixel_size: float = PIXEL_SIZE_M,
    padding: float = PADDING_M,
):
    """
    Scan all slice_*.ply files to build a single global raster transform.

    Returns a dict with:
      - xmin, ymax: world-space origin of the raster frame
      - pixel_size: metres per pixel
      - W, H: raster width/height in pixels
    """
    xs_min, xs_max, ys_min, ys_max = [], [], [], []

    # Compute world-space bounds across all slices
    for p in sorted(slices_dir.glob("slice_*.ply")):
        pts = load_ply_xyz_ascii(p)
        if pts.size:
            xs, ys = pts[:, 0], pts[:, 1]
            xs_min.append(xs.min()); xs_max.append(xs.max())
            ys_min.append(ys.min()); ys_max.append(ys.max())

    if not xs_min:
        raise FileNotFoundError(f"No slice_*.ply in {slices_dir}")

    # Pad the bounding box and convert to pixel dimensions
    xmin = float(min(xs_min) - padding)
    xmax = float(max(xs_max) + padding)
    ymin = float(min(ys_min) - padding)
    ymax = float(max(ys_max) + padding)

    W = int(np.ceil((xmax - xmin) / pixel_size))
    H = int(np.ceil((ymax - ymin) / pixel_size))

    return {"xmin": xmin, "ymax": ymax, "pixel_size": pixel_size, "W": W, "H": H}


def world_to_image_fixed(xy: np.ndarray, tf: dict):
    """
    Map world XY points to pixel coordinates in a fixed global frame.

    Returns:
      - (H, W): image shape (rows, cols)
      - pix: Nx2 float32 array of [x_px, y_px]
    """
    px = (xy[:, 0] - tf["xmin"]) / tf["pixel_size"]
    py = (tf["ymax"] - xy[:, 1]) / tf["pixel_size"]
    return (tf["H"], tf["W"]), np.stack([px, py], axis=1).astype(np.float32)


def rasterize_heatmap_fixed(xy: np.ndarray, tf: dict) -> np.ndarray:
    """
    Make a normalized occupancy heatmap (uint8) in the fixed global frame.
    Each input point contributes +1 to its pixel bin, then the heatmap is blurred
    and scaled to [0, 255]. If empty, returns a zero image.
    """
    H, W = tf["H"], tf["W"]
    heat = np.zeros((H, W), dtype=np.float32)

    _, pix = world_to_image_fixed(xy, tf)
    ij = np.rint(pix).astype(np.int32)
    ij[:, 0] = np.clip(ij[:, 0], 0, W - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, H - 1)

    # Accumulate per-pixel counts
    np.add.at(heat, (ij[:, 1], ij[:, 0]), 1.0)

    # Smooth and normalize
    heat = cv2.GaussianBlur(heat, (0, 0), 1.2)
    if heat.max() > 0:
        heat = (255.0 * (heat / heat.max())).astype(np.uint8)
    else:
        heat = np.zeros_like(heat, dtype=np.uint8)

    return heat

# --------- run() entrypoint (refactor) ---------
def run(params: Params):
    """
    End-to-end driver for the circle-detection pipeline, per-slice.
      1) Build a global raster transform from all slices.
      2) Iterate slices in height order and write per-slice CSVs of detections.
      3) Bridge missing slices using relaxed ROI recovery.
      4) Write a stems_summary.csv of track statistics.
    """
    ensure_dir(params.out_dir)

    # 1) Global raster frame (used by all subsequent steps)
    global GLOBAL_TF
    GLOBAL_TF = compute_global_tf_from_slices(params.slices_dir)
    print("GLOBAL raster:", GLOBAL_TF["H"], "x", GLOBAL_TF["W"])

    # Collect slice files and sort by parsed height
    slice_files = sorted(
        (p for p in params.slices_dir.glob("slice_*.ply") if p.is_file()),
        key=lambda p: (parse_slice_height(p.name, params.slice_regex) or 0.0),
    )

    # Tracking state across slices
    tracks: List[Track] = []
    cache_occ_tf: dict = {}   # height_m -> (occupancy_image, tf)
    all_heights: List[float] = []

    # 2) Process each slice and write per-slice detections
    fieldnames = ["center_x_m", "center_y_m", "radius_m", "diameter_m", "ringness"]
    for ply in slice_files:
        h = parse_slice_height(ply.name, params.slice_regex)
        all_heights.append(h)

        rows = process_slice(
            ply,
            params=params,
            tracks=tracks,
            cache_occ_tf=cache_occ_tf,
            allow_global=True,
        )

        csv_path = params.out_dir / f"{ply.stem}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # 3) Attempt to fill gaps (heights without detections) along existing tracks
    heights_sorted = sorted(set(all_heights))
    bridge_missing_slices(tracks, heights_sorted, cache_occ_tf, params)

    # 4) Summarize tracks
    summary_path = params.out_dir / "stems_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "n_obs", "median_DBH_m", "mean_x_m", "mean_y_m"])
        for t in tracks:
            xs = [c[0] for c in t.centers_m]
            ys = [c[1] for c in t.centers_m]
            writer.writerow([t.id, len(t.diam_m), t.dbh_m, float(np.mean(xs)), float(np.mean(ys))])

    print(f"Wrote {len(tracks)} stem tracks → stems_summary.csv")

if __name__ == "__main__":
    run(Params(
        slices_dir=Path("sliced_ply_outputs_2/plot_annotations_ct03t1b_01"),
        out_dir=Path("circle_detections"),
        pixel_size_m=0.02,
    ))
