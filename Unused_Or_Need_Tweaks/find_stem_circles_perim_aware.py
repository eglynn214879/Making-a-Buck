# ================================================================
# circle_detector.py — Slice-wise ring detection + tracking for stems
# ---------------------------------------------------------------
# Purpose
#   Detect stem cross-sections in LiDAR slice_*.ply files, track them across
#   height, and export per-slice detections + a stems summary (with DBH).
#
# What it does
#   • Builds a fixed global XY raster over all slices (with padding).
#   • Rasterizes each slice to an occupancy/heatmap image.
#   • Detects rings (Hough + RANSAC) with strict global gates.
#   • Recovers misses via relaxed ROI/perimeter detection guided by tracks.
#   • Associates detections across heights by NN + diameter-jump gating.
#   • Writes per-slice CSVs + debug PNG overlays; computes per-track DBH
#     (median within 1.3–1.7 m; falls back to overall median if none).
#
# Inputs
#   • slices_dir: folder containing ASCII PLY slices named like: slice_<h>m.ply
#                 (height h parsed from filename; expects at least x y z columns)
#
# Outputs (under out_dir)
#   • slice_<h>m.csv             — center_x_m, center_y_m, radius_m, diameter_m, ringness
#   • slice_<h>m_debug.png       — visualization overlay of detected circles
#   • stems_summary.csv          — track_id, n_obs, median_DBH_m, mean_x_m, mean_y_m
#
# Configure
#   • Tune Params dataclass: pixel_size_m, padding_m, gates (global/relaxed/perimeter),
#     tracking radii, and inlier/coverage thresholds.
#
# Usage (programmatic)
#   from pathlib import Path
#   run(Params(
#       slices_dir=Path(".../slices"),
#       out_dir=Path(".../circle_detections"),
#       pixel_size_m=0.02,
#   ))
#
# Dependencies
#   Python 3.9+, numpy, opencv-python, dataclasses
#
# Author
#   Ethan Glynn (University of Sydney)
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
    # Input / Output
    slices_dir: Path   # Directory containing slice_*.ply files
    out_dir: Path      # Directory for output CSVs and debug images

    # Rasterization
    pixel_size_m: float = 0.02   # Meters per pixel in XY (adjust for LiDAR density)
    padding_m: float = 0.5       # Extra spatial border around the full point extent

    # Preprocessing
    gaussian_sigma_px: int = 3   # Gaussian blur for denoising before thresholding
    min_blob_area_px: int = 150  # Minimum connected component area (remove small specks)
    morph_open_px: int = 3       # Morphological opening radius to separate touching blobs

    # Detection constraints (in meters)
    min_diam_m: float = 0.26     # Minimum detectable stem diameter (≈36 cm)
    max_diam_m: float = 0.6      # Maximum detectable stem diameter (≈53 cm)
    min_fill_ratio: float = 0.45 # Minimum area ratio (blob / enclosing circle)
    min_circ_compactness: float = 0.65 # Minimum contour compactness (4πA/P²)

    # File naming pattern
    slice_regex: str = r"slice_(?P<h>[0-9]+\.[0-9]+)m\.ply"


@dataclass
class Params(Params):
    # --- Tracking / ROI parameters ---
    max_match_dist_m: float = 0.35          # Max distance (m) to associate detections between slices
    max_diam_jump_frac: float = 0.28        # Max fractional change in diameter between slices
    roi_radius_m: float = 0.55              # Radius (m) for ROI-based recovery around predicted center

    # Relaxed (ROI / tracking) gates
    relaxed_grad_align_tol_deg: float = 38.0  # Max gradient misalignment for relaxed detection
    relaxed_min_ringness: float = 0.30        # Minimum ringness score in relaxed mode
    relaxed_min_arc_deg: float = 110.0        # Minimum arc coverage for relaxed detection

    # Global (per-slice, full-image) gates
    global_grad_align_deg: float = 20.0       # Max gradient misalignment for global detection
    global_min_ringness: float = 0.50         # Minimum ringness score for global detection
    global_min_arc_deg: float = 130.0         # Minimum arc coverage (degrees) for global detection

    # RANSAC inlier requirements
    global_min_inliers_px: int = 30   # Min inliers for full-slice RANSAC
    relaxed_min_inliers_px: int = 18  # Min inliers for relaxed / ROI RANSAC

    # Perimeter relaxations
    perimeter_band_px: int = 6                 # Pixel distance from border considered “perimeter”
    perimeter_min_arc_deg: float = 85.0        # Minimum arc coverage near borders
    perimeter_min_inliers_px: int = 14         # Minimum inliers near borders
    perimeter_center_inside_bias: float = 1.15 # Bias factor (>1) if circle center lies fully inside image


@dataclass
class Track:
    """Stores the tracked properties of a single detected stem."""
    id: int
    centers_m: List[Tuple[float, float]]  # (x, y) coordinates in world space
    diam_m: List[float]                   # Diameter for each observation
    heights_m: List[float]                # Height (z) for each observation
    missed: int = 0                       # Count of consecutive missing detections

    @property
    def center(self) -> Tuple[float, float]:
        """Return the most recent stem center (x, y)."""
        return self.centers_m[-1]

    @property
    def dbh_m(self) -> float:
        """Return the median diameter at breast height (1.3–1.7 m)."""
        vals = [d for d, h in zip(self.diam_m, self.heights_m) if 1.3 <= h <= 1.7]
        vals = vals or self.diam_m  # Fallback: use all diameters if no 1.3–1.7 m data
        return float(np.median(vals))


# ---------- I/O ----------
def load_ply_xyz_ascii(filepath: Path) -> np.ndarray:
    """Reads an ASCII PLY file containing at least (x, y, z) columns."""
    with open(filepath, "r") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Invalid PLY: missing 'end_header'")
            if line.strip() == "end_header":
                break
        data = np.loadtxt(f, dtype=np.float32)

    if data.ndim == 1:
        if data.size < 3:
            return np.empty((0, 3), dtype=np.float32)
        data = data[None, :]

    return data[:, :3]  # Return only (x, y, z) columns


def ensure_dir(p: Path):
    """Ensure a directory exists (create parents if missing)."""
    p.mkdir(parents=True, exist_ok=True)


# ---------- Geometry helpers ----------
def circle_intersects_border(cx: float, cy: float, r: float, H: int, W: int, band: int = 6) -> bool:
    """
    Check whether a circle (centered at cx, cy with radius r) intersects
    any image border, considering an optional band (px) from the edge.
    """
    left   = cx
    right  = W - 1 - cx
    top    = cy
    bottom = H - 1 - cy
    return (left <= r + band) or (right <= r + band) or (top <= r + band) or (bottom <= r + band)


def world_to_image(points_xy: np.ndarray, pixel_size: float, padding_m: float):
    """
    Build a per-slice image frame and map world (x, y) → pixel (px, py).

    Returns
    -------
    (H, W) : tuple[int, int]
        Image height and width in pixels.
    tf : dict
        Minimal transform dict with:
          - "xmin": world x corresponding to pixel column 0
          - "ymax": world y corresponding to pixel row 0 (top)
          - "pixel_size": metres per pixel
    pix : np.ndarray, shape (N, 2), dtype float32
        Pixel coordinates corresponding to input points in the created frame.
    """
    x = points_xy[:, 0]
    y = points_xy[:, 1]

    # Expand world bounds with padding, then rasterize at pixel_size
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xmin -= padding_m; ymin -= padding_m
    xmax += padding_m; ymax += padding_m

    width_m  = max(xmax - xmin, pixel_size)
    height_m = max(ymax - ymin, pixel_size)

    W = int(np.ceil(width_m  / pixel_size))
    H = int(np.ceil(height_m / pixel_size))

    # y→py is flipped so that larger y is lower in image space
    px = (x - xmin) / pixel_size
    py = (ymax - y) / pixel_size
    pix = np.stack([px, py], axis=1).astype(np.float32)

    tf = {"xmin": xmin, "ymax": ymax, "pixel_size": pixel_size}
    return (H, W), tf, pix


def image_to_world(cx_px: float, cy_px: float, tf) -> tuple[float, float]:
    """
    Convert a pixel coordinate (cx_px, cy_px) back to world (x, y).

    Parameters
    ----------
    cx_px, cy_px : float
        Pixel coordinates in the frame defined by `tf`.
    tf : dict
        Transform dict from world_to_image (expects "xmin", "ymax", "pixel_size").

    Returns
    -------
    (x, y) : tuple[float, float]
        World-space coordinates in metres.
    """
    x = tf["xmin"] + cx_px * tf["pixel_size"]
    y = tf["ymax"] - cy_px * tf["pixel_size"]
    return float(x), float(y)


def world_point_to_pixel(tf, x: float, y: float) -> Tuple[float, float]:
    """
    Convert a world point (x, y) to pixel (px, py) using transform `tf`.

    Notes
    -----
    - Uses the same axis conventions as world_to_image/image_to_world.
    - py decreases as world y increases (image origin at top-left).
    """
    px = (x - tf["xmin"]) / tf["pixel_size"]
    py = (tf["ymax"] - y) / tf["pixel_size"]
    return float(px), float(py)


def crop_roi(img: np.ndarray, cx_px: float, cy_px: float, r_px: float):
    """
    Crop a square ROI from `img` centered at (cx_px, cy_px) with radius `r_px`.

    Parameters
    ----------
    img : np.ndarray
        Source image (H×W or H×W×C).
    cx_px, cy_px : float
        Center of the ROI in pixel coordinates.
    r_px : float
        Half-size of the square crop in pixels.

    Returns
    -------
    crop : np.ndarray
        Cropped view (clipped to image bounds).
    (x0, y0) : tuple[int, int]
        Top-left pixel offset of the crop within the original image.
    """
    H, W = img.shape[:2]

    # Clamp ROI to image bounds
    x0 = int(max(0, np.floor(cx_px - r_px)))
    y0 = int(max(0, np.floor(cy_px - r_px)))
    x1 = int(min(W, np.ceil(cx_px + r_px)))
    y1 = int(min(H, np.ceil(cy_px + r_px)))

    crop = img[y0:y1, x0:x1].copy()
    return crop, (x0, y0)


def nms_world(dets, min_dist_m=0.15, diam_tol_frac=0.20):
    """
    Non-maximum suppression in world space on circle detections.

    Strategy
    --------
    - Sort by 'ringness' (desc).
    - Keep a detection unless there exists a kept one:
        * whose center is within `min_dist_m`, AND
        * whose diameter differs by ≤ `diam_tol_frac` of the larger diameter.

    Parameters
    ----------
    dets : list[dict]
        Each dict must include 'center_x_m', 'center_y_m', 'diameter_m', and optionally 'ringness'.
    min_dist_m : float
        Minimum spatial separation (in metres) between kept centers.
    diam_tol_frac : float
        Relative diameter tolerance to consider two circles a duplicate.

    Returns
    -------
    kept : list[dict]
        Pruned detections.
    """
    dets = sorted(dets, key=lambda d: d.get("ringness", 0.0), reverse=True)
    kept = []
    for d in dets:
        keep = True
        for k in kept:
            dx = d["center_x_m"] - k["center_x_m"]
            dy = d["center_y_m"] - k["center_y_m"]
            if (dx*dx + dy*dy)**0.5 < min_dist_m:
                if abs(d["diameter_m"] - k["diameter_m"]) <= diam_tol_frac*max(d["diameter_m"], k["diameter_m"]):
                    keep = False
                    break
        if keep:
            kept.append(d)
    return kept


def bridge_missing_slices(tracks, heights_sorted, cache_occ_tf, params):
    """
    Fill in detection gaps for each track by attempting relaxed ROI recovery
    at intermediate slice heights (linear interpolation of center/diameter).

    Parameters
    ----------
    tracks : list[Track]
        Existing tracks with observations at some heights.
    heights_sorted : list[float]
        All slice heights, sorted ascending.
    cache_occ_tf : dict[float -> (occ, tf)]
        Cached per-height occupancy image and its transform.
    params : Params
        Detection/association configuration (uses relaxed thresholds).

    Notes
    -----
    For each consecutive observed pair (h1, h2), we:
      1) Find missing heights h with h1 < h < h2.
      2) Linearly interpolate expected (x, y, diameter) at h.
      3) Run detect_rings_relaxed_roi() near the expected center/diameter.
      4) If a candidate is found, associate_and_update() the track at height h.
    """
    for t in tracks:
        obs = sorted(zip(t.heights_m, t.centers_m, t.diam_m))
        for (h1,(x1,y1),d1), (h2,(x2,y2),d2) in zip(obs, obs[1:]):
            gaps = [h for h in heights_sorted if h1 < h < h2 and h not in t.heights_m]
            for h in gaps:
                occ, tf = cache_occ_tf[h]
                a  = (h - h1) / max(h2 - h1, 1e-6)
                ex = (1-a)*x1 + a*x2
                ey = (1-a)*y1 + a*y2
                ed = (1-a)*d1 + a*d2

                best = detect_rings_relaxed_roi(
                    occ, tf, params.pixel_size_m,
                    center_xy_m=(ex, ey), roi_radius_m=min(params.roi_radius_m, 0.30),
                    min_diam_m=ed*(1.0 - params.max_diam_jump_frac),
                    max_diam_m=ed*(1.0 + params.max_diam_jump_frac),
                    grad_tol_deg=params.relaxed_grad_align_tol_deg,
                    min_ringness=params.relaxed_min_ringness,
                    expected_center_xy_m=(ex, ey),
                    expected_diam_m=ed,
                    center_gate_m=min(params.max_match_dist_m, 0.20),
                    diam_gate_frac=min(params.max_diam_jump_frac, 0.12),
                    top_k=1
                )
                if best:
                    associate_and_update(tracks, best, h, params)


# ---------- Detection pipeline ----------
def rasterize(points_xy: np.ndarray, img_shape, pix: np.ndarray) -> np.ndarray:
    """
    Rasterize a set of world points (already mapped to pixel coords `pix`)
    into a binary occupancy image with the specified shape.

    Parameters
    ----------
    points_xy : np.ndarray
        Unused by the function body (kept for signature parity).
    img_shape : tuple[int, int]
        (H, W) of the output image.
    pix : np.ndarray, shape (N, 2)
        Pixel coordinates (float) to be rounded and splatted.

    Returns
    -------
    img : np.ndarray, dtype=uint8
        Binary image with 255 at occupied pixels, 0 elsewhere.
    """
    H, W = img_shape
    img = np.zeros((H, W), dtype=np.uint8)

    # Round to nearest pixel, clamp to frame bounds, then splat
    ij = np.rint(pix).astype(np.int32)
    ij[:, 0] = np.clip(ij[:, 0], 0, W - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, H - 1)
    img[ij[:, 1], ij[:, 0]] = 255
    return img

def rasterize_heatmap(points_xy: np.ndarray, img_shape, pix: np.ndarray) -> np.ndarray:
    """
    Convert sparse point samples (already mapped to pixel coords `pix`)
    into a smoothed occupancy *heatmap* (0–255).

    Parameters
    ----------
    points_xy : np.ndarray
        Unused (kept for signature symmetry with other rasterizers).
    img_shape : tuple[int, int]
        (H, W) for the output canvas.
    pix : np.ndarray, shape (N, 2)
        Pixel coordinates corresponding to the points.

    Returns
    -------
    heat : np.ndarray, dtype=uint8
        Smoothed density image scaled to 0–255. Zero when no points.
    """
    H, W = img_shape
    heat = np.zeros((H, W), dtype=np.float32)

    # Round to nearest pixel, clamp to bounds, and splat into a float canvas.
    ij = np.rint(pix).astype(np.int32)
    x = np.clip(ij[:, 0], 0, W - 1)
    y = np.clip(ij[:, 1], 0, H - 1)
    np.add.at(heat, (y, x), 1.0)

    # Light Gaussian to diffuse sparse hits into a soft density map.
    heat = cv2.GaussianBlur(heat, (0, 0), 1.2)

    # Normalize to 0–255 only if non-empty; otherwise keep zeros.
    if heat.max() > 0:
        heat = (255.0 * (heat / heat.max())).astype(np.uint8)
    else:
        heat = heat.astype(np.uint8)
    return heat


def preprocess(mask: np.ndarray, sigma: int, min_area: int, open_r: int) -> np.ndarray:
    """
    Clean a binary-ish occupancy mask before contour-based processing.

    Steps
    -----
    1) Optional Gaussian blur (denoise before threshold).
    2) Binary threshold at > 0.
    3) Optional morphological opening to separate/fix small connections.
    4) Remove connected components smaller than `min_area`.

    Parameters
    ----------
    mask : np.ndarray
        Input grayscale/binary image (uint8).
    sigma : int
        Std dev for Gaussian blur in pixels (0 disables).
    min_area : int
        Minimum connected-component area to keep (in pixels).
    open_r : int
        Radius for elliptical opening (0 disables).

    Returns
    -------
    cleaned : np.ndarray, dtype=uint8
        Cleaned binary mask.
    """
    if sigma > 0:
        # Kernel size ≈ 6*sigma rounded to odd.
        k = max(1, int(2 * round(3 * sigma) + 1))
        mask = cv2.GaussianBlur(mask, (k, k), sigmaX=sigma, sigmaY=sigma)

    # Anything >0 becomes foreground.
    _, bw = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    if open_r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_r + 1, 2 * open_r + 1))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)

    # Remove tiny components (noise specks).
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    cleaned = np.zeros_like(bw)
    for i in range(1, num):  # skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def edgeify(occ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Produce edges and per-pixel gradient direction from an occupancy image.

    Returns
    -------
    edges : np.ndarray, dtype=uint8
        Canny edges (0/255), tuned for thin, stable contours.
    grad_dir : np.ndarray, dtype=float32
        Gradient direction (radians) from Sobel gx/gy, same shape as `occ`.
        Use with arctan2(gy, gx). Helpful for ringness alignment checks.
    """
    # Gentle blur to stabilize edges.
    blurred = cv2.GaussianBlur(occ, (5, 5), 1.0)

    # Thin edges suitable for circle fitting / ringness scoring.
    edges = cv2.Canny(blurred, threshold1=20, threshold2=60, L2gradient=True)

    # Gradient direction (in radians) for alignment tests.
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad_dir = np.arctan2(gy, gx)
    return edges, grad_dir


def contour_compactness(contour: np.ndarray) -> float:
    """
    Compute the circular compactness metric for a contour.

    Definition
    ----------
    compactness = 4π * area / perimeter²
      • = 1.0 for a perfect circle
      • → 0.0 for highly elongated/irregular shapes

    Parameters
    ----------
    contour : np.ndarray
        OpenCV contour array (N×1×2).

    Returns
    -------
    float
        Compactness in [0, 1]. Returns 0.0 if perimeter is degenerate.
    """
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    if peri <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (peri * peri))

def detect_circles(mask: np.ndarray,
                   pixel_size: float,
                   min_diam_m: float,
                   max_diam_m: float,
                   min_fill_ratio: float,
                   min_compactness: float):
    """
    Fast blob-based circle proposals from a *binary* mask.

    Pipeline
    --------
    1) Find external contours.
    2) Fit a min-enclosing circle for each contour.
    3) Compute:
         • fill_ratio = (blob area) / (enclosing-circle area)
         • compactness = 4πA / P²  (≈1 for true circles)
    4) Apply diameter/fill/compactness gates in *metric* units via `pixel_size`.
    5) Lightweight NMS in pixel space to suppress duplicates.

    Returns
    -------
    dets : List[dict]
        Each with:
          - cx_px, cy_px, r_px
          - radius_m, diameter_m
          - fill_ratio, compactness
    """
    dets = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        if len(c) < 5:
            continue
        (cx_px, cy_px), r_px = cv2.minEnclosingCircle(c)
        circle_area_px = np.pi * (r_px ** 2)
        blob_area_px = cv2.contourArea(c)
        if circle_area_px <= 1.0:
            continue

        fill_ratio = float(blob_area_px / circle_area_px)
        compact = contour_compactness(c)

        # Convert to metres for diameter gating.
        r_m = r_px * pixel_size
        d_m = 2.0 * r_m

        # Hard gates in physical units + blob quality checks.
        if d_m < min_diam_m or d_m > max_diam_m:
            continue
        if fill_ratio < min_fill_ratio:
            continue
        if compact < min_compactness:
            continue

        dets.append({
            "cx_px": cx_px, "cy_px": cy_px, "r_px": r_px,
            "radius_m": r_m, "diameter_m": d_m,
            "fill_ratio": fill_ratio, "compactness": compact
        })

    # Suppress near-duplicates in image space.
    dets = nms_circles(dets, pixel_thresh=6.0)
    return dets


# Fixed seed for deterministic RANSAC sampling across runs.
rng = np.random.default_rng(1234)


def hough_ring_candidates(edges: np.ndarray,
                          pixel_size: float,
                          min_diam_m: float, max_diam_m: float,
                          minDist_px: float = 12.0):
    """
    Proposal stage using OpenCV's HoughCircles on a thin edge map.

    Notes
    -----
    - Radius bounds are derived from (min_diam_m, max_diam_m) and `pixel_size`.
    - Returns (x, y, r) in *pixel* units; filtering is deferred to later stages.

    Returns
    -------
    out : List[Tuple[float, float, float]]
        Candidate circle centers and radii in pixels.
    """
    H, W = edges.shape
    minR = max(3, int((min_diam_m / pixel_size) / 2.0))
    maxR = int((max_diam_m / pixel_size) / 2.0)
    if maxR <= minR:
        return []

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.0,
                               minDist=minDist_px,
                               param1=120, param2=18,
                               minRadius=minR, maxRadius=maxR)

    out = []
    if circles is not None:
        for c in circles[0]:
            x, y, r = float(c[0]), float(c[1]), float(c[2])
            if 0 <= x < W and 0 <= y < H:
                out.append((x, y, r))
    return out


def ransac_circles_from_edges(edges: np.ndarray,
                              pixel_size: float,
                              min_diam_m: float, max_diam_m: float,
                              trials: int = 1800,
                              inlier_tol_px: float = 2.8,
                              min_arc_deg: float = 130.0,
                              subsample: int = 9000,
                              min_inliers_px: int = 30):   # <— NEW
    """
    Robust circle proposals via 3-point RANSAC on edge pixels.

    Strategy
    --------
    - Optionally subsample edge points for speed.
    - Loop:
        * Sample 3 points → define circle (skip degenerate).
        * Score by inlier count (|d - r| <= tol) and angular coverage.
        * Keep proposals within radius bounds derived from metric limits.
    - Sort by score and suppress near-duplicates.

    Parameters
    ----------
    trials : int
        RANSAC iterations.
    inlier_tol_px : float
        Radial residual tolerance in pixels.
    min_arc_deg : float
        Required angular coverage from inliers (0–360°).
    subsample : int
        Max number of edge points used in fitting.
    min_inliers_px : int
        Minimum inlier count to accept a hypothesis.

    Returns
    -------
    kept : List[Tuple[float, float, float, float]]
        (cx, cy, r, score) in pixel units, sorted best-first.
    """
    ys, xs = np.nonzero(edges)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    if pts.shape[0] == 0:
        return []
    if pts.shape[0] > subsample:
        idx = rng.choice(pts.shape[0], size=subsample, replace=False)
        pts = pts[idx]

    minR = max(3, int((min_diam_m / pixel_size) / 2.0))
    maxR = int((max_diam_m / pixel_size) / 2.0)

    best = []
    for _ in range(trials):
        i, j, k = rng.choice(len(pts), size=3, replace=False)
        p1, p2, p3 = pts[i], pts[j], pts[k]

        # Solve circle through 3 points (linearized form).
        A = 2 * np.array([[p2[0]-p1[0], p2[1]-p1[1]],
                          [p3[0]-p1[0], p3[1]-p1[1]]], dtype=np.float32)
        b = np.array([[p2[0]**2 - p1[0]**2 + p2[1]**2 - p1[1]**2],
                      [p3[0]**2 - p1[0]**2 + p3[1]**2 - p1[1]**2]], dtype=np.float32)
        det = np.linalg.det(A)
        if abs(det) < 1e-6:
            continue

        center = np.linalg.solve(A, b).ravel()
        cx, cy = center[0], center[1]
        r = np.hypot(p1[0]-cx, p1[1]-cy)
        if not (minR <= r <= maxR):
            continue

        # Inliers by radial residual.
        d = np.hypot(pts[:,0]-cx, pts[:,1]-cy)
        resid = np.abs(d - r)
        inliers = resid <= inlier_tol_px
        n_in = int(inliers.sum())
        if n_in < min_inliers_px:
            continue

        # Angular coverage from inlier angles (penalize partial arcs).
        angles = np.degrees(np.arctan2(pts[inliers,1]-cy, pts[inliers,0]-cx))
        ang_sorted = np.sort((angles + 360) % 360)
        diffs = np.diff(np.r_[ang_sorted, ang_sorted[0]+360])
        coverage = 360.0 - float(diffs.max())
        if coverage < min_arc_deg:
            continue

        score = n_in * (coverage / 360.0)
        best.append((cx, cy, r, score))

    # Rank by score and NMS in (cx, cy, r)-space.
    best.sort(key=lambda t: t[3], reverse=True)
    kept = []
    taken = np.zeros(len(best), dtype=bool)
    for a_idx, a in enumerate(best):
        if taken[a_idx]:
            continue
        kept.append(a)
        for b_idx in range(a_idx+1, len(best)):
            if taken[b_idx]:
                continue
            b = best[b_idx]
            if math.hypot(a[0]-b[0], a[1]-b[1]) < 6.0 and abs(a[2]-b[2]) < 3.0:
                taken[b_idx] = True
    return kept


def score_ringness(edges: np.ndarray, grad_dir: np.ndarray, cx: float, cy: float, r_px: float,
                   radial_tol_px: float = 2.5, align_tol_deg: float = 30.0):
    """
    Fraction of edge pixels on the ring band whose gradient aligns with the
    *radial* direction from (cx, cy). Higher is "more ring-like".

    Method
    ------
    - Select edge pixels within ±`radial_tol_px` of radius `r_px`.
    - For each, compute:
        radial_theta = atan2(y - cy, x - cx)
        grad_dir[y, x] from Sobel (via `edgeify`)
        alignment = |radial_theta - grad_dir| in degrees
    - Return (#aligned ≤ align_tol_deg) / (#band pixels).

    Returns
    -------
    float
        Proportion in [0, 1]. Returns 0 when there are no valid band pixels.
    """
    H, W = edges.shape
    ys, xs = np.nonzero(edges)

    # Distance of each edge pixel to the hypothesized circle.
    dx = xs - cx
    dy = ys - cy
    dist = np.hypot(dx, dy)

    # Keep a thin annulus around r_px.
    band = np.abs(dist - r_px) <= radial_tol_px
    if not np.any(band):
        return 0.0

    # Require local neighbourhood to be in-bounds (avoid sampling off-image).
    x_ok = (xs[band] - radial_tol_px >= 0) & (xs[band] + radial_tol_px < W)
    y_ok = (ys[band] - radial_tol_px >= 0) & (ys[band] + radial_tol_px < H)
    vis = x_ok & y_ok
    if not np.any(vis):
        return 0.0

    # Compute alignment between radial direction and gradient direction.
    bx = xs[band][vis]
    by = ys[band][vis]
    radial_theta = np.arctan2(by - cy, bx - cx)
    g = grad_dir[by, bx]

    # Smallest angular difference in [0, π].
    delta = np.abs(np.unwrap(radial_theta) - np.unwrap(g))
    delta = np.minimum(delta, 2*np.pi - delta)

    ok = (np.degrees(delta) <= align_tol_deg).sum()
    return float(ok) / float(bx.size)


def detect_rings(occ: np.ndarray,
                 pixel_size: float,
                 min_diam_m: float, max_diam_m: float,
                 grad_align_tol_deg: float = 35.0,
                 min_ringness: float = 0.35,
                 min_arc_deg: float = 130.0,
                 min_inliers_px: int = 30):
    """
    Detect circular ring structures in a slice occupancy image.

    Approach
    --------
    1) Build an edge map + gradient directions (Sobel).
    2) Generate proposals:
         • HoughCircles (quick/rough)
         • RANSAC over edges (robust, uses arc coverage & inliers)
    3) Merge near-duplicate proposals in pixel space.
    4) Score each ring with `score_ringness` (gradient alignment on a narrow band).
       • Apply relaxed thresholds for perimeter-touching circles.
    5) NMS over (cx, cy, r), return best candidates with metric radii.

    Parameters
    ----------
    occ : np.ndarray
        8-bit occupancy/heatmap image of a slice.
    pixel_size : float
        Metres per pixel (used to convert r_px → r_m).
    min_diam_m, max_diam_m : float
        Physical diameter gates for the circle search.
    grad_align_tol_deg : float
        Max allowed angle difference (radial vs gradient) in `score_ringness`.
    min_ringness : float
        Minimum ringness score to accept a detection.
    min_arc_deg : float
        Required arc coverage for RANSAC candidates (0–360).
    min_inliers_px : int
        Minimum inlier count for RANSAC hypotheses.

    Returns
    -------
    List[dict]
        Each dict has:
          - "cx_px", "cy_px", "r_px"
          - "radius_m", "diameter_m"
          - "ringness"
    """
    edges, grad_dir = edgeify(occ)
    H, W = edges.shape

    # Proposals from two sources: Hough (fast) + RANSAC (robust).
    cand_a = hough_ring_candidates(edges, pixel_size, min_diam_m, max_diam_m)
    cand_b = ransac_circles_from_edges(
        edges, pixel_size, min_diam_m, max_diam_m,
        min_arc_deg=min_arc_deg, min_inliers_px=min_inliers_px
    )
    cand = [(cx, cy, r) for (cx, cy, r) in cand_a] + [(cx, cy, r) for (cx, cy, r, _) in cand_b]
    if not cand:
        return []

    # Merge close duplicates (center & radius proximity).
    merged = []
    used = np.zeros(len(cand), dtype=bool)
    for i, (cx, cy, r) in enumerate(cand):
        if used[i]:
            continue
        group = [(cx, cy, r)]
        for j in range(i+1, len(cand)):
            if used[j]:
                continue
            cx2, cy2, r2 = cand[j]
            if math.hypot(cx-cx2, cy-cy2) < 5.0 and abs(r-r2) < 3.0:
                used[j] = True
                group.append((cx2, cy2, r2))
        g = np.array(group)
        merged.append(tuple(np.mean(g, axis=0)))
    cand = merged

    dets = []
    for (cx, cy, r) in cand:
        # Perimeter-aware gating: relax requirements if ring touches image edge.
        is_perim = circle_intersects_border(cx, cy, r, H, W, band=Params.perimeter_band_px if hasattr(Params, 'perimeter_band_px') else 6)
        local_min_arc = min_arc_deg
        local_min_inl = min_inliers_px
        local_min_ring = min_ringness

        if is_perim:
            local_min_arc = getattr(Params, 'perimeter_min_arc_deg', 90.0)
            local_min_inl = getattr(Params, 'perimeter_min_inliers_px', 14)
            # Optionally relax ringness slightly for border cases:
            # local_min_ring = max(0.25, min_ringness - 0.05)

        # Score "how ring-like" using gradient alignment in a thin annulus.
        # (Acts as a lightweight proxy instead of re-running RANSAC per candidate.)
        ringness = score_ringness(edges, grad_dir, cx, cy, r, radial_tol_px=2.5, align_tol_deg=grad_align_tol_deg)
        if ringness < local_min_ring:
            continue

        # Small bias if center lies strictly inside the image.
        center_inside = (0 <= cx < W) and (0 <= cy < H)
        bias = getattr(Params, 'perimeter_center_inside_bias', 1.0) if center_inside else 1.0

        dets.append((cx, cy, r, ringness * bias))

    # Rank by score (ringness × bias).
    dets.sort(key=lambda t: t[3], reverse=True)

    # Final NMS in (cx, cy, r)-space.
    final = []
    taken = np.zeros(len(dets), dtype=bool)
    for i, a in enumerate(dets):
        if taken[i]:
            continue
        final.append(a)
        for j in range(i+1, len(dets)):
            if taken[j]:
                continue
            b = dets[j]
            if math.hypot(a[0]-b[0], a[1]-b[1]) < 6.0 and abs(a[2]-b[2]) < 3.0:
                taken[j] = True

    # Convert radii to metric units and package fields.
    out = []
    for (cx, cy, r, ringness) in final:
        out.append({
            "cx_px": cx, "cy_px": cy, "r_px": r,
            "radius_m": r * pixel_size,
            "diameter_m": 2.0 * r * pixel_size,
            "ringness": ringness
        })
    return out


def detect_rings_relaxed_roi(occ: np.ndarray, tf, pixel_size_m: float,
                             center_xy_m: tuple[float, float], roi_radius_m: float,
                             min_diam_m: float, max_diam_m: float,
                             grad_tol_deg: float, min_ringness: float,
                             expected_center_xy_m: tuple[float, float],
                             expected_diam_m: float,
                             center_gate_m: float = 0.20,
                             diam_gate_frac: float = 0.15,
                             top_k: int = 1,
                             min_arc_deg: float = 130.0,
                             min_inliers_px: int = 18):
    """
    ROI-limited re-detection around a predicted track position.

    Steps
    -----
    1) Crop a square patch around `center_xy_m` with radius `roi_radius_m`.
    2) Run `detect_rings` inside the patch with (typically) relaxed gates.
    3) Gate candidates against expected center/diameter (track prediction).
    4) Score by ringness × proximity × diameter agreement; return top-K.

    Returns
    -------
    List[dict]
        Top-K detections with pixel + metric fields and a "score".
    """
    cx_px, cy_px = world_point_to_pixel(tf, *center_xy_m)
    r_px = roi_radius_m / pixel_size_m
    patch, (x0, y0) = crop_roi(occ, cx_px, cy_px, r_px)

    # Reuse the main detector on the cropped patch.
    dets = detect_rings(
        patch,
        pixel_size=pixel_size_m,
        min_diam_m=min_diam_m,
        max_diam_m=max_diam_m,
        grad_align_tol_deg=grad_tol_deg,
        min_ringness=min_ringness,
        min_arc_deg=min_arc_deg,
        min_inliers_px=min_inliers_px,   # ROI-specific inlier floor
    )

    kept = []
    ex, ey = expected_center_xy_m
    for d in dets:
        # Translate patch coordinates back to full image space.
        d["cx_px"] += x0; d["cy_px"] += y0
        cx_m, cy_m = image_to_world(d["cx_px"], d["cy_px"], tf)
        d["center_x_m"] = cx_m; d["center_y_m"] = cy_m

        # Track-consistency gates: center distance + diameter agreement.
        dist = ((cx_m - ex)**2 + (cy_m - ey)**2) ** 0.5
        if dist > center_gate_m:
            continue
        diam_err = abs(d["diameter_m"] - expected_diam_m) / max(expected_diam_m, 1e-6)
        if diam_err > diam_gate_frac:
            continue

        # Composite score: prefer ring-like, near-by, and diameter-consistent.
        score = d.get("ringness", 0.0) * (1.0 - dist/center_gate_m) * (1.0 - diam_err/diam_gate_frac)
        d["score"] = float(score)
        kept.append(d)

    kept.sort(key=lambda x: x["score"], reverse=True)
    return kept[:top_k]


def nms_circles(dets, pixel_thresh=6.0):
    """
    Non-maximum suppression for blob-based circle proposals (image space).

    - Sort by a simple quality proxy: fill_ratio × compactness.
    - Suppress proposals whose centers are within `pixel_thresh` of a better one.

    Parameters
    ----------
    dets : List[dict]
        Each dict must include:
          - "cx_px", "cy_px"
          - "fill_ratio", "compactness"

    Returns
    -------
    List[dict]
        Filtered detections, highest quality first, with an added "score".
    """
    if not dets:
        return dets

    # Rank by blob quality proxy.
    for d in dets:
        d["score"] = d["fill_ratio"] * d["compactness"]
    dets = sorted(dets, key=lambda x: x["score"], reverse=True)

    kept = []
    taken = np.zeros(len(dets), dtype=bool)
    for i, a in enumerate(dets):
        if taken[i]:
            continue
        kept.append(a)
        for j in range(i + 1, len(dets)):
            if taken[j]:
                continue
            b = dets[j]
            dx = a["cx_px"] - b["cx_px"]
            dy = a["cy_px"] - b["cy_px"]
            if (dx * dx + dy * dy) ** 0.5 <= pixel_thresh:
                taken[j] = True
    return kept

def draw_debug(points_xy: np.ndarray, img_shape, pix: np.ndarray, dets, tf, out_png: Path):
    """
    Render a per-slice debug overlay.

    Layers
    ------
    • grey dots: rasterized slice points (XY projected)
    • green circles: detected rings (r_px)
    • yellow dot: ring center
    • label: "<diameter_cm> cm | r<ringness>"

    Parameters
    ----------
    points_xy : np.ndarray
        Original slice points (N×2) in world metres (x, y).
    img_shape : Tuple[int, int]
        (H, W) for the canvas.
    pix : np.ndarray
        Pixel-space coordinates for points_xy (N×2).
    dets : List[dict]
        Detections with "cx_px", "cy_px", "r_px", "diameter_m", (optional) "ringness".
    tf : dict
        World↔image transform (unused here except kept for symmetry with other fns).
    out_png : Path
        Output path for the PNG overlay.
    """
    H, W = img_shape
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # Plot points as grey pixels
    dots = np.rint(pix).astype(int)
    dots[:, 0] = np.clip(dots[:, 0], 0, W - 1)
    dots[:, 1] = np.clip(dots[:, 1], 0, H - 1)
    canvas[dots[:, 1], dots[:, 0]] = (180, 180, 180)

    # Draw each detection: circle + center + label
    for d in dets:
        center = (int(round(d["cx_px"])), int(round(d["cy_px"])))
        cv2.circle(canvas, center, int(round(d["r_px"])), (0, 220, 0), 2)   # ring
        cv2.circle(canvas, center, 2, (0, 255, 255), -1)                   # center dot

        rn = d.get("ringness")
        if rn is not None:
            label = f"{d['diameter_m']*100:.0f} cm | r{rn:.2f}"
        else:
            label = f"{d['diameter_m']*100:.0f} cm"

        # Keep label inside the image near the center
        y = max(15, center[1] - 8)
        cv2.putText(canvas, label, (max(0, center[0]-20), y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_png), canvas)


def per_slice_diam_bounds(height_m: float,
                          dbh_min_m: float, dbh_max_m: float,
                          bh_m: float = 1.5, taper_per_m: float = 0.15) -> tuple[float, float]:
    """
    Compute diameter bounds for a slice based on simple linear taper.

    Idea
    ----
    Scale the allowable DBH range by a linear taper factor relative to breast height.

    Parameters
    ----------
    height_m : float
        Slice height in metres. If None, returns raw DBH bounds.
    dbh_min_m, dbh_max_m : float
        Min/max diameter (m) around DBH for the species/site.
    bh_m : float
        Breast height (m), default 1.5 m.
    taper_per_m : float
        Linear taper factor per metre (unitless fraction / m).

    Returns
    -------
    (float, float)
        (min_diam_m, max_diam_m) for this slice height.
    """
    if height_m is None:
        return dbh_min_m, dbh_max_m
    factor = 1.0 + taper_per_m * (bh_m - height_m)
    return dbh_min_m * factor, dbh_max_m * factor


def associate_and_update(tracks: List[Track], dets: List[dict], height_m: float, params: Params):
    """
    Greedy nearest-neighbour association with gating, then track update/spawn.

    Steps
    -----
    1) For each existing track, pick the nearest unused detection within
       `max_match_dist_m` and with diameter jump <= `max_diam_jump_frac`.
       • If matched: append (center, diameter, height), reset missed.
       • Else: missed += 1
    2) For remaining unused detections, try to attach to any viable track
       (same gates). If none, start a new track.

    Notes
    -----
    - Greedy order is track-then-best-remaining; good enough for moderate densities.
    - Distance in XY metres; diameter gating prevents cross-assignments on crossings.
    """
    used = [False]*len(dets)

    # Pass 1: extend existing tracks (one best detection per track)
    for t in tracks:
        best_j, best_d = None, 1e9
        tx, ty = t.center
        for j, d in enumerate(dets):
            if used[j]:
                continue
            dx = d["center_x_m"] - tx
            dy = d["center_y_m"] - ty
            dist = (dx*dx + dy*dy)**0.5
            if dist < best_d:
                best_d = dist; best_j = j

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

    # Pass 2: attach remaining detections to any viable track, else spawn
    next_id = (max([tr.id for tr in tracks]) + 1) if tracks else 1
    for j, d in enumerate(dets):
        if used[j]:
            continue

        attached = False
        for t in tracks:
            tx, ty = t.center
            dist = ((d["center_x_m"]-tx)**2 + (d["center_y_m"]-ty)**2)**0.5
            if dist <= params.max_match_dist_m:
                pred = t.diam_m[-1]
                if abs(d["diameter_m"] - pred) <= params.max_diam_jump_frac*pred:
                    t.centers_m.append((d["center_x_m"], d["center_y_m"]))
                    t.diam_m.append(d["diameter_m"])
                    t.heights_m.append(height_m)
                    t.missed = 0
                    attached = True
                    break

        if attached:
            continue

        # Start a fresh track for unclaimed detection
        tracks.append(Track(
            id=next_id,
            centers_m=[(d["center_x_m"], d["center_y_m"])],
            diam_m=[d["diameter_m"]],
            heights_m=[height_m],
            missed=0
        ))
        next_id += 1


# ---------- Main-like helpers ----------
def parse_slice_height(name: str, pattern: str) -> float | None:
    """
    Extract numeric height from a slice filename by regex with named group 'h'.

    Example
    -------
    pattern: r"slice_(?P<h>[0-9]+\.[0-9]+)m\.ply"
    name   : "slice_1.40m.ply" → 1.40
    """
    m = re.match(pattern, name)
    if not m:
        return None
    return float(m.group("h"))


def process_slice(ply_path: Path, params: Params, tracks: list[Track],
                  cache_occ_tf: dict, allow_global: bool = True):
    """
    Process one slice file:
      • load points → global rasterize → global detection
      • world-pack outputs → NMS → associate
      • ROI re-detect around un-updated tracks → associate
      • draw overlay and return per-slice rows for CSV

    Parameters
    ----------
    ply_path : Path
        Path to ASCII PLY slice "slice_<h>m.ply".
    params : Params
        All algorithm/config parameters.
    tracks : list[Track]
        Mutable list of active tracks (updated in-place).
    cache_occ_tf : dict
        Cache mapping {height_m: (occ, GLOBAL_TF)} for later bridging.
    allow_global : bool
        Hook for toggling the global pass (kept for symmetry/future).

    Returns
    -------
    List[dict]
        Per-slice detections with world-space center, radius/diameter, ringness.
    """
    pts = load_ply_xyz_ascii(ply_path)
    if pts.size == 0:
        return []

    # Project to global raster frame (precomputed for the whole plot)
    xy = pts[:, :2]
    img_shape = (GLOBAL_TF["H"], GLOBAL_TF["W"])
    _, pix = world_to_image_fixed(xy, GLOBAL_TF)
    occ = rasterize_heatmap_fixed(xy, GLOBAL_TF)

    # Height → per-slice diameter gates via taper model
    h = parse_slice_height(ply_path.name, params.slice_regex)
    minD, maxD = per_slice_diam_bounds(h, dbh_min_m=0.3592, dbh_max_m=0.5342, bh_m=1.5, taper_per_m=0.15)
    cache_occ_tf[h] = (occ, GLOBAL_TF)

    # Global detection (full image) with stricter gates
    dets = detect_rings(
        occ,
        pixel_size=GLOBAL_TF["pixel_size"],
        min_diam_m=minD, max_diam_m=maxD,
        grad_align_tol_deg = params.global_grad_align_deg,
        min_ringness       = params.global_min_ringness,
        min_arc_deg        = params.global_min_arc_deg,
        min_inliers_px=params.global_min_inliers_px,   # stricter for global
    )

    # Convert pixel centers to world metres and suppress near-duplicates
    for d in dets:
        cx_m, cy_m = image_to_world(d["cx_px"], d["cy_px"], GLOBAL_TF)
        d["center_x_m"] = cx_m; d["center_y_m"] = cy_m
    dets = nms_world(dets, min_dist_m=0.15, diam_tol_frac=0.20)

    # Track association for global detections
    associate_and_update(tracks, dets, h, params)

    # ROI recovery for tracks that did not get an update at this height
    extra = []
    for t in tracks:
        # Skip tracks already updated at this height
        if t.heights_m[-1] == h:
            continue

        # Predict local diameter bounds around last observation (guard jumps)
        pred_d = t.diam_m[-1]
        dmin = max(minD, pred_d*(1.0 - params.max_diam_jump_frac))
        dmax = min(maxD, pred_d*(1.0 + params.max_diam_jump_frac))

        # ROI-limited relaxed search near the predicted center
        best = detect_rings_relaxed_roi(
            occ, GLOBAL_TF, GLOBAL_TF["pixel_size"],
            center_xy_m=t.center, roi_radius_m=params.roi_radius_m,
            min_diam_m=dmin, max_diam_m=dmax,
            grad_tol_deg=params.relaxed_grad_align_tol_deg,
            min_ringness=params.relaxed_min_ringness,
            expected_center_xy_m=t.center, expected_diam_m=pred_d,
            center_gate_m=params.max_match_dist_m,
            diam_gate_frac=params.max_diam_jump_frac, top_k=1,
            min_arc_deg=params.relaxed_min_arc_deg,
            min_inliers_px=params.relaxed_min_inliers_px,    # relaxed for ROI
        )

        if best:
            # World-space NMS + association for ROI hits
            best = nms_world(best, min_dist_m=0.15, diam_tol_frac=0.20)
            associate_and_update(tracks, best, h, params)
            extra.extend(best)

    # Aggregate (global + ROI) for this slice's overlay + CSV
    dets.extend(extra)
    draw_debug(xy, img_shape, pix, dets, GLOBAL_TF, params.out_dir / (ply_path.stem + "_debug.png"))

    # Emit compact rows for CSV writer
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
PIXEL_SIZE_M = 0.02   # Default global raster resolution in metres/pixel
PADDING_M    = 0.5    # Extra border (m) around min/max XY extents when framing all slices


def compute_global_tf_from_slices(slices_dir: Path, pixel_size=PIXEL_SIZE_M, padding=PADDING_M):
    """
    Build a *global* image frame that comfortably contains **all** slice point clouds.

    For every "slice_*.ply" file:
      • read ASCII PLY → collect min/max of x and y
      • expand bounds by `padding`
      • convert span to pixel dimensions using `pixel_size`

    Parameters
    ----------
    slices_dir : Path
        Directory containing "slice_*.ply" files.
    pixel_size : float
        Metres per pixel for the global raster.
    padding : float
        Extra margin (m) added on every side of the XY extent.

    Returns
    -------
    dict
        {"xmin", "ymax", "pixel_size", "W", "H"} describing the world→image transform.
        - Image X axis grows with +x; Y axis grows downward from ymax.
    """
    xs_min, xs_max, ys_min, ys_max = [], [], [], []
    for p in sorted(slices_dir.glob("slice_*.ply")):
        pts = load_ply_xyz_ascii(p)
        if pts.size:
            xs, ys = pts[:,0], pts[:,1]
            xs_min.append(xs.min()); xs_max.append(xs.max())
            ys_min.append(ys.min()); ys_max.append(ys.max())
    if not xs_min:
        raise FileNotFoundError(f"No slice_*.ply in {slices_dir}")

    # Expand world bounds with padding
    xmin = float(min(xs_min) - padding); xmax = float(max(xs_max) + padding)
    ymin = float(min(ys_min) - padding); ymax = float(max(ys_max) + padding)

    # Convert to integer pixel dimensions
    W = int(np.ceil((xmax - xmin)/pixel_size))
    H = int(np.ceil((ymax - ymin)/pixel_size))
    return {"xmin": xmin, "ymax": ymax, "pixel_size": pixel_size, "W": W, "H": H}


def world_to_image_fixed(xy: np.ndarray, tf: dict):
    """
    Project world XY points (metres) into the fixed global image frame.

    Image convention
    ----------------
    • Pixel X: (x - xmin) / pixel_size
    • Pixel Y: (ymax - y) / pixel_size   (i.e., origin at top-left, y grows downward)

    Parameters
    ----------
    xy : np.ndarray
        N×2 array of world coordinates (x, y).
    tf : dict
        Transform dict from `compute_global_tf_from_slices`.

    Returns
    -------
    (Tuple[int, int], np.ndarray)
        - (H, W) image shape
        - N×2 array of pixel coordinates (float32)
    """
    px = (xy[:,0] - tf["xmin"]) / tf["pixel_size"]
    py = (tf["ymax"] - xy[:,1]) / tf["pixel_size"]
    return (tf["H"], tf["W"]), np.stack([px,py], 1).astype(np.float32)


def rasterize_heatmap_fixed(xy: np.ndarray, tf: dict) -> np.ndarray:
    """
    Turn world XY points into a smoothed occupancy heatmap in the global frame.

    Steps
    -----
    1) world→image projection
    2) integer rounding + in-bounds clipping
    3) increment per-pixel counts
    4) Gaussian blur → normalize to [0, 255] (uint8)

    Parameters
    ----------
    xy : np.ndarray
        N×2 world coordinates.
    tf : dict
        Transform dict describing the raster frame.

    Returns
    -------
    np.ndarray
        H×W uint8 heatmap image (0 if empty).
    """
    H, W = tf["H"], tf["W"]
    heat = np.zeros((H, W), np.float32)

    _, pix = world_to_image_fixed(xy, tf)
    ij = np.rint(pix).astype(np.int32)
    ij[:,0] = np.clip(ij[:,0], 0, W-1)
    ij[:,1] = np.clip(ij[:,1], 0, H-1)

    np.add.at(heat, (ij[:,1], ij[:,0]), 1.0)
    heat = cv2.GaussianBlur(heat, (0,0), 1.2)

    return (255.0 * (heat/heat.max())).astype(np.uint8) if heat.max()>0 else np.zeros_like(heat, np.uint8)


# --------- run() entrypoint (refactor) ---------
def run(params: Params):
    """
    End-to-end pipeline:
      1) Prepare output dir and global raster transform.
      2) Sort slices by height.
      3) For each slice:
         • detect globally
         • attempt ROI recovery for missed tracks
         • write per-slice CSV and debug overlay
      4) Bridge gaps between observed heights (interpolate → ROI detect).
      5) Emit stems_summary.csv with per-track DBH and mean center.

    Files written to params.out_dir
    -------------------------------
    • slice_XXXX.csv
    • slice_XXXX_debug.png
    • stems_summary.csv
    """
    ensure_dir(params.out_dir)

    # 1) Global frame over all slices
    global GLOBAL_TF
    GLOBAL_TF = compute_global_tf_from_slices(params.slices_dir)
    print("GLOBAL raster:", GLOBAL_TF["H"], "x", GLOBAL_TF["W"])

    # 2) Enumerate and sort slices by extracted height
    slice_files = sorted(
        [p for p in params.slices_dir.glob("slice_*.ply") if p.is_file()],
        key=lambda p: (parse_slice_height(p.name, params.slice_regex) or 0.0)
    )

    tracks: List[Track] = []
    cache_occ_tf = {}   # height_m → (occ, GLOBAL_TF)
    all_heights = []

    # 3) Per-slice processing
    for ply in slice_files:
        h = parse_slice_height(ply.name, params.slice_regex)
        all_heights.append(h)

        rows = process_slice(ply, params, tracks, cache_occ_tf, allow_global=True)

        # Per-slice detections CSV
        csv_path = params.out_dir / f"{ply.stem}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["center_x_m", "center_y_m", "radius_m", "diameter_m", "ringness"])
            w.writeheader(); w.writerows(rows)

    # 4) Gap-bridging: attempt ROI recovery on heights skipped by tracks
    heights_sorted = sorted(set(all_heights))
    bridge_missing_slices(tracks, heights_sorted, cache_occ_tf, params)

    # 5) Summary of tracks (DBH via median @ 1.3–1.7 m, else median of all)
    with open(params.out_dir / "stems_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "n_obs", "median_DBH_m", "mean_x_m", "mean_y_m"])
        for t in tracks:
            xs = [c[0] for c in t.centers_m]; ys = [c[1] for c in t.centers_m]
            w.writerow([t.id, len(t.diam_m), t.dbh_m, float(np.mean(xs)), float(np.mean(ys))])

    print(f"Wrote {len(tracks)} stem tracks → stems_summary.csv")


if __name__ == "__main__":
    run(Params(
        slices_dir=Path("sliced_ply_outputs_2/plot_annotations_ct03t1b_01"),
        out_dir=Path("circle_detections"),
        pixel_size_m=0.02,
    ))
