# ======================================================================
#  refine_circles_from_tracks.py
#  -------------------------------------------------------------
#  Stage: Track-guided per-slice circle refinement (+ crown onset logging)
#  -------------------------------------------------------------
#  Purpose:
#    • Re-run circle fitting inside tight ROIs predicted from merged stem tracks.
#    • Use slice-to-slice drift corrections to translate between raw/corrected XY.
#    • Be sweep-aware: search along the local motion direction of each track.
#    • Apply a taper prior (from DBH) to keep radii plausible with height.
#    • Emit rich per-slice telemetry and PNG debug overlays.
#    • Detect/record likely crown onset (branching) via clutter change.
#
#  Inputs (directories):
#    SLICES_DIR  : folder of slice_*.ply (ASCII) with XYZ points per horizontal slice.
#    MERGED_DIR  : folder containing the merger outputs:
#                  - detections_with_clusters.csv (columns: cluster_id,height_m,x,y,diameter_m,…)
#                  - slice_translations.csv       (columns: height_m,dx_m,dy_m,n_matches)
#
#  Outputs:
#    MERGED_DIR/refined_per_slice_circles.csv  # per-cluster per-slice refined circle (x_corr,y_corr,radius, ok, telemetry)
#    MERGED_DIR/branch_points.csv              # first detected crown-onset height per cluster (+ z-scores/metrics)
#    OUT_DIR/*_debug.png                       # per-slice heatmap with ROIs, priors, refined fits, branch markers
#
#  Key knobs (see constants below):
#    Raster/ROI:     PIXEL_SIZE_M, PADDING_M, BASE_ROI_RADIUS_M, ROI_MAX_M, ROI_DRIFT_GAIN
#    Band/tolerances:RING_BAND_M, RANSAC_TRIALS, RANSAC_INLIER_TOL_M, RANSAC_MIN_ARC_DEG
#    Evidence gates: MIN_BAND_INLIERS, MIN_SECTORS, CLEAN_* thresholds
#    Sweep search:   SWEEP_STEP_M, SWEEP_STEPS
#    Motion clamps:  MAX_CENTER_JUMP_M, MAX_SWEEP_RATE_M_PER_M, MIN_SWEEP_FLOOR_M
#    Taper prior:    expected_diam(), TAPER_PER_M, TAPER_SLACK, DIAM_JUMP_FRAC
#    Clutter logic:  BRANCH_MIN_HEIGHT_M, MIN_CLEAN_RUN_SLICES, BRANCH_PERSIST_SLICES,
#                    CLUTTER_Z_THRESHOLD, OUTER/EDGE/INNER density bands
#
#  Typical usage:
#    # Standalone (writes refined CSV into MERGED_DIR and PNGs into OUT_DIR):
#    python refine_circles_from_tracks.py
#
#    # As a module (from main pipeline):
#    import refine_circles_from_tracks as refiner
#    refiner.run(slices_dir=..., merged_dir=..., out_dir=...)
#
#  Notes:
#    • DBH for taper prior is taken per cluster (median of diameters near 1.5 m if available).
#    • “Corrected” coords = raw + (dx,dy) from slice_translations.csv.
#    • PNG overlay legend:
#        - Magenta box/cross: ROI on raw coords
#        - Yellow circle: prior (track prediction) on raw coords
#        - Green circle/cross: refined result (if ok)
#        - Red star: detected branch-onset slice for that cluster
#    • ASCII PLY is expected; loader is minimal (reads after end_header).
#
#  Dependencies:
#    numpy, opencv-python, tqdm (optional)
#
#  Author: Ethan Glynn
# ======================================================================

import os, re, csv, math, random
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional
import sys, time
from time import perf_counter
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

import numpy as np
import cv2

# =========================
# Config (defaults) — all distances in metres unless noted
# =========================

# --- I/O directories ---
SLICES_DIR = Path("sliced_ply_outputs_2/plot_annotations_ct03t1b_01")  # where slice_*.ply live
MERGED_DIR = Path("merged_stems")                                       # merger outputs (detections + translations)
OUT_DIR    = Path("circle_refine_debug")                                # per-slice PNG overlays
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Filenames like: slice_1.25m.ply → extract height
SLICE_RE = re.compile(r"slice_(?P<h>[0-9]+\.[0-9]+)m\.ply")

# --- Rasterization (debug overlays) ---
PIXEL_SIZE_M = 0.02   # image scale for heatmaps (smaller = higher res)
PADDING_M    = 0.50   # border around global canvas to avoid clipping

# --- ROI geometry around the predicted circle ---
BASE_ROI_RADIUS_M = 0.75   # minimal ROI radius around prior center (↑ catches drift; ↑ cost)
RING_BAND_M       = 0.045  # ± band thickness around radius when counting ring inliers

# --- RANSAC circle fitting (arc-friendly) ---
RANSAC_TRIALS        = 2200   # random 3-pt proposals
RANSAC_INLIER_TOL_M  = 0.038  # distance-to-circle threshold for inliers
RANSAC_MIN_ARC_DEG   = 130.0  # require this much angular coverage (rejects tiny arcs)
RANSAC_MAX_SAMPLES   = 15000  # cap points used per slice for speed

# --- Evidence gates (post-fit health checks) ---
MIN_BAND_INLIERS = 12    # minimum ring-band inliers
MIN_SECTORS      = 8     # minimum occupied angular sectors (uniformity proxy)

# --- Plausibility relative to prior/taper ---
MAX_CENTER_JUMP_M = 0.25  # hard cap on center movement vs prior (outlier guard)
DIAM_JUMP_FRAC    = 0.25  # allowed fractional change vs prior radius

# --- Sweep-aware search (search along motion direction; no height hard-coding) ---
SWEEP_STEP_M   = 0.18   # step size along sweep direction when probing candidates
SWEEP_STEPS    = 10     # number of ± steps from the prior (→ 2*steps+1 candidates)
ROI_DRIFT_GAIN = 1.25   # enlarge ROI with observed drift (larger drift → wider ROI)
ROI_MAX_M      = 1.35   # absolute cap on ROI radius (keeps runtime bounded)
DERIV_DH       = 0.60   # finite-difference step (m) when estimating sweep derivative

# --- Movement clamps between accepted slices (rate limiter) ---
MAX_SWEEP_RATE_M_PER_M = 0.14  # max allowed center change per metre of height
MIN_SWEEP_FLOOR_M      = 0.08  # minimum allowed movement regardless of small dh
DRIFT_PENALTY_GAIN     = 0.45  # score penalty per metre of drift from prior (higher → stricter)

# --- Taper prior (expected diameter vs height) ---
TAPER_PER_M = 0.08   # linear taper: fraction per metre relative to DBH (positive = thinner higher)
TAPER_SLACK = 0.35   # tolerance around expected radius from taper prior

# --- Crown onset (branch) logging via rising clutter ---
BRANCH_MIN_HEIGHT_M   = 1.60  # ignore very low clutter (butt swell / debris)
MIN_CLEAN_RUN_SLICES  = 3     # clean baseline length before watching for change
BRANCH_PERSIST_SLICES = 2     # require change to persist to avoid flicker
CLUTTER_Z_THRESHOLD   = 2.0   # z-score vs baseline to flag a change (per-feature)

# --- “Clean ring” thresholds to build clutter baseline ---
CLEAN_MIN_SECTORS = 10   # sectors occupied
CLEAN_MIN_INLIERS = 80   # ring-band inliers
CLEAN_MIN_OCC     = 0.45 # mean sector occupancy fraction

# --- Clutter measurement bands (relative to fitted radius) ---
OUTER_CLUTTER_INNER_M = 0.05  # annulus start outside the ring
OUTER_CLUTTER_OUTER_M = 0.20  # annulus end   outside the ring
EDGE_NOISE_BAND_M     = 0.03  # very tight strip around ring (roughness)
INNER_DENSITY_FRAC    = 0.55  # inside this fraction of radius counts as trunk core

# --- Data sufficiency for clutter metrics ---
MIN_ROI_POINTS_FOR_CLUTTER = 300  # skip clutter stats if ROI too sparse


def finite_diff(f, h, dh=DERIV_DH):
    return float((f(h+dh) - f(max(h-dh, 0.0))) / max(2.0*dh, 1e-6))

# =========================
# I/O helpers
# =========================
def _flush_file(f):
    f.flush()
    os.fsync(f.fileno())

def parse_h(name: str) -> Optional[float]:
    m = SLICE_RE.match(name)
    return float(m.group("h")) if m else None

def load_ply_xyz(path: Path) -> np.ndarray:
    """ASCII PLY → Nx3 float32 (x y z)."""
    with open(path, "r", errors="ignore") as f:
        header = []
        for ln in f:
            header.append(ln)
            if ln.strip() == "end_header":
                break
        arr = np.loadtxt(path, skiprows=len(header), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, :3].astype(np.float32) if arr.size else np.zeros((0,3), np.float32)

def read_detections_with_clusters(path: Path):
    rows = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            d = row.get("diameter_m", "")
            rows.append({
                "cid": int(row["cluster_id"]),
                "h":   float(row["height_m"]),
                "x":   float(row["x"]),
                "y":   float(row["y"]),
                "diam": float(d) if d not in ("", "nan") else float("nan"),
            })
    return rows

def read_translations(path: Path):
    """height -> (dx, dy) where corrected = raw + (dx, dy)."""
    tr = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            tr[float(row["height_m"])] = (float(row["dx_m"]), float(row["dy_m"]))
    return tr

# =========================
# Geometry / raster helpers
# =========================
def expected_diam(h: float, dbh: float, bh: float = 1.5, taper: float = TAPER_PER_M) -> float:
    """
    Predict diameter at height h (metres) from DBH using a simple linear taper model.

    Args:
        h:   Height above ground [m].
        dbh: Diameter at breast height (≈1.3–1.5 m) [m].
        bh:  Reference breast-height used in the taper model [m].
        taper: Fractional change in diameter per metre (positive → thinner with height).

    Returns:
        Predicted diameter at height h [m], floored at 0.05 m to avoid degenerate radii.
    """
    return max(0.05, float(dbh * (1.0 + taper * (bh - h))))


def compute_global_tf_from_slices(
    slices_dir: Path,
    pixel_size: float = PIXEL_SIZE_M,
    padding: float = PADDING_M
) -> dict:
    """
    Build a global world→image transform for debug rasters by scanning all slice_*.ply files.

    Image convention:
        • x → right, y → down
        • We flip Y so that larger world-Y appears higher in the image: py = (ymax - y)/pixel
    The transform dict contains:
        { "xmin": float, "ymax": float, "pixel_size": float, "W": int, "H": int }

    Args:
        slices_dir: Directory containing slice_*.ply (ASCII) with XYZ columns.
        pixel_size: World metres per pixel [m/px].
        padding:    Extra world margin added around the global bounds [m].

    Returns:
        Transform dict usable by world_to_image_fixed / world_point_to_pixel.

    Raises:
        FileNotFoundError: If no slice_*.ply files with points are found.
    """
    xs_min, xs_max, ys_min, ys_max = [], [], [], []
    for p in sorted(slices_dir.glob("slice_*.ply")):
        pts = load_ply_xyz(p)
        if pts.size:
            xs, ys = pts[:, 0], pts[:, 1]
            xs_min.append(xs.min()); xs_max.append(xs.max())
            ys_min.append(ys.min()); ys_max.append(ys.max())

    if not xs_min:
        raise FileNotFoundError(f"No slice_*.ply in {slices_dir}")

    xmin = float(min(xs_min) - padding)
    xmax = float(max(xs_max) + padding)
    ymin = float(min(ys_min) - padding)
    ymax = float(max(ys_max) + padding)

    W = int(np.ceil((xmax - xmin) / pixel_size))
    H = int(np.ceil((ymax - ymin) / pixel_size))

    return {"xmin": xmin, "ymax": ymax, "pixel_size": pixel_size, "W": W, "H": H}


def world_to_image_fixed(xy: np.ndarray, tf: dict) -> Tuple[Tuple[int, int], np.ndarray]:
    """
    Vectorized world→image mapping for a batch of (x,y) points.

    Args:
        xy: (N, 2) array of world coordinates in metres.
        tf: Transform dict from compute_global_tf_from_slices.

    Returns:
        (H, W), pix:
            • (H, W): image shape tuple
            • pix: (N, 2) float32 pixel coords (px, py), not clipped/rounded
    """
    # Gracefully handle empty/Nan inputs to keep downstream robust
    if xy.size == 0:
        return (tf["H"], tf["W"]), np.zeros((0, 2), dtype=np.float32)

    # Remove NaNs if any slipped in (keeps alignment if caller expects 1:1 -> skip filtering)
    px = (xy[:, 0] - tf["xmin"]) / tf["pixel_size"]
    py = (tf["ymax"] - xy[:, 1]) / tf["pixel_size"]
    return (tf["H"], tf["W"]), np.stack([px, py], axis=1).astype(np.float32)


def world_point_to_pixel(tf: dict, x: float, y: float) -> Tuple[float, float]:
    """
    Scalar convenience wrapper for world→image mapping.

    Args:
        tf: Transform dict from compute_global_tf_from_slices.
        x, y: World coordinates [m].

    Returns:
        (px, py) float pixel coordinates (not rounded).
    """
    px = (x - tf["xmin"]) / tf["pixel_size"]
    py = (tf["ymax"] - y) / tf["pixel_size"]
    return float(px), float(py)


def rasterize_heatmap_fixed(xy: np.ndarray, tf: dict) -> np.ndarray:
    """
    Make a debug occupancy heatmap by splatting points onto a global canvas.

    Steps:
        1) Map world points to pixels.
        2) Round to integer bins and clip to image bounds.
        3) Accumulate counts and lightly blur for visibility.
        4) Normalize to 0–255 uint8 (safe if empty).

    Args:
        xy: (N, 2) world points [m].
        tf: Transform dict from compute_global_tf_from_slices.

    Returns:
        (H, W) uint8 heatmap.
    """
    H, W = tf["H"], tf["W"]
    heat = np.zeros((H, W), np.float32)

    _, pix = world_to_image_fixed(xy, tf)
    if pix.size == 0:
        return np.zeros((H, W), np.uint8)

    ij = np.rint(pix).astype(np.int32)
    # Clip to valid indices
    ij[:, 0] = np.clip(ij[:, 0], 0, W - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, H - 1)

    # Accumulate (row, col) = (y, x)
    np.add.at(heat, (ij[:, 1], ij[:, 0]), 1.0)

    # Gentle blur makes sparse data visible without washing out dense areas
    heat = cv2.GaussianBlur(heat, (0, 0), 1.2)

    vmax = float(heat.max())
    if vmax <= 0.0 or not np.isfinite(vmax):
        return np.zeros((H, W), np.uint8)

    return (255.0 * (heat / vmax)).astype(np.uint8)

# =========================
# Smoothing (simple, robust)
# =========================
def mad(arr: np.ndarray) -> float:
    """
    Robust scale estimate: Median Absolute Deviation (MAD) scaled to ~σ for normal data.

    Args:
        arr: 1D array of values.

    Returns:
        Robust standard-deviation proxy (float). Returns 0.0 for empty input.
    """
    if arr.size == 0:
        return 0.0
    med = np.median(arr)
    return float(1.4826 * np.median(np.abs(arr - med)))


def kalman_1d(hs: np.ndarray, zs: np.ndarray, q: float = 1e-5, r: float = 1e-3) -> np.ndarray:
    """
    Tiny scalar Kalman smoother over an ordered sequence (e.g., vs height).
    Model: x_k = x_{k-1} + ε_q,  z_k = x_k + ε_r

    Args:
        hs: Monotone support (e.g., heights). Only used for alignment; not in equations.
        zs: Observations aligned with hs (same length).
        q:  Process noise variance (larger → smoother/laggier response).
        r:  Baseline measurement noise variance. We also inflate using robust MAD of residuals.

    Returns:
        Smoothed values, same length as zs (np.float32).
    """
    # Guard: empty or scalar
    if zs.size == 0:
        return np.zeros(0, dtype=np.float32)
    if zs.size == 1:
        return zs.astype(np.float32)

    x = float(zs[0])   # state
    P = 1.0            # variance
    out = [x]

    # Robust measurement noise scaling from residual dispersion
    resid = zs - np.median(zs)
    R = max(float(r), mad(resid) or float(r))

    for k in range(1, len(zs)):
        # predict
        x_pred, P_pred = x, P + q
        # update
        K = P_pred / (P_pred + R)
        x = x_pred + K * (float(zs[k]) - x_pred)
        P = (1.0 - K) * P_pred
        out.append(x)

    return np.asarray(out, dtype=np.float32)


def smooth_track(
    heights: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    rs: np.ndarray
) -> dict | None:
    """
    Smooth a single stem track (x(h), y(h), r(h)) with robust trimming + tiny Kalman.

    Pipeline:
        1) Sort by height.
        2) Trim gross outliers in x/y via 3.5×MAD gates.
        3) Fill NaNs in radius with robust median.
        4) Smooth each series independently with kalman_1d.
        5) Return interpolation callables fx(h), fy(h), fr(h).

    Args:
        heights: 1D array of slice heights [m].
        xs, ys:  1D arrays of centers [m].
        rs:      1D array of radii [m].

    Returns:
        {"fx": f(h)->x, "fy": f(h)->y, "fr": f(h)->r} or None if no valid samples.
    """
    if np.size(heights) == 0:
        return None

    order = np.argsort(heights)
    h = np.asarray(heights, np.float32)[order]
    x = np.asarray(xs,      np.float32)[order]
    y = np.asarray(ys,      np.float32)[order]
    r = np.asarray(rs,      np.float32)[order]

    if h.size == 0:
        return None

    # --- Robust outlier trim on x/y (3.5×MAD)
    keep = np.ones(h.shape[0], dtype=bool)

    med_x = np.median(x[keep]); mx = mad(x[keep]) or 1e-3
    keep &= (np.abs(x - med_x) <= 3.5 * mx)

    med_y = np.median(y[keep]); my = mad(y[keep]) or 1e-3
    keep &= (np.abs(y - med_y) <= 3.5 * my)

    h, x, y, r = h[keep], x[keep], y[keep], r[keep]
    if h.size == 0:
        return None

    # --- Fill NaNs in radius with robust median (fallback 0.20 m)
    if np.isnan(r).any():
        fill = np.nanmedian(r) if not np.isnan(r).all() else 0.20
        r = np.where(np.isnan(r), float(fill), r)

    # --- Smooth each channel independently
    xs_s = kalman_1d(h, x)
    ys_s = kalman_1d(h, y)
    rs_s = kalman_1d(h, r)

    # --- Build lightweight interpolators (monotone linear)
    def _interp(hs_arr: np.ndarray, vs_arr: np.ndarray):
        hs_arr = np.asarray(hs_arr, np.float32)
        vs_arr = np.asarray(vs_arr, np.float32)

        def f(hq: float) -> float:
            if hs_arr.size == 1:
                return float(vs_arr[0])
            return float(np.interp(hq, hs_arr, vs_arr))
        return f

    return {"fx": _interp(h, xs_s), "fy": _interp(h, ys_s), "fr": _interp(h, rs_s)}

# =========================
# Evidence checks
# =========================
def sector_occupancy_points(
    xy: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    band_m: float = RING_BAND_M,
    n_sectors: int = 24
) -> float:
    """
    Fraction of angular sectors (0..360°) that contain at least one point within a thin
    annulus around radius r (a quick completeness proxy for a circular ring).

    Args:
        xy:        Nx2 points (x,y) for a single slice [m].
        cx, cy:    Candidate circle center [m].
        r:         Candidate radius [m] (must be > 0).
        band_m:    Half-width of the ring band around r [m]. Points with |dist-r|<=band_m count.
        n_sectors: Number of uniform angular bins over [0, 360).

    Returns:
        Occupancy fraction in [0,1]: (# non-empty sectors) / n_sectors.
        Returns 0.0 if inputs are empty/degenerate.
    """
    if xy.size == 0 or r <= 0.0 or band_m <= 0.0 or n_sectors < 1:
        return 0.0

    # radial distances from candidate center
    dx = xy[:, 0] - cx
    dy = xy[:, 1] - cy
    dist = np.hypot(dx, dy)

    # select points within the ring band
    band_mask = np.abs(dist - r) <= band_m
    if not np.any(band_mask):
        return 0.0

    # angles of band points mapped to [0, 360)
    ang = (np.degrees(np.arctan2(dy[band_mask], dx[band_mask])) + 360.0) % 360.0

    # uniform sectoring and occupancy
    sector_width = 360.0 / float(n_sectors)
    bins = np.floor(ang / sector_width).astype(np.int32)
    # clamp in case floating error puts a value at exactly 360°
    bins = np.clip(bins, 0, n_sectors - 1)

    counts = np.bincount(bins, minlength=n_sectors)
    return float(np.count_nonzero(counts)) / float(n_sectors)


def ring_voidness_ratio(
    xy: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    band_m: float = RING_BAND_M,
    inner_frac: float = 0.65
) -> float:
    """
    Ratio of ring-band points to inner-disk points (coarse "hollowness" check).
    Clean stems tend to have many points on the ring and few in the inner disk.

    Args:
        xy:         Nx2 points (x,y) for a single slice [m].
        cx, cy:     Candidate circle center [m].
        r:          Candidate radius [m] (must be > 0).
        band_m:     Half-width of the ring band around r [m].
        inner_frac: Inner region radius as a fraction of r (clamped to ≥0.01 m).

    Returns:
        (ring_count) / max(1, inner_count) as a float. Returns 0.0 for empty/degenerate cases.
    """
    if xy.size == 0 or r <= 0.0 or band_m <= 0.0:
        return 0.0

    dx = xy[:, 0] - cx
    dy = xy[:, 1] - cy
    dist = np.hypot(dx, dy)

    # ring band
    ring_mask = np.abs(dist - r) <= band_m
    n_ring = int(np.count_nonzero(ring_mask))

    # inner disk (avoid vanishing radius)
    inner_r = max(0.01, float(inner_frac) * r)
    inner_mask = dist <= inner_r
    n_inner = int(np.count_nonzero(inner_mask))

    return float(n_ring) / float(max(1, n_inner))

def clutter_metrics(xy: np.ndarray, cx: float, cy: float, r: float,
                    outer_in=OUTER_CLUTTER_INNER_M, outer_out=OUTER_CLUTTER_OUTER_M,
                    edge_band=EDGE_NOISE_BAND_M, inner_frac=INNER_DENSITY_FRAC):
    """Compute lightweight clutter features around a fitted ring."""
    if xy.size == 0:
        return dict(outer_density=0.0, edge_noise=0.0, inner_density=0.0, resid_mad=0.0, n=0)

    dx = xy[:,0] - cx
    dy = xy[:,1] - cy
    dist = np.hypot(dx, dy)

    # Edge noise: points extremely close to ring (wiggly/rough ring contour)
    edge = np.abs(dist - r) <= edge_band
    edge_noise = float(np.count_nonzero(edge)) / max(1.0, 2*np.pi*r*edge_band)  # ~perimeter-normalized

    # Outer clutter: annulus outside ring
    outer = (dist >= (r + outer_in)) & (dist <= (r + outer_out))
    # normalize by annulus area
    outer_area = np.pi*((r+outer_out)**2 - (r+outer_in)**2)
    outer_density = float(np.count_nonzero(outer)) / max(outer_area, 1e-6)

    # Inner density: inside fraction of radius (should be low for clean trunk)
    inner = dist <= (max(0.02, inner_frac*r))
    inner_density = float(np.count_nonzero(inner)) / max(np.pi*(inner_frac*r)**2, 1e-6)

    # Residual rugosity around ring (how wobbly are distances near ring)
    near = np.abs(dist - r) <= max(edge_band*2.0, 0.06)
    resid = np.abs(dist[near] - r) if np.any(near) else np.array([0.0])
    resid_mad = float(1.4826 * np.median(np.abs(resid - np.median(resid)))) if resid.size else 0.0

    return dict(
        outer_density=outer_density,
        edge_noise=edge_noise,
        inner_density=inner_density,
        resid_mad=resid_mad,
        n=int(xy.shape[0])
    )

# =========================
# Circle fitting
# =========================

def fit_circle_taubin(pts):
    """
    Algebraic circle fitting using Taubin's method.
    Computes the least-squares circle through a set of 2D points.
    """
    # Split into x, y
    x = pts[:, 0]
    y = pts[:, 1]

    # Center data
    x_m = np.mean(x)
    y_m = np.mean(y)
    u = x - x_m
    v = y - y_m

    # Compute moments
    Suu = np.dot(u, u)
    Svv = np.dot(v, v)
    Suv = np.dot(u, v)
    Suuu = np.dot(u, u * u)
    Svvv = np.dot(v, v * v)
    Suvv = np.dot(u, v * v)
    Svuu = np.dot(v, u * u)

    # Build system A·[uc vc]^T = b
    A = np.array([[Suu, Suv],
                  [Suv, Svv]], dtype=np.float64)
    b = 0.5 * np.array([Suuu + Suvv,
                        Svvv + Svuu], dtype=np.float64)

    # Solve for offsets from mean
    det = np.linalg.det(A)
    if abs(det) < 1e-12:
        return None  # Degenerate / collinear

    uc, vc = np.linalg.solve(A, b)
    cx = x_m + uc
    cy = y_m + vc

    # Mean distance to center → radius
    r = float(np.mean(np.hypot(x - cx, y - cy)))
    return cx, cy, r


def ransac_circle(pts, trials=RANSAC_TRIALS, tol=RANSAC_INLIER_TOL_M,
                  min_arc_deg=RANSAC_MIN_ARC_DEG, seed=123):
    """
    RANSAC-based robust circle fitting.
    Randomly samples 3 points, fits a circle, evaluates by inlier count and arc coverage.
    """
    if pts.shape[0] < 6:
        return None

    rng = random.Random(seed)
    best, best_score = None, -1.0
    idxs = list(range(pts.shape[0]))

    # Run randomized hypotheses
    for _ in range(trials):
        # Randomly sample 3 unique points
        i, j, k = rng.sample(idxs, 3)
        p1, p2, p3 = pts[i], pts[j], pts[k]

        # Solve for circle center (intersection of perpendicular bisectors)
        A = 2 * np.array([[p2[0] - p1[0], p2[1] - p1[1]],
                          [p3[0] - p1[0], p3[1] - p1[1]]], dtype=np.float64)
        b = np.array([[p2[0]**2 - p1[0]**2 + p2[1]**2 - p1[1]**2],
                      [p3[0]**2 - p1[0]**2 + p3[1]**2 - p1[1]**2]], dtype=np.float64)

        if abs(np.linalg.det(A)) < 1e-12:
            continue  # Nearly collinear sample

        c = np.linalg.solve(A, b).ravel()
        cx, cy = c[0], c[1]
        r = np.hypot(p1[0] - cx, p1[1] - cy)

        # Compute residuals and inliers
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        resid = np.abs(d - r)
        inliers = resid <= tol
        n_in = int(inliers.sum())

        if n_in < 12:
            continue  # Too few inliers

        # Angular coverage of inliers
        ang = (np.degrees(np.arctan2(pts[inliers, 1] - cy,
                                     pts[inliers, 0] - cx)) + 360) % 360
        ang = np.sort(ang)
        gaps = np.diff(np.r_[ang, ang[0] + 360])
        coverage = 360.0 - float(gaps.max())

        if coverage < min_arc_deg:
            continue  # Partial arc too small

        # Score: weighted by inlier count and coverage
        score = n_in * (coverage / 360.0)
        if score > best_score:
            best_score = score
            best = (cx, cy, r, inliers)

    return best

# =========================
# ROI refine (WITH TAPER PRIOR)
# =========================
def refine_circle_roi_for_slice(
    xyz: np.ndarray,
    cx_corr: float, cy_corr: float,
    dx: float, dy: float,
    r_pred: float,
    h: float, dbh: float,
    roi_center_raw: Tuple[float, float] = None,
    roi_radius_m: Optional[float] = None
):
    """
    Refine a circle fit within a local ROI around a predicted center/radius.
    Inputs are in world metres. Returns:
      (ok, x_ref, y_ref, r_ref, info_dict, (roi_cx_raw, roi_cy_raw, roi_R), score)
    """

    # ---- 1) Decide ROI center/radius in RAW coords (pre-translation) ----
    cx_raw_prior, cy_raw_prior = cx_corr - dx, cy_corr - dy
    if roi_center_raw is None:
        cx_roi, cy_roi = cx_raw_prior, cy_raw_prior
    else:
        cx_roi, cy_roi = roi_center_raw

    roi_R = roi_radius_m if roi_radius_m is not None else max(BASE_ROI_RADIUS_M, r_pred + 0.22)

    # ---- 2) Collect candidate points inside ROI (XY only) ----
    xy = xyz[:, :2]
    dist_roi = np.hypot(xy[:, 0] - cx_roi, xy[:, 1] - cy_roi)
    roi_mask = dist_roi <= roi_R
    if not np.any(roi_mask):
        return False, cx_corr, cy_corr, r_pred, {"reason": "empty_roi"}, (cx_roi, cy_roi, roi_R), 0.0

    roi_xy = xy[roi_mask]

    # Cap point count for RANSAC speed
    if roi_xy.shape[0] > RANSAC_MAX_SAMPLES:
        sel = np.random.choice(roi_xy.shape[0], RANSAC_MAX_SAMPLES, replace=False)
        roi_xy = roi_xy[sel]

    # ---- 3) Gate a thin "ring band" around r_pred to focus the fit ----
    band_allow = min(0.12, max(0.06, 0.25 * max(r_pred, 0.15), 0.15 * roi_R))
    cand_mask = np.abs(np.hypot(roi_xy[:, 0] - cx_roi, roi_xy[:, 1] - cy_roi) - r_pred) <= band_allow
    cand_pts = roi_xy[cand_mask] if np.any(cand_mask) else roi_xy

    # If band is too sparse, do a coarse RANSAC to re-center band, then refocus
    if cand_pts.shape[0] < 120:
        coarse = ransac_circle(
            roi_xy,
            trials=int(RANSAC_TRIALS * 0.6),
            tol=max(RANSAC_INLIER_TOL_M, 0.045),
            min_arc_deg=max(160.0, RANSAC_MIN_ARC_DEG - 20.0)
        )
        if coarse is not None:
            cx_c, cy_c, r_c, _ = coarse
            band2 = np.abs(np.hypot(roi_xy[:, 0] - cx_c, roi_xy[:, 1] - cy_c) - r_c) <= max(0.05, RING_BAND_M * 1.8)
            cand_pts = roi_xy[band2] if np.any(band2) else roi_xy

    # ---- 4) Robust fit: primary RANSAC, fallback to Taubin if weak but plausible ----
    fit = ransac_circle(cand_pts)

    if fit is None:
        # Quick plausibility: do we have any ring-like occupancy at r_pred?
        occ = sector_occupancy_points(cand_pts, cx_roi, cy_roi, r_pred, band_m=band_allow, n_sectors=24)
        if occ < 0.30:
            return False, cx_corr, cy_corr, r_pred, {"reason": "no_fit_weak_band"}, (cx_roi, cy_roi, roi_R), 0.0

        # Try Taubin on a capped subset (algebraic, not robust)
        sub = cand_pts[:2000] if cand_pts.shape[0] > 2000 else cand_pts
        ta = fit_circle_taubin(sub) if sub.shape[0] >= 6 else None
        if ta is None:
            return False, cx_corr, cy_corr, r_pred, {"reason": "no_fit"}, (cx_roi, cy_roi, roi_R), 0.0
        cx_fit, cy_fit, r_fit = ta
    else:
        cx_fit, cy_fit, r_fit, _ = fit

    # ---- 5) Center sanity: don't jump too far from prior raw center ----
    if math.hypot(cx_fit - cx_raw_prior, cy_fit - cy_raw_prior) > MAX_CENTER_JUMP_M:
        return False, cx_corr, cy_corr, r_pred, {"reason": "jump_gate"}, (cx_roi, cy_roi, roi_R), 0.0

    # ---- 6) Radius sanity: combine prior radius window and taper-based expectation ----
    r_exp = 0.5 * expected_diam(h, dbh)  # predicted radius from taper model (diam/2)

    rmin1, rmax1 = r_pred * (1.0 - DIAM_JUMP_FRAC), r_pred * (1.0 + DIAM_JUMP_FRAC)
    rmin2, rmax2 = r_exp  * (1.0 - TAPER_SLACK),    r_exp  * (1.0 + TAPER_SLACK)
    rmin, rmax = min(rmin1, rmin2), max(rmax1, rmax2)

    if not (rmin <= r_fit <= rmax):
        taper_dev = abs(r_fit - r_exp) / max(r_exp, 1e-6)
        return False, cx_corr, cy_corr, r_pred, {
            "reason": "radius_gate",
            "r_exp": float(r_exp),
            "taper_dev": float(taper_dev)
        }, (cx_roi, cy_roi, roi_R), 0.0

    # ---- 7) Evidence checks around the fitted ring ----
    dist_ref = np.hypot(roi_xy[:, 0] - cx_fit, roi_xy[:, 1] - cy_fit)
    band     = np.abs(dist_ref - r_fit) <= RING_BAND_M
    n_in     = int(np.count_nonzero(band))

    occ_frac = sector_occupancy_points(roi_xy, cx_fit, cy_fit, r_fit, band_m=RING_BAND_M, n_sectors=24)
    voidness = ring_voidness_ratio(roi_xy, cx_fit, cy_fit, r_fit, band_m=RING_BAND_M, inner_frac=0.65)
    sectors  = int(round(occ_frac * 24))

    ok = (
        (sectors >= MIN_SECTORS) and
        (n_in >= MIN_BAND_INLIERS) and
        (occ_frac >= 0.35) and
        (voidness >= 1.3)
    )
    if not ok:
        return False, cx_corr, cy_corr, r_pred, {
            "reason": f"evidence(sectors={sectors},inliers={n_in},occ={occ_frac:.2f},void={voidness:.2f})",
            "r_exp": float(r_exp)
        }, (cx_roi, cy_roi, roi_R), 0.0

    # ---- 8) Clutter metrics + final score (higher = better) ----
    clut = clutter_metrics(roi_xy, cx_fit, cy_fit, r_fit)
    taper_dev = abs(r_fit - r_exp) / max(r_exp, 1e-6)
    score = (occ_frac * voidness) * (1.0 + 0.02 * n_in)

    # Return center in CORRECTED coords (add back dx, dy)
    return True, cx_fit + dx, cy_fit + dy, float(r_fit), {
        "inliers": n_in, "sectors": sectors, "occ": occ_frac, "void": voidness,
        "r_exp": float(r_exp), "taper_dev": float(taper_dev),
        "outer_density": clut["outer_density"],
        "edge_noise":    clut["edge_noise"],
        "inner_density": clut["inner_density"],
        "resid_mad":     clut["resid_mad"],
        "roi_points":    clut["n"]
    }, (cx_roi, cy_roi, roi_R), float(score)


# =========================
# Debug drawing
# =========================
def draw_slice_debug(xy_raw: np.ndarray, tf: dict, dets_for_slice: List[dict], out_png: Path):
    occ = rasterize_heatmap_fixed(xy_raw, tf)
    canvas = cv2.cvtColor(occ, cv2.COLOR_GRAY2BGR)
    H, W = occ.shape
    for d in dets_for_slice:
        cxr, cyr, rr = d["roi_raw"]
        x0w, y0w = cxr - rr, cyr - rr
        x1w, y1w = cxr + rr, cyr + rr
        x0p, y0p = world_point_to_pixel(tf, x0w, y0w)
        x1p, y1p = world_point_to_pixel(tf, x1w, y1w)
        x0i, y0i = int(round(x0p)), int(round(y0p))
        x1i, y1i = int(round(x1p)), int(round(y1p))
        cv2.rectangle(canvas, (x0i,y0i), (x1i,y1i), (255, 0, 255), 1)
        cxp, cyp = world_point_to_pixel(tf, cxr, cyr)
        cv2.drawMarker(canvas, (int(round(cxp)), int(round(cyp))), (255, 0, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)
        px, py = world_point_to_pixel(tf, d["cx_raw_prior"], d["cy_raw_prior"])
        pr = d["r_prior"] / tf["pixel_size"]
        cv2.circle(canvas, (int(round(px)), int(round(py))), int(round(pr)), (0, 255, 255), 1)
        cv2.circle(canvas, (int(round(px)), int(round(py))), 2, (0, 255, 255), -1)
        if d["ok"]:
            rx, ry = world_point_to_pixel(tf, d["cx_raw_ref"], d["cy_raw_ref"])
            rrpx   = d["r_ref"] / tf["pixel_size"]
            cv2.circle(canvas, (int(round(rx)), int(round(ry))), int(round(rrpx)), (0, 220, 0), 2)
            cv2.circle(canvas, (int(round(rx)), int(round(ry))), 2, (0, 220, 0), -1)
        # branch (crown onset) marker
        if d.get("branch_here"):
            rx, ry = world_point_to_pixel(tf, d["cx_raw_ref"], d["cy_raw_ref"])
            cv2.drawMarker(
                canvas,
                (int(round(rx)), int(round(ry))),
                (0, 0, 255),                # red
                markerType=cv2.MARKER_STAR,
                markerSize=18,
                thickness=2
            )
        label = f"CID {d['cid']}" + (" ✓" if d["ok"] else " ×")
        cv2.putText(canvas, label, (max(2, x0i+2), max(12, y0i+12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
        if not d["ok"] and d.get("reason"):
            cv2.putText(canvas, d["reason"][:32], (max(2, x0i+2), min(H-6, y0i+26)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_png), canvas)

# =========================
# Corridor Search
# =========================
def corridor_search(
    xyz: np.ndarray, h: float,
    sm_fx, sm_fy,
    dx: float, dy: float,
    r_pred: float,
    last_ok: Optional[Tuple[float, float, float, float]],
    dbh: float
):
    """
    Sweep along the predicted track at height h and try a few candidate ROI centers.
    Returns the best-scoring refinement tuple from refine_circle_roi_for_slice.

    Args:
        xyz:   slice point cloud (N x 3), metres
        h:     slice height (m)
        sm_fx, sm_fy: smoothed center predictors x(h), y(h)
        dx, dy: per-slice drift corrections (add to raw → corrected)
        r_pred: prior radius at height h (m)
        last_ok: last accepted (h, x, y, r) for rate gating; None if none yet
        dbh:  median diameter at breast height (m)

    Returns:
        (ok, x_ref, y_ref, r_ref, info_dict, (roi_cx_raw, roi_cy_raw, roi_R), score, sweep_log)
    """

    # Predicted corrected center at height h from smoother
    cx0, cy0 = sm_fx(h), sm_fy(h)

    # ---- Velocity estimate (for sweep direction) ----
    if last_ok is not None:
        # derive velocity from last refined point to current prediction
        h_last, x_last, y_last, _ = last_ok
        dh = max(h - h_last, 1e-6)
        vx = (cx0 - x_last) / dh
        vy = (cy0 - y_last) / dh
    else:
        # fallback: numerical derivative of the smooth functions
        vx = finite_diff(sm_fx, h)
        vy = finite_diff(sm_fy, h)

    vnorm = math.hypot(vx, vy)
    # Direction to probe along (unit vector); default to +x if nearly stationary
    dirx, diry = (1.0, 0.0) if vnorm < 1e-6 else (vx / vnorm, vy / vnorm)

    # For logging: the zero-offset prediction point
    cx_pred, cy_pred = cx0, cy0

    # Best-so-far holder (matches refine_circle_roi_for_slice() return shape)
    best = (
        False,                # ok
        cx0, cy0, r_pred,     # x_ref, y_ref, r_ref (init with priors)
        {"reason": "no_candidate"},
        (cx0 - dx, cy0 - dy, BASE_ROI_RADIUS_M),  # ROI in raw coords (cx_raw, cy_raw, R)
        0.0,                  # score
        {   # sweep log
            "vx": float(vx), "vy": float(vy), "speed": float(vnorm),
            "cx_pred": float(cx_pred), "cy_pred": float(cy_pred),
            "drift_from_prior": 0.0, "drift_from_last": 0.0, "allowed_drift": 0.0
        }
    )

    # ---- Candidate centers along sweep direction (0 and ±k*step) ----
    candidates = [(0.0, cx0, cy0)]
    for k in range(1, SWEEP_STEPS + 1):
        off = k * SWEEP_STEP_M
        candidates.append((+off, cx0 + dirx * off, cy0 + diry * off))
        candidates.append((-off, cx0 - dirx * off, cy0 - diry * off))

    # ---- Evaluate each candidate ROI ----
    for _, cxc, cyc in candidates:
        # Candidate center in RAW coords for ROI placement
        cx_raw = cxc - dx
        cy_raw = cyc - dy

        # Drift vs last good corrected center; if none, use step size
        drift = (
            math.hypot(cxc - last_ok[1], cyc - last_ok[2]) if last_ok
            else SWEEP_STEP_M
        )

        # ROI radius: base, plus allowance for drift
        roi_R = min(
            ROI_MAX_M,
            max(BASE_ROI_RADIUS_M, r_pred + 0.2, ROI_DRIFT_GAIN * drift)
        )

        # Try to refine within this ROI
        ok, x_ref, y_ref, r_ref, info, roi_box, score = refine_circle_roi_for_slice(
            xyz,
            cxc, cyc,                 # corrected prior center
            dx, dy,                   # drift corrections
            r_pred,                   # predicted radius
            h=h, dbh=dbh,             # for taper prior
            roi_center_raw=(cx_raw, cy_raw),
            roi_radius_m=roi_R
        )

        # ---- Rate gate vs last OK (limit jump per Δh) ----
        drift_from_last = 0.0
        allowed = 0.0
        if ok and last_ok is not None:
            h_last, x_last, y_last, _ = last_ok
            dh = max(h - h_last, 1e-6)
            allowed = min(
                MAX_CENTER_JUMP_M,
                MIN_SWEEP_FLOOR_M + MAX_SWEEP_RATE_M_PER_M * dh
            )
            drift_from_last = math.hypot(x_ref - x_last, y_ref - y_last)

            if drift_from_last > allowed:
                # Too fast laterally for the available Δh → reject this candidate
                ok = False
                info = {
                    "reason": f"rate_gate(drift={drift_from_last:.2f}m>allowed={allowed:.2f}m)"
                }
                score = 0.0

        # Penalize distance from the smoother’s predicted center
        drift_from_prior = math.hypot(x_ref - cx0, y_ref - cy0)
        score *= 1.0 / (1.0 + DRIFT_PENALTY_GAIN * drift_from_prior)

        # Build sweep diagnostics
        log = {
            "vx": float(vx), "vy": float(vy), "speed": float(vnorm),
            "cx_pred": float(cx_pred), "cy_pred": float(cy_pred),
            "drift_from_prior": float(drift_from_prior),
            "drift_from_last":  float(drift_from_last),
            "allowed_drift":    float(allowed)
        }

        # Keep the highest-scoring candidate
        if score > best[6]:
            best = (ok, x_ref, y_ref, r_ref, info, roi_box, score, log)

    return best


# =========================
# run() entrypoint (refactor)
# =========================
def run(slices_dir: Path = SLICES_DIR,
        merged_dir: Path = MERGED_DIR,
        out_dir: Path = OUT_DIR):
    """
    Execute the refinement using the provided directories.
    - slices_dir: where slice_*.ply live
    - merged_dir: where detections_with_clusters.csv and slice_translations.csv live
    - out_dir:    where debug PNGs go; refined CSV is written into merged_dir
    """
    csv_dets = merged_dir / "detections_with_clusters.csv"
    csv_tr   = merged_dir / "slice_translations.csv"
    out_csv  = merged_dir / "refined_per_slice_circles.csv"

    out_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    # 1) slices
    slice_paths = sorted([p for p in slices_dir.glob("slice_*.ply")], key=lambda p: parse_h(p.name) or 0)
    if not slice_paths:
        raise SystemExit(f"No slice_*.ply in {slices_dir}")
    heights = [parse_h(p.name) for p in slice_paths]
    n_slices = len(slice_paths)

    # 2) merged tracks + translations
    if not csv_dets.exists() or not csv_tr.exists():
        raise SystemExit("Missing merged CSVs. Run the merger first.")
    dets_rows = read_detections_with_clusters(csv_dets)
    tr        = read_translations(csv_tr)

    # group by cluster
    groups = defaultdict(list)
    for r in dets_rows:
        groups[r["cid"]].append(r)
    cids = sorted(groups.keys())
    n_clusters = len(cids)

    # Branch logging state
    branch_streak     = {cid: 0 for cid in cids}     # persistence after change
    branch_fired      = {cid: False for cid in cids}
    branch_height     = {cid: None for cid in cids}

    # Per-cluster clean-run baseline (rolling stats)
    baseline_count    = {cid: 0 for cid in cids}
    baseline_means    = {cid: {"outer_density":0.0, "edge_noise":0.0, "resid_mad":0.0} for cid in cids}
    baseline_mads     = {cid: {"outer_density":1e-6, "edge_noise":1e-6, "resid_mad":1e-6} for cid in cids}

    branch_rows       = []

    # 3) global raster for debug
    tf_global = compute_global_tf_from_slices(slices_dir)

    # 4) per-cluster smoothing + DBH
    dbh_by_cid = {}
    for cid in cids:
        G = groups[cid]
        vals = [g["diam"] for g in G if not math.isnan(g["diam"]) and 1.3 <= g["h"] <= 1.7]
        vals = vals or [g["diam"] for g in G if not math.isnan(g["diam"])]
        dbh_by_cid[cid] = float(np.median(vals)) if vals else float(np.nanmedian([g["diam"] for g in G]) or 0.40)

    smoothers = {}
    for cid in cids:
        G = groups[cid]
        hs = np.array([g["h"] for g in G], np.float32)
        xs = np.array([g["x"] for g in G], np.float32)
        ys = np.array([g["y"] for g in G], np.float32)
        rs = np.array([(g["diam"]/2.0) if not math.isnan(g["diam"]) else np.nan for g in G], np.float32)
        if np.isnan(rs).any():
            fill = np.nanmedian(rs) if not np.isnan(rs).all() else 0.20
            rs = np.where(np.isnan(rs), fill, rs)
        sm = smooth_track(hs, xs, ys, rs)
        if sm is None:
            cx0, cy0 = float(np.median(xs)), float(np.median(ys))
            r0 = float(np.median(rs))
            sm = {"fx": lambda h, cx=cx0: cx,
                  "fy": lambda h, cy=cy0: cy,
                  "fr": lambda h, r=r0:  r}
        smoothers[cid] = sm

    last_refined = {cid: None for cid in cids}

    fieldnames = [
        "cluster_id","height_m","x_corr_m","y_corr_m","radius_m","ok",
        # sweep logging
        "vx_m_per_m","vy_m_per_m","sweep_speed_m_per_m",
        "cx_pred_m","cy_pred_m",
        "drift_from_prior_m","drift_from_last_m","allowed_drift_m",
        # taper/evidence logging
        "r_exp_m","taper_dev","sectors","inliers","occ","void"
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"[refine] slices={n_slices}, clusters={n_clusters}, out_csv={out_csv}")
    print(f"[refine] debug images → {out_dir}")

    iterable = list(zip(slice_paths, heights))
    if _HAS_TQDM:
        pbar = tqdm(iterable, total=len(iterable), desc="Refining slices", unit="slice", ncols=100)
        iterator = pbar
    else:
        iterator = iterable

    t_start = perf_counter()
    with open(out_csv, "w", newline="") as f_csv:
        w = csv.DictWriter(f_csv, fieldnames=fieldnames)
        w.writeheader()
        _flush_file(f_csv)

        for idx, (p, h) in enumerate(iterator, start=1):
            t0 = perf_counter()

            xyz = load_ply_xyz(p)
            if xyz.size == 0:
                draw_slice_debug(np.zeros((0,2), np.float32), tf_global, [], out_dir / f"{p.stem}_debug.png")
                if _HAS_TQDM:
                    pbar.set_postfix_str("empty slice")
                else:
                    print(f"[{idx}/{n_slices}] {p.name}: empty slice", flush=True)
                continue

            dets_for_slice = []
            rows_for_slice = []
            ok_count = 0

            for cid in cids:
                sm = smoothers[cid]
                cx_corr_prior, cy_corr_prior = sm["fx"](h), sm["fy"](h)
                r_prior = sm["fr"](h)
                dx, dy = tr.get(h, (0.0, 0.0))

                ok, cx_corr_ref, cy_corr_ref, r_ref, info, roi, score, slog = corridor_search(
                    xyz, h, sm["fx"], sm["fy"], dx, dy, r_prior, last_refined[cid], dbh_by_cid[cid]
                )

                # Guard: need enough points to trust clutter
                roi_pts = int(info.get("roi_points", 0))

                # Clean ring definition (to build baseline)
                is_clean = (
                    ok and (h >= BRANCH_MIN_HEIGHT_M) and (roi_pts >= MIN_ROI_POINTS_FOR_CLUTTER) and
                    (info.get("sectors", 0) >= CLEAN_MIN_SECTORS) and
                    (info.get("inliers", 0) >= CLEAN_MIN_INLIERS) and
                    (float(info.get("occ", 0.0)) >= CLEAN_MIN_OCC)
                )

                # Update baseline during clean run (rolling robust mean/MAD)
                if not branch_fired[cid] and is_clean:
                    # online update with simple EMA for stability
                    b = baseline_means[cid]; m = baseline_mads[cid]
                    alpha = 0.25  # smoothing
                    for k in ("outer_density","edge_noise","resid_mad"):
                        x = float(info.get(k, 0.0))
                        b[k] = (1-alpha)*b[k] + alpha*x if baseline_count[cid] > 0 else x
                    # track MAD via rolling (approx): update absolute deviations to current mean
                    for k in ("outer_density","edge_noise","resid_mad"):
                        x = float(info.get(k, 0.0))
                        m[k] = (1-alpha)*m[k] + alpha*abs(x - b[k])  # approx MAD scaler
                    baseline_count[cid] += 1

                # If we already have a clean run of sufficient length, watch for clutter change
                branch_here = False
                if (not branch_fired[cid]) and (baseline_count[cid] >= MIN_CLEAN_RUN_SLICES) and (h >= BRANCH_MIN_HEIGHT_M) and ok and (roi_pts >= MIN_ROI_POINTS_FOR_CLUTTER):
                    # z-scores vs baseline
                    z_outer = (float(info.get("outer_density", 0.0)) - baseline_means[cid]["outer_density"]) / max(baseline_mads[cid]["outer_density"], 1e-6)
                    z_edge  = (float(info.get("edge_noise", 0.0))    - baseline_means[cid]["edge_noise"])    / max(baseline_mads[cid]["edge_noise"], 1e-6)
                    z_rug   = (float(info.get("resid_mad", 0.0))     - baseline_means[cid]["resid_mad"])     / max(baseline_mads[cid]["resid_mad"], 1e-6)

                    clutter_change = (z_outer >= CLUTTER_Z_THRESHOLD) or (z_edge >= CLUTTER_Z_THRESHOLD) or (z_rug >= CLUTTER_Z_THRESHOLD)

                    if clutter_change:
                        branch_streak[cid] += 1
                    else:
                        branch_streak[cid] = 0

                    if branch_streak[cid] >= BRANCH_PERSIST_SLICES:
                        branch_fired[cid]  = True
                        branch_height[cid] = h
                        branch_here        = True
                        branch_rows.append({
                            "cluster_id": cid,
                            "branch_height_m": h,
                            "z_outer": float(z_outer),
                            "z_edge":  float(z_edge),
                            "z_rug":   float(z_rug),
                            "outer_density": float(info.get("outer_density", 0.0)),
                            "edge_noise":    float(info.get("edge_noise", 0.0)),
                            "resid_mad":     float(info.get("resid_mad", 0.0)),
                            "baseline_outer": float(baseline_means[cid]["outer_density"]),
                            "baseline_edge":  float(baseline_means[cid]["edge_noise"]),
                            "baseline_rug":   float(baseline_means[cid]["resid_mad"]),
                            "sectors": int(info.get("sectors", 0)),
                            "inliers": int(info.get("inliers", 0)),
                            "occ":     float(info.get("occ", 0.0))
                        })

                rows_for_slice.append({
                    "cluster_id": cid,
                    "height_m":   h,
                    "x_corr_m":   cx_corr_ref if ok else cx_corr_prior,
                    "y_corr_m":   cy_corr_ref if ok else cy_corr_prior,
                    "radius_m":   r_ref       if ok else r_prior,
                    "ok":         int(ok),

                    # sweep
                    "vx_m_per_m":           slog.get("vx", 0.0),
                    "vy_m_per_m":           slog.get("vy", 0.0),
                    "sweep_speed_m_per_m":  slog.get("speed", 0.0),
                    "cx_pred_m":            slog.get("cx_pred", cx_corr_prior),
                    "cy_pred_m":            slog.get("cy_pred", cy_corr_prior),
                    "drift_from_prior_m":   slog.get("drift_from_prior", 0.0),
                    "drift_from_last_m":    slog.get("drift_from_last", 0.0),
                    "allowed_drift_m":      slog.get("allowed_drift", 0.0),

                    # taper/evidence
                    "r_exp_m":    float(info.get("r_exp", 0.0)),
                    "taper_dev":  float(info.get("taper_dev", 0.0)),
                    "sectors":    int(info.get("sectors", 0)),
                    "inliers":    int(info.get("inliers", 0)),
                    "occ":        float(info.get("occ", 0.0)),
                    "void":       float(info.get("void", 0.0)),
                })

                if ok:
                    ok_count += 1

                cx_raw_prior, cy_raw_prior = cx_corr_prior - dx, cy_corr_prior - dy
                if ok:
                    cx_raw_ref, cy_raw_ref = cx_corr_ref - dx, cy_corr_ref - dy
                    last_refined[cid] = (h, cx_corr_ref, cy_corr_ref, r_ref)
                else:
                    cx_raw_ref, cy_raw_ref = cx_raw_prior, cy_raw_prior

                dets_for_slice.append({
                    "cid": cid,
                    "cx_raw_prior": cx_raw_prior, "cy_raw_prior": cy_raw_prior, "r_prior": r_prior,
                    "ok": ok,
                    "cx_raw_ref": cx_raw_ref, "cy_raw_ref": cy_raw_ref, "r_ref": r_ref,
                    "roi_raw": roi,
                    "reason": info.get("reason", "") if isinstance(info, dict) else "",
                    "branch_here": bool(branch_here)
                })

            w.writerows(rows_for_slice)
            _flush_file(f_csv)

            draw_slice_debug(xyz[:, :2], tf_global, dets_for_slice, out_dir / f"{p.stem}_debug.png")

            dt = perf_counter() - t0
            if _HAS_TQDM:
                pbar.set_postfix(ok=f"{ok_count}/{n_clusters}", dt=f"{dt:.1f}s")
            else:
                print(f"[{idx}/{n_slices}] {p.name} | ok {ok_count}/{n_clusters} | {dt:.1f}s", flush=True)
    
    # write branch points once
    branch_csv = merged_dir / "branch_points.csv"
    with open(branch_csv, "w", newline="") as fbp:
        cols = ["cluster_id","branch_height_m",
                "z_outer","z_edge","z_rug",
                "outer_density","edge_noise","resid_mad",
                "baseline_outer","baseline_edge","baseline_rug",
                "sectors","inliers","occ"]
        wbp = csv.DictWriter(fbp, fieldnames=cols)
        wbp.writeheader()
        wbp.writerows(branch_rows)
    print(f"[branch] wrote {len(branch_rows)} branch points → {branch_csv}")


    total_dt = perf_counter() - t_start
    print(f"Done. Wrote refined circles → {out_csv}")
    print(f"Per-slice debug PNGs → {out_dir}")
    print(f"Total time: {total_dt/60.0:.1f} min")

# keep standalone behavior identical
if __name__ == "__main__":
    run(
        slices_dir=SLICES_DIR,
        merged_dir=MERGED_DIR,
        out_dir=OUT_DIR
    )
