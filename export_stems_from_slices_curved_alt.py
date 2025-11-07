# ================================================================
# export_stems_from_slices_curved_alt.py
# ---------------------------------------------------------------
# Build smooth, curved (swept) stem reconstructions from per-slice
# detections, preferring refined circle centers/radii where available.
#
# Workflow:
#   • Reads per-slice PLY point clouds, detection CSVs, and translations.
#   • Groups detections by stem cluster ID and builds height-parametric
#     smoothers (Kalman + MAD outlier removal).
#   • For each stem and slice:
#       – Apply translation offsets.
#       – Use refined correction if available, otherwise fall back
#         to smoothed predictions.
#       – Optionally refine locally via RANSAC circle fit.
#       – Select ring points within ±RING_BAND_M of predicted radius.
#   • Exports one colored PLY per stem and (optionally) a combined
#     “stems_all_curved_colored_alt.ply” visualization.
#
# Inputs:
#   SLICES_DIR  — Directory of slice_*.ply files (ASCII, with xyz/rgb)
#   MERGED_DIR  — Directory containing detection/translation CSVs
#   REFINED_CSV — Refined per-slice centers/radii (optional)
#
# Outputs:
#   OUT_DIR/stem_###.ply             — Per-stem point clouds
#   MERGED_DIR/stems_all_curved_colored_alt.ply (optional)
#
# Notes:
#   • Results are saved separately from the default curved export
#     (“stems_curved”) to avoid overwriting previous runs.
#   • Parameters are configured at top of file; the script can be run
#     standalone or imported as a callable `run()` function.

# Author 
#   Ethan Glynn [University of Sydney]
# ================================================================

import os, csv, re, math, random
from pathlib import Path
from collections import defaultdict
import numpy as np

# === CONFIG (defaults) ===
SLICES_DIR = Path("sliced_ply_outputs_2/plot_annotations_ct03t1b_01")
MERGED_DIR = Path("merged_stems")
OUT_DIR    = MERGED_DIR / "stems_curved"
OUT_DIR    = MERGED_DIR / "stems_curved_alt"   # <- keep results separate
REFINED_CSV = MERGED_DIR / "refined_per_slice_circles.csv"
CSV_DETS    = MERGED_DIR / "detections_with_clusters.csv"
CSV_TR      = MERGED_DIR / "slice_translations.csv"

WRITE_COLORED_ALL = True

# Ring selection / refine
RING_BAND_M = 0.035
ALLOW_RANSAC_REFINE = False      # usually False if you trust refined CSV
RANSAC_TRIALS = 1200
RANSAC_INLIER_TOL_M = 0.02
RANSAC_MIN_ARC_DEG = 180.0
RANSAC_MAX_SAMPLES = 12000

# Smoothing for fallback tracks
KALMAN_Q = 1e-5
KALMAN_R = 1e-3
OUTLIER_SIGMA = 3.5

SLICE_RE = re.compile(r"slice_(?P<h>[0-9]+\.[0-9]+)m\.ply")

# --- taper helper (if you ever want it) ---
def expected_diam_at_h(h: float, dbh_m: float, bh_m: float = 1.5, taper_per_m: float = 0.06) -> float:
    return max(0.05, float(dbh_m * (1.0 + taper_per_m * (bh_m - h))))

# =============== I/O ==================
def parse_h(name: str):
    m = SLICE_RE.match(name)
    return float(m.group("h")) if m else None

def load_ply_xyz_optional_rgb(path: Path):
    """Return (xyz: Nx3 float32, rgb: Nx3 uint8 or None) for ASCII PLY."""
    header = []
    with open(path, "r", errors="ignore") as f:
        for ln in f:
            header.append(ln.rstrip("\n"))
            if ln.strip() == "end_header": break
        props = [ln.split()[-1] for ln in header if ln.startswith("property ")]
        arr = np.loadtxt(path, skiprows=len(header), dtype=np.float32)
    if arr.ndim == 1: arr = arr[None, :]
    idx_x = props.index("x"); idx_y = props.index("y"); idx_z = props.index("z")
    xyz = arr[:, [idx_x, idx_y, idx_z]].astype(np.float32)
    rgb_cols = []
    for name in ("diffuse_red","red"):
        if name in props: rgb_cols.append(props.index(name)); break
    for name in ("diffuse_green","green"):
        if name in props: rgb_cols.append(props.index(name)); break
    for name in ("diffuse_blue","blue"):
        if name in props: rgb_cols.append(props.index(name)); break
    rgb = arr[:, rgb_cols].astype(np.uint8) if len(rgb_cols)==3 else None
    return xyz, rgb

def save_ply_xyzrgb_diffuse(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    n = int(xyz.shape[0])
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar diffuse_red\nproperty uchar diffuse_green\nproperty uchar diffuse_blue\n")
        f.write("end_header\n")
        for (x,y,z),(r,g,b) in zip(xyz, rgb):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

def read_detections_with_clusters(path: Path):
    rows = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            d = row.get("diameter_m", "")
            rows.append({
                "cid": int(row["cluster_id"]),
                "h": float(row["height_m"]),
                "x": float(row["x"]), "y": float(row["y"]),
                "diam": float(d) if d not in ("", "nan") else np.nan
            })
    return rows

def read_translations(path: Path):
    tr = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            tr[float(row["height_m"])] = (float(row["dx_m"]), float(row["dy_m"]))
    return tr

def read_refined_per_slice(path: Path):
    """
    Returns: dict[cid][hkey] = {"x": x_corr, "y": y_corr, "r": radius, "ok": 0/1}
    """
    HKEY = lambda h: round(float(h), 3)
    out = defaultdict(dict)
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            cid = int(row["cluster_id"])
            h   = HKEY(float(row["height_m"]))
            out[cid][h] = {
                "x": float(row["x_corr_m"]),
                "y": float(row["y_corr_m"]),
                "r": float(row["radius_m"]),
                "ok": int(row.get("ok", "1"))
            }
    return out

# =============== utils =================

def palette(n):
    """
    Generate a palette of n distinct RGB colors (uint8).
    Repeats a predefined 20-color base if n > 20.
    """
    base = np.array([
        [230,25,75],[60,180,75],[255,225,25],[0,130,200],[245,130,48],
        [145,30,180],[70,240,240],[240,50,230],[210,245,60],[250,190,190],
        [0,128,128],[230,190,255],[170,110,40],[255,250,200],[128,0,0],
        [170,255,195],[128,128,0],[0,0,128],[128,128,128],[255,215,0]
    ], dtype=np.uint8)
    # Repeat colors if more stems than base colors
    reps = int(np.ceil(n / len(base)))
    return np.vstack([base] * reps)[:n]


def mad(arr):
    """
    Compute the Median Absolute Deviation (robust std-like measure).
    Returns 0.0 if array is empty.
    """
    if arr.size == 0:
        return 0.0
    med = np.median(arr)
    # 1.4826 is a scaling constant to make MAD ~ std for normal data
    return 1.4826 * np.median(np.abs(arr - med))


def kalman_1d(hs, zs, q=KALMAN_Q, r=KALMAN_R):
    """
    1D Kalman smoothing over ordered measurements.
    - hs: independent variable (heights)
    - zs: observed variable (x, y, or radius)
    - q: process noise
    - r: measurement noise base
    Returns smoothed values as np.array.
    """
    x = zs[0]       # initial state estimate
    P = 1.0         # initial covariance
    out = [x]

    # Robustify R using MAD of residuals
    resid = zs - np.median(zs)
    R = max(r, (mad(resid) or r))

    # Recursive Kalman filter update for each observation
    for k in range(1, len(zs)):
        x_pred, P_pred = x, P + q
        K = P_pred / (P_pred + R)               # Kalman gain
        x = x_pred + K * (zs[k] - x_pred)       # state update
        P = (1 - K) * P_pred                    # covariance update
        out.append(x)

    return np.array(out, dtype=np.float32)


def smooth_track(heights, xs, ys, rs):
    """
    Smooth and clean (x, y, r) trajectories across height slices.
    - Removes outliers using MAD-based sigma gating.
    - Fills missing radii with nanmedian.
    - Applies 1D Kalman smoothing independently to x, y, r.
    - Returns interpolation functions for each variable.
    """
    # Sort observations by height
    order = np.argsort(heights)
    h = np.asarray(heights, np.float32)[order]
    x = np.asarray(xs, np.float32)[order]
    y = np.asarray(ys, np.float32)[order]
    r = np.asarray(rs, np.float32)[order]
    if h.size == 0:
        return None

    # Remove outliers using robust MAD thresholds
    keep = np.ones(h.shape[0], dtype=bool)
    med_x = np.median(x[keep]); mad_x = mad(x[keep]) or 1e-3
    keep &= (np.abs(x - med_x) <= OUTLIER_SIGMA * mad_x)
    med_y = np.median(y[keep]); mad_y = mad(y[keep]) or 1e-3
    keep &= (np.abs(y - med_y) <= OUTLIER_SIGMA * mad_y)

    # Filter data after outlier rejection
    h, x, y, r = h[keep], x[keep], y[keep], r[keep]
    if h.size == 0:
        return None

    # Replace NaN radii with median fallback
    if np.isnan(r).any():
        fill = np.nanmedian(r) if not np.isnan(r).all() else 0.20
        r = np.where(np.isnan(r), fill, r)

    # Kalman smooth x, y, r separately
    xs_s = kalman_1d(h, x)
    ys_s = kalman_1d(h, y)
    rs_s = kalman_1d(h, r)

    # Build linear interpolation functions for each smoothed dimension
    def interp(hs, vs):
        hs = np.asarray(hs, np.float32)
        vs = np.asarray(vs, np.float32)
        def f(hq):
            if hs.size == 1:
                return float(vs[0])
            return float(np.interp(hq, hs, vs))
        return f

    return {"fx": interp(h, xs_s),
            "fy": interp(h, ys_s),
            "fr": interp(h, rs_s)}


def sector_occupancy_points(xy: np.ndarray, cx: float, cy: float, r: float,
                            band_m: float = RING_BAND_M, n_sectors: int = 24) -> float:
    """
    Compute the angular occupancy fraction of points around a circle.

    Parameters:
        xy: Nx2 array of (x, y) points.
        cx, cy: Center of the candidate ring.
        r: Expected ring radius (in metres).
        band_m: Radial tolerance around r for accepting points.
        n_sectors: Number of angular bins to divide the circle into.

    Returns:
        Fraction of sectors (0–1) that contain at least one point within the ring band.
        Higher values → more complete, uniform ring coverage.
    """
    if xy.size == 0:
        return 0.0

    # Compute point offsets and distances from center
    dx = xy[:, 0] - cx
    dy = xy[:, 1] - cy
    dist = np.hypot(dx, dy)

    # Select points near the expected ring radius
    band = np.abs(dist - r) <= band_m
    if not np.any(band):
        return 0.0

    # Convert selected points to angles (0–360°)
    ang = (np.degrees(np.arctan2(dy[band], dx[band])) + 360.0) % 360.0

    # Bin points into equal angular sectors
    bins = np.floor(ang / (360.0 / n_sectors)).astype(np.int32)
    counts = np.bincount(bins, minlength=n_sectors)

    # Fraction of non-empty sectors = ring completeness metric
    return float(np.mean(counts > 0))


def ring_voidness_ratio(xy: np.ndarray, cx: float, cy: float, r: float,
                        band_m: float = RING_BAND_M, inner_frac: float = 0.65) -> float:
    """
    Estimate how 'solid' or 'hollow' a detected ring is.

    Parameters:
        xy: Nx2 array of (x, y) points.
        cx, cy: Circle center.
        r: Ring radius (metres).
        band_m: Radial band width for defining ring points.
        inner_frac: Fraction of r defining inner core region.

    Returns:
        Ratio = (# of ring-band points) / (# of inner-core points)
        • Higher → thicker/more filled ring
        • Lower → thinner or hollow ring
    """
    if xy.size == 0:
        return 0.0

    dx = xy[:, 0] - cx
    dy = xy[:, 1] - cy
    dist = np.hypot(dx, dy)

    # Points near the ring vs points inside the core
    ring = np.abs(dist - r) <= band_m
    inner = dist <= max(0.01, inner_frac * r)

    n_ring = int(np.count_nonzero(ring))
    n_inner = int(np.count_nonzero(inner))

    # Avoid divide-by-zero; 1.0 denominator min
    return n_ring / float(max(1, n_inner))


def fit_circle_taubin(pts):
    """
    Fit a circle to 2D points using the Taubin algebraic method.

    Parameters:
        pts: Nx2 array of (x, y) coordinates.

    Returns:
        (cx, cy, r): Circle center and radius.
        None if degenerate (ill-conditioned) configuration.
    """
    x = pts[:, 0]
    y = pts[:, 1]

    # Translate to mean-centered coordinates
    x_m = np.mean(x)
    y_m = np.mean(y)
    u = x - x_m
    v = y - y_m

    # Compute required moment sums for the linear system
    Suu = np.dot(u, u)
    Svv = np.dot(v, v)
    Suv = np.dot(u, v)
    Suuu = np.dot(u, u * u)
    Svvv = np.dot(v, v * v)
    Suvv = np.dot(u, v * v)
    Svuu = np.dot(v, u * u)

    # Solve linear system for circle center offsets (uc, vc)
    A = np.array([[Suu, Suv], [Suv, Svv]], dtype=np.float64)
    b = 0.5 * np.array([Suuu + Suvv, Svvv + Svuu], dtype=np.float64)
    det = np.linalg.det(A)
    if abs(det) < 1e-12:
        return None  # Degenerate (e.g., collinear) points

    uc, vc = np.linalg.solve(A, b)

    # Recover center and mean radius in original coordinates
    cx = x_m + uc
    cy = y_m + vc
    r = float(np.mean(np.hypot(x - cx, y - cy)))

    return cx, cy, r

def ransac_circle(pts, trials=RANSAC_TRIALS, tol=RANSAC_INLIER_TOL_M, 
                  min_arc_deg=RANSAC_MIN_ARC_DEG, seed=123):
    """
    Fit a circle to 2D points using RANSAC (Random Sample Consensus).

    Parameters:
        pts : np.ndarray
            Nx2 array of (x, y) coordinates.
        trials : int
            Number of random samples to draw.
        tol : float
            Distance tolerance (metres) for a point to be considered an inlier.
        min_arc_deg : float
            Minimum angular coverage (degrees) of inlier points to accept a fit.
        seed : int
            Random seed for reproducibility.

    Returns:
        best : tuple or None
            (cx, cy, r, inliers_mask) for the best-fit circle,
            or None if no valid circle is found.
    """
    # Need at least 3 points to define a circle, 6 for stability
    if pts.shape[0] < 6:
        return None

    rng = random.Random(seed)
    best, best_score = None, -1
    idxs = list(range(pts.shape[0]))

    # --- Main RANSAC loop ---
    for _ in range(trials):
        # Randomly sample 3 unique points
        i, j, k = rng.sample(idxs, 3)
        p1, p2, p3 = pts[i], pts[j], pts[k]

        # Solve for circle center using perpendicular bisector equations
        A = 2 * np.array([
            [p2[0] - p1[0], p2[1] - p1[1]],
            [p3[0] - p1[0], p3[1] - p1[1]]
        ], dtype=np.float64)
        b = np.array([
            [p2[0]**2 - p1[0]**2 + p2[1]**2 - p1[1]**2],
            [p3[0]**2 - p1[0]**2 + p3[1]**2 - p1[1]**2]
        ], dtype=np.float64)

        # Skip degenerate configurations (e.g., collinear points)
        if abs(np.linalg.det(A)) < 1e-12:
            continue

        # Compute circle parameters from the linear system
        c = np.linalg.solve(A, b).ravel()
        cx, cy = c[0], c[1]
        r = np.hypot(p1[0] - cx, p1[1] - cy)

        # Compute residual distances of all points to the fitted circle
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        resid = np.abs(d - r)
        inliers = resid <= tol
        n_in = int(inliers.sum())
        if n_in < 12:
            continue  # Not enough inliers to be a stable fit

        # Compute angular coverage of inliers
        ang = (np.degrees(np.arctan2(pts[inliers, 1] - cy, pts[inliers, 0] - cx)) + 360) % 360
        ang = np.sort(ang)
        gaps = np.diff(np.r_[ang, ang[0] + 360])
        coverage = 360.0 - float(gaps.max())
        if coverage < min_arc_deg:
            continue

        # Score combines number of inliers and coverage proportion
        score = n_in * (coverage / 360.0)
        if score > best_score:
            best_score = score
            best = (cx, cy, r, inliers)

    return best


# =============== run() =================
def run(slices_dir: Path = SLICES_DIR,
        merged_dir: Path = MERGED_DIR,
        out_dir: Path = OUT_DIR,
        refined_csv: Path = REFINED_CSV,
        csv_dets: Path = CSV_DETS,
        csv_tr: Path = CSV_TR,
        write_colored_all: bool = WRITE_COLORED_ALL,
        allow_ransac_refine: bool = ALLOW_RANSAC_REFINE):
    """
    Export curved (swept) stems as per-stem PLYs by selecting ring-band points per slice.

    Inputs (files/dirs):
        slices_dir    : Directory of slice_*.ply point clouds (ASCII PLY), one per height.
        merged_dir    : Directory containing merged CSVs and where combined outputs are written.
        out_dir       : Destination for per-stem PLYs.

        refined_csv   : CSV with refined per-slice circle fits per cluster_id (x_corr_m, y_corr_m, radius_m, ok).
                        If present and ok==1, this is used as primary source for (cx, cy, r).
        csv_dets      : CSV of detections_with_clusters (cluster_id, height_m, x, y, diameter_m).
                        Used to build fallback smoothers if refined row is missing.
        csv_tr        : CSV of per-slice translations (height_m → dx_m, dy_m) to un-translate corrected centers.

    Behaviour:
        1) Load all slice PLYs (xyz and optional RGB).
        2) Read per-slice translations and refined per-slice circles (if available).
        3) For each cluster_id (stem):
           - Prefer refined centers/radii at each height; otherwise use a smoothed fallback
             built from detections (Kalman + robust outlier gating).
           - Optionally run local RANSAC refine around predicted center (if allow_ransac_refine).
           - Select a thin ring-band of points |dist - r| <= RING_BAND_M for that slice.
        4) Save one PLY per stem with per-point colors (slice RGB if available, else palette color).
        5) Optionally also save a single combined colored PLY of all stems.

    Notes:
        - Uses translations to map corrected (x_corr_m, y_corr_m) back to raw slice coordinates.
        - RANSAC refine is off by default; enable only if refined_csv is absent/unreliable.
        - No changes to original algorithmic thresholds/logic.
    """
    # Ensure output directories exist
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    # Discover & load slices (sorted by height)
    slice_paths = sorted([p for p in slices_dir.glob("slice_*.ply")], key=lambda p: parse_h(p.name) or 0)
    if not slice_paths: 
        raise SystemExit(f"No slice_*.ply in {slices_dir}")
    heights = [parse_h(p.name) for p in slice_paths]

    # Cache per-slice xyz (+ optional rgb)
    slices_xyz, slices_rgb = {}, {}
    for p, h in zip(slice_paths, heights):
        xyz, rgb = load_ply_xyz_optional_rgb(p)
        slices_xyz[h] = xyz
        slices_rgb[h] = rgb

    # Required merged CSVs
    if not csv_dets.exists() or not csv_tr.exists():
        raise SystemExit("Missing merged CSVs. Run the merger first.")
    dets = read_detections_with_clusters(csv_dets)  # fallback detection rows
    tr   = read_translations(csv_tr)                # per-height (dx, dy)

    # Optional refined per-slice circles (preferred when ok==1)
    refined = read_refined_per_slice(refined_csv) if refined_csv.exists() else defaultdict(dict)
    HKEY = lambda h: round(float(h), 3)  # stable height key

    # Group detections by cluster_id → stems
    groups = defaultdict(list)
    for r in dets:
        groups[r["cid"]].append(r)
    cids = sorted(groups.keys())
    colors = palette(len(cids))  # fallback color per stem

    # Build fallback smoothers (cx(h), cy(h), r(h)) for each stem
    smoothers = {}
    for cid in cids:
        G = groups[cid]
        hs = np.array([g["h"] for g in G], np.float32)
        xs = np.array([g["x"] for g in G], np.float32)
        ys = np.array([g["y"] for g in G], np.float32)
        rs = np.array([(g["diam"]/2.0) if not np.isnan(g["diam"]) else np.nan for g in G], np.float32)

        # Fill missing radii to keep the smoother defined
        if np.isnan(rs).any():
            fill = np.nanmedian(rs) if not np.isnan(rs).all() else 0.20
            rs = np.where(np.isnan(rs), fill, rs)

        sm = smooth_track(hs, xs, ys, rs)
        # Degenerate: fall back to constant functions around medians
        if sm is None:
            cx0, cy0 = float(np.median(xs)), float(np.median(ys))
            r0 = float(np.median(rs)) if rs.size else 0.20
            sm = {"fx": (lambda _h, cx=cx0: cx),
                  "fy": (lambda _h, cy=cy0: cy),
                  "fr": (lambda _h, rr=r0: rr)}
        smoothers[cid] = sm

    all_pts = []
    all_rgb = []
    used_refined, used_fallback = 0, 0

    # --- Per-stem assembly ---
    for idx, cid in enumerate(cids):
        stem_pts = []
        stem_cols = []
        color = colors[idx]  # palette fallback

        # Accumulate ring-band points across all heights
        for h in heights:
            xyz = slices_xyz[h]
            if xyz.size == 0:
                continue

            # Undo per-slice translation for corrected centers
            dx, dy = tr.get(h, (0.0, 0.0))

            # Prefer refined per-slice circle if valid; else smoother prediction
            use_ref = False
            row = refined.get(cid, {}).get(HKEY(h))
            if row and row["ok"] == 1 and np.isfinite(row["r"]):
                cx_corr, cy_corr, r_pred = row["x"], row["y"], row["r"]
                use_ref = True
            else:
                sm = smoothers[cid]
                cx_corr, cy_corr, r_pred = sm["fx"](h), sm["fy"](h), sm["fr"](h)

            # Map corrected center back to raw coordinates
            cx_raw, cy_raw = cx_corr - dx, cy_corr - dy
            xy = xyz[:, :2]

            # Optional local RANSAC refine around the predicted center/radius
            if allow_ransac_refine:
                roi_R = max(0.36, r_pred + 0.22)
                roi_mask = (np.hypot(xy[:, 0] - cx_raw, xy[:, 1] - cy_raw) <= roi_R)
                cand = xy[roi_mask] if np.any(roi_mask) else xy
                if cand.shape[0] > RANSAC_MAX_SAMPLES:
                    sel = np.random.choice(cand.shape[0], RANSAC_MAX_SAMPLES, replace=False)
                    cand = cand[sel]
                fit = ransac_circle(cand, trials=RANSAC_TRIALS,
                                    tol=RANSAC_INLIER_TOL_M,
                                    min_arc_deg=RANSAC_MIN_ARC_DEG)
                if fit is not None:
                    cx_raw, cy_raw, r_pred, _ = fit

            # Final ring-band selection |dist - r| <= RING_BAND_M
            dist = np.hypot(xy[:, 0] - cx_raw, xy[:, 1] - cy_raw)
            band = np.abs(dist - r_pred) <= RING_BAND_M
            if not np.any(band):
                continue

            # Collect selected points (and per-point colors)
            pts_here = xyz[band]
            stem_pts.append(pts_here)
            if slices_rgb[h] is not None:
                stem_cols.append(slices_rgb[h][band])
            else:
                stem_cols.append(np.tile(color, (int(band.sum()), 1)))

            used_refined += int(use_ref)
            used_fallback += int(not use_ref)

        # Emit per-stem PLY
        if stem_pts:
            pts = np.vstack(stem_pts)
            cols = np.vstack(stem_cols)
            save_ply_xyzrgb_diffuse(out_dir / f"stem_{cid:03d}.ply", pts, cols)
            if write_colored_all:
                all_pts.append(pts)
                all_rgb.append(cols)

    # Optionally emit combined colored PLY of all stems
    if write_colored_all and all_pts:
        save_ply_xyzrgb_diffuse(
            merged_dir / "stems_all_curved_colored_alt.ply",
            np.vstack(all_pts),
            np.vstack(all_rgb)
        )

    # Summary / counters
    print(f"Done. Per-stem curved PLYs → {out_dir}")
    if write_colored_all:
        print(f"Combined colored PLY → {merged_dir/'stems_all_curved_colored_alt.ply'}")
    print(f"Used refined rows: {used_refined}, fallback rows: {used_fallback}")


# keep standalone behavior
if __name__ == "__main__":
    run()  # uses the defaults above
