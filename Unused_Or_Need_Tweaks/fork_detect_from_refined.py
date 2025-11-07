# ================================================================
# fork_detect_from_refined.py — Detect child forks from refined stems
# ---------------------------------------------------------------
# Purpose
#   Starting from refined per-slice stem circles (parents), search local
#   annuli for additional rings that persist across consecutive slices and
#   promote them as "child" clusters (forks/headers).
#
# What it does
#   • Loads refined parent circles (x_corr,y_corr,r) and per-slice XY translations.
#   • Builds one fixed global raster over all slices and rasterizes each slice
#     to an occupancy heatmap.
#   • Around each parent, scans an annulus (r_ref + inner_pad .. + outer_pad).
#   • Runs a lightweight ring detector (Canny + Hough) and scores “ringness”
#     via gradient alignment.
#   • Filters candidates by (a) separation from parent and (b) diameter similarity.
#   • Tracks candidates per parent across height; promotes those seen in
#     ≥ CONFIRM_CONSEC consecutive slices.
#   • Writes new child rows with corrected coordinates (x_corr,y_corr) and
#     links them to the parent cluster_id.
#
# Inputs
#   • slices_dir: folder of ASCII PLY slices named slice_<h>m.ply
#   • merged_dir:
#       - refined_per_slice_circles.csv  (expects: cluster_id, height_m, x_corr_m, y_corr_m, radius_m, ok)
#       - slice_translations.csv         (expects: height_m, dx_m, dy_m)
#
# Outputs
#   • out_csv: CSV with detected forks (children):
#       cluster_id, height_m, x_corr_m, y_corr_m, radius_m, ok, parent_cid
#
# Tunables (safe defaults for double/triple headers)
#   • FORK_MIN_SEP_M         : minimum parent→child center separation
#   • FORK_DIAM_TOL_FRAC     : relative diameter tolerance vs parent
#   • ANNULUS_INNER/OUTER    : search band around parent radius (metres)
#   • RINGNESS_MIN           : gradient-alignment score threshold
#   • GRAD_ALIGN_TOL_DEG     : degrees allowed between radial and gradient dirs
#   • CONFIRM_CONSEC         : # consecutive slices required to promote a child
#   • MAX_CHILDREN_PER_SLICE : per-parent cap to keep outputs sensible
#
# Usage (programmatic)
#   from pathlib import Path
#   run(
#       slices_dir=Path(".../slices"),
#       merged_dir=Path(".../merged_stems"),
#       out_csv=Path(".../forks_detected.csv")
#   )
#
# Assumptions / Notes
#   • PLY files are ASCII with an 'end_header' line; only XYZ are read.
#   • Heights are parsed from filenames: slice_<h>m.ply (float metres).
#   • “Corrected” coords = RAW coords + per-slice translation (dx,dy).
#   • Child cluster_ids are generated from a high range (starting at 10_000)
#     to avoid collisions with existing parent ids.
#
# Dependencies
#   Python 3.9+, numpy, opencv-python
#
# Author
#   Ethan Glynn [University of Sydney]
# ================================================================

from pathlib import Path
import csv, math, numpy as np, cv2, re
from collections import defaultdict

# --- Tunables (calibrated defaults for detecting double/triple headers) ---
# Minimum allowed separation between a parent stem center and a candidate child (metres).
FORK_MIN_SEP_M       = 0.22

# Diameter similarity gate for a child vs its parent:
# accept if |d_child - d_parent| / d_parent <= FORK_DIAM_TOL_FRAC
FORK_DIAM_TOL_FRAC   = 0.25

# Search band around the parent radius (metres): we scan an annulus
# from (r_parent + ANNULUS_INNER_PAD_M) to (r_parent + ANNULUS_OUTER_PAD_M).
ANNULUS_INNER_PAD_M  = 0.10
ANNULUS_OUTER_PAD_M  = 0.40

# Width (in pixels) of the radial band used to score ringness on edges.
RING_BAND_PX         = 3.0

# Minimum “ringness” score (gradient alignment ratio) to keep a candidate.
RINGNESS_MIN         = 0.32

# Maximum angular mismatch (degrees) between radial direction and image gradient.
GRAD_ALIGN_TOL_DEG   = 36.0

# Number of consecutive slices a candidate must appear in before promotion to “child”.
CONFIRM_CONSEC       = 2

# Safety cap: at most this many children are kept per parent per slice.
MAX_CHILDREN_PER_SLICE = 3

# Expected slice filename pattern: slice_<height>m.ply (height in metres).
SLICE_RE = re.compile(r"slice_(?P<h>[0-9]+\.[0-9]+)m\.ply")


# --- I/O helpers (format- and schema-aligned with the rest of your code) ---
def parse_h(name):
    """
    Extract height (metres) from a slice filename like 'slice_1.50m.ply'.
    Returns float height if matched, otherwise None.
    """
    m = SLICE_RE.match(name)
    return float(m.group("h")) if m else None


def load_ply_xyz(path: Path) -> np.ndarray:
    """
    Load an ASCII PLY and return an (N,3) float32 array of XYZ coordinates.
    - Skips the header up to 'end_header'.
    - If the file has a single XYZ row, returns shape (1,3).
    - If empty, returns shape (0,3).
    """
    with open(path, "r", errors="ignore") as f:
        header = []
        for ln in f:
            header.append(ln)
            if ln.strip() == "end_header":
                break
        arr = np.loadtxt(path, skiprows=len(header), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, :3].astype(np.float32) if arr.size else np.zeros((0, 3), np.float32)


def read_translations(path: Path):
    """
    Read per-slice XY translations (metres) keyed by height.
    CSV schema: height_m, dx_m, dy_m
    Returns: dict[height_m] -> (dx_m, dy_m)
    """
    tr = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            tr[float(row["height_m"])] = (float(row["dx_m"]), float(row["dy_m"]))
    return tr


def read_refined(path: Path):
    """
    Load refined per-slice parent circles, grouped by cluster_id and height.
    CSV expected fields:
      - cluster_id, height_m, x_corr_m, y_corr_m, radius_m, ok (optional; defaults to 1)
    Returns:
      dict[int cluster_id] -> dict[rounded height -> {x,y,r,ok}]
    """
    HKEY = lambda h: round(float(h), 3)
    by_cid = defaultdict(dict)
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            cid = int(row["cluster_id"])
            h = HKEY(float(row["height_m"]))
            by_cid[cid][h] = {
                "x": float(row["x_corr_m"]),
                "y": float(row["y_corr_m"]),
                "r": float(row["radius_m"]),
                "ok": int(row.get("ok", "1")),
            }
    return by_cid


# --- Global XY → fixed image (shared convention with raster helpers elsewhere) ---
def compute_global_tf_from_slices(slices_dir: Path, pixel_size=0.02, padding=0.5):
    """
    Build a single global image transform that covers all slice extents,
    with extra padding, at the requested pixel resolution.
    Returns:
      dict with {xmin, ymax, pixel_size, W, H}
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
    xmin = float(min(xs_min) - padding); xmax = float(max(xs_max) + padding)
    ymin = float(min(ys_min) - padding); ymax = float(max(ys_max) + padding)
    W = int(np.ceil((xmax - xmin) / pixel_size))
    H = int(np.ceil((ymax - ymin) / pixel_size))
    return {"xmin": xmin, "ymax": ymax, "pixel_size": pixel_size, "W": W, "H": H}


def world_point_to_pixel(tf, x, y):
    """
    Map a world-space point (metres) into global image pixel coordinates (float).
    """
    return (x - tf["xmin"]) / tf["pixel_size"], (tf["ymax"] - y) / tf["pixel_size"]


def rasterize_heatmap_fixed(xy: np.ndarray, tf: dict) -> np.ndarray:
    """
    Rasterize a set of XY points into a fixed global occupancy heatmap:
    - Votes points into pixels
    - Gaussian blur
    - Scales to uint8 [0,255] (or zeros if empty)
    """
    H, W = tf["H"], tf["W"]
    heat = np.zeros((H, W), np.float32)
    px = (xy[:, 0] - tf["xmin"]) / tf["pixel_size"]
    py = (tf["ymax"] - xy[:, 1]) / tf["pixel_size"]
    ij = np.rint(np.stack([px, py], 1)).astype(np.int32)
    ij[:, 0] = np.clip(ij[:, 0], 0, W - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, H - 1)
    np.add.at(heat, (ij[:, 1], ij[:, 0]), 1.0)
    heat = cv2.GaussianBlur(heat, (0, 0), 1.2)
    return (255.0 * (heat / heat.max())).astype(np.uint8) if heat.max() > 0 else np.zeros_like(heat, np.uint8)


# --- Ring detector mini (edge + Hough; reuses thresholds defined above) ---
def edgeify(occ):
    """
    Compute edges and per-pixel gradient direction for a uint8 occupancy map.
    Returns: (edges uint8, grad_dir float32 radians)
    """
    blurred = cv2.GaussianBlur(occ, (5, 5), 1.0)
    edges = cv2.Canny(blurred, 20, 60, L2gradient=True)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    return edges, np.arctan2(gy, gx)


def score_ringness(edges, grad_dir, cx, cy, r_px, radial_tol_px=RING_BAND_PX, align_tol_deg=GRAD_ALIGN_TOL_DEG):
    """
    Measure how “ring-like” a candidate circle is by checking whether edge
    gradients align with the local radial direction within an annular band.
    Returns: ratio in [0,1].
    """
    ys, xs = np.nonzero(edges)
    dx = xs - cx; dy = ys - cy
    dist = np.hypot(dx, dy)
    band = np.abs(dist - r_px) <= radial_tol_px
    if not np.any(band):
        return 0.0
    radial = np.arctan2(dy[band], dx[band])
    g = grad_dir[ys[band], xs[band]]
    delta = np.abs(np.unwrap(radial) - np.unwrap(g))
    delta = np.minimum(delta, 2 * np.pi - delta)
    ok = (np.degrees(delta) <= align_tol_deg).sum()
    return float(ok) / float(np.count_nonzero(band))


def detect_rings_in_patch(patch, pixel_size, min_diam_m, max_diam_m):
    """
    Detect circular rings in a cropped occupancy patch using:
      1) Canny/Sobel for edges + gradients,
      2) HoughCircles for circle proposals,
      3) ringness scoring with gradient alignment.
    Returns: list of (cx, cy, r, ringness), sorted by ringness desc.
    """
    H, W = patch.shape
    minR = max(3, int((min_diam_m / pixel_size) / 2.0))
    maxR = int((max_diam_m / pixel_size) / 2.0)
    if maxR <= minR:
        return []
    edges, gdir = edgeify(patch)
    circles = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT, dp=1.0,
        minDist=12.0, param1=120, param2=14,
        minRadius=minR, maxRadius=maxR
    )
    out = []
    if circles is not None:
        for c in circles[0]:
            cx, cy, r = float(c[0]), float(c[1]), float(c[2])
            rn = score_ringness(edges, gdir, cx, cy, r)
            if rn >= RINGNESS_MIN:
                out.append((cx, cy, r, rn))
    out.sort(key=lambda t: t[3], reverse=True)
    return out


# --- Simple NMS in world space (merge near-duplicate children) ---
def nms_world(dets, min_dist_m=0.15, diam_tol_frac=0.20):
    """
    Non-maximum suppression across world-space detections:
    - Sort by 'ringness' (desc)
    - Suppress a detection if it is both close in center (min_dist_m) and
      similar in diameter (fractional tolerance) to a kept detection.
    Returns: filtered list.
    """
    dets = sorted(dets, key=lambda d: d.get("ringness", 0.0), reverse=True)
    kept = []
    for d in dets:
        keep = True
        for k in kept:
            dx = d["center_x_m"] - k["center_x_m"]
            dy = d["center_y_m"] - k["center_y_m"]
            if (dx * dx + dy * dy) ** 0.5 < min_dist_m:
                if abs(d["diameter_m"] - k["diameter_m"]) <= diam_tol_frac * max(d["diameter_m"], k["diameter_m"]):
                    keep = False
                    break
        if keep:
            kept.append(d)
    return kept

def run(slices_dir: Path, merged_dir: Path, out_csv: Path):
    """
    Detect and confirm forked child stems (double/triple headers) relative to refined parent circles.

    Inputs
    -------
    slices_dir : Path
        Folder containing slice_*.ply point-cloud slices (ASCII PLY).
    merged_dir : Path
        Folder containing:
          - refined_per_slice_circles.csv : corrected parent circles per slice (x_corr_m, y_corr_m, radius_m, ok)
          - slice_translations.csv         : per-slice (height_m) world<->raw XY translations (dx_m, dy_m)
    out_csv : Path
        Output CSV path where confirmed children will be written (cluster_id, height_m, x_corr_m, y_corr_m, radius_m, ok, parent_cid).

    What it does
    ------------
    1) Builds a global raster transform over all slices (for consistent pixelization).
    2) For each height/slice:
         - Rasterizes the XY points to a heatmap (occ).
         - Looks up the raw↔corrected XY translation (dx, dy) for that height.
         - For every parent cluster that has an 'ok' refined circle at this height:
              a) Defines an annular ROI (outside the parent radius).
              b) Detects ring candidates (potential children) in the ROI.
              c) Filters by separation and diameter similarity to the parent.
              d) Applies world-space NMS and caps children per slice.
              e) Tracks candidates across consecutive slices (‘pending’).
              f) Promotes a track to a real child when it persists CONFIRM_CONSEC slices.
    3) Saves all promoted children to out_csv (with corrected coordinates applied).

    Notes
    -----
    - 'pending' stores transient detections keyed by parent cluster:
        { parent_cid : { tmp_track_id : {x, y, r, streak, last_h} } }
      A candidate must reach 'streak' == CONFIRM_CONSEC to be promoted.
    - Child coordinates are written in corrected space (x_corr_m, y_corr_m),
      i.e., raw detection plus the per-slice translation (dx, dy).
    """

    # ---------- Load inputs ----------
    refined = read_refined(merged_dir / "refined_per_slice_circles.csv")   # parent circles per height (corrected)
    tr = read_translations(merged_dir / "slice_translations.csv")          # per-height (dx, dy) for raw↔corrected mapping
    tf = compute_global_tf_from_slices(slices_dir)                          # global raster transform for all slices

    # ---------- Cache slice paths and ordered heights ----------
    slice_paths = {parse_h(p.name): p for p in slices_dir.glob("slice_*.ply")}
    heights = sorted(slice_paths.keys())

    # ---------- Promotion bookkeeping ----------
    # pending: per-parent transient children, promoted when seen in CONFIRM_CONSEC consecutive slices.
    pending = defaultdict(dict)  # parent_cid -> {tmp_id: {"streak":k, "last_h":h, "x":..., "y":..., "r":...}}
    next_child_cid = 10_000      # child cluster id range (kept separate from parents)

    rows_out = []

    # ---------- Iterate over slices by height ----------
    for h in heights:
        p = slice_paths[h]
        xyz = load_ply_xyz(p)
        if xyz.size == 0:
            continue

        # Occupancy heatmap for the current slice in the global frame.
        occ = rasterize_heatmap_fixed(xyz[:, :2], tf)

        # Per-slice raw<->corrected XY translation (default to 0 if missing).
        dx, dy = tr.get(h, (0.0, 0.0))

        # ---------- For each parent cluster that has a refined circle at this height ----------
        for cid, by_h in refined.items():
            row = by_h.get(round(h, 3))
            if not row or row["ok"] != 1:
                continue

            # Parent (corrected coords) and its radius.
            cx_corr, cy_corr, r_ref = row["x"], row["y"], row["r"]

            # Convert to RAW coords for searching in the rasterized occupancy.
            cx_raw, cy_raw = cx_corr - dx, cy_corr - dy

            # ----- Define ROI patch: annulus outside the parent radius -----
            cx_px, cy_px = world_point_to_pixel(tf, cx_raw, cy_raw)
            ann_inner = r_ref + ANNULUS_INNER_PAD_M
            ann_outer = r_ref + ANNULUS_OUTER_PAD_M
            r_px = ann_outer / tf["pixel_size"]

            # Crop a square around the annulus outer radius; detection function will score candidates.
            x0 = int(max(0, np.floor(cx_px - r_px))); y0 = int(max(0, np.floor(cy_px - r_px)))
            x1 = int(min(tf["W"], np.ceil(cx_px + r_px))); y1 = int(min(tf["H"], np.ceil(cy_px + r_px)))
            patch = occ[y0:y1, x0:x1]

            # ----- Detect ring candidates in the patch (loose diameter guard; refined later) -----
            cand = detect_rings_in_patch(
                patch, tf["pixel_size"],
                min_diam_m=max(0.05, 2*ann_inner - 2*r_ref),   # lower bound loosely beyond parent rim
                max_diam_m=2*(r_ref*(1.0+FORK_DIAM_TOL_FRAC))  # upper bound from parent + tolerance
            )

            # ----- Filter: enforce separation from parent center and diameter similarity -----
            picks = []
            for (cx, cy, r, rn) in cand:
                # Shift back from patch coords to global pixel coords.
                cx += x0; cy += y0

                # Convert to world (metres).
                wx = tf["xmin"] + cx*tf["pixel_size"]
                wy = tf["ymax"] - cy*tf["pixel_size"]

                # Separation test vs parent raw center.
                sep = math.hypot(wx - cx_raw, wy - cy_raw)
                if sep < FORK_MIN_SEP_M:
                    continue

                # Diameter similarity test vs parent.
                d_child = 2*r*tf["pixel_size"]
                d_parent = 2*r_ref
                if abs(d_child - d_parent) / max(d_parent, 1e-6) > FORK_DIAM_TOL_FRAC:
                    continue

                # Keep candidate (world coords + ringness).
                picks.append({
                    "center_x_m": wx, "center_y_m": wy,
                    "radius_m": r*tf["pixel_size"],
                    "diameter_m": d_child,
                    "ringness": rn
                })

            # Merge near-duplicate children (world-space NMS) and cap count per slice.
            picks = nms_world(
                picks,
                min_dist_m=FORK_MIN_SEP_M*0.8,
                diam_tol_frac=FORK_DIAM_TOL_FRAC
            )[:MAX_CHILDREN_PER_SLICE]

            # ---------- Update 'pending' tracks for this parent and possibly promote ----------
            pend = pending[cid]
            used = set()

            # Associate each pick to the nearest pending track (within FORK_MIN_SEP_M), else start new.
            for d in picks:
                best_id, best_dist = None, 1e9
                for tid, s in pend.items():
                    dist = math.hypot(d["center_x_m"] - s["x"], d["center_y_m"] - s["y"])
                    if dist < best_dist:
                        best_id, best_dist = tid, dist

                if best_id is not None and best_dist <= FORK_MIN_SEP_M:
                    # Continue existing track (increase streak)
                    s = pend[best_id]
                    s.update({"x": d["center_x_m"], "y": d["center_y_m"], "r": d["radius_m"],
                              "streak": s["streak"] + 1, "last_h": h})
                    used.add(best_id)
                else:
                    # Start a new temporary track id at this height.
                    tid = f"t{len(pend)+1}_{int(h*1000)}"
                    pend[tid] = {"x": d["center_x_m"], "y": d["center_y_m"], "r": d["radius_m"],
                                 "streak": 1, "last_h": h}
                    used.add(tid)

            # Drop stale tracks that weren’t matched at this height (simple one-slice patience).
            for tid in [k for k, v in pend.items() if k not in used and v["last_h"] < h - 1e-6]:
                pend.pop(tid, None)

            # Promote any track that reached the required consecutive streak.
            promote = [tid for tid, s in pend.items() if s["streak"] >= CONFIRM_CONSEC]
            for tid in promote:
                s = pend.pop(tid)

                # Allocate a new child cluster id in the child range.
                child = next_child_cid
                next_child_cid += 1

                # IMPORTANT: Write corrected coords for downstream compatibility (add back dx, dy).
                rows_out.append({
                    "cluster_id": child,
                    "height_m":   h,
                    "x_corr_m":   s["x"] + dx,
                    "y_corr_m":   s["y"] + dy,
                    "radius_m":   s["r"],
                    "ok":         1,
                    "parent_cid": cid
                })

    # ---------- Save all promoted child rows ----------
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cluster_id", "height_m", "x_corr_m", "y_corr_m", "radius_m", "ok", "parent_cid"])
        w.writeheader()
        w.writerows(rows_out)

    print(f"[fork] wrote {len(rows_out)} fork rows → {out_csv}")

