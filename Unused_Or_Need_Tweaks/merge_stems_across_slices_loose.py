# ======================================================================
#  merge_slice_circles_to_stems.py
#  -------------------------------------------------------------
#  Stage: Drift correction across slices + stem clustering
#  -------------------------------------------------------------
#  Purpose:
#     • Load per-slice circle detections (x, y, diameter) from CSVs.
#     • Choose a reference slice (near 1.5 m, high detection count).
#     • Estimate per-slice XY drift vs. reference using fast grid-NN
#       matches (optionally weighted by diameter similarity).
#     • Smooth the drift with a tiny 1-D Kalman filter across height.
#     • Apply drift corrections, aggregate all detections, and cluster
#       into stems using a simple grid-DBSCAN (ε, min_samples).
#     • Write clustered detections, per-stem summaries, and the
#       per-slice drift table for sanity checking.
#
#  Inputs (from config section):
#     INPUT_DIR : folder containing slice_*.csv files with columns:
#                 center_x_m, center_y_m, (optional) diameter_m
#
#  Outputs (written to OUT_DIR):
#     detections_with_clusters.csv  # per-detection rows + cluster_id
#     stems_summary.csv             # one row per stem cluster
#     slice_translations.csv        # per-slice dx, dy, n_matches
#
#  Key knobs:
#     NN_RADIUS_M         # match radius for drift estimation
#     MATCH_ITERS         # ICP-like refinement iterations
#     USE_DIAMETER_WEIGHT # weight matches by |d1-d2| similarity
#     DIAM_TOL_FRAC       # tolerance for diameter weighting
#     CLUSTER_EPS_M       # clustering ε (meters)
#     CLUSTER_MIN_SAMPLES # min detections to form a stem
#
#  Typical usage:
#     python merge_slice_circles_to_stems.py
#       (uses INPUT_DIR and OUT_DIR defaults from the file)
#
#  Notes:
#     • CSV filenames must follow: slice_<height>m.csv
#     • Reference slice is auto-picked via a simple score combining
#       detection count and proximity to 1.5 m.
#     • Clustering is near-linear via a uniform grid index; labels -1
#       denote noise (unclustered).
#
#  Dependencies:
#     numpy
#
#  Author: Ethan Glynn
# ======================================================================

import os, re, csv, math
from pathlib import Path
from collections import defaultdict
import numpy as np

# ---------- Config ----------
INPUT_DIR = Path("circle_detections")   # folder containing slice_*.csv files
OUT_DIR   = Path("merged_stems")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# drift / matching
NN_RADIUS_M = 0.90        # max distance to consider a match when estimating slice drift
MATCH_ITERS = 3           # ICP-like refinement steps

# clustering (after drift correction)
CLUSTER_EPS_M       = 0.35  # points within this are the same stem (≈ 25 cm)
CLUSTER_MIN_SAMPLES = 2     # require detections from at least 2 slices

# robust DBH prior; only used for weighting during drift estimation (optional)
USE_DIAMETER_WEIGHT = True
DIAM_TOL_FRAC = 0.20      # prefer matches with |d1-d2| <= 20%

SLICE_RE = re.compile(r"slice_(?P<h>[0-9]+\.[0-9]+)m\.csv")

# ---------- Utils ----------
def parse_height_from_name(name: str):
    m = SLICE_RE.match(name)
    return float(m.group("h")) if m else None

def read_slice_csv(path: Path):
    rows = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append({
                    "x": float(row["center_x_m"]),
                    "y": float(row["center_y_m"]),
                    "diam": float(row.get("diameter_m", "nan")),
                })
            except KeyError:
                rows.append({
                    "x": float(row["center_x_m"]),
                    "y": float(row["center_y_m"]),
                    "diam": float("nan"),
                })
    if not rows:
        return np.empty((0, 3), dtype=np.float32)
    arr = np.array([[r["x"], r["y"], r["diam"]] for r in rows], dtype=np.float32)
    return arr  # (N, 3) -> x,y,diam

def pick_reference_slice(slices):
    """
    Choose a reference slice near 1.5m with many detections.
    slices: list of (height, arr)
    """
    if not slices:
        return None
    scores = []
    for h, arr in slices:
        n = len(arr)
        score = n - 0.8 * abs((h or 1.5) - 1.5) / 0.25
        scores.append((score, h))
    scores.sort(reverse=True)
    return scores[0][1]

# ---------- Fast grid index for neighbor search ----------
def build_grid(points_xy, cell):
    """hash (i,j) -> indices"""
    if points_xy.size == 0:
        return {}, np.empty((0,2), dtype=np.float32), np.empty(0, dtype=np.int32)
    gx = np.floor(points_xy[:,0] / cell).astype(np.int64)
    gy = np.floor(points_xy[:,1] / cell).astype(np.int64)
    keys = np.stack([gx, gy], axis=1)
    mapping = defaultdict(list)
    for idx, (i,j) in enumerate(keys):
        mapping[(i,j)].append(idx)
    return mapping, keys, np.arange(points_xy.shape[0])

def nn_matches(ref_xy, ref_d, cur_xy, cur_d, radius):
    """
    For each cur point, find nearest ref point within radius using a grid.
    Returns arrays of displacement vectors (ref - cur), weighted by diam similarity if enabled.
    """
    if ref_xy.size == 0 or cur_xy.size == 0:
        return np.empty((0,2), dtype=np.float32)

    cell = radius
    grid, _, _ = build_grid(ref_xy, cell)

    out = []
    for i, (x, y) in enumerate(cur_xy):
        gi = int(math.floor(x / cell))
        gj = int(math.floor(y / cell))
        best_j, best_d2 = -1, 1e18
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                cand = grid.get((gi+di, gj+dj))
                if not cand:
                    continue
                pts = ref_xy[cand]
                dx = pts[:,0] - x
                dy = pts[:,1] - y
                d2 = dx*dx + dy*dy
                m = np.argmin(d2)
                if d2[m] < best_d2:
                    best_d2 = d2[m]
                    best_j = cand[m]
        if best_j >= 0 and best_d2 <= radius*radius:
            w = 1.0
            if USE_DIAMETER_WEIGHT and not (np.isnan(ref_d[best_j]) or np.isnan(cur_d[i])):
                e = abs(ref_d[best_j] - cur_d[i]) / max(ref_d[best_j], cur_d[i], 1e-6)
                w = 0.0 if e > DIAM_TOL_FRAC else (1.0 - e/DIAM_TOL_FRAC)
            out.append([ (ref_xy[best_j,0] - x)*w, (ref_xy[best_j,1] - y)*w ])
    if not out:
        return np.empty((0,2), dtype=np.float32)
    return np.array(out, dtype=np.float32)

def robust_translation(ref_pts, cur_pts, radius=0.60, iters=2):
    """
    Estimate translation to move 'cur' into 'ref' (translation only).
    Returns (dx, dy, used_matches).
    """
    if ref_pts.shape[0] == 0 or cur_pts.shape[0] == 0:
        return 0.0, 0.0, 0
    cur_xy = cur_pts[:, :2].copy()
    cur_d  = cur_pts[:, 2].copy()
    ref_xy = ref_pts[:, :2]
    ref_d  = ref_pts[:, 2]

    dx, dy = 0.0, 0.0
    disps = np.empty((0,2), dtype=np.float32)
    for _ in range(max(1, iters)):
        disps = nn_matches(ref_xy, ref_d, cur_xy, cur_d, radius)
        if disps.shape[0] == 0:
            break
        md = np.median(disps, axis=0)
        dx += md[0]; dy += md[1]
        cur_xy[:,0] += md[0]
        cur_xy[:,1] += md[1]
    return float(dx), float(dy), int(disps.shape[0])

# ---------- Tiny 1D Kalman for smoothing per-slice drift ----------
def kalman_smooth_1d(meas, q=1e-4, r_default=1e-2):
    """
    meas: list of (z, r) where r is variance estimate (can be None)
    """
    x, P = meas[0][0], 1.0
    out = []
    for z, r in meas:
        R = r if (r is not None and r > 0) else r_default
        x_pred, P_pred = x, P + q
        K = P_pred / (P_pred + R)
        x = x_pred + K * (z - x_pred)
        P = (1 - K) * P_pred
        out.append(x)
    return np.array(out, dtype=np.float32)

# ---------- Simple grid-DBSCAN (pure NumPy; near-linear) ----------
def cluster_points(points_xy, eps, min_samples):
    """
    Returns labels array of size N with -1 for noise.
    """
    if points_xy.shape[0] == 0:
        return np.empty(0, dtype=np.int32)
    cell = eps
    grid, keys, order = build_grid(points_xy, cell)

    N = points_xy.shape[0]
    labels = -np.ones(N, dtype=np.int32)
    visited = np.zeros(N, dtype=bool)
    cid = 0

    def neighbors(idx):
        x, y = points_xy[idx]
        gi = int(math.floor(x / cell)); gj = int(math.floor(y / cell))
        ns = []
        for di in (-1,0,1):
            for dj in (-1,0,1):
                cand = grid.get((gi+di, gj+dj), [])
                if not cand:
                    continue
                pts = points_xy[cand]
                dx = pts[:,0] - x
                dy = pts[:,1] - y
                d2 = dx*dx + dy*dy
                for k, d2v in zip(cand, d2):
                    if d2v <= eps*eps:
                        ns.append(k)
        return ns

    for i in range(N):
        if visited[i]: continue
        visited[i] = True
        Ns = neighbors(i)
        if len(Ns) < min_samples:
            labels[i] = -1
            continue
        labels[i] = cid
        seeds = [n for n in Ns if n != i]
        while seeds:
            j = seeds.pop()
            if not visited[j]:
                visited[j] = True
                Ns2 = neighbors(j)
                if len(Ns2) >= min_samples:
                    seeds.extend([n for n in Ns2 if labels[n] < 0])
            if labels[j] < 0:
                labels[j] = cid
        cid += 1
    return labels

# ---------- run() pipeline ----------
def run(input_dir: Path = INPUT_DIR, out_dir: Path = OUT_DIR):
    # 1) Load all slice CSVs
    files = [p for p in input_dir.glob("slice_*.csv")]
    if not files:
        print(f"No slice_*.csv in {input_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    slices = []
    height_to_file = {}
    for p in files:
        h = parse_height_from_name(p.name)
        arr = read_slice_csv(p)
        if h is None or arr.shape[0] == 0:
            continue
        slices.append((h, arr, p))
        height_to_file[h] = p
    slices.sort(key=lambda t: t[0])
    heights = [h for h,_,_ in slices]

    # 2) Pick reference slice
    ref_h = pick_reference_slice([(h, a) for (h,a,_) in slices]) or slices[0][0]
    ref_pts = next(a for (h,a,_) in slices if h == ref_h)
    print(f"Reference slice: {ref_h:.2f} m, detections={len(ref_pts)}")

    # 3) Estimate per-slice translation vs reference (robust, iterative)
    est_dx, est_dy, used = {}, {}, {}
    for h, arr, _ in slices:
        dx, dy, n = (0.0, 0.0, 0)
        if h != ref_h:
            dx, dy, n = robust_translation(ref_pts, arr, radius=NN_RADIUS_M, iters=MATCH_ITERS)
        est_dx[h], est_dy[h], used[h] = dx, dy, n

    # 4) Smooth translations along height (tiny Kalman)
    order = np.argsort(heights)
    hs_sorted = [heights[i] for i in order]
    dx_series = [(est_dx[h], None) for h in hs_sorted]
    dy_series = [(est_dy[h], None) for h in hs_sorted]
    dx_smooth = kalman_smooth_1d(dx_series)
    dy_smooth = kalman_smooth_1d(dy_series)

    dx_map = {h: float(dx_smooth[i]) for i,h in enumerate(hs_sorted)}
    dy_map = {h: float(dy_smooth[i]) for i,h in enumerate(hs_sorted)}

    # 5) Apply smoothed translations & accumulate all detections
    all_rows = []
    for h, arr, src in slices:
        T = np.array([dx_map[h], dy_map[h]], dtype=np.float32)
        pts_corr = arr.copy()
        pts_corr[:,0:2] += T
        for x,y,d in pts_corr:
            all_rows.append({"height_m": h, "x": float(x), "y": float(y), "diameter_m": float(d), "source": src.name})

    # 6) Cluster all corrected detections into stems
    all_xy = np.array([[r["x"], r["y"]] for r in all_rows], dtype=np.float32)
    labels = cluster_points(all_xy, eps=CLUSTER_EPS_M, min_samples=CLUSTER_MIN_SAMPLES)

    # 7) Summaries
    with open(out_dir / "detections_with_clusters.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cluster_id","height_m","x","y","diameter_m","source"])
        w.writeheader()
        for r, lab in zip(all_rows, labels):
            w.writerow({"cluster_id": int(lab), **r})

    clusters = defaultdict(list)
    for r, lab in zip(all_rows, labels):
        if lab >= 0:
            clusters[int(lab)].append(r)

    with open(out_dir / "stems_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster_id","n_detections","n_slices","mean_x_m","mean_y_m","median_DBH_m","dbh_std_m"])
        for cid, pts in sorted(clusters.items()):
            xs = np.array([p["x"] for p in pts], dtype=np.float32)
            ys = np.array([p["y"] for p in pts], dtype=np.float32)
            ds = np.array([p["diameter_m"] for p in pts if not math.isnan(p["diameter_m"])], dtype=np.float32)
            hs = np.array([p["height_m"] for p in pts], dtype=np.float32)
            mean_x, mean_y = float(xs.mean()), float(ys.mean())
            med_dbh = float(np.median(ds)) if ds.size else float("nan")
            std_dbh = float(np.std(ds)) if ds.size else float("nan")
            w.writerow([cid, len(pts), len(np.unique(hs)), mean_x, mean_y, med_dbh, std_dbh])

    # 8) Save per-slice drift (for sanity)
    with open(out_dir / "slice_translations.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["height_m","dx_m","dy_m","n_matches"])
        for h in hs_sorted:
            w.writerow([h, dx_map[h], dy_map[h], used.get(h, 0)])

    print(f"Done. Stems: {len(clusters)}")
    print(f"Wrote:\n  - {out_dir/'stems_summary.csv'}\n  - {out_dir/'detections_with_clusters.csv'}\n  - {out_dir/'slice_translations.csv'}")

if __name__ == "__main__":
    run()  # uses INPUT_DIR / OUT_DIR defaults