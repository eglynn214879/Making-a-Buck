# --------------------------------------------------------------
#  branchpoint_survey.py
#  --------------------------------------------------------------
#  Purpose
#    Survey per-slice clutter around refined stem rings and infer
#    branch (crown onset) heights. Primary detector = z-score rise
#    in a clutter score; fallback = streak of "bad" slices.
#
#  Inputs
#    --slices-dir  : folder with slice_*.ply (ASCII) point clouds
#    --merged-dir  : folder with refined_per_slice_circles.csv
#                    (needs: cluster_id, height_m, x_corr_m, y_corr_m, radius_m, ok)
#
#  Outputs
#    clutter_profiles.csv       : per-cluster, per-slice metrics + z-scores
#    branch_points_survey.csv   : one row per detected branch height
#
#  Usage
#    python branchpoint_survey.py \
#      --slices-dir Work/<dataset>/slices \
#      --merged-dir Work/<dataset>/merged_stems \
#      --out Work/<dataset>/merged_stems \
#      [--z 1.2 --persist 1 --min-h 1.3 --clean-frac 0.4 --smooth 1 \
#       --bad-streak 3 --bad-min-h-delta 0.4]
#
#  Notes
#    • Clutter score = 0.45*z(outer_density) + 0.35*z(edge_noise)
#                      + 0.20*z(resid_mad) + 0.15*z(inner_density)
#    • Baseline = lowest CLEAN_FRACTION of smoothed scores (by height).
#    • Fallback triggers if a run of “bad” slices (no metrics / too few ROI pts)
#      appears after a clean region above min height.
# --------------------------------------------------------------

import argparse, csv, math
from pathlib import Path
from collections import defaultdict
import numpy as np

# ---------- config defaults ----------
BRANCH_MIN_H = 1.3        # ignore ground-y clutter below this
OUTER_IN = 0.05           # m beyond ring for outer clutter start
OUTER_OUT = 0.20          # m beyond ring for outer clutter end
EDGE_BAND = 0.03          # m around ring for edge noise
INNER_FRAC = 0.55         # inside this fraction of r for inner density
ROI_PAD = 0.35            # m beyond (r+OUTER_OUT) for ROI cut
MIN_ROI_POINTS = 150      # trust threshold
CLEAN_FRACTION = 0.4      # % of lowest-clutter slices to form baseline
Z_THRESH = 1.2            # z-score threshold vs baseline
PERSIST = 1               # consecutive slices required
SMOOTH = 1                # moving-average half-window for the score (0=off)

def read_refined_csv(path: Path):
    """
    Read refined_per_slice_circles.csv into tuples of
    (cluster_id, height_m, x_corr_m, y_corr_m, radius_m, ok).

    Notes:
      • Tolerates missing 'ok' column (defaults to 1 = good).
      • Accepts both 'cid' and 'cluster_id' naming.
    """
    rows = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                cid = int(row.get("cluster_id") or row.get("cid") or row["cluster_id"])
                h   = float(row["height_m"])
                x   = float(row.get("x_corr_m") or row["x_corr_m"])
                y   = float(row.get("y_corr_m") or row["y_corr_m"])
                rad = float(row.get("radius_m")  or row["radius_m"])
                ok  = int(row.get("ok", "1"))   # default ok=1 if missing
                rows.append((cid, h, x, y, rad, ok))
            except Exception:
                # Skip malformed or incomplete rows silently
                continue
    return rows


def load_ply_xyz_ascii(filepath: Path) -> np.ndarray:
    """
    Reads ASCII PLY containing x y z ... coordinates.
    Skips header lines until 'end_header' and returns Nx3 array.
    """
    lines = []
    with open(filepath, "r", errors="ignore") as f:
        # Skip header
        for ln in f:
            if ln.strip() == "end_header":
                break

        # Parse numeric lines after header
        for ln in f:
            if not ln.strip():
                continue
            parts = ln.strip().split()

            # Basic numeric check: first three must look like floats
            if len(parts) >= 3 and parts[0].replace('.', '', 1).replace('-', '', 1).isdigit():
                lines.append([float(parts[0]), float(parts[1]), float(parts[2])])

    if not lines:
        return np.zeros((0, 3), np.float32)

    return np.array(lines, dtype=np.float32)


def parse_h_from_name(name: str) -> float | None:
    """
    Extract numeric height (m) from filenames like 'slice_1.50m.ply'.
    Returns None if pattern not found.
    """
    try:
        s = name.split("slice_", 1)[1]
        m = s.split("m.", 1)[0]
        return float(m)
    except Exception:
        return None


def slice_map(slices_dir: Path):
    """
    Build {height_m : Path} mapping for all slice_*.ply files.
    Ignores files without a valid numeric height.
    """
    m = {}
    for p in slices_dir.glob("slice_*.ply"):
        h = parse_h_from_name(p.name)
        if h is not None:
            m[h] = p
    return m


def clutter_metrics(xy: np.ndarray, cx: float, cy: float, r: float):
    """
    Lightweight clutter features around a fitted ring centered at (cx, cy) with radius r.

    Returns a dict with:
      outer_density : point density in an annulus outside the ring  (~branch clutter)
      edge_noise    : points very close to the ring (wiggly/rough ring contour)
      inner_density : points well inside the ring (should be low for clean trunks)
      resid_mad     : MAD of radial residuals near the ring (ring roughness)
      n             : total points considered (xy rows)

    Notes:
      • Band/annulus widths scale with r (proportional gates).
      • Safe against empty inputs and divides by small numbers.
    """
    if xy.shape[0] == 0:
        return dict(outer_density=0.0, edge_noise=0.0, inner_density=0.0, resid_mad=0.0, n=0)

    # Scale bands with radius to keep behavior consistent across stem sizes
    OUTER_IN  = 0.3 * r
    OUTER_OUT = 1.0 * r
    EDGE_BAND = 0.15 * r
    INNER_FRAC = 0.55

    # Distances of all points from center
    dx = xy[:, 0] - cx
    dy = xy[:, 1] - cy
    dist = np.hypot(dx, dy)

    # --- Edge noise: points extremely close to ring circumference
    edge = np.abs(dist - r) <= EDGE_BAND
    edge_noise = float(np.count_nonzero(edge)) / max(1.0, 2 * np.pi * r * EDGE_BAND)

    # --- Outer clutter: annulus just outside the ring
    outer = (dist >= (r + OUTER_IN)) & (dist <= (r + OUTER_OUT))
    outer_area = math.pi * ((r + OUTER_OUT) ** 2 - (r + OUTER_IN) ** 2)
    outer_density = float(np.count_nonzero(outer)) / max(outer_area, 1e-6)

    # --- Inner density: area within a fraction of the ring radius
    inner = dist <= max(0.02, INNER_FRAC * r)
    inner_density = float(np.count_nonzero(inner)) / max(math.pi * (INNER_FRAC * r) ** 2, 1e-6)

    # --- Residual roughness: MAD of distances near the ring
    near = np.abs(dist - r) <= max(EDGE_BAND * 2.0, 0.06)
    resid = np.abs(dist[near] - r) if np.any(near) else np.array([0.0])
    resid_mad = float(1.4826 * np.median(np.abs(resid - np.median(resid)))) if resid.size else 0.0

    return dict(
        outer_density=outer_density,
        edge_noise=edge_noise,
        inner_density=inner_density,
        resid_mad=resid_mad,
        n=int(xy.shape[0]),
    )


def zscore(x, mean, mad):
    """
    Robust z-score using a MAD-like scale.
    Safe for tiny scales via a floor on the denominator.
    """
    return (x - mean) / max(mad, 1e-6)


def moving_mean(y, halfwin=1):
    """
    Simple centered moving average with edge padding.
      halfwin = 0 disables smoothing.
      Window size = 2*halfwin + 1.
    """
    if halfwin <= 0 or len(y) == 0:
        return np.array(y, dtype=np.float32)

    k = 2 * halfwin + 1
    pad = np.pad(y, (halfwin, halfwin), mode="edge")
    c = np.convolve(pad, np.ones(k, dtype=np.float32) / k, mode="valid")
    return c.astype(np.float32)

def main():
    ap = argparse.ArgumentParser(description="Survey branch points via clutter rise around refined stem rings.")
    ap.add_argument("--slices-dir", required=True, type=Path, help="Folder with slice_*.ply")
    ap.add_argument("--merged-dir", required=True, type=Path, help="Folder with refined_per_slice_circles.csv")
    ap.add_argument("--out", required=True, type=Path, help="Output folder")
    ap.add_argument("--z", type=float, default=Z_THRESH, help="Z-score threshold to flag rise")
    ap.add_argument("--persist", type=int, default=PERSIST, help="Consecutive slices required")
    ap.add_argument("--min-h", type=float, default=BRANCH_MIN_H, help="Ignore heights below this")
    ap.add_argument("--clean-frac", type=float, default=CLEAN_FRACTION, help="Fraction of lowest clutter slices to form baseline")
    ap.add_argument("--smooth", type=int, default=SMOOTH, help="Half-window for score moving average (0=off)")
    ap.add_argument("--bad-streak", type=int, default=3,
                help="Consecutive bad slices required to infer branch by drop-off.")
    ap.add_argument("--bad-min-h-delta", type=float, default=0.4,
                    help="Only consider bad-streaks starting above (min_h + delta).")

    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    refined_csv = args.merged_dir / "refined_per_slice_circles.csv"
    if not refined_csv.exists():
        raise SystemExit(f"Missing refined_per_slice_circles.csv in {args.merged_dir}")

    rows = read_refined_csv(refined_csv)
    if not rows:
        raise SystemExit("No refined rows found.")

    slice_files = slice_map(args.slices_dir)
    if not slice_files:
        raise SystemExit(f"No slice_*.ply in {args.slices_dir}")

    by_cid = defaultdict(list)
    for cid, h, x, y, r, ok in rows:
        if h in slice_files and r > 0:
            by_cid[cid].append((h, x, y, r, ok))

    # outputs
    branch_rows = []
    per_cluster_series = []

    for cid, items in by_cid.items():
        print(f"[debug] processing cluster {cid} with {len(items)} slices")

        items.sort(key=lambda t: t[0])
        Hs, Xs, Ys, Rs, OKs = [np.array([t[i] for t in items]) for i in range(5)]

        # ---- compute metrics for each slice ----
        metrics = []
        for h, x, y, r, ok in items:
            if h < args.min_h:
                metrics.append((h, np.nan, np.nan, np.nan, np.nan, 0))
                continue
            p = slice_files[h]
            xyz = load_ply_xyz_ascii(p)
            if xyz.size == 0:
                metrics.append((h, np.nan, np.nan, np.nan, np.nan, 0))
                continue
            # ROI cut (radius scaled off r)
            roi_R = max(0.6, min(2.5, r * 2.25))
            dx = xyz[:,0] - x; dy = xyz[:,1] - y
            keep = (dx*dx + dy*dy) <= roi_R*roi_R
            roi_xy = xyz[keep, :2]
            if roi_xy.shape[0] < MIN_ROI_POINTS:
                metrics.append((h, np.nan, np.nan, np.nan, np.nan, roi_xy.shape[0]))
                continue
            m = clutter_metrics(roi_xy, x, y, r)
            metrics.append((h, m["outer_density"], m["edge_noise"], m["inner_density"], m["resid_mad"], m["n"]))

        # ---- pack series arrays ----
        Hm = np.array([m[0] for m in metrics], dtype=np.float32)
        OD = np.array([m[1] for m in metrics], dtype=np.float32)
        EN = np.array([m[2] for m in metrics], dtype=np.float32)
        IN = np.array([m[3] for m in metrics], dtype=np.float32)
        RU = np.array([m[4] for m in metrics], dtype=np.float32)
        Np = np.array([m[5] for m in metrics], dtype=np.int32)

        # keep valid rows for scoring
        valid = (~np.isnan(OD)) & (~np.isnan(EN)) & (~np.isnan(RU))
        if valid.sum() < max(3, int(len(valid)*0.15)):
            continue

        # ---- robust normalization + score ----
        def robust_norm(v):
            vv = v[valid]
            med = float(np.median(vv))
            mad = float(1.4826*np.median(np.abs(vv - med))) or 1e-6
            z = (v - med) / mad
            return z, med, mad

        zOD, od_med, od_mad = robust_norm(OD)
        zEN, en_med, en_mad = robust_norm(EN)
        zRU, ru_med, ru_mad = robust_norm(RU)
        zIN, in_med, in_mad = robust_norm(IN)   # inner density weight is lower below

        score_raw = 0.45*zOD + 0.35*zEN + 0.20*zRU + 0.15*zIN

        # sort by height + smooth
        order = np.argsort(Hm)
        H_sorted = Hm[order]
        s_sorted = moving_mean(score_raw[order], halfwin=args.smooth)
        valid_sorted = valid[order] & (H_sorted >= args.min_h)

        sv = s_sorted[valid_sorted]
        if sv.size < 4:
            continue
        k = max(2, int(len(sv)*args.clean_frac))
        idx = np.argsort(sv)[:k]
        base_mean = float(np.median(sv[idx]))
        base_mad  = float(np.std(sv[idx])) or 1e-6  # std for stability across mixes

        z = (s_sorted - base_mean) / base_mad

        # ---- primary detection: z-score exceedance ----
        branch_h = None
        note = "none"
        run = 0
        for h, zz in zip(H_sorted, z):
            if h < args.min_h:
                run = 0
                continue
            if zz >= args.z:
                run += 1
                if run >= args.persist:
                    branch_h = float(h)
                    note = "zscore_trigger"
                    break
            else:
                run = 0

        # ---- fallback: streak of bad slices after a clean region ----
        if branch_h is None:
            H_ord  = H_sorted
            valid_ord = valid[order]
            Np_ord = Np[order]
            # bad := failed metrics OR too few points
            bad_ord = (~valid_ord) | (Np_ord < MIN_ROI_POINTS)

            last_good_h = None
            streak = 0
            for h, bad in zip(H_ord, bad_ord):
                if h < (args.min_h + args.bad_min_h_delta):
                    streak = 0
                    if not bad:
                        last_good_h = h
                    continue
                if bad:
                    streak += 1
                    if streak >= args.bad_streak and last_good_h is not None:
                        branch_h = float(h)   # or float(last_good_h) if you prefer last clean
                        note = f"inferred_from_bad_streak(k={streak})"
                        break
                else:
                    streak = 0
                    last_good_h = h

        # ---- write per-slice series (for debugging/plots) ----
        # Use same baseline to compute unsorted z for export
        z_unsorted = (score_raw - base_mean) / base_mad
        for h, od, en, inn, ru, npnts, sr, zz in zip(Hm, OD, EN, IN, RU, Np, score_raw, z_unsorted):
            per_cluster_series.append({
                "cluster_id": cid, "height_m": float(h),
                "outer_density": float(od) if not math.isnan(od) else "",
                "edge_noise": float(en) if not math.isnan(en) else "",
                "inner_density": float(inn) if not math.isnan(inn) else "",
                "resid_mad": float(ru) if not math.isnan(ru) else "",
                "roi_points": int(npnts),
                "score": float(sr) if not math.isnan(sr) else "",
                "zscore": float(zz) if not math.isnan(zz) else ""
            })

        # ---- record branch if found ----
        if branch_h is not None:
            branch_rows.append({
                "cluster_id": cid,
                "branch_height_m": branch_h,
                "method": note,
                "z_threshold": args.z,
                "persist": args.persist,
                "baseline_mean": base_mean,
                "baseline_mad": base_mad
            })

    # write outputs
    series_csv = args.out / "clutter_profiles.csv"
    with open(series_csv, "w", newline="") as f:
        cols = ["cluster_id","height_m","outer_density","edge_noise","inner_density","resid_mad","roi_points","score","zscore"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(per_cluster_series)

    branch_csv = args.out / "branch_points_survey.csv"
    with open(branch_csv, "w", newline="") as f:
        cols = ["cluster_id","branch_height_m","method","z_threshold","persist","baseline_mean","baseline_mad"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(branch_rows)


    print(f"[debug] built profiles for {len(per_cluster_series)} cluster–slice rows total")

    print(f"[ok] wrote profiles → {series_csv}")
    print(f"[ok] wrote branch points → {branch_csv} (n={len(branch_rows)})")

if __name__ == "__main__":
    main()
