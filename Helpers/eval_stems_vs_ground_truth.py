# eval_stems_vs_ground_truth.py
# Detection metrics (precision/recall/FP/missed) use stems_summary.csv XY.
# Advanced metrics (location error & DBH) use refined_per_slice_circles.csv at ~1.30 m,
# with graceful fallback to summary when refined info is unavailable.

from pathlib import Path
import json, math, csv, argparse
import numpy as np
from typing import Tuple, Dict, List, Optional

# -------------------- CONFIG --------------------
WORK_ROOT = Path("Work")
GT_DIR    = Path("ground_truth")
MATCH_RADIUS_M = 0.50
DATASET_NAME_PREFIX = "plot_annotations_"

# Breast height selection
BH_TARGET_M = 1.30   # target BH height (m)
# ------------------------------------------------

def read_gt_csv(path: Path) -> np.ndarray:
    """Return Nx3: x_m, y_m, rbh_m. Header/no-header robust."""
    raw = []
    with path.open("r", newline="") as f:
        first = f.read(1024); f.seek(0)
        try:
            has_header = csv.Sniffer().has_header(first)
        except Exception:
            has_header = False
        r = csv.reader(f)
        if has_header:
            _ = next(r, None)
        for row in r:
            if not row or len(row) < 3:
                continue
            try:
                x = float(row[0]); y = float(row[1]); rbh = float(row[2])
                raw.append((x,y,rbh))
            except ValueError:
                continue
    return np.array(raw, dtype=float) if raw else np.zeros((0,3), float)

def _get_xy_from_row(row: Dict[str,str]) -> Optional[Tuple[float,float]]:
    """Robust XY extraction with fallbacks."""
    for kx, ky in (("x_corr_m","y_corr_m"), ("x_m","y_m"), ("x","y")):
        if kx in row and ky in row:
            try:
                return float(row[kx]), float(row[ky])
            except Exception:
                pass
    return None

# ---------- Summary (detection set + fallback info) ----------
def read_valid_clusters_from_summary(summary_csv: Path) -> Dict[int, Dict[str, float]]:
    """
    From stems_summary.csv, collect valid cluster_ids and fallback XY/DBH.
    Returns: {cluster_id: {"x": x, "y": y, "dbh": dbh_fallback_or_nan}}
    """
    out: Dict[int, Dict[str, float]] = {}
    with summary_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            try:
                cid = int(row.get("cluster_id", i))
            except Exception:
                continue
            # XY (summary)
            x = None; y = None
            for kx, ky in (("mean_x_m","mean_y_m"), ("mean_x","mean_y"), ("x","y")):
                if kx in row and ky in row:
                    try:
                        x = float(row[kx]); y = float(row[ky]); break
                    except Exception:
                        pass
            if x is None or y is None:
                continue  # no XY even in summary → skip from detection set

            # DBH fallback (summary)
            dbh_fb = math.nan
            for kd in ("median_DBH_m", "dbh_m", "median_dbh_m"):
                if kd in row:
                    try:
                        dbh_fb = float(row[kd]); break
                    except Exception:
                        pass
            out[cid] = {"x": x, "y": y, "dbh": dbh_fb}
    return out

# ---------- Refined (advanced metrics source) ----------
def read_refined_rows_by_cid(refined_csv: Path) -> Dict[int, List[Dict[str,str]]]:
    """Group all refined rows by cluster_id (no 'ok' filtering)."""
    by_cid: Dict[int, List[Dict[str,str]]] = {}
    with refined_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                cid = int(row["cluster_id"])
                _ = float(row["height_m"])
                _ = float(row["radius_m"])
            except Exception:
                continue
            by_cid.setdefault(cid, []).append(row)
    return by_cid

def pick_refined_bh_row(rows: List[Dict[str,str]], target: float = BH_TARGET_M) -> Optional[Dict[str,str]]:
    """Choose the row closest to BH_TARGET_M by |height - target|."""
    ranked = []
    for rw in rows:
        try:
            ranked.append((abs(float(rw["height_m"]) - target), rw))
        except Exception:
            pass
    ranked.sort(key=lambda t: t[0])
    return ranked[0][1] if ranked else None

def build_detection_arrays(summary_csv: Path, refined_csv: Path):
    """
    Build:
      - det_xy_summary: Mx2 summary XY used for matching
      - det_dbh:        M refined DBH (fallback to summary DBH)
      - det_xy_refined: Mx2 refined XY (None rows become NaNs)
      - idx_map:        cluster_id -> index
    """
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing {summary_csv}")
    if not refined_csv.exists():
        raise FileNotFoundError(f"Missing {refined_csv}")

    valid = read_valid_clusters_from_summary(summary_csv)   # cid -> summary XY/DBH
    refined_by = read_refined_rows_by_cid(refined_csv)      # cid -> refined rows

    det_xy_summary: List[Tuple[float,float]] = []
    det_dbh: List[float] = []
    det_xy_refined: List[Tuple[float,float]] = []
    idx_map: Dict[int,int] = {}

    for cid, fb in valid.items():
        rows   = refined_by.get(cid, [])
        chosen = pick_refined_bh_row(rows) if rows else None

        # Refined XY & DBH (preferred)
        xy_ref = _get_xy_from_row(chosen) if chosen else None
        rad    = None
        if chosen is not None:
            try:
                rad = float(chosen["radius_m"])
            except Exception:
                rad = None

        # Push summary XY for matching
        idx_map[cid] = len(det_xy_summary)
        det_xy_summary.append((fb["x"], fb["y"]))

        # DBH: prefer refined (2*radius); else summary fallback
        if rad is not None and math.isfinite(rad):
            det_dbh.append(2.0 * rad)
        else:
            det_dbh.append(fb["dbh"] if math.isfinite(fb["dbh"]) else math.nan)

        # Store refined XY (or NaNs)
        if xy_ref is not None:
            det_xy_refined.append(xy_ref)
        else:
            det_xy_refined.append((math.nan, math.nan))

    det_xy_summary = np.array(det_xy_summary, dtype=float) if det_xy_summary else np.zeros((0,2), float)
    det_dbh        = np.array(det_dbh, dtype=float) if det_dbh else np.zeros((0,), float)
    det_xy_refined = np.array(det_xy_refined, dtype=float) if det_xy_refined else np.zeros((0,2), float)

    return det_xy_summary, det_dbh, det_xy_refined, idx_map

# ---------- Matching ----------
def match_sets(gt_xy: np.ndarray, det_xy_for_matching: np.ndarray, radius: float):
    """Greedy one-to-one NN matching within radius using provided detection XY."""
    if gt_xy.shape[0] == 0 and det_xy_for_matching.shape[0] == 0:
        return [], [], []
    used_gt = np.zeros(gt_xy.shape[0], dtype=bool)
    used_dt = np.zeros(det_xy_for_matching.shape[0], dtype=bool)
    matches = []
    if gt_xy.size and det_xy_for_matching.size:
        D = np.linalg.norm(gt_xy[:,None,:] - det_xy_for_matching[None,:,:], axis=2)
        order = np.argsort(D, axis=None)
        for idx in order:
            g = idx // D.shape[1]
            d = idx %  D.shape[1]
            if used_gt[g] or used_dt[d]:
                continue
            dist = D[g,d]
            if dist <= radius:
                used_gt[g] = True
                used_dt[d] = True
                matches.append((g, d, float(dist)))  # dist based on summary XY
    missed = [i for i,u in enumerate(used_gt) if not u]
    fp     = [j for j,u in enumerate(used_dt) if not u]
    return matches, missed, fp

# ---------- Evaluation ----------
def eval_one_dataset(gt_csv: Path, work_root: Path, match_radius: float) -> Dict:
    suffix  = gt_csv.stem.replace("stem_rings_", "")
    dataset = DATASET_NAME_PREFIX + suffix
    merged_dir   = work_root / dataset / "merged_stems"
    summary_csv  = merged_dir / "stems_summary.csv"
    refined_csv  = merged_dir / "refined_per_slice_circles.csv"

    if not summary_csv.exists():
        raise FileNotFoundError(f"Detections missing: {summary_csv}")
    if not refined_csv.exists():
        raise FileNotFoundError(f"Detections missing: {refined_csv}")

    # Ground truth
    gt = read_gt_csv(gt_csv)                 # (N,3): x,y,rbh
    gt_xy  = gt[:, :2]
    gt_dbh = 2.0 * gt[:, 2]

    # Detections: summary XY for matching; refined for advanced metrics
    det_xy_summary, det_dbh, det_xy_refined, _ = build_detection_arrays(summary_csv, refined_csv)

    # Matching is done using summary XY
    matches, missed, fp = match_sets(gt_xy, det_xy_summary, radius=match_radius)

    # Metrics
    rows_match = []
    loc_errs   = []
    dbh_errs   = []

    for gidx, didx, dist_match in matches:
        # Refined distance (preferred for metrics)
        dxr = det_xy_refined[didx, 0] - gt_xy[gidx, 0]
        dyr = det_xy_refined[didx, 1] - gt_xy[gidx, 1]
        if math.isfinite(dxr) and math.isfinite(dyr):
            dist_refined = float(math.hypot(dxr, dyr))
        else:
            dist_refined = dist_match  # fallback to summary distance

        # DBH metric: prefer refined; det_dbh already prefers refined with fallback
        dbh_dt = det_dbh[didx]
        dbh_gt = gt_dbh[gidx]
        dbh_abs = (dbh_dt - dbh_gt) if (math.isfinite(dbh_dt) and math.isfinite(dbh_gt)) else math.nan

        rows_match.append({
            "gt_index": gidx,
            "det_index": didx,
            "gt_x_m": gt_xy[gidx,0],
            "gt_y_m": gt_xy[gidx,1],
            "det_x_summary_m": det_xy_summary[didx,0],
            "det_y_summary_m": det_xy_summary[didx,1],
            "det_x_refined_m": det_xy_refined[didx,0],
            "det_y_refined_m": det_xy_refined[didx,1],
            "distance_match_m": dist_match,     # distance used for matching (summary XY)
            "distance_metric_m": dist_refined,  # distance used for reported location error (refined XY if available)
            "dbh_gt_m": dbh_gt,
            "dbh_det_m": dbh_dt,
            "dbh_abs_err_m": dbh_abs if math.isfinite(dbh_abs) else math.nan,
            "dbh_rel_err": (abs(dbh_abs)/max(dbh_gt,1e-6)) if math.isfinite(dbh_abs) else math.nan,
        })

        loc_errs.append(dist_refined)
        if math.isfinite(dbh_abs):
            dbh_errs.append(abs(dbh_abs))

    precision = len(matches) / max(len(matches) + len(fp), 1)
    recall    = len(matches) / max(len(matches) + len(missed), 1)

    summary = {
        "dataset": dataset,
        "gt_count": int(gt.shape[0]),
        "det_count": int(det_xy_summary.shape[0]),
        "matches": int(len(matches)),
        "missed_gt": int(len(missed)),
        "false_positives": int(len(fp)),
        "precision": precision,
        "recall": recall,
        # Location error is the refined distance distribution (fallback to summary if refined missing)
        "mean_loc_err_m": (float(np.mean(loc_errs)) if loc_errs else None),
        "median_loc_err_m": (float(np.median(loc_errs)) if loc_errs else None),
        "mean_dbh_abs_err_m": (float(np.mean(dbh_errs)) if dbh_errs else None),
        "median_dbh_abs_err_m": (float(np.median(dbh_errs)) if dbh_errs else None),
        "match_radius_m": match_radius,
        "paths": {
            "gt_csv": str(gt_csv),
            "refined_csv": str(refined_csv),
            "stems_summary_csv": str(summary_csv),
        }
    }

    out_dir = Path("eval_outputs") / suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / f"matches_{suffix}.csv").open("w", newline="") as f:
        hdr = [
            "gt_index","det_index",
            "gt_x_m","gt_y_m",
            "det_x_summary_m","det_y_summary_m",
            "det_x_refined_m","det_y_refined_m",
            "distance_match_m","distance_metric_m",
            "dbh_gt_m","dbh_det_m","dbh_abs_err_m","dbh_rel_err"
        ]
        w = csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(rows_match)

    with (out_dir / f"missed_gt_{suffix}.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["gt_index","gt_x_m","gt_y_m","dbh_gt_m"])
        for i in missed:
            w.writerow([i, gt_xy[i,0], gt_xy[i,1], gt_dbh[i]])

    with (out_dir / f"false_positives_{suffix}.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["det_index","det_x_summary_m","det_y_summary_m"])
        for j in fp:
            w.writerow([j, det_xy_summary[j,0], det_xy_summary[j,1]])

    with (out_dir / f"summary_{suffix}.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[eval] {dataset} → matches={len(matches)}, missed={len(missed)}, fp={len(fp)}, "
          f"precision={precision:.3f}, recall={recall:.3f}")
    print(f"[eval] outputs → {out_dir}")
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-radius", type=float, default=MATCH_RADIUS_M,
                    help="Greedy NN match radius in meters (default 0.50).")
    args = ap.parse_args()

    gt_files = sorted(GT_DIR.glob("stem_rings_*.csv"))
    if not gt_files:
        raise SystemExit(f"No ground-truth CSVs found in {GT_DIR}")

    all_summ = []
    for gt in gt_files:
        try:
            s = eval_one_dataset(gt, WORK_ROOT, args.match_radius)
            all_summ.append(s)
        except FileNotFoundError as e:
            print(f"[skip] {gt.name}: {e}")

    rollup = {
        "datasets": all_summ,
        "macro_precision": (sum(s["matches"] for s in all_summ) /
                            max(sum(s["matches"] + s["false_positives"] for s in all_summ), 1)),
        "macro_recall": (sum(s["matches"] for s in all_summ) /
                         max(sum(s["matches"] + s["missed_gt"] for s in all_summ), 1)),
        "match_radius_m": args.match_radius,
    }
    Path("eval_outputs").mkdir(exist_ok=True)
    with (Path("eval_outputs") / "rollup.json").open("w") as f:
        json.dump(rollup, f, indent=2)
    print("[eval] rollup saved → eval_outputs/rollup.json")

if __name__ == "__main__":
    main()
