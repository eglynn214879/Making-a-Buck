# ======================================================================
#  branch_points_from_trackdrop.py
#  -------------------------------------------------------------
#  Purpose:
#     Infer per-stem branch (crown onset) heights from refined slice tracks
#     by detecting a *persistent loss* of circle fits (ok == 0 / missing).
#     The idea: once a stem’s ring becomes occluded by branches/leaves,
#     refined detections start failing for several consecutive slices.
#
#  How it works (per cluster_id):
#     1) Load refined_per_slice_circles.csv and collect (height_m, ok) pairs.
#     2) Sort by height and ignore slices below --min-h.
#     3) Require a warm-up of --good-streak consecutive good detections.
#     4) When good is established, watch for --bad-streak consecutive bads.
#     5) Declare branch at either:
#          • the first bad slice ("first_bad"), or
#          • the last good slice before the bad run ("last_good").
#
#  Inputs (required):
#     merged_dir/refined_per_slice_circles.csv
#         CSV with at least: cluster_id (or cid), height_m, ok
#         ok: 1 = success, 0 = fail/missing.
#         If ok is missing, it defaults to 1 (conservative).
#
#  Outputs:
#     out/branch_points_trackdrop.csv
#         Columns: cluster_id, branch_height_m, first_bad_height_m,
#                  last_good_height_m, good_streak_used, bad_streak_used, method
#     out/trackdrop_debug.csv
#         Minimal per-cluster diagnostics (counts, found flag, branch height).
#
#  CLI Usage:
#     python branch_points_from_trackdrop.py \
#         --merged-dir Work/<dataset>/merged_stems \
#         --out Work/<dataset>/merged_stems  \
#         --min-h 1.3 --good-streak 3 --bad-streak 3 --branch-at last_good
#
#  Arguments:
#     --merged-dir     Folder containing refined_per_slice_circles.csv  (required)
#     --out            Output folder for CSVs                           (required)
#     --min-h          Only consider slices at/above this height (m)    [default: 1.3]
#     --good-streak    # consecutive ok==1 before branch can be called  [default: 3]
#     --bad-streak     # consecutive bad/missing to declare a branch    [default: 3]
#     --branch-at      "first_bad" | "last_good" selection              [default: first_bad]
#     --skip-noise     If set, skip cluster_id < 0 (noise labels)
#
#  Notes & Assumptions:
#     • This is a lightweight heuristic suitable when refined ok flags are reliable.
#     • If your pipeline sparsely samples heights, you may want smaller streaks.
#     • If ok is noisy, increase --bad-streak and/or --good-streak.
#     • Branch height is reported per cluster_id; multiple branches per stem are
#       not modeled in this simple track-drop detector.
#
#  Dependencies: argparse, csv, pathlib, collections (defaultdict)
#  Author: Ethan Glynn
# ======================================================================

import argparse, csv
from pathlib import Path
from collections import defaultdict

def read_refined_csv(path: Path):
    rows = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                cid = int(row.get("cluster_id") or row.get("cid") or row["cluster_id"])
                h   = float(row["height_m"])
                okv = row.get("ok")
                ok  = int(okv) if okv not in (None, "", "nan") else 1  # default to 1 if missing
                rows.append((cid, h, ok))
            except Exception:
                continue
    return rows

def main():
    ap = argparse.ArgumentParser(
        description="Find branch points by first persistent loss of refined stem track (ok==0/missing)."
    )
    ap.add_argument("--merged-dir", required=True, type=Path,
                    help="Folder containing refined_per_slice_circles.csv")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output folder")
    ap.add_argument("--min-h", type=float, default=1.3,
                    help="Only consider branch after this height (meters)")
    ap.add_argument("--good-streak", type=int, default=3,
                    help="Required consecutive good detections before a branch can be called")
    ap.add_argument("--bad-streak", type=int, default=3,
                    help="Required consecutive bad (missing/ok==0) to declare branch")
    ap.add_argument("--branch-at", choices=["first_bad","last_good"], default="first_bad",
                    help="Report branch height at the first bad slice, or at the last good slice before it.")
    ap.add_argument("--skip-noise", action="store_true",
                    help="Skip cluster_id < 0 (noise/unclustered)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    refined_csv = args.merged_dir / "refined_per_slice_circles.csv"
    if not refined_csv.exists():
        raise SystemExit(f"Missing refined_per_slice_circles.csv in {args.merged_dir}")

    rows = read_refined_csv(refined_csv)
    if not rows:
        raise SystemExit("No rows in refined_per_slice_circles.csv")

    # group by cluster
    by_cid = defaultdict(list)
    for cid, h, ok in rows:
        if args.skip_noise and cid < 0:
            continue
        by_cid[cid].append((h, ok))

    branch_rows = []
    per_cluster_debug = []

    for cid, items in by_cid.items():
        # sort by height
        items.sort(key=lambda t: t[0])
        H  = [h for h,_ in items]
        OK = [ok for _,ok in items]

        # compress to above min-h (keep indices so we can map back)
        idxs = [i for i, h in enumerate(H) if h >= args.min_h]
        if not idxs:
            continue

        good_run = 0
        bad_run  = 0
        last_good_h = None
        branch_h = None
        first_bad_h = None

        for i in idxs:
            h  = H[i]
            ok = int(OK[i])  # 1 = good detection, 0 = missing/failed

            # count runs
            if ok == 1:
                good_run += 1
                bad_run = 0
                last_good_h = h
            else:
                # only start counting bad streak once we have a prior good streak
                if good_run >= args.good_streak:
                    bad_run += 1
                    if first_bad_h is None:
                        first_bad_h = h
                    if bad_run >= args.bad_streak:
                        # declare branch
                        branch_h = first_bad_h if args.branch_at == "first_bad" else (last_good_h or first_bad_h)
                        break
                else:
                    # still warming up (haven't proven this stem had a clean section yet)
                    good_run = 0
                    bad_run = 0
                    first_bad_h = None
                    last_good_h = None

        if branch_h is not None:
            branch_rows.append({
                "cluster_id": cid,
                "branch_height_m": float(branch_h),
                "first_bad_height_m": float(first_bad_h) if first_bad_h is not None else "",
                "last_good_height_m": float(last_good_h) if last_good_h is not None else "",
                "good_streak_used": args.good_streak,
                "bad_streak_used":  args.bad_streak,
                "method": f"track_drop({args.branch_at})"
            })

        # minimal debug record
        per_cluster_debug.append({
            "cluster_id": cid,
            "n_slices_total": len(items),
            "n_slices_above_min_h": len(idxs),
            "found_branch": int(branch_h is not None),
            "branch_height_m": float(branch_h) if branch_h is not None else "",
        })

    # write outputs
    out_csv = args.out / "branch_points_trackdrop.csv"
    with open(out_csv, "w", newline="") as f:
        cols = ["cluster_id","branch_height_m","first_bad_height_m","last_good_height_m",
                "good_streak_used","bad_streak_used","method"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(branch_rows)

    dbg_csv = args.out / "trackdrop_debug.csv"
    with open(dbg_csv, "w", newline="") as f:
        cols = ["cluster_id","n_slices_total","n_slices_above_min_h","found_branch","branch_height_m"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(per_cluster_debug)

    print(f"[ok] wrote branch points → {out_csv} (n={len(branch_rows)})")
    print(f"[ok] wrote debug → {dbg_csv}")

if __name__ == "__main__":
    main()
