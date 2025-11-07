# make_showcase_plots.py
# Create per-stem taper & sweep plots for every dataset under a Work/ root,
# plus per-dataset and cross-dataset summary tables.

import argparse, math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# --------------------------
# Config (defaults)
# --------------------------
BH_MIN = 1.3     # breast-height lower bound (m)
BH_MAX = 1.7     # breast-height upper bound (m)
IQR_K  = 1.5     # IQR multiplier for per-stem diameter outlier filtering
MIN_OBS_PER_STEM = 6  # skip super sparse tracks for plotting
SMOOTH_WIN = 5        # odd integer; moving-median window for diameter smoothing (set to 0 to disable)

# paths relative to dataset root
MERGED_DIRNAME = "merged_stems"
REFINED_NAME   = "refined_per_slice_circles.csv"

# --------------------------
# Helpers
# --------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def iqr_filter(series: pd.Series, k: float = IQR_K) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return (series >= lo) & (series <= hi)

def moving_median(y: np.ndarray, win: int) -> np.ndarray:
    if win is None or win <= 1 or win % 2 == 0 or y.size == 0:
        return y
    pad = win // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    out = np.empty_like(y, dtype=float)
    for i in range(len(y)):
        out[i] = np.median(ypad[i:i+win])
    return out

def simple_taper_slope(h: np.ndarray, d: np.ndarray,
                       h_lo: float = BH_MIN, h_hi: float = BH_MAX) -> float:
    """Least-squares slope of diameter vs height inside [h_lo, h_hi] (m per m)."""
    m = (h >= h_lo) & (h <= h_hi)
    if m.sum() >= 2:
        hh = h[m]; dd = d[m]
    else:
        # fall back to entire range
        hh = h; dd = d
    if hh.size < 2:
        return float("nan")
    A = np.vstack([hh, np.ones_like(hh)]).T
    slope, _ = np.linalg.lstsq(A, dd, rcond=None)[0]
    return float(slope)

def lateral_drift(h: np.ndarray, x: np.ndarray, y: np.ndarray,
                  h_lo: float = BH_MIN, h_hi: float = BH_MAX) -> float:
    """Euclidean displacement between medians at h_lo..h_hi vs full-range medians (robust-ish)."""
    m = (h >= h_lo) & (h <= h_hi)
    if m.sum() < 2 or h.size < 2:
        return float("nan")
    x1 = np.median(x[m]); y1 = np.median(y[m])
    x0 = np.median(x);    y0 = np.median(y)
    return float(math.hypot(x1 - x0, y1 - y0))

def compute_dbh(h: np.ndarray, d: np.ndarray) -> float:
    """Median diameter within BH window; falls back to global median."""
    m = (h >= BH_MIN) & (h <= BH_MAX)
    vals = d[m] if m.any() else d
    if vals.size == 0: return float("nan")
    return float(np.nanmedian(vals))

@dataclass
class StemStats:
    dataset: str
    stem_id: int
    n_obs: int
    dbh_m: float
    taper_slope_m_per_m: float
    lateral_drift_m: float

# --------------------------
# Plotting
# --------------------------
def plot_taper_for_stem(ax, h: np.ndarray, d: np.ndarray,
                        d_smooth: Optional[np.ndarray] = None,
                        stem_id: int = 0, dataset_name: str = ""):
    ax.plot(h, d, ".", alpha=0.6, label="diameter (obs)")
    if d_smooth is not None:
        ax.plot(h, d_smooth, "-", lw=2, label="smoothed")
    ax.axvspan(BH_MIN, BH_MAX, color="grey", alpha=0.12, label="BH window")
    ax.set_xlabel("Height (m)")
    ax.set_ylabel("Diameter (m)")
    ax.set_title(f"{dataset_name} — Stem {stem_id:03d} | Taper")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

def plot_sweep_for_stem(fig, main_ax, ax_xh, ax_yh,
                        h: np.ndarray, x: np.ndarray, y: np.ndarray,
                        stem_id: int = 0, dataset_name: str = ""):
    # XY colored by height
    norm = Normalize(vmin=np.nanmin(h), vmax=np.nanmax(h))
    sm = ScalarMappable(norm=norm, cmap="viridis")
    colors = sm.to_rgba(h)
    main_ax.scatter(x, y, c=colors, s=12)
    main_ax.set_aspect("equal", adjustable="datalim")
    main_ax.set_title(f"{dataset_name} — Stem {stem_id:03d} | Sweep (XY, colored by height)")
    main_ax.set_xlabel("X (m)"); main_ax.set_ylabel("Y (m)")
    main_ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sm, ax=main_ax, fraction=0.046, pad=0.04)
    cbar.set_label("Height (m)")

    # x(h) and y(h)
    ax_xh.plot(h, x, ".", alpha=0.6)
    ax_xh.set_ylabel("X (m)"); ax_xh.grid(True, alpha=0.3)
    ax_yh.plot(h, y, ".", alpha=0.6, color="tab:orange")
    ax_yh.set_xlabel("Height (m)"); ax_yh.set_ylabel("Y (m)"); ax_yh.grid(True, alpha=0.3)

# --------------------------
# Core
# --------------------------
def process_dataset(ds_root: Path) -> List[StemStats]:
    merged_dir = ds_root / MERGED_DIRNAME
    refined_csv = merged_dir / REFINED_NAME
    if not refined_csv.exists():
        print(f"[skip] No refined CSV at {refined_csv}")
        return []

    df = pd.read_csv(refined_csv)
    # expected columns (from your refiner): cluster_id, height_m, x_corr_m, y_corr_m, radius_m, ok, ...
    cols_needed = {"cluster_id","height_m","x_corr_m","y_corr_m","radius_m"}
    if not cols_needed.issubset(df.columns):
        print(f"[warn] Missing columns in {refined_csv.name}; found: {list(df.columns)}")
        return []

    # compute diameter
    df["diameter_m"] = 2.0 * df["radius_m"].astype(float)

    # outlier filter per stem (based on diameter)
    stats: List[StemStats] = []
    out_dir = ds_root / "showcase"
    figs_taper = out_dir / "stems_taper"
    figs_sweep = out_dir / "stems_sweep"
    ensure_dir(figs_taper); ensure_dir(figs_sweep)

    dataset_name = ds_root.name
    per_stem_summary = []

    for stem_id, g in df.groupby("cluster_id"):
        g = g.copy().sort_values("height_m")
        if g.shape[0] < MIN_OBS_PER_STEM:
            continue

        # filter on diameter outliers (IQR) – keep heights aligned
        mask = iqr_filter(g["diameter_m"])
        g = g[mask].copy()

        if g.shape[0] < MIN_OBS_PER_STEM:
            continue

        h = g["height_m"].to_numpy(np.float32)
        x = g["x_corr_m"].to_numpy(np.float32)
        y = g["y_corr_m"].to_numpy(np.float32)
        d = g["diameter_m"].to_numpy(np.float32)

        # optional smoothing for d(h)
        d_smooth = moving_median(d, SMOOTH_WIN) if SMOOTH_WIN and SMOOTH_WIN > 1 else None

        # --- TAPER PLOT ---
        fig_t, ax_t = plt.subplots(figsize=(6, 4))
        plot_taper_for_stem(ax_t, h, d, d_smooth, stem_id=int(stem_id), dataset_name=dataset_name)
        fig_t.tight_layout()
        fig_t.savefig(figs_taper / f"taper_stem_{int(stem_id):03d}.png", dpi=200)
        plt.close(fig_t)

        # --- SWEEP PLOT ---
        fig_s = plt.figure(figsize=(7.5, 5))
        gs = fig_s.add_gridspec(2, 2, height_ratios=[2.2, 1.2])
        ax_xy = fig_s.add_subplot(gs[0, :])
        ax_xh = fig_s.add_subplot(gs[1, 0])
        ax_yh = fig_s.add_subplot(gs[1, 1])
        plot_sweep_for_stem(fig_s, ax_xy, ax_xh, ax_yh, h, x, y, stem_id=int(stem_id), dataset_name=dataset_name)
        fig_s.tight_layout()
        fig_s.savefig(figs_sweep / f"sweep_stem_{int(stem_id):03d}.png", dpi=200)
        plt.close(fig_s)

        # --- per-stem stats ---
        dbh = compute_dbh(h, d)
        slope = simple_taper_slope(h, d)
        drift = lateral_drift(h, x, y)
        stats.append(StemStats(dataset=dataset_name,
                               stem_id=int(stem_id),
                               n_obs=int(g.shape[0]),
                               dbh_m=dbh,
                               taper_slope_m_per_m=slope,
                               lateral_drift_m=drift))
        per_stem_summary.append({
            "dataset": dataset_name,
            "stem_id": int(stem_id),
            "n_obs": int(g.shape[0]),
            "median_DBH_m": dbh,
            "taper_slope_m_per_m": slope,
            "lateral_drift_m": drift
        })

    # save per-dataset stem table
    if per_stem_summary:
        stem_df = pd.DataFrame(per_stem_summary).sort_values(["dataset","stem_id"])
        stem_df.to_csv(out_dir / "stem_summary_table.csv", index=False)

        # dataset-level stats file (small)
        agg = {
            "n_stems": stem_df.shape[0],
            "median_DBH_m": float(stem_df["median_DBH_m"].median()),
            "mean_taper_slope_m_per_m": float(stem_df["taper_slope_m_per_m"].mean()),
            "mean_lateral_drift_m": float(stem_df["lateral_drift_m"].mean())
        }
        pd.Series(agg).to_csv(out_dir / "dataset_stats.csv")

    print(f"[ok] {dataset_name}: stems plotted={len(stats)} → {out_dir}")
    return stats


def main():
    ap = argparse.ArgumentParser(description="Produce per-stem taper & sweep plots and summary tables.")
    ap.add_argument("--work-root", type=str, default=".",
                    help="Path to the Work/ folder (containing per-dataset subfolders).")
    ap.add_argument("--min-obs", type=int, default=6,
                    help="Minimum observations per stem to plot (default 6).")
    ap.add_argument("--iqr-k", type=float, default=1.5,
                    help="IQR multiplier for outlier removal on diameter (default 1.5).")
    ap.add_argument("--smooth-win", type=int, default=5,
                    help="Odd window size for moving-median smoothing of diameter (0=off, default 5).")
    args = ap.parse_args()

    # now safely override the globals
    global MIN_OBS_PER_STEM, IQR_K, SMOOTH_WIN
    MIN_OBS_PER_STEM = args.min_obs
    IQR_K = args.iqr_k
    SMOOTH_WIN = args.smooth_win

    work_root = Path(args.work_root).resolve()
    if not work_root.exists():
        raise SystemExit(f"Work root does not exist: {work_root}")

    # find dataset roots = immediate subfolders that contain merged_stems/refined_per_slice_circles.csv
    # find dataset roots either directly under work_root OR anywhere below it
    dataset_roots = []
    for refined in work_root.rglob(f"{MERGED_DIRNAME}/{REFINED_NAME}"):
        dataset_roots.append(refined.parent.parent)  # …/dataset/merged_stems/refined_per_slice_circles.csv -> dataset

    dataset_roots = sorted(set(dataset_roots))

    for ds in sorted(work_root.iterdir()):
        if not ds.is_dir():
            continue
        refined = ds / MERGED_DIRNAME / REFINED_NAME
        if refined.exists():
            dataset_roots.append(ds)

    if not dataset_roots:
        raise SystemExit(f"No datasets found under {work_root} containing {MERGED_DIRNAME}/{REFINED_NAME}")

    all_stats: List[StemStats] = []
    for ds_root in dataset_roots:
        all_stats.extend(process_dataset(ds_root))

    # Cross-dataset summary
    out_global = work_root / "_showcase_summary"
    ensure_dir(out_global)
    if all_stats:
        rows = [{
            "dataset": s.dataset,
            "stem_id": s.stem_id,
            "n_obs": s.n_obs,
            "median_DBH_m": s.dbh_m,
            "taper_slope_m_per_m": s.taper_slope_m_per_m,
            "lateral_drift_m": s.lateral_drift_m
        } for s in all_stats]
        df = pd.DataFrame(rows)
        df.to_csv(out_global / "all_datasets_stem_summary.csv", index=False)

        # also dump a small global rollup
        rollup = df.groupby("dataset").agg(
            n_stems=("stem_id","count"),
            median_DBH_m=("median_DBH_m","median"),
            mean_taper_slope_m_per_m=("taper_slope_m_per_m","mean"),
            mean_lateral_drift_m=("lateral_drift_m","mean")
        ).reset_index()
        rollup.to_csv(out_global / "per_dataset_rollup.csv", index=False)

        # overall summary
        overall = {
            "datasets": int(rollup.shape[0]),
            "total_stems": int(df.shape[0]),
            "global_median_DBH_m": float(df["median_DBH_m"].median()),
            "global_mean_taper_slope_m_per_m": float(df["taper_slope_m_per_m"].mean()),
            "global_mean_lateral_drift_m": float(df["lateral_drift_m"].mean())
        }
        pd.Series(overall).to_csv(out_global / "overall_stats.csv")
        print(f"[done] Global summaries → {out_global}")
    else:
        print("[note] No stems plotted; nothing to summarize.")


if __name__ == "__main__":
    main()
