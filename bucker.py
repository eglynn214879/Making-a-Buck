# ================================================================
# bucker.py — Tri-method bucking comparison (Greedy / DP / Advanced DP)
# ---------------------------------------------------------------
# Compares three log-bucking strategies on per-stem LiDAR reconstructions:
#   • Greedy — top-down selection by product price hierarchy
#   • DP-base — dynamic programming using nominal prices only
#   • DP-advanced — DP with LiDAR-derived taper/sweep/occupancy modifiers
#
# The script discovers merged_stems/*.csv files, processes each stem to
# compute optimal bucking plans, aggregates results, and generates charts
# comparing value, volume, and product mix across methods.
#
# Typical usage:
#   $ python bucker.py --base-dir /path/to/project --verbose
#
# Outputs:
#   • Per-plot CSVs alongside each input file:
#       *_tri_method_summary.csv
#       *_tri_method_cuts.csv
#   • Combined CSVs under <base-dir>:
#       tri_method_summary_all_plots.csv
#       tri_method_cuts_all_plots.csv
#   • Figures under <base-dir>/figs/
#   • Configuration JSON:
#       tri_method_config.json
#
# Dependencies:
#   Python ≥ 3.8, numpy, pandas, matplotlib
#
# Author: Ethan Glynn [University of Sydney]
# ================================================================

import os, sys, glob, math, json, re, argparse
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fnmatch

# -----------------------
# Args
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Tri-method bucking comparison")

    # paths & selection
    ap.add_argument("--base-dir", type=str, default=".",
                    help="Project root (folder containing Work/). Defaults to current directory.")
    ap.add_argument("--only-original", action="store_true",
                    help="Process only canonical originals: .../merged_stems/refined_per_slice_circles.csv")
    ap.add_argument("--only-trunc", action="store_true",
                    help="Process only truncated files: .../merged_stems/*_TRUNC.csv")
    ap.add_argument("--prefer-trunc", action="store_true",
                    help="If a stem has both original and *_TRUNC, prefer *_TRUNC.")
    ap.add_argument("--include", action="append", default=[],
                    help="Additional glob(s) to include. Can be repeated.")
    ap.add_argument("--exclude", action="append", default=[],
                    help="Glob(s) to exclude (e.g. '*forks_*', '*_border*'). Can be repeated.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List files that would be processed and exit.")

    # modelling
    ap.add_argument("--slice-step-m", type=float, default=0.3,
                    help="Vertical discretization (m). Should match your slice spacing.")
    ap.add_argument("--kerf-m", type=float, default=0.01,
                    help="Kerf per cut in meters.")
    ap.add_argument("--tip-min-diam-m", type=float, default=0.08,
                    help="Stop bucking when diameter falls below this.")

    # hygiene
    ap.add_argument("--min-stems", type=int, default=1,
                    help="Skip writing per-plot outputs if < N stems survive filters (default 1).")
    ap.add_argument("--verbose", action="store_true", help="Print extra info.")
    return ap.parse_args()

# -----------------------
# Config (also written to JSON for easy edits)
# -----------------------
def default_config(args):
    return {
        "slice_step_m": float(args.slice_step_m),
        "tip_min_diam_m": float(args.tip_min_diam_m),
        "kerf_m": float(args.kerf_m),
        "products": [
            {"name": "Sawn_Grade_A", "lengths_m": [6.0, 5.4, 4.8, 4.2, 3.6, 3.0, 2.4], "sed_min_m": 0.18, "price_per_m3": 180},
            {"name": "Sawn_Grade_B", "lengths_m": [4.8, 4.2, 3.6, 3.0, 2.4], "sed_min_m": 0.16, "price_per_m3": 140},
            {"name": "Pulp",          "lengths_m": [3.0, 2.7, 2.4],            "sed_min_m": 0.10, "price_per_m3": 60}
        ],
        "lidar": {
            "sweep_caps": {
                "Sawn_Grade_A": [
                    {"thresh_m_per_m": 0.03, "max_len_m": 4.8},
                    {"thresh_m_per_m": 0.05, "max_len_m": 3.6}
                ],
                "Sawn_Grade_B": [
                    {"thresh_m_per_m": 0.05, "max_len_m": 3.6}
                ],
                "Pulp": []
            },
            "taper_penalty": {
                "Sawn_Grade_A": {"t0": 0.12, "beta": 1.2},
                "Sawn_Grade_B": {"t0": 0.15, "beta": 0.6},
                "Pulp":         {"t0": 1.00, "beta": 0.0}
            },
            "confidence": {
                "a_occ": 0.6,
                "b_inliers_per": 0.4,
                "K_inliers": 120.0,
                "gamma_resid": 6.0,
                "min_conf": 0.6,
                "max_conf": 1.0
            }
        }
    }

# -----------------------
# Model helpers 
# -----------------------
def smalian_volume(d0_m: float, d1_m: float, L_m: float) -> float:
    """Compute log volume using Smalian’s formula."""
    A0 = math.pi * (d0_m / 2) ** 2
    A1 = math.pi * (d1_m / 2) ** 2
    return ((A0 + A1) / 2.0) * L_m


def build_profile(stem_df: pd.DataFrame, dz: float):
    """Interpolate stem diameter profile on a uniform height grid."""
    prof = (
        stem_df.sort_values("height_m")[["height_m", "radius_m"]]
        .dropna()
        .reset_index(drop=True)
    )

    z = prof["height_m"].to_numpy()
    diam = (2.0 * prof["radius_m"]).to_numpy()

    if len(z) < 2:
        return None, None, None

    z_grid = np.arange(z.min(), z.max() + 1e-9, dz)
    diam_grid = np.interp(z_grid, z, diam, left=0.0, right=0.0)

    return z, z_grid, diam_grid


def segment_stats(slice_df: pd.DataFrame, z0: float, z1: float):
    """Compute mean sweep, taper, and confidence stats between z0 and z1."""
    mask = (slice_df["height_m"] >= z0 - 1e-6) & (slice_df["height_m"] <= z1 + 1e-6)
    seg = slice_df[mask]

    if seg.empty:
        return dict(
            mean_sweep=0.0,
            mean_taper=0.0,
            mean_occ=1.0,
            mean_inliers=120.0,
            mean_resid=0.0,
        )

    return dict(
        mean_sweep=float(seg["sweep_speed_m_per_m"].mean()),
        mean_taper=float(seg["taper_dev"].mean()),
        mean_occ=float(seg["occ"].mean()),
        mean_inliers=float(seg["inliers"].mean()),
        mean_resid=float(seg["resid_mad"].mean() if "resid_mad" in seg.columns else 0.0),
    )

def base_segment_value(d0: float, d1: float, L: float, prod: Dict) -> float | None:
    """Compute base (nominal) segment value given product specs."""
    sed = min(d0, d1)
    if sed < prod["sed_min_m"]:
        return None

    vol = smalian_volume(d0, d1, L)
    return vol * prod["price_per_m3"]


def advanced_price_multiplier(prod_name: str, stats: Dict, cfg: Dict) -> float:
    """Apply taper and confidence-based multipliers to segment value."""
    # --- Taper penalty ---
    taper_cfg = cfg["lidar"]["taper_penalty"].get(prod_name, {"t0": 1.0, "beta": 0.0})
    t0 = float(taper_cfg["t0"])
    beta = float(taper_cfg["beta"])
    taper_mult = math.exp(-beta * max(0.0, stats["mean_taper"] - t0))

    # --- Confidence weighting ---
    conf = cfg["lidar"]["confidence"]
    occ = float(stats["mean_occ"])
    inliers = float(stats["mean_inliers"])
    resid = float(stats["mean_resid"])

    conf_raw = (
        conf["a_occ"] * occ
        + conf["b_inliers_per"] * (inliers / max(1.0, conf["K_inliers"]))
    )
    conf_raw = max(conf["min_conf"], min(conf["max_conf"], conf_raw))
    conf_mult = conf_raw * math.exp(-conf["gamma_resid"] * max(0.0, resid))

    return float(taper_mult * conf_mult)


def length_allowed_by_sweep(prod_name: str, L: float, stats: Dict, cfg: Dict) -> bool:
    """Check whether segment length is allowed given sweep severity rules."""
    caps = cfg["lidar"]["sweep_caps"].get(prod_name, [])
    sweep = float(stats["mean_sweep"])

    for rule in caps:
        thresh = float(rule["thresh_m_per_m"])
        max_len = float(rule["max_len_m"])
        if sweep > thresh and L > max_len:
            return False

    return True

def plan_dp(
    z_grid: np.ndarray,
    diam_grid: np.ndarray,
    cfg: Dict,
    slice_df: Optional[pd.DataFrame] = None,
    advanced: bool = False,
) -> List[Dict]:
    """
    Dynamic programming bucking planner.

    Args:
        z_grid: Monotonic heights (m).
        diam_grid: Diameters (m) at each z in z_grid.
        cfg: Config dict (products, kerf_m, slice_step_m, lidar modifiers).
        slice_df: Per-slice metrics (optional, required if advanced=True).
        advanced: If True, applies sweep/taper/occ confidence adjustments.

    Returns:
        List of cut dicts with start/end z, length, product, values, volume.
    """
    dz = float(cfg["slice_step_m"])
    kerf_m = float(cfg["kerf_m"])
    kerf_steps = int(round(kerf_m / dz)) if kerf_m > 0 else 0

    n = len(z_grid)
    dp = np.zeros(n, dtype=float)  # best value from i..end
    choice: List[Optional[Tuple[int, int, str, float, float, float, float, float]]] = [None] * n

    # Backward pass
    for i in range(n - 2, -1, -1):
        best_val = 0.0
        best_choice = None
        d_i = float(diam_grid[i])

        for prod in cfg["products"]:
            pname = prod["name"]
            for L in prod["lengths_m"]:
                j = i + int(round(float(L) / dz))
                if j >= n:
                    continue

                d_j = float(diam_grid[j])
                base = base_segment_value(d_i, d_j, float(L), prod)
                if base is None:
                    continue

                val = base
                if advanced and slice_df is not None:
                    stats = segment_stats(slice_df, float(z_grid[i]), float(z_grid[j]))
                    if not length_allowed_by_sweep(pname, float(L), stats, cfg):
                        continue
                    val = base * advanced_price_multiplier(pname, stats, cfg)

                j_after = j + kerf_steps
                future = dp[j_after] if j_after < n else 0.0
                total = float(val) + float(future)

                if total > best_val:
                    best_val = total
                    best_choice = (
                        j,               # next index after segment (before kerf)
                        j_after,         # index after kerf
                        pname,           # product name
                        float(L),        # length
                        d_i, d_j,        # start/end diameters
                        float(val),      # effective value (after modifiers if any)
                        float(base),     # base value (nominal)
                    )

        dp[i] = best_val
        choice[i] = best_choice

    # Forward reconstruction
    cuts: List[Dict] = []
    i = 0
    while i < n - 1 and choice[i] is not None:
        j, j_after, pname, L, d_start, d_end, val_eff, val_base = choice[i]  # type: ignore[misc]
        cuts.append(
            {
                "product": pname,
                "z_start_m": float(z_grid[i]),
                "z_end_m": float(z_grid[j]),
                "length_m": float(L),
                "diam_start_m": float(d_start),
                "diam_end_m": float(d_end),
                "value_eff_AUD": float(val_eff),
                "value_base_AUD": float(val_base),
                "volume_m3": float(smalian_volume(float(d_start), float(d_end), float(L))),
            }
        )
        i = j_after if j_after is not None else (j + 1)

    return cuts


def plan_greedy(
    z_grid: np.ndarray,
    diam_grid: np.ndarray,
    cfg: Dict,
) -> List[Dict]:
    """
    Greedy bucking planner: pick the highest-price product first, longest length first.
    """
    dz = float(cfg["slice_step_m"])
    kerf_m = float(cfg["kerf_m"])
    kerf_steps = int(round(kerf_m / dz)) if kerf_m > 0 else 0

    prods_sorted = sorted(cfg["products"], key=lambda p: p["price_per_m3"], reverse=True)

    cuts: List[Dict] = []
    i = 0
    n = len(z_grid)

    while i < n - 1:
        d_i = float(diam_grid[i])
        made_cut = False

        for prod in prods_sorted:
            pname = prod["name"]
            # Try longer lengths first
            for L in sorted(prod["lengths_m"], reverse=True):
                j = i + int(round(float(L) / dz))
                if j >= n:
                    continue

                d_j = float(diam_grid[j])
                base = base_segment_value(d_i, d_j, float(L), prod)
                if base is None:
                    continue

                cuts.append(
                    {
                        "product": pname,
                        "z_start_m": float(z_grid[i]),
                        "z_end_m": float(z_grid[j]),
                        "length_m": float(L),
                        "diam_start_m": d_i,
                        "diam_end_m": d_j,
                        "value_eff_AUD": float(base),
                        "value_base_AUD": float(base),
                        "volume_m3": float(smalian_volume(d_i, d_j, float(L))),
                    }
                )

                i = j + kerf_steps
                made_cut = True
                break  # next position
            if made_cut:
                break

        if not made_cut:
            break  # no feasible product/length at this position

    return cuts

def reevaluate_advanced(cuts: List[Dict], slice_df: pd.DataFrame, cfg: Dict) -> float:
    """Recompute total value using LiDAR-derived modifiers on existing cuts."""
    vals = []
    for c in cuts:
        stats = segment_stats(slice_df, c["z_start_m"], c["z_end_m"])
        mult = advanced_price_multiplier(c["product"], stats, cfg)
        vals.append(c["value_base_AUD"] * mult)
    return float(sum(vals))


def process_csv(
    csv_path: str,
    cfg: Dict,
    min_stems: int = 1,
    verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process a single merged_stems CSV and return (summary_df, cuts_df)."""
    if verbose:
        print(f"[process] Reading {csv_path}")

    df = pd.read_csv(csv_path)

    # Required columns
    need = {"cluster_id", "height_m", "radius_m", "ok"}
    if not need.issubset(df.columns):
        if verbose:
            missing = need - set(df.columns)
            print(f"[skip] Missing columns in {csv_path}: {missing}")
        return pd.DataFrame(), pd.DataFrame()

    # Basic hygiene: valid slices only
    df = df[(df["ok"] == 1) & (df["cluster_id"] >= 0)].copy()
    if df["cluster_id"].nunique() < min_stems:
        if verbose:
            print(f"[skip] <{min_stems} valid stems in {csv_path}")
        return pd.DataFrame(), pd.DataFrame()

    dz = float(cfg["slice_step_m"])
    plot_id = os.path.splitext(csv_path)[0].replace("\\", "/")

    summary_rows: List[Dict] = []
    cuts_rows: List[Dict] = []

    # Helpers
    def totals(cuts: List[Dict]) -> Tuple[float, float, float]:
        if not cuts:
            return 0.0, 0.0, 0.0
        base = float(sum(x["value_base_AUD"] for x in cuts))
        eff  = float(sum(x["value_eff_AUD"]  for x in cuts))
        vol  = float(sum(x["volume_m3"]      for x in cuts))
        return base, eff, vol

    def mape(spec: float, true: float) -> float:
        return float(abs(spec - true) / true) if true > 1e-9 else 0.0

    # Per-stem processing
    for stem_id, sdf in df.groupby("cluster_id"):
        prof = sdf.sort_values("height_m")[["height_m", "radius_m"]].dropna()
        if len(prof) < 2:
            continue

        z = prof["height_m"].to_numpy()
        diam = (2.0 * prof["radius_m"]).to_numpy()

        # Uniform grid and diameter interpolation
        z_grid = np.arange(z.min(), z.max() + 1e-9, dz)
        diam_grid = np.interp(z_grid, z, diam, left=0.0, right=0.0)

        # Truncate where diameter falls below tip threshold
        valid = diam_grid >= float(cfg["tip_min_diam_m"])
        if not valid.any():
            continue
        last_valid = int(np.where(valid)[0][-1])
        z_grid = z_grid[: last_valid + 1]
        diam_grid = diam_grid[: last_valid + 1]

        # Plans
        cuts_g  = plan_greedy(z_grid, diam_grid, cfg)
        cuts_dp = plan_dp(z_grid, diam_grid, cfg, advanced=False)
        cuts_ad = plan_dp(z_grid, diam_grid, cfg, slice_df=sdf, advanced=True)

        # Totals (speculated/base vs effective where applicable)
        base_g, eff_g,  vol_g  = totals(cuts_g)
        base_dp, eff_dp, vol_dp = totals(cuts_dp)
        base_ad, eff_ad, vol_ad = totals(cuts_ad)

        # Re-evaluate Greedy/DP-base with advanced modifiers for parity comparisons
        adv_g  = reevaluate_advanced(cuts_g,  sdf, cfg)
        adv_dp = reevaluate_advanced(cuts_dp, sdf, cfg)

        # Summary row
        summary_rows.append(
            {
                "plot_id": plot_id,
                "stem_id": int(stem_id),
                "z_min_m": float(z_grid.min()),
                "z_max_m": float(z_grid.max()),

                "Greedy_base_AUD": base_g,
                "DPbase_base_AUD": base_dp,
                "DPadv_adv_AUD":   eff_ad,

                "Greedy_advReval_AUD": adv_g,
                "DPbase_advReval_AUD": adv_dp,

                "Greedy_bias_AUD": base_g - adv_g,
                "DPbase_bias_AUD": base_dp - adv_dp,

                "Greedy_MAPE_vs_advReval": mape(base_g, adv_g),
                "DPbase_MAPE_vs_advReval": mape(base_dp, adv_dp),

                "Greedy_volume_m3": vol_g,
                "DPbase_volume_m3": vol_dp,
                "DPadv_volume_m3":  vol_ad,

                "Lift_DPadv_vs_GreedyAdv_AUD":  eff_ad - adv_g,
                "Lift_DPadv_vs_DPbaseAdv_AUD":  eff_ad - adv_dp,
            }
        )

        # Cut-level rows
        for k, c in enumerate(cuts_g):
            cuts_rows.append(
                {"plot_id": plot_id, "stem_id": int(stem_id), "method": "Greedy", **c, "cut_index": k}
            )
        for k, c in enumerate(cuts_dp):
            cuts_rows.append(
                {"plot_id": plot_id, "stem_id": int(stem_id), "method": "DP_base", **c, "cut_index": k}
            )
        for k, c in enumerate(cuts_ad):
            cuts_rows.append(
                {"plot_id": plot_id, "stem_id": int(stem_id), "method": "DP_advanced", **c, "cut_index": k}
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(cuts_rows)

# -----------------------
# Charts
# -----------------------
def save_fig(fig, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

def make_charts(summary_combined: pd.DataFrame, figs_dir: str, verbose=False):
    os.makedirs(figs_dir, exist_ok=True)

    # ---- Lift histogram ----
    if "Lift_DPadv_vs_GreedyAdv_AUD" in summary_combined.columns and not summary_combined.empty:
        fig = plt.figure()
        vals = summary_combined["Lift_DPadv_vs_GreedyAdv_AUD"].dropna().values
        if vals.size == 0:
            vals = np.array([0.0])
        plt.hist(vals, bins=30)
        plt.title("Value Lift per Stem: Advanced DP − Greedy (Advanced Re-eval)")
        plt.xlabel("Lift (AUD per stem)")
        plt.ylabel("Count of stems")
        save_fig(fig, os.path.join(figs_dir, "hist_lift_adv_vs_greedy.png"))
        if verbose: print("[charts] hist_lift_adv_vs_greedy.png")

    # ---- Greedy parity ----
    need = {"Greedy_base_AUD","Greedy_advReval_AUD"}
    if need.issubset(summary_combined.columns) and not summary_combined.empty:
        fig = plt.figure()
        x = summary_combined["Greedy_advReval_AUD"].fillna(0.0).values
        y = summary_combined["Greedy_base_AUD"].fillna(0.0).values
        plt.scatter(x, y, s=12)
        lim = max(1.0, float(np.nanmax([x.max() if x.size else 0, y.max() if y.size else 0])))
        plt.plot([0, lim], [0, lim], color="gray", linestyle="--")
        plt.title("Greedy: Speculated vs LiDAR Re-evaluated Stem Values")
        plt.xlabel("Re-evaluated Value (AUD per stem)")
        plt.ylabel("Speculated Value (AUD per stem)")
        save_fig(fig, os.path.join(figs_dir, "parity_greedy.png"))
        if verbose: print("[charts] parity_greedy.png")

    # ---- DP-base parity ----
    need = {"DPbase_base_AUD","DPbase_advReval_AUD"}
    if need.issubset(summary_combined.columns) and not summary_combined.empty:
        fig = plt.figure()
        x = summary_combined["DPbase_advReval_AUD"].fillna(0.0).values
        y = summary_combined["DPbase_base_AUD"].fillna(0.0).values
        plt.scatter(x, y, s=12)
        lim = max(1.0, float(np.nanmax([x.max() if x.size else 0, y.max() if y.size else 0])))
        plt.plot([0, lim], [0, lim], color="gray", linestyle="--")
        plt.title("DP-base: Speculated vs LiDAR Re-evaluated Stem Values")
        plt.xlabel("Re-evaluated Value (AUD per stem)")
        plt.ylabel("Speculated Value (AUD per stem)")
        save_fig(fig, os.path.join(figs_dir, "parity_dpbase.png"))
        if verbose: print("[charts] parity_dpbase.png")

    # ---- Mean as-planned value per method ----
    need = {"Greedy_base_AUD","DPbase_base_AUD","DPadv_adv_AUD"}
    if need.issubset(summary_combined.columns) and not summary_combined.empty:
        fig = plt.figure()
        means = [
            float(summary_combined["Greedy_base_AUD"].mean()),
            float(summary_combined["DPbase_base_AUD"].mean()),
            float(summary_combined["DPadv_adv_AUD"].mean())
        ]
        labels = ["Greedy (spec.)", "DP-base (spec.)", "DP-advanced"]
        plt.bar(labels, means)
        plt.title("Mean As-Planned Stem Value per Method")
        plt.ylabel("Mean Value (AUD per stem)")
        save_fig(fig, os.path.join(figs_dir, "bar_means_per_method.png"))
        if verbose: print("[charts] bar_means_per_method.png")

    # ---- Split Speculated vs Re-evaluated ----
    need = {"Greedy_base_AUD","Greedy_advReval_AUD",
            "DPbase_base_AUD","DPbase_advReval_AUD","DPadv_adv_AUD"}
    if need.issubset(summary_combined.columns) and not summary_combined.empty:
        labels = ["Greedy", "DP-base", "DP-advanced"]

        # Speculated-only means
        spec_means = [
            float(summary_combined["Greedy_base_AUD"].mean()),
            float(summary_combined["DPbase_base_AUD"].mean()),
            float(summary_combined["DPadv_adv_AUD"].mean())
        ]
        fig = plt.figure(figsize=(5.5, 3.8))
        plt.bar(labels, spec_means, color="#1f77b4")
        plt.ylabel("Mean Value (AUD per stem)")
        plt.title("Speculated Mean Stem Values by Method")
        save_fig(fig, os.path.join(figs_dir, "bar_spec_only.png"))
        if verbose: print("[charts] bar_spec_only.png")

        # Re-evaluated-only means
        reeval_means = [
            float(summary_combined["Greedy_advReval_AUD"].mean()),
            float(summary_combined["DPbase_advReval_AUD"].mean()),
            float(summary_combined["DPadv_adv_AUD"].mean())
        ]
        fig = plt.figure(figsize=(5.5, 3.8))
        plt.bar(labels, reeval_means, color="#ff7f0e")
        plt.ylabel("Mean Value (AUD per stem)")
        plt.title("LiDAR Re-evaluated Mean Stem Values by Method")
        save_fig(fig, os.path.join(figs_dir, "bar_reeval_only.png"))
        if verbose: print("[charts] bar_reeval_only.png")
    
    # --- Mean harvested volume per method ---
    need = {"Greedy_volume_m3","DPbase_volume_m3","DPadv_volume_m3"}
    if need.issubset(summary_combined.columns) and not summary_combined.empty:
        fig = plt.figure()
        means = [
            float(summary_combined["Greedy_volume_m3"].mean()),
            float(summary_combined["DPbase_volume_m3"].mean()),
            float(summary_combined["DPadv_volume_m3"].mean())
        ]
        labels = ["Greedy", "DP-base", "DP-advanced"]
        plt.bar(labels, means)
        plt.title("Mean Harvested Volume per Stem by Method")
        plt.ylabel("Mean Volume (m³ per stem)")
        save_fig(fig, os.path.join(figs_dir, "bar_volume_means_per_method.png"))

def plot_product_mix(cuts_combined: pd.DataFrame, figs_dir: str):
    if cuts_combined.empty: return
    df = cuts_combined.copy()
    # Volume mix
    vol_piv = df.pivot_table(index="method", columns="product", values="volume_m3", aggfunc="sum").fillna(0.0)
    vol_piv_pct = vol_piv.div(vol_piv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    fig = plt.figure()
    bottom = np.zeros(len(vol_piv_pct))
    for prod in vol_piv_pct.columns:
        vals = vol_piv_pct[prod].values
        plt.bar(vol_piv_pct.index, vals, bottom=bottom, label=prod)
        bottom += vals
    plt.title("Product Mix by Method (Volume Share)")
    plt.ylabel("Share of Volume")
    plt.legend(title="Product", fontsize=8)
    save_fig(fig, os.path.join(figs_dir, "stack_product_mix_volume.png"))

    # Value mix (use re-evaluated where available; else value_base_AUD)
    val_col = "value_eff_AUD" if "value_eff_AUD" in df.columns else "value_base_AUD"
    val_piv = df.pivot_table(index="method", columns="product", values=val_col, aggfunc="sum").fillna(0.0)
    val_piv_pct = val_piv.div(val_piv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    fig = plt.figure()
    bottom = np.zeros(len(val_piv_pct))
    for prod in val_piv_pct.columns:
        vals = val_piv_pct[prod].values
        plt.bar(val_piv_pct.index, vals, bottom=bottom, label=prod)
        bottom += vals
    plt.title("Product Mix by Method (Value Share)")
    plt.ylabel("Share of Value (AUD)")
    plt.legend(title="Product", fontsize=8)
    save_fig(fig, os.path.join(figs_dir, "stack_product_mix_value.png"))

def plot_kerf_loss(cuts_combined: pd.DataFrame, kerf_m: float, figs_dir: str):
    if cuts_combined.empty or kerf_m <= 0: return
    df = cuts_combined.copy()
    area_min = np.pi * (np.minimum(df["diam_start_m"], df["diam_end_m"]) / 2.0)**2
    kerf_loss = kerf_m * area_min
    kerf_by_method = kerf_loss.groupby(df["method"]).sum()
    fig = plt.figure()
    kerf_by_method.plot(kind="bar")
    plt.title("Estimated Kerf Loss by Method")
    plt.ylabel("Kerf Volume (m³)")
    plt.xlabel("")
    save_fig(fig, os.path.join(figs_dir, "bar_kerf_loss.png"))



# -----------------------
# Discovery
# -----------------------
def discover_files(base_dir: str, args) -> List[str]:
    base_dir = os.path.abspath(base_dir)
    cands = set()

    if args.only_trunc:
        pats = [os.path.join(base_dir, "Work", "plot_annotations_*", "merged_stems", "*_TRUNC.csv")]
    elif args.only_original:
        pats = [os.path.join(base_dir, "Work", "plot_annotations_*", "merged_stems", "refined_per_slice_circles.csv")]
    else:
        # default: originals + any *_TRUNC.csv, but NOT forks_* and NOT *_border*
        pats = [
            os.path.join(base_dir, "Work", "plot_annotations_*", "merged_stems", "refined_per_slice_circles.csv"),
            os.path.join(base_dir, "Work", "plot_annotations_*", "merged_stems", "*_TRUNC.csv"),
        ]

    # apply patterns
    for pat in pats:
        for p in glob.glob(pat, recursive=True):
            cands.add(os.path.normpath(p))

    # user include/exclude
    for inc in (args.include or []):
        for p in glob.glob(os.path.join(base_dir, inc), recursive=True):
            cands.add(os.path.normpath(p))

    # default excludes unless explicitly included
    default_excludes = ["*forks_*", "*merged_stems_border*"]
    ex_patterns = (args.exclude or []) + ([] if args.only_original or args.only_trunc else default_excludes)
    if ex_patterns:
        keep = []
        for p in cands:
            rel = os.path.relpath(p, base_dir)
            if any(fnmatch.fnmatch(rel, ex) for ex in ex_patterns):
                continue
            keep.append(p)
        cands = set(keep)

    # prefer *_TRUNC when both present (same folder)
    if args.prefer_trunc:
        by_dir = {}
        for p in cands:
            d = os.path.dirname(p)
            by_dir.setdefault(d, []).append(p)
        final = set()
        for d, files in by_dir.items():
            truncs = [f for f in files if f.endswith("_TRUNC.csv")]
            if truncs:
                final.update(truncs)
            else:
                final.update(files)
        cands = final

    return sorted(cands)

# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    BASE_DIR = os.path.abspath(args.base_dir)
    os.makedirs(BASE_DIR, exist_ok=True)

    # config JSON
    cfg = default_config(args)
    config_path = os.path.join(BASE_DIR, "tri_method_config.json")
    with open(config_path, "w") as f: json.dump(cfg, f, indent=2)
    if args.verbose:
        print(f"[setup] base-dir={BASE_DIR}")
        print(f"[setup] config -> {config_path}")

    # choose files
    candidates = discover_files(BASE_DIR, args)
    if args.verbose:
        print(f"[discover] found {len(candidates)} CSV(s)")
        for p in candidates:
            print("  -", os.path.relpath(p, BASE_DIR))

    if args.dry_run:
        return

    if not candidates:
        print("[warn] No candidate CSVs. Adjust --only-*, --include/--exclude, or base-dir.")
        return

    # Process each CSV
    summaries, cuts_all = [], []
    for csv_path in candidates:
        s_df, c_df = process_csv(csv_path, cfg, min_stems=args.min_stems, verbose=args.verbose)
        if s_df.empty and c_df.empty:
            continue
        base = os.path.splitext(csv_path)[0]
        out_sum = base + "_tri_method_summary.csv"
        out_cuts = base + "_tri_method_cuts.csv"
        s_df.to_csv(out_sum, index=False)
        c_df.to_csv(out_cuts, index=False)
        if args.verbose:
            print(f"[write] {os.path.relpath(out_sum, BASE_DIR)}  ({len(s_df)} rows)")
            print(f"[write] {os.path.relpath(out_cuts, BASE_DIR)} ({len(c_df)} rows)")
        summaries.append(s_df); cuts_all.append(c_df)

    # Combined outputs
    summary_combined = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    cuts_combined = pd.concat(cuts_all, ignore_index=True) if cuts_all else pd.DataFrame()
    combined_summary_path = os.path.join(BASE_DIR, "tri_method_summary_all_plots.csv")
    combined_cuts_path = os.path.join(BASE_DIR, "tri_method_cuts_all_plots.csv")
    summary_combined.to_csv(combined_summary_path, index=False)
    cuts_combined.to_csv(combined_cuts_path, index=False)
    if args.verbose:
        print(f"[write] {os.path.relpath(combined_summary_path, BASE_DIR)} ({len(summary_combined)} rows)")
        print(f"[write] {os.path.relpath(combined_cuts_path, BASE_DIR)} ({len(cuts_combined)} rows)")

    # Charts
    figs_dir = os.path.join(BASE_DIR, "figs")
    make_charts(summary_combined, figs_dir, verbose=args.verbose)
    plot_product_mix(cuts_combined, figs_dir)
    plot_kerf_loss(cuts_combined, cfg["kerf_m"], figs_dir)



    print("\n[done]")
    print(" Per-plot CSVs were written next to each source CSV (…/merged_stems/).")
    print(" Combined CSVs:")
    print("  -", combined_summary_path)
    print("  -", combined_cuts_path)
    print(" Figures folder:")
    print("  -", figs_dir)
    print(" Config you can edit and re-run:")
    print("  -", config_path)

if __name__ == "__main__":
    main()
