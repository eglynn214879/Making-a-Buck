# eval_showcase.py
"""
Creates graphs to shwocase evaluation of ground truth data.
"""
import json, argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _val_or_nan(d, k):
    v = d.get(k, None)
    return np.nan if v is None else float(v)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollup", required=True, help="path to eval_outputs/rollup.json")
    ap.add_argument("--out", required=True, help="output dir for csv/png")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    with open(args.rollup, "r") as f:
        roll = json.load(f)

    # ---- DataFrame from per-dataset summaries ----
    df = pd.DataFrame(roll["datasets"]).copy()

    # Ensure numeric dtypes & handle None
    num_cols = ["precision","recall","mean_loc_err_m","median_loc_err_m",
                "mean_dbh_abs_err_m","median_dbh_abs_err_m",
                "gt_count","det_count","matches","missed_gt","false_positives"]
    for c in num_cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # F1 per dataset (safe divide)
    df["F1"] = 2 * (df["precision"] * df["recall"]) / (df["precision"] + df["recall"])
    df["F1"] = df["F1"].fillna(0.0)

    # Optionally sort by dataset name so plots have stable order
    if "dataset" in df:
        df = df.sort_values("dataset").reset_index(drop=True)

    # Save a pretty rollup CSV
    cols = ["dataset","gt_count","det_count","matches","missed_gt","false_positives",
            "precision","recall","F1","mean_loc_err_m","median_loc_err_m",
            "mean_dbh_abs_err_m","median_dbh_abs_err_m"]
    existing_cols = [c for c in cols if c in df.columns]
    df[existing_cols].to_csv(out / "rollup_pretty.csv", index=False)

    # ---- Macro metrics from rollup (already provided) ----
    macro_p = float(roll.get("macro_precision", np.nan))
    macro_r = float(roll.get("macro_recall", np.nan))
    macro_f1 = (2 * macro_p * macro_r / (macro_p + macro_r)) if (macro_p + macro_r) > 0 else np.nan

    # Write a tiny summary.txt for quick eyeballing
    with open(out / "summary.txt", "w") as fsum:
        fsum.write(f"Datasets: {len(df)}\n")
        fsum.write(f"Macro precision: {macro_p:.3f}\n")
        fsum.write(f"Macro recall:    {macro_r:.3f}\n")
        fsum.write(f"Macro F1:        {macro_f1:.3f}\n")
        if "mean_dbh_abs_err_m" in df:
            fsum.write(f"Mean of mean |DBH err| (m):   {df['mean_dbh_abs_err_m'].mean():.4f}\n")
        if "median_dbh_abs_err_m" in df:
            fsum.write(f"Mean of median |DBH err| (m): {df['median_dbh_abs_err_m'].mean():.4f}\n")

    # Reusable x positions / labels
    x = np.arange(len(df))
    labels = df["dataset"] if "dataset" in df else pd.Series([f"d{i}" for i in range(len(df))])

    # ---- 1) Precision/Recall bars with value labels + macro banner ----
    w = 0.35
    plt.figure(figsize=(11,6))
    plt.bar(x - w/2, df["precision"], w, label="Precision")
    plt.bar(x + w/2, df["recall"],    w, label="Recall")
    for xi, p in zip(x, df["precision"]):
        if not np.isnan(p):
            plt.text(xi - w/2, p + 0.015, f"{p:.2f}", ha="center", va="bottom", fontsize=8)
    for xi, r in zip(x, df["recall"]):
        if not np.isnan(r):
            plt.text(xi + w/2, r + 0.015, f"{r:.2f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(f"Per-dataset Precision & Recall  |  Macro P={macro_p:.3f}  R={macro_r:.3f}  F1={macro_f1:.3f}")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(out / "precision_recall.png", dpi=200)
    plt.close()


    # ---- 2) DBH error bars (mean only) ----
    if "mean_dbh_abs_err_m" in df:
        plt.figure(figsize=(10,7))
        plt.bar(x, df["mean_dbh_abs_err_m"], width=0.6, label="Mean |DBH error|")
        for xi, v in zip(x, df["mean_dbh_abs_err_m"]):
            if not np.isnan(v):
                plt.text(xi, v + max(1e-3, 0.01 * np.nanmax(df["mean_dbh_abs_err_m"])),
                         f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        plt.xticks(x, labels, rotation=20, ha="right")
        plt.ylabel("|DBH error| (m)")
        plt.title("Per-dataset |DBH error| (mean)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "dbh_error.png", dpi=200)
        plt.close()


    # ---- 3) Detection outcomes stacked ----
    plt.figure(figsize=(11,6))
    matches = df["matches"] if "matches" in df else np.zeros(len(df))
    fps     = df["false_positives"] if "false_positives" in df else np.zeros(len(df))
    missed  = df["missed_gt"] if "missed_gt" in df else np.zeros(len(df))
    plt.bar(x, matches, label="Matches")
    plt.bar(x, fps, bottom=matches, label="False +")
    plt.bar(x, missed, bottom=matches+fps, label="Missed GT", hatch="//")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Count")
    plt.title("Per-dataset detection outcomes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "detection_outcomes.png", dpi=200)
    plt.close()

    # ---- 4) Location error (mean vs median) ----
    if "mean_loc_err_m" in df and "median_loc_err_m" in df:
        plt.figure(figsize=(8,6))
        plt.scatter(df["mean_loc_err_m"], df["median_loc_err_m"])
        for i, name in enumerate(labels):
            mx = df["mean_loc_err_m"].iloc[i]; md = df["median_loc_err_m"].iloc[i]
            if not (np.isnan(mx) or np.isnan(md)):
                plt.text(mx, md, str(name), fontsize=8, ha="left", va="bottom")
        plt.xlabel("Mean location error (m)")
        plt.ylabel("Median location error (m)")
        plt.title("Location error (mean vs median)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "loc_error_mean_vs_median.png", dpi=200)
        plt.close()
    
    # ---- 5) Mean location error per dataset (bar) ----
    if "mean_loc_err_m" in df:
        vals = pd.to_numeric(df["mean_loc_err_m"], errors="coerce").to_numpy()
        overall = np.nanmean(vals) if np.isfinite(vals).any() else np.nan

        plt.figure(figsize=(10,6))
        plt.bar(x, vals, width=0.6, label="Mean location error (m)")
        # annotate bars
        for xi, v in zip(x, vals):
            if not np.isnan(v):
                plt.text(xi, v + max(1e-3, 0.01 * np.nanmax(vals)), f"{v:.3f}",
                         ha="center", va="bottom", fontsize=8)
        # overall mean line (if defined)
        if not np.isnan(overall):
            plt.axhline(overall, linestyle="--", linewidth=1.2, label=f"Overall mean = {overall:.3f} m")

        plt.xticks(x, labels, rotation=20, ha="right")
        plt.ylabel("Mean location error (m)")
        plt.title("Per-dataset mean location error")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "mean_loc_error.png", dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
