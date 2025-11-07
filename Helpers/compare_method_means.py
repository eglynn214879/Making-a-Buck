#!/usr/bin/env python3
# export_plot_numerics.py — dump per-dataset numerics used by the plots

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def dataset_from_plot_id(plot_id: str) -> str:
    """
    plot_id in the pipeline looks like:
      .../Work/plot_annotations_<NAME>/merged_stems/refined_per_slice_circles
    want <NAME>.
    """
    p = Path(str(plot_id))
    try:
        return p.parent.parent.name  # .../<plot_annotations_NAME>/merged_stems/<file_stem>
    except Exception:
        return p.stem

def main():
    ap = argparse.ArgumentParser(description="Export per-dataset numerics used by bucker.py plots.")
    ap.add_argument("--summary", default="tri_method_summary_all_plots.csv",
                    help="Path to tri_method_summary_all_plots.csv")
    ap.add_argument("--out-dir", default="figs/numerics",
                    help="Folder to write CSV outputs")
    args = ap.parse_args()

    summary = Path(args.summary)
    if not summary.exists():
        raise SystemExit(f"Missing: {summary}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary)

    # Robust dataset column
    df["dataset"] = df["plot_id"].apply(dataset_from_plot_id)

    # ------- Per-dataset means (the direct plot inputs) -------
    cols_means = [
        "Greedy_base_AUD", "Greedy_advReval_AUD",
        "DPbase_base_AUD", "DPbase_advReval_AUD",
        "DPadv_adv_AUD",   # treated as 'truth' in your comparisons
        "Greedy_volume_m3","DPbase_volume_m3","DPadv_volume_m3"
    ]
    have = [c for c in cols_means if c in df.columns]
    per_ds_means = df.groupby("dataset")[have].mean().reset_index()

    # Also add a global row (ALL)
    global_means = per_ds_means[have].mean(numeric_only=True)
    per_ds_means = pd.concat(
        [per_ds_means, pd.DataFrame([{"dataset":"ALL", **global_means.to_dict()}])],
        ignore_index=True
    )

    # ------- Differences vs DP-advanced (absolute + percent) -------
    # DP-advanced is “closest to real” reference.
    def safe_pct(diff, ref):
        return float(diff / ref) if abs(ref) > 1e-9 else np.nan

    diffs = []
    for _, r in per_ds_means.iterrows():
        row = dict(dataset=r["dataset"])
        adv = float(r.get("DPadv_adv_AUD", np.nan))

        # Greedy vs Advanced
        g_base = float(r.get("Greedy_base_AUD", np.nan))
        row["Diff_Greedy_minus_Advanced_AUD"] = g_base - adv
        row["Diff_Greedy_vs_Advanced_pct"] = safe_pct(row["Diff_Greedy_minus_Advanced_AUD"], adv)

        # DP-base vs Advanced
        d_base = float(r.get("DPbase_base_AUD", np.nan))
        row["Diff_DPbase_minus_Advanced_AUD"] = d_base - adv
        row["Diff_DPbase_vs_Advanced_pct"] = safe_pct(row["Diff_DPbase_minus_Advanced_AUD"], adv)

        # Speculated vs Re-evaluated (method-internal bias)
        g_reev = float(r.get("Greedy_advReval_AUD", np.nan))
        row["Greedy_spec_minus_reeval_AUD"] = g_base - g_reev
        row["Greedy_spec_vs_reeval_pct"] = safe_pct(row["Greedy_spec_minus_reeval_AUD"], g_reev)

        d_reev = float(r.get("DPbase_advReval_AUD", np.nan))
        row["DPbase_spec_minus_reeval_AUD"] = d_base - d_reev
        row["DPbase_spec_vs_reeval_pct"] = safe_pct(row["DPbase_spec_minus_reeval_AUD"], d_reev)

        diffs.append(row)

    per_ds_diffs = pd.DataFrame(diffs)

    # ------- Write CSVs -------
    out_means = out_dir / "per_dataset_means.csv"
    out_diffs = out_dir / "per_dataset_differences_vs_advanced.csv"
    per_ds_means.to_csv(out_means, index=False)
    per_ds_diffs.to_csv(out_diffs, index=False)

    # Console preview
    print("\n=== Per-dataset MEANS (values used by plots) ===")
    print(per_ds_means.to_string(index=False, float_format="%.2f"))
    print(f"\nSaved → {out_means}")

    print("\n=== Per-dataset DIFFERENCES vs DP-advanced (absolute & %) ===")
    show_cols = [
        "dataset",
        "Diff_Greedy_minus_Advanced_AUD","Diff_Greedy_vs_Advanced_pct",
        "Diff_DPbase_minus_Advanced_AUD","Diff_DPbase_vs_Advanced_pct",
        "Greedy_spec_minus_reeval_AUD","Greedy_spec_vs_reeval_pct",
        "DPbase_spec_minus_reeval_AUD","DPbase_spec_vs_reeval_pct",
    ]
    print(per_ds_diffs[show_cols].to_string(index=False, float_format="%.3f"))
    print(f"\nSaved → {out_diffs}")

if __name__ == "__main__":
    main()
