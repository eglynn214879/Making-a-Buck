# compare_branch_points.py
import argparse, pandas as pd, numpy as np, os

def norm(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def main():
    ap = argparse.ArgumentParser(description="Compare branch heights from two methods")
    ap.add_argument("--survey",     required=True, help="branch_points_survey.csv")
    ap.add_argument("--trackdrop",  required=True, help="branch_points_trackdrop.csv")
    ap.add_argument("--out",        default="branch_points_comparison.csv")
    args = ap.parse_args()

    survey = norm(pd.read_csv(args.survey))
    drop   = norm(pd.read_csv(args.trackdrop))

    # expected columns (robust to names)
    id_col = "cluster_id"
    h_aliases = ["branch_height_m","branch_height","first_branch_m","first_branch_height_m"]

    def pick(df, aliases):
        for a in aliases:
            if a in df.columns: return a
        raise SystemExit(f"Couldn't find a branch-height column in: {df.columns.tolist()}")

    hs = pick(survey, h_aliases)
    hd = pick(drop,   h_aliases)

    keep_s = survey[[id_col, hs]].drop_duplicates().rename(columns={hs:"survey_branch_m"})
    # keep useful extra diagnostics if present
    cols_d = [c for c in [id_col, hd, "first_bad_height_m","last_good_height_m","good_streak_used","bad_streak_used"] if c in drop.columns]
    keep_d = drop[cols_d].drop_duplicates().rename(columns={hd:"trackdrop_branch_m"})

    cmp = keep_s.merge(keep_d, on=id_col, how="outer")
    cmp["diff_m"] = cmp["trackdrop_branch_m"] - cmp["survey_branch_m"]
    cmp["abs_diff_m"] = np.abs(cmp["diff_m"])

    overlap = cmp.dropna(subset=["survey_branch_m","trackdrop_branch_m"])
    mae  = overlap["abs_diff_m"].mean() if not overlap.empty else np.nan
    bias = overlap["diff_m"].mean()     if not overlap.empty else np.nan
    r    = overlap[["survey_branch_m","trackdrop_branch_m"]].corr().iloc[0,1] if len(overlap)>=2 else np.nan

    cmp.to_csv(args.out, index=False)

    print("\n[summary]")
    print(f" survey stems:     {keep_s[id_col].nunique()}")
    print(f" trackdrop stems:  {keep_d[id_col].nunique()}")
    print(f" overlap stems:    {overlap[id_col].nunique()}")
    print(f" MAE (m):          {mae:.3f}" if pd.notna(mae) else " MAE: n/a")
    print(f" Bias TD-SV (m):   {bias:.3f}" if pd.notna(bias) else " Bias: n/a")
    print(f" Pearson r:        {r:.3f}" if pd.notna(r) else " r: n/a")
    print(f"\n[write] {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
