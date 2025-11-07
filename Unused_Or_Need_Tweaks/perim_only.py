# perim_only.py
"""
Version of the main_pipeline for just testing on the perimeter for 
validation purposes.
"""
from pathlib import Path
import argparse, shutil

# --- Use the perimeter-aware detector ---
import find_stem_circles_perim_aware as detector
from find_stem_circles_perim_aware import Params as DetectParams

# Other pipeline stages (unchanged)
import merge_stems_across_slices as merger
import refine_circles_from_tracks as refiner
import export_stems_from_slices_curved_alt as exporter
import fork_detect_from_refined as forks

# Perimeter cropping util
import crop_perimeter_slices as border_crop


def run_perimeter_for_dataset(
    ds_root: Path,
    perim_frac: float,
    perim_stem_expand: float,
    perim_min_ringness: float,
    perim_grad_deg: float,
    allow_ransac_refine: bool,
    fresh_perim: bool,
    perim_arc_deg: float,
):
    dataset_name = ds_root.name
    print(f"\n================= {dataset_name} (perimeter only) =================")

    # Expected main outputs already present
    slices_dir      = ds_root / "slices_12p5cm"
    merged_dir      = ds_root / "merged_stems"
    refined_csv     = merged_dir / "refined_per_slice_circles.csv"

    if not slices_dir.exists() or not any(slices_dir.glob("slice_*.ply")):
        print(f"[skip] {dataset_name}: missing/empty slices_12p5cm/ — run the main pipeline first.")
        return
    if not refined_csv.exists():
        print(f"[skip] {dataset_name}: missing refined_per_slice_circles.csv — run refine first.")
        return

    # Perimeter outputs
    perim_slices_dir   = ds_root / "slices_border"
    perim_dets_dir     = ds_root / "circle_detections_border"
    perim_merged_dir   = ds_root / "merged_stems_border"
    perim_refine_debug = ds_root / "circle_refine_debug_border"

    if fresh_perim:
        for d in (perim_slices_dir, perim_dets_dir, perim_merged_dir, perim_refine_debug):
            if d.exists():
                shutil.rmtree(d)

    perim_slices_dir.mkdir(parents=True, exist_ok=True)
    perim_dets_dir.mkdir(parents=True, exist_ok=True)
    perim_merged_dir.mkdir(parents=True, exist_ok=True)
    perim_refine_debug.mkdir(parents=True, exist_ok=True)

    # 1) Crop outer band while excluding known stems (expanded radii)
    print("[perim] cropping outer band & excluding refined stems …")
    border_crop.run(
        slices_dir=slices_dir,
        out_dir=perim_slices_dir,
        perim_frac=perim_frac,
        stems_csv=refined_csv,
        stem_expand=perim_stem_expand,
        write_debug=True,
    )

    if not any(perim_slices_dir.glob("slice_*.ply")):
        print("[perim] no points left after cropping/exclusion; skipping perimeter pass.")
        return

    # 2) Detect on border with perimeter-aware gates
    print("[perim][detect] starting …")
    dp = DetectParams(
        slices_dir=perim_slices_dir,
        out_dir=perim_dets_dir,
        pixel_size_m=0.02,
    )

    # --- Perimeter-aware thresholds ---
    # Arc completeness: accept shorter arcs at the border (driven by CLI)
    if hasattr(dp, "global_min_arc_deg"):     dp.global_min_arc_deg     = perim_arc_deg
    if hasattr(dp, "relaxed_min_arc_deg"):    dp.relaxed_min_arc_deg    = perim_arc_deg

    # Inlier counts: lighter at perimeter (kept conservative but looser than main)
    # You can tweak these if needed or expose as CLI args later.
    if hasattr(dp, "global_min_inliers_px"):  dp.global_min_inliers_px  = 18
    if hasattr(dp, "relaxed_min_inliers_px"): dp.relaxed_min_inliers_px = 14

    # Ringness & gradient alignment: use CLI for both global and relaxed
    if hasattr(dp, "global_min_ringness"):        dp.global_min_ringness        = perim_min_ringness
    if hasattr(dp, "relaxed_min_ringness"):       dp.relaxed_min_ringness       = perim_min_ringness
    if hasattr(dp, "global_grad_align_deg"):      dp.global_grad_align_deg      = perim_grad_deg
    if hasattr(dp, "relaxed_grad_align_tol_deg"): dp.relaxed_grad_align_tol_deg = perim_grad_deg

    # New perimeter-aware knobs in the detector (keep defaults unless you want to override)
    # Example overrides shown; comment out if you prefer the detector's defaults.
    if hasattr(dp, "perimeter_band_px"):            dp.perimeter_band_px = 6
    if hasattr(dp, "perimeter_min_arc_deg"):        dp.perimeter_min_arc_deg = perim_arc_deg
    if hasattr(dp, "perimeter_min_inliers_px"):     dp.perimeter_min_inliers_px = 14
    if hasattr(dp, "perimeter_center_inside_bias"): dp.perimeter_center_inside_bias = 1.15

    detector.run(dp)
    print("[perim][detect] done.")

    # 3) Merge across slices
    print("[perim][merge ] starting …")
    merger.run(input_dir=perim_dets_dir, out_dir=perim_merged_dir)
    print("[perim][merge ] done.")

    # 4) Refine on border slices
    print("[perim][refine] starting …")
    refiner.run(slices_dir=perim_slices_dir, merged_dir=perim_merged_dir, out_dir=perim_refine_debug)
    print("[perim][refine] done.")

    # 4.5) Fork detection on border set (if refined csv exists)
    perim_refined_csv = perim_merged_dir / "refined_per_slice_circles.csv"
    if perim_refined_csv.exists():
        perim_forks_out = perim_merged_dir / "forks_refined_per_slice_circles.csv"
        print("[perim][fork ] starting …")
        forks.run(slices_dir=perim_slices_dir, merged_dir=perim_merged_dir, out_csv=perim_forks_out)
        print(f"[perim][fork ] wrote → {perim_forks_out}")
    else:
        print("[perim] skip forks (no refined_per_slice_circles.csv)")

    # 5) Export curved stems for border detections
    if perim_refined_csv.exists():
        perim_stems_out_dir = perim_merged_dir / "stems_curved_alt"
        perim_stems_out_dir.mkdir(parents=True, exist_ok=True)
        print("[perim][export] starting …")
        exporter.run(
            slices_dir=perim_slices_dir,
            merged_dir=perim_merged_dir,
            out_dir=perim_stems_out_dir,
            refined_csv=perim_refined_csv,
            csv_dets=perim_merged_dir / "detections_with_clusters.csv",
            csv_tr=perim_merged_dir / "slice_translations.csv",
            write_colored_all=True,
            allow_ransac_refine=allow_ransac_refine,
        )
        print("[perim][export] done.")
    else:
        print("[perim] skip export (no refined_per_slice_circles.csv)")


def main():
    ap = argparse.ArgumentParser(description="Perimeter-only pipeline: crop outer band → detect → merge → refine → forks → export.")
    ap.add_argument("--work-root", type=str, default="Work",
                    help="Root folder containing per-dataset subfolders (each with slices_12p5cm/ and merged_stems/).")
    ap.add_argument("--only", type=str, default="", nargs="*",
                    help="Optional one or more dataset folder names to process (exact match). If omitted, process all under work-root.")
    ap.add_argument("--perim-run", action="store_true",
                    help="Required flag to actually run the perimeter pass (safety latch).")
    ap.add_argument("--perim-frac", type=float, default=0.10,
                    help="Outer band thickness as fraction of footprint radius.")
    ap.add_argument("--perim-stem-expand", type=float, default=1.25,
                    help="Multiply refined radius to carve out stems from the border crop.")
    ap.add_argument("--perim-min-ringness", type=float, default=0.30,
                    help="Lower ringness gate for border detection.")
    ap.add_argument("--perim-grad-deg", type=float, default=38.0,
                    help="Looser gradient alignment (deg) for border detection.")
    ap.add_argument("--allow-ransac-refine", action="store_true",
                    help="Exporter: optional local ring refine.")
    ap.add_argument("--fresh-perim", action="store_true",
                    help="Delete existing perimeter folders before running.")
    ap.add_argument("--perim-arc-deg", type=float, default=90.0,
                    help="Arc completeness (deg) threshold for perimeter detect (lower = accept smaller arcs).")
    args = ap.parse_args()

    if not args.perim_run:
        print("Nothing to do: add --perim-run to confirm you want to run the perimeter pass.")
        return

    work_root = Path(args.work_root).resolve()
    if not work_root.exists():
        raise SystemExit(f"Work root does not exist: {work_root}")

    # Choose datasets
    dataset_dirs = [d for d in sorted(work_root.iterdir()) if d.is_dir()]
    if args.only:
        names = set(args.only)
        dataset_dirs = [d for d in dataset_dirs if d.name in names]

    # Process each selected dataset
    any_done = False
    for ds_root in dataset_dirs:
        run_perimeter_for_dataset(
            ds_root=ds_root,
            perim_frac=args.perim_frac,
            perim_stem_expand=args.perim_stem_expand,
            perim_min_ringness=args.perim_min_ringness,
            perim_grad_deg=args.perim_grad_deg,
            allow_ransac_refine=args.allow_ransac_refine,
            fresh_perim=args.fresh_perim,
            perim_arc_deg=args.perim_arc_deg,
        )
        any_done = True

    if not any_done:
        print("No datasets processed (check --work-root and/or --only filters).")
    else:
        print("\nPerimeter-only pass complete.")


if __name__ == "__main__":
    main()
