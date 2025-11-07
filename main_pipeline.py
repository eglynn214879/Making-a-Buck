# main_pipeline.py
"""
Complete 3D point-cloud processing pipeline.

This script performs full reconstruction and extraction of stems from a raw LiDAR
point cloud. It:
  1. Estimates and saves a ground mesh.
  2. Creates evenly spaced horizontal slices.
  3. Detects stem cross-sections (circles) per slice.
  4. Merges detections across slices into stem tracks.
  5. Refines detections using track geometry.
  6. Exports a 3D .ply of the reconstructed stems.

After export, the outputs are ready for analysis in bucker.py to generate a
bucking plan.

Notes:
- Commented sections include optional logic for fork detection and perimeter sweep.
- Designed to run on multiple .ply datasets within a “Data_Sets” directory.
"""

from pathlib import Path
import argparse, shutil

# Stages
import ground_mesh
import find_stem_circles as detector
from find_stem_circles import Params as DetectParams
import merge_stems_across_slices as merger
import refine_circles_from_tracks as refiner
import export_stems_from_slices_curved_alt as exporter
# import fork_detect_from_refined as forks
# import crop_perimeter_slices as border_crop


def main():
    ap = argparse.ArgumentParser(
        description="Full pipeline: ground→slices→detect→merge→refine→(perimeter pass)→export"
    )
    ap.add_argument("--data-sets", default="Data_Sets", help="Folder containing raw *.ply datasets")
    ap.add_argument("--work-root", default="Work", help="Root folder for per-dataset outputs")
    ap.add_argument("--grid-size", type=float, default=1.0)

    # DENSE slicing defaults:
    ap.add_argument("--slice-start", type=float, default=0.30, help="Start height above ground (m)")
    ap.add_argument("--slice-step", type=float, default=0.125, help="Slice step (m)")
    ap.add_argument("--min-points", type=int, default=1000)
    ap.add_argument("--allow-ransac-refine", action="store_true", help="Exporter: optional local ring refine")
    ap.add_argument("--use-tqdm", action="store_true", help="Show tqdm progress in ground slicer")
    ap.add_argument("--fresh", action="store_true", help="Delete existing slices dir before slicing")

    # --- NEW: perimeter pass controls ---
    ap.add_argument("--perim-frac", type=float, default=0.10, help="Outer band thickness as fraction of footprint radius")
    ap.add_argument("--perim-stem-expand", type=float, default=1.25,
                    help="Multiply refined radius to carve out stems from the border crop")
    ap.add_argument("--perim-run", action="store_true",
                    help="Enable the perimeter pass (crop outer band, re-detect, merge, refine)")
    # Optional: slightly relax detector for border occlusions
    ap.add_argument("--perim-min-ringness", type=float, default=0.30,
                    help="Lower ringness gate for border detection")
    ap.add_argument("--perim-grad-deg", type=float, default=38.0,
                    help="Looser gradient alignment (deg) for border detection")

    args = ap.parse_args()

    data_dir = Path(args.data_sets).resolve()
    work_root = Path(args.work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    # Synthesized/collected grounds live here (keeps raw folder clean)
    grounds_dir = work_root / "ground_meshes"
    grounds_dir.mkdir(parents=True, exist_ok=True)

    # Process each dataset .ply
    for raw_ply in sorted(data_dir.glob("*.ply")):
        dataset_name = raw_ply.stem
        print(f"\n================= {dataset_name} =================")
        print(f"Input: {raw_ply}")

        # Per-dataset folders
        ds_root          = work_root / dataset_name
        slices_dir       = ds_root / "slices_12p5cm"
        circle_dets_dir  = ds_root / "circle_detections"
        merged_dir       = ds_root / "merged_stems"
        refine_debug_dir = ds_root / "circle_refine_debug"
        stems_out_dir    = merged_dir / "stems_curved_alt"

        # Perimeter pass folders (NEW)
        perim_slices_dir      = ds_root / "slices_border"
        perim_dets_dir        = ds_root / "circle_detections_border"
        perim_merged_dir      = ds_root / "merged_stems_border"
        perim_refine_debug    = ds_root / "circle_refine_debug_border"

        # 0) Ground + slicing (creates ground if missing)
        if args.fresh and slices_dir.exists():
            print("[ground] --fresh: removing existing slices dir …")
            shutil.rmtree(slices_dir)
        slices_dir.mkdir(parents=True, exist_ok=True)

        ground_out_path = grounds_dir / f"ground_{raw_ply.name}"
        gm_cfg = {
            "raw_path": str(raw_ply),
            "ground_out_path": str(ground_out_path),
            "grid_size": float(args.grid_size),
            "slice_start": float(args.slice_start),
            "slice_step": float(args.slice_step),
            "min_points": int(args.min_points),
            "use_tqdm": bool(args.use_tqdm),
        }
        print("[ground] make ground (if missing) + slice …")
        m = ground_mesh.run(dataset_dir=raw_ply.parent, out_dir=slices_dir, config=gm_cfg)
        print(f"[ground] slices → {slices_dir}")
        print(f"[ground] made {len(m.get('outputs', {}).get('slices', []))} slices")

        if not any(slices_dir.glob("slice_*.ply")):
            print("[skip] No slices produced (too sparse or filtering removed all).")
            continue

        # 1) Per-slice circle detection (standard)
        circle_dets_dir.mkdir(parents=True, exist_ok=True)
        print("[detect] starting …")
        detector.run(DetectParams(
            slices_dir=slices_dir,
            out_dir=circle_dets_dir,
            pixel_size_m=0.02,
        ))
        print("[detect] done.")

        # 2) Merge detections across slices
        merged_dir.mkdir(parents=True, exist_ok=True)
        print("[merge ] starting …")
        merger.run(input_dir=circle_dets_dir, out_dir=merged_dir)
        print("[merge ] done.")

        # 3) Refine circles guided by tracks
        refine_debug_dir.mkdir(parents=True, exist_ok=True)
        print("[refine] starting …")
        refiner.run(slices_dir=slices_dir, merged_dir=merged_dir, out_dir=refine_debug_dir)
        print("[refine] done.")

        # Below is the unsed fork detection logic
        """
        # 3.5) forks on refined (your script)
        refined_csv = merged_dir / "refined_per_slice_circles.csv"
        if refined_csv.exists():
            forks_out = merged_dir / "forks_refined_per_slice_circles.csv"
            print("[fork ] starting …")
            forks.run(slices_dir=slices_dir, merged_dir=merged_dir, out_csv=forks_out)
            print(f"[fork ] wrote → {forks_out}")
        else:
            print("[fork ] skipped (missing refined_per_slice_circles.csv)")
        
        """

        # Below is the unused perimeter pass logic

        """
        # -------- NEW: Perimeter pass --------
        if args.perim_run:
            print("[perim] cropping outer band & excluding refined stems …")
            perim_slices_dir.mkdir(parents=True, exist_ok=True)
            border_crop.run(
                slices_dir=slices_dir,
                out_dir=perim_slices_dir,
                perim_frac=args.perim_frac,
                stems_csv=merged_dir / "refined_per_slice_circles.csv",
                stem_expand=args.perim_stem_expand,
                write_debug=True
            )

            if any(perim_slices_dir.glob("slice_*.ply")):
                # Detect again on the border with relaxed thresholds
                print("[perim][detect] starting …")
                perim_dets_dir.mkdir(parents=True, exist_ok=True)
                dp = DetectParams(
                    slices_dir=perim_slices_dir,
                    out_dir=perim_dets_dir,
                    pixel_size_m=0.02,
                )
                # Looser global gates for rim occlusions
                dp.global_min_ringness    = float(args.perim_min_ringness)
                dp.global_grad_align_deg  = float(args.perim_grad_deg)
                dp.global_min_arc_deg     = 100.0

                # Looser ROI recovery gates
                dp.relaxed_min_ringness       = float(args.perim_min_ringness)
                dp.relaxed_grad_align_tol_deg = float(args.perim_grad_deg)
                dp.relaxed_min_arc_deg        = 100.0

                detector.run(dp)
                print("[perim][detect] done.")

                # Merge + refine the border-only detections
                print("[perim][merge ] starting …")
                perim_merged_dir.mkdir(parents=True, exist_ok=True)
                merger.run(input_dir=perim_dets_dir, out_dir=perim_merged_dir)
                print("[perim][merge ] done.")

                print("[perim][refine] starting …")
                perim_refine_debug.mkdir(parents=True, exist_ok=True)
                refiner.run(slices_dir=perim_slices_dir, merged_dir=perim_merged_dir, out_dir=perim_refine_debug)
                print("[perim][refine] done.")

                # Export perimeter PLYs only if refinement produced a CSV
                perim_refined_csv = perim_merged_dir / "refined_per_slice_circles.csv"

                perim_refined_csv = perim_merged_dir / "refined_per_slice_circles.csv"
                if perim_refined_csv.exists():
                    perim_forks_out = perim_merged_dir / "forks_refined_per_slice_circles.csv"
                    print("[perim][fork ] starting …")
                    forks.run(slices_dir=perim_slices_dir, merged_dir=perim_merged_dir, out_csv=perim_forks_out)
                    print(f"[perim][fork ] wrote → {perim_forks_out}")

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
                        allow_ransac_refine=args.allow_ransac_refine,
                    )
                    print("[perim][export] done.")
                else:
                    print("[perim] skip export (no refined_per_slice_circles.csv)")
            else:
                print("[perim] no points left after cropping/exclusion; skipping perimeter pass.")

        """

        # 4) Export curved stems from the MAIN refined set
        stems_out_dir.mkdir(parents=True, exist_ok=True)
        print("[export] starting …")
        exporter.run(
            slices_dir=slices_dir,
            merged_dir=merged_dir,
            out_dir=stems_out_dir,
            refined_csv=merged_dir / "refined_per_slice_circles.csv",
            csv_dets=merged_dir / "detections_with_clusters.csv",
            csv_tr=merged_dir / "slice_translations.csv",
            write_colored_all=True,
            allow_ransac_refine=args.allow_ransac_refine,
        )
        print("[export] done.")

    print("\nAll datasets complete.")


if __name__ == "__main__":
    main()
