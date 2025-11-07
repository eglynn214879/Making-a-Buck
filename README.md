# 🌲 Making-a-Buck

**LiDAR-based tree stem reconstruction and bucking optimisation**

This repository provides a complete Python pipeline for reconstructing tree stems from terrestrial LiDAR point clouds and computing optimal log value recovery (“bucking”) using geometric analysis and dynamic programming.
Each stage can be run independently or via the unified orchestrator script.

---

## 🧩 Pipeline Overview

| Stage | Script                                   | Description                                                                                   |
| ----- | ---------------------------------------- | --------------------------------------------------------------------------------------------- |
| 0     | `ground_mesh.py`                         | Estimate ground surface, remove ground points, and generate height-aligned slice_*.ply files. |
| 1     | `find_stem_circles.py`                   | Detect circular stem cross-sections per slice using Hough + RANSAC on occupancy maps.         |
| 2     | `merge_stems_across_slices.py`           | Merge detections vertically, correct drift, and form stem tracks.                             |
| 3     | `refine_circles_from_tracks.py`          | Re-fit circles guided by predicted track positions (taper / sweep aware).                     |
| 4     | `export_stems_from_slices_curved_alt.py` | Export smoothed 3-D stems and per-stem statistics as PLY / CSV.                               |
| 5     | `bucker.py`                              | Optimise commercial log segmentation using configurable product rules.                        |
| —     | `main_pipeline.py`                       | Runs all stages sequentially for a dataset.                                                   |

Optional utilities cover perimeter recovery, branch/fork detection, and evaluation.

---

## ⚙️ Installation

```bash
git clone https://github.com/ethanglynn/Making-a-Buck.git
cd Making-a-Buck
pip install -r requirements.txt
```

Typical dependencies: `numpy`, `opencv-python`, `pandas`, `matplotlib`.

---

## 🚀 Usage by Stage

### 0️⃣ Ground mesh & slicing

```bash
python ground_mesh.py \
  --in_ply data/plot01.ply \
  --out_dir Work/plot01/slices \
  --step_m 0.0125
```

### 1️⃣ Circle detection

```bash
python find_stem_circles.py \
  --slices_dir Work/plot01/slices \
  --out_dir Work/plot01/detected
```

### 2️⃣ Merge detections across slices

```bash
python merge_stems_across_slices.py \
  --circles_dir Work/plot01/detected \
  --out_dir Work/plot01/merged_stems
```

### 3️⃣ Refine using stem tracks

```bash
python refine_circles_from_tracks.py \
  --slices_dir Work/plot01/slices \
  --merged_dir Work/plot01/merged_stems \
  --out_dir Work/plot01/refined
```

### 4️⃣ Export curved 3-D stems

```bash
python export_stems_from_slices_curved_alt.py \
  --slices_dir Work/plot01/slices \
  --merged_dir Work/plot01/refined \
  --out_csv Work/plot01/stems_curved.csv
```

### 5️⃣ Bucking optimisation

```bash
python bucker.py \
  --in_csv Work/plot01/stems_curved.csv \
  --config tri_method_config.json \
  --out_dir Work/plot01/bucking_results
```

### 🌲 Run end-to-end

```bash
python main_pipeline.py --data_dir Work/plot01
```

---

## 🧪 Optional Tools

### Perimeter recovery

Recover occluded border stems:

```bash
python crop_perimeter_slices.py \
  --slices_dir Work/plot01/slices \
  --out_dir Work/plot01/perim_only \
  --perim_frac 0.12
python find_stem_circles_perim_aware.py --slices_dir Work/plot01/perim_only
```

### Branch / fork analysis

```bash
python branchpoint_from_track_drop.py --refined_csv refined_per_slice_circles.csv
python fork_detect_from_refined.py --slices_dir slices --merged_dir refined
```

### Evaluation & visualisation

```bash
python eval_stems_vs_ground_truth.py --pred_csv stems_curved.csv --truth_csv survey.csv
python make_showcase_plots.py --stems_csv stems_curved.csv --out_dir plots
```

🧪 Experimental / Legacy Scripts

Files in Unused_or_Need_Tweaks/ contain early or alternate versions of some modules.
They are provided for reference and reproducibility but may not be fully functional or up to date.
Details about their intent and limitations are discussed in the accompanying report.

---

## 📁 Output Structure

```
Work/
 └── plot01/
      ├── slices/                   # slice_*.ply (height-indexed)
      ├── detected/                 # per-slice circle detections
      ├── merged_stems/             # cluster summaries + translations
      ├── refined/                  # refined_per_slice_circles.csv
      ├── stems_curved.csv          # final per-stem geometry
      └── bucking_results/          # DP/greedy value tables + logs
```

---

## 💡 Notes

* All coordinates are **metres** in LiDAR frame.
* Slice files follow pattern `slice_<height>m.ply` (ASCII, x y z).
* CSV schemas are consistent across modules (`height_m`, `x_corr_m`, `y_corr_m`, `radius_m`, `ok`).
* The bucking configuration (`tri_method_config.json`) defines product specs and prices.

---

## 🧾 Citation

If this code or workflow is useful in your research, please cite:

> **Ethan Glynn (2025). Making-a-Buck: Developing Dynamic Bucking Algorithms with LiDAR-Based 3D Point Clouds to Optimise Timber Profitability**
> [https://github.com/ethanglynn/Making-a-Buck](https://github.com/ethanglynn/Making-a-Buck)

---

## 🪶 License

MIT License — see `LICENSE` for details.

---
