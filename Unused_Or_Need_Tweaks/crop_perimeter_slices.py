# ======================================================================
#  perimeter_band_crop.py
#  -------------------------------------------------------------
#  Purpose
#    From each ASCII PLY slice (slice_<height>m.ply), keep only a thin
#    perimeter band of points and optionally exclude points belonging
#    to known stems. Intended to create “border-only” datasets for
#    perimeter search / recovery experiments.
#
#  How it works
#    • Parses height from filename via regex: slice_<h>m.ply
#    • Computes a footprint from XY (bbox + circumscribed radius)
#    • Builds a perimeter mask as either:
#        - a circular band near the outer radius, and/or
#        - a thin band along the bounding box edges
#      controlled by --perim_frac (fraction of extent).
#    • Optionally loads stem centers/radii from a CSV
#      (x_corr_m, y_corr_m, radius_m, height_m, ok) and removes points
#      within stem_expand * radius for slices at the same (rounded) height.
#    • Writes cropped PLY per slice and a color PNG debug raster:
#        gray = all points, red = excluded (stems), green = kept perimeter.
#
#  Inputs
#    --slices_dir   Directory containing ASCII PLY slices: slice_<h>m.ply
#    --out_dir      Output directory for cropped PLY (and debug/ PNGs)
#    --perim_frac   Perimeter band thickness as a fraction of extent [0.10]
#    --stems_csv    Optional CSV of refined circles (columns: height_m,
#                   x_corr_m, y_corr_m, radius_m, ok). Rows with ok==0
#                   are ignored. Heights are matched with 50 mm rounding
#                   and a small ±0.01 m tolerance.
#    --stem_expand  Multiplier applied to radius when excluding stems [1.2]
#    --no_debug     Disable debug PNGs
#
#  Outputs
#    <out_dir>/slice_<h>m.ply            Cropped points (per slice)
#    <out_dir>/debug/slice_<h>m_crop.png Color overlay for QA
#
#  Assumptions & Notes
#    • PLY files are ASCII with x y z columns; units are metres.
#    • Height token in filenames is decimal metres (e.g., 1.20).
#    • If a slice has no points, an empty PLY is still written.
#    • Stem heights are grouped by 50 mm buckets (round_mm=50) and also
#      accepted if |h_csv - h_file| ≤ 0.01 m.
#    • OpenCV is only required for debug PNGs.
#
#  Example
#    python perimeter_band_crop.py \
#      --slices_dir Work/CT03/slices \
#      --out_dir Work/CT03/perim_only \
#      --perim_frac 0.12 \
#      --stems_csv Work/CT03/merged_stems/refined_per_slice_circles.csv
#
#  Dependencies
#    Python 3.9+, numpy, opencv-python (cv2) for debug images
#
#  Author
#    Ethan Glynn (header added for readability)
# ======================================================================

import re, csv, math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

# Regex to pull the numeric height (in metres) from filenames like: slice_1.20m.ply
SLICE_RE = re.compile(r"slice_(?P<h>[0-9]+\.[0-9]+)m\.ply")

def parse_h(name: str) -> Optional[float]:
    """
    Extract height (m) from a slice filename.
    Returns None if the pattern doesn't match.
    """
    m = SLICE_RE.match(name)
    return float(m.group("h")) if m else None


# ---------- PLY I/O ----------
def load_ply_xyz_ascii(filepath: Path) -> np.ndarray:
    """
    Load an ASCII PLY file and return an (N,3) float32 array of [x, y, z].
    Assumes the header ends with the line 'end_header' and that the
    remaining lines are numeric. Returns an empty (0,3) if no data.
    """
    with open(filepath, "r", errors="ignore") as f:
        header = []
        # Consume header until 'end_header'
        for ln in f:
            header.append(ln)
            if ln.strip() == "end_header":
                break
        # np.loadtxt reads the numeric section only
        arr = np.loadtxt(filepath, skiprows=len(header), dtype=np.float32)

    # If there is exactly one vertex, ensure we still return (N,3)
    if arr.ndim == 1:
        arr = arr[None, :]

    # Trim to XYZ if there are extra columns
    return arr[:, :3].astype(np.float32) if arr.size else np.zeros((0,3), np.float32)

def write_ply_xyz_ascii(filepath: Path, xyz: np.ndarray):
    """
    Write an (N,3) array of [x, y, z] to an ASCII PLY file with a minimal header.
    Creates parent directories as needed.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    n = int(xyz.shape[0])
    with open(filepath, "w") as f:
        # Minimal header
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        # Body
        if n:
            np.savetxt(f, xyz, fmt="%.6f")


# ---------- Footprint + band ----------
def footprint_from_xy(xy: np.ndarray) -> Dict[str, float]:
    """
    Compute a simple footprint of 2D points:
      - axis-aligned bbox (xmin/xmax/ymin/ymax)
      - center (cx, cy) = bbox midpoint
      - radius r = max distance from center to any point (circumscribed)
    Returns a dict with bbox, center, and r.
    """
    xmin, ymin = np.min(xy, axis=0)
    xmax, ymax = np.max(xy, axis=0)
    cx = 0.5*(xmin + xmax)
    cy = 0.5*(ymin + ymax)
    # Circumscribed radius from bbox-center
    r = float(np.sqrt(((xy[:,0]-cx)**2 + (xy[:,1]-cy)**2)).max())
    return dict(cx=float(cx), cy=float(cy), r=r,
                xmin=float(xmin), xmax=float(xmax), ymin=float(ymin), ymax=float(ymax))

def perimeter_mask_world(xy: np.ndarray, fp: Dict[str, float], frac: float) -> np.ndarray:
    """
    Build a boolean mask selecting points in a thin perimeter band.
    Two complementary bands are OR'ed together:
      1) Circular band near the outer radius of the footprint circle.
      2) Rectangular band hugging the bounding-box edges.
    'frac' is a thickness fraction (e.g., 0.10 = 10% near the edge).
    """
    # ---- circular band near fp['r'] ----
    r = np.hypot(xy[:,0] - fp["cx"], xy[:,1] - fp["cy"])
    r_norm = r / max(fp["r"], 1e-6)
    circ_band = (r_norm >= (1.0 - frac)) & (r_norm <= 1.0001)

    # ---- rectangular band near bbox edges ----
    hx = 0.5*(fp["xmax"] - fp["xmin"])
    hy = 0.5*(fp["ymax"] - fp["ymin"])
    # Distance to nearest vertical edge, normalized by half-width
    dx = np.minimum(xy[:,0] - fp["xmin"], fp["xmax"] - xy[:,0]) / max(hx, 1e-6)
    # Distance to nearest horizontal edge, normalized by half-height
    dy = np.minimum(xy[:,1] - fp["ymin"], fp["ymax"] - xy[:,1]) / max(hy, 1e-6)
    # Near any edge if min normalized distance <= frac
    box_band = (np.minimum(dx, dy) <= frac + 1e-6)

    # Keep a point if it's in *either* perimeter band
    return circ_band | box_band


# ---------- Stem exclusion ----------
def load_stem_circles(csv_path: Path,
                      x_col="x_corr_m", y_col="y_corr_m", r_col="radius_m",
                      h_col="height_m", ok_col="ok",
                      round_mm=50) -> Dict[float, List[Tuple[float,float,float]]]:
    """
    Load per-slice stem circle parameters from a CSV (refined circles).
    Returns a dict keyed by rounded height (metres) -> list of (x, y, r).
    - Rows with ok==0/False are ignored (keeps only valid circles).
    - Heights are grouped into round_mm buckets (default 50 mm).
    """
    def round_h(v: float) -> float:
        step = round_mm / 1000.0  # convert mm to metres
        return round(v / step) * step

    out: Dict[float, List[Tuple[float,float,float]]] = {}
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                # skip failed detections if 'ok' is present
                if ok_col in row and str(row[ok_col]).strip() in ("0", "False", "false"):
                    continue
                h  = float(row[h_col])
                xc = float(row[x_col]); yc = float(row[y_col])
                rr = float(row[r_col])
            except Exception:
                # silently skip malformed rows
                continue
            key = round_h(h)
            out.setdefault(key, []).append((xc, yc, rr))
    return out


# ---------- Debug ----------
def save_debug_png(xy, kept_mask, excl_mask, out_png, pixel_size=0.02, padding=0.3):
    """
    Rasterize a tiny debug PNG for a slice:
      - gray  = all points,
      - red   = excluded by stem masks,
      - green = kept by perimeter band (and not excluded).
    The world→pixel mapping is built from the data bounds with 'padding'.
    """
    if xy.size == 0:
        return

    # World bounds with padding
    x, y = xy[:,0], xy[:,1]
    xmin, xmax = x.min()-padding, x.max()+padding
    ymin, ymax = y.min()-padding, y.max()+padding

    # Output raster size (keep it simple and metric)
    W = int(np.ceil((xmax-xmin)/pixel_size))
    H = int(np.ceil((ymax-ymin)/pixel_size))

    # World → pixel helper
    def to_px(xw, yw):
        px = (xw - xmin)/pixel_size
        py = (ymax - yw)/pixel_size   # flip Y for image coordinates
        return int(np.clip(round(px), 0, W-1)), int(np.clip(round(py), 0, H-1))

    img = np.zeros((H, W, 3), np.uint8)

    # Plot all points as gray
    for xi, yi in xy:
        px, py = to_px(xi, yi)
        img[py, px] = (110, 110, 110)

    # Excluded (stems) as red
    for (xi, yi) in xy[excl_mask]:
        px, py = to_px(xi, yi)
        img[py, px] = (0, 0, 220)

    # Kept perimeter (that are not excluded) as green
    for (xi, yi) in xy[kept_mask & ~excl_mask]:
        px, py = to_px(xi, yi)
        img[py, px] = (0, 220, 0)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), img)

# ---------- Core ----------
def crop_one_slice(ply_path: Path,
                   out_dir: Path,
                   frac: float,
                   stems_by_h: Optional[Dict[float, List[Tuple[float,float,float]]]] = None,
                   stem_expand: float = 1.2,
                   height_tolerance_m: float = 1e-3,
                   write_debug: bool = True) -> int:
    """
    Crop a single slice to a perimeter band and (optionally) exclude known stems.

    Args:
        ply_path: Path to slice_XX.XXm.ply (ASCII PLY with XYZ).
        out_dir:  Folder to write the cropped PLY.
        frac:     Thickness of the perimeter band as a fraction of footprint size,
                  e.g. 0.10 = keep ~10% near the outer edge.
        stems_by_h: Optional dict of {rounded_height_m: [(x, y, r), ...]} used to
                    exclude points near previously detected stem rings.
        stem_expand: Multiplier on r when masking stems (gives a little buffer).
        height_tolerance_m: Height matching tolerance for stems_by_h buckets.
        write_debug: If True, write a small PNG showing keep/exclude masks.

    Returns:
        Number of points kept (written) for this slice.
    """
    # Load full slice point cloud (XYZ); short-circuit if empty.
    xyz = load_ply_xyz_ascii(ply_path)
    if xyz.size == 0:
        write_ply_xyz_ascii(out_dir / ply_path.name, xyz)
        return 0

    # Parse the numeric height from the filename (e.g., slice_1.20m.ply -> 1.20)
    h = parse_h(ply_path.name)

    # Build perimeter band mask in world coordinates
    xy = xyz[:, :2]
    fp = footprint_from_xy(xy)                 # bbox + center + circumscribed radius
    perim = perimeter_mask_world(xy, fp, frac) # boolean mask: perimeter band

    # Optional exclusion mask from known stem circles at (approximately) this height
    excl = np.zeros(xy.shape[0], dtype=bool)
    if stems_by_h and (h is not None):
        # Accept exact bucket or a close neighbour within tolerance (plus a 1 cm leniency)
        keys = [k for k in stems_by_h.keys()
                if (abs(k - h) <= height_tolerance_m) or (abs(k - h) <= 0.01)]
        for key in keys:
            for (sx, sy, sr) in stems_by_h[key]:
                rr = sr * stem_expand
                # Mark points within the (expanded) stem circle for exclusion
                excl |= (np.hypot(xy[:, 0] - sx, xy[:, 1] - sy) <= rr)

    # Keep points that lie in the perimeter band and are NOT inside an excluded stem
    keep = perim & (~excl)
    out_xyz = xyz[keep]

    # Write cropped slice
    write_ply_xyz_ascii(out_dir / ply_path.name, out_xyz)

    # Optional debug raster: gray=all, red=excluded stems, green=kept perimeter
    if write_debug:
        dbg = out_dir / "debug" / f"{ply_path.stem}_crop.png"
        save_debug_png(xy, perim, excl, dbg)

    return int(out_xyz.shape[0])


# ---------- Run entrypoint ----------
def run(slices_dir: Path,
        out_dir: Path,
        perim_frac: float = 0.10,
        stems_csv: Optional[Path] = None,
        stem_expand: float = 1.2,
        write_debug: bool = True) -> None:
    """
    Batch-process all slice_*.ply in a directory:
      1) Keep only a perimeter band (perim_frac) of each slice.
      2) Optionally exclude points near known stem rings loaded from stems_csv.
      3) Write cropped PLYs and a summary of kept/total points.

    Args:
        slices_dir: Directory containing slice_*.ply files.
        out_dir:    Output directory for the cropped PLYs (+ optional debug PNGs).
        perim_frac: Perimeter band thickness as a fraction of footprint size.
        stems_csv:  Optional CSV with refined circles (x_corr_m, y_corr_m, radius_m, height_m[, ok]).
        stem_expand:Multiply circle radius when masking stems.
        write_debug:Write debug PNG overlays for each slice.
    """
    # Load stems (if provided) into a {rounded_height_m: [(x, y, r), ...]} map.
    stems_by_h = None
    if stems_csv and stems_csv.exists():
        stems_by_h = load_stem_circles(stems_csv)

    # Enumerate slices in height order for reproducible processing
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_files = sorted(
        [p for p in slices_dir.glob("slice_*.ply") if p.is_file()],
        key=lambda p: parse_h(p.name) or 0.0
    )

    total_in, total_out = 0, 0

    # Process slices
    for p in ply_files:
        n_kept = crop_one_slice(
            p, out_dir, perim_frac,
            stems_by_h=stems_by_h,
            stem_expand=stem_expand,
            write_debug=write_debug
        )
        total_out += n_kept

        # Count original points for a simple kept/total summary
        xyz = load_ply_xyz_ascii(p)
        total_in += xyz.shape[0]

    # Final summary
    print(f"[done] wrote cropped slices to {out_dir}")
    print(f"       kept {total_out}/{total_in} points (~{100.0*total_out/max(total_in,1):.1f}%)")

# keep CLI for standalone use
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--perim_frac", type=float, default=0.10)
    ap.add_argument("--stems_csv", type=Path, default=None)
    ap.add_argument("--stem_expand", type=float, default=1.2)
    ap.add_argument("--no_debug", action="store_true")
    args = ap.parse_args()
    run(args.slices_dir, args.out_dir,
        perim_frac=args.perim_frac,
        stems_csv=args.stems_csv,
        stem_expand=args.stem_expand,
        write_debug=not args.no_debug)

