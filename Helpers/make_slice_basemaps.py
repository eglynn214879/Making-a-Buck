# make_slice_basemaps.py
"""
Exports the slice cross-sections as debug images before circle detection 
for inspection and validation.
"""
import argparse, re
from pathlib import Path
import numpy as np
import cv2

# --------- config defaults ----------
PIXEL_SIZE_M = 0.02   # metres per pixel
PADDING_M    = 0.5    # border around overall XY extent
GAUSS_SIGMA  = 1.2    # for the heatmap blur

SLICE_RE = re.compile(r"slice_(?P<h>[0-9]+\.[0-9]+)m\.ply")

# --------- I/O ----------
def load_ply_xyz_ascii(path: Path) -> np.ndarray:
    """
    Robust ASCII PLY reader — reads until end_header, then parses XYZ float32.
    Works even for single-line data or files with extra columns.
    """
    with open(path, "r", errors="ignore") as f:
        # read header line by line until "end_header"
        while True:
            ln = f.readline()
            if not ln:
                raise ValueError(f"Invalid PLY (no end_header): {path}")
            if ln.strip() == "end_header":
                break

        # np.loadtxt directly from the open handle (no tell() issues)
        arr = np.loadtxt(f, dtype=np.float32)

    if arr.ndim == 1:
        if arr.size < 3:
            return np.zeros((0, 3), np.float32)
        arr = arr[None, :]

    if arr.size == 0:
        return np.zeros((0, 3), np.float32)

    return arr[:, :3].astype(np.float32)

# --------- global transform over all slices ----------
def compute_global_tf(slices_dir: Path, pixel_size=PIXEL_SIZE_M, padding=PADDING_M):
    xs_min, xs_max, ys_min, ys_max = [], [], [], []
    for p in sorted(slices_dir.glob("slice_*.ply")):
        try:
            xyz = load_ply_xyz_ascii(p)
        except Exception:
            continue
        if xyz.size:
            xs, ys = xyz[:, 0], xyz[:, 1]
            xs_min.append(xs.min()); xs_max.append(xs.max())
            ys_min.append(ys.min()); ys_max.append(ys.max())
    if not xs_min:
        raise SystemExit(f"No readable slice_*.ply found in {slices_dir}")
    xmin = float(min(xs_min) - padding); xmax = float(max(xs_max) + padding)
    ymin = float(min(ys_min) - padding); ymax = float(max(ys_max) + padding)
    W = int(np.ceil((xmax - xmin) / pixel_size))
    H = int(np.ceil((ymax - ymin) / pixel_size))
    return {"xmin": xmin, "ymax": ymax, "pixel": pixel_size, "W": W, "H": H}

def world_to_pixel(tf, xy):
    px = (xy[:, 0] - tf["xmin"]) / tf["pixel"]
    py = (tf["ymax"] - xy[:, 1]) / tf["pixel"]
    return np.stack([px, py], 1).astype(np.float32)

# --------- rasterizers ----------
def raster_points(tf, xy):
    H, W = tf["H"], tf["W"]
    img = np.zeros((H, W), np.uint8)
    if xy.size == 0:
        return img
    ij = np.rint(world_to_pixel(tf, xy)).astype(np.int32)
    ij[:, 0] = np.clip(ij[:, 0], 0, W - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, H - 1)
    img[ij[:, 1], ij[:, 0]] = 255
    return img

def raster_heat(tf, xy, sigma=GAUSS_SIGMA):
    H, W = tf["H"], tf["W"]
    heat = np.zeros((H, W), np.float32)
    if xy.size == 0:
        return np.zeros((H, W), np.uint8)
    ij = np.rint(world_to_pixel(tf, xy)).astype(np.int32)
    ij[:, 0] = np.clip(ij[:, 0], 0, W - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, H - 1)
    np.add.at(heat, (ij[:, 1], ij[:, 0]), 1.0)
    heat = cv2.GaussianBlur(heat, (0, 0), sigma)
    return (255.0 * (heat / heat.max())).astype(np.uint8) if heat.max() > 0 else np.zeros((H, W), np.uint8)

# --------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Render pre-annotation slice images (no circles).")
    ap.add_argument("--slices", required=True, help="Folder with slice_*.ply")
    ap.add_argument("--out", required=True, help="Output folder for PNGs")
    ap.add_argument("--pixel", type=float, default=PIXEL_SIZE_M, help="Metres per pixel (default 0.02)")
    ap.add_argument("--padding", type=float, default=PADDING_M, help="Border around XY extent (m)")
    ap.add_argument("--sigma", type=float, default=GAUSS_SIGMA, help="Gaussian sigma for heatmap blur")
    args = ap.parse_args()

    slices_dir = Path(args.slices)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tf = compute_global_tf(slices_dir, pixel_size=args.pixel, padding=args.padding)
    print(f"[make_slice_basemaps] global raster {tf['H']}x{tf['W']} @ {tf['pixel']} m/px")

    for p in sorted(slices_dir.glob("slice_*.ply"),
                    key=lambda q: float(SLICE_RE.match(q.name).group("h")) if SLICE_RE.match(q.name) else 0.0):
        xyz = load_ply_xyz_ascii(p)
        xy = xyz[:, :2] if xyz.size else np.zeros((0, 2), np.float32)

        dots = raster_points(tf, xy)
        cv2.imwrite(str(out_dir / f"{p.stem}_dots.png"), dots)

        heat = raster_heat(tf, xy, sigma=args.sigma)
        cv2.imwrite(str(out_dir / f"{p.stem}_heat.png"), heat)

        color = cv2.applyColorMap(heat, cv2.COLORMAP_OCEAN)
        cv2.imwrite(str(out_dir / f"{p.stem}_heat_color.png"), color)

    print(f"[make_slice_basemaps] wrote PNGs → {out_dir}")

if __name__ == "__main__":
    main()
