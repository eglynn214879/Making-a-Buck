# ======================================================================
#  ground_mesh.py
#  -------------------------------------------------------------
#  Stage: Ground synthesis + slicing above ground
#  -------------------------------------------------------------
#  Purpose:
#     • Ensures each dataset directory contains a valid ground mesh.
#     • If none exists, synthesizes one from raw point cloud data by
#       taking minimum Z per XY grid cell.
#     • Builds a ground height map and produces horizontal slices
#       above the ground until point density becomes sparse.
#
#  Inputs:
#     - raw_*.ply              : Full point cloud (XYZ, ASCII PLY)
#     - ground_*.ply (optional): Existing ground mesh
#
#  Outputs:
#     - ground_*.ply           : Synthesized ground (if missing)
#     - slice_*.ply            : Height-banded slices above ground
#     - manifest.json (stdout) : Metadata summary of operation
#
#  Typical usage:
#     python ground_mesh.py --dataset_dir data/plot_HQPLR118V1 \
#                           --out_dir Work/HQPLR118V1/slices
#
#  Dependencies:
#     numpy, tqdm (optional for progress bar)
#
#  Author 
#   Ethan Glynn
# ======================================================================

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


# ----------------- I/O helpers -----------------

def load_ply_xyz(filepath: Path) -> np.ndarray:
    """
    Load ASCII PLY with at least x y z as the first 3 columns.
    """
    with filepath.open("r", errors="ignore") as f:
        header_lines = 0
        for line in f:
            header_lines += 1
            if line.strip() == "end_header":
                break
    arr = np.loadtxt(filepath, skiprows=header_lines, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, :3].astype(np.float32) if arr.size else np.zeros((0, 3), np.float32)

def save_ply_xyz(filename: Path, points: np.ndarray) -> None:
    n = int(points.shape[0])
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w") as f:
        f.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "end_header\n"
        )
        for x, y, z in points:
            f.write(f"{float(x)} {float(y)} {float(z)}\n")


# ----------------- core logic -----------------

def build_ground_map(ground_points: np.ndarray, grid_size: float = 1.0) -> Dict[Tuple[int, int], float]:
    """
    For each XY grid cell, store the minimum z (ground) by finding local minima
    """
    ground_map: Dict[Tuple[int, int], float] = {}
    if ground_points.size == 0:
        return ground_map
    gx = (ground_points[:, 0] / grid_size).astype(np.int64)
    gy = (ground_points[:, 1] / grid_size).astype(np.int64)
    for (cx, cy, z) in zip(gx, gy, ground_points[:, 2]):
        key = (int(cx), int(cy))
        v = ground_map.get(key)
        if v is None or z < v:
            ground_map[key] = float(z)
    return ground_map

def slice_above_ground_until_sparse(
    all_points: np.ndarray,
    ground_map: Dict[Tuple[int, int], float],
    grid_size: float = 1.0,
    slice_start: float = 0.5,
    slice_step: float = 0.25,
    min_points: int = 1000,
    use_tqdm: bool = False,
) -> Dict[float, np.ndarray]:
    """
    Generate horizontal slices at heights (ground_z + h .. + h+step) for each cell,
    stopping when a slice returns fewer than min_points.
    """
    slices: Dict[float, np.ndarray] = {}
    current_level = float(slice_start)

    while True:
        iterator = tqdm(all_points, desc=f"Slicing {current_level:.2f}–{current_level + slice_step:.2f} m", leave=False) \
            if (use_tqdm and _HAS_TQDM) else all_points

        buf: List[Tuple[float, float, float]] = []
        for x, y, z in iterator:
            key = (int(x / grid_size), int(y / grid_size))
            gz = ground_map.get(key)
            if gz is None:
                continue
            z0 = gz + current_level
            if z0 <= z < z0 + slice_step:
                buf.append((x, y, z))

        if len(buf) < int(min_points):
            break

        slices[current_level] = np.asarray(buf, dtype=np.float32)
        current_level += float(slice_step)

    return slices


# --------- ground synthesis (min-Z per XY grid cell) ----------

def synthesize_ground_from_raw(raw_points: np.ndarray, grid_size: float = 1.0) -> np.ndarray:
    """
    Build a sparse ground cloud by taking the minimum Z per XY grid cell.
    We place each ground point at the *cell center* with z = min z of that cell.
    """
    if raw_points.size == 0:
        return np.zeros((0, 3), np.float32)

    gx = np.floor(raw_points[:, 0] / grid_size).astype(np.int64)
    gy = np.floor(raw_points[:, 1] / grid_size).astype(np.int64)
    z = raw_points[:, 2]

    # Map (gx,gy) -> min z
    cell_min: Dict[Tuple[int, int], float] = {}
    for i in range(raw_points.shape[0]):
        key = (int(gx[i]), int(gy[i]))
        v = cell_min.get(key)
        zi = float(z[i])
        if v is None or zi < v:
            cell_min[key] = zi

    # Emit one point per cell at cell center
    out = []
    half = 0.5 * float(grid_size)
    for (cx, cy), zmin in cell_min.items():
        x = cx * grid_size + half
        y = cy * grid_size + half
        out.append((float(x), float(y), float(zmin)))
    return np.asarray(out, dtype=np.float32)


# ----------------- discovery & run -----------------

def _discover_inputs(dataset_dir: Path, cfg: Dict) -> Tuple[Path, Optional[Path]]:
    """
    Try to find raw and (optionally) an existing ground PLY.
    - explicit: cfg['raw_path'], cfg['ground_path']
    - else: search raw_glob (default '*.ply', excluding 'ground_*') and ground_glob (default 'ground_*.ply')
    Returns (raw_path, ground_path_or_None)
    """
    # explicit overrides
    raw_path = cfg.get("raw_path")
    ground_path = cfg.get("ground_path")
    if raw_path:
        return Path(raw_path), (Path(ground_path) if ground_path else None)

    raw_glob = cfg.get("raw_glob", "*.ply")
    ground_glob = cfg.get("ground_glob", "ground_*.ply")

    # candidates
    raw_candidates = sorted([p for p in dataset_dir.glob(raw_glob) if p.is_file() and not p.name.startswith("ground_")])
    if not raw_candidates:
        raise FileNotFoundError(f"No raw PLY found in {dataset_dir} (glob='{raw_glob}', excluding 'ground_*').")
    # pick largest raw by size (heuristic)
    raw = max(raw_candidates, key=lambda p: p.stat().st_size)

    ground_candidates = sorted([p for p in dataset_dir.glob(ground_glob) if p.is_file()])
    ground = ground_candidates[0] if ground_candidates else None
    return raw, ground


def run(
    dataset_dir: Path,
    out_dir: Path,
    config: Optional[Dict] = None,
    logger: Optional["logging.Logger"] = None,
) -> Dict:
    """
    Stage: (1) ensure a ground mesh exists (auto-synthesize if missing), then
           (2) slice the cloud into horizontal bands relative to ground.

    Inputs are discovered within dataset_dir unless overridden by config.

    Config keys (all optional):
      raw_path: str (explicit raw file)
      ground_path: str (explicit existing ground file)
      ground_out_path: str (where to write synthesized ground if missing)
      raw_glob: '*.ply'
      ground_glob: 'ground_*.ply'
      grid_size: 1.0
      slice_start: 0.5
      slice_step: 0.25
      min_points: 1000
      use_tqdm: False
    """
    import hashlib
    from datetime import datetime, timezone

    cfg = dict(
        grid_size=1.0,
        slice_start=0.3,
        slice_step=0.125,
        min_points=1000,
        use_tqdm=False,
        ground_out_path=None,
    )
    if config:
        cfg.update(config)

    log = (logger.info if logger else print)
    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path, ground_path = _discover_inputs(dataset_dir, cfg)

    # Load raw once (we may need it for ground synthesis)
    all_points = load_ply_xyz(raw_path)
    if all_points.size == 0:
        raise FileNotFoundError(f"Raw PLY appears empty: {raw_path}")

    # Ensure we have a ground cloud
    if ground_path is None:
        # choose output path
        ground_out_path = Path(cfg["ground_out_path"]) if cfg.get("ground_out_path") else (dataset_dir / f"ground_{raw_path.name}")
        log(f"[ground_mesh] No ground found → synthesizing: {ground_out_path.name}")
        ground_points = synthesize_ground_from_raw(all_points, grid_size=float(cfg["grid_size"]))
        save_ply_xyz(ground_out_path, ground_points)
        ground_path = ground_out_path
    else:
        log(f"[ground_mesh] Using existing ground: {ground_path.name}")
        ground_points = load_ply_xyz(ground_path)

    # Build ground map and slice
    ground_map = build_ground_map(ground_points, grid_size=float(cfg["grid_size"]))
    slices = slice_above_ground_until_sparse(
        all_points=all_points,
        ground_map=ground_map,
        grid_size=float(cfg["grid_size"]),
        slice_start=float(cfg["slice_start"]),
        slice_step=float(cfg["slice_step"]),
        min_points=int(cfg["min_points"]),
        use_tqdm=bool(cfg["use_tqdm"]),
    )

    # Save slices
    saved = []
    for height, pts in sorted(slices.items(), key=lambda kv: kv[0]):
        fn = out_dir / f"slice_{height:.2f}m.ply"
        save_ply_xyz(fn, pts)
        saved.append(str(fn))

    # Manifest & content hash
    h = hashlib.sha256()
    try:
        h.update(raw_path.read_bytes())
    except Exception:
        pass
    try:
        h.update(Path(ground_path).read_bytes())
    except Exception:
        pass
    for k in ("grid_size", "slice_start", "slice_step", "min_points"):
        h.update(str(cfg[k]).encode())

    manifest = {
        "stage": "ground_mesh",
        "version": "2025-09-24",
        "inputs": {
            "dataset_dir": str(dataset_dir),
            "raw_path": str(raw_path),
            "ground_path": str(ground_path),
        },
        "outputs": {
            "slices_dir": str(out_dir),
            "slices": saved,
        },
        "metrics": {
            "total_points_in_raw": int(all_points.shape[0]),
            "num_slices": len(saved),
            "grid_size": float(cfg["grid_size"]),
            "slice_start": float(cfg["slice_start"]),
            "slice_step": float(cfg["slice_step"]),
            "min_points": int(cfg["min_points"]),
        },
        "hash": "sha256:" + h.hexdigest(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }

    log(f"[ground_mesh] Saved {len(saved)} slices → {out_dir}")
    return manifest


# ----------------- tiny CLI wrapper -----------------

if __name__ == "__main__":
    import argparse, logging
    parser = argparse.ArgumentParser(description="Make ground if missing, then slice point cloud above ground.")
    parser.add_argument("--dataset_dir", type=Path, required=True, help="Folder containing raw PLY (and optionally ground_*.ply).")
    parser.add_argument("--out_dir", type=Path, required=True, help="Where to write slice_*.ply files")
    parser.add_argument("--config", type=Path, help="JSON file with config overrides")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = {}
    if args.config and args.config.exists():
        cfg = json.loads(args.config.read_text())

    logger = None
    if not args.quiet:
        logger = logging.getLogger("ground_mesh")
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(ch)

    m = run(args.dataset_dir, args.out_dir, cfg, logger)
    print(json.dumps(m, indent=2))
