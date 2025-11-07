#!/usr/bin/env python3
# extra_ply_with_bucking.py
# Exports MeshLab-friendly overlay PLYs, incl. NEW bucking annotations.
#
# Outputs in --out-dir:
#   wire_mesh_overlays.ply
#   branch_markers.ply
#   circle_rings_only.ply
#   dbh_annotations.ply
#   bucking_annotations.ply   <-- NEW
#
# Inputs (under --merged-dir):
#   refined_per_slice_circles.csv  (cluster_id, height_m, x_corr_m, y_corr_m, radius_m, ok)
#   forks_refined_per_slice_circles.csv OR --branch-csv (optional)
#   stems_summary.csv (optional)
#   --bucking-csv (optional; e.g. tri_method_cuts_all_plots.csv)

import csv, math, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

# ---------------- PLY utils ----------------
def write_ply_vertices_edges(path: Path, verts_xyzrgb: np.ndarray, edges_v1v2: np.ndarray):
    n_v = int(verts_xyzrgb.shape[0])
    n_e = int(edges_v1v2.shape[0])
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n_v}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element edge {n_e}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")
        for row in verts_xyzrgb:
            x,y,z,r,g,b = row
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        for v1,v2 in edges_v1v2:
            f.write(f"{int(v1)} {int(v2)}\n")

def palette(n):
    base = np.array([
        [230,25,75],[60,180,75],[255,225,25],[0,130,200],[245,130,48],
        [145,30,180],[70,240,240],[240,50,230],[210,245,60],[250,190,190],
        [0,128,128],[230,190,255],[170,110,40],[255,250,200],[128,0,0],
        [170,255,195],[128,128,0],[0,0,128],[128,128,128],[255,215,0]
    ], dtype=np.uint8)
    reps = int(np.ceil(max(1, n)/len(base)))
    return np.vstack([base]*reps)[:n]

# --------------- CSV readers ---------------
def read_refined_tracks(refined_csv: Path):
    """
    tracks[cid] = [{"h":..., "x":..., "y":..., "r":..., "ok":1/0}, ...] sorted by h
    """
    tracks = defaultdict(list)
    with open(refined_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                cid = int(row["cluster_id"])
                h   = float(row["height_m"])
                x   = float(row.get("x_corr_m", row.get("x", "nan")))
                y   = float(row.get("y_corr_m", row.get("y", "nan")))
                r_m = float(row.get("radius_m", row.get("r", "nan")))
                ok  = int(row.get("ok","1"))
            except Exception:
                continue
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(r_m)):
                continue
            tracks[cid].append({"h":h,"x":x,"y":y,"r":r_m,"ok":ok})
    for cid in list(tracks.keys()):
        tracks[cid] = sorted(tracks[cid], key=lambda d: d["h"])
    return tracks

def _guess_col(cols, *cands):
    for c in cands:
        if c in cols: return c
    low = [c.lower() for c in cols]
    for cand in cands:
        for i,name in enumerate(low):
            if cand.lower() == name or cand.lower() in name:
                return cols[i]
    return None

def read_branch_heights(branch_csv: Path):
    branch_h, centers = {}, {}
    if not branch_csv or not branch_csv.exists():
        return branch_h, centers
    with open(branch_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames or []
        cid_col = _guess_col(cols, "cluster_id","cid","stem_id","id")
        h_col   = _guess_col(cols, "branch_h_m","branch_height_m","fork_h_m","branch_point_h_m","bp_h_m","height_m","h_m")
        x_col   = _guess_col(cols, "x_corr_m","x_m","x")
        y_col   = _guess_col(cols, "y_corr_m","y_m","y")
        for row in r:
            try:
                cid = int(row[cid_col]) if cid_col else None
                h   = float(row[h_col])  if h_col  else None
            except Exception:
                continue
            if cid is None or h is None or not np.isfinite(h):
                continue
            branch_h[cid] = h
            if x_col and y_col:
                try:
                    x = float(row[x_col]); y = float(row[y_col])
                    if np.isfinite(x) and np.isfinite(y):
                        centers[cid] = (x,y)
                except Exception:
                    pass
    return branch_h, centers

def read_stems_summary(stems_summary_csv: Path):
    out = {}
    if not stems_summary_csv or not stems_summary_csv.exists():
        return out
    with open(stems_summary_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames or []
        cid_col = _guess_col(cols, "cluster_id","cid","stem_id","id")
        dbh_m_col  = _guess_col(cols, "dbh_m","DBH_m","diameter_m","median_DBH_m")
        dbh_mm_col = _guess_col(cols, "dbh_mm","DBH_mm")
        bh_col     = _guess_col(cols, "bh_m","breast_h_m","breast_height_m")
        x_col      = _guess_col(cols, "x_corr_m","x_m","x")
        y_col      = _guess_col(cols, "y_corr_m","y_m","y")
        for row in r:
            try:
                cid = int(row[cid_col]) if cid_col else None
            except Exception:
                continue
            if cid is None:
                continue
            dbh_m = None
            if dbh_m_col and row.get(dbh_m_col, "") not in ("", None):
                try: dbh_m = float(row[dbh_m_col])
                except: pass
            if dbh_m is None and dbh_mm_col and row.get(dbh_mm_col, "") not in ("", None):
                try: dbh_m = float(row[dbh_mm_col]) / 1000.0
                except: pass
            if dbh_m is None or not np.isfinite(dbh_m) or dbh_m <= 0:
                continue
            try:
                bh_m = float(row[bh_col]) if bh_col and row.get(bh_col,"") not in ("",None) else 1.3
            except Exception:
                bh_m = 1.3
            cx = cy = None
            if x_col and y_col:
                try:
                    cx = float(row[x_col]); cy = float(row[y_col])
                    if not (np.isfinite(cx) and np.isfinite(cy)):
                        cx = cy = None
                except Exception:
                    cx = cy = None
            out[cid] = {"dbh_m": dbh_m, "bh_m": bh_m, "cx": cx, "cy": cy}
    return out

# -------- NEW: bucking cuts reader --------
def read_bucking_csv(bucking_csv: Path, merged_dir: Path):
    """
    Reads bucking cuts from a multi-plot CSV.
    - Accepts columns: stem_id/cluster_id/...; z_start_m/z_end_m (intervals); cut_h_m/height_m (points).
    - Maps methods:
        dp_advanced -> advanced
        dp_base     -> heuristic
        greedy      -> IGNORED
    - If a 'plot_id' column exists, only rows whose plot_id contains the current merged_dir are kept.
    Returns: dict[cid] -> list of cuts dicts.
    """
    cuts = defaultdict(list)
    if not bucking_csv or not bucking_csv.exists():
        return cuts

    def _guess_col(cols, *cands):
        for c in cands:
            if c in cols: return c
        low = [c.lower() for c in cols]
        for cand in cands:
            cl = cand.lower()
            for i,name in enumerate(low):
                if cl == name or cl in name:
                    return cols[i]
        return None

    with open(bucking_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames or []

        # columns in your file:
        plot_col  = _guess_col(cols, "plot_id")
        cid_col   = _guess_col(cols, "stem_id","cluster_id","cid","stem","id","track_id")
        method_col= _guess_col(cols, "method","algo","algorithm")
        # interval heights (your file uses z_start_m / z_end_m)
        h1_col    = _guess_col(cols, "z_start_m","start_h_m","h_start_m","start_m","h1_m","h1")
        h2_col    = _guess_col(cols, "z_end_m","end_h_m","h_end_m","end_m","h2_m","h2")
        # optional point cuts
        h_col     = _guess_col(cols, "cut_h_m","height_m","h_m","cut_height_m","cut_h")

        merged_key = merged_dir.as_posix()

        def map_method(s: str) -> str:
            s = (s or "").strip().lower()
            if s.startswith("dp_advanced"): return "advanced"
            if s.startswith("dp_base"):     return "heuristic"
            if s.startswith("heur"):        return "heuristic"
            if s.startswith("adv"):         return "advanced"
            return ""  # includes 'greedy' -> ignored

        def to_float(v):
            try: 
                return float(v)
            except Exception:
                return None

        for row in r:
            # filter to current plot if plot_id exists
            if plot_col and row.get(plot_col):
                if merged_key not in row[plot_col].replace("\\","/"):
                    continue

            try:
                cid = int(row[cid_col]) if cid_col and row.get(cid_col,"")!="" else None
            except Exception:
                cid = None
            if cid is None:
                continue

            method = map_method(row.get(method_col,"") if method_col else "")
            if method not in ("heuristic","advanced"):
                continue  # ignore 'greedy' and unknowns

            # interval first
            h1 = to_float(row.get(h1_col)) if h1_col else None
            h2 = to_float(row.get(h2_col)) if h2_col else None
            if h1 is not None and h2 is not None and np.isfinite(h1) and np.isfinite(h2):
                if h2 < h1: h1, h2 = h2, h1
                cuts[cid].append({"type":"interval","h1":float(h1),"h2":float(h2),"method":method})
                continue

            # point (rare with your file, but supported)
            hh = to_float(row.get(h_col)) if h_col else None
            if hh is not None and np.isfinite(hh):
                cuts[cid].append({"type":"point","h":float(hh),"method":method})

    for cid in list(cuts.keys()):
        cuts[cid] = sorted(cuts[cid], key=lambda c: (c.get("h", c.get("h1", 0.0))))
    return cuts


# --------------- Interp helpers ---------------
def interp_at_h(samples, key):
    hs = np.array([s["h"] for s in samples], dtype=np.float32)
    vs = np.array([s[key] for s in samples], dtype=np.float32)
    if hs.size == 0:
        return lambda _: np.nan
    if hs.size == 1:
        v0 = float(vs[0])
        return lambda _h: v0
    def f(hq):
        return float(np.interp(hq, hs, vs))
    return f

# --------------- Geometry builders ---------------
def build_wire_mesh(tracks: dict, n_angles: int = 48):
    color_map = {cid: col for cid, col in zip(sorted(tracks.keys()), palette(len(tracks)))}
    verts, edges = [], []
    for cid in sorted(tracks.keys()):
        samples = [s for s in tracks[cid] if s["ok"] == 1]
        if len(samples) < 2:
            continue
        cols = color_map[cid]
        angs = np.linspace(0.0, 2.0*math.pi, num=n_angles, endpoint=False)
        base_idx = []
        for s in samples:
            base = len(verts); base_idx.append(base)
            cx, cy, r, h = s["x"], s["y"], s["r"], s["h"]
            for th in angs:
                x = cx + r*math.cos(th)
                y = cy + r*math.sin(th)
                z = h
                verts.append([x,y,z, int(cols[0]), int(cols[1]), int(cols[2])])
            for ai in range(n_angles):
                v1 = base + ai
                v2 = base + ((ai+1) % n_angles)
                edges.append([v1, v2])
        for i in range(len(base_idx)-1):
            b1, b2 = base_idx[i], base_idx[i+1]
            for ai in range(n_angles):
                edges.append([b1 + ai, b2 + ai])
    if not verts: return None, None
    return np.asarray(verts, dtype=np.float32), np.asarray(edges, dtype=np.int32)

def build_circle_rings_only(tracks: dict, n_angles: int = 64):
    color_map = {cid: col for cid, col in zip(sorted(tracks.keys()), palette(len(tracks)))}
    verts, edges = [], []
    for cid in sorted(tracks.keys()):
        samples = [s for s in tracks[cid] if s["ok"] == 1]
        if not samples:
            continue
        cols = color_map[cid]
        angs = np.linspace(0.0, 2.0*math.pi, num=n_angles, endpoint=False)
        for s in samples:
            base = len(verts)
            cx, cy, r, h = s["x"], s["y"], s["r"], s["h"]
            for th in angs:
                x = cx + r*math.cos(th)
                y = cy + r*math.sin(th)
                z = h
                verts.append([x,y,z, int(cols[0]), int(cols[1]), int(cols[2])])
            for ai in range(n_angles):
                v1 = base + ai
                v2 = base + ((ai+1) % n_angles)
                edges.append([v1, v2])
    if not verts: return None, None
    return np.asarray(verts, dtype=np.float32), np.asarray(edges, dtype=np.int32)

def build_branch_markers(
    tracks: dict,
    branch_h: dict,
    centers_hint: dict,
    cross_len: float = 0.40,   # min full cross length (m) – bigger default
    overhang: float = 0.10     # extend beyond local radius so tips stick out
):
    """
    High-contrast branch markers:
      - bold cross that extends past the stem
      - small ring around the cross
      - short vertical 'pin' so it's visible from any angle
    All vertices are bright white (good contrast on red/green clouds).
    """
    color = np.array([245, 245, 245], dtype=np.uint8)  # bright white
    ring_scale = 1.15   # ring radius = ring_scale * local stem radius
    ring_angles = 32
    pin_len = 0.20      # total length of the vertical pin (m)

    verts, edges = [], []

    for cid, h in branch_h.items():
        samples = tracks.get(cid, [])
        if not samples:
            continue

        # center at branch height (hint -> interp -> nearest)
        if cid in centers_hint:
            cx, cy = centers_hint[cid]
        else:
            fx = interp_at_h(samples, "x")
            fy = interp_at_h(samples, "y")
            cx, cy = fx(h), fy(h)
            if not (np.isfinite(cx) and np.isfinite(cy)):
                nearest = min(samples, key=lambda s: abs(s["h"] - h))
                cx, cy = nearest["x"], nearest["y"]

        # local radius at branch height (interp -> nearest)
        fr = interp_at_h(samples, "r")
        r_at_h = fr(h)
        if not (np.isfinite(r_at_h) and r_at_h > 0):
            r_at_h = min(samples, key=lambda s: abs(s["h"] - h))["r"]

        # half-length so cross tips stick out
        L = max(r_at_h + overhang, cross_len * 0.5)

        # --- CROSS (two edges) ---
        pts = [(cx - L, cy, h), (cx + L, cy, h), (cx, cy - L, h), (cx, cy + L, h)]
        base = len(verts)
        for (x, y, z) in pts:
            verts.append([x, y, z, int(color[0]), int(color[1]), int(color[2])])
        edges.append([base + 0, base + 1])
        edges.append([base + 2, base + 3])

        # --- RING (small ring around the cross) ---
        rr = max(0.05, r_at_h * ring_scale)
        angs = np.linspace(0.0, 2.0*math.pi, num=ring_angles, endpoint=False)
        rbase = len(verts)
        for th in angs:
            x = cx + rr * math.cos(th)
            y = cy + rr * math.sin(th)
            verts.append([x, y, h, int(color[0]), int(color[1]), int(color[2])])
        for ai in range(ring_angles):
            edges.append([rbase + ai, rbase + ((ai + 1) % ring_angles)])

        # --- PIN (short vertical segment through center) ---
        dz = pin_len * 0.5
        pbase = len(verts)
        verts.append([cx, cy, h - dz, int(color[0]), int(color[1]), int(color[2])])
        verts.append([cx, cy, h + dz, int(color[0]), int(color[1]), int(color[2])])
        edges.append([pbase + 0, pbase + 1])

    if not verts:
        return None, None
    return np.asarray(verts, dtype=np.float32), np.asarray(edges, dtype=np.int32)


def build_dbh_annotations(tracks: dict, stems_summary: dict, n_angles: int = 64):
    """Build circle + cross annotations at breast height (BH) for each stem."""
    gold, white = np.array([255, 215, 0]), np.array([245, 245, 245])
    verts, edges = [], []
    angs = np.linspace(0.0, math.tau, num=n_angles, endpoint=False)

    for cid, info in stems_summary.items():
        dbh, bh = info.get("dbh_m"), info.get("bh_m", 1.3)
        if dbh is None or not np.isfinite(dbh) or dbh <= 0:
            continue
        r = dbh / 2.0
        cx, cy = info.get("cx"), info.get("cy")

        # If no center, interpolate or take nearest
        if not (np.isfinite(cx) and np.isfinite(cy)):
            samples = tracks.get(cid, [])
            if not samples:
                continue
            fx, fy = interp_at_h(samples, "x"), interp_at_h(samples, "y")  # noqa
            cx, cy = fx(bh), fy(bh)
            if not (np.isfinite(cx) and np.isfinite(cy)):
                n = min(samples, key=lambda s: abs(s["h"] - bh))
                cx, cy = n["x"], n["y"]

        base = len(verts)
        for th in angs:
            x, y = cx + r * math.cos(th), cy + r * math.sin(th)
            verts.append([x, y, bh, *gold])
        for i in range(n_angles):
            edges.append([base + i, base + (i + 1) % n_angles])

        # Cross
        cross = [(cx - r, cy, bh), (cx + r, cy, bh), (cx, cy - r, bh), (cx, cy + r, bh)]
        cbase = len(verts)
        for (x, y, z) in cross:
            verts.append([x, y, z, *white])
        edges += [[cbase, cbase + 1], [cbase + 2, cbase + 3]]

    if not verts:
        return None, None
    return np.array(verts, np.float32), np.array(edges, np.int32)

# -------- NEW: bucking annotations builder --------
def build_bucking_annotations(tracks: dict, bucking: dict, n_angles: int = 64):
    """
    For each cut:
      - POINT: draw one ring at Z=h using radius from track at h
      - INTERVAL: draw two rings at Z=h1 and Z=h2
    Colors:
      heuristic -> cyan   (0, 200, 255)
      advanced  -> magenta(255, 50, 255)
    """
    COLORS = {
        "heuristic": np.array([0, 200, 255], dtype=np.uint8),
        "advanced":  np.array([255, 50, 255], dtype=np.uint8),
    }
    angs = np.linspace(0.0, 2.0*math.pi, num=n_angles, endpoint=False)
    verts, edges = [], []

    for cid, cuts in bucking.items():
        samples = tracks.get(cid, [])
        if not samples:
            continue
        fx = interp_at_h(samples, "x")
        fy = interp_at_h(samples, "y")
        fr = interp_at_h(samples, "r")
        r_fallback = float(np.median([s["r"] for s in samples])) if samples else 0.15

        for c in cuts:
            method = c["method"]
            col = COLORS.get(method, np.array([200,200,200], dtype=np.uint8))

            def ring_at(hh: float):
                cx, cy = fx(hh), fy(hh)
                rr = fr(hh)
                if not np.isfinite(rr) or rr <= 0:
                    rr = r_fallback
                if not (np.isfinite(cx) and np.isfinite(cy)):
                    near = min(samples, key=lambda s: abs(s["h"] - hh))
                    cx, cy, rr = near["x"], near["y"], near["r"]
                base = len(verts)
                for th in angs:
                    x = cx + rr*math.cos(th)
                    y = cy + rr*math.sin(th)
                    verts.append([x,y,hh, int(col[0]), int(col[1]), int(col[2])])
                for ai in range(n_angles):
                    edges.append([base + ai, base + ((ai+1) % n_angles)])

            if c["type"] == "point":
                ring_at(c["h"])
            elif c["type"] == "interval":
                ring_at(c["h1"]); ring_at(c["h2"])

    if not verts:
        return None, None
    return np.asarray(verts, dtype=np.float32), np.asarray(edges, dtype=np.int32)

# --------------- CLI ---------------
def main():
    ap = argparse.ArgumentParser(description="Export wire-mesh, rings-only, branch, DBH, and bucking overlay PLYs.")
    ap.add_argument("--merged-dir", required=True, help="Folder with refined_per_slice_circles.csv")
    ap.add_argument("--out-dir",    required=True, help="Output folder for overlays")
    ap.add_argument("--refined-csv", default="", help="Override path to refined CSV")
    ap.add_argument("--branch-csv",  default="", help="Path to branch/fork survey CSV (optional)")
    ap.add_argument("--angles", type=int, default=48, help="Angles for wire mesh circles")
    ap.add_argument("--ring-angles", type=int, default=64, help="Angles for rings-only, DBH & bucking rings")
    ap.add_argument("--stems-summary", default="", help="Override path to stems_summary.csv for DBH")
    # NEW
    ap.add_argument("--bucking-csv", default="", help="CSV with bucking cut annotations (heuristic/advanced). Greedy is ignored.")
    ap.add_argument("--branch-overhang", type=float, default=0.05,
                    help="How far branch cross should extend beyond local stem radius (m).")
    ap.add_argument("--cross-len", type=float, default=0.25,
                    help="Minimum full length (tip-to-tip) for branch cross (m).")

    args = ap.parse_args()

    merged_dir = Path(args.merged_dir)
    out_dir    = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    refined_csv = Path(args.refined_csv) if args.refined_csv else (merged_dir / "refined_per_slice_circles.csv")
    if not refined_csv.exists():
        raise SystemExit(f"Missing refined CSV: {refined_csv}")
    tracks = read_refined_tracks(refined_csv)

    # 1) Wire mesh
    v_wire, e_wire = build_wire_mesh(tracks, n_angles=args.angles)
    if v_wire is not None:
        p = out_dir / "wire_mesh_overlays.ply"
        write_ply_vertices_edges(p, v_wire, e_wire)
        print(f"[wire ] {p}  (V={v_wire.shape[0]}, E={e_wire.shape[0]})")
    else:
        print("[wire ] no valid tracks.")

    # 2) Branch markers
    branch_csv = Path(args.branch_csv) if args.branch_csv else (merged_dir / "forks_refined_per_slice_circles.csv")
    branch_h, centers = read_branch_heights(branch_csv)
    if branch_h:
        v_b, e_b = build_branch_markers(
            tracks, branch_h, centers,
            cross_len=args.cross_len,
            overhang=args.branch_overhang
        )
        if v_b is not None:
            p = out_dir / "branch_markers.ply"
            write_ply_vertices_edges(p, v_b, e_b)
            print(f"[branch] {p}  (V={v_b.shape[0]}, E={e_b.shape[0]})")
    else:
        print(f"[branch] no branch heights found in {branch_csv}; skipped.")

    # 3) Rings only
    v_ro, e_ro = build_circle_rings_only(tracks, n_angles=args.ring_angles)
    if v_ro is not None:
        p = out_dir / "circle_rings_only.ply"
        write_ply_vertices_edges(p, v_ro, e_ro)
        print(f"[rings] {p}  (V={v_ro.shape[0]}, E={e_ro.shape[0]})")

    # 4) DBH annotations
    stems_summary_csv = Path(args.stems_summary) if args.stems_summary else (merged_dir / "stems_summary.csv")
    stems = read_stems_summary(stems_summary_csv)
    if stems:
        v_dbh, e_dbh = build_dbh_annotations(tracks, stems, n_angles=args.ring_angles)
        if v_dbh is not None:
            p = out_dir / "dbh_annotations.ply"
            write_ply_vertices_edges(p, v_dbh, e_dbh)
            print(f"[dbh  ] {p}  (V={v_dbh.shape[0]}, E={e_dbh.shape[0]})")
    else:
        print(f"[dbh  ] no usable DBH found in {stems_summary_csv}; skipped.")

    # 5) Bucking annotations
    bucking_csv = Path(args.bucking_csv) if args.bucking_csv else None
    bucking = read_bucking_csv(bucking_csv, merged_dir) if bucking_csv else {}
    if bucking:
        v_buck, e_buck = build_bucking_annotations(tracks, bucking, n_angles=args.ring_angles)
        if v_buck is not None:
            p = out_dir / "bucking_annotations.ply"
            write_ply_vertices_edges(p, v_buck, e_buck)
            print(f"[buck ] {p}  (V={v_buck.shape[0]}, E={e_buck.shape[0]})")
    else:
        print("[buck ] no bucking CSV provided or no usable rows; skipped.")

if __name__ == "__main__":
    main()

