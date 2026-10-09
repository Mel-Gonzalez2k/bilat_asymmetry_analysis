"""Shared helpers for group_heatmaps.py and group_latency.py (bilat asymmetry).

Discovers each mouse's 100ms session folder under a parent, assigns group from
the opto<N> id, and extracts per-PULSE whisker-angle windows aligned to each
pulse onset, baseline-subtracted, on a common time grid.

Side convention (screen -> mouse):
  left  whisker  <-  Angle_*Right*.csv
  right whisker  <-  Angle_*Left*reverse*.csv   (must contain 'reverse')
Prefers FILTERED files (filt/filter), else falls back to raw.
"""
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

CAM_BIT, PULSE_BIT, FPS = 0, 1, 500.0
FILT_TOKENS = ("filt", "filter")
GROUPS = ("experimental", "control")


def skip_path(folder) -> bool:
    s = str(folder).lower()
    return ("outdated" in s) or ("not_in_use" in s) or ("not in use" in s)


def mouse_id(path) -> int:
    m = re.search(r"opto[_ ]?(\d+)", str(path), re.I)
    return int(m.group(1)) if m else -1


def find_acq_h5(folder: Path):
    cands = []
    for p in glob.glob(str(folder / "*.h5")):
        n = Path(p).name.lower()
        if any(t in n for t in ("line_masks", "dlc", "resnet", "shuffle")):
            continue
        cands.append(Path(p))
    runlike = [p for p in cands if re.search(r"_\d{4}\.h5$", p.name)]
    cc = sorted(runlike) or sorted(cands)
    return cc[0] if cc else None


def _is_filtered(n): return any(t in n for t in FILT_TOKENS)


def find_angle_csv(folder: Path, mouse_side: str, use_filtered=False):
    """RAW (non-filtered) Angle files by default; pass use_filtered=True to use
    the Angle_Filt/Filter versions. Returns None if the requested kind is absent."""
    screen = "right" if mouse_side == "left" else "left"
    angle = [Path(p) for p in glob.glob(str(folder / "*.csv"))
             if Path(p).name.lower().startswith("angle")]
    pool = [p for p in angle
            if screen in p.name.lower()
            and (screen != "right" or "left" not in p.name.lower())]
    if mouse_side == "right":
        pool = [p for p in pool if "reverse" in p.name.lower()]
    if not pool:
        return None
    filt = [p for p in pool if _is_filtered(p.name.lower())]
    raw = [p for p in pool if not _is_filtered(p.name.lower())]
    chosen = filt if use_filtered else raw
    return sorted(chosen)[0] if chosen else None


def sweep_key(h5_name: str) -> str:
    m = re.search(r"_(\d{4})\.h5$", h5_name)
    if m:
        return f"sweep_{m.group(1)}/digitalScans"
    g = re.findall(r"\d{4}", h5_name)
    return f"sweep_{g[-1] if g else '0001'}/digitalScans"


def read_h5(h5_path: Path):
    import h5py
    key = sweep_key(h5_path.name)
    with h5py.File(h5_path, "r") as f:
        if key not in f:
            raise KeyError(f"'{key}' not in {h5_path.name}; keys={list(f.keys())}")
        try:
            sr = float(f["header/AcquisitionSampleRate"][()].ravel()[0])
        except Exception:
            sr = 250000.0
        raw = np.asarray(f[key][()]).ravel()
    cam = ((raw >> CAM_BIT) & 1).astype(np.int8)
    pul = ((raw >> PULSE_BIT) & 1).astype(np.int8)
    cam_rise_s = (np.where(np.diff(cam) == 1)[0] + 1) / sr
    d = np.diff(pul)
    ps = (np.where(d == 1)[0] + 1) / sr
    pe = (np.where(d == -1)[0] + 1) / sr
    n = min(len(ps), len(pe))
    return cam_rise_s, ps[:n], pe[:n]


def load_angle(csv_path: Path):
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    tcol = "Time" if "Time" in cols else cols[0]
    acol = "Data" if "Data" in cols else ("Angle" if "Angle" in cols
                                          else next(c for c in cols if c != tcol))
    frame = pd.to_numeric(df[tcol], errors="coerce").to_numpy(float)
    ang = pd.to_numeric(df[acol], errors="coerce").to_numpy(float)
    m = np.isfinite(frame) & np.isfinite(ang)
    frame, ang = frame[m], ang[m]
    order = np.argsort(frame)
    return frame[order], ang[order]


def frame_to_abs_s(frames, cam_rise_s):
    idx = np.clip(frames, 0, len(cam_rise_s) - 1).astype(int)
    out = cam_rise_s[idx].astype(float)
    over = frames > (len(cam_rise_s) - 1)
    if np.any(over):
        out[over] = cam_rise_s[-1] + (frames[over] - (len(cam_rise_s) - 1)) / FPS
    return out


def per_pulse_windows(abs_s, ang, onsets, grid, base_ms):
    """Return (n_pulse, n_grid) of angle per pulse, baseline-subtracted over
    [-base_ms, 0] (None = no subtraction)."""
    out = np.full((len(onsets), len(grid)), np.nan)
    for k, t0 in enumerate(onsets):
        rel = abs_s - t0
        m = (rel >= grid[0] - 0.02) & (rel <= grid[-1] + 0.02)
        if m.sum() < 2:
            continue
        out[k] = np.interp(grid, rel[m], ang[m], left=np.nan, right=np.nan)
        if base_ms is not None:
            b = (grid >= -base_ms / 1000.0) & (grid < 0)
            if b.any():
                out[k] -= np.nanmean(out[k][b])
    return out


def coverage_keep(W, grid, width_s, min_cover):
    """Bool mask of pulses that have angle data BEFORE, DURING and AFTER the pulse.
    A pulse is kept only if each of the three sub-windows has at least
    `min_cover` fraction of finite (non-NaN) samples."""
    before = grid < 0
    during = (grid >= 0) & (grid <= width_s)
    after = grid > width_s
    keep = np.zeros(W.shape[0], bool)
    for r in range(W.shape[0]):
        f = np.isfinite(W[r])
        cb = f[before].mean() if before.any() else 1.0
        cd = f[during].mean() if during.any() else 1.0
        ca = f[after].mean() if after.any() else 1.0
        keep[r] = min(cb, cd, ca) >= min_cover
    return keep


def discover_folders(parent, exp_ids, ctrl_ids, session_match, use_filtered=False):
    want = {**{i: "experimental" for i in exp_ids},
            **{i: "control" for i in ctrl_ids}}
    by_id = {}
    for root, _dirs, _files in os.walk(parent):
        folder = Path(root)
        if skip_path(folder):
            continue
        if session_match and session_match.lower() not in str(folder).lower():
            continue
        if not (find_angle_csv(folder, "left", use_filtered)
                and find_angle_csv(folder, "right", use_filtered)
                and find_acq_h5(folder)):
            continue
        mid = mouse_id(folder)
        if mid in want:
            by_id.setdefault(mid, []).append(folder)
    rows = []
    for mid, grp in want.items():
        fs = sorted(set(by_id.get(mid, [])))
        if len(fs) == 1:
            rows.append((grp, fs[0], mid))
        elif len(fs) == 0:
            print(f"  [!] MRN_opto{mid} ({grp}): no matching folder")
        else:
            print(f"  [!] MRN_opto{mid} ({grp}): {len(fs)} folders; using {fs[0].name}")
            rows.append((grp, fs[0], mid))
    return rows


def collect(parent, exp_ids, ctrl_ids, session_match, grid, base_ms, min_cover=0.5,
            use_filtered=False):
    """data[group][side] = list of (mouse_id, trials[n_pulse, n_grid]), keeping
    ONLY pulses with angle data before+during+after (coverage >= min_cover).
    Uses RAW angle files unless use_filtered=True.
    Also returns median pulse width (ms) and n mice per group."""
    print(f"  angle source: {'FILTERED (Angle_Filt/Filter)' if use_filtered else 'RAW (non-filtered)'}")
    rows = discover_folders(parent, exp_ids, ctrl_ids, session_match, use_filtered)
    data = {g: {"left": [], "right": []} for g in GROUPS}
    widths = []
    for grp, folder, mid in rows:
        h5 = find_acq_h5(folder)
        cam_rise_s, ps, pe = read_h5(h5)
        if len(ps) == 0:
            print(f"    [skip] opto{mid}: no pulses"); continue
        width_s = float(np.median(pe - ps))
        widths.append(width_s * 1000.0)
        kept = {}
        for side in ("left", "right"):
            f, a = load_angle(find_angle_csv(folder, side, use_filtered))
            W = per_pulse_windows(frame_to_abs_s(f, cam_rise_s), a, ps, grid, base_ms)
            keep = coverage_keep(W, grid, width_s, min_cover)
            data[grp][side].append((mid, W[keep]))
            kept[side] = int(keep.sum())
        print(f"    [{grp}] opto{mid}: {len(ps)} pulses in h5 → kept with full "
              f"coverage: left {kept['left']}, right {kept['right']}")
    width_ms = float(np.median(widths)) if widths else 100.0
    # a mouse counts for a group only if it has >=1 covered pulse (either side)
    ns = {}
    for g in GROUPS:
        mids = set(m for m, W in data[g]["left"] if W.shape[0] > 0)
        mids |= set(m for m, W in data[g]["right"] if W.shape[0] > 0)
        ns[g] = len(mids)
    print(f"\nmice with covered pulses -> experimental: {ns['experimental']}, "
          f"control: {ns['control']}")
    return data, width_ms, ns
