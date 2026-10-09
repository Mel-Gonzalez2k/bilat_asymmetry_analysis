#!/usr/bin/env python3
r"""
Group analysis: control vs experimental whisker angle, averaged ACROSS MICE.

Statistics are hierarchical (the correct way for a group comparison):
  1. each mouse -> ONE trace (default: the whole 1 s-baseline + 5x-stim + 1 s-post
     block, aligned to the first pulse onset at t = 0; or, with --per-pulse, the
     mean of the 5 single-pulse responses within the mouse);
  2. across mice within a group -> mean, and error bars computed over MICE
     (SEM = sd / sqrt(n_mice), or SD). n = number of mice, never number of stims,
     so there is no pseudoreplication.

For each whisker it makes the requested plots, in BOTH baseline-subtracted and
raw-angle form:
      left  whisker : mean +/- SEM   and   mean +/- SD   (control vs experimental)
      right whisker : mean +/- SEM   and   mean +/- SD
-> 8 figures total (4 baseline-subtracted, 4 raw), plus group/per-mouse CSVs.

Side convention (screen -> mouse, same as your overlay scripts):
  left  whisker  <-  Angle_*Right*.csv          (raw)
  right whisker  <-  Angle_*Left*reverse*.csv    (raw; must contain 'reverse')

INPUT: a manifest listing each mouse's SESSION folder and its group.
  manifest.csv (one mouse per line; '#' comments and blank lines ignored):
      group,path
      experimental,E:\...\MRN_opto12\20260517\1_at_rest_first_stim_100ms_1s
      control,E:\...\<ctrl mouse session folder>
      ...
  'group' may be: experimental / exp / e   or   control / ctrl / c.

Usage
-----
    # 1) find candidate session folders to paste into the manifest:
    uv run group_whisker_plots.py --discover "E:\bilat_asymmetry_analysis\data\Optogenetic_Activation"

    # 2) run the analysis:
    uv run group_whisker_plots.py --manifest manifest.csv --out-dir "E:\...\group_out"
    uv run group_whisker_plots.py --manifest manifest.csv --per-pulse   # per-pulse mode

Dependencies: h5py, numpy, pandas, matplotlib.
"""
# /// script
# requires-python = ">=3.9"
# dependencies = ["h5py", "numpy", "pandas", "matplotlib"]
# ///

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAM_BIT, PULSE_BIT, FPS = 0, 1, 500.0
EXP_COLOR, CTRL_COLOR = "#e63946", "#1d3557"   # experimental = red, control = navy
STIM_COLOR, STIM_ALPHA = "#00bcd4", 0.10
GROUPS = ("experimental", "control")


# ----------------------------- file discovery -------------------------------
def norm_group(s: str):
    s = s.strip().lower()
    if s in ("experimental", "exp", "e", "stim", "mrn"):
        return "experimental"
    if s in ("control", "ctrl", "ctl", "c"):
        return "control"
    return None


def skip_path(folder) -> bool:
    s = str(folder).lower()
    return ("outdated" in s) or ("not_in_use" in s) or ("not in use" in s)


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


def find_angle_csv(folder: Path, mouse_side: str):
    """mouse_side 'left' -> screen Right file; 'right' -> screen Left+reverse file.
    Uses RAW files (skips Angle_Filter_*, *filt*, *filter*)."""
    screen = "right" if mouse_side == "left" else "left"
    angle = [Path(p) for p in glob.glob(str(folder / "*.csv"))
             if Path(p).name.lower().startswith("angle")]
    out = []
    for p in angle:
        n = p.name.lower()
        if "filt" in n or "filter" in n:          # RAW only
            continue
        if screen not in n:
            continue
        if screen == "right" and "left" in n:      # exclude the other side
            continue
        if mouse_side == "right" and "reverse" not in n:
            continue
        out.append(p)
    return sorted(out)[0] if out else None


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


# ----------------------------- per-mouse trace ------------------------------
def interp_window(abs_s, ang, t0, grid):
    rel = abs_s - t0
    m = (rel >= grid[0] - 0.02) & (rel <= grid[-1] + 0.02)
    if m.sum() < 2:
        return np.full(len(grid), np.nan)
    return np.interp(grid, rel[m], ang[m], left=np.nan, right=np.nan)


def baseline_sub(trace, grid, pre_s):
    b = (grid >= -pre_s) & (grid < 0)
    if b.any():
        return trace - np.nanmean(trace[b])
    return trace


def smooth(trace, win):
    """NaN-aware centered rolling mean for display (win in samples)."""
    if win and win > 1:
        return (pd.Series(trace)
                .rolling(win, center=True, min_periods=max(1, win // 3))
                .mean().to_numpy())
    return trace


def process_mouse(folder: Path, grid, pre_s, per_pulse, smooth_win=0):
    """Return dict: left/right traces (raw & baseline-subtracted) on `grid`,
    plus pulse timing, for one mouse. None if files are missing."""
    h5 = find_acq_h5(folder)
    lcsv = find_angle_csv(folder, "left")    # mouse LEFT  <- Angle_*Right*
    rcsv = find_angle_csv(folder, "right")   # mouse RIGHT <- Angle_*Left*reverse*
    miss = [n for n, v in (("acq .h5", h5), ("left(Angle_Right)", lcsv),
                            ("right(Angle_Left_reverse)", rcsv)) if v is None]
    if miss:
        print(f"    [skip] {folder.name}: missing {miss}")
        return None
    cam_rise_s, ps, pe = read_h5(h5)
    if len(ps) == 0:
        print(f"    [skip] {folder.name}: no pulses in h5")
        return None
    lf, la = load_angle(lcsv)
    rf, ra = load_angle(rcsv)
    la_abs, ra_abs = frame_to_abs_s(lf, cam_rise_s), frame_to_abs_s(rf, cam_rise_s)

    rel_on = ps - ps[0]
    rel_off = pe - ps[0]
    if per_pulse:
        # mean of the single-pulse responses within this mouse
        Ls, Rs = [], []
        for t0 in ps:
            Ls.append(interp_window(la_abs, la, t0, grid))
            Rs.append(interp_window(ra_abs, ra, t0, grid))
        left = np.nanmean(np.vstack(Ls), 0)
        right = np.nanmean(np.vstack(Rs), 0)
    else:
        # whole block, aligned to first pulse onset
        left = interp_window(la_abs, la, ps[0], grid)
        right = interp_window(ra_abs, ra, ps[0], grid)

    left_bs = baseline_sub(left, grid, pre_s)
    right_bs = baseline_sub(right, grid, pre_s)
    if smooth_win and smooth_win > 1:
        left, right = smooth(left, smooth_win), smooth(right, smooth_win)
        left_bs, right_bs = smooth(left_bs, smooth_win), smooth(right_bs, smooth_win)
    return dict(
        left_raw=left, right_raw=right, left_bs=left_bs, right_bs=right_bs,
        n_pulses=len(ps), rel_on=rel_on, rel_off=rel_off, name=folder.name,
    )


# ----------------------------- aggregation / plot ---------------------------
def group_stat(mat):
    """mat: (n_mice, n_grid) -> (mean, sem, sd, n_per_col)."""
    mean = np.nanmean(mat, 0)
    sd = np.nanstd(mat, 0, ddof=1) if mat.shape[0] > 1 else np.full(mat.shape[1], np.nan)
    n = np.sum(np.isfinite(mat), 0)
    sem = sd / np.sqrt(np.where(n > 0, n, np.nan))
    return mean, sem, sd, n


def plot_metric(grid, by_group, metric, whisker, mode, stim_bands, out_dir, ns,
                group_offset=0.0, align="block"):
    """metric 'SEM'|'SD'; whisker 'left'|'right'; mode 'baseline'|'raw';
    align 'block' (whole 5x block) | 'perpulse' (avg of the 5 single pulses)."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for (lo, hi) in stim_bands:
        ax.axvspan(lo, hi, color=STIM_COLOR, alpha=STIM_ALPHA, lw=0)
    ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.4)
    # experimental shifted up by group_offset (0 = overlaid, the default)
    shifts = {"experimental": group_offset, "control": 0.0}
    for grp, color in (("experimental", EXP_COLOR), ("control", CTRL_COLOR)):
        mat = by_group.get(grp)
        if mat is None or mat.shape[0] == 0:
            continue
        mean, sem, sd, _ = group_stat(mat)
        err = sem if metric == "SEM" else sd
        off = shifts[grp]
        lab = f"{grp} (n={ns[grp]})"
        if group_offset:
            ax.axhline(off, color=color, lw=0.7, ls=":", alpha=0.5)
        ax.fill_between(grid, mean - err + off, mean + err + off,
                        color=color, alpha=0.30, lw=0, zorder=1)
        ax.plot(grid, mean - err + off, color=color, lw=0.6, alpha=0.6, zorder=2)
        ax.plot(grid, mean + err + off, color=color, lw=0.6, alpha=0.6, zorder=2)
        ax.plot(grid, mean + off, color=color, lw=1.6, label=lab, zorder=3)
    if not group_offset:
        ax.axhline(0, color="k", lw=0.6, ls=":", alpha=0.4)
    ax.set_xlabel("Time from first stim onset (s)")
    ylab = f"{whisker.capitalize()} whisker angle (deg)"
    ylab += "  (baseline-subtracted)" if mode == "baseline" else "  (raw)"
    ax.set_ylabel(ylab)
    ax.set_xlim(grid[0], grid[-1])
    align_lbl = "per-pulse" if align == "perpulse" else "whole block"
    ax.set_title(f"{whisker.capitalize()} whisker \u2014 control vs experimental "
                 f"(mean \u00b1 {metric}, across mice; {align_lbl})")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.12)
    fig.tight_layout()
    stem = f"group_{whisker}_whisker_{metric}_{mode}_{align}"
    for ext in ("png", "svg"):
        p = out_dir / f"{stem}.{ext}"
        try:
            fig.savefig(p, dpi=150)
        except (PermissionError, OSError):
            for i in range(1, 100):
                try:
                    fig.savefig(out_dir / f"{stem}_{i}.{ext}"); break
                except (PermissionError, OSError):
                    continue
    plt.close(fig)
    print(f"  wrote {stem}.png/.svg")


# ----------------------------- manifest / discover --------------------------
def read_manifest(path: Path):
    rows = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "," not in line:
            continue
        g, p = line.split(",", 1)
        grp = norm_group(g)
        if grp is None:
            if norm_group(p.split(",")[0]) is not None:   # columns swapped
                p, g = g, p.split(",")[0]
                grp = norm_group(g)
        if grp is None:
            print(f"  [manifest] skipping line (unknown group): {line}")
            continue
        rows.append((grp, p.strip().strip('"')))
    return rows


def mouse_id(path) -> int:
    m = re.search(r"opto[_ ]?(\d+)", str(path), re.I)
    return int(m.group(1)) if m else -1


def build_rows_auto(parent: Path, exp_ids, ctrl_ids, session_match):
    """Auto-assign group from the opto<N> number in each folder's path."""
    want = {**{i: "experimental" for i in exp_ids},
            **{i: "control" for i in ctrl_ids}}
    by_id = {}
    for root, _dirs, _files in os.walk(parent):
        folder = Path(root)
        if skip_path(folder):
            continue
        if session_match and session_match.lower() not in str(folder).lower():
            continue
        if not (find_angle_csv(folder, "left") and find_angle_csv(folder, "right")
                and find_acq_h5(folder)):
            continue
        mid = mouse_id(folder)
        if mid in want:
            by_id.setdefault(mid, []).append(folder)

    rows, multi = [], {}
    for mid, grp in want.items():
        fs = sorted(set(by_id.get(mid, [])))
        if len(fs) == 0:
            print(f"  [auto] MRN_opto{mid} ({grp}): NO folder with "
                  f"Angle_Right + Angle_Left_reverse + acquisition .h5 found")
        elif len(fs) > 1:
            multi[mid] = fs
        else:
            rows.append((grp, str(fs[0])))
    if multi:
        print("\n[auto] several candidate folders for some mice. Narrow with "
              "--session-match <substring> (e.g. 100ms_1s), or list them in a "
              "--manifest. Candidates:")
        for mid, fs in multi.items():
            print(f"  MRN_opto{mid} ({want[mid]}):")
            for f in fs:
                print(f"     {f}")
        raise SystemExit("resolve the ambiguous mice above, then re-run.")
    return rows


def discover(parent: Path, session_match="100ms", exp_ids=(), ctrl_ids=()):
    print(f"Scanning {parent} for session folders with the needed files"
          + (f" (path must contain '{session_match}')" if session_match else "") + "...\n")
    want = {**{i: "experimental" for i in exp_ids}, **{i: "control" for i in ctrl_ids}}
    hits = []
    for root, _dirs, _files in os.walk(parent):
        folder = Path(root)
        if skip_path(folder):
            continue
        if session_match and session_match.lower() not in str(folder).lower():
            continue
        if (find_angle_csv(folder, "left") and find_angle_csv(folder, "right")
                and find_acq_h5(folder)):
            hits.append(folder)
    if not hits:
        print("  (none found)")
        return
    print("Folders found, with the group each will get in --parent mode:\n")
    used = 0
    for h in sorted(hits):
        grp = want.get(mouse_id(h))
        if grp:
            print(f"  [{grp:12s}] {h}"); used += 1
        else:
            print(f"  [IGNORED      ] {h}   (opto{mouse_id(h)} not in exp/ctrl lists)")
    print(f"\n{used} folder(s) will be used ({len(hits)} matched the file+100ms filter).")


# --------------------------------- main -------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", help="CSV of 'group,path' lines (one mouse each)")
    ap.add_argument("--parent", help="auto mode: scan this parent dir and assign "
                    "group from the opto<N> number in each path")
    ap.add_argument("--exp", default="7,12,13,14,15",
                    help="experimental opto IDs (comma-sep), for --parent mode")
    ap.add_argument("--ctrl", default="5,6,8",
                    help="control opto IDs (comma-sep), for --parent mode")
    ap.add_argument("--session-match", default="100ms",
                    help="only use folders whose path contains this substring "
                         "(default '100ms' = this stimulation duration; set to '' "
                         "for no filter, or e.g. '100ms_1s' to narrow further)")
    ap.add_argument("--discover", help="scan this parent dir and list candidate folders")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--pre-ms", type=float, default=1000.0, help="baseline window (ms)")
    ap.add_argument("--post-ms", type=float, default=1000.0,
                    help="how long after the LAST pulse to keep (ms)")
    ap.add_argument("--per-pulse", action="store_true",
                    help="average the 5 single-pulse responses within mouse instead "
                         "of using the whole block")
    ap.add_argument("--smooth-ms", type=float, default=25.0,
                    help="display smoothing of each trace (ms rolling mean) so the "
                         "SEM/SD band is visible under the 500 fps jitter; 0 = off")
    ap.add_argument("--group-offset", type=float, default=0.0,
                    help="vertical gap (deg) to lift the experimental trace above "
                         "control. Default 0 = overlaid (best for comparing size).")
    args = ap.parse_args()

    if args.discover:
        discover(Path(args.discover), args.session_match,
                 [int(x) for x in args.exp.split(",") if x.strip()],
                 [int(x) for x in args.ctrl.split(",") if x.strip()])
        return

    if args.parent:
        exp_ids = [int(x) for x in args.exp.split(",") if x.strip()]
        ctrl_ids = [int(x) for x in args.ctrl.split(",") if x.strip()]
        print(f"auto mode: experimental opto {exp_ids}, control opto {ctrl_ids}")
        rows = build_rows_auto(Path(args.parent), exp_ids, ctrl_ids, args.session_match)
    elif args.manifest:
        rows = read_manifest(Path(args.manifest))
    else:
        raise SystemExit("give --parent PARENT_DIR  (or --manifest manifest.csv, "
                         "or --discover PARENT_DIR)")
    if not rows:
        raise SystemExit("no usable mouse folders found")

    # a provisional grid; widen 'post' after we know the block length
    pre_s = args.pre_ms / 1000.0
    dt = 1.0 / FPS
    # first pass just to learn block durations
    block_ends = []
    for _grp, p in rows:
        h5 = find_acq_h5(Path(p))
        if h5 is None:
            continue
        try:
            _c, ps, pe = read_h5(h5)
            if len(ps):
                block_ends.append(float(pe[-1] - ps[0]))
        except Exception:
            pass
    block_s = float(np.median(block_ends)) if block_ends else 4.5
    end_s = (0.0 if args.per_pulse else block_s) + args.post_ms / 1000.0
    grid = np.arange(-pre_s, end_s + dt, dt)
    print(f"window: -{pre_s:.1f}s .. +{end_s:.2f}s  (median block {block_s:.2f}s, "
          f"{'per-pulse' if args.per_pulse else 'whole-block'} mode)")

    # process every mouse
    data = {g: {k: [] for k in ("left_raw", "right_raw", "left_bs", "right_bs")}
            for g in GROUPS}
    rel_ons, rel_offs = [], []
    per_mouse_rows = []
    for grp, p in rows:
        folder = Path(p)
        if not folder.is_dir():
            print(f"    [skip] not a folder: {p}"); continue
        res = process_mouse(folder, grid, pre_s, args.per_pulse,
                             smooth_win=int(round(args.smooth_ms / 1000.0 * FPS)))
        if res is None:
            continue
        for k in ("left_raw", "right_raw", "left_bs", "right_bs"):
            data[grp][k].append(res[k])
        rel_ons.append(res["rel_on"][:5]); rel_offs.append(res["rel_off"][:5])
        per_mouse_rows.append((grp, res["name"], res["n_pulses"]))
        print(f"    [{grp}] {res['name']}: {res['n_pulses']} pulses")

    ns = {g: len(data[g]["left_raw"]) for g in GROUPS}
    print(f"\nmice used -> experimental: {ns['experimental']}, control: {ns['control']}")
    if ns["experimental"] == 0 and ns["control"] == 0:
        raise SystemExit("no mice processed; check the manifest paths")

    # stim bands (median pulse windows across mice), only for whole-block mode
    stim_bands = []
    if not args.per_pulse and rel_ons:
        on = np.nanmedian(np.vstack([r for r in rel_ons if len(r)]), 0)
        off = np.nanmedian(np.vstack([r for r in rel_offs if len(r)]), 0)
        stim_bands = list(zip(on, off))
    elif args.per_pulse:
        stim_bands = [(0.0, 0.1)]

    align_tag = "perpulse" if args.per_pulse else "block"
    out_dir = Path(args.out_dir) if args.out_dir else Path(rows[0][1]).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # build matrices and plot all 8
    def mat(grp, key):
        lst = data[grp][key]
        return np.vstack(lst) if lst else np.empty((0, len(grid)))

    for mode, suffix in (("baseline", "_bs"), ("raw", "_raw")):
        for whisker in ("left", "right"):
            by_group = {g: mat(g, f"{whisker}{suffix}") for g in GROUPS}
            for metric in ("SEM", "SD"):
                plot_metric(grid, by_group, metric, whisker, mode, stim_bands,
                            out_dir, ns, group_offset=args.group_offset,
                            align=align_tag)

    # save the group curves + per-mouse list
    rows_out = {"time_s": grid}
    for mode, suffix in (("baseline", "_bs"), ("raw", "_raw")):
        for whisker in ("left", "right"):
            for g in GROUPS:
                m = mat(g, f"{whisker}{suffix}")
                if m.shape[0] == 0:
                    continue
                mean, sem, sd, nn = group_stat(m)
                tag = f"{whisker}_{mode}_{g}"
                rows_out[f"{tag}_mean"] = mean
                rows_out[f"{tag}_sem"] = sem
                rows_out[f"{tag}_sd"] = sd
    pd.DataFrame(rows_out).to_csv(out_dir / f"group_curves_{align_tag}.csv", index=False)
    pd.DataFrame(per_mouse_rows, columns=["group", "folder", "n_pulses"]).to_csv(
        out_dir / f"mice_used_{align_tag}.csv", index=False)
    print(f"\nwrote group_curves.csv and mice_used.csv to {out_dir}\nDone.")


if __name__ == "__main__":
    main()
