#!/usr/bin/env python3
r"""
Plot a whisker (LEFT = Data_Inverted, RIGHT = Data) around a chosen RANGE of
pulse-pal intervals, producing FOUR SVGs into the mirrored results folder:

  1. continuous            : whole trace on a real time axis (ms), absolute
                             angle (resting posture visible), fixed y-limits.
  2. offset_raw            : per-trial stacked traces, ABSOLUTE angle (resting
                             posture visible), first selected trial at bottom.
  3. offset_baseline       : per-trial stacked traces, BASELINE-SUBTRACTED
                             (each trial minus its pre-poke mean), fixed scale.
  4. heatmap_baseline      : aligned matrix, BASELINE-SUBTRACTED change encoded
                             by color on a FIXED diverging scale (+/- CLIM).

Why baseline subtraction + fixed limits: to compare plots across whiskers,
pokes, videos and animals, every figure must use the SAME degrees->color and
degrees->deflection mapping, and must be insensitive to each whisker's
arbitrary resting angle. Baseline subtraction (change from the pre-poke mean)
removes the resting offset; a fixed CLIM makes a given color/deflection mean the
same angle change everywhere. Glitch frames are despiked first (clipped to NaN
beyond CLIP_DEG) so a handful of tracking outliers can't distort the scale.

Choosing CLIM scientifically: run the script; it prints the 99th percentile of
|baseline-subtracted change| (a robust coverage-based limit, unaffected by
outliers unlike a standard deviation). Pool that across representative
videos/animals, round it, and set --clim once so every plot shares it. For this
dataset the pooled 99th percentile is ~54 deg -> default CLIM = 50.

Side selection:
  --side left   -> Data_Inverted column (left whisker inverted to share sign).
  --side right  -> raw Data column.

Interval range:
  --intervals START END  (1-based, inclusive), e.g. --intervals 17 38.

Windowing: each trial spans [-pre_ms, interval_end + post_ms], aligned to poke
ONSET (t = 0). Baseline window = pre-poke [-pre_ms, 0).

Output: <RESULTS_ROOT>\<session sub-path>\<name>.svg  (folders auto-created).

Usage
-----
    uv run plot_whisker_trials_norm.py
    uv run plot_whisker_trials_norm.py --side right --intervals 17 38
    uv run plot_whisker_trials_norm.py --side left --intervals 1 16 --clim 50

Dependencies: h5py, numpy, pandas, matplotlib.
"""
# /// script
# requires-python = ">=3.9"
# dependencies = ["h5py", "numpy", "pandas", "matplotlib"]
# ///

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_DATA = r"E:\bilat_asymmetry_analysis\data\TeLC_Silencing\WA023\20260413\Post_Day7"
DEFAULT_RAW = r"E:\bilat_asymmetry_analysis\raw\TeLC_Silencing\WA023\20260413\Post_Day7"
RESULTS_ROOT = r"E:\bilat_asymmetry_analysis\results"

DATA_KEY = "sweep_0001/digitalScans"
CAM_BIT = 0
PULSE_BIT = 1

SIDE_COLOR = {"left": "#d62728", "right": "#1f77b4"}  # red / blue

# Fixed scale for baseline-subtracted plots (degrees of change). Calibrated from
# the pooled 99th percentile of |change| across both whiskers (~54 deg -> 50).
DEFAULT_CLIM = 50.0
# Despike: samples with |angle| beyond this are set to NaN before anything else.
DEFAULT_CLIP_DEG = 90.0


# ---------- CSV loading ----------
def _pick_csv(folder: Path) -> Path:
    csvs = [Path(p) for p in sorted(glob.glob(str(folder / "*.csv")))]
    if not csvs:
        raise FileNotFoundError(f"no .csv found in {folder}")

    def header_ok(p):
        try:
            cols = list(pd.read_csv(p, nrows=0).columns)
        except Exception:
            return False
        return any(str(c) == "Time" for c in cols) and \
               any(str(c).startswith("Data") for c in cols)

    valid = [p for p in csvs if header_ok(p)]
    if not valid:
        raise ValueError(f"no CSV in {folder} has Time+Data columns; found "
                         f"{[p.name for p in csvs]}")
    filt = [p for p in valid if "angle_filt" in p.name.lower()]
    ang = [p for p in valid if "angle" in p.name.lower()]
    chosen = (filt or ang or valid)[0]
    if len(valid) > 1:
        print(f"  [note] {folder.name}: {len(csvs)} CSVs present, using {chosen.name}")
    return chosen


def load_whisker(data_dir: Path, side: str, clip_deg: float) -> pd.DataFrame:
    """Return ['frame','angle'] for the given side, despiked.
    left -> Data_Inverted (or negated Data); right -> raw Data.
    Samples with |angle| > clip_deg are set to NaN (tracking glitches)."""
    folder = data_dir / side
    csv_path = _pick_csv(folder)
    df = pd.read_csv(csv_path)
    time_col = "Time" if "Time" in df.columns else df.columns[0]
    inv = [c for c in df.columns if str(c) in ("Data_Inverted", "Data.1")]
    raw = [c for c in df.columns if str(c) == "Data"]

    if side == "left":
        if inv:
            angle = pd.to_numeric(df[inv[0]], errors="coerce"); which = inv[0]
        elif raw:
            angle = -pd.to_numeric(df[raw[0]], errors="coerce")
            which = f"-{raw[0]} (negated fallback)"
        else:
            raise ValueError(f"{csv_path.name}: no Data/Data_Inverted column")
    else:
        if not raw:
            raise ValueError(f"{csv_path.name}: no Data column")
        angle = pd.to_numeric(df[raw[0]], errors="coerce"); which = raw[0]

    angle = angle.copy()
    n_spike = int((angle.abs() > clip_deg).sum())
    angle[angle.abs() > clip_deg] = np.nan
    print(f"  [{side}] file: {csv_path.name}  column: {which}  "
          f"despiked {n_spike} samples > |{clip_deg:.0f}|deg")
    out = pd.DataFrame({"frame": pd.to_numeric(df[time_col], errors="coerce"),
                        "angle": angle}).dropna()
    if out.empty:
        raise ValueError(f"{csv_path.name}: no usable rows after parsing")
    return out


# ---------- h5 timing ----------
def find_h5(raw_dir: Path) -> Path:
    h5s = sorted(glob.glob(str(raw_dir / "*.h5")))
    if not h5s:
        raise FileNotFoundError(f"no .h5 found in {raw_dir}")
    if len(h5s) > 1:
        print(f"  [note] {len(h5s)} .h5 files, using {Path(h5s[0]).name}")
    return Path(h5s[0])


def read_timing(h5_path: Path, first_1based: int, last_1based: int):
    import h5py
    with h5py.File(h5_path, "r") as f:
        sr = float(f["header/AcquisitionSampleRate"][()].ravel()[0])
        raw = f[DATA_KEY][()]
    raw = np.asarray(raw).ravel()
    cam = ((raw >> CAM_BIT) & 1).astype(np.int8)
    pulse = ((raw >> PULSE_BIT) & 1).astype(np.int8)
    cam_rise_s = (np.where(np.diff(cam) == 1)[0] + 1) / sr
    d = np.diff(pulse)
    starts = (np.where(d == 1)[0] + 1) / sr
    ends = (np.where(d == -1)[0] + 1) / sr
    n_total = min(len(starts), len(ends))
    lo, hi = first_1based, last_1based
    if lo < 1 or hi < lo:
        raise ValueError(f"bad interval range {lo}..{hi}")
    if lo > n_total:
        raise IndexError(f"range starts at {lo} but only {n_total} intervals exist")
    hi = min(hi, n_total)
    sel = list(zip(starts[lo - 1:hi], ends[lo - 1:hi]))
    return cam_rise_s, sr, sel, n_total, (lo, hi)


def frame_time_s(frames, cam_rise_s):
    idx = np.clip(np.asarray(frames, dtype=int), 0, len(cam_rise_s) - 1)
    return cam_rise_s[idx]


# ---------- results path ----------
def results_path(src_dir: Path, results_root: Path, filename: str) -> Path:
    parts = src_dir.resolve().parts
    sub = None
    for anchor in ("data", "raw"):
        idxs = [i for i, p in enumerate(parts) if p.lower() == anchor]
        if idxs:
            sub = Path(*parts[idxs[-1] + 1:]); break
    if sub is None:
        sub = Path(src_dir.name)
    out_dir = Path(results_root) / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


# ---------- trial extraction ----------
def extract_trials(frame_s, angle, intervals_s, pre_ms, post_ms, dt_ms=2.0):
    """Resample each trial onto a common ms grid (poke onset = 0). Returns
    (grid, trials_raw, dur_ms). trials_raw is absolute angle; NaN where missing."""
    dur_ms = np.median([(e - s) for s, e in intervals_s]) * 1000.0
    t_lo, t_hi = -pre_ms, dur_ms + post_ms
    grid = np.arange(t_lo, t_hi + dt_ms, dt_ms)
    trials = np.full((len(intervals_s), len(grid)), np.nan)
    for k, (s0, _e0) in enumerate(intervals_s):
        rel_ms = (frame_s - s0) * 1000.0
        m = (rel_ms >= t_lo - dt_ms) & (rel_ms <= t_hi + dt_ms)
        if m.sum() < 2:
            continue
        trials[k] = np.interp(grid, rel_ms[m], angle[m], left=np.nan, right=np.nan)
    return grid, trials, dur_ms


def baseline_subtract(grid, trials, pre_ms):
    """Subtract each trial's mean over the pre-poke window [-pre_ms, 0)."""
    base_mask = (grid >= -pre_ms) & (grid < 0)
    base = np.nanmedian(trials[:, base_mask], axis=1, keepdims=True)   # was np.nanmean
    #base = np.nanmean(trials[:, base_mask], axis=1, keepdims=True)
    return trials - base


def report_robust_limit(trials_bs, tag):
    ch = np.abs(trials_bs[np.isfinite(trials_bs)])
    if ch.size:
        p99 = np.nanpercentile(ch, 99)
        print(f"  [{tag}] robust scale suggestion: 99th pct |change| = "
              f"{p99:.1f} deg (pool across videos, round, set --clim)")


# ---------- plots ----------
def plot_continuous(frame_s, angle, intervals_s, t0_s, out_path, title,
                    color, label, ylim):
    ms = (frame_s - t0_s) * 1000.0
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.plot(ms, angle, lw=0.6, color=color, label=label)
    for j, (s0, e0) in enumerate(intervals_s):
        ax.axvspan((s0 - t0_s) * 1000, (e0 - t0_s) * 1000, color="0.5",
                   alpha=0.25, label="Cotton Swab Poke" if j == 0 else None)
    ax.set_xlabel("Time (ms, 0 = first poke onset)")
    ax.set_ylabel("Whisker angle (deg)")
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    lo = (intervals_s[0][0] - t0_s) * 1000 - 500
    hi = (intervals_s[-1][1] - t0_s) * 1000 + 500
    ax.set_xlim(lo, hi)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    print(f"Saved: {out_path}")


def plot_offset(grid, trials, dur_ms, out_path, title, color, labels,
                spacing, ylabel_note):
    n = trials.shape[0]
    fig, ax = plt.subplots(figsize=(9, 12))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvspan(0, dur_ms, color="0.5", alpha=0.18, zorder=0,
               label="Cotton Swab Poke")
    ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.5)
    for k in range(n):
        ax.plot(grid, trials[k] + k * spacing, lw=0.7, color=color)
    ax.set_yticks([k * spacing for k in range(n)])
    ax.set_yticklabels([str(x) for x in labels])
    ax.set_ylabel(f"Trial (interval #){ylabel_note}")
    ax.set_xlabel("Time from poke onset (ms)")
    ax.set_title(title)
    ax.plot([grid[-1] + 20, grid[-1] + 20], [0, spacing], color="k", lw=2,
            clip_on=False)
    ax.text(grid[-1] + 40, spacing / 2, f"{spacing:.0f}\u00b0 (1 trial step)",
            rotation=90, va="center", fontsize=8)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    print(f"Saved: {out_path}")


def plot_heatmap(grid, trials, dur_ms, out_path, title, labels, clim):
    n = trials.shape[0]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(trials, aspect="auto", origin="lower",
                   extent=[grid[0], grid[-1], 0.5, n + 0.5],
                   cmap="RdBu_r", vmin=-clim, vmax=clim,
                   interpolation="nearest")
    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.7)
    ax.axvline(dur_ms, color="k", lw=0.8, ls="--", alpha=0.4)
    ax.set_xlabel("Time from poke onset (ms)")
    ax.set_ylabel("Trial (interval #)")
    ax.set_yticks(range(1, n + 1))
    ax.set_yticklabels([str(x) for x in labels])
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Whisker angle change from baseline (deg)")
    cb.ax.text(0.5, 1.02, "Protraction (+)", transform=cb.ax.transAxes,
               ha="left", va="bottom", fontsize=12)
    cb.ax.text(0.5, -0.02, "Retraction (\u2212)", transform=cb.ax.transAxes,
               ha="left", va="top", fontsize=12)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--raw-dir", default=DEFAULT_RAW)
    ap.add_argument("--results-root", default=RESULTS_ROOT)
    ap.add_argument("--side", choices=["left", "right"], default="left",
                    help="left (Data_Inverted) or right (Data). Default left.")
    ap.add_argument("--intervals", type=int, nargs=2, metavar=("START", "END"),
                    default=[1, 16],
                    help="Pulse-pal interval range, 1-based inclusive. Default 1 16.")
    ap.add_argument("--pre-ms", type=float, default=500.0,
                    help="ms before poke onset; also the baseline window. Default 500.")
    ap.add_argument("--post-ms", type=float, default=500.0,
                    help="ms after interval end. Default 500.")
    ap.add_argument("--clim", type=float, default=DEFAULT_CLIM,
                    help=f"Fixed +/- scale (deg) for baseline-subtracted heatmap "
                         f"and offset. Use the same value across all plots/animals. "
                         f"Default {DEFAULT_CLIM:.0f}.")
    ap.add_argument("--clip-deg", type=float, default=DEFAULT_CLIP_DEG,
                    help=f"Despike: |angle| beyond this -> NaN. Default {DEFAULT_CLIP_DEG:.0f}.")
    ap.add_argument("--ylim", type=float, default=90.0,
                    help="Absolute-angle y-limit (deg) for continuous & raw offset. "
                         "Default 90.")
    args = ap.parse_args()

    data_dir, raw_dir = Path(args.data_dir), Path(args.raw_dir)
    side = args.side
    color = SIDE_COLOR[side]
    side_label = f"{side.capitalize()} whisker"
    inv_note = " (inverted)" if side == "left" else ""

    print(f"Loading {side.upper()} whisker from {data_dir}")
    w = load_whisker(data_dir, side, args.clip_deg)
    print(f"  {len(w):,} frames ({w['frame'].min():.0f}-{w['frame'].max():.0f})")

    h5_path = find_h5(raw_dir)
    print(f"Reading pulse-pal timing from {h5_path.name}")
    lo_req, hi_req = args.intervals
    cam_rise_s, sr, intervals_s, n_total, (lo, hi) = read_timing(
        h5_path, lo_req, hi_req)
    print(f"  {n_total} intervals available; plotting intervals {lo}..{hi} "
          f"({len(intervals_s)} trials)")

    frame_s = frame_time_s(w["frame"].values, cam_rise_s)
    angle = w["angle"].values
    t0_s = intervals_s[0][0]

    grid, trials_raw, dur_ms = extract_trials(
        frame_s, angle, intervals_s, args.pre_ms, args.post_ms,
        dt_ms=1000.0 / 500.0)
    trials_bs = baseline_subtract(grid, trials_raw, args.pre_ms)
    print(f"  trial window: [-{args.pre_ms:.0f}, {dur_ms + args.post_ms:.0f}] ms, "
          f"{trials_raw.shape[1]} timepoints x {trials_raw.shape[0]} trials")
    report_robust_limit(trials_bs, side)
    print(f"  using fixed clim = +/-{args.clim:.0f} deg for baseline-subtracted plots")

    labels = list(range(lo, hi + 1))
    parts = raw_dir.resolve().parts
    sess = " / ".join(parts[-3:]) if len(parts) >= 3 else raw_dir.name
    sess_tag = "_".join(parts[-3:])
    rng = f"int{lo}-{hi}"

    def rp(name):
        return results_path(raw_dir, Path(args.results_root),
                            f"{side}whisker_{name}_{sess_tag}_{rng}.svg")

    # 1) continuous, absolute angle
    plot_continuous(
        frame_s, angle, intervals_s, t0_s, rp("continuous"),
        f"{side_label}{inv_note} \u2014 continuous, intervals {lo}\u2013{hi}\n{sess}",
        color, side_label, ylim=(-args.ylim, args.ylim))

    # 2) offset, absolute angle (resting posture). Fixed spacing = 2*ylim so the
    #    absolute scale is comparable across plots.
    plot_offset(
        grid, trials_raw, dur_ms, rp("offset_raw"),
        f"{side_label}{inv_note} \u2014 intervals {lo}\u2013{hi}, absolute angle\n{sess}",
        color, labels, spacing=2 * args.ylim, ylabel_note="  (absolute angle)")

    # 3) offset, baseline-subtracted. Fixed spacing = 2*clim.
    plot_offset(
        grid, trials_bs, dur_ms, rp("offset_baseline"),
        f"{side_label}{inv_note} \u2014 intervals {lo}\u2013{hi}, baseline-subtracted\n{sess}",
        color, labels, spacing=2 * args.clim, ylabel_note="  (\u0394 from baseline)")

    # 4) heatmap, baseline-subtracted, fixed clim
    plot_heatmap(
        grid, trials_bs, dur_ms, rp("heatmap_baseline"),
        f"{side_label}{inv_note} \u2014 intervals {lo}\u2013{hi}, baseline-subtracted\n{sess}",
        labels, clim=args.clim)


if __name__ == "__main__":
    main()
