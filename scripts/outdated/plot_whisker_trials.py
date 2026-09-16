#!/usr/bin/env python3
r"""
Plot a whisker (LEFT = Data_Inverted, RIGHT = Data) around a chosen RANGE of
pulse-pal intervals in three ways, each saved as its own SVG into the mirrored
results folder:

  1. continuous  : the whole whisker trace on a real time axis (ms), with every
                   selected interval shaded.
  2. offset      : each interval extracted as a trial window
                   [-pre, interval, +post], aligned so poke onset = 0 ms, drawn
                   as stacked/offset line traces (first trial at bottom).
  3. heatmap     : the same trial windows as an aligned matrix, whisker angle
                   encoded by color (diverging map centered at 0), with a
                   colorbar in degrees.

Side selection:
  --side left   -> reads the Data_Inverted column (left whisker, inverted so it
                   shares a protraction/retraction sign with the right whisker).
  --side right  -> reads the raw Data column.

Interval range:
  --intervals START END selects pulse-pal intervals START..END (1-based,
  inclusive), e.g. "--intervals 1 16" or "--intervals 5 20". The first selected
  interval's onset is t = 0 in the continuous plot; every trial is aligned to
  its own poke onset in the offset/heatmap plots.

Windowing: each trial spans [-pre_ms, interval_end + post_ms], aligned to the
interval (poke) ONSET at t = 0. Intervals here are 500 ms; the default window is
500 ms before, the 500 ms interval, and 500 ms after.

Output: <RESULTS_ROOT>\<session sub-path>\<name>.svg  (folders auto-created).

Usage
-----
    uv run plot_whisker_trials.py
    uv run plot_whisker_trials.py --side left --intervals 1 16
    uv run plot_whisker_trials.py --side left --intervals 17 38
    uv run plot_whisker_trials.py --side left  --intervals 5 20 --pre-ms 500 --post-ms 500

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


def load_whisker(data_dir: Path, side: str) -> pd.DataFrame:
    """Return ['frame','angle'] for the given side.
    left  -> Data_Inverted column (falls back to negating Data).
    right -> raw Data column.
    """
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
    else:  # right -> raw Data
        if not raw:
            raise ValueError(f"{csv_path.name}: no Data column")
        angle = pd.to_numeric(df[raw[0]], errors="coerce"); which = raw[0]

    print(f"  [{side}] file: {csv_path.name}  angle column: {which}")
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
    """Return cam_rise_s (per-frame times, s), sample_rate, the selected
    pulse-pal (start_s, end_s) tuples for intervals first..last (1-based,
    inclusive), and the total number of intervals available."""
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
    """Resample each trial window onto a common time grid (ms, poke onset = 0)."""
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


# ---------- plots ----------
def plot_continuous(frame_s, angle, intervals_s, t0_s, out_path, title,
                    color, label):
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
    ax.set_ylim(-90, 90)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    lo = (intervals_s[0][0] - t0_s) * 1000 - 500
    hi = (intervals_s[-1][1] - t0_s) * 1000 + 500
    ax.set_xlim(lo, hi)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    print(f"Saved: {out_path}")


def plot_offset(grid, trials, dur_ms, out_path, title, color, labels,
                spacing=None):
    n = trials.shape[0]
    finite = trials[np.isfinite(trials)]
    if spacing is None:
        p2p = np.nanpercentile(finite, 97.5) - np.nanpercentile(finite, 2.5)
        spacing = max(p2p * 1.2, 1.0)
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
    ax.set_ylabel("Trial (interval #)")
    ax.set_xlabel("Time from poke onset (ms)")
    ax.set_title(title)
    ax.plot([grid[-1] + 20, grid[-1] + 20], [0, spacing], color="k", lw=2,
            clip_on=False)
    ax.text(grid[-1] + 40, spacing / 2, f"{spacing:.0f}\u00b0 (1 trial step)",
            rotation=90, va="center", fontsize=8)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    print(f"Saved: {out_path}")


def plot_heatmap(grid, trials, dur_ms, out_path, title, labels, clim=None):
    n = trials.shape[0]
    if clim is None:
        v = np.nanpercentile(np.abs(trials), 98)
        clim = (-v, v)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(trials, aspect="auto", origin="lower",
                   extent=[grid[0], grid[-1], 0.5, n + 0.5],
                   cmap="RdBu_r", vmin=clim[0], vmax=clim[1],
                   interpolation="nearest")
    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.7)
    ax.axvline(dur_ms, color="k", lw=0.8, ls="--", alpha=0.4)
    ax.set_xlabel("Time from poke onset (ms)")
    ax.set_ylabel("Trial (interval #)")
    ax.set_yticks(range(1, n + 1))
    ax.set_yticklabels([str(x) for x in labels])
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Whisker angle (deg)")
    # annotate direction at the ends of the colorbar
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
                    help="Which whisker to plot: left (Data_Inverted) or right "
                         "(Data). Default left.")
    ap.add_argument("--intervals", type=int, nargs=2, metavar=("START", "END"),
                    default=[1, 16],
                    help="Pulse-pal interval range to plot, 1-based inclusive "
                         "(e.g. --intervals 5 20). Default: 1 16.")
    ap.add_argument("--pre-ms", type=float, default=500.0,
                    help="Milliseconds before each poke onset (default 500).")
    ap.add_argument("--post-ms", type=float, default=500.0,
                    help="Milliseconds after each interval end (default 500).")
    args = ap.parse_args()

    data_dir, raw_dir = Path(args.data_dir), Path(args.raw_dir)
    side = args.side
    color = SIDE_COLOR[side]
    side_label = f"{side.capitalize()} whisker"
    inv_note = " (inverted)" if side == "left" else ""

    print(f"Loading {side.upper()} whisker from {data_dir}")
    w = load_whisker(data_dir, side)
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
    t0_s = intervals_s[0][0]  # first selected poke onset = 0 (continuous plot)

    grid, trials, dur_ms = extract_trials(
        frame_s, angle, intervals_s, args.pre_ms, args.post_ms,
        dt_ms=1000.0 / 500.0)  # 500 fps
    print(f"  trial window: [-{args.pre_ms:.0f}, {dur_ms + args.post_ms:.0f}] ms, "
          f"{trials.shape[1]} timepoints x {trials.shape[0]} trials")

    # per-trial labels are the actual 1-based interval numbers in the range
    labels = list(range(lo, hi + 1))

    parts = raw_dir.resolve().parts
    sess = " / ".join(parts[-3:]) if len(parts) >= 3 else raw_dir.name
    sess_tag = "_".join(parts[-3:])
    rng = f"int{lo}-{hi}"

    plot_continuous(
        frame_s, angle, intervals_s, t0_s,
        results_path(raw_dir, Path(args.results_root),
                     f"{side}whisker_continuous_{sess_tag}_{rng}.svg"),
        f"{side_label}{inv_note} \u2014 continuous, intervals {lo}\u2013{hi}\n{sess}",
        color, side_label)

    plot_offset(
        grid, trials, dur_ms,
        results_path(raw_dir, Path(args.results_root),
                     f"{side}whisker_offset_{sess_tag}_{rng}.svg"),
        f"{side_label}{inv_note} \u2014 intervals {lo}\u2013{hi} stacked\n{sess}",
        color, labels)

    plot_heatmap(
        grid, trials, dur_ms,
        results_path(raw_dir, Path(args.results_root),
                     f"{side}whisker_heatmap_{sess_tag}_{rng}.svg"),
        f"{side_label}{inv_note} \u2014 intervals {lo}\u2013{hi} (color = angle)\n{sess}",
        labels)


if __name__ == "__main__":
    main()