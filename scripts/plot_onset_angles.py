#!/usr/bin/env python3
r"""
Collect the whisker angle at POKE ONSET (t = 0) for each pulse-pal interval in a
chosen range, store them in an array, and plot them.

For each selected interval the script samples the whisker angle at t = 0 (the
interval/poke onset) and builds an array with one value per trial. Two versions
are computed:
  * absolute onset angle (deg)
  * baseline-subtracted onset angle = onset minus the pre-poke mean over
    [-pre_ms, 0)  ->  "how far from rest the whisker was at the moment of poke".
--metric selects which is plotted (default baseline; the other is raw).

The figure has two panels:
  left  : onset angle vs. trial (interval #), as points + connecting line.
  right : the distribution of onset angles (horizontal box + jittered points),
          with the mean marked.
Both share the y-axis (angle), and a fixed +/- --clim is used when metric =
baseline so plots are comparable across sides / videos / animals.

Same arguments as plot_whisker_trials.py: --side (left=Data_Inverted,
right=Data) and --intervals START END (1-based inclusive). Glitch frames are
despiked (|angle| > --clip-deg -> NaN) before sampling.

Output: <RESULTS_ROOT>\<session sub-path>\<name>.svg  (folders auto-created);
the raw array is also written next to it as a .csv.

Usage
-----
    uv run plot_onset_angles.py --side left --intervals 17 38
    uv run plot_onset_angles.py --side right --intervals 1 16 --metric raw

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

SIDE_COLOR = {"left": "#d62728", "right": "#1f77b4"}
DEFAULT_CLIM = 50.0
DEFAULT_CLIP_DEG = 90.0


# ---------- CSV loading (shared with plot_whisker_trials.py) ----------
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


# ---------- onset sampling ----------
def onset_angles(frame_s, angle, intervals_s, pre_ms, onset_win_ms=10.0):
    """For each interval, sample the angle at t=0 (poke onset) and the pre-poke
    baseline. Returns arrays (onset_abs, onset_baseline_sub), one value per
    interval. The onset value is the mean angle in [0, onset_win_ms] to be robust
    to a single missing frame; baseline is the median over [-pre_ms, 0)."""
    onset_abs = np.full(len(intervals_s), np.nan)
    onset_bs = np.full(len(intervals_s), np.nan)
    for k, (s0, _e0) in enumerate(intervals_s):
        rel_ms = (frame_s - s0) * 1000.0
        onset_m = (rel_ms >= 0) & (rel_ms <= onset_win_ms)
        base_m = (rel_ms >= -pre_ms) & (rel_ms < 0)
        if onset_m.any():
            o = np.nanmean(angle[onset_m])
            onset_abs[k] = o
            if base_m.any():
                onset_bs[k] = o - np.nanmedian(angle[base_m])
    return onset_abs, onset_bs


# ---------- plot ----------
def plot_onset(labels, vals, out_path, title, color, ylabel, ylim):
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [3, 1]}, sharey=True)

    # left: onset angle vs trial
    axL.spines["top"].set_visible(False)
    axL.spines["right"].set_visible(False)
    axL.axhline(0, color="0.6", lw=0.8, ls="--")
    axL.plot(labels, vals, "-", color=color, lw=1, alpha=0.5, zorder=1)
    axL.plot(labels, vals, "o", color=color, ms=6, zorder=2)
    axL.set_xlabel("Trial (interval #)")
    axL.set_ylabel(ylabel)
    axL.set_title(title, loc="left")
    if ylim is not None:
        axL.set_ylim(*ylim)
    axL.set_xticks(labels)
    axL.tick_params(axis="x", labelrotation=90, labelsize=7)

    # right: distribution
    axR.spines["top"].set_visible(False)
    axR.spines["right"].set_visible(False)
    axR.spines["left"].set_visible(False)
    axR.axhline(0, color="0.6", lw=0.8, ls="--")
    finite = vals[np.isfinite(vals)]
    bp = axR.boxplot(finite, vert=True, widths=0.5, showfliers=False,
                     patch_artist=True, medianprops=dict(color="k"))
    for b in bp["boxes"]:
        b.set(facecolor=color, alpha=0.25)
    jitter = (np.random.default_rng(0).random(finite.size) - 0.5) * 0.25 + 1
    axR.plot(jitter, finite, "o", color=color, ms=5, alpha=0.7)
    med = np.median(finite)
    axR.plot([0.7, 1.3], [med] * 2, color="k", lw=2)
    axR.text(1.35, med, f" median {med:.1f}\u00b0",
             va="center", fontsize=8)
    axR.set_xticks([1]); axR.set_xticklabels(["all trials"])
    axR.set_title(f"n={finite.size}", loc="left", fontsize=9)

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
    ap.add_argument("--metric", choices=["baseline", "raw"],
                    default="baseline",
                    help="Onset angle relative to pre-poke rest (baseline) or "
                         "raw (absolute angle). Default baseline.")
    ap.add_argument("--pre-ms", type=float, default=500.0,
                    help="Pre-poke baseline window length (default 500).")
    ap.add_argument("--onset-win-ms", type=float, default=10.0,
                    help="Average the onset angle over [0, this] ms (default 10).")
    ap.add_argument("--clim", type=float, default=DEFAULT_CLIM,
                    help=f"Fixed +/- y-limit (deg) for the baseline metric. "
                         f"Default {DEFAULT_CLIM:.0f}.")
    ap.add_argument("--ylim", type=float, default=90.0,
                    help="Fixed +/- y-limit (deg) for the absolute metric. "
                         "Default 90.")
    ap.add_argument("--clip-deg", type=float, default=DEFAULT_CLIP_DEG,
                    help=f"Despike: |angle| beyond this -> NaN. Default {DEFAULT_CLIP_DEG:.0f}.")
    args = ap.parse_args()

    data_dir, raw_dir = Path(args.data_dir), Path(args.raw_dir)
    side = args.side
    color = SIDE_COLOR[side]
    side_label = f"{side.capitalize()} whisker"
    inv_note = " (inverted)" if side == "left" else ""

    print(f"Loading {side.upper()} whisker from {data_dir}")
    w = load_whisker(data_dir, side, args.clip_deg)

    h5_path = find_h5(raw_dir)
    print(f"Reading pulse-pal timing from {h5_path.name}")
    lo_req, hi_req = args.intervals
    cam_rise_s, sr, intervals_s, n_total, (lo, hi) = read_timing(
        h5_path, lo_req, hi_req)
    print(f"  {n_total} intervals available; sampling onsets for intervals {lo}..{hi}")

    frame_s = frame_time_s(w["frame"].values, cam_rise_s)
    angle = w["angle"].values
    onset_abs, onset_bs = onset_angles(
        frame_s, angle, intervals_s, args.pre_ms, args.onset_win_ms)

    labels = list(range(lo, hi + 1))
    if args.metric == "baseline":
        vals = onset_bs
        ylabel = "Onset angle \u0394 from baseline (deg)"
        ylim = (-args.clim, args.clim)
        metric_title = "baseline-subtracted"
    else:  # raw
        vals = onset_abs
        ylabel = "Onset angle (deg)"
        ylim = (-args.ylim, args.ylim)
        metric_title = "raw"

    print(f"  onset angles ({args.metric}): "
          f"mean {np.nanmean(vals):.1f}, sd {np.nanstd(vals):.1f}, "
          f"n={np.isfinite(vals).sum()}")

    parts = raw_dir.resolve().parts
    sess = " / ".join(parts[-3:]) if len(parts) >= 3 else raw_dir.name
    sess_tag = "_".join(parts[-3:])
    rng = f"int{lo}-{hi}"

    # write the array as CSV alongside the figure
    arr_path = results_path(raw_dir, Path(args.results_root),
                            f"{side}whisker_onsetangles_{args.metric}_{sess_tag}_{rng}.csv")
    pd.DataFrame({"interval": labels,
                  "onset_absolute_deg": onset_abs,
                  "onset_baseline_sub_deg": onset_bs}).to_csv(arr_path, index=False)
    print(f"Saved array: {arr_path}")

    svg_path = results_path(raw_dir, Path(args.results_root),
                            f"{side}whisker_onsetangles_{args.metric}_{sess_tag}_{rng}.svg")
    plot_onset(
        labels, vals, svg_path,
        f"{side_label}{inv_note} \u2014 angle at poke onset ({metric_title}), "
        f"intervals {lo}\u2013{hi}\n{sess}",
        color, ylabel, ylim)


if __name__ == "__main__":
    main()
