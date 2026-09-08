#!/usr/bin/env python3
r"""
Plot left and right whisker angles (x-axis = time in ms, 0 = start of the first
shown pulse-pal interval) with selected pulse-pal stimulation intervals drawn as
shaded regions.

Data sources
------------
Whisker angle CSVs (x = frame number in a "Time" column, y = angle in a "Data"
or "Data_Inverted" column):
    <DATA_DIR>\left\*.csv     (left whisker)
    <DATA_DIR>\right\*.csv    (right whisker)

Digital TTLs (WaveSurfer .h5, bit-packed at sweep_0001/digitalScans):
    bit 0 = cam        -> one rising edge per camera frame
    bit 1 = pulse_pal  -> high intervals = stimulation pulses
The .h5 lives directly in <RAW_DIR>.

Whisker traces are indexed in frames; each frame is mapped to a time using the
camera rising-edge times from the .h5, then shifted so the first shown pulse-pal
interval starts at 0 ms.

Output
------
By default the figure is written as an SVG into the results tree, mirroring the
session sub-path after "data\" or "raw\":
    <RESULTS_ROOT>\TeLC_Silencing\WA023\20260413\Post_Day7\<name>.svg
Missing folders are created automatically. Use --save to override.

Usage
-----
    uv run plot_whisker_pulsepal.py
    uv run plot_whisker_pulsepal.py --intervals 3 4
    uv run plot_whisker_pulsepal.py --data-dir "E:\...\data\...\Post_Day7" \
                                    --raw-dir  "E:\...\raw\...\Post_Day7"

Dependencies: h5py, numpy, pandas, matplotlib (installed by uv via the block).
"""
# /// script
# requires-python = ">=3.9"
# dependencies = ["h5py", "numpy", "pandas", "matplotlib"]
# ///

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe default; comment out to show interactively
import matplotlib.pyplot as plt

# Defaults point at the WA023 Post_Day7 session described in the request.
DEFAULT_DATA = r"E:\bilat_asymmetry_analysis\data\TeLC_Silencing\WA023\20260413\Post_Day7"
DEFAULT_RAW = r"E:\bilat_asymmetry_analysis\raw\TeLC_Silencing\WA023\20260413\Post_Day7"
# Plots are written here, mirroring the sub-path after "data\" or "raw\".
RESULTS_ROOT = r"E:\bilat_asymmetry_analysis\results"

DATA_KEY = "sweep_0001/digitalScans"
CAM_BIT = 0
PULSE_BIT = 1


def _pick_csv(folder: Path) -> Path:
    """Choose the right angle CSV in a folder that may contain several.

    Preference order:
      1. files whose name contains 'angle_filt' (the filtered angle trace),
      2. then files whose name contains 'angle',
      3. then any CSV whose header has a 'Time' column and a 'Data' column.
    Raster/coordinate files like 'left_whisker2.csv' (header 'Frame,X,Y') are
    skipped because they lack Time/Data columns.
    """
    csvs = [Path(p) for p in sorted(glob.glob(str(folder / "*.csv")))]
    if not csvs:
        raise FileNotFoundError(f"no .csv found in {folder}")

    def header_ok(p):
        try:
            cols = list(pd.read_csv(p, nrows=0).columns)
        except Exception:
            return False
        has_time = any(str(c) == "Time" for c in cols)
        has_data = any(str(c).startswith("Data") for c in cols)
        return has_time and has_data

    valid = [p for p in csvs if header_ok(p)]
    if not valid:
        raise ValueError(
            f"no CSV in {folder} has 'Time' + 'Data' columns; "
            f"found: {[p.name for p in csvs]}")

    filt = [p for p in valid if "angle_filt" in p.name.lower()]
    ang = [p for p in valid if "angle" in p.name.lower()]
    chosen = (filt or ang or valid)[0]
    if len(valid) > 1:
        print(f"  [note] {folder.name}: {len(csvs)} CSVs present, using {chosen.name}")
    return chosen


def load_whisker_csv(folder: Path, use_inverted: bool = False) -> pd.DataFrame:
    """Read the angle CSV in a left/ or right/ folder.
    Returns a DataFrame with columns ['frame', 'angle'].

    The "Time" column is the frame index. For the angle column:
      * use_inverted=False -> the first "Data" column (raw angle).
      * use_inverted=True  -> the inverted angle, stored as a column named
        "Data_Inverted" (older files store it as an unnamed second "Data",
        which pandas renames "Data.1"); it equals -1 * Data. If neither is
        present, the raw column is negated as a fallback.
    """
    csv_path = _pick_csv(folder)
    df = pd.read_csv(csv_path)

    time_col = "Time" if "Time" in df.columns else df.columns[0]

    # candidate inverted columns, by preference
    inv_candidates = [c for c in df.columns
                      if str(c) in ("Data_Inverted", "Data.1")]
    raw_candidates = [c for c in df.columns if str(c) == "Data"]
    if not raw_candidates:
        raw_candidates = [c for c in df.columns if str(c).startswith("Data")
                          and c not in inv_candidates]

    if use_inverted:
        if inv_candidates:
            angle = pd.to_numeric(df[inv_candidates[0]], errors="coerce")
            which = inv_candidates[0]
        elif raw_candidates:
            angle = -pd.to_numeric(df[raw_candidates[0]], errors="coerce")
            which = f"-{raw_candidates[0]} (negated fallback)"
        else:
            raise ValueError(f"{csv_path.name}: no Data/Data_Inverted column found")
    else:
        if not raw_candidates:
            raise ValueError(f"{csv_path.name}: no Data column found")
        angle = pd.to_numeric(df[raw_candidates[0]], errors="coerce")
        which = raw_candidates[0]

    print(f"  [{folder.name}] file: {csv_path.name}  angle column: {which}")
    out = pd.DataFrame({
        "frame": pd.to_numeric(df[time_col], errors="coerce"),
        "angle": angle,
    }).dropna()
    if out.empty:
        raise ValueError(
            f"{csv_path.name}: no usable rows after parsing "
            f"(time_col='{time_col}', angle='{which}'). Check the file's columns.")
    return out


def find_h5(raw_dir: Path) -> Path:
    h5s = sorted(glob.glob(str(raw_dir / "*.h5")))
    if not h5s:
        raise FileNotFoundError(f"no .h5 found in {raw_dir}")
    if len(h5s) > 1:
        print(f"  [note] {len(h5s)} .h5 files found, using {Path(h5s[0]).name}")
    return Path(h5s[0])


def get_pulsepal_and_timing(h5_path: Path, which_intervals):
    """Extract selected pulse-pal intervals plus camera timing.

    which_intervals: 1-based list of consecutive interval numbers to show,
        e.g. [1, 2] for the first two or [3, 4] for the third and fourth.

    Returns a dict with:
      cam_rise_s   : array of camera-frame times in SECONDS (one per frame;
                     frame N of the whisker data = cam_rise_s[N])
      sample_rate  : acquisition sample rate (Hz)
      t0_s         : absolute time (s) of the START of the first selected
                     interval -> this becomes 0 ms on the plot
      intervals_ms : list of (start_ms, end_ms) for the selected intervals,
                     already shifted so t0 = 0
      intervals_frame : list of (start_frame, end_frame) for the selected
                     intervals (0-based; first cam TTL = frame 0)
      n_total      : total number of pulse-pal intervals available
    """
    import h5py
    with h5py.File(h5_path, "r") as f:
        sample_rate = float(f["header/AcquisitionSampleRate"][()].ravel()[0])
        raw = f[DATA_KEY][()]
    raw = np.asarray(raw).ravel()

    cam = ((raw >> CAM_BIT) & 1).astype(np.int8)
    pulse = ((raw >> PULSE_BIT) & 1).astype(np.int8)

    cam_rise = np.where(np.diff(cam) == 1)[0] + 1        # sample idx per frame
    cam_rise_s = cam_rise / sample_rate                   # seconds per frame

    d = np.diff(pulse)
    pp_start = np.where(d == 1)[0] + 1
    pp_end = np.where(d == -1)[0] + 1
    n_total = min(len(pp_start), len(pp_end))

    # 1-based -> 0-based indices, validated
    idx = [i - 1 for i in which_intervals]
    for i, orig in zip(idx, which_intervals):
        if i < 0 or i >= n_total:
            raise IndexError(
                f"requested interval {orig} but only {n_total} pulse-pal "
                f"intervals exist in {h5_path.name}")

    # t=0 at the start of the FIRST selected interval
    t0_s = pp_start[idx[0]] / sample_rate

    def sample_to_frame(s):
        return int(np.searchsorted(cam_rise, s, side="right") - 1)

    intervals_ms, intervals_frame = [], []
    for i in idx:
        s0, s1 = pp_start[i] / sample_rate, pp_end[i] / sample_rate
        intervals_ms.append(((s0 - t0_s) * 1000.0, (s1 - t0_s) * 1000.0))
        intervals_frame.append((sample_to_frame(pp_start[i]),
                                sample_to_frame(pp_end[i])))

    return {
        "cam_rise_s": cam_rise_s,
        "sample_rate": sample_rate,
        "t0_s": t0_s,
        "intervals_ms": intervals_ms,
        "intervals_frame": intervals_frame,
        "n_total": n_total,
    }


def frames_to_ms(frames, cam_rise_s, t0_s):
    """Convert whisker frame numbers to milliseconds relative to t0.
    frame N -> cam_rise_s[N]; frames beyond the recorded range are clipped."""
    f = np.asarray(frames, dtype=float)
    idx = np.clip(f.astype(int), 0, len(cam_rise_s) - 1)
    return (cam_rise_s[idx] - t0_s) * 1000.0


def results_path(src_dir: Path, results_root: Path, filename: str) -> Path:
    """Mirror the session sub-path (everything after a 'data' or 'raw' folder)
    under results_root, and return the full path for `filename`.

    e.g. src = ...\\raw\\TeLC_Silencing\\WA023\\20260413\\Post_Day7
         ->  <results_root>\\TeLC_Silencing\\WA023\\20260413\\Post_Day7\\<filename>
    If no 'data'/'raw' component is found, the session folder name is used.
    """
    parts = src_dir.resolve().parts
    sub = None
    for anchor in ("data", "raw"):
        idxs = [i for i, p in enumerate(parts) if p.lower() == anchor]
        if idxs:
            sub = Path(*parts[idxs[-1] + 1:])
            break
    if sub is None:
        sub = Path(src_dir.name)
    out_dir = Path(results_root) / sub
    out_dir.mkdir(parents=True, exist_ok=True)  # create folders if missing
    return out_dir / filename


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DEFAULT_DATA,
                    help="Folder containing left/ and right/ whisker CSVs.")
    ap.add_argument("--raw-dir", default=DEFAULT_RAW,
                    help="Folder containing the WaveSurfer .h5.")
    ap.add_argument("--intervals", type=int, nargs="+", default=[1, 2],
                    help="Which pulse-pal intervals to show, 1-based. Give the "
                         "consecutive pair you want: '1 2' (default) for the "
                         "first and second, '3 4' for the third and fourth, etc. "
                         "You can also pass a single number or more than two.")
    ap.add_argument("--left-inverted", action="store_true", default=True,
                    help="Use the inverted (Data_Inverted) column for the LEFT "
                         "whisker. On by default for this dataset.")
    ap.add_argument("--left-raw", dest="left_inverted", action="store_false",
                    help="Use the raw (non-inverted) column for the LEFT whisker.")
    ap.add_argument("--right-inverted", action="store_true", default=False,
                    help="Use the inverted column for the RIGHT whisker too.")
    ap.add_argument("--pad-ms", type=float, default=500.0,
                    help="Milliseconds of context to show on each side of the "
                         "shaded region when auto-zooming (default 500).")
    ap.add_argument("--no-zoom", action="store_true",
                    help="Show the whole recording instead of zooming to the "
                         "shaded intervals.")
    ap.add_argument("--results-root", default=RESULTS_ROOT,
                    help="Top of the results tree. The session sub-path (after "
                         "data\\ or raw\\) is recreated here and the plot saved "
                         "inside it.")
    ap.add_argument("--save", default=None,
                    help="Explicit output path. Overrides the automatic "
                         "results-folder location. Extension sets the format "
                         "(.svg, .png, ...).")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir = Path(args.raw_dir)

    print(f"Loading whisker CSVs from {data_dir}")
    left = load_whisker_csv(data_dir / "left", use_inverted=args.left_inverted)
    right = load_whisker_csv(data_dir / "right", use_inverted=args.right_inverted)
    print(f"  left : {len(left):,} frames ({left['frame'].min():.0f}-{left['frame'].max():.0f})")
    print(f"  right: {len(right):,} frames ({right['frame'].min():.0f}-{right['frame'].max():.0f})")

    h5_path = find_h5(raw_dir)
    print(f"Reading pulse-pal from {h5_path.name}")
    info = get_pulsepal_and_timing(h5_path, args.intervals)
    cam_rise_s, t0_s = info["cam_rise_s"], info["t0_s"]
    print(f"  pulse-pal intervals available: {info['n_total']}")
    print(f"  showing intervals {args.intervals} "
          f"(t=0 ms at start of interval {args.intervals[0]})")
    for k, (ms0, ms1) in zip(args.intervals, info["intervals_ms"]):
        print(f"    interval {k}: {ms0:8.1f} - {ms1:8.1f} ms")

    # frame -> ms (relative to t0) for the whisker traces
    left_ms = frames_to_ms(left["frame"].values, cam_rise_s, t0_s)
    right_ms = frames_to_ms(right["frame"].values, cam_rise_s, t0_s)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.plot(left_ms, left["angle"].values, lw=0.7, color="#d62728", label="Left whisker")
    ax.plot(right_ms, right["angle"].values, lw=0.7, color="#1f77b4", label="Right whisker")

    for j, (ms0, ms1) in enumerate(info["intervals_ms"]):
        ax.axvspan(ms0, ms1, color="0.5", alpha=0.25,
                   label="Cotton Swab Poke" if j == 0 else None)

    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.6)  # mark t=0
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Whisker angle (deg)")
    ax.set_ylim(-90, 90)

    parts = raw_dir.resolve().parts
    label = " / ".join(parts[-3:]) if len(parts) >= 3 else raw_dir.name
    shown = ", ".join(str(i) for i in args.intervals)
    ax.set_title(f"Whisker angle with cotton swab pokes {shown}\n{label}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0)

    if not args.no_zoom and info["intervals_ms"]:
        lo = min(m0 for m0, _ in info["intervals_ms"]) - args.pad_ms
        hi = max(m1 for _, m1 in info["intervals_ms"]) + args.pad_ms
        ax.set_xlim(lo, hi)

    fig.tight_layout()

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Auto-name: mirror the session sub-path under results/, save as SVG.
        # Filename encodes the session and which intervals were shown.
        sess_tag = "_".join(p for p in raw_dir.resolve().parts[-3:])
        ivl_tag = "-".join(str(i) for i in args.intervals)
        fname = f"whisker_pulsepal_{sess_tag}_int{ivl_tag}.svg"
        out_path = results_path(raw_dir, Path(args.results_root), fname)

    fig.savefig(out_path)  # format inferred from extension (.svg = vector)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
