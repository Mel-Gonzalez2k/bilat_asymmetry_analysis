#!/usr/bin/env python3
r"""
Plot the RAW left and right whisker angles for one optogenetic-activation session
and overlay the pulse-pal stimulation TRIALS (sets of pulses) extracted from the .h5.

Data (all in one session folder, e.g.
  E:\bilat_asymmetry_analysis\data\Optogenetic_Activation\MRN_opto13\202260527\3_whisking_first_stim_001_20Hz):

  Angle_Left_<range>.csv   -> raw LEFT angle   (the files WITHOUT "Filt" in the name)
  Angle_Right_<range>.csv  -> raw RIGHT angle   (columns: Time=frame, Data=angle)
  MRN_opto..._001_0001.h5  -> WaveSurfer digital TTLs
      bit 0 = cam        (one rising edge per camera frame, 500 fps)
      bit 1 = pulse_pal  (high = light pulse)

The "Filt" angle CSVs are ignored on purpose: this plots the angles you calculated,
not the band-pass-filtered ones.

h5 data key: sweep_NNNN from the LAST 4-digit group before ".h5"
(e.g. "..._001_0001.h5" -> sweep_0001/digitalScans).

TRIALS (sets of intervals)
--------------------------
A "20 Hz for 3 s" run is one SET = one trial = a train of ~60 pulses at 20 Hz.
A recording may hold several trials (here: 3 trains, ~33 s apart). The pulses are
grouped into trials automatically (a new trial starts whenever the gap to the
previous pulse exceeds --gap seconds, default 0.5 s). Choose which to show with
--trials: e.g. "--trials 1" for only the first trial (default: all trials that
fall inside the angle window). --pulses shades each individual 5 ms pulse within
the chosen trials instead of the whole train.

Usage
-----
    uv run plot_opto_angles.py                       # all trials in the angle window
    uv run plot_opto_angles.py --trials 1            # only the first trial
    uv run plot_opto_angles.py --trials 1 --x-origin trial1   # 0 s at trial-1 onset
    uv run plot_opto_angles.py --trials 1 --pad-ms 1000       # zoom to trial 1 +/-1 s
    uv run plot_opto_angles.py --pulses              # shade individual pulses
    uv run plot_opto_angles.py --clip-deg 90         # NaN out tracking glitches > 90 deg

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

#DEFAULT_SESSION = r"E:\bilat_asymmetry_analysis\data\Optogenetic_Activation\MRN_opto13\202260527\3_whisking_first_stim_001_20Hz"
DEFAULT_SESSION = r"E:\bilat_asymmetry_analysis\data\Optogenetic_Activation\MRN_opto14\20260518\2_100ms_1s"

CAM_BIT = 0
PULSE_BIT = 1
FPS = 500.0

# NOTE on sides: the CSV "Left"/"Right" refer to the LEFT/RIGHT of the SCREEN,
# which are mirror-flipped relative to the mouse. So:
#   Angle_Right_*.csv  -> the mouse's LEFT whisker
#   Angle_Left_*.csv   -> the mouse's RIGHT whisker
LEFTWHISKER_COLOR = "#00ff00"   # green   (R0 G255 B0)   = mouse LEFT whisker  (from Angle_Right)
RIGHTWHISKER_COLOR = "#ff00ff"  # magenta (R255 G0 B255) = mouse RIGHT whisker (from Angle_Left)
STIM_COLOR = "#00ffff"          # teal/cyan (R0 G255 B255)
STIM_ALPHA = 0.45               # a little faded


# ---------------- angle CSVs ----------------
def find_angle_csv(session: Path, side: str) -> Path:
    cands = [Path(p) for p in glob.glob(str(session / f"Angle_{side}*.csv"))]
    cands = [p for p in cands if "filt" not in p.name.lower()]
    if not cands:
        raise FileNotFoundError(f"no raw Angle_{side}*.csv (non-Filt) in {session}")
    if len(cands) > 1:
        print(f"  [note] {len(cands)} Angle_{side} files, using {sorted(cands)[0].name}")
    return sorted(cands)[0]


def load_angle(csv_path: Path, clip_deg):
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    tcol = "Time" if "Time" in cols else cols[0]
    if "Data" in cols:
        acol = "Data"
    elif "Angle" in cols:
        acol = "Angle"
    else:
        acol = next((c for c in cols if c != tcol), cols[-1])
    frame = pd.to_numeric(df[tcol], errors="coerce").to_numpy(float)
    ang = pd.to_numeric(df[acol], errors="coerce").to_numpy(float).copy()
    if clip_deg is not None:
        ang[np.abs(ang) > clip_deg] = np.nan
    print(f"    {csv_path.name}: time='{tcol}', angle='{acol}', "
          f"{np.isfinite(ang).sum()} pts, frames "
          f"{np.nanmin(frame):.0f}-{np.nanmax(frame):.0f}")
    return frame, ang


# ---------------- h5 timing + stim ----------------
def find_h5(session: Path) -> Path:
    h5s = sorted(glob.glob(str(session / "*.h5")))
    if not h5s:
        raise FileNotFoundError(f"no .h5 in {session}")
    if len(h5s) > 1:
        print(f"  [note] {len(h5s)} .h5 files, using {Path(h5s[0]).name}")
    return Path(h5s[0])


def sweep_key(h5_name: str) -> str:
    """sweep_NNNN from the LAST 4-digit group in the name (e.g. '..._001_0003.h5'
    -> sweep_0003). Robust to a trailing copy suffix like '..._0003_1.h5'."""
    import re
    groups = re.findall(r"\d{4}", h5_name)
    return f"sweep_{groups[-1] if groups else '0001'}/digitalScans"


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
            print("  [warn] sample rate not in header, assuming 250 kHz")
        raw = np.asarray(f[key][()]).ravel()
    cam = ((raw >> CAM_BIT) & 1).astype(np.int8)
    pul = ((raw >> PULSE_BIT) & 1).astype(np.int8)
    cam_rise_s = (np.where(np.diff(cam) == 1)[0] + 1) / sr
    d = np.diff(pul)
    ps = (np.where(d == 1)[0] + 1) / sr
    pe = (np.where(d == -1)[0] + 1) / sr
    n = min(len(ps), len(pe))
    print(f"  h5 key {key}: {len(cam_rise_s)} cam frames, {n} pulses, {raw.size/sr:.1f} s")
    return cam_rise_s, sr, ps[:n], pe[:n]


def group_trials(ps, pe, gap_s):
    """Group pulses into trials/trains. Returns list of dicts:
    {'start','end','n','pulses':[(s,e),...]} in recording order (1-based when shown)."""
    if len(ps) == 0:
        return []
    trials = []
    cur = {"start": ps[0], "end": pe[0], "n": 1, "pulses": [(ps[0], pe[0])]}
    for i in range(1, len(ps)):
        if ps[i] - cur["end"] > gap_s:
            trials.append(cur)
            cur = {"start": ps[i], "end": pe[i], "n": 1, "pulses": [(ps[i], pe[i])]}
        else:
            cur["end"] = pe[i]; cur["n"] += 1; cur["pulses"].append((ps[i], pe[i]))
    trials.append(cur)
    return trials


def save_fig_safe(fig, path):
    """Save the figure; if the target file is locked (e.g. open in an image
    viewer on Windows), write to a numbered sibling instead of crashing, so
    re-running the script never errors out on an existing file."""
    try:
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
        return path
    except (PermissionError, OSError) as e:
        base, ext = path.with_suffix(""), path.suffix
        for i in range(1, 100):
            alt = Path(f"{base}_{i}{ext}")
            try:
                fig.savefig(alt, dpi=150)
                print(f"  [note] {path.name} was locked ({e.__class__.__name__}); "
                      f"saved as {alt.name}. Close the open viewer to overwrite "
                      f"the original.")
                return alt
            except (PermissionError, OSError):
                continue
        print(f"  [error] could not save {path.name}: {e}")
        return None


def frame_to_abs_s(frames, cam_rise_s):
    f = np.asarray(frames, dtype=float)
    idx = np.clip(f, 0, len(cam_rise_s) - 1).astype(int)
    out = cam_rise_s[idx].astype(float)
    over = f > (len(cam_rise_s) - 1)
    if over.any():
        out[over] = cam_rise_s[-1] + (f[over] - (len(cam_rise_s) - 1)) / FPS
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-dir", default=DEFAULT_SESSION)
    ap.add_argument("--out-dir", default=None,
                    help="Where to save the figure (default: the session folder).")
    ap.add_argument("--gap", type=float, default=0.5,
                    help="Seconds between pulses that starts a new trial (default 0.5).")
    ap.add_argument("--trials", nargs="+", default=["all"],
                    help="Which trial(s)/set(s) to show: trial numbers (1-based, in "
                         "recording order), or 'all' (default). e.g. --trials 1")
    ap.add_argument("--whole-train", action="store_true",
                    help="Shade each trial as one solid block instead of the "
                         "individual pulses (default: individual pulses as skinny bars).")
    ap.add_argument("--x-origin", choices=["recording", "snippet", "trial1"],
                    default="recording",
                    help="Time axis zero: recording clock (default), start of the "
                         "plotted angles (snippet), or the first shown trial's onset.")
    ap.add_argument("--pad-ms", type=float, default=None,
                    help="If set, zoom the x-axis to the shown trials +/- this many ms "
                         "(clipped to the angle data). Default: show the whole snippet.")
    ap.add_argument("--clip-deg", type=float, default=None,
                    help="NaN out |angle| above this many deg (despike). Default: off.")
    ap.add_argument("--offset", type=float, default=None,
                    help="Vertical gap (deg) added to the RIGHT-whisker trace so the two "
                         "traces are separated instead of overlapping. Default: "
                         "auto-separate. Use --offset 0 to overlay on the same scale.")
    ap.add_argument("--sep-pad", type=float, default=15.0,
                    help="Extra margin (deg) between the two traces when auto-separating "
                         "(default 15).")
    args = ap.parse_args()

    session = Path(args.session_dir)
    if not session.is_dir():
        raise SystemExit(f"Not a folder: {session}")
    print(f"Session: {session}")

    print("Loading raw angles (screen side -> mouse side):")
    # Angle_Right file = mouse LEFT whisker; Angle_Left file = mouse RIGHT whisker
    lw_f, lw_a = load_angle(find_angle_csv(session, "Right"), args.clip_deg)  # mouse LEFT
    rw_f, rw_a = load_angle(find_angle_csv(session, "Left"), args.clip_deg)   # mouse RIGHT

    cam_rise_s, sr, ps, pe = read_h5(find_h5(session))

    lw_abs = frame_to_abs_s(lw_f, cam_rise_s)
    rw_abs = frame_to_abs_s(rw_f, cam_rise_s)
    win_lo = float(np.nanmin([lw_abs.min(), rw_abs.min()]))
    win_hi = float(np.nanmax([lw_abs.max(), rw_abs.max()]))

    trials = group_trials(ps, pe, args.gap)
    print(f"Found {len(trials)} trial(s) (20 Hz trains):")
    for k, t in enumerate(trials, 1):
        dur = t["end"] - t["start"]
        inwin = not (t["end"] < win_lo or t["start"] > win_hi)
        print(f"  trial {k}: {t['start']:8.3f}-{t['end']:8.3f} s  "
              f"{t['n']} pulses  {t['n']/dur:.1f} Hz" +
              ("   <-- in angle window" if inwin else "   (outside angle window)"))

    # select trials
    if [s.lower() for s in args.trials] == ["all"]:
        sel = list(range(1, len(trials) + 1))
    else:
        sel = []
        for s in args.trials:
            try:
                k = int(s)
            except ValueError:
                raise SystemExit(f"--trials takes integers or 'all', got {s!r}")
            if 1 <= k <= len(trials):
                sel.append(k)
            else:
                print(f"  [warn] trial {k} does not exist (have 1..{len(trials)})")
    sel_trials = [trials[k - 1] for k in sel]
    print(f"Showing trial(s): {sel}")

    # epochs to shade: individual pulses (default, skinny bars) or whole-train blocks
    if args.whole_train:
        epochs = [(t["start"], t["end"]) for t in sel_trials]
    else:
        epochs = [(s, e) for t in sel_trials for (s, e) in t["pulses"]]

    # x-axis origin
    if args.x_origin == "snippet":
        origin, xlab = win_lo, "Time (s, 0 = start of plotted angles)"
    elif args.x_origin == "trial1":
        origin = sel_trials[0]["start"] if sel_trials else win_lo
        xlab = "Time (s, 0 = first shown trial onset)"
    else:
        origin, xlab = 0.0, "Time (s, recording clock)"

    # x-limits (compute first so skinny-bar min width can be set from the view)
    if args.pad_ms is not None and sel_trials:
        lo = min(t["start"] for t in sel_trials) - args.pad_ms / 1000.0
        hi = max(t["end"] for t in sel_trials) + args.pad_ms / 1000.0
        lo, hi = max(lo, win_lo), min(hi, win_hi)
    else:
        lo, hi = win_lo, win_hi
    # a 5 ms pulse can render sub-pixel on a multi-second axis; give each bar a
    # floor width so the individual pulses stay VISIBLE as separate bars. The
    # floor (0.15% of the view) stays far below the 50 ms pulse period, so the
    # gaps between bars are preserved and the stim never looks continuous.
    min_w = (hi - lo) * 0.0015 if not args.whole_train else 0.0

    # ---------------- plot ----------------
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    shaded = 0
    for a, b in epochs:
        if b < win_lo or a > win_hi:
            continue
        if (b - a) < min_w:
            c = 0.5 * (a + b); a, b = c - min_w / 2, c + min_w / 2
        ax.axvspan(a - origin, b - origin, color=STIM_COLOR, alpha=STIM_ALPHA,
                   lw=0, label=("Stim (20 Hz pulses)" if shaded == 0 else None))
        shaded += 1
    if shaded == 0:
        print("  [warn] no selected stim falls inside the angle window; nothing "
              "shaded. Try a different --trials, or export angles over the stim.")
    else:
        print(f"  shaded {shaded} stim "
              f"{'block(s)' if args.whole_train else 'pulse bar(s)'}")

    # vertical spacing so the two traces don't overlap: lift the RIGHT whisker
    if args.offset is None:  # auto: put right-whisker min just above left-whisker max
        top_left = np.nanpercentile(lw_a, 98)
        bot_right = np.nanpercentile(rw_a, 2)
        offset = max(0.0, top_left - bot_right) + args.sep_pad
    else:
        offset = args.offset
    rlabel = "Right whisker" + (f" (+{offset:.0f}°)" if offset else "")

    ax.plot(lw_abs - origin, lw_a, lw=0.8, color=LEFTWHISKER_COLOR,
            label="Left whisker")
    ax.plot(rw_abs - origin, rw_a + offset, lw=0.8, color=RIGHTWHISKER_COLOR,
            label=rlabel)
    if offset:
        # dotted baselines mark 0 deg for each trace (right trace is shifted up)
        ax.axhline(0, color=LEFTWHISKER_COLOR, lw=0.6, ls=":", alpha=0.45)
        ax.axhline(offset, color=RIGHTWHISKER_COLOR, lw=0.6, ls=":", alpha=0.45)

    ax.set_xlabel(xlab)
    ax.set_ylabel("Whisker angle (deg)" +
                  ("  (right trace offset)" if offset else ""))
    ax.set_xlim(lo - origin, hi - origin)

    parts = session.resolve().parts
    sess_lbl = " / ".join(parts[-3:]) if len(parts) >= 3 else session.name
    trstr = "all trials" if sel == list(range(1, len(trials) + 1)) else \
            ("trial " + ", ".join(map(str, sel)))
    pulsestr = " (20 Hz pulses)" if not args.whole_train else ""
    ax.set_title(f"Raw whisker angles — {trstr}{pulsestr}\n{sess_lbl}")
    ax.legend(loc="upper right", fontsize=9, ncol=3)
    ax.grid(alpha=0.12)

    out_dir = Path(args.out_dir) if args.out_dir else session
    out_dir.mkdir(parents=True, exist_ok=True)
    sess_tag = "_".join(parts[-3:]) if len(parts) >= 3 else session.name
    sess_tag = "".join(c if (c.isalnum() or c in "-_") else "_" for c in sess_tag)
    sel_tag = "all" if sel == list(range(1, len(trials) + 1)) else \
              "trial" + "-".join(map(str, sel))
    stem = f"angles_raw_LR_{sel_tag}_{sess_tag}"
    fig.tight_layout()
    for ext in ("svg", "png"):
        save_fig_safe(fig, out_dir / f"{stem}.{ext}")


if __name__ == "__main__":
    main()
