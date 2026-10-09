#!/usr/bin/env python3
r"""
Overlay LEFT vs RIGHT whisker angle across all stimulations, aligned to each
pulse onset (t = 0), on a single axes.  **FILTERED version.**

Same as overlay_left_right_per_stim.py, but it selects the FILTERED angle
traces and requires the left file to be the REVERSED one:

  RIGHT side (mouse LEFT whisker):  Angle_Right*.csv  containing "filt"/"filter"
  LEFT  side (mouse RIGHT whisker): Angle_Left*.csv   containing "reverse" AND "filt"/"filter"

(matching is case-insensitive). Everything else — pulse extraction, overlay,
mean, baseline option, save-safe output — is unchanged.

For every pulse-pal stimulation in the .h5, a window [-pre, +post] around its
onset is cut from both whisker traces and plotted on a common time axis. All
stimulations are overlaid: each individual stim is a faint line, and the mean
across stims is drawn bold on top, for the left and the right whisker.

Usage
-----
    uv run overlay_left_right_per_stim_filt.py \
        --session-dir "E:\...\MRN_opto14\...\2_100ms_1s"
    uv run overlay_left_right_per_stim_filt.py --trials 1 2 3
    uv run overlay_left_right_per_stim_filt.py --baseline-ms 200
    uv run overlay_left_right_per_stim_filt.py --no-mean

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

DEFAULT_SESSION = r"E:\bilat_asymmetry_analysis\data\Optogenetic_Activation\MRN_opto14\20260518\2_100ms_1s"
#DEFAULT_SESSION = r"E:\bilat_asymmetry_analysis\data\Optogenetic_Activation\MRN_opto12\20260517\1_at_rest_first_stim_100ms_1s"
CAM_BIT, PULSE_BIT, FPS = 0, 1, 500.0
LEFTWHISKER_COLOR = "#00b140"   # green  (mouse LEFT, from Angle_Right)
RIGHTWHISKER_COLOR = "#d400d4"  # magenta(mouse RIGHT, from Angle_Left)
STIM_COLOR = "#00ffff"
STIM_ALPHA = 0.30

# ---- file-selection requirements (case-insensitive substrings) -------------
FILT_TOKENS = ("filt", "filter")   # a file is "filtered" if its name has either
# left file must ALSO say "reverse"; right file only needs to be filtered
LEFT_REQUIRE_REVERSE = "reverse"
# ----------------------------------------------------------------------------


def _is_filtered(name_lower: str) -> bool:
    return any(tok in name_lower for tok in FILT_TOKENS)


def find_angle_csv(session: Path, side: str) -> Path:
    """side is 'Right' or 'Left' (screen side).

    Matches the side word ANYWHERE in the name, so both naming styles work:
        Angle_Right_90461_93711.csv                      (raw)
        Angle_Filter_8_30Hz_Right_90461_93711.csv        (filtered)
        Angle_Left_reverse_90461_93711.csv               (raw, reversed)
        Angle_Filter_8_30Hz_Left_reverse_90461_93711.csv (filtered, reversed)

    LEFT  : MUST contain 'reverse' (hard requirement). Among those, prefer a
            FILTERED file; else fall back to the reverse file with a note.
    RIGHT : no 'reverse' requirement. Prefer a FILTERED file; else fall back to
            the available Angle_*Right* file with a note.
    """
    other = "left" if side == "Right" else "right"
    # all angle CSVs (Angle_* prefix), excluding Lines_/Mask_ etc.
    angle = [Path(p) for p in glob.glob(str(session / "*.csv"))
             if Path(p).name.lower().startswith("angle")]
    # this side only (contains this side word, not the other)
    pool = [p for p in angle
            if side.lower() in p.name.lower() and other not in p.name.lower()]

    if side == "Left":
        pool = [p for p in pool if LEFT_REQUIRE_REVERSE in p.name.lower()]
        if not pool:
            raise FileNotFoundError(
                f"No 'reverse' Angle_*Left*.csv in {session}.\n"
                f"  Angle CSVs present: {[p.name for p in sorted(angle)]}")
    elif not pool:
        raise FileNotFoundError(
            f"No Angle_*Right*.csv in {session}.\n"
            f"  Angle CSVs present: {[p.name for p in sorted(angle)]}")

    # prefer filtered; otherwise fall back with a loud note
    filt = [p for p in pool if _is_filtered(p.name.lower())]
    if filt:
        chosen = sorted(filt)
    else:
        chosen = sorted(pool)
        print(f"  [!] no FILTERED Angle_*{side}* found; using non-filtered "
              f"{chosen[0].name}")

    if len(chosen) > 1:
        print(f"  [!] {len(chosen)} candidates for {side}; using {chosen[0].name}")
        for c in chosen[1:]:
            print(f"        (ignored: {c.name})")
    return chosen[0]


def load_angle(csv_path: Path, clip_deg):
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    tcol = "Time" if "Time" in cols else cols[0]
    acol = "Data" if "Data" in cols else ("Angle" if "Angle" in cols
                                          else next(c for c in cols if c != tcol))
    frame = pd.to_numeric(df[tcol], errors="coerce").to_numpy(float)
    ang = pd.to_numeric(df[acol], errors="coerce").to_numpy(float).copy()
    if clip_deg is not None:
        ang[np.abs(ang) > clip_deg] = np.nan
    m = np.isfinite(frame) & np.isfinite(ang)
    frame, ang = frame[m], ang[m]
    order = np.argsort(frame)
    print(f"    {csv_path.name}: {m.sum()} pts, frames {frame.min():.0f}-{frame.max():.0f}")
    return frame[order], ang[order]


def find_h5(session: Path) -> Path:
    h5s = sorted(glob.glob(str(session / "*.h5")))
    h5s = [h for h in h5s if "line_masks" not in Path(h).name.lower()]
    if not h5s:
        raise FileNotFoundError(f"no acquisition .h5 in {session}")
    return Path(h5s[0])


def sweep_key(h5_name: str) -> str:
    import re
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
    print(f"  {key}: {len(cam_rise_s)} cam frames, {n} pulses, {raw.size/sr:.1f} s")
    return cam_rise_s, ps[:n], pe[:n]


def frame_to_abs_s(frames, cam_rise_s):
    idx = np.clip(frames, 0, len(cam_rise_s) - 1).astype(int)
    out = cam_rise_s[idx].astype(float)
    over = frames > (len(cam_rise_s) - 1)
    if np.any(over):
        out[over] = cam_rise_s[-1] + (frames[over] - (len(cam_rise_s) - 1)) / FPS
    return out


def stack(abs_s, ang, onsets, grid_s, baseline_ms):
    """Return (n_stim, n_grid) array of angle resampled onto grid_s per stim."""
    out = np.full((len(onsets), len(grid_s)), np.nan)
    for k, t0 in enumerate(onsets):
        rel = abs_s - t0
        m = (rel >= grid_s[0] - 0.01) & (rel <= grid_s[-1] + 0.01)
        if m.sum() < 2:
            continue
        out[k] = np.interp(grid_s, rel[m], ang[m], left=np.nan, right=np.nan)
        if baseline_ms is not None:
            b = (grid_s >= -baseline_ms / 1000.0) & (grid_s < 0)
            if b.any():
                out[k] -= np.nanmean(out[k][b])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-dir", default=DEFAULT_SESSION)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--pre-ms", type=float, default=1000.0)
    ap.add_argument("--post-ms", type=float, default=1000.0)
    ap.add_argument("--trials", nargs="+", default=["all"],
                    help="Which stimulations to include (1-based) or 'all'.")
    ap.add_argument("--baseline-ms", type=float, default=None,
                    help="Subtract each trace's mean over this many ms before onset "
                         "so all stims start from 0 (cleaner overlay). Default: off (raw).")
    ap.add_argument("--no-mean", action="store_true",
                    help="Don't draw the bold mean trace, only the individual stims.")
    ap.add_argument("--clip-deg", type=float, default=None)
    ap.add_argument("--offset", type=float, default=None,
                    help="Vertical gap (deg) between the left and right whisker. "
                         "Default: auto-separate so the traces don't overlap. "
                         "Use --offset 0 to overlay on one baseline.")
    ap.add_argument("--sep-pad", type=float, default=8.0,
                    help="Extra padding (deg) added to the auto offset.")
    args = ap.parse_args()

    session = Path(args.session_dir)
    if not session.is_dir():
        raise SystemExit(f"Not a folder: {session}")
    print(f"Session: {session}\nLoading FILTERED angles:")
    lw_f, lw_a = load_angle(find_angle_csv(session, "Right"), args.clip_deg)  # mouse LEFT
    rw_f, rw_a = load_angle(find_angle_csv(session, "Left"), args.clip_deg)   # mouse RIGHT (reversed)
    cam_rise_s, ps, pe = read_h5(find_h5(session))
    if len(ps) == 0:
        raise SystemExit("no pulses found in h5")

    # select stims
    if [s.lower() for s in args.trials] == ["all"]:
        sel = list(range(len(ps)))
    else:
        sel = [int(s) - 1 for s in args.trials if 0 < int(s) <= len(ps)]
    onsets = ps[sel]
    widths_ms = (pe[sel] - ps[sel]) * 1000.0
    print(f"  {len(onsets)} stimulation(s), pulse width ~{np.median(widths_ms):.0f} ms")

    lw_abs = frame_to_abs_s(lw_f, cam_rise_s)
    rw_abs = frame_to_abs_s(rw_f, cam_rise_s)

    dt = 1000.0 / FPS
    grid = np.arange(-args.pre_ms, args.post_ms + dt, dt) / 1000.0  # seconds
    L = stack(lw_abs, lw_a, onsets, grid, args.baseline_ms)
    R = stack(rw_abs, rw_a, onsets, grid, args.baseline_ms)

    # ---- vertical offset: put RIGHT whisker above LEFT so they don't overlap ----
    if args.offset is not None:
        offset = args.offset
    else:
        top_left = np.nanpercentile(L, 98)
        bot_right = np.nanpercentile(R, 2)
        offset = max(0.0, top_left - bot_right) + args.sep_pad
    print(f"  vertical offset (right above left): {offset:.1f} deg")
    Rp = R + offset  # shifted right whisker

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvspan(0, np.median(widths_ms) / 1000.0, color=STIM_COLOR, alpha=STIM_ALPHA,
               lw=0, label=f"Stim (~{np.median(widths_ms):.0f} ms)")
    ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.5)
    # dotted baselines marking each whisker's zero
    if offset != 0:
        ax.axhline(0, color=LEFTWHISKER_COLOR, lw=0.8, ls=":", alpha=0.6)
        ax.axhline(offset, color=RIGHTWHISKER_COLOR, lw=0.8, ls=":", alpha=0.6)

    for k in range(len(onsets)):
        ax.plot(grid, L[k], color=LEFTWHISKER_COLOR, lw=0.7, alpha=0.45)
        ax.plot(grid, Rp[k], color=RIGHTWHISKER_COLOR, lw=0.7, alpha=0.45)
    if not args.no_mean:
        ax.plot(grid, np.nanmean(L, 0), color=LEFTWHISKER_COLOR, lw=2.4,
                label="Left whisker (mean)")
        ax.plot(grid, np.nanmean(Rp, 0), color=RIGHTWHISKER_COLOR, lw=2.4,
                label="Right whisker (mean)")
    else:
        ax.plot([], [], color=LEFTWHISKER_COLOR, lw=2, label="Left whisker")
        ax.plot([], [], color=RIGHTWHISKER_COLOR, lw=2, label="Right whisker")

    ax.set_xlabel("Time from stim onset (s)")
    ylab = "Whisker angle (deg)"
    if offset != 0:
        ylab += f"  (right shifted +{offset:.0f}°)"
    if args.baseline_ms:
        ylab += f"  (baseline -{args.baseline_ms:.0f} ms)"
    ax.set_ylabel(ylab)
    ax.set_xlim(grid[0], grid[-1])
    parts = session.resolve().parts
    sess_lbl = " / ".join(parts[-3:]) if len(parts) >= 3 else session.name
    ax.set_title(f"Left vs right whisker (filtered) \u2014 {len(onsets)} stimulations overlaid\n{sess_lbl}")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.12)

    out_dir = Path(args.out_dir) if args.out_dir else session
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_".join(parts[-3:]) if len(parts) >= 3 else session.name
    tag = "".join(c if (c.isalnum() or c in "-_") else "_" for c in tag)
    bl = f"_bl{int(args.baseline_ms)}" if args.baseline_ms else ""
    stem = f"overlay_LvR_perstim_filt{bl}_{tag}"
    fig.tight_layout()
    for ext in ("svg", "png"):
        p = out_dir / f"{stem}.{ext}"
        try:
            fig.savefig(p, dpi=150); print(f"Saved: {p}")
        except (PermissionError, OSError) as e:
            for i in range(1, 100):
                alt = out_dir / f"{stem}_{i}.{ext}"
                try:
                    fig.savefig(alt, dpi=150)
                    print(f"  [note] {p.name} locked; saved {alt.name}"); break
                except (PermissionError, OSError):
                    continue


if __name__ == "__main__":
    main()
