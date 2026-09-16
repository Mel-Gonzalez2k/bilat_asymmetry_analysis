#!/usr/bin/env python3
r"""
Decompose a whisker angle trace into phase, amplitude, and midpoint following
Hill, Curtis, Moore & Kleinfeld (Neuron 2011), using scipy's Hilbert transform.

Method (Hill et al. 2011, "Decomposition of Rhythmic Whisks", Eq. 1):
    theta_hat(t) = amplitude(t) * cos[phase(t)] + midpoint(t)

  * The angle is band-passed to isolate the whisking rhythm (fast component).
  * A Hilbert transform of the band-passed signal gives the rapidly varying
    PHASE phi(t), increasing from -pi to pi on each whisk cycle. Per Hill's
    convention, (-pi, 0) is protraction and (0, pi) is retraction.
  * AMPLITUDE and MIDPOINT (the slow components) are estimated PER WHISK CYCLE
    from the protracted and retracted extremes, then interpolated to every
    timepoint:
        amplitude = (peak - trough) / 2
        midpoint  = (peak + trough) / 2
    (Hill sample these at the phase landmarks phi = 0 and phi = +/-pi; taking the
    per-cycle max/min of the angle is the equivalent, robust implementation.)
  * Reconstructing theta_hat via Eq. 1 and comparing to the real angle gives a
    consistency check; Hill report ~2.7 deg mean error on long recordings.

Note the difference from a plain continuous Hilbert envelope: amplitude here is a
PER-CYCLE quantity (Hill's method), not |analytic signal| at every sample. Both
are reported so they can be compared. Midpoint is NOT from the Hilbert transform
-- it is the slow setpoint measured from the cycle extremes.

Input CSV: columns "Time" (frame) and "Data" (angle, degrees).

Usage
-----
    uv run hill_decompose.py
    uv run hill_decompose.py --csv "E:\...\right\right_whisker_angle_snipet.csv" --fps 500

Output: a decomposition figure (SVG) and a per-sample CSV next to the input,
or in --out-dir if given.

Dependencies: numpy, pandas, scipy, matplotlib.
"""
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "pandas", "scipy", "matplotlib"]
# ///

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt

DEFAULT_CSV = r"E:\bilat_asymmetry_analysis\data\TeLC_Silencing\WA023\20260413\Post_Day7\right\right_whisker_angle_snipet.csv"
BP_LO, BP_HI = 4.0, 25.0      # whisking band (Hz), Hill 2011 range
CLIP_DEG = 90.0               # despike tracking glitches


def pick_csv(path_arg: str) -> Path:
    p = Path(path_arg)
    if p.is_file():
        return p
    if p.is_dir():  # a folder was given -> take the angle CSV in it
        cands = sorted(glob.glob(str(p / "*.csv")))
        filt = [c for c in cands if "angle" in Path(c).name.lower()]
        if filt:
            return Path(filt[0])
        if cands:
            return Path(cands[0])
    raise FileNotFoundError(f"no CSV found at {path_arg}")


def load_angle(csv_path: Path):
    df = pd.read_csv(csv_path)
    tcol = "Time" if "Time" in df.columns else df.columns[0]
    # prefer a plain "Data" angle column; fall back to first numeric non-time col
    if "Baseline_Subtracted" in df.columns:
        acol = "Baseline_Subtracted"
    else:
        acol = [c for c in df.columns if c != tcol][0]
    frame = pd.to_numeric(df[tcol], errors="coerce").to_numpy(float)
    ang = pd.to_numeric(df[acol], errors="coerce").to_numpy(float).copy()
    ang[np.abs(ang) > CLIP_DEG] = np.nan
    # interpolate small gaps so filtering works
    ang = pd.Series(ang).interpolate(limit=10).to_numpy()
    return frame, ang, acol


def decompose(ang, fps):
    """Return dict with band-passed signal, phase, per-cycle amplitude & midpoint
    (interpolated to every sample), the continuous Hilbert envelope, and the
    Eq.1 reconstruction with its mean absolute error."""
    nyq = fps / 2.0
    fill = np.where(np.isnan(ang), np.nanmean(ang), ang)

    # --- band-pass, then Hilbert for phase (Hill: fast component) ---
    b, a = butter(2, [BP_LO / nyq, BP_HI / nyq], btype="band")
    bp = filtfilt(b, a, fill)
    z = hilbert(bp)
    phase = np.angle(z)                 # -pi..pi each cycle
    cont_env = np.abs(z)                # continuous |analytic| envelope (for comparison)

    # --- per-whisk-cycle amplitude & midpoint from the extremes (Hill) ---
    # cycle boundaries at phase wraps (pi -> -pi)
    wraps = np.where(np.diff(phase) < -np.pi)[0] + 1
    bounds = np.r_[0, wraps, len(ang)]
    amp_c, mid_c, t_c = [], [], []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if e - s < 3:
            continue
        seg = ang[s:e]
        if np.all(np.isnan(seg)):
            continue
        pk, tr = np.nanmax(seg), np.nanmin(seg)
        amp_c.append((pk - tr) / 2.0)
        mid_c.append((pk + tr) / 2.0)
        t_c.append((s + e) / 2.0)
    t_c = np.asarray(t_c, float)
    amp_c = np.asarray(amp_c, float)
    mid_c = np.asarray(mid_c, float)

    tt = np.arange(len(ang), dtype=float)
    if len(t_c) >= 2:
        amplitude = np.interp(tt, t_c, amp_c)
        midpoint = np.interp(tt, t_c, mid_c)
    else:  # too few cycles to interpolate; fall back to constants
        amplitude = np.full_like(tt, np.nanmedian(amp_c) if len(amp_c) else np.nan)
        midpoint = np.full_like(tt, np.nanmedian(mid_c) if len(mid_c) else np.nan)

    # --- Eq. 1 reconstruction + consistency check ---
    recon = amplitude * np.cos(phase) + midpoint
    err = float(np.nanmean(np.abs(ang - recon)))

    return dict(bandpassed=bp, phase=phase, amplitude=amplitude, midpoint=midpoint,
                cont_env=cont_env, recon=recon, err=err,
                n_cycles=len(amp_c),
                amp_cycle=amp_c, mid_cycle=mid_c, t_cycle=t_c)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=DEFAULT_CSV,
                    help="Angle CSV (or a folder containing it).")
    ap.add_argument("--fps", type=float, default=500.0,
                    help="Camera frame rate (Hz). Default 500.")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write outputs. Default: next to the input CSV.")
    args = ap.parse_args()

    csv_path = pick_csv(args.csv)
    print(f"Loading {csv_path}")
    frame, ang, acol = load_angle(csv_path)
    fps = args.fps
    print(f"  {len(ang)} samples ({len(ang)/fps*1000:.0f} ms at {fps:.0f} fps), "
          f"angle column '{acol}', range {np.nanmin(ang):.1f} to {np.nanmax(ang):.1f} deg")

    dec = decompose(ang, fps)
    print(f"  whisk cycles: {dec['n_cycles']}")
    print(f"  median amplitude (per-cycle): {np.nanmedian(dec['amplitude']):.1f} deg")
    print(f"  median midpoint: {np.nanmedian(dec['midpoint']):.1f} deg")
    print(f"  Eq.1 reconstruction mean error: {dec['err']:.2f} deg "
          f"(Hill 2011 report ~2.7 deg on long recordings)")

    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem

    # per-sample CSV
    out_csv = out_dir / f"{stem}_hill_decomp.csv"
    pd.DataFrame({
        "frame": frame, "angle": ang,
        "bandpassed": dec["bandpassed"],
        "phase_rad": dec["phase"],
        "amplitude_deg": dec["amplitude"],
        "midpoint_deg": dec["midpoint"],
        "reconstruction_deg": dec["recon"],
        "continuous_envelope_deg": dec["cont_env"],
    }).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # figure
    t_ms = (frame - frame[0]) / fps * 1000.0
    fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True,
                           gridspec_kw={"height_ratios": [3, 2, 2, 2]})
    for a in ax:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)

    ax[0].plot(t_ms, ang, color="#1f77b4", lw=1.2, label="angle")
    ax[0].plot(t_ms, dec["recon"], color="k", lw=1, ls="--", label="reconstruction (Eq.1)")
    ax[0].plot(t_ms, dec["midpoint"], color="#2ca02c", lw=1.5, label="midpoint")
    ax[0].set_ylabel("Angle (deg)")
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].set_title(f"Hill et al. 2011 decomposition  \u2014  {stem}\n"
                    f"reconstruction error {dec['err']:.2f}\u00b0, {dec['n_cycles']} cycles")

    ax[1].plot(t_ms, dec["amplitude"], color="#d62728", lw=1.5, label="amplitude (per-cycle)")
    ax[1].plot(t_ms, dec["cont_env"], color="0.6", lw=1, label="continuous |Hilbert| (compare)")
    ax[1].set_ylabel("Amplitude (deg)")
    ax[1].legend(fontsize=8, loc="upper right")

    ax[2].plot(t_ms, dec["midpoint"], color="#2ca02c", lw=1.5)
    ax[2].set_ylabel("Midpoint (deg)")

    ax[3].plot(t_ms, dec["phase"], color="#9467bd", lw=1)
    ax[3].set_ylabel("Phase (rad)")
    ax[3].set_yticks([-np.pi, 0, np.pi]); ax[3].set_yticklabels(["-\u03c0", "0", "\u03c0"])
    ax[3].set_xlabel("Time (ms)")

    fig.tight_layout()
    out_svg = out_dir / f"{stem}_hill_decomp.svg"
    fig.savefig(out_svg)
    print(f"Saved: {out_svg}")


if __name__ == "__main__":
    main()