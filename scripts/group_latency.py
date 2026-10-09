#!/usr/bin/env python3
r"""
Whisker-movement LATENCY from the stim pulse, control vs experimental (panels J/K).

Latency is velocity-based and referenced to EACH PULSE'S OWN pre-stim movement,
so it does not depend on a control animal and is not fooled by a control that is
whisking vs stationary:

  1. smooth the angle, take the first derivative -> whisker velocity (deg/s);
  2. per pulse, measure the velocity noise over the pre-stim baseline window;
  3. latency = first time after onset the |velocity| exceeds
        max(k * baseline-velocity-SD, --min-vel)  for >= --min-dur-ms
     (i.e. the light drives a velocity change beyond that animal's own ongoing
     movement). If the pulse never crosses, its latency is NaN (no movement).
  (--method peak instead = time to the largest |velocity| in the window.)

Then: median latency per mouse over its pulses -> bars = group mean +/- SEM over
MICE, one dot per mouse, for left and right whisker.

Needs _grp_common.py in the same folder.

Usage
-----
    uv run group_latency.py --parent "E:\...\Optogenetic_Activation" --out-dir "E:\...\group_plots"
    uv run group_latency.py --parent "..." --vel-k 3 --resp-ms 200 --min-vel 10
    uv run group_latency.py --parent "..." --method peak

Dependencies: h5py, numpy, pandas, matplotlib.
"""
# /// script
# requires-python = ">=3.9"
# dependencies = ["h5py", "numpy", "pandas", "matplotlib"]
# ///

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _grp_common as gc

EXP_COLOR, CTRL_COLOR = "#e63946", "#1d3557"


def smooth(a, win):
    if win and win > 1:
        return pd.Series(a).rolling(win, center=True,
                                    min_periods=max(1, win // 3)).mean().to_numpy()
    return a


def pulse_latency(angle_row, grid, base_ms, resp_ms, k, min_vel, min_dur_samp,
                  smooth_win, method):
    """Latency (ms) for one pulse, or NaN."""
    a = smooth(angle_row, smooth_win)
    if np.all(~np.isfinite(a)):
        return np.nan
    vel = np.gradient(a, grid)                       # deg/s
    base = (grid >= -base_ms / 1000.0) & (grid < 0)
    resp = (grid >= 0) & (grid <= resp_ms / 1000.0)
    ridx = np.where(resp)[0]
    if ridx.size == 0:
        return np.nan
    if method == "peak":
        v = np.abs(vel[ridx])
        if np.all(~np.isfinite(v)):
            return np.nan
        return grid[ridx[int(np.nanargmax(v))]] * 1000.0
    sd = np.nanstd(vel[base]) if base.any() else np.nan
    if not np.isfinite(sd):
        sd = np.nanstd(vel)
    thr = max(k * (sd if np.isfinite(sd) else 0.0), min_vel)
    over = np.abs(vel) > thr
    for i in ridx:
        j = min(i + min_dur_samp, len(over))
        if over[i] and np.all(over[i:j]):
            return grid[i] * 1000.0
    return np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--exp", default="7,12,13,14,15")
    ap.add_argument("--ctrl", default="5,6,8")
    ap.add_argument("--session-match", default="100ms")
    ap.add_argument("--pre-ms", type=float, default=200.0)
    ap.add_argument("--post-ms", type=float, default=400.0)
    ap.add_argument("--base-ms", type=float, default=200.0, help="pre-stim baseline window")
    ap.add_argument("--min-cover", type=float, default=0.5,
                    help="keep a pulse only if >= this fraction of samples is present "
                         "in each of before/during/after windows")
    ap.add_argument("--use-filtered", action="store_true",
                    help="use the Angle_Filt/Filter files instead of RAW (default: RAW)")
    ap.add_argument("--resp-ms", type=float, default=200.0,
                    help="search window after onset for movement (ms)")
    ap.add_argument("--vel-k", type=float, default=3.0,
                    help="threshold = k x baseline velocity SD")
    ap.add_argument("--min-vel", type=float, default=10.0,
                    help="floor on the velocity threshold (deg/s) so a perfectly "
                         "still animal isn't triggered by noise; 0 = purely relative")
    ap.add_argument("--min-dur-ms", type=float, default=10.0,
                    help="velocity must stay above threshold this long")
    ap.add_argument("--smooth-ms", type=float, default=20.0,
                    help="angle smoothing before the derivative")
    ap.add_argument("--method", choices=["onset", "peak"], default="onset")
    args = ap.parse_args()

    exp_ids = [int(x) for x in args.exp.split(",") if x.strip()]
    ctrl_ids = [int(x) for x in args.ctrl.split(",") if x.strip()]
    dt = 1000.0 / gc.FPS
    grid = np.arange(-args.pre_ms, args.post_ms + dt, dt) / 1000.0
    smooth_win = int(round(args.smooth_ms / 1000.0 * gc.FPS))
    min_dur_samp = max(1, int(round(args.min_dur_ms / 1000.0 * gc.FPS)))

    print(f"Latency ({args.method}): k={args.vel_k}, min_vel={args.min_vel} deg/s, "
          f"resp 0..{args.resp_ms:.0f} ms")
    # no baseline subtraction needed for velocity -> pass base_ms=None to keep raw angle
    data, width_ms, ns = gc.collect(args.parent, exp_ids, ctrl_ids,
                                    args.session_match, grid, base_ms=None,
                                    min_cover=args.min_cover,
                                    use_filtered=args.use_filtered)

    per_mouse_rows = []
    permouse = {g: {"left": [], "right": []} for g in gc.GROUPS}
    for grp in gc.GROUPS:
        for side in ("left", "right"):
            for mid, W in data[grp][side]:
                lats = np.array([pulse_latency(W[r], grid, args.base_ms, args.resp_ms,
                                               args.vel_k, args.min_vel, min_dur_samp,
                                               smooth_win, args.method)
                                 for r in range(W.shape[0])])
                ndet = int(np.sum(np.isfinite(lats)))
                med = float(np.nanmedian(lats)) if ndet else np.nan
                permouse[grp][side].append(med)
                per_mouse_rows.append((grp, mid, side, med, ndet, W.shape[0]))

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_mouse_rows,
                 columns=["group", "opto", "side", "latency_ms",
                          "n_pulses_detected", "n_pulses"]).to_csv(
        out_dir / "latency_per_mouse.csv", index=False)

    # ---- bar plot: one subplot per whisker, control vs experimental ----
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, side in zip(axes, ("left", "right")):
        for xi, (grp, color) in enumerate((("control", CTRL_COLOR),
                                           ("experimental", EXP_COLOR))):
            vals = np.array(permouse[grp][side], float)
            v = vals[np.isfinite(vals)]
            mean = np.nanmean(v) if v.size else np.nan
            sem = (np.nanstd(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan
            ax.bar(xi, mean, width=0.6, color=color, alpha=0.45,
                   edgecolor=color, lw=1.5, zorder=1)
            if np.isfinite(sem):
                ax.errorbar(xi, mean, yerr=sem, color=color, lw=1.5, capsize=4, zorder=3)
            jit = (rng.random(v.size) - 0.5) * 0.22
            ax.scatter(np.full(v.size, xi) + jit, v, s=36, color=color,
                       edgecolor="k", lw=0.5, zorder=4)
            lab = f"n={v.size}/{len(vals)}"
            ax.text(xi, -0.06, lab, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, color=color)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["control", "experimental"])
        ax.set_title(f"{side.capitalize()} whisker")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.set_xlim(-0.6, 1.6)
    axes[0].set_ylabel("Movement latency (ms)")
    fig.suptitle(f"Whisker movement latency from stim onset "
                 f"({args.method}; median/mouse, mean \u00b1 SEM over mice)")
    fig.tight_layout()
    for ext in ("png", "svg"):
        p = out_dir / f"latency_bars_ctrl_vs_exp_{args.method}.{ext}"
        try:
            fig.savefig(p, dpi=150)
        except (PermissionError, OSError):
            for i in range(1, 100):
                try:
                    fig.savefig(out_dir / f"latency_bars_ctrl_vs_exp_{args.method}_{i}.{ext}"); break
                except (PermissionError, OSError):
                    continue
    plt.close(fig)
    print(f"  wrote latency_bars_ctrl_vs_exp_{args.method}.png/.svg and latency_per_mouse.csv")

    # ---- experimental-only: left vs right whisker ----
    LW, RW = "#00b140", "#d400d4"   # green left, magenta right
    fig2, ax2 = plt.subplots(figsize=(4.8, 4.6))
    for xi, (side, color) in enumerate((("left", LW), ("right", RW))):
        vals = np.array(permouse["experimental"][side], float)
        v = vals[np.isfinite(vals)]
        mean = np.nanmean(v) if v.size else np.nan
        sem = (np.nanstd(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan
        ax2.bar(xi, mean, width=0.6, color=color, alpha=0.45, edgecolor=color,
                lw=1.5, zorder=1)
        if np.isfinite(sem):
            ax2.errorbar(xi, mean, yerr=sem, color=color, lw=1.5, capsize=4, zorder=3)
        jit = (rng.random(v.size) - 0.5) * 0.22
        ax2.scatter(np.full(v.size, xi) + jit, v, s=36, color=color,
                    edgecolor="k", lw=0.5, zorder=4)
        ax2.text(xi, -0.06, f"n={v.size}/{len(vals)}",
                 transform=ax2.get_xaxis_transform(), ha="center", va="top",
                 fontsize=8, color=color)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["left whisker", "right whisker"])
    ax2.set_ylabel("Movement latency (ms)")
    ax2.set_xlim(-0.6, 1.6)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.set_title(f"Experimental: left vs right whisker latency\n"
                  f"({args.method}; median/mouse, mean ± SEM over mice)")
    fig2.tight_layout()
    for ext in ("png", "svg"):
        p = out_dir / f"latency_bars_exp_LvR_{args.method}.{ext}"
        try:
            fig2.savefig(p, dpi=150)
        except (PermissionError, OSError):
            for i in range(1, 100):
                try:
                    fig2.savefig(out_dir / f"latency_bars_exp_LvR_{args.method}_{i}.{ext}"); break
                except (PermissionError, OSError):
                    continue
    plt.close(fig2)
    print(f"  wrote latency_bars_exp_LvR_{args.method}.png/.svg")

    for grp in gc.GROUPS:
        for side in ("left", "right"):
            v = np.array(permouse[grp][side], float)
            vv = v[np.isfinite(v)]
            m = f"{np.nanmean(vv):.0f} ms" if vv.size else "no movement detected"
            print(f"  {grp:12s} {side:5s}: {m}  ({vv.size}/{len(v)} mice responded)")


if __name__ == "__main__":
    main()
