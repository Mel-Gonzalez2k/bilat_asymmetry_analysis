#!/usr/bin/env python3
r"""
Diagnostic: are the LEFT and RIGHT whisker heatmaps similar because of a bug
(same file loaded twice) or because of real synchronized bilateral whisking?

For every mouse it reports:
  * the exact CSV file used for each whisker (must be DIFFERENT files),
  * whether the two loaded traces are byte-for-byte identical (-> bug),
  * their Pearson correlation over the whole session, and separately over the
    pre-stim baseline vs the post-stim window (synchronized whisking -> high
    everywhere; a real stim effect should make them DIVERGE after onset).
It also saves one figure overlaying the raw left/right angle per mouse so you
can eyeball them.

Needs _grp_common.py in the same folder.

Usage
-----
    uv run check_left_right.py --parent "E:\...\Optogenetic_Activation" --out-dir "E:\...\group_plots"

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _grp_common as gc


def gwp_find(folder, mouse_side, use_filtered=False):
    """Replicate group_whisker_plots.py's own file-finder, so we can confirm it
    resolves to the SAME file the heatmap/latency scripts use."""
    screen = "right" if mouse_side == "left" else "left"
    angle = [Path(p) for p in glob.glob(str(Path(folder) / "*.csv"))
             if Path(p).name.lower().startswith("angle")]
    out = []
    for p in angle:
        n = p.name.lower()
        is_filt = ("filt" in n) or ("filter" in n)
        if is_filt != use_filtered:          # raw unless use_filtered
            continue
        if screen not in n:
            continue
        if screen == "right" and "left" in n:
            continue
        if mouse_side == "right" and "reverse" not in n:
            continue
        out.append(p)
    return sorted(out)[0] if out else None


def pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return np.nan
    a, b = a[m], b[m]
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--exp", default="7,12,13,14,15")
    ap.add_argument("--ctrl", default="5,6,8")
    ap.add_argument("--session-match", default="100ms")
    ap.add_argument("--win-s", type=float, default=6.0,
                    help="seconds of raw trace to draw in the overlay")
    ap.add_argument("--use-filtered", action="store_true",
                    help="use the Angle_Filt/Filter files instead of RAW (default: RAW)")
    args = ap.parse_args()

    exp_ids = [int(x) for x in args.exp.split(",") if x.strip()]
    ctrl_ids = [int(x) for x in args.ctrl.split(",") if x.strip()]
    print(f"angle source: {'FILTERED' if args.use_filtered else 'RAW (non-filtered)'}")
    rows = gc.discover_folders(args.parent, exp_ids, ctrl_ids, args.session_match,
                               args.use_filtered)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'mouse':>6} {'group':>12} {'same_file':>9} {'identical':>9} "
          f"{'plots_match':>11} {'r_all':>7} {'r_base':>7} {'r_post':>7}   files")
    print("-" * 125)

    panels = []
    n_mismatch = 0
    for grp, folder, mid in rows:
        lpath = gc.find_angle_csv(folder, "left", args.use_filtered)    # Angle_*Right*
        rpath = gc.find_angle_csv(folder, "right", args.use_filtered)   # Angle_*Left*reverse*
        # what group_whisker_plots.py would load, for the same side:
        gwp_l = gwp_find(folder, "left", args.use_filtered)
        gwp_r = gwp_find(folder, "right", args.use_filtered)
        plots_match = (gwp_l == lpath) and (gwp_r == rpath)
        if not plots_match:
            n_mismatch += 1
        same_file = (lpath == rpath)
        lf, la = gc.load_angle(lpath)
        rf, ra = gc.load_angle(rpath)

        # align right onto left's frames for a fair comparison
        ri = np.interp(lf, rf, ra, left=np.nan, right=np.nan)
        identical = bool(np.nanmax(np.abs(la - ri)) < 1e-6) if np.isfinite(ri).any() else False
        r_all = pearson(la, ri)

        # baseline vs post relative to first pulse (needs the h5)
        r_base = r_post = np.nan
        try:
            cam, ps, pe = gc.read_h5(gc.find_acq_h5(folder))
            if len(ps):
                la_abs = gc.frame_to_abs_s(lf, cam)
                t0 = ps[0]
                base = (la_abs - t0 >= -0.2) & (la_abs - t0 < 0)
                post = (la_abs - t0 >= 0) & (la_abs - t0 <= 0.3)
                r_base = pearson(la[base], ri[base])
                r_post = pearson(la[post], ri[post])
        except Exception:
            pass

        flag = "  <-- BUG" if (same_file or identical) else ""
        if not plots_match:
            flag += "  <-- PLOTS USE DIFFERENT FILES"
        print(f"{mid:>6} {grp:>12} {str(same_file):>9} {str(identical):>9} "
              f"{str(plots_match):>11} {r_all:>7.3f} {r_base:>7.3f} {r_post:>7.3f}   "
              f"{lpath.name}  |  {rpath.name}{flag}")
        panels.append((grp, mid, lf, la, ri))

    # overlay figure (experimental mice)
    exp_panels = [p for p in panels if p[0] == "experimental"]
    if exp_panels:
        n = len(exp_panels)
        fig, axes = plt.subplots(n, 1, figsize=(11, 1.8 * n), sharex=False)
        if n == 1:
            axes = [axes]
        for ax, (grp, mid, lf, la, ri) in zip(axes, exp_panels):
            k = min(len(lf), int(args.win_s * gc.FPS))
            x = np.arange(k) / gc.FPS
            ax.plot(x, la[:k], color="#00b140", lw=0.8, label="left (Angle_Right)")
            ax.plot(x, ri[:k], color="#d400d4", lw=0.8, alpha=0.8,
                    label="right (Angle_Left_reverse)")
            ax.set_ylabel(f"opto{mid}\n(deg)", fontsize=8)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            if ax is axes[0]:
                ax.legend(loc="upper right", fontsize=8, ncol=2)
        axes[-1].set_xlabel("Time (s, from start of angle file)")
        fig.suptitle("Raw left vs right whisker angle (experimental) \u2014 "
                     "identical = bug; correlated-but-distinct = synchronized whisking")
        fig.tight_layout()
        fig.savefig(out_dir / "check_left_right_overlay.png", dpi=150)
        fig.savefig(out_dir / "check_left_right_overlay.svg")
        print(f"\nwrote check_left_right_overlay.png/.svg to {out_dir}")

    print("\nHow to read this:")
    print("  same_file=True or identical=True  -> real bug (same data both sides).")
    print("  plots_match=True                  -> group_whisker_plots, heatmaps and")
    print("     latency all load the SAME file for each whisker (what you plotted).")
    print("  identical=False but r_all high    -> synchronized bilateral whisking")
    print("     (expected). If a stim effect exists, r_post should drop below r_base.")
    src = "FILTERED" if args.use_filtered else "RAW (non-filtered)"
    if n_mismatch == 0:
        print(f"\nOK: all scripts resolve to the same {src} angle files for every mouse.")
    else:
        print(f"\n[!] {n_mismatch} mouse(ren) where the plotting scripts would pick "
              f"DIFFERENT files - investigate those rows above.")


if __name__ == "__main__":
    main()
