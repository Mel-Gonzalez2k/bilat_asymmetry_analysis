#!/usr/bin/env python3
r"""
Per-pulse whisker-angle HEATMAPS (like panels D/E), control vs experimental.

Four heatmaps are produced -- left/right whisker x control/experimental -- each
with rows = every stimulation pulse pooled across that group's mice, columns =
time from pulse onset (-pre .. +post ms), colour = baseline-subtracted whisker
angle (deg). All four share ONE colour scale (computed across all four, or set
with --vmin/--vmax) so control and experimental are directly comparable even
though they are separate files.

Needs _grp_common.py in the same folder.

Usage
-----
    uv run group_heatmaps.py --parent "E:\...\Optogenetic_Activation" --out-dir "E:\...\group_plots"
    uv run group_heatmaps.py --parent "..." --pre-ms 200 --post-ms 400
    uv run group_heatmaps.py --parent "..." --vmin -3 --vmax 5 --cmap viridis

Dependencies: h5py, numpy, pandas, matplotlib.
"""
# /// script
# requires-python = ">=3.9"
# dependencies = ["h5py", "numpy", "pandas", "matplotlib"]
# ///

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _grp_common as gc


def pooled(data, grp, side):
    """Stack all pulses across the group's mice -> (n_trials, n_grid) + mouse sizes.
    Mice with zero covered pulses are skipped (no empty blocks)."""
    mats, sizes, ids = [], [], []
    for mid, W in data[grp][side]:
        if W.shape[0] == 0:
            continue
        mats.append(W); sizes.append(W.shape[0]); ids.append(mid)
    if not mats:
        return np.empty((0, 0)), [], []
    return np.vstack(mats), sizes, ids


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
    ap.add_argument("--base-ms", type=float, default=200.0,
                    help="baseline window before onset to subtract per pulse")
    ap.add_argument("--min-cover", type=float, default=0.5,
                    help="keep a pulse only if >= this fraction of samples is present "
                         "in each of before/during/after windows (0.5 = 50%%)")
    ap.add_argument("--use-filtered", action="store_true",
                    help="use the Angle_Filt/Filter files instead of RAW (default: RAW)")
    ap.add_argument("--vmin", type=float, default=None, help="shared colour min (deg)")
    ap.add_argument("--vmax", type=float, default=None, help="shared colour max (deg)")
    ap.add_argument("--pctl", type=float, default=98.0,
                    help="symmetric colour limit percentile when vmin/vmax not given")
    ap.add_argument("--cmap", default="RdBu_r",
                    help="colormap (RdBu_r = blue retraction / red protraction; "
                         "or viridis/turbo to match the paper)")
    args = ap.parse_args()

    exp_ids = [int(x) for x in args.exp.split(",") if x.strip()]
    ctrl_ids = [int(x) for x in args.ctrl.split(",") if x.strip()]
    dt = 1000.0 / gc.FPS
    grid = np.arange(-args.pre_ms, args.post_ms + dt, dt) / 1000.0  # seconds

    print(f"Heatmaps: window -{args.pre_ms:.0f}..+{args.post_ms:.0f} ms, "
          f"rows = pulses pooled per group")
    data, width_ms, ns = gc.collect(args.parent, exp_ids, ctrl_ids,
                                     args.session_match, grid, args.base_ms,
                                     min_cover=args.min_cover,
                                     use_filtered=args.use_filtered)

    # shared colour scale across all 4
    panels = {}
    allvals = []
    for grp in gc.GROUPS:
        for side in ("left", "right"):
            M, sizes, ids = pooled(data, grp, side)
            panels[(grp, side)] = (M, sizes, ids)
            if M.size:
                allvals.append(M[np.isfinite(M)].ravel())
    if args.vmin is not None and args.vmax is not None:
        vmin, vmax = args.vmin, args.vmax
    else:
        a = np.concatenate(allvals) if allvals else np.array([0.0, 1.0])
        L = float(np.nanpercentile(np.abs(a), args.pctl))
        vmin, vmax = -L, L
    print(f"shared colour scale: [{vmin:.2f}, {vmax:.2f}] deg (cmap {args.cmap})")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tgrid_ms = grid * 1000.0

    for grp in gc.GROUPS:
        for side in ("left", "right"):
            M, sizes, ids = panels[(grp, side)]
            fig, ax = plt.subplots(figsize=(5.2, 5.0))
            if M.size == 0:
                ax.text(0.5, 0.5, f"no {grp} data", ha="center", va="center")
            else:
                im = ax.imshow(M, aspect="auto", cmap=args.cmap, vmin=vmin, vmax=vmax,
                               extent=[tgrid_ms[0], tgrid_ms[-1], M.shape[0], 0],
                               interpolation="nearest")
                # stim pulse window + onset
                ax.axvline(0, color="k", lw=1.0)
                ax.axvline(width_ms, color="k", lw=1.0, ls="--", alpha=0.6)
                # mouse boundaries + one centered "mouse N" tick per block
                y = 0
                centers, labels = [], []
                for s, mid in zip(sizes, ids):
                    centers.append(y + s / 2.0)
                    labels.append(f"mouse {mid}")
                    y += s
                    if y < M.shape[0]:
                        ax.axhline(y, color="w", lw=1.0, alpha=0.8)
                ax.set_yticks(centers)
                ax.set_yticklabels(labels, fontsize=9)
                ax.tick_params(axis="y", length=0)
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.set_label("Angle (deg)  \u2212 retraction / + protraction")
            ax.set_xlabel("Time from stim onset (ms)")
            ax.set_ylabel("")
            ax.set_title(f"{side.capitalize()} whisker \u2014 {grp}\n"
                         f"n={ns[grp]} mice, {M.shape[0] if M.size else 0} pulses")
            fig.tight_layout()
            stem = f"heatmap_{side}_whisker_{grp}"
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

    # a small text file recording the shared scale
    (out_dir / "heatmap_colorscale.txt").write_text(
        f"shared color scale: vmin={vmin:.3f}, vmax={vmax:.3f} deg, cmap={args.cmap}\n"
        f"window: -{args.pre_ms:.0f}..+{args.post_ms:.0f} ms, baseline -{args.base_ms:.0f} ms\n")
    print(f"\nDone. Shared scale saved to heatmap_colorscale.txt")


if __name__ == "__main__":
    main()
