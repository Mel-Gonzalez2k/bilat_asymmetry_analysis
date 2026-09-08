#!/usr/bin/env python3
r"""
For each (.h5, .mp4) recording pair in a directory, report:

  * number of camera-TTL events   (rising edges on channel 0)   -> "time points"
  * number of pulse-pal intervals (rising->falling on channel 1)
  * number of video frames in the .mp4

The .h5 stores a single bit-packed digital dataset at
"sweep_0001/digitalScans": each sample is an integer whose bit 0 is the
camera TTL and bit 1 is the pulse-pal TTL (this mirrors the channel/
transition fields in data_MRN12.json). We unpack those two bits, count
rising edges for the camera, and count high-intervals for pulse-pal.

Pairing (same as the JSON generator):
    h5  : <base>_2026-05-15_001_0001.h5
    mp4 : <base>0_noB.mp4

Usage:
    uv run extract_counts.py "E:\bilat_asymmetry_analysis\raw\Sept_test_ttl\opto_testing_ttl_alignment"

Optional:
    --data-key sweep_0001/digitalScans   (override the HDF5 dataset path)
    --cam-bit 0        (bit index for the camera TTL)
    --pulse-bit 1      (bit index for the pulse-pal TTL)
    --csv counts.csv   (also write a summary table)

Dependencies: h5py, opencv-python, numpy.
With uv you can run it standalone thanks to the inline metadata block below.
"""
# /// script
# requires-python = ">=3.9"
# dependencies = ["h5py", "numpy", "opencv-python"]
# ///

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

H5_SUFFIX = "_2026-05-15_001_0001.h5"
MP4_SUFFIX = "0_noB.mp4"
DEFAULT_KEY = "sweep_0001/digitalScans"


def h5_base(name: str):
    return name[: -len(H5_SUFFIX)] if name.endswith(H5_SUFFIX) else None


def mp4_base(name: str):
    return name[: -len(MP4_SUFFIX)] if name.endswith(MP4_SUFFIX) else None


def count_rising_edges(bit_trace: np.ndarray) -> int:
    """Number of 0->1 transitions in a 0/1 array."""
    # rising edge where previous sample is 0 and current is 1
    return int(np.count_nonzero((bit_trace[1:] == 1) & (bit_trace[:-1] == 0)))


def count_intervals(bit_trace: np.ndarray) -> int:
    """Number of contiguous high (==1) runs. Equals rising-edge count when the
    trace starts low; handles a trace that begins already-high as one interval."""
    d = np.diff(bit_trace.astype(np.int8))
    rises = int(np.count_nonzero(d == 1))
    if bit_trace.size and bit_trace[0] == 1:
        rises += 1  # trace already high at sample 0 = one interval in progress
    return rises


def read_h5_counts(h5_path: Path, data_key: str, cam_bit: int, pulse_bit: int):
    """Return (n_cam_events, n_pulse_intervals, n_samples)."""
    import h5py  # imported here so --help works without the dep installed

    with h5py.File(h5_path, "r") as f:
        if data_key not in f:
            raise KeyError(
                f"dataset '{data_key}' not found in {h5_path.name}. "
                f"Available top-level keys: {list(f.keys())}"
            )
        raw = f[data_key][()]

    raw = np.asarray(raw).ravel()
    n_samples = raw.size

    # Bit-unpack the two channels of interest.
    cam = ((raw >> cam_bit) & 1).astype(np.int8)
    pulse = ((raw >> pulse_bit) & 1).astype(np.int8)

    n_cam = count_rising_edges(cam)
    n_pulse = count_intervals(pulse)
    return n_cam, n_pulse, n_samples


def read_video_frames(mp4_path: Path) -> int:
    """Frame count from the mp4 container. Falls back to decoding if the
    container's reported count is missing or clearly wrong."""
    import cv2

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise IOError(f"could not open video {mp4_path.name}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:  # some codecs don't report a count; decode to be sure
        n = 0
        while cap.grab():
            n += 1
    cap.release()
    return n


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("directory", help="Folder containing the .h5 and .mp4 files.")
    ap.add_argument("--data-key", default=DEFAULT_KEY,
                    help=f"HDF5 dataset path (default: {DEFAULT_KEY}).")
    ap.add_argument("--cam-bit", type=int, default=0,
                    help="Bit index of the camera TTL (default 0).")
    ap.add_argument("--pulse-bit", type=int, default=1,
                    help="Bit index of the pulse-pal TTL (default 1).")
    ap.add_argument("--csv", default=None,
                    help="Optional path to write a summary CSV.")
    args = ap.parse_args()

    root = Path(args.directory)
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    names = [p.name for p in root.iterdir() if p.is_file()]
    h5_by_base = {b: n for n in names if (b := h5_base(n)) is not None}
    mp4_by_base = {b: n for n in names if (b := mp4_base(n)) is not None}
    bases = sorted(set(h5_by_base) & set(mp4_by_base))

    for b in sorted(set(h5_by_base) - set(mp4_by_base)):
        print(f"[warn] h5 with no matching mp4: {h5_by_base[b]}")
    for b in sorted(set(mp4_by_base) - set(h5_by_base)):
        print(f"[warn] mp4 with no matching h5: {mp4_by_base[b]}")

    if not bases:
        sys.exit("No matched .h5 / .mp4 pairs found.")

    rows = []
    for b in bases:
        h5_name = h5_by_base[b]
        mp4_name = mp4_by_base[b]
        print(f"\n=== {b} ===")
        try:
            n_cam, n_pulse, n_samp = read_h5_counts(
                root / h5_name, args.data_key, args.cam_bit, args.pulse_bit
            )
        except Exception as e:
            print(f"  [error reading h5] {e}")
            continue
        try:
            n_frames = read_video_frames(root / mp4_name)
        except Exception as e:
            print(f"  [error reading mp4] {e}")
            n_frames = -1

        diff = n_frames - n_cam if n_frames >= 0 else None
        print(f"  h5  : {h5_name}")
        print(f"        samples in dataset : {n_samp:,}")
        print(f"        camera TTL events (time points) : {n_cam:,}")
        print(f"        pulse-pal intervals             : {n_pulse}")
        print(f"  mp4 : {mp4_name}")
        print(f"        video frames                    : {n_frames:,}")
        if diff is not None:
            flag = "  <-- MISMATCH" if diff != 0 else "  (match)"
            print(f"        frames - cam events             : {diff:+d}{flag}")

        rows.append({
            "base": b,
            "h5": h5_name,
            "mp4": mp4_name,
            "n_samples": n_samp,
            "cam_ttl_events": n_cam,
            "pulse_pal_intervals": n_pulse,
            "video_frames": n_frames,
            "frames_minus_cam": diff,
        })

    print(f"\nProcessed {len(rows)} pair(s).")

    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote summary: {args.csv}")


if __name__ == "__main__":
    main()
