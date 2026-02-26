from __future__ import annotations

import argparse
from pathlib import Path

from ringdown.io import load_waveform_csv, load_waveform_npz
from ringdown.preprocess import (
    align_to_peak,
    build_start_time_grid,
    crop_time,
    resample_uniform,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--format", choices=["csv", "npz"], required=True)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--crop-min", type=float, default=-50.0)
    parser.add_argument("--crop-max", type=float, default=100.0)
    args = parser.parse_args()

    if args.format == "csv":
        wf = load_waveform_csv(args.input)
    else:
        wf = load_waveform_npz(args.input)

    wf_aligned, t_peak_original = align_to_peak(wf)
    wf_crop = crop_time(wf_aligned, args.crop_min, args.crop_max)
    wf_uniform = resample_uniform(wf_crop, args.dt)
    t0_grid = build_start_time_grid(
        t_peak=0.0, m_total=1.0, rel_start_m=-25.0, rel_end_m=60.0, step_m=1.0
    )

    print(f"source={wf.source}")
    print(f"n_raw={wf.t.size}")
    print(f"t_peak_original={t_peak_original:.6f}")
    print(f"n_crop={wf_crop.t.size}")
    print(f"n_uniform={wf_uniform.t.size}, dt={args.dt}")
    print(f"t_range_uniform=[{wf_uniform.t[0]:.3f}, {wf_uniform.t[-1]:.3f}]")
    print(f"start_time_grid_size={t0_grid.size}")
    print(f"start_time_grid_first_last=({t0_grid[0]:.3f}, {t0_grid[-1]:.3f})")


if __name__ == "__main__":
    main()

