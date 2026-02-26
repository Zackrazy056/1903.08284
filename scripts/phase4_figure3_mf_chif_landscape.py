from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ringdown.frequencies import make_omega_provider_22
from ringdown.io import load_waveform_csv, load_waveform_npz
from ringdown.metrics import remnant_error_epsilon
from ringdown.preprocess import align_to_peak
from ringdown.scan import grid_search_remnant
from ringdown.sxs_io import load_sxs_waveform22
from ringdown.types import Waveform22


def load_waveform(args: argparse.Namespace) -> tuple[Waveform22, float | None, float | None, float | None]:
    if args.input_format == "csv":
        if args.input is None:
            raise ValueError("--input is required for csv format")
        wf = load_waveform_csv(args.input)
        return wf, None, None, None

    if args.input_format == "npz":
        if args.input is None:
            raise ValueError("--input is required for npz format")
        wf = load_waveform_npz(args.input)
        return wf, None, None, None

    wf, info = load_sxs_waveform22(location=args.sxs_location, download=not args.no_download)
    return wf, info.remnant_mass, info.remnant_chif_z, info.initial_total_mass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-format", choices=["csv", "npz", "sxs"], default="sxs")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--sxs-location", type=str, default="SXS:BBH:0305/Lev6")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--n-overtones", type=int, default=7)
    parser.add_argument("--t0", type=float, default=0.0)
    parser.add_argument("--t-end", type=float, default=90.0)
    parser.add_argument("--lstsq-rcond", type=float, default=1e-12)
    parser.add_argument("--max-condition-number", type=float, default=1e12)
    parser.add_argument("--max-overtone-ratio", type=float, default=1e4)
    parser.add_argument("--min-signal-norm", type=float, default=1e-14)

    parser.add_argument("--mf-center", type=float, default=None)
    parser.add_argument("--chif-center", type=float, default=None)
    parser.add_argument("--mf-half-width", type=float, default=0.04)
    parser.add_argument("--chif-half-width", type=float, default=0.08)
    parser.add_argument("--mf-points", type=int, default=121)
    parser.add_argument("--chif-points", type=int, default=121)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figure3_mf_chif_landscape.png"),
    )
    args = parser.parse_args()

    wf_raw, mf_data, chif_data, total_mass = load_waveform(args)
    wf, t_peak_original = align_to_peak(wf_raw)

    mf_center = args.mf_center if args.mf_center is not None else mf_data
    chif_center = args.chif_center if args.chif_center is not None else chif_data
    if mf_center is None or chif_center is None:
        raise ValueError(
            "mf/chif center not available from data source; provide --mf-center and --chif-center"
        )

    mf_min = mf_center - args.mf_half_width
    mf_max = mf_center + args.mf_half_width
    chif_min = chif_center - args.chif_half_width
    chif_max = chif_center + args.chif_half_width
    mf_grid = np.linspace(mf_min, mf_max, args.mf_points)
    chif_grid = np.linspace(chif_min, chif_max, args.chif_points)

    provider = make_omega_provider_22()
    result = grid_search_remnant(
        wf=wf,
        n_overtones=args.n_overtones,
        t0=args.t0,
        mf_grid=mf_grid,
        chif_grid=chif_grid,
        omega_provider=provider,
        t_end=args.t_end,
        lstsq_rcond=args.lstsq_rcond,
        max_condition_number=args.max_condition_number,
        max_overtone_to_fund_ratio=args.max_overtone_ratio,
        min_signal_norm=args.min_signal_norm,
    )

    if not np.any(result.valid_mask):
        raise RuntimeError("no valid points survived numerical-quality filters")
    mismatch_grid = result.mismatch_grid.copy()
    mismatch_grid[result.valid_mask] = np.maximum(mismatch_grid[result.valid_mask], 1e-20)
    log_mismatch = np.full_like(mismatch_grid, np.nan, dtype=float)
    log_mismatch[result.valid_mask] = np.log10(mismatch_grid[result.valid_mask])

    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6.8))
    valid_vals = log_mismatch[np.isfinite(log_mismatch)]
    levels = np.linspace(float(valid_vals.min()), float(valid_vals.max()), 40)
    cs = plt.contourf(chif_grid, mf_grid, log_mismatch, levels=levels, cmap="viridis")
    cbar = plt.colorbar(cs)
    cbar.set_label(r"$\log_{10}\, \mathcal{M}$")
    plt.scatter([result.best_chif], [result.best_mf], c="red", s=42, label="Best fit")
    if mf_data is not None and chif_data is not None:
        plt.scatter([chif_data], [mf_data], c="white", edgecolors="black", s=42, label="NR remnant")
    plt.xlabel(r"$\chi_f$")
    plt.ylabel(r"$M_f / M$")
    plt.title(f"Mismatch landscape (N={args.n_overtones}, t0={args.t0:g})")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(args.output, dpi=180)

    print(f"source={wf.source}")
    print(f"t_peak_original={t_peak_original:.6f}")
    print(f"best_mf={result.best_mf:.9f}, best_chif={result.best_chif:.9f}")
    print(f"best_mismatch={result.best_mismatch:.6e}")
    print(f"valid_grid_points={int(np.count_nonzero(result.valid_mask))}/{result.valid_mask.size}")
    if mf_data is not None and chif_data is not None:
        mass_scale = total_mass if total_mass is not None else 1.0
        eps = remnant_error_epsilon(
            mf_fit=result.best_mf,
            mf_true=mf_data,
            chi_fit=result.best_chif,
            chi_true=chif_data,
            total_mass=mass_scale,
        )
        print(f"nr_mf={mf_data:.9f}, nr_chif={chif_data:.9f}, epsilon={eps:.6e}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
