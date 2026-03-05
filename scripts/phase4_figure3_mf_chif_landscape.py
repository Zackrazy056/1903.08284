from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ringdown.frequencies import make_omega_provider_22
from ringdown.io import load_waveform_csv, load_waveform_npz
from ringdown.metrics import remnant_error_epsilon
from ringdown.preprocess import align_to_peak
from ringdown.scan import grid_search_remnant
from ringdown.sxs_io import load_sxs_waveform22
from ringdown.types import Waveform22


def parse_n_list(text: str) -> list[int]:
    vals: list[int] = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(int(p))
    if not vals:
        raise ValueError("empty n-overtones list")
    return sorted(set(vals))


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
    parser.add_argument(
        "--n-overtones-list",
        type=str,
        default=None,
        help="Comma-separated list, e.g. '0,7'. When set, draws multi-panel comparison.",
    )
    parser.add_argument("--t0", type=float, default=0.0)
    parser.add_argument("--t-end", type=float, default=90.0)
    parser.add_argument("--lstsq-rcond", type=float, default=1e-12)
    parser.add_argument(
        "--no-constant-offset",
        action="store_true",
        help="Disable default complex constant-offset basis term b in fitting.",
    )
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
        "--paper-fig45-style",
        action="store_true",
        help="Use paper-like panel-specific domains for N=7 (Fig.4) and N=0 (Fig.5).",
    )
    parser.add_argument("--paper-chif-min", type=float, default=0.5)
    parser.add_argument("--paper-chif-max", type=float, default=0.9)
    parser.add_argument("--paper-n7-mf-min", type=float, default=0.875)
    parser.add_argument("--paper-n7-mf-max", type=float, default=1.0)
    parser.add_argument("--paper-n0-mf-min", type=float, default=0.9)
    parser.add_argument("--paper-n0-mf-max", type=float, default=1.45)

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

    provider = make_omega_provider_22()

    n_values: Sequence[int]
    if args.n_overtones_list is not None:
        n_values = parse_n_list(args.n_overtones_list)
    else:
        n_values = [int(args.n_overtones)]

    result_by_n: dict[int, Any] = {}
    log_by_n: dict[int, np.ndarray] = {}
    mf_grid_by_n: dict[int, np.ndarray] = {}
    chif_grid_by_n: dict[int, np.ndarray] = {}
    for n in n_values:
        if args.paper_fig45_style:
            if n == 7:
                mf_min = args.paper_n7_mf_min
                mf_max = args.paper_n7_mf_max
            elif n == 0:
                mf_min = args.paper_n0_mf_min
                mf_max = args.paper_n0_mf_max
            else:
                mf_min = mf_center - args.mf_half_width
                mf_max = mf_center + args.mf_half_width
            chif_min = args.paper_chif_min
            chif_max = args.paper_chif_max
        else:
            mf_min = mf_center - args.mf_half_width
            mf_max = mf_center + args.mf_half_width
            chif_min = chif_center - args.chif_half_width
            chif_max = chif_center + args.chif_half_width
        mf_grid = np.linspace(mf_min, mf_max, args.mf_points)
        chif_grid = np.linspace(chif_min, chif_max, args.chif_points)

        result = grid_search_remnant(
            wf=wf,
            n_overtones=n,
            t0=args.t0,
            mf_grid=mf_grid,
            chif_grid=chif_grid,
            omega_provider=provider,
            t_end=args.t_end,
            lstsq_rcond=args.lstsq_rcond,
            include_constant_offset=not args.no_constant_offset,
            max_condition_number=args.max_condition_number,
            max_overtone_to_fund_ratio=args.max_overtone_ratio,
            min_signal_norm=args.min_signal_norm,
        )
        if not np.any(result.valid_mask):
            raise RuntimeError(f"no valid points survived filters for N={n}")
        mismatch_grid = result.mismatch_grid.copy()
        mismatch_grid[result.valid_mask] = np.maximum(mismatch_grid[result.valid_mask], 1e-20)
        log_mismatch = np.full_like(mismatch_grid, np.nan, dtype=float)
        log_mismatch[result.valid_mask] = np.log10(mismatch_grid[result.valid_mask])
        result_by_n[n] = result
        log_by_n[n] = log_mismatch
        mf_grid_by_n[n] = mf_grid
        chif_grid_by_n[n] = chif_grid

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        len(n_values),
        figsize=(8.0 * len(n_values), 6.6),
        squeeze=False,
        constrained_layout=True,
    )
    axes1 = axes[0]
    for idx, n in enumerate(n_values):
        ax = axes1[idx]
        log_mismatch = log_by_n[n]
        result = result_by_n[n]
        mf_grid = mf_grid_by_n[n]
        chif_grid = chif_grid_by_n[n]
        valid_vals = log_mismatch[np.isfinite(log_mismatch)]
        vmin = float(np.min(valid_vals))
        vmax = float(np.max(valid_vals))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        levels = np.linspace(vmin, vmax, 40)
        cs = ax.contourf(chif_grid, mf_grid, log_mismatch, levels=levels, cmap="viridis")
        ax.scatter([result.best_chif], [result.best_mf], c="red", s=42, label="Best fit")
        if mf_data is not None and chif_data is not None:
            # Draw NR truth as crosshairs for precise visual anchoring.
            ax.axvline(float(chif_data), color="black", lw=2.0, alpha=0.45, zorder=5)
            ax.axhline(float(mf_data), color="black", lw=2.0, alpha=0.45, zorder=5)
            ax.axvline(float(chif_data), color="white", lw=1.2, alpha=0.95, zorder=6)
            ax.axhline(float(mf_data), color="white", lw=1.2, alpha=0.95, zorder=6)
        ax.set_xlabel(r"$\chi_f$")
        if idx == 0:
            ax.set_ylabel(r"$M_f / M$")
        ax.set_xlim(float(chif_grid[0]), float(chif_grid[-1]))
        ax.set_ylim(float(mf_grid[0]), float(mf_grid[-1]))
        ax.set_title(f"N={n}, t0={args.t0:g}")
        ax.grid(True, alpha=0.15)
        cbar = fig.colorbar(cs, ax=ax, shrink=0.92, pad=0.02)
        cbar.set_label(r"$\log_{10}\, \mathcal{M}$")
    axes1[0].legend(loc="best", fontsize=9)
    fig.suptitle("Mismatch landscape", fontsize=12)
    fig.savefig(args.output, dpi=180)

    print(f"source={wf.source}")
    print(f"t_peak_original={t_peak_original:.6f}")
    print(f"include_constant_offset={not args.no_constant_offset}")
    print(f"paper_fig45_style={args.paper_fig45_style}")
    for n in n_values:
        result = result_by_n[n]
        mf_grid = mf_grid_by_n[n]
        chif_grid = chif_grid_by_n[n]
        print(f"N={n} best_mf={result.best_mf:.9f}, best_chif={result.best_chif:.9f}")
        print(f"N={n} best_mismatch={result.best_mismatch:.6e}")
        print(f"N={n} valid_grid_points={int(np.count_nonzero(result.valid_mask))}/{result.valid_mask.size}")
        print(
            f"N={n} panel_range: mf=[{float(mf_grid[0]):.3f},{float(mf_grid[-1]):.3f}], "
            f"chif=[{float(chif_grid[0]):.3f},{float(chif_grid[-1]):.3f}]"
        )
        if mf_data is not None and chif_data is not None:
            mass_scale = total_mass if total_mass is not None else 1.0
            eps = remnant_error_epsilon(
                mf_fit=result.best_mf,
                mf_true=mf_data,
                chi_fit=result.best_chif,
                chi_true=chif_data,
                total_mass=mass_scale,
            )
            print(f"N={n} nr_mf={mf_data:.9f}, nr_chif={chif_data:.9f}, epsilon={eps:.6e}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
