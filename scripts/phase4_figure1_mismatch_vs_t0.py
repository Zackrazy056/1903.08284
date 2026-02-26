from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ringdown.frequencies import kerr_qnm_omegas_22n
from ringdown.io import load_waveform_csv, load_waveform_npz
from ringdown.preprocess import align_to_peak, build_start_time_grid
from ringdown.scan import scan_start_times_fixed_omegas
from ringdown.sxs_io import load_sxs_waveform22
from ringdown.types import Waveform22


def load_waveform(args: argparse.Namespace) -> tuple[Waveform22, float | None, float | None]:
    if args.input_format == "csv":
        if args.input is None:
            raise ValueError("--input is required for csv format")
        wf = load_waveform_csv(args.input)
        return wf, None, None

    if args.input_format == "npz":
        if args.input is None:
            raise ValueError("--input is required for npz format")
        wf = load_waveform_npz(args.input)
        return wf, None, None

    wf, info = load_sxs_waveform22(location=args.sxs_location, download=not args.no_download)
    return wf, info.remnant_mass, info.remnant_chif_z


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-format", choices=["csv", "npz", "sxs"], default="sxs")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--sxs-location", type=str, default="SXS:BBH:0305v2.0/Lev6")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--mf", type=float, default=None)
    parser.add_argument("--chif", type=float, default=None)
    parser.add_argument("--n-max", type=int, default=7)
    parser.add_argument("--t0-start", type=float, default=-25.0)
    parser.add_argument("--t0-end", type=float, default=60.0)
    parser.add_argument("--t0-step", type=float, default=1.0)
    parser.add_argument("--t-end", type=float, default=90.0)
    parser.add_argument("--lstsq-rcond", type=float, default=None)
    parser.add_argument("--max-condition-number", type=float, default=None)
    parser.add_argument("--max-overtone-ratio", type=float, default=None)
    parser.add_argument("--min-signal-norm", type=float, default=None)
    parser.add_argument(
        "--use-stability-guards",
        action="store_true",
        help="Enable non-paper numerical guards (condition/overtone/signal filters).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figure1_mismatch_vs_t0.png"),
    )
    args = parser.parse_args()

    if args.use_stability_guards:
        if args.lstsq_rcond is None:
            args.lstsq_rcond = 1e-12
        if args.max_condition_number is None:
            args.max_condition_number = 1e12
        if args.max_overtone_ratio is None:
            args.max_overtone_ratio = 100.0
        if args.min_signal_norm is None:
            args.min_signal_norm = 1e-14
    else:
        if args.min_signal_norm is None:
            args.min_signal_norm = 0.0

    wf, mf_from_data, chif_from_data = load_waveform(args)
    wf_peak_aligned, original_peak_time = align_to_peak(wf)

    mf = args.mf if args.mf is not None else mf_from_data
    chif = args.chif if args.chif is not None else chif_from_data
    if mf is None or chif is None:
        raise ValueError(
            "mf/chif not available from data source; provide --mf and --chif explicitly"
        )

    t0_grid = build_start_time_grid(
        t_peak=0.0,
        m_total=1.0,
        rel_start_m=args.t0_start,
        rel_end_m=args.t0_end,
        step_m=args.t0_step,
    )

    curves: list[tuple[int, np.ndarray, np.ndarray]] = []
    for n in range(args.n_max + 1):
        omegas = kerr_qnm_omegas_22n(mf=mf, chif=chif, n_max=n)
        results = scan_start_times_fixed_omegas(
            wf=wf_peak_aligned,
            omegas=omegas,
            t0_grid=t0_grid,
            t_end=args.t_end,
            lstsq_rcond=args.lstsq_rcond,
            max_condition_number=args.max_condition_number,
            max_overtone_to_fund_ratio=args.max_overtone_ratio,
            min_signal_norm=args.min_signal_norm,
        )
        if not results:
            continue
        xs = np.array([r.t0 for r in results], dtype=float)
        ys = np.array([r.mismatch for r in results], dtype=float)
        curves.append((n, xs, ys))
        best_idx = int(np.argmin(ys))
        print(
            f"N={n} valid_points={len(results)}/{len(t0_grid)} "
            f"best_t0={xs[best_idx]:.3f} best_mismatch={ys[best_idx]:.6e}"
        )

    if not curves:
        raise RuntimeError("no valid fits computed; check time window and waveform length")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))
    for n, xs, ys in curves:
        plt.semilogy(xs, ys, label=f"N={n}")
    plt.xlabel(r"$t_0 - t_{\mathrm{peak}} \ [M]$")
    plt.ylabel("Mismatch")
    plt.title(f"Mismatch vs start time ({wf.source})")
    plt.grid(True, which="both", alpha=0.2)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(args.output, dpi=180)

    print(f"source={wf.source}")
    print(f"t_peak_original={original_peak_time:.6f}")
    print(f"mf={mf:.9f}, chif={chif:.9f}")
    print(
        "fit_controls="
        f"lstsq_rcond={args.lstsq_rcond}, "
        f"max_condition_number={args.max_condition_number}, "
        f"max_overtone_ratio={args.max_overtone_ratio}, "
        f"min_signal_norm={args.min_signal_norm}"
    )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
