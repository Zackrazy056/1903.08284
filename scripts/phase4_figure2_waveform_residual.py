from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ringdown.compare import (
    align_time_and_phase_by_window,
    interp_complex,
    phase_align_to_reference_at_tref,
    window_waveform,
)
from ringdown.frequencies import kerr_qnm_omegas_22n
from ringdown.io import load_waveform_csv, load_waveform_npz
from ringdown.preprocess import align_to_peak
from ringdown.scan import fit_at_start_time
from ringdown.sxs_io import SXSRemnantInfo, load_sxs_waveform22
from ringdown.types import Waveform22


def load_waveform_main(
    args: argparse.Namespace,
) -> tuple[Waveform22, float | None, float | None, float | None]:
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


def try_load_reference(
    args: argparse.Namespace,
) -> tuple[Waveform22 | None, SXSRemnantInfo | None]:
    if args.input_format != "sxs":
        return None, None
    if args.sxs_reference_location is None:
        return None, None
    wf_ref, info_ref = load_sxs_waveform22(
        location=args.sxs_reference_location, download=not args.no_download
    )
    return wf_ref, info_ref


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-format", choices=["csv", "npz", "sxs"], default="sxs")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--sxs-location", type=str, default="SXS:BBH:0305v2.0/Lev6")
    parser.add_argument("--sxs-reference-location", type=str, default="SXS:BBH:0305v2.0/Lev5")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--mf", type=float, default=None)
    parser.add_argument("--chif", type=float, default=None)
    parser.add_argument("--n-overtones", type=int, default=7)
    parser.add_argument("--t0", type=float, default=0.0)
    parser.add_argument("--t-end", type=float, default=90.0)
    parser.add_argument(
        "--ref-align-mode",
        choices=["point_phase", "window_time_phase"],
        default="window_time_phase",
    )
    parser.add_argument("--ref-align-window", type=float, default=20.0)
    parser.add_argument("--ref-dt-half-width", type=float, default=2.0)
    parser.add_argument("--ref-dt-step", type=float, default=0.002)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figure2_waveform_residual.png"),
    )
    args = parser.parse_args()

    wf_raw, mf_data, chif_data, _ = load_waveform_main(args)
    wf, t_peak_original = align_to_peak(wf_raw)

    mf = args.mf if args.mf is not None else mf_data
    chif = args.chif if args.chif is not None else chif_data
    if mf is None or chif is None:
        raise ValueError(
            "mf/chif not available from data source; provide --mf and --chif explicitly"
        )

    omegas = kerr_qnm_omegas_22n(mf=mf, chif=chif, n_max=args.n_overtones)
    fit_result, lin_result = fit_at_start_time(
        wf=wf, omegas=omegas, t0=args.t0, t_end=args.t_end
    )

    t_win, h_win = window_waveform(wf=wf, t0=args.t0, t_end=args.t_end)
    h_model = lin_result.model
    residual = np.abs(h_win - h_model)

    wf_ref_raw, _ = try_load_reference(args)
    ref_t = None
    ref_residual = None
    best_dt = None
    best_phase = None
    if wf_ref_raw is not None:
        wf_ref, _ = align_to_peak(wf_ref_raw)
        if args.ref_align_mode == "point_phase":
            t_ref_aligned = wf_ref.t
            h_ref_aligned = phase_align_to_reference_at_tref(
                t_ref=args.t0,
                t_reference=t_win,
                h_reference=h_win,
                t_target=wf_ref.t,
                h_target=wf_ref.h,
            )
            best_dt = 0.0
            best_phase = float(
                np.angle(
                    interp_complex(t_win, h_win, np.array([args.t0], dtype=float))[0]
                    / interp_complex(wf_ref.t, wf_ref.h, np.array([args.t0], dtype=float))[0]
                )
            )
        else:
            t_align_end = min(args.t_end, args.t0 + args.ref_align_window)
            t_ref_aligned, h_ref_aligned, best_dt, best_phase = align_time_and_phase_by_window(
                t_reference=t_win,
                h_reference=h_win,
                t_target=wf_ref.t,
                h_target=wf_ref.h,
                t_start=args.t0,
                t_end=t_align_end,
                dt_search_half_width=args.ref_dt_half_width,
                dt_step=args.ref_dt_step,
            )

        valid = (t_win >= t_ref_aligned[0]) & (t_win <= t_ref_aligned[-1])
        if np.any(valid):
            ref_t = t_win[valid]
            h_ref_on_grid = interp_complex(t_ref_aligned, h_ref_aligned, ref_t)
            ref_residual = np.abs(h_win[valid] - h_ref_on_grid)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    axes[0].plot(t_win, h_win.real, label=r"$h_{22}^{\mathrm{NR}}$ (Re)", lw=1.6)
    axes[0].plot(t_win, h_model.real, label=rf"$h_{{22}}^{{N={args.n_overtones}}}$ (Re)", lw=1.2)
    axes[0].set_ylabel("Strain")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(True, alpha=0.2)

    eps = 1e-16
    axes[1].semilogy(t_win, np.maximum(residual, eps), label=r"$|h_{22}^{\mathrm{NR}}-h_{22}^{\mathrm{model}}|$")
    if ref_t is not None and ref_residual is not None:
        axes[1].semilogy(
            ref_t,
            np.maximum(ref_residual, eps),
            label=r"$|h_{22}^{\mathrm{Lev6}}-h_{22}^{\mathrm{Lev5}}|$",
            alpha=0.9,
        )
    axes[1].set_xlabel(r"$t - t_{\mathrm{peak}} \ [M]$")
    axes[1].set_ylabel("Residual")
    axes[1].grid(True, which="both", alpha=0.2)
    axes[1].legend(loc="best", fontsize=9)

    fig.suptitle(
        f"Waveform fit and residuals ({wf.source}, N={args.n_overtones}, t0={args.t0:g})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)

    print(f"source={wf.source}")
    print(f"t_peak_original={t_peak_original:.6f}")
    print(f"mf={mf:.9f}, chif={chif:.9f}")
    print(f"n_overtones={args.n_overtones}")
    print(f"fit_mismatch={fit_result.mismatch:.6e}")
    print(f"fit_residual_norm={fit_result.residual_norm:.6e}")
    print(
        "model_residual_stats="
        f"(min={residual.min():.6e}, median={np.median(residual):.6e}, max={residual.max():.6e})"
    )
    if ref_residual is not None and ref_residual.size > 0:
        print(
            "nr_error_proxy_stats="
            f"(min={ref_residual.min():.6e}, median={np.median(ref_residual):.6e}, max={ref_residual.max():.6e})"
        )
    if best_dt is not None and best_phase is not None:
        print(f"ref_align_mode={args.ref_align_mode}, ref_best_dt={best_dt:.6f}, ref_best_phase={best_phase:.6f}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
