from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ringdown.compare import align_time_and_phase_by_window, interp_complex
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


def infer_reference_location(main_location: str) -> str | None:
    if "Lev6" in main_location:
        return main_location.replace("Lev6", "Lev5")
    return None


def estimate_noise_safe_t_end(
    wf_main_peak: Waveform22,
    wf_ref_peak: Waveform22,
    *,
    requested_t_end: float,
    ratio_threshold: float,
    min_time: float,
    align_start: float,
    align_end: float,
    smooth_window_m: float,
) -> tuple[float, dict[str, float]]:
    """
    Estimate a conservative t_end by requiring |h_main|/|h_main-h_ref| above threshold.
    """
    if ratio_threshold <= 0:
        raise ValueError("ratio_threshold must be positive")

    t_ref_aligned, h_ref_aligned, best_dt, best_phase = align_time_and_phase_by_window(
        t_reference=wf_main_peak.t,
        h_reference=wf_main_peak.h,
        t_target=wf_ref_peak.t,
        h_target=wf_ref_peak.h,
        t_start=align_start,
        t_end=align_end,
    )

    valid = (wf_main_peak.t >= t_ref_aligned[0]) & (wf_main_peak.t <= t_ref_aligned[-1])
    t = wf_main_peak.t[valid]
    h_main = wf_main_peak.h[valid]
    h_ref = interp_complex(t_ref_aligned, h_ref_aligned, t)

    ratio = np.abs(h_main) / (np.abs(h_main - h_ref) + 1e-30)
    ratio = np.clip(ratio, 1e-30, None)

    if t.size >= 2 and smooth_window_m > 0:
        dt = float(np.median(np.diff(t)))
        window_n = max(1, int(round(smooth_window_m / dt)))
    else:
        window_n = 1

    if window_n > 1:
        kernel = np.ones(window_n, dtype=float) / float(window_n)
        log_ratio_smooth = np.convolve(np.log10(ratio), kernel, mode="same")
    else:
        log_ratio_smooth = np.log10(ratio)

    threshold_log = float(np.log10(ratio_threshold))
    mask = (t >= min_time) & (t <= requested_t_end) & (log_ratio_smooth >= threshold_log)
    if not np.any(mask):
        return requested_t_end, {
            "best_dt": float(best_dt),
            "best_phase": float(best_phase),
            "t_est": float("nan"),
            "ratio_threshold": float(ratio_threshold),
            "min_time": float(min_time),
            "window_n": float(window_n),
            "status": "fallback_requested_t_end",
        }

    t_est = float(np.max(t[mask]))
    return min(requested_t_end, t_est), {
        "best_dt": float(best_dt),
        "best_phase": float(best_phase),
        "t_est": float(t_est),
        "ratio_threshold": float(ratio_threshold),
        "min_time": float(min_time),
        "window_n": float(window_n),
        "status": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-format", choices=["csv", "npz", "sxs"], default="sxs")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--sxs-location", type=str, default="SXS:BBH:0305v2.0/Lev6")
    parser.add_argument("--sxs-reference-location", type=str, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--mf", type=float, default=None)
    parser.add_argument("--chif", type=float, default=None)
    parser.add_argument("--n-max", type=int, default=7)
    parser.add_argument("--t0-start", type=float, default=-25.0)
    parser.add_argument("--t0-end", type=float, default=60.0)
    parser.add_argument("--t0-step", type=float, default=1.0)
    parser.add_argument("--t-end", type=float, default=90.0)
    parser.add_argument("--adaptive-t-end-from-reference", action="store_true")
    parser.add_argument("--noise-ratio-threshold", type=float, default=300.0)
    parser.add_argument("--noise-min-time", type=float, default=40.0)
    parser.add_argument("--noise-align-start", type=float, default=0.0)
    parser.add_argument("--noise-align-end", type=float, default=20.0)
    parser.add_argument("--noise-smooth-window", type=float, default=2.0)
    parser.add_argument("--lstsq-rcond", type=float, default=None)
    parser.add_argument(
        "--no-constant-offset",
        action="store_true",
        help="Disable default complex constant-offset basis term b in fitting.",
    )
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
    effective_t_end = float(args.t_end)

    mf = args.mf if args.mf is not None else mf_from_data
    chif = args.chif if args.chif is not None else chif_from_data
    if mf is None or chif is None:
        raise ValueError(
            "mf/chif not available from data source; provide --mf and --chif explicitly"
        )

    if args.adaptive_t_end_from_reference:
        if args.input_format != "sxs":
            raise ValueError("--adaptive-t-end-from-reference currently supports only --input-format sxs")

        ref_location = args.sxs_reference_location
        if ref_location is None:
            ref_location = infer_reference_location(args.sxs_location)
        if ref_location is None:
            raise ValueError("cannot infer --sxs-reference-location from --sxs-location; provide explicitly")

        wf_ref, _, _ = load_waveform(
            argparse.Namespace(
                input_format="sxs",
                input=None,
                sxs_location=ref_location,
                no_download=args.no_download,
            )
        )
        wf_ref_peak, _ = align_to_peak(wf_ref)
        effective_t_end, meta = estimate_noise_safe_t_end(
            wf_main_peak=wf_peak_aligned,
            wf_ref_peak=wf_ref_peak,
            requested_t_end=float(args.t_end),
            ratio_threshold=float(args.noise_ratio_threshold),
            min_time=float(args.noise_min_time),
            align_start=float(args.noise_align_start),
            align_end=float(args.noise_align_end),
            smooth_window_m=float(args.noise_smooth_window),
        )
        print(
            "adaptive_t_end="
            f"requested={args.t_end:.3f}, estimated={effective_t_end:.3f}, "
            f"status={meta['status']}, threshold={meta['ratio_threshold']:.3f}, "
            f"align_dt={meta['best_dt']:.6f}, align_phase={meta['best_phase']:.6f}"
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
            t_end=effective_t_end,
            lstsq_rcond=args.lstsq_rcond,
            include_constant_offset=not args.no_constant_offset,
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
    print(f"t_end_effective={effective_t_end:.6f}")
    print(
        "fit_controls="
        f"lstsq_rcond={args.lstsq_rcond}, "
        f"include_constant_offset={not args.no_constant_offset}, "
        f"max_condition_number={args.max_condition_number}, "
        f"max_overtone_ratio={args.max_overtone_ratio}, "
        f"min_signal_norm={args.min_signal_norm}"
    )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
