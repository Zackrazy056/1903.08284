from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ringdown.fd_likelihood import FrequencyDomainRingdownLikelihood, real_ringdown_mode_tilde
from ringdown.frequencies import kerr_qnm_omegas_22n
from ringdown.paper_fig10 import (
    MSUN_SEC,
    PaperFigure10Config,
    PaperFigure10Priors,
    build_paper_fig10_signal,
    inject_paper_fig10_noise,
    paper_fig10_signal_diagnostics,
)


@dataclass(frozen=True)
class ProfileResult:
    log_likelihood: float
    amplitudes: np.ndarray
    phases: np.ndarray
    coefficients_alpha_beta: np.ndarray


def parse_int_list(text: str) -> list[int]:
    vals = [int(s.strip()) for s in text.split(",") if s.strip()]
    if not vals:
        raise ValueError("empty integer list")
    return sorted(set(vals))


def profile_real_channel_coefficients(
    fd_like: FrequencyDomainRingdownLikelihood,
    omegas_rad_s: np.ndarray,
) -> ProfileResult:
    if fd_like.channel != "real":
        raise ValueError("profile_real_channel_coefficients requires channel='real'")

    design_cols: list[np.ndarray] = []
    for omega in omegas_rad_s:
        omega_arr = np.array([omega], dtype=complex)
        basis_alpha = real_ringdown_mode_tilde(
            fd_like.f_calc,
            omega_arr,
            np.array([1.0], dtype=float),
            np.array([0.0], dtype=float),
            duration_sec=fd_like.duration_sec,
            t0_sec=fd_like.t0_sec,
            include_finite_duration=fd_like.include_finite_duration,
        )
        basis_beta = real_ringdown_mode_tilde(
            fd_like.f_calc,
            omega_arr,
            np.array([1.0], dtype=float),
            np.array([0.5 * np.pi], dtype=float),
            duration_sec=fd_like.duration_sec,
            t0_sec=fd_like.t0_sec,
            include_finite_duration=fd_like.include_finite_duration,
        )
        design_cols.extend([basis_alpha, basis_beta])
    design = np.column_stack(design_cols)

    weight = np.sqrt((4.0 * fd_like.df) / fd_like.psd_calc)
    y = np.concatenate([weight * fd_like.d_calc.real, weight * fd_like.d_calc.imag])
    a = np.concatenate([weight[:, None] * design.real, weight[:, None] * design.imag], axis=0)
    coeffs, *_ = np.linalg.lstsq(a, y, rcond=1e-12)
    model = design @ coeffs
    d_h = 4.0 * fd_like.df * np.sum(np.real((fd_like.d_calc / fd_like.psd_calc) * np.conjugate(model)))
    h_h = 4.0 * fd_like.df * np.sum((np.abs(model) ** 2) / fd_like.psd_calc)
    logl = float(d_h - 0.5 * h_h)

    alpha = coeffs[0::2]
    beta = coeffs[1::2]
    amps = np.sqrt(alpha**2 + beta**2)
    phases = np.mod(np.arctan2(beta, alpha), 2.0 * np.pi)
    return ProfileResult(
        log_likelihood=logl,
        amplitudes=amps,
        phases=phases,
        coefficients_alpha_beta=coeffs,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sxs-location", type=str, default="SXS:BBH:0305v2.0/Lev6")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--n-values", type=str, default="0,3")
    p.add_argument("--m-total-msun", type=float, default=72.0)
    p.add_argument("--distance-mpc", type=float, default=400.0)
    p.add_argument("--delta-t0-ms", type=float, default=0.0)
    p.add_argument("--t-end", type=float, default=90.0)
    p.add_argument("--f-min-hz", type=float, default=20.0)
    p.add_argument("--f-max-hz", type=float, default=1024.0)
    p.add_argument("--df-hz", type=float, default=1.0)
    p.add_argument("--mf-min-msun", type=float, default=58.0)
    p.add_argument("--mf-max-msun", type=float, default=78.0)
    p.add_argument("--chif-min", type=float, default=0.45)
    p.add_argument("--chif-max", type=float, default=0.85)
    p.add_argument("--mf-points", type=int, default=50)
    p.add_argument("--chif-points", type=int, default=50)
    p.add_argument("--slice-n", type=int, default=3)
    p.add_argument("--slice-amp-points", type=int, default=160)
    p.add_argument("--slice-phase-points", type=int, default=180)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--output-heatmap", type=Path, default=Path("results/fig10_profile_geometry_shared_heatmap.png"))
    p.add_argument("--output-slices", type=Path, default=Path("results/fig10_profile_geometry_shared_slices.png"))
    p.add_argument("--summary-json", type=Path, default=Path("results/fig10_profile_geometry_shared.json"))
    p.add_argument("--summary-md", type=Path, default=Path("results/fig10_profile_geometry_shared.md"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n_values = parse_int_list(args.n_values)

    priors = PaperFigure10Priors()
    signal = build_paper_fig10_signal(
        PaperFigure10Config(
            sxs_location=args.sxs_location,
            total_mass_msun=args.m_total_msun,
            distance_mpc=args.distance_mpc,
            delta_t0_ms=args.delta_t0_ms,
            t_end_m=args.t_end,
            f_min_hz=args.f_min_hz,
            f_max_hz=args.f_max_hz,
            df_hz=args.df_hz,
            priors=priors,
            download=not args.no_download,
        )
    )
    observation = inject_paper_fig10_noise(signal, np.random.default_rng(args.seed))
    fd_like = FrequencyDomainRingdownLikelihood(
        freqs_hz=signal.freqs_hz,
        d_tilde=observation.d_tilde,
        psd=signal.psd,
        df=signal.config.df_hz,
        duration_sec=signal.duration_sec,
        t0_sec=0.0,
        f_min_hz=args.f_min_hz,
        f_max_hz=args.f_max_hz,
        include_finite_duration=True,
        channel="real",
    )

    mf_grid = np.linspace(args.mf_min_msun, args.mf_max_msun, args.mf_points)
    chif_grid = np.linspace(args.chif_min, args.chif_max, args.chif_points)

    summary: dict[str, object] = {
        "signal_diagnostics": paper_fig10_signal_diagnostics(signal),
        "grid": {
            "mf_min_msun": float(args.mf_min_msun),
            "mf_max_msun": float(args.mf_max_msun),
            "mf_points": int(args.mf_points),
            "chif_min": float(args.chif_min),
            "chif_max": float(args.chif_max),
            "chif_points": int(args.chif_points),
        },
        "per_n": {},
    }

    fig_heat, axes = plt.subplots(1, len(n_values), figsize=(6.4 * len(n_values), 5.0), constrained_layout=True)
    if len(n_values) == 1:
        axes = np.array([axes])

    best_profiles: dict[int, ProfileResult] = {}
    best_locations: dict[int, tuple[float, float]] = {}

    for ax, n in zip(axes, n_values):
        logl_grid = np.full((mf_grid.size, chif_grid.size), np.nan, dtype=float)
        profiles: dict[tuple[int, int], ProfileResult] = {}
        for i, mf_msun in enumerate(mf_grid):
            mf_frac = float(mf_msun / signal.config.total_mass_msun)
            for j, chif in enumerate(chif_grid):
                omegas_m = kerr_qnm_omegas_22n(mf=mf_frac, chif=float(chif), n_max=n)
                result = profile_real_channel_coefficients(fd_like, omegas_m / (MSUN_SEC * signal.config.total_mass_msun))
                profiles[(i, j)] = result
                logl_grid[i, j] = result.log_likelihood

        idx_best = np.unravel_index(int(np.nanargmax(logl_grid)), logl_grid.shape)
        best_i, best_j = int(idx_best[0]), int(idx_best[1])
        best_profile = profiles[(best_i, best_j)]
        best_mf = float(mf_grid[best_i])
        best_chif = float(chif_grid[best_j])
        best_profiles[n] = best_profile
        best_locations[n] = (best_mf, best_chif)

        truth_profile = profile_real_channel_coefficients(
            fd_like,
            kerr_qnm_omegas_22n(
                mf=float(signal.true_mf_msun / signal.config.total_mass_msun),
                chif=float(signal.true_chif),
                n_max=n,
            )
            / (MSUN_SEC * signal.config.total_mass_msun),
        )

        delta_logl = logl_grid.T - float(np.nanmax(logl_grid))
        im = ax.imshow(
            delta_logl,
            origin="lower",
            aspect="auto",
            extent=[mf_grid[0], mf_grid[-1], chif_grid[0], chif_grid[-1]],
            cmap="gist_heat_r",
            vmin=float(np.nanpercentile(delta_logl, 5)),
            vmax=0.0,
        )
        ax.axvline(signal.true_mf_msun, color="white", lw=1.0, alpha=0.85)
        ax.axhline(signal.true_chif, color="white", lw=1.0, alpha=0.85)
        ax.plot(best_mf, best_chif, marker="o", ms=4.5, color="cyan")
        ax.set_xlabel(r"$M_f\ [M_\odot]$")
        ax.set_ylabel(r"$\chi_f$")
        ax.set_title(
            rf"$N={n}$ profile $\Delta \log \mathcal{{L}}$"
            "\n"
            rf"best=({best_mf:.2f}, {best_chif:.3f})"
        )
        cbar = fig_heat.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(r"$\Delta \log \mathcal{L}$")

        summary["per_n"][str(n)] = {
            "best_mf_msun": best_mf,
            "best_chif": best_chif,
            "max_log_likelihood": float(np.nanmax(logl_grid)),
            "truth_log_likelihood": float(truth_profile.log_likelihood),
            "truth_delta_log_likelihood": float(truth_profile.log_likelihood - np.nanmax(logl_grid)),
            "best_amplitudes_rel": (best_profile.amplitudes / signal.h_peak).tolist(),
            "best_phases": best_profile.phases.tolist(),
        }

    args.output_heatmap.parent.mkdir(parents=True, exist_ok=True)
    fig_heat.savefig(args.output_heatmap, dpi=180)

    if args.slice_n not in best_profiles:
        raise ValueError(f"slice-n={args.slice_n} not found in n-values={n_values}")
    ref_profile = best_profiles[args.slice_n]
    ref_mf, ref_chif = best_locations[args.slice_n]
    omegas_ref = (
        kerr_qnm_omegas_22n(
            mf=float(ref_mf / signal.config.total_mass_msun),
            chif=float(ref_chif),
            n_max=args.slice_n,
        )
        / (MSUN_SEC * signal.config.total_mass_msun)
    )

    n_modes = args.slice_n + 1
    fig_slices, axes_slices = plt.subplots(n_modes, 2, figsize=(10.0, 2.2 * n_modes), constrained_layout=True)
    if n_modes == 1:
        axes_slices = np.array([axes_slices])
    local_summary: dict[str, object] = {
        "slice_n": int(args.slice_n),
        "reference_best_mf_msun": float(ref_mf),
        "reference_best_chif": float(ref_chif),
        "per_mode": {},
    }

    amp_ref_rel = ref_profile.amplitudes / signal.h_peak
    phi_ref = ref_profile.phases.copy()

    for k in range(n_modes):
        amp_hi = max(1.0, 2.0 * float(amp_ref_rel[k]))
        amp_grid = np.linspace(0.0, amp_hi, args.slice_amp_points)
        ll_amp = np.empty_like(amp_grid)
        for i, amp_rel in enumerate(amp_grid):
            amps = ref_profile.amplitudes.copy()
            phis = ref_profile.phases.copy()
            amps[k] = amp_rel * signal.h_peak
            ll_amp[i] = fd_like.log_likelihood(omegas_ref, amps, phis)

        dphi_grid = np.linspace(-np.pi, np.pi, args.slice_phase_points)
        ll_phi = np.empty_like(dphi_grid)
        for i, dphi in enumerate(dphi_grid):
            amps = ref_profile.amplitudes.copy()
            phis = ref_profile.phases.copy()
            phis[k] = np.mod(phi_ref[k] + dphi, 2.0 * np.pi)
            ll_phi[i] = fd_like.log_likelihood(omegas_ref, amps, phis)

        axes_slices[k, 0].plot(amp_grid, ll_amp - np.max(ll_amp), color="#1f77b4", lw=1.6)
        axes_slices[k, 0].axvline(float(amp_ref_rel[k]), color="k", ls=":", lw=1.0)
        axes_slices[k, 0].set_ylabel(rf"$\Delta \log \mathcal{{L}}$ mode {k}")
        axes_slices[k, 0].set_xlabel(rf"$A_{k}/h_{{peak}}$")
        axes_slices[k, 0].grid(True, alpha=0.15)

        axes_slices[k, 1].plot(dphi_grid, ll_phi - np.max(ll_phi), color="#d62728", lw=1.6)
        axes_slices[k, 1].axvline(0.0, color="k", ls=":", lw=1.0)
        axes_slices[k, 1].set_xlabel(rf"$\Delta \phi_{k}$")
        axes_slices[k, 1].grid(True, alpha=0.15)

        local_summary["per_mode"][str(k)] = {
            "amp_ref_rel": float(amp_ref_rel[k]),
            "phi_ref": float(phi_ref[k]),
            "amp_delta_logl_min": float(np.min(ll_amp - np.max(ll_amp))),
            "phi_delta_logl_min": float(np.min(ll_phi - np.max(ll_phi))),
        }

    axes_slices[0, 0].set_title(rf"$N={args.slice_n}$ local amplitude slices")
    axes_slices[0, 1].set_title(rf"$N={args.slice_n}$ local phase slices")
    args.output_slices.parent.mkdir(parents=True, exist_ok=True)
    fig_slices.savefig(args.output_slices, dpi=180)

    summary["local_slices"] = local_summary
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# Fig.10 Shared-Forward Profile Geometry",
        "",
        "## Signal Diagnostics",
    ]
    for key, value in paper_fig10_signal_diagnostics(signal).items():
        md_lines.append(f"- `{key}`: {value}")
    md_lines.extend(["", "## Profile Grid Results"])
    for n in n_values:
        payload = summary["per_n"][str(n)]
        md_lines.append(
            f"- `N={n}`: best `(Mf, chif)=({payload['best_mf_msun']:.4f}, {payload['best_chif']:.4f})`, "
            f"`truth_delta_logL={payload['truth_delta_log_likelihood']:.6f}`"
        )
    md_lines.extend(["", "## Local Slice Reference"])
    md_lines.append(
        f"- `N={args.slice_n}` reference best `(Mf, chif)=({ref_mf:.4f}, {ref_chif:.4f})`"
    )
    for k in range(n_modes):
        payload = local_summary["per_mode"][str(k)]
        md_lines.append(
            f"- mode `{k}`: `A/h_peak={payload['amp_ref_rel']:.6f}`, `phi={payload['phi_ref']:.6f}`, "
            f"`amp_min_dlogL={payload['amp_delta_logl_min']:.6f}`, `phi_min_dlogL={payload['phi_delta_logl_min']:.6f}`"
        )
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"output_heatmap={args.output_heatmap}")
    print(f"output_slices={args.output_slices}")
    print(f"summary_json={args.summary_json}")
    print(f"summary_md={args.summary_md}")


if __name__ == "__main__":
    main()
