from __future__ import annotations

import argparse
import json
from pathlib import Path

import dynesty
import matplotlib.pyplot as plt
import numpy as np
from dynesty.utils import resample_equal

from ringdown.fd_likelihood import FrequencyDomainRingdownLikelihood, real_ringdown_mode_tilde
from ringdown.frequencies import kerr_qnm_omegas_22n
from ringdown.metrics import remnant_error_epsilon
from ringdown.paper_fig10 import (
    MSUN_SEC,
    PaperFigure10Config,
    PaperFigure10Priors,
    build_paper_fig10_signal,
    inject_paper_fig10_noise,
    paper_fig10_signal_diagnostics,
)


N_OVERTONES = 3
N_MODES = N_OVERTONES + 1
TWO_PI = 2.0 * np.pi


def normalize_1d_pdf(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    z = float(np.trapezoid(p, x))
    return p / z if z > 0 else p


def credible_level_2d(pdf: np.ndarray, cred: float = 0.9) -> float:
    flat = pdf.ravel()
    order = np.argsort(flat)[::-1]
    cdf = np.cumsum(flat[order])
    idx = int(np.searchsorted(cdf, cred, side="left"))
    idx = min(max(idx, 0), order.size - 1)
    return float(flat[order[idx]])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sxs-location", type=str, default="SXS:BBH:0305v2.0/Lev6")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--m-total-msun", type=float, default=72.0)
    p.add_argument("--distance-mpc", type=float, default=400.0)
    p.add_argument("--delta-t0-ms", type=float, default=0.0)
    p.add_argument("--t-end", type=float, default=90.0)
    p.add_argument("--f-min-hz", type=float, default=20.0)
    p.add_argument("--f-max-hz", type=float, default=1024.0)
    p.add_argument("--df-hz", type=float, default=1.0)
    p.add_argument("--mf-min-msun", type=float, default=10.0)
    p.add_argument("--mf-max-msun", type=float, default=100.0)
    p.add_argument("--chif-min", type=float, default=0.0)
    p.add_argument("--chif-max", type=float, default=1.0)
    p.add_argument("--profile-mf-min-msun", type=float, default=64.0)
    p.add_argument("--profile-mf-max-msun", type=float, default=76.0)
    p.add_argument("--profile-chif-min", type=float, default=0.64)
    p.add_argument("--profile-chif-max", type=float, default=0.76)
    p.add_argument("--profile-mf-points", type=int, default=30)
    p.add_argument("--profile-chif-points", type=int, default=30)
    p.add_argument("--truth-epsilon-threshold", type=float, default=0.03)
    p.add_argument("--dynesty-nlive", type=int, default=120)
    p.add_argument("--dynesty-bound", choices=["multi", "single", "balls", "cubes"], default="single")
    p.add_argument("--dynesty-sample", choices=["auto", "unif", "rwalk", "slice", "rslice", "hslice"], default="rwalk")
    p.add_argument("--dynesty-walks", type=int, default=16)
    p.add_argument("--dynesty-slices", type=int, default=8)
    p.add_argument("--dynesty-bootstrap", type=int, default=0)
    p.add_argument("--dynesty-update-interval", type=float, default=0.6)
    p.add_argument("--dynesty-dlogz", type=float, default=1.5)
    p.add_argument("--dynesty-maxcall", type=int, default=60000)
    p.add_argument("--posterior-ess-target", type=float, default=80.0)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--output", type=Path, default=Path("results/fig10_n3_logrel_relphase_candidate.png"))
    p.add_argument("--trace-output", type=Path, default=Path("results/fig10_n3_logrel_relphase_candidate_traces.png"))
    p.add_argument("--local-slices-output", type=Path, default=Path("results/fig10_n3_logrel_relphase_candidate_slices.png"))
    p.add_argument("--summary-json", type=Path, default=Path("results/fig10_n3_logrel_relphase_candidate.json"))
    p.add_argument("--summary-md", type=Path, default=Path("results/fig10_n3_logrel_relphase_candidate.md"))
    p.add_argument("--samples-output", type=Path, default=Path("results/fig10_n3_logrel_relphase_candidate_samples.npz"))
    return p.parse_args()


def qnm_interp_tables(chif_bounds: tuple[float, float]) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    chi_interp = np.linspace(max(chif_bounds[0], 0.0), min(chif_bounds[1], 0.999), 260)
    qnm_table_re: dict[int, np.ndarray] = {}
    qnm_table_im: dict[int, np.ndarray] = {}
    for k in range(N_MODES):
        vals = np.array([kerr_qnm_omegas_22n(mf=1.0, chif=float(c), n_max=N_OVERTONES)[k] for c in chi_interp], dtype=complex)
        qnm_table_re[k] = vals.real
        qnm_table_im[k] = vals.imag
    return chi_interp, qnm_table_re, qnm_table_im


def omegas_from_mf_chif(
    mf_msun: float,
    chif: float,
    *,
    total_mass_msun: float,
    chi_interp: np.ndarray,
    qnm_table_re: dict[int, np.ndarray],
    qnm_table_im: dict[int, np.ndarray],
) -> np.ndarray:
    mf_frac = float(mf_msun / total_mass_msun)
    omegas = np.empty(N_MODES, dtype=complex)
    for k in range(N_MODES):
        wr = np.interp(chif, chi_interp, qnm_table_re[k])
        wi = np.interp(chif, chi_interp, qnm_table_im[k])
        omegas[k] = (wr + 1j * wi) / mf_frac
    return omegas


def profile_real_channel_coefficients(
    fd_like: FrequencyDomainRingdownLikelihood,
    omegas_rad_s: np.ndarray,
) -> float:
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
    return float(d_h - 0.5 * h_h)


def decode_logrel_relphase(theta: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    mf_msun = float(theta[0])
    chif = float(theta[1])
    log_a0 = float(theta[2])
    log_ratios = np.asarray(theta[3 : 3 + N_OVERTONES], dtype=float)
    phi0 = float(theta[3 + N_OVERTONES])
    dphis = np.asarray(theta[4 + N_OVERTONES : 4 + 2 * N_OVERTONES], dtype=float)

    a0 = 10.0**log_a0
    ratios = 10.0**log_ratios
    amp_rel = np.empty(N_MODES, dtype=float)
    amp_rel[0] = a0
    amp_rel[1:] = a0 * ratios
    phases = np.empty(N_MODES, dtype=float)
    phases[0] = np.mod(phi0, TWO_PI)
    phases[1:] = np.mod(phi0 + dphis, TWO_PI)
    return mf_msun, chif, amp_rel, phases


def make_prior_transform(
    *,
    mf_bounds: tuple[float, float],
    chif_bounds: tuple[float, float],
    amp_bounds_rel: tuple[float, float],
    phi_bounds: tuple[float, float],
) -> tuple[int, callable]:
    ndim = 2 + 1 + N_OVERTONES + 1 + N_OVERTONES
    log_amp_lo = float(np.log10(amp_bounds_rel[0]))
    log_amp_hi = float(np.log10(amp_bounds_rel[1]))
    log_ratio_lo = -2.0
    log_ratio_hi = 2.0

    def prior_transform(u: np.ndarray) -> np.ndarray:
        theta = np.empty(ndim, dtype=float)
        theta[0] = mf_bounds[0] + u[0] * (mf_bounds[1] - mf_bounds[0])
        theta[1] = chif_bounds[0] + u[1] * (chif_bounds[1] - chif_bounds[0])
        theta[2] = log_amp_lo + u[2] * (log_amp_hi - log_amp_lo)
        theta[3 : 3 + N_OVERTONES] = log_ratio_lo + u[3 : 3 + N_OVERTONES] * (log_ratio_hi - log_ratio_lo)
        theta[3 + N_OVERTONES] = phi_bounds[0] + u[3 + N_OVERTONES] * (phi_bounds[1] - phi_bounds[0])
        theta[4 + N_OVERTONES : 4 + 2 * N_OVERTONES] = (
            phi_bounds[0]
            + u[4 + N_OVERTONES : 4 + 2 * N_OVERTONES] * (phi_bounds[1] - phi_bounds[0])
        )
        return theta

    return ndim, prior_transform


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    priors = PaperFigure10Priors(
        mf_bounds_msun=(float(args.mf_min_msun), float(args.mf_max_msun)),
        chif_bounds=(float(args.chif_min), float(args.chif_max)),
    )
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
    observation = inject_paper_fig10_noise(signal, rng)
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

    chi_interp, qnm_table_re, qnm_table_im = qnm_interp_tables((args.chif_min, min(args.chif_max, 0.999)))
    m_sec = MSUN_SEC * args.m_total_msun

    # Local profile reference around the truth-adjacent basin.
    prof_mf = np.linspace(args.profile_mf_min_msun, args.profile_mf_max_msun, args.profile_mf_points)
    prof_chif = np.linspace(args.profile_chif_min, args.profile_chif_max, args.profile_chif_points)
    profile_grid = np.full((prof_mf.size, prof_chif.size), np.nan, dtype=float)
    profile_max_logl = -np.inf
    best_profile_mf = float("nan")
    best_profile_chif = float("nan")
    for i, mf in enumerate(prof_mf):
        for j, chif in enumerate(prof_chif):
            omegas = omegas_from_mf_chif(
                mf,
                float(chif),
                total_mass_msun=args.m_total_msun,
                chi_interp=chi_interp,
                qnm_table_re=qnm_table_re,
                qnm_table_im=qnm_table_im,
            )
            logl = profile_real_channel_coefficients(fd_like, omegas / m_sec)
            profile_grid[i, j] = logl
            if logl > profile_max_logl:
                profile_max_logl = logl
                best_profile_mf = float(mf)
                best_profile_chif = float(chif)

    ndim, prior_transform = make_prior_transform(
        mf_bounds=priors.mf_bounds_msun,
        chif_bounds=(args.chif_min, min(args.chif_max, 0.999)),
        amp_bounds_rel=priors.amp_bounds_rel,
        phi_bounds=priors.phi_bounds,
    )

    def log_likelihood(theta: np.ndarray) -> float:
        mf_msun, chif, amp_rel, phases = decode_logrel_relphase(theta)
        if not (priors.mf_bounds_msun[0] <= mf_msun <= priors.mf_bounds_msun[1]):
            return -np.inf
        if not (args.chif_min <= chif <= min(args.chif_max, 0.999)):
            return -np.inf
        if np.any((amp_rel < priors.amp_bounds_rel[0]) | (amp_rel > priors.amp_bounds_rel[1])):
            return -np.inf
        omegas = omegas_from_mf_chif(
            mf_msun,
            chif,
            total_mass_msun=args.m_total_msun,
            chi_interp=chi_interp,
            qnm_table_re=qnm_table_re,
            qnm_table_im=qnm_table_im,
        )
        return float(
            fd_like.log_likelihood(
                omegas_rad_s=omegas / m_sec,
                amplitudes=amp_rel * signal.h_peak,
                phases=phases,
            )
        )

    dynesty_kwargs: dict[str, object] = dict(
        bound=args.dynesty_bound,
        sample=args.dynesty_sample,
        bootstrap=args.dynesty_bootstrap,
        update_interval=args.dynesty_update_interval,
        rstate=np.random.default_rng(args.seed + 2026),
    )
    if args.dynesty_sample == "rwalk":
        dynesty_kwargs["walks"] = args.dynesty_walks
    if args.dynesty_sample in {"slice", "rslice", "hslice"}:
        dynesty_kwargs["slices"] = args.dynesty_slices

    sampler = dynesty.NestedSampler(
        loglikelihood=log_likelihood,
        prior_transform=prior_transform,
        ndim=ndim,
        nlive=args.dynesty_nlive,
        **dynesty_kwargs,
    )
    run_kwargs: dict[str, object] = dict(dlogz=args.dynesty_dlogz, print_progress=False)
    if args.dynesty_maxcall > 0:
        run_kwargs["maxcall"] = args.dynesty_maxcall
    sampler.run_nested(**run_kwargs)
    res = sampler.results

    samples_raw = np.asarray(res.samples, dtype=float)
    logwt = np.asarray(res.logwt, dtype=float)
    logl = np.asarray(res.logl, dtype=float)
    logz_final = float(res.logz[-1])
    w = np.exp(logwt - logz_final)
    w = np.clip(w, 0.0, np.inf)
    w = w / np.sum(w)
    ess_kish = float(1.0 / np.sum(w**2))
    samples_eq = resample_equal(samples_raw, w)

    phys = np.array([decode_logrel_relphase(s)[:2] for s in samples_eq], dtype=float)
    mf_s = phys[:, 0]
    chif_s = phys[:, 1]
    amp_phys = np.array([decode_logrel_relphase(s)[2] for s in samples_eq], dtype=float)

    idx_map = int(np.argmax(logl))
    theta_map = samples_raw[idx_map]
    map_mf, map_chif, map_amp_rel, map_phases = decode_logrel_relphase(theta_map)
    map_eps = remnant_error_epsilon(map_mf, signal.true_mf_msun, map_chif, signal.true_chif, args.m_total_msun)
    q50_mf = float(np.quantile(mf_s, 0.5))
    q50_chif = float(np.quantile(chif_s, 0.5))
    q50_eps = remnant_error_epsilon(q50_mf, signal.true_mf_msun, q50_chif, signal.true_chif, args.m_total_msun)
    truth_ball_frac = float(
        np.mean(
            [
                remnant_error_epsilon(mf_v, signal.true_mf_msun, ch_v, signal.true_chif, args.m_total_msun)
                < args.truth_epsilon_threshold
                for mf_v, ch_v in zip(mf_s, chif_s)
            ]
        )
    )

    eff_arr = np.asarray(res.eff).reshape(-1)
    eff_percent = float(eff_arr[-1]) if eff_arr.size > 0 else float("nan")
    converged = bool(float(res.logzerr[-1]) <= args.dynesty_dlogz and ess_kish >= args.posterior_ess_target)
    max_logl = float(np.max(logl))
    gap_to_profile_max = float(profile_max_logl - max_logl)

    # Posterior summary figure
    mf_grid = np.linspace(args.mf_min_msun, args.mf_max_msun, 220)
    chif_grid = np.linspace(args.chif_min, min(args.chif_max, 0.999), 220)
    hist2d, _, _ = np.histogram2d(mf_s, chif_s, bins=[mf_grid, chif_grid], density=False)
    pdf2d = hist2d.T
    pdf2d /= np.sum(pdf2d)
    level90 = credible_level_2d(pdf2d, 0.9)
    mf_centers = 0.5 * (mf_grid[:-1] + mf_grid[1:])
    chif_centers = 0.5 * (chif_grid[:-1] + chif_grid[1:])
    post_mf = normalize_1d_pdf(mf_centers, np.histogram(mf_s, bins=mf_grid, density=False)[0].astype(float))
    post_chif = normalize_1d_pdf(chif_centers, np.histogram(chif_s, bins=chif_grid, density=False)[0].astype(float))

    fig = plt.figure(figsize=(9, 8))
    gs = fig.add_gridspec(4, 4, hspace=0.05, wspace=0.05)
    ax_top = fig.add_subplot(gs[0, 0:3])
    ax_main = fig.add_subplot(gs[1:4, 0:3], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
    ax_main.contour(mf_centers, chif_centers, pdf2d, levels=[level90], colors=["#d62728"], linewidths=2.0)
    ax_top.plot(mf_centers, post_mf, color="#d62728", lw=1.8)
    ax_right.plot(post_chif, chif_centers, color="#d62728", lw=1.8)
    ax_main.axvline(signal.true_mf_msun, color="k", ls=":", lw=1.2)
    ax_main.axhline(signal.true_chif, color="k", ls=":", lw=1.2)
    ax_main.plot(map_mf, map_chif, marker="o", ms=4.5, color="cyan")
    ax_main.set_xlabel(r"$M_f\ [M_\odot]$")
    ax_main.set_ylabel(r"$\chi_f$")
    ax_main.set_xlim(args.mf_min_msun, args.mf_max_msun)
    ax_main.set_ylim(args.chif_min, min(args.chif_max, 0.999))
    ax_main.grid(True, alpha=0.15)
    ax_top.set_ylabel("Posterior")
    ax_top.grid(True, alpha=0.15)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_right.set_xlabel("Posterior")
    ax_right.grid(True, alpha=0.15)
    ax_right.tick_params(axis="y", labelleft=False)
    ax_top.set_title(
        rf"N=3 logrel_relphase candidate ($\Delta t_0={args.delta_t0_ms:.3f}$ ms)"
        "\n"
        rf"gap={gap_to_profile_max:.2f}, MAP $\epsilon$={map_eps:.3f}, ESS={ess_kish:.1f}"
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)

    # Transformed-coordinate local slices around the MAP point.
    fig_s, axes = plt.subplots(2, 4, figsize=(12, 6.5), constrained_layout=True)
    axes = axes.ravel()
    labels = [r"$\log_{10} A_0$", r"$\log_{10}(A_1/A_0)$", r"$\log_{10}(A_2/A_0)$", r"$\log_{10}(A_3/A_0)$", r"$\phi_0$", r"$\Delta\phi_1$", r"$\Delta\phi_2$", r"$\Delta\phi_3$"]
    centers = np.array(theta_map[2:], dtype=float)
    widths = np.array([0.35, 0.45, 0.45, 0.45, 0.7, 0.7, 0.7, 0.7], dtype=float)
    slice_min_dlogl: dict[str, float] = {}
    for i, ax in enumerate(axes):
        grid = np.linspace(centers[i] - widths[i], centers[i] + widths[i], 160)
        ll = np.empty_like(grid)
        for j, val in enumerate(grid):
            theta = np.array(theta_map, copy=True)
            theta[2 + i] = val
            ll[j] = log_likelihood(theta)
        dlogl = ll - np.max(ll)
        ax.plot(grid, dlogl, color="#1f77b4", lw=1.6)
        ax.axvline(centers[i], color="k", ls=":", lw=1.0)
        ax.set_title(labels[i], fontsize=10)
        ax.grid(True, alpha=0.15)
        slice_min_dlogl[str(i)] = float(np.min(dlogl))
    axes[0].set_ylabel(r"$\Delta \log \mathcal{L}$")
    axes[4].set_ylabel(r"$\Delta \log \mathcal{L}$")
    args.local_slices_output.parent.mkdir(parents=True, exist_ok=True)
    fig_s.savefig(args.local_slices_output, dpi=180)

    # Traces
    fig_t, axes_t = plt.subplots(ndim, 1, figsize=(10, 1.55 * ndim), sharex=True)
    if ndim == 1:
        axes_t = [axes_t]
    trace_labels = ["$M_f$", "$\\chi_f$", "$\\log A_0$", "$\\log R_1$", "$\\log R_2$", "$\\log R_3$", "$\\phi_0$", "$\\Delta\\phi_1$", "$\\Delta\\phi_2$", "$\\Delta\\phi_3$"]
    for p in range(ndim):
        axes_t[p].plot(samples_raw[:, p], lw=0.4, alpha=0.8)
        axes_t[p].set_ylabel(trace_labels[p], fontsize=8)
        axes_t[p].grid(True, alpha=0.15)
    axes_t[-1].set_xlabel("Nested sample index")
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    fig_t.savefig(args.trace_output, dpi=180)

    args.samples_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.samples_output,
        samples_equal=samples_eq,
        samples_raw=samples_raw,
        weights=w,
        mf=mf_s,
        chif=chif_s,
    )

    summary = {
        "signal_diagnostics": paper_fig10_signal_diagnostics(signal),
        "profile_reference": {
            "profile_max_logl": float(profile_max_logl),
            "best_profile_mf_msun": float(best_profile_mf),
            "best_profile_chif": float(best_profile_chif),
        },
        "run_diagnostics": {
            "niter": int(res.niter),
            "ncall_total": int(np.sum(np.asarray(res.ncall))),
            "eff_percent": eff_percent,
            "logz": logz_final,
            "logzerr": float(res.logzerr[-1]),
            "ess_kish": ess_kish,
            "converged": int(converged),
            "max_logl": max_logl,
            "gap_to_profile_max": gap_to_profile_max,
            "map_mf_msun": float(map_mf),
            "map_chif": float(map_chif),
            "map_epsilon": float(map_eps),
            "q50_mf_msun": q50_mf,
            "q50_chif": q50_chif,
            "q50_epsilon": float(q50_eps),
            "truth_ball_frac": truth_ball_frac,
            "amp_min_rel": float(np.min(amp_phys)),
            "amp_max_rel": float(np.max(amp_phys)),
        },
        "map_point_transformed": {
            "log10_A0": float(theta_map[2]),
            "log10_R1": float(theta_map[3]),
            "log10_R2": float(theta_map[4]),
            "log10_R3": float(theta_map[5]),
            "phi0": float(theta_map[6]),
            "dphi1": float(theta_map[7]),
            "dphi2": float(theta_map[8]),
            "dphi3": float(theta_map[9]),
        },
        "map_point_physical": {
            "amp_rel": map_amp_rel.tolist(),
            "phases": map_phases.tolist(),
        },
        "local_slice_min_dlogl": slice_min_dlogl,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# Fig.10 N=3 Logrel-Relphase Candidate",
        "",
        "## Signal Diagnostics",
    ]
    for key, value in paper_fig10_signal_diagnostics(signal).items():
        md_lines.append(f"- `{key}`: {value}")
    md_lines.extend(
        [
            "",
            "## Profile Reference",
            f"- `profile_max_logl`: {profile_max_logl}",
            f"- `best_profile_(Mf, chif)`: ({best_profile_mf:.6f}, {best_profile_chif:.6f})",
            "",
            "## Run Diagnostics",
            f"- `gap_to_profile_max`: {gap_to_profile_max:.6f}",
            f"- `ess_kish`: {ess_kish:.6f}",
            f"- `logzerr`: {float(res.logzerr[-1]):.6f}",
            f"- `converged`: {int(converged)}",
            f"- `map_(Mf, chif)`: ({map_mf:.6f}, {map_chif:.6f})",
            f"- `map_epsilon`: {map_eps:.6f}",
            f"- `q50_(Mf, chif)`: ({q50_mf:.6f}, {q50_chif:.6f})",
            f"- `q50_epsilon`: {q50_eps:.6f}",
            f"- `truth_ball_frac`: {truth_ball_frac:.6f}",
            "",
            "## Local Slice Readout",
        ]
    )
    for key, value in slice_min_dlogl.items():
        md_lines.append(f"- param `{key}` min `Delta logL`: {value}")
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"profile_max_logl={profile_max_logl:.6f}")
    print(f"gap_to_profile_max={gap_to_profile_max:.6f}")
    print(f"ess_kish={ess_kish:.6f}")
    print(f"map_epsilon={map_eps:.6f}")
    print(f"truth_ball_frac={truth_ball_frac:.6f}")
    print(f"output={args.output}")
    print(f"trace_output={args.trace_output}")
    print(f"local_slices_output={args.local_slices_output}")
    print(f"summary_json={args.summary_json}")
    print(f"summary_md={args.summary_md}")


if __name__ == "__main__":
    main()
