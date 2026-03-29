from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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


TWO_PI = 2.0 * np.pi
N_OVERTONES = 3
N_MODES = N_OVERTONES + 1


@dataclass(frozen=True)
class ComparisonResult:
    name: str
    niter: int
    ncall_total: int
    eff_percent: float
    logz: float
    logzerr: float
    ess_kish: float
    converged: bool
    max_logl: float
    gap_to_profile_max: float
    map_mf_msun: float
    map_chif: float
    map_epsilon: float
    q50_mf_msun: float
    q50_chif: float
    q50_epsilon: float
    truth_ball_frac: float
    amp_max_rel: float
    amp_min_rel: float


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
    p.add_argument("--profile-mf-points", type=int, default=28)
    p.add_argument("--profile-chif-points", type=int, default=28)
    p.add_argument("--truth-epsilon-threshold", type=float, default=0.03)
    p.add_argument("--parametrizations", type=str, default="baseline,logamp,relphase,logrel_relphase")
    p.add_argument("--dynesty-nlive", type=int, default=80)
    p.add_argument("--dynesty-bound", choices=["multi", "single", "balls", "cubes"], default="single")
    p.add_argument("--dynesty-sample", choices=["auto", "unif", "rwalk", "slice", "rslice", "hslice"], default="rwalk")
    p.add_argument("--dynesty-walks", type=int, default=16)
    p.add_argument("--dynesty-slices", type=int, default=8)
    p.add_argument("--dynesty-bootstrap", type=int, default=0)
    p.add_argument("--dynesty-update-interval", type=float, default=0.6)
    p.add_argument("--dynesty-dlogz", type=float, default=2.5)
    p.add_argument("--dynesty-maxcall", type=int, default=35000)
    p.add_argument("--posterior-ess-target", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--output-plot", type=Path, default=Path("results/fig10_n3_reparam_compare.png"))
    p.add_argument("--summary-json", type=Path, default=Path("results/fig10_n3_reparam_compare.json"))
    p.add_argument("--summary-md", type=Path, default=Path("results/fig10_n3_reparam_compare.md"))
    return p.parse_args()


def parse_parametrizations(text: str) -> list[str]:
    items = [s.strip() for s in text.split(",") if s.strip()]
    if not items:
        raise ValueError("empty parametrizations list")
    valid = {"baseline", "logamp", "relphase", "logrel_relphase"}
    unknown = [s for s in items if s not in valid]
    if unknown:
        raise ValueError(f"unknown parametrizations: {unknown}")
    return items


def qnm_interp_tables(chif_bounds: tuple[float, float]) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    chi_interp = np.linspace(max(chif_bounds[0], 0.0), min(chif_bounds[1], 0.999), 260)
    qnm_table_re: dict[int, np.ndarray] = {}
    qnm_table_im: dict[int, np.ndarray] = {}
    for k in range(N_MODES):
        vals = kerr_qnm_omegas_22n(mf=1.0, chif=float(chi_interp[0]), n_max=N_OVERTONES)
        break
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


def decode_physical_params(name: str, theta: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    mf_msun = float(theta[0])
    chif = float(theta[1])
    if name == "baseline":
        amp_rel = np.asarray(theta[2 : 2 + N_MODES], dtype=float)
        phases = np.mod(np.asarray(theta[2 + N_MODES : 2 + 2 * N_MODES], dtype=float), TWO_PI)
        return mf_msun, chif, amp_rel, phases
    if name == "logamp":
        log_amp = np.asarray(theta[2 : 2 + N_MODES], dtype=float)
        phases = np.mod(np.asarray(theta[2 + N_MODES : 2 + 2 * N_MODES], dtype=float), TWO_PI)
        return mf_msun, chif, 10.0**log_amp, phases
    if name == "relphase":
        amp_rel = np.asarray(theta[2 : 2 + N_MODES], dtype=float)
        phi0 = float(theta[2 + N_MODES])
        dphis = np.asarray(theta[3 + N_MODES : 2 + 2 * N_MODES], dtype=float)
        phases = np.empty(N_MODES, dtype=float)
        phases[0] = np.mod(phi0, TWO_PI)
        phases[1:] = np.mod(phi0 + dphis, TWO_PI)
        return mf_msun, chif, amp_rel, phases
    if name == "logrel_relphase":
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
    raise ValueError(f"unsupported parametrization: {name}")


def prior_transform_factory(
    name: str,
    *,
    mf_bounds: tuple[float, float],
    chif_bounds: tuple[float, float],
    amp_bounds_rel: tuple[float, float],
    phi_bounds: tuple[float, float],
) -> tuple[int, callable]:
    log_amp_lo = float(np.log10(amp_bounds_rel[0]))
    log_amp_hi = float(np.log10(amp_bounds_rel[1]))
    log_ratio_lo = -2.0
    log_ratio_hi = 2.0

    if name == "baseline":
        ndim = 2 + 2 * N_MODES

        def prior_transform(u: np.ndarray) -> np.ndarray:
            theta = np.empty(ndim, dtype=float)
            theta[0] = mf_bounds[0] + u[0] * (mf_bounds[1] - mf_bounds[0])
            theta[1] = chif_bounds[0] + u[1] * (chif_bounds[1] - chif_bounds[0])
            theta[2 : 2 + N_MODES] = amp_bounds_rel[0] + u[2 : 2 + N_MODES] * (amp_bounds_rel[1] - amp_bounds_rel[0])
            theta[2 + N_MODES : 2 + 2 * N_MODES] = phi_bounds[0] + u[2 + N_MODES : 2 + 2 * N_MODES] * (phi_bounds[1] - phi_bounds[0])
            return theta

        return ndim, prior_transform

    if name == "logamp":
        ndim = 2 + 2 * N_MODES

        def prior_transform(u: np.ndarray) -> np.ndarray:
            theta = np.empty(ndim, dtype=float)
            theta[0] = mf_bounds[0] + u[0] * (mf_bounds[1] - mf_bounds[0])
            theta[1] = chif_bounds[0] + u[1] * (chif_bounds[1] - chif_bounds[0])
            theta[2 : 2 + N_MODES] = log_amp_lo + u[2 : 2 + N_MODES] * (log_amp_hi - log_amp_lo)
            theta[2 + N_MODES : 2 + 2 * N_MODES] = phi_bounds[0] + u[2 + N_MODES : 2 + 2 * N_MODES] * (phi_bounds[1] - phi_bounds[0])
            return theta

        return ndim, prior_transform

    if name == "relphase":
        ndim = 2 + N_MODES + 1 + N_OVERTONES

        def prior_transform(u: np.ndarray) -> np.ndarray:
            theta = np.empty(ndim, dtype=float)
            theta[0] = mf_bounds[0] + u[0] * (mf_bounds[1] - mf_bounds[0])
            theta[1] = chif_bounds[0] + u[1] * (chif_bounds[1] - chif_bounds[0])
            theta[2 : 2 + N_MODES] = amp_bounds_rel[0] + u[2 : 2 + N_MODES] * (amp_bounds_rel[1] - amp_bounds_rel[0])
            theta[2 + N_MODES] = phi_bounds[0] + u[2 + N_MODES] * (phi_bounds[1] - phi_bounds[0])
            theta[3 + N_MODES : 3 + N_MODES + N_OVERTONES] = phi_bounds[0] + u[3 + N_MODES : 3 + N_MODES + N_OVERTONES] * (phi_bounds[1] - phi_bounds[0])
            return theta

        return ndim, prior_transform

    if name == "logrel_relphase":
        ndim = 2 + 1 + N_OVERTONES + 1 + N_OVERTONES

        def prior_transform(u: np.ndarray) -> np.ndarray:
            theta = np.empty(ndim, dtype=float)
            theta[0] = mf_bounds[0] + u[0] * (mf_bounds[1] - mf_bounds[0])
            theta[1] = chif_bounds[0] + u[1] * (chif_bounds[1] - chif_bounds[0])
            theta[2] = log_amp_lo + u[2] * (log_amp_hi - log_amp_lo)
            theta[3 : 3 + N_OVERTONES] = log_ratio_lo + u[3 : 3 + N_OVERTONES] * (log_ratio_hi - log_ratio_lo)
            theta[3 + N_OVERTONES] = phi_bounds[0] + u[3 + N_OVERTONES] * (phi_bounds[1] - phi_bounds[0])
            theta[4 + N_OVERTONES : 4 + 2 * N_OVERTONES] = phi_bounds[0] + u[4 + N_OVERTONES : 4 + 2 * N_OVERTONES] * (phi_bounds[1] - phi_bounds[0])
            return theta

        return ndim, prior_transform

    raise ValueError(f"unsupported parametrization: {name}")


def main() -> None:
    args = parse_args()
    parametrizations = parse_parametrizations(args.parametrizations)
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

    # Profile-likelihood reference for the same observation.
    prof_mf = np.linspace(args.profile_mf_min_msun, args.profile_mf_max_msun, args.profile_mf_points)
    prof_chif = np.linspace(args.profile_chif_min, args.profile_chif_max, args.profile_chif_points)
    profile_max_logl = -np.inf
    for mf in prof_mf:
        for chif in prof_chif:
            omegas = omegas_from_mf_chif(
                mf,
                float(chif),
                total_mass_msun=args.m_total_msun,
                chi_interp=chi_interp,
                qnm_table_re=qnm_table_re,
                qnm_table_im=qnm_table_im,
            )
            logl = profile_real_channel_coefficients(fd_like, omegas / m_sec)
            if logl > profile_max_logl:
                profile_max_logl = logl

    results: list[ComparisonResult] = []
    summary_payload: dict[str, object] = {
        "signal_diagnostics": paper_fig10_signal_diagnostics(signal),
        "profile_reference_max_logl": float(profile_max_logl),
        "parametrizations": {},
    }

    for idx, name in enumerate(parametrizations):
        ndim, prior_transform = prior_transform_factory(
            name,
            mf_bounds=priors.mf_bounds_msun,
            chif_bounds=(args.chif_min, min(args.chif_max, 0.999)),
            amp_bounds_rel=priors.amp_bounds_rel,
            phi_bounds=priors.phi_bounds,
        )

        def log_likelihood(theta: np.ndarray) -> float:
            mf_msun, chif, amp_rel, phases = decode_physical_params(name, theta)
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
            rstate=np.random.default_rng(args.seed + 1000 * (idx + 1)),
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

        phys = np.array([decode_physical_params(name, s)[:2] for s in samples_eq], dtype=float)
        mf_s = phys[:, 0]
        chif_s = phys[:, 1]
        amp_phys = np.array([decode_physical_params(name, s)[2] for s in samples_eq], dtype=float)

        idx_map = int(np.argmax(logl))
        map_mf, map_chif, map_amp_rel, _ = decode_physical_params(name, samples_raw[idx_map])
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

        result = ComparisonResult(
            name=name,
            niter=int(res.niter),
            ncall_total=int(np.sum(np.asarray(res.ncall))),
            eff_percent=eff_percent,
            logz=logz_final,
            logzerr=float(res.logzerr[-1]),
            ess_kish=ess_kish,
            converged=converged,
            max_logl=max_logl,
            gap_to_profile_max=gap_to_profile_max,
            map_mf_msun=float(map_mf),
            map_chif=float(map_chif),
            map_epsilon=float(map_eps),
            q50_mf_msun=q50_mf,
            q50_chif=q50_chif,
            q50_epsilon=float(q50_eps),
            truth_ball_frac=truth_ball_frac,
            amp_max_rel=float(np.max(amp_phys)),
            amp_min_rel=float(np.min(amp_phys)),
        )
        results.append(result)

        summary_payload["parametrizations"][name] = {
            "ndim": int(ndim),
            "niter": result.niter,
            "ncall_total": result.ncall_total,
            "eff_percent": result.eff_percent,
            "logz": result.logz,
            "logzerr": result.logzerr,
            "ess_kish": result.ess_kish,
            "converged": int(result.converged),
            "max_logl": result.max_logl,
            "gap_to_profile_max": result.gap_to_profile_max,
            "map_mf_msun": result.map_mf_msun,
            "map_chif": result.map_chif,
            "map_epsilon": result.map_epsilon,
            "q50_mf_msun": result.q50_mf_msun,
            "q50_chif": result.q50_chif,
            "q50_epsilon": result.q50_epsilon,
            "truth_ball_frac": result.truth_ball_frac,
            "amp_min_rel": result.amp_min_rel,
            "amp_max_rel": result.amp_max_rel,
        }
        print(
            f"{name}: ess={result.ess_kish:.1f} logzerr={result.logzerr:.3f} "
            f"max_logl={result.max_logl:.3f} gap_to_profile={result.gap_to_profile_max:.3f} "
            f"map_eps={result.map_epsilon:.4f} truth_ball_frac={result.truth_ball_frac:.4f}"
        )

    names = [r.name for r in results]
    x = np.arange(len(results))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    axes = axes.ravel()

    axes[0].bar(x, [r.gap_to_profile_max for r in results], color="#1f77b4")
    axes[0].set_title("Gap To Profile Max")
    axes[0].set_ylabel(r"$\log \mathcal{L}_{\rm profile,max} - \log \mathcal{L}_{\rm sample,max}$")
    axes[0].set_xticks(x, names, rotation=20)

    axes[1].bar(x, [r.ess_kish for r in results], color="#2ca02c")
    axes[1].axhline(args.posterior_ess_target, color="k", ls=":", lw=1.0)
    axes[1].set_title("Posterior ESS")
    axes[1].set_ylabel("Kish ESS")
    axes[1].set_xticks(x, names, rotation=20)

    axes[2].bar(x, [r.map_epsilon for r in results], color="#d62728")
    axes[2].set_title("MAP Epsilon")
    axes[2].set_ylabel(r"$\epsilon$")
    axes[2].set_xticks(x, names, rotation=20)

    axes[3].bar(x, [r.truth_ball_frac for r in results], color="#9467bd")
    axes[3].set_title("Truth-Ball Fraction")
    axes[3].set_ylabel(rf"Frac($\epsilon < {args.truth_epsilon_threshold:.3f}$)")
    axes[3].set_xticks(x, names, rotation=20)

    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_plot, dpi=180)

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    best_gap = min(results, key=lambda r: r.gap_to_profile_max)
    best_truth = max(results, key=lambda r: r.truth_ball_frac)
    best_map = min(results, key=lambda r: r.map_epsilon)
    truth_best_label = best_truth.name if best_truth.truth_ball_frac > 0.0 else "none (all 0)"

    md_lines = [
        "# Fig.10 N=3 Reparameterization Compare",
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
            "",
            "## Per-Parameterization Results",
        ]
    )
    for r in results:
        md_lines.append(
            f"- `{r.name}`: `gap_to_profile={r.gap_to_profile_max:.6f}`, `ess={r.ess_kish:.2f}`, "
            f"`map_epsilon={r.map_epsilon:.6f}`, `truth_ball_frac={r.truth_ball_frac:.6f}`, "
            f"`q50_epsilon={r.q50_epsilon:.6f}`, `converged={int(r.converged)}`"
        )
    md_lines.extend(
        [
            "",
            "## Best By Metric",
            f"- smallest profile gap: `{best_gap.name}`",
            f"- largest truth-ball fraction: `{truth_best_label}`",
            f"- smallest MAP epsilon: `{best_map.name}`",
        ]
    )
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"output_plot={args.output_plot}")
    print(f"summary_json={args.summary_json}")
    print(f"summary_md={args.summary_md}")


if __name__ == "__main__":
    main()
