from __future__ import annotations

import argparse
from pathlib import Path

import dynesty
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from dynesty.utils import resample_equal

from ringdown.fd_likelihood import (
    FrequencyDomainRingdownLikelihood,
    aligo_zero_det_high_power_psd,
    continuous_ft_from_time_series,
    draw_colored_noise_rfft,
    optimal_snr,
)
from ringdown.frequencies import kerr_qnm_omega_lmn
from ringdown.preprocess import align_to_peak
from ringdown.sxs_io import load_sxs_waveform22


MSUN_SEC = 4.92549095e-6
AMP_PRIOR_MIN_REL = 0.01
AMP_PRIOR_MAX_REL = 250.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sxs-location", type=str, default="SXS:BBH:0305v2.0/Lev6")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--n-values", type=str, default="0,1,2,3")
    parser.add_argument("--m-total-msun", type=float, default=72.0)
    parser.add_argument("--distance-mpc", type=float, default=400.0)
    parser.add_argument("--target-hpeak", type=float, default=2e-21)
    parser.add_argument("--target-snr-postpeak", type=float, default=42.3)
    parser.add_argument(
        "--likelihood-domain",
        choices=["frequency", "time"],
        default="frequency",
        help="Use PSD-weighted frequency-domain likelihood (default) or legacy time-domain white-noise likelihood.",
    )
    parser.add_argument(
        "--psd-model",
        choices=["aligo_design", "flat"],
        default="aligo_design",
        help="PSD model for frequency-domain injection/likelihood.",
    )
    parser.add_argument("--f-min-hz", type=float, default=20.0, help="Minimum frequency for PSD-weighted inner products.")
    parser.add_argument("--f-max-hz", type=float, default=1024.0, help="Maximum frequency for PSD-weighted inner products.")
    parser.add_argument(
        "--fd-grid-df-hz",
        type=float,
        default=1.0,
        help="Frequency spacing for frequency-domain likelihood grid (Hz).",
    )
    parser.add_argument(
        "--disable-psd-snr-rescale",
        action="store_true",
        help="Disable amplitude rescaling that enforces target post-peak SNR under PSD-weighted inner product.",
    )
    parser.add_argument(
        "--disable-finite-duration-model",
        action="store_true",
        help="Use infinite-duration analytic FT i*C/(2*pi*f + omega) instead of finite-window model.",
    )
    parser.add_argument(
        "--strain-channel",
        choices=["complex", "plus"],
        default="complex",
        help="Data/likelihood channel: complex h22 or detector-like plus polarization only.",
    )
    parser.add_argument(
        "--t0-reference",
        choices=["complex_peak", "plus_peak"],
        default="complex_peak",
        help="Reference peak used for t0; paper-style setting is complex_peak.",
    )
    parser.add_argument("--delta-t0-ms", type=float, default=0.0)
    parser.add_argument("--t-end", type=float, default=90.0)
    parser.add_argument("--mf-min-msun", type=float, default=50.0)
    parser.add_argument("--mf-max-msun", type=float, default=100.0)
    parser.add_argument("--chif-min", type=float, default=0.0)
    parser.add_argument("--chif-max", type=float, default=1.0)
    parser.add_argument("--boundary-eps-frac", type=float, default=0.001)

    parser.add_argument("--dynesty-mode", choices=["static", "dynamic"], default="static")
    parser.add_argument("--dynesty-nlive", type=int, default=1200)
    parser.add_argument("--dynesty-nlive-init", type=int, default=800)
    parser.add_argument("--dynesty-nlive-batch", type=int, default=400)
    parser.add_argument("--dynesty-maxbatch", type=int, default=0, help="0 means no explicit maxbatch bound (dynamic mode).")
    parser.add_argument("--dynesty-n-effective", type=float, default=0.0, help="0 means disabled.")
    parser.add_argument("--dynesty-bound", choices=["multi", "single", "balls", "cubes"], default="multi")
    parser.add_argument("--dynesty-sample", choices=["auto", "unif", "rwalk", "slice", "rslice", "hslice"], default="rslice")
    parser.add_argument("--dynesty-walks", type=int, default=64)
    parser.add_argument("--dynesty-slices", type=int, default=12)
    parser.add_argument("--dynesty-bootstrap", type=int, default=0)
    parser.add_argument("--dynesty-update-interval", type=float, default=0.6)
    parser.add_argument("--dynesty-dlogz", type=float, default=0.10)
    parser.add_argument("--dynesty-dlogz-init", type=float, default=0.30, help="DynamicNestedSampler dlogz_init.")
    parser.add_argument("--dynesty-maxiter", type=int, default=0, help="0 means no explicit maxiter bound.")
    parser.add_argument("--dynesty-maxcall", type=int, default=0, help="0 means no explicit maxcall bound.")
    parser.add_argument("--posterior-ess-target", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fig10_reproduction_dynesty.png"),
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=Path("results/fig10_reproduction_dynesty_traces.png"),
    )
    parser.add_argument(
        "--trace-all-prefix",
        type=Path,
        default=Path("results/fig10_reproduction_dynesty_trace_all.png"),
    )
    parser.add_argument(
        "--diag-csv",
        type=Path,
        default=Path("results/fig10_reproduction_dynesty_diagnostics.csv"),
    )
    parser.add_argument("--samples-prefix", type=Path, default=None)
    return parser.parse_args()


def parse_n_values(text: str) -> list[int]:
    vals: list[int] = []
    for token in text.split(","):
        s = token.strip()
        if not s:
            continue
        vals.append(int(s))
    if not vals:
        raise ValueError("empty n-values")
    return sorted(set(vals))


def normalize_1d_pdf(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    z = float(np.trapezoid(p, x))
    if z > 0:
        return p / z
    return p


def credible_level_2d(pdf: np.ndarray, cred: float) -> float:
    flat = pdf.ravel()
    order = np.argsort(flat)[::-1]
    cdf = np.cumsum(flat[order])
    idx = int(np.searchsorted(cdf, cred, side="left"))
    idx = min(max(idx, 0), order.size - 1)
    return float(flat[order[idx]])


def mass_to_ms(total_mass_msun: float) -> float:
    return MSUN_SEC * total_mass_msun * 1e3


def n_tagged_path(base: Path, n: int) -> Path:
    suffix = base.suffix if base.suffix else ".png"
    stem = base.stem if base.suffix else str(base)
    p = Path(stem)
    return base.parent / f"{p.stem}_N{n}{suffix}"


def build_psd(freqs_hz: np.ndarray, model: str) -> np.ndarray:
    if model == "aligo_design":
        return aligo_zero_det_high_power_psd(freqs_hz, f_low_hz=10.0)
    if model == "flat":
        return np.ones_like(freqs_hz, dtype=float)
    raise ValueError(f"unsupported psd model: {model}")


def boundary_hit_fraction(samples: np.ndarray, lo: float, hi: float, eps_frac: float) -> float:
    span = hi - lo
    if span <= 0:
        return float("nan")
    eps = eps_frac * span
    hit = (samples <= (lo + eps)) | (samples >= (hi - eps))
    return float(np.mean(hit))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    n_values = parse_n_values(args.n_values)

    if args.f_min_hz < 0:
        raise ValueError("--f-min-hz must be >= 0")
    if args.f_max_hz <= args.f_min_hz:
        raise ValueError("--f-max-hz must be > --f-min-hz")
    if args.fd_grid_df_hz <= 0:
        raise ValueError("--fd-grid-df-hz must be positive")
    if args.boundary_eps_frac <= 0 or args.boundary_eps_frac >= 0.5:
        raise ValueError("--boundary-eps-frac must be in (0, 0.5)")
    if args.dynesty_nlive < 2:
        raise ValueError("--dynesty-nlive must be >= 2")
    if args.dynesty_nlive_init < 2:
        raise ValueError("--dynesty-nlive-init must be >= 2")
    if args.dynesty_nlive_batch < 2:
        raise ValueError("--dynesty-nlive-batch must be >= 2")
    if args.dynesty_maxbatch < 0:
        raise ValueError("--dynesty-maxbatch must be >= 0")
    if args.dynesty_n_effective < 0:
        raise ValueError("--dynesty-n-effective must be >= 0")
    if args.dynesty_dlogz <= 0:
        raise ValueError("--dynesty-dlogz must be positive")
    if args.dynesty_dlogz_init <= 0:
        raise ValueError("--dynesty-dlogz-init must be positive")
    if args.posterior_ess_target <= 0:
        raise ValueError("--posterior-ess-target must be positive")
    if args.likelihood_domain == "frequency" and args.strain_channel != "complex":
        raise ValueError("frequency-domain likelihood currently supports only --strain-channel complex")

    wf_raw, info = load_sxs_waveform22(location=args.sxs_location, download=not args.no_download)
    wf, t_peak_complex_raw = align_to_peak(wf_raw)
    if info.remnant_mass is None or info.remnant_chif_z is None:
        raise ValueError("missing remnant mass/spin metadata")

    t_all = wf.t
    h_complex_raw = wf.h
    h_plus_raw = h_complex_raw.real
    idx_peak_complex = int(np.argmax(np.abs(h_complex_raw)))
    idx_peak_plus = int(np.argmax(np.abs(h_plus_raw)))
    t_peak_complex_aligned = float(t_all[idx_peak_complex])
    t_peak_plus_aligned = float(t_all[idx_peak_plus])

    if args.strain_channel == "complex":
        h_obs_raw: np.ndarray = h_complex_raw
    else:
        h_obs_raw = h_plus_raw

    h_peak_raw = float(np.max(np.abs(h_obs_raw)))
    if h_peak_raw <= 0:
        raise ValueError("invalid analysis-channel peak")
    amp_scale_hpeak = float(args.target_hpeak / h_peak_raw)
    overall_amp_scale = amp_scale_hpeak
    h_obs = h_obs_raw * overall_amp_scale
    h_peak = float(np.max(np.abs(h_obs)))

    ms_per_M = mass_to_ms(args.m_total_msun)
    m_sec = MSUN_SEC * args.m_total_msun
    delta_t0_M = float(args.delta_t0_ms / ms_per_M)
    t_peak_ref = t_peak_complex_aligned if args.t0_reference == "complex_peak" else t_peak_plus_aligned
    t0 = t_peak_ref + delta_t0_M

    mask = (t_all >= t0) & (t_all <= args.t_end)
    t = t_all[mask]
    h_sig = h_obs[mask]
    if t.size < 50:
        raise ValueError("insufficient samples in analysis window")
    tau_m = t - t0

    sigma = float("nan")
    snr_before_rescale = float("nan")
    snr_achieved = float("nan")
    snr_rescale_factor = 1.0
    freq_df_hz = float("nan")
    freq_n_valid = 0
    fd_like: FrequencyDomainRingdownLikelihood | None = None
    data: np.ndarray | None = None

    if args.likelihood_domain == "frequency":
        dt_m = float(np.median(np.diff(t)))
        tau_end_m = float(tau_m[-1])
        tau_u_m = np.arange(0.0, tau_end_m + 0.5 * dt_m, dt_m)
        if tau_u_m.size < 64:
            raise ValueError("insufficient uniform samples for frequency-domain likelihood")

        h_u_re = np.interp(tau_u_m, tau_m, h_sig.real)
        h_u_im = np.interp(tau_u_m, tau_m, h_sig.imag)
        h_sig_u = h_u_re + 1j * h_u_im

        tau_u_sec = tau_u_m * m_sec
        freq_df_hz = float(args.fd_grid_df_hz)
        freqs_hz = np.arange(args.f_min_hz, args.f_max_hz + 0.5 * freq_df_hz, freq_df_hz)
        if freqs_hz.size < 16:
            raise ValueError("frequency grid too small; reduce --fd-grid-df-hz or widen frequency range")

        psd_full = build_psd(freqs_hz, args.psd_model)
        ip_valid = (
            (freqs_hz >= args.f_min_hz)
            & (freqs_hz <= args.f_max_hz)
            & np.isfinite(psd_full)
            & (psd_full > 0)
            & (freqs_hz > 0.0)
        )
        if np.count_nonzero(ip_valid) < 16:
            raise ValueError("too few valid frequency bins; relax f-range or PSD settings")

        signal_tilde = continuous_ft_from_time_series(tau_u_sec, h_sig_u, freqs_hz)
        snr_before_rescale = optimal_snr(signal_tilde, psd_full, freq_df_hz, valid_mask=ip_valid)
        if (not args.disable_psd_snr_rescale) and snr_before_rescale > 0:
            snr_rescale_factor = float(args.target_snr_postpeak / snr_before_rescale)
            overall_amp_scale *= snr_rescale_factor
            h_obs = h_obs_raw * overall_amp_scale
            h_peak = float(np.max(np.abs(h_obs)))
            h_sig = h_obs[mask]
            h_u_re = np.interp(tau_u_m, tau_m, h_sig.real)
            h_u_im = np.interp(tau_u_m, tau_m, h_sig.imag)
            h_sig_u = h_u_re + 1j * h_u_im
            signal_tilde = continuous_ft_from_time_series(tau_u_sec, h_sig_u, freqs_hz)
        snr_achieved = optimal_snr(signal_tilde, psd_full, freq_df_hz, valid_mask=ip_valid)

        noise_tilde = draw_colored_noise_rfft(
            rng,
            freqs_hz.size,
            psd_full,
            freq_df_hz,
            enforce_real_endpoints=False,
        )
        data_tilde = signal_tilde + noise_tilde
        fd_like = FrequencyDomainRingdownLikelihood(
            freqs_hz=freqs_hz,
            d_tilde=data_tilde,
            psd=psd_full,
            df=freq_df_hz,
            duration_sec=float(tau_u_sec[-1]),
            t0_sec=0.0,
            f_min_hz=args.f_min_hz,
            f_max_hz=args.f_max_hz,
            include_finite_duration=not args.disable_finite_duration_model,
        )
        freq_n_valid = fd_like.n_valid
        data = h_sig
    else:
        sigma = float(np.linalg.norm(h_sig) / args.target_snr_postpeak)
        if args.strain_channel == "complex":
            noise = rng.normal(loc=0.0, scale=sigma / np.sqrt(2.0), size=h_sig.size) + 1j * rng.normal(
                loc=0.0,
                scale=sigma / np.sqrt(2.0),
                size=h_sig.size,
            )
        else:
            noise = rng.normal(loc=0.0, scale=sigma, size=h_sig.size)
        data = h_sig + noise
        snr_achieved = float(np.linalg.norm(h_sig) / sigma)

    true_mf_msun = float(info.remnant_mass * args.m_total_msun)
    true_chif = float(info.remnant_chif_z)
    mf_bounds = (float(args.mf_min_msun), float(args.mf_max_msun))
    chif_hi = min(float(args.chif_max), 0.999)
    chif_bounds = (float(args.chif_min), chif_hi)
    amp_bounds = (AMP_PRIOR_MIN_REL, AMP_PRIOR_MAX_REL)
    phi_bounds = (0.0, 2.0 * np.pi)

    colors = {0: "#1f77b4", 1: "#6f2da8", 2: "#d4a017", 3: "#d62728"}
    linestyles = {0: "-", 1: "--", 2: "--", 3: "-"}

    mf_grid = np.linspace(args.mf_min_msun, args.mf_max_msun, 200)
    chif_grid = np.linspace(args.chif_min, chif_hi, 200)

    pdf2d_by_n: dict[int, np.ndarray] = {}
    post_mf_by_n: dict[int, np.ndarray] = {}
    post_chif_by_n: dict[int, np.ndarray] = {}
    level90_by_n: dict[int, float] = {}
    sample_by_n: dict[int, np.ndarray] = {}
    trace_by_n: dict[int, np.ndarray] = {}

    diag_rows: list[str] = [
        "N,ndim,dynesty_mode,nlive,nlive_init,nlive_batch,maxbatch,niter,ncall_total,eff_percent,logz,logzerr,dlogz_target,dlogz_init_target,converged,posterior_ess_kish,posterior_ess_target,boundary_mf_frac,boundary_chif_frac,boundary_amp_frac,map_mf_msun,map_chif,mf_q16,mf_q50,mf_q84,chif_q16,chif_q50,chif_q84"
    ]

    chi_interp = np.linspace(max(args.chif_min, 0.0), chif_hi, 240)
    qnm_table_re: dict[int, np.ndarray] = {}
    qnm_table_im: dict[int, np.ndarray] = {}
    max_n = max(n_values)
    for k in range(max_n + 1):
        vals = np.array([kerr_qnm_omega_lmn(mf=1.0, chif=float(c), ell=2, m=2, n=k) for c in chi_interp], dtype=complex)
        qnm_table_re[k] = vals.real
        qnm_table_im[k] = vals.imag

    for n in n_values:
        ndim = 2 + 2 * (n + 1)

        def prior_transform(u: np.ndarray) -> np.ndarray:
            theta = np.empty_like(u)
            theta[0] = mf_bounds[0] + u[0] * (mf_bounds[1] - mf_bounds[0])
            theta[1] = chif_bounds[0] + u[1] * (chif_bounds[1] - chif_bounds[0])
            amps_u = u[2 : 2 + (n + 1)]
            phis_u = u[2 + (n + 1) : 2 + 2 * (n + 1)]
            theta[2 : 2 + (n + 1)] = amp_bounds[0] + amps_u * (amp_bounds[1] - amp_bounds[0])
            theta[2 + (n + 1) : 2 + 2 * (n + 1)] = phi_bounds[0] + phis_u * (phi_bounds[1] - phi_bounds[0])
            return theta

        def log_likelihood(theta: np.ndarray) -> float:
            mf_msun = float(theta[0])
            chif = float(theta[1])
            amp_rel = theta[2 : 2 + (n + 1)]
            phis = theta[2 + (n + 1) : 2 + 2 * (n + 1)]

            if not (mf_bounds[0] <= mf_msun <= mf_bounds[1]):
                return -np.inf
            if not (chif_bounds[0] <= chif <= chif_bounds[1]):
                return -np.inf
            if np.any((amp_rel < amp_bounds[0]) | (amp_rel > amp_bounds[1])):
                return -np.inf
            if np.any((phis < phi_bounds[0]) | (phis > phi_bounds[1])):
                return -np.inf

            mf_frac = mf_msun / args.m_total_msun
            if mf_frac <= 0:
                return -np.inf

            omegas = np.empty(n + 1, dtype=complex)
            for k in range(n + 1):
                wr = np.interp(chif, chi_interp, qnm_table_re[k])
                wi = np.interp(chif, chi_interp, qnm_table_im[k])
                omegas[k] = (wr + 1j * wi) / mf_frac

            if args.likelihood_domain == "frequency":
                if fd_like is None:
                    return -np.inf
                return float(
                    fd_like.log_likelihood(
                        omegas_rad_s=omegas / m_sec,
                        amplitudes=amp_rel * h_peak,
                        phases=phis,
                    )
                )

            if data is None:
                return -np.inf
            tau = t - t0
            model_c = np.zeros_like(t, dtype=complex)
            for k in range(n + 1):
                model_c += (amp_rel[k] * h_peak) * np.exp(-1j * (omegas[k] * tau + phis[k]))
            model = model_c if args.strain_channel == "complex" else model_c.real
            resid = data - model
            if np.iscomplexobj(resid):
                chi2 = float(np.vdot(resid, resid).real / (sigma**2))
            else:
                chi2 = float(np.dot(resid, resid) / (sigma**2))
            return float(-0.5 * chi2)

        dynesty_kwargs: dict[str, object] = dict(
            bound=args.dynesty_bound,
            sample=args.dynesty_sample,
            bootstrap=args.dynesty_bootstrap,
            update_interval=args.dynesty_update_interval,
            rstate=np.random.default_rng(args.seed + 1000 * (n + 1)),
        )
        if args.dynesty_sample == "rwalk":
            dynesty_kwargs["walks"] = args.dynesty_walks
        if args.dynesty_sample in {"slice", "rslice", "hslice"}:
            dynesty_kwargs["slices"] = args.dynesty_slices

        run_maxiter: int | None = args.dynesty_maxiter if args.dynesty_maxiter > 0 else None
        run_maxcall: int | None = args.dynesty_maxcall if args.dynesty_maxcall > 0 else None
        nlive_report = args.dynesty_nlive
        if args.dynesty_mode == "static":
            sampler = dynesty.NestedSampler(
                loglikelihood=log_likelihood,
                prior_transform=prior_transform,
                ndim=ndim,
                nlive=args.dynesty_nlive,
                **dynesty_kwargs,
            )
            run_kwargs: dict[str, object] = dict(dlogz=args.dynesty_dlogz, print_progress=False)
            if run_maxiter is not None:
                run_kwargs["maxiter"] = run_maxiter
            if run_maxcall is not None:
                run_kwargs["maxcall"] = run_maxcall
            if args.dynesty_n_effective > 0:
                run_kwargs["n_effective"] = args.dynesty_n_effective
            sampler.run_nested(**run_kwargs)
        else:
            sampler = dynesty.DynamicNestedSampler(
                loglikelihood=log_likelihood,
                prior_transform=prior_transform,
                ndim=ndim,
                nlive=args.dynesty_nlive_init,
                **dynesty_kwargs,
            )
            nlive_report = args.dynesty_nlive_init
            run_kwargs = dict(
                nlive_init=args.dynesty_nlive_init,
                nlive_batch=args.dynesty_nlive_batch,
                dlogz_init=args.dynesty_dlogz_init,
                print_progress=False,
            )
            if run_maxiter is not None:
                run_kwargs["maxiter"] = run_maxiter
            if run_maxcall is not None:
                run_kwargs["maxcall"] = run_maxcall
            if args.dynesty_maxbatch > 0:
                run_kwargs["maxbatch"] = args.dynesty_maxbatch
            if args.dynesty_n_effective > 0:
                run_kwargs["n_effective"] = args.dynesty_n_effective
            sampler.run_nested(**run_kwargs)
        res = sampler.results

        samples_raw = np.asarray(res.samples, dtype=float)
        logwt = np.asarray(res.logwt, dtype=float)
        logz_final = float(res.logz[-1])
        w = np.exp(logwt - logz_final)
        w = np.clip(w, 0.0, np.inf)
        wsum = float(np.sum(w))
        if wsum <= 0:
            raise RuntimeError(f"invalid dynesty weights for N={n}")
        w = w / wsum
        ess_kish = float(1.0 / np.sum(w**2))
        samples_eq = resample_equal(samples_raw, w)

        trace_by_n[n] = samples_raw
        sample_by_n[n] = samples_eq
        mf_s = samples_eq[:, 0]
        chif_s = samples_eq[:, 1]

        hist2d, xedges, yedges = np.histogram2d(
            mf_s,
            chif_s,
            bins=[mf_grid, chif_grid],
            density=False,
        )
        pdf2d = hist2d.T
        if np.sum(pdf2d) <= 0:
            raise RuntimeError(f"invalid posterior histogram for N={n}")
        pdf2d /= np.sum(pdf2d)
        pdf2d_by_n[n] = pdf2d
        level90_by_n[n] = credible_level_2d(pdf2d, cred=0.9)

        post_mf = np.histogram(mf_s, bins=mf_grid, density=False)[0].astype(float)
        post_chi = np.histogram(chif_s, bins=chif_grid, density=False)[0].astype(float)
        mf_centers = 0.5 * (mf_grid[:-1] + mf_grid[1:])
        chi_centers = 0.5 * (chif_grid[:-1] + chif_grid[1:])
        post_mf_by_n[n] = normalize_1d_pdf(mf_centers, post_mf)
        post_chif_by_n[n] = normalize_1d_pdf(chi_centers, post_chi)

        idx_map = int(np.argmax(np.asarray(res.logl)))
        map_mf = float(samples_raw[idx_map, 0])
        map_chi = float(samples_raw[idx_map, 1])

        mf_q16, mf_q50, mf_q84 = [float(v) for v in np.quantile(mf_s, [0.16, 0.50, 0.84])]
        ch_q16, ch_q50, ch_q84 = [float(v) for v in np.quantile(chif_s, [0.16, 0.50, 0.84])]

        amp_eq = samples_eq[:, 2 : 2 + (n + 1)]
        mf_boundary = boundary_hit_fraction(mf_s, mf_bounds[0], mf_bounds[1], args.boundary_eps_frac)
        chif_boundary = boundary_hit_fraction(chif_s, chif_bounds[0], chif_bounds[1], args.boundary_eps_frac)
        amp_boundary = boundary_hit_fraction(amp_eq.reshape(-1), amp_bounds[0], amp_bounds[1], args.boundary_eps_frac)

        niter = int(res.niter)
        ncall_total = int(np.sum(np.asarray(res.ncall)))
        eff_arr = np.asarray(res.eff).reshape(-1)
        eff_percent = float(eff_arr[-1]) if eff_arr.size > 0 else float("nan")
        logzerr = float(res.logzerr[-1])
        converged = bool(logzerr <= args.dynesty_dlogz and ess_kish >= args.posterior_ess_target)

        diag_rows.append(
            f"{n},{ndim},{args.dynesty_mode},{nlive_report},{args.dynesty_nlive_init},{args.dynesty_nlive_batch},"
            f"{args.dynesty_maxbatch},{niter},{ncall_total},{eff_percent:.6f},{logz_final:.6f},"
            f"{logzerr:.6f},{args.dynesty_dlogz:.6f},{args.dynesty_dlogz_init:.6f},{int(converged)},"
            f"{ess_kish:.3f},{args.posterior_ess_target:.3f},"
            f"{mf_boundary:.6f},{chif_boundary:.6f},{amp_boundary:.6f},{map_mf:.6f},{map_chi:.6f},"
            f"{mf_q16:.6f},{mf_q50:.6f},{mf_q84:.6f},{ch_q16:.6f},{ch_q50:.6f},{ch_q84:.6f}"
        )
        print(
            f"N={n} mode={args.dynesty_mode} nlive={nlive_report} niter={niter} ncall={ncall_total} "
            f"eff={eff_percent:.2f}% logzerr={logzerr:.4f} ess_kish={ess_kish:.1f} "
            f"converged={converged} mf_q50={mf_q50:.3f} chif_q50={ch_q50:.3f}"
        )

        if args.samples_prefix is not None:
            out_samples = n_tagged_path(args.samples_prefix, n).with_suffix(".npz")
            out_samples.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_samples,
                samples_equal=samples_eq,
                samples_raw=samples_raw,
                weights=w,
                mf=mf_s,
                chif=chif_s,
            )

    fig = plt.figure(figsize=(9, 8))
    gs = fig.add_gridspec(4, 4, hspace=0.05, wspace=0.05)
    ax_top = fig.add_subplot(gs[0, 0:3])
    ax_main = fig.add_subplot(gs[1:4, 0:3], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

    mf_centers = 0.5 * (mf_grid[:-1] + mf_grid[1:])
    chi_centers = 0.5 * (chif_grid[:-1] + chif_grid[1:])
    handles: list[mlines.Line2D] = []

    for n in n_values:
        pdf2d = pdf2d_by_n[n]
        level90 = level90_by_n[n]
        color = colors.get(n, None)
        ls = linestyles.get(n, "-")
        ax_main.contour(
            mf_centers,
            chi_centers,
            pdf2d,
            levels=[level90],
            colors=[color],
            linestyles=[ls],
            linewidths=2.0,
        )
        ax_top.plot(mf_centers, post_mf_by_n[n], color=color, ls=ls, lw=1.8)
        ax_right.plot(post_chif_by_n[n], chi_centers, color=color, ls=ls, lw=1.8)
        handles.append(mlines.Line2D([], [], color=color, ls=ls, lw=2.0, label=f"N={n}"))

    ax_main.axvline(true_mf_msun, color="k", ls=":", lw=1.2)
    ax_main.axhline(true_chif, color="k", ls=":", lw=1.2)
    ax_main.set_xlabel(r"$M_f\ [M_\odot]$")
    ax_main.set_ylabel(r"$\chi_f$")
    ax_main.set_xlim(args.mf_min_msun, args.mf_max_msun)
    ax_main.set_ylim(args.chif_min, chif_hi)
    ax_main.grid(True, alpha=0.15)
    ax_main.legend(handles=handles, loc="upper left", fontsize=9)

    ax_top.set_ylabel("Posterior")
    ax_top.grid(True, alpha=0.15)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_right.set_xlabel("Posterior")
    ax_right.grid(True, alpha=0.15)
    ax_right.tick_params(axis="y", labelleft=False)
    ax_top.set_title(rf"Fig.10-style Dynesty posteriors ($\Delta t_0={args.delta_t0_ms:.3f}$ ms)")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)

    fig_t, axes_t = plt.subplots(len(n_values), 2, figsize=(10, 2.4 * len(n_values)), sharex=False)
    if len(n_values) == 1:
        axes_t = np.array([axes_t])
    for r, n in enumerate(n_values):
        trace = trace_by_n[n]
        axes_t[r, 0].plot(trace[:, 0], lw=0.45, alpha=0.8)
        axes_t[r, 1].plot(trace[:, 1], lw=0.45, alpha=0.8)
        axes_t[r, 0].set_ylabel(f"N={n}")
        axes_t[r, 0].grid(True, alpha=0.15)
        axes_t[r, 1].grid(True, alpha=0.15)
    axes_t[0, 0].set_title("Sequence: $M_f$")
    axes_t[0, 1].set_title("Sequence: $\\chi_f$")
    axes_t[-1, 0].set_xlabel("Nested sample index")
    axes_t[-1, 1].set_xlabel("Nested sample index")
    fig_t.tight_layout()
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    fig_t.savefig(args.trace_output, dpi=180)

    for n in n_values:
        trace = trace_by_n[n]
        ndim = trace.shape[1]
        labels = ["$M_f$", "$\\chi_f$"] + [f"$A_{k}/h_{{peak}}$" for k in range(n + 1)] + [f"$\\phi_{k}$" for k in range(n + 1)]
        fig_all, axes_all = plt.subplots(ndim, 1, figsize=(10, 1.55 * ndim), sharex=True)
        if ndim == 1:
            axes_all = [axes_all]
        for p in range(ndim):
            axes_all[p].plot(trace[:, p], lw=0.4, alpha=0.8)
            axes_all[p].set_ylabel(labels[p], fontsize=8)
            axes_all[p].grid(True, alpha=0.15)
        axes_all[-1].set_xlabel("Nested sample index")
        fig_all.tight_layout()
        out_trace_all = n_tagged_path(args.trace_all_prefix, n)
        out_trace_all.parent.mkdir(parents=True, exist_ok=True)
        fig_all.savefig(out_trace_all, dpi=180)
        plt.close(fig_all)

    args.diag_csv.parent.mkdir(parents=True, exist_ok=True)
    args.diag_csv.write_text("\n".join(diag_rows) + "\n", encoding="utf-8")

    print(f"source={wf.source}")
    print(f"distance_mpc={args.distance_mpc:.3f}")
    print(
        f"sampler=dynesty mode={args.dynesty_mode} bound={args.dynesty_bound} sample={args.dynesty_sample} "
        f"nlive={args.dynesty_nlive} nlive_init={args.dynesty_nlive_init} "
        f"nlive_batch={args.dynesty_nlive_batch} dlogz={args.dynesty_dlogz:.4f} "
        f"dlogz_init={args.dynesty_dlogz_init:.4f}"
    )
    print(f"strain_channel={args.strain_channel}, likelihood_domain={args.likelihood_domain}, psd_model={args.psd_model}")
    print(f"t_peak_complex_raw_M={t_peak_complex_raw:.6f}")
    print(f"t_peak_complex_aligned_M={t_peak_complex_aligned:.6f}")
    print(f"t_peak_plus_aligned_M={t_peak_plus_aligned:.6f}")
    print(f"delta_t0_ms={args.delta_t0_ms:.6f}, delta_t0_M={delta_t0_M:.6f}, t0_ref_M={t_peak_ref:.6f}, t0_M={t0:.6f}")
    print(f"n_values={n_values}")
    print(f"true_mf_msun={true_mf_msun:.6f}, true_chif={true_chif:.6f}")
    print(f"target_hpeak={args.target_hpeak:.3e}, achieved_hpeak={h_peak:.3e}")
    print(
        f"target_postpeak_snr={args.target_snr_postpeak:.3f}, "
        f"pre_rescale_psd_snr={snr_before_rescale:.3f}, snr_rescale_factor={snr_rescale_factor:.6f}, "
        f"achieved_postpeak_snr={snr_achieved:.3f}"
    )
    if args.likelihood_domain == "frequency":
        print(
            f"window_samples={t.size}, valid_freq_bins={freq_n_valid}, "
            f"f_range_hz=[{args.f_min_hz:.3f}, {args.f_max_hz:.3f}], df_hz={freq_df_hz:.6f}"
        )
    else:
        print(f"window_samples={t.size}, sigma={sigma:.3e}")
    print(f"diag_csv={args.diag_csv}")
    print(f"trace_output={args.trace_output}")
    print(f"trace_all_prefix={args.trace_all_prefix}")
    if args.samples_prefix is not None:
        print(f"samples_prefix={args.samples_prefix}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
