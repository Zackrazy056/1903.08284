from __future__ import annotations

import argparse
from pathlib import Path

import emcee
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from ringdown.fd_likelihood import (
    aligo_zero_det_high_power_psd,
    continuous_ft_from_time_series,
    draw_colored_noise_rfft,
    irfft_continuous,
    optimal_snr,
    real_ringdown_mode_tilde,
)
from ringdown.frequencies import kerr_qnm_omegas_22n
from ringdown.sxs_io import load_sxs_waveform22


MSUN_SEC = 4.92549095e-6


def parse_float_list(text: str) -> list[float]:
    vals = [float(s.strip()) for s in text.split(",") if s.strip()]
    if not vals:
        raise ValueError("empty float list")
    return vals


def credible_level_2d(pdf: np.ndarray, cred: float = 0.9) -> float:
    flat = pdf.ravel()
    order = np.argsort(flat)[::-1]
    cdf = np.cumsum(flat[order])
    idx = int(np.searchsorted(cdf, cred, side="left"))
    idx = min(max(idx, 0), order.size - 1)
    return float(flat[order[idx]])


def normalize_1d_pdf(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    z = float(np.trapezoid(p, x))
    return p / z if z > 0 else p


def robust_tau_ess(chain: np.ndarray) -> tuple[float, float, float]:
    try:
        tau = emcee.autocorr.integrated_time(chain, tol=0, quiet=True)
        tau_mf = float(tau[0])
        tau_chif = float(tau[1])
        ess = float(chain.shape[0] * chain.shape[1] / max(np.max(tau[:2]), 1.0))
        return tau_mf, tau_chif, ess
    except Exception:
        return float("nan"), float("nan"), float("nan")


def truth_hpd_rank(
    hist2d: np.ndarray,
    truth_mf: float,
    truth_chif: float,
    mf_range: tuple[float, float],
    chif_range: tuple[float, float],
) -> float:
    h = np.asarray(hist2d, dtype=float)
    total = float(np.sum(h))
    if total <= 0:
        return 1.0
    mf_min, mf_max = mf_range
    chif_min, chif_max = chif_range
    if not (mf_min <= truth_mf <= mf_max and chif_min <= truth_chif <= chif_max):
        return 1.0
    n_chif, n_mf = h.shape
    mf_bin = int(np.floor((truth_mf - mf_min) / (mf_max - mf_min) * n_mf))
    chif_bin = int(np.floor((truth_chif - chif_min) / (chif_max - chif_min) * n_chif))
    mf_bin = min(max(mf_bin, 0), n_mf - 1)
    chif_bin = min(max(chif_bin, 0), n_chif - 1)
    truth_density = float(h[chif_bin, mf_bin])
    return float(np.sum(h[h >= truth_density])) / total


def detector_strain_from_mode22(h22: np.ndarray) -> np.ndarray:
    if h22.ndim != 1:
        raise ValueError("h22 must be 1D")
    i_ref = int(np.argmax(np.abs(h22)))
    href = h22[i_ref]
    phase_ref = float(np.angle(href)) if np.abs(href) > 0 else 0.0
    return np.real(h22 * np.exp(-1j * phase_ref))


def peak_time_from_detector_strain(t: np.ndarray, h_det: np.ndarray) -> float:
    if t.ndim != 1 or h_det.ndim != 1 or t.size != h_det.size:
        raise ValueError("t and h_det must be 1D arrays with identical length")
    idx = int(np.argmax(np.abs(h_det)))
    return float(t[idx])


def format_dt0_tag(dt0_ms: float) -> str:
    text = f"{dt0_ms:.3f}".replace("-", "m").replace(".", "p")
    return f"dt0_{text}ms"


class DetectorRingdownPosterior:
    def __init__(
        self,
        *,
        freqs_hz: np.ndarray,
        d_tilde: np.ndarray,
        psd: np.ndarray,
        df_hz: float,
        duration_sec: float,
        h_peak: float,
        n_overtones: int,
        m_total_msun: float,
        mf_bounds: tuple[float, float],
        chif_bounds: tuple[float, float],
        amp_bounds_rel: tuple[float, float],
        phi_bounds: tuple[float, float],
        qnm_chi_grid_size: int,
    ) -> None:
        self.n_overtones = int(n_overtones)
        self.n_modes = self.n_overtones + 1
        self.ndim = 2 + 2 * self.n_modes
        self.duration_sec = float(duration_sec)
        self.h_peak = float(h_peak)
        self.m_total_msun = float(m_total_msun)
        self.m_sec = MSUN_SEC * self.m_total_msun
        self.mf_bounds = mf_bounds
        self.chif_bounds = chif_bounds
        self.amp_bounds_rel = amp_bounds_rel
        self.phi_bounds = phi_bounds

        valid = (freqs_hz > 0.0) & np.isfinite(psd) & (psd > 0.0)
        if np.count_nonzero(valid) < 16:
            raise ValueError("too few valid frequency bins")
        self.f_calc = np.asarray(freqs_hz[valid], dtype=float)
        self.d_calc = np.asarray(d_tilde[valid], dtype=complex)
        self.psd_calc = np.asarray(psd[valid], dtype=float)
        self.df_hz = float(df_hz)
        self.d_weighted = self.d_calc / self.psd_calc

        chi_lo = max(self.chif_bounds[0], 0.0)
        chi_hi = min(self.chif_bounds[1], 0.999)
        if qnm_chi_grid_size < 64:
            raise ValueError("qnm chi grid too small")
        self.chi_grid = np.linspace(chi_lo, chi_hi, qnm_chi_grid_size)
        self.qnm_re = np.empty((self.n_modes, self.chi_grid.size), dtype=float)
        self.qnm_im = np.empty((self.n_modes, self.chi_grid.size), dtype=float)
        for j, chi in enumerate(self.chi_grid):
            vals = kerr_qnm_omegas_22n(mf=1.0, chif=float(chi), n_max=self.n_overtones)
            self.qnm_re[:, j] = vals.real
            self.qnm_im[:, j] = vals.imag

    def sample_initial_walkers(
        self,
        rng: np.random.Generator,
        nwalkers: int,
        *,
        true_mf_msun: float,
        true_chif: float,
        init_strategy: str,
        init_mf_sigma_msun: float,
        init_chif_sigma: float,
    ) -> np.ndarray:
        p = np.empty((nwalkers, self.ndim), dtype=float)
        p[:, 0] = rng.uniform(self.mf_bounds[0], self.mf_bounds[1], size=nwalkers)
        p[:, 1] = rng.uniform(self.chif_bounds[0], self.chif_bounds[1], size=nwalkers)
        if init_strategy == "truth_ball":
            p[:, 0] = np.clip(
                rng.normal(true_mf_msun, init_mf_sigma_msun, size=nwalkers),
                self.mf_bounds[0],
                self.mf_bounds[1],
            )
            p[:, 1] = np.clip(
                rng.normal(true_chif, init_chif_sigma, size=nwalkers),
                self.chif_bounds[0] + 1e-6,
                self.chif_bounds[1] - 1e-6,
            )
        for k in range(self.n_modes):
            ia = 2 + 2 * k
            ip = 3 + 2 * k
            p[:, ia] = rng.uniform(self.amp_bounds_rel[0], self.amp_bounds_rel[1], size=nwalkers)
            p[:, ip] = rng.uniform(self.phi_bounds[0], self.phi_bounds[1], size=nwalkers)
        return p

    def log_prior(self, theta: np.ndarray) -> float:
        mf = float(theta[0])
        chif = float(theta[1])
        if not (self.mf_bounds[0] <= mf <= self.mf_bounds[1]):
            return -np.inf
        if not (self.chif_bounds[0] <= chif <= self.chif_bounds[1]):
            return -np.inf
        amps = theta[2::2]
        phis = theta[3::2]
        if np.any((amps < self.amp_bounds_rel[0]) | (amps > self.amp_bounds_rel[1])):
            return -np.inf
        if np.any((phis < self.phi_bounds[0]) | (phis > self.phi_bounds[1])):
            return -np.inf
        return 0.0

    def model_tilde(self, theta: np.ndarray) -> np.ndarray:
        mf = float(theta[0])
        chif = float(theta[1])
        mf_frac = mf / self.m_total_msun
        if mf_frac <= 0.0:
            raise ValueError("invalid mf_frac")
        amps = theta[2::2] * self.h_peak
        phis = theta[3::2]
        omegas_m = np.empty(self.n_modes, dtype=complex)
        for k in range(self.n_modes):
            wr = np.interp(chif, self.chi_grid, self.qnm_re[k])
            wi = np.interp(chif, self.chi_grid, self.qnm_im[k])
            omegas_m[k] = (wr + 1j * wi) / mf_frac
        omegas_rad_s = omegas_m / self.m_sec

        return real_ringdown_mode_tilde(
            self.f_calc,
            omegas_rad_s,
            amps,
            phis,
            duration_sec=self.duration_sec,
        )

    def log_likelihood(self, theta: np.ndarray) -> float:
        h_tilde = self.model_tilde(theta)
        d_h = 4.0 * self.df_hz * np.sum(np.real(self.d_weighted * np.conjugate(h_tilde)))
        h_h = 4.0 * self.df_hz * np.sum((np.abs(h_tilde) ** 2) / self.psd_calc)
        return float(d_h - 0.5 * h_h)

    def log_posterior(self, theta: np.ndarray) -> float:
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll


def run_emcee_chain(
    *,
    posterior: DetectorRingdownPosterior,
    p0: np.ndarray,
    burnin_steps: int,
    prod_steps: int,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    em = emcee.EnsembleSampler(p0.shape[0], posterior.ndim, posterior.log_posterior)
    state = em.run_mcmc(p0, burnin_steps, progress=False)
    em.reset()
    em.run_mcmc(state, prod_steps, progress=False)
    chain = em.get_chain()
    flat = em.get_chain(flat=True)
    acc = float(np.mean(em.acceptance_fraction))
    tau_mf, tau_chif, ess = robust_tau_ess(chain)
    return chain, flat, acc, tau_mf, tau_chif, ess


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sxs-location", type=str, default="SXS:BBH:0305v2.0/Lev6")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--m-total-msun", type=float, default=72.0)
    p.add_argument("--target-hpeak", type=float, default=2e-21)
    p.add_argument("--target-snr-postpeak", type=float, default=42.3)
    p.add_argument("--dt0-ms-list", type=str, default="0,3,6,10")
    p.add_argument("--reference-n", type=int, default=3)
    p.add_argument("--reference-dt0-ms", type=float, default=0.0)
    p.add_argument("--t-end", type=float, default=90.0)
    p.add_argument("--f-min-hz", type=float, default=20.0)
    p.add_argument("--f-max-hz", type=float, default=1024.0)
    p.add_argument("--df-hz", type=float, default=1.0)
    p.add_argument("--mf-min-msun", type=float, default=50.0)
    p.add_argument("--mf-max-msun", type=float, default=100.0)
    p.add_argument("--chif-min", type=float, default=0.0)
    p.add_argument("--chif-max", type=float, default=1.0)
    p.add_argument("--amp-min-rel", type=float, default=0.01)
    p.add_argument("--amp-max-rel", type=float, default=250.0)
    p.add_argument("--qnm-chi-grid-size", type=int, default=240)
    p.add_argument("--nwalkers", type=int, default=64)
    p.add_argument("--emcee-burnin", type=int, default=1000)
    p.add_argument("--emcee-steps", type=int, default=3000)
    p.add_argument("--init-strategy", choices=["prior", "truth_ball"], default="truth_ball")
    p.add_argument("--init-mf-sigma-msun", type=float, default=1.5)
    p.add_argument("--init-chif-sigma", type=float, default=0.05)
    p.add_argument("--min-acceptance", type=float, default=0.01)
    p.add_argument("--require-health", action="store_true")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--output", type=Path, default=Path("results/fig11_emcee_compare.png"))
    p.add_argument("--diag-csv", type=Path, default=Path("results/fig11_emcee_compare_diag.csv"))
    p.add_argument("--samples-prefix", type=Path, default=Path("results/fig11_emcee_compare_samples.npz"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dt0_ms_list = parse_float_list(args.dt0_ms_list)
    if args.reference_n < 0:
        raise ValueError("reference-n must be nonnegative")
    max_ndim = 2 + 2 * (max(args.reference_n, 0) + 1)
    if args.nwalkers <= max_ndim:
        raise ValueError(f"nwalkers must be > max ndim ({max_ndim})")

    rng = np.random.default_rng(args.seed)
    wf, info = load_sxs_waveform22(location=args.sxs_location, download=not args.no_download)
    if info.remnant_mass is None or info.remnant_chif_z is None:
        raise ValueError("missing remnant metadata")

    t_raw = wf.t
    h22 = wf.h
    h_det_raw = detector_strain_from_mode22(h22)
    t_hpeak = peak_time_from_detector_strain(t_raw, h_det_raw)
    t = t_raw - t_hpeak

    mask_full = (t >= 0.0) & (t <= args.t_end)
    t_full = t[mask_full]
    h_full = h_det_raw[mask_full]
    if t_full.size < 64:
        raise ValueError("post-peak window is too short")

    h_peak_raw = float(np.max(np.abs(h_full)))
    if h_peak_raw <= 0.0:
        raise RuntimeError("detector strain has non-positive peak amplitude")
    h_full = h_full * (args.target_hpeak / h_peak_raw)

    dt_m = float(np.median(np.diff(t_full)))
    tau_full_m = np.arange(0.0, float(t_full[-1]) + 0.5 * dt_m, dt_m)
    signal_full = np.interp(tau_full_m, t_full, h_full)
    m_sec = MSUN_SEC * args.m_total_msun
    tau_full_sec = tau_full_m * m_sec

    freqs_eval = np.arange(args.f_min_hz, args.f_max_hz + 0.5 * args.df_hz, args.df_hz)
    psd_eval = aligo_zero_det_high_power_psd(freqs_eval, f_low_hz=10.0)
    valid_eval = (freqs_eval >= args.f_min_hz) & (freqs_eval <= args.f_max_hz) & np.isfinite(psd_eval) & (psd_eval > 0.0)
    signal_tilde_full = continuous_ft_from_time_series(tau_full_sec, signal_full, freqs_eval)
    snr_before = optimal_snr(signal_tilde_full, psd_eval, args.df_hz, valid_mask=valid_eval)
    if snr_before <= 0.0:
        raise RuntimeError("invalid pre-rescale SNR")
    snr_scale = float(args.target_snr_postpeak / snr_before)
    signal_full = signal_full * snr_scale
    signal_tilde_full = continuous_ft_from_time_series(tau_full_sec, signal_full, freqs_eval)
    snr_after = optimal_snr(signal_tilde_full, psd_eval, args.df_hz, valid_mask=valid_eval)
    h_peak = float(np.max(np.abs(signal_full)))

    dt_sec = dt_m * m_sec
    noise_freqs = np.fft.rfftfreq(tau_full_m.size, d=dt_sec)
    if noise_freqs.size < 2:
        raise RuntimeError("noise frequency grid too small")
    if noise_freqs[-1] < args.f_max_hz:
        raise ValueError("Nyquist frequency is below f-max-hz; reduce f-max-hz or refine time sampling")
    noise_df_hz = float(noise_freqs[1] - noise_freqs[0])
    noise_psd = aligo_zero_det_high_power_psd(noise_freqs, f_low_hz=10.0)
    noise_rfft = draw_colored_noise_rfft(
        rng,
        noise_freqs.size,
        noise_psd,
        noise_df_hz,
        enforce_real_endpoints=True,
    )
    noise_full = irfft_continuous(noise_rfft, tau_full_m.size, dt_sec)
    data_full = signal_full + noise_full

    true_mf = float(info.remnant_mass * args.m_total_msun)
    true_chif = float(info.remnant_chif_z)
    mf_bounds = (args.mf_min_msun, args.mf_max_msun)
    chif_bounds = (args.chif_min, min(args.chif_max, 0.999))

    series = [
        {"n": 0, "dt0_ms": float(dt0_ms), "color": "#1f77b4", "ls": ls, "label": f"N=0, dt0={dt0_ms:g} ms"}
        for dt0_ms, ls in zip(dt0_ms_list, ["-", "--", "-.", ":"] + ["--"] * 32)
    ]
    series.append(
        {
            "n": int(args.reference_n),
            "dt0_ms": float(args.reference_dt0_ms),
            "color": "#d62728",
            "ls": "-",
            "label": f"N={args.reference_n}, dt0={args.reference_dt0_ms:g} ms",
        }
    )

    mf_grid = np.linspace(args.mf_min_msun, args.mf_max_msun, 220)
    chif_grid = np.linspace(args.chif_min, chif_bounds[1], 220)
    mf_centers = 0.5 * (mf_grid[:-1] + mf_grid[1:])
    chif_centers = 0.5 * (chif_grid[:-1] + chif_grid[1:])

    pdf2d_by_label: dict[str, np.ndarray] = {}
    post_mf_by_label: dict[str, np.ndarray] = {}
    post_chif_by_label: dict[str, np.ndarray] = {}
    level90_by_label: dict[str, float] = {}

    diag_rows = [
        "label,N,dt0_requested_ms,dt0_realized_ms,nwalkers,burnin,steps,acceptance,tau_mf,tau_chif,ess,snr_window,mf_q16,mf_q50,mf_q84,chif_q16,chif_q50,chif_q84,truth_hpd_rank,truth_in_90_hpd,healthy_minimum"
    ]

    for item in series:
        n = int(item["n"])
        dt0_ms = float(item["dt0_ms"])
        dt0_m = dt0_ms / (m_sec * 1e3)
        start_idx = int(np.searchsorted(tau_full_m, dt0_m, side="left"))
        if start_idx >= tau_full_m.size - 32:
            raise ValueError(f"dt0={dt0_ms:.3f} ms leaves too short a window")

        tau_window_m = tau_full_m[start_idx:]
        tau_local_sec = (tau_window_m - tau_window_m[0]) * m_sec
        signal_window = signal_full[start_idx:]
        data_window = data_full[start_idx:]
        duration_sec = float(tau_local_sec[-1])
        if duration_sec <= 0.0:
            raise ValueError("window duration must be positive")

        d_tilde = continuous_ft_from_time_series(tau_local_sec, data_window, freqs_eval)
        signal_tilde = continuous_ft_from_time_series(tau_local_sec, signal_window, freqs_eval)
        snr_window = optimal_snr(signal_tilde, psd_eval, args.df_hz, valid_mask=valid_eval)

        posterior = DetectorRingdownPosterior(
            freqs_hz=freqs_eval,
            d_tilde=d_tilde,
            psd=psd_eval,
            df_hz=args.df_hz,
            duration_sec=duration_sec,
            h_peak=h_peak,
            n_overtones=n,
            m_total_msun=args.m_total_msun,
            mf_bounds=mf_bounds,
            chif_bounds=chif_bounds,
            amp_bounds_rel=(args.amp_min_rel, args.amp_max_rel),
            phi_bounds=(0.0, 2.0 * np.pi),
            qnm_chi_grid_size=args.qnm_chi_grid_size,
        )
        p0 = posterior.sample_initial_walkers(
            rng,
            args.nwalkers,
            true_mf_msun=true_mf,
            true_chif=true_chif,
            init_strategy=args.init_strategy,
            init_mf_sigma_msun=args.init_mf_sigma_msun,
            init_chif_sigma=args.init_chif_sigma,
        )
        chain, flat, acc, tau_mf, tau_chif, ess = run_emcee_chain(
            posterior=posterior,
            p0=p0,
            burnin_steps=args.emcee_burnin,
            prod_steps=args.emcee_steps,
        )
        healthy = np.isfinite(acc) and acc >= args.min_acceptance and np.isfinite(tau_mf) and np.isfinite(tau_chif)
        if args.require_health and not healthy:
            raise RuntimeError(
                f"health check failed for {item['label']}: acceptance={acc}, tau_mf={tau_mf}, tau_chif={tau_chif}"
            )

        mf = flat[:, 0]
        chif = flat[:, 1]
        hist2d, _, _ = np.histogram2d(mf, chif, bins=[mf_grid, chif_grid], density=False)
        pdf2d = hist2d.T
        if np.sum(pdf2d) <= 0:
            raise RuntimeError(f"invalid posterior histogram for {item['label']}")
        pdf2d /= np.sum(pdf2d)
        truth_rank = truth_hpd_rank(pdf2d, true_mf, true_chif, mf_bounds, chif_bounds)

        pdf2d_by_label[item["label"]] = pdf2d
        level90_by_label[item["label"]] = credible_level_2d(pdf2d, 0.9)
        post_mf_by_label[item["label"]] = normalize_1d_pdf(
            mf_centers,
            np.histogram(mf, bins=mf_grid, density=False)[0].astype(float),
        )
        post_chif_by_label[item["label"]] = normalize_1d_pdf(
            chif_centers,
            np.histogram(chif, bins=chif_grid, density=False)[0].astype(float),
        )

        mf_q16, mf_q50, mf_q84 = [float(v) for v in np.quantile(mf, [0.16, 0.5, 0.84])]
        chif_q16, chif_q50, chif_q84 = [float(v) for v in np.quantile(chif, [0.16, 0.5, 0.84])]
        realized_ms = float(tau_window_m[0] * m_sec * 1e3)
        diag_rows.append(
            f"{item['label']},{n},{dt0_ms:.6f},{realized_ms:.6f},{args.nwalkers},{args.emcee_burnin},{args.emcee_steps},"
            f"{acc:.6f},{tau_mf:.3f},{tau_chif:.3f},{ess:.2f},{snr_window:.6f},"
            f"{mf_q16:.6f},{mf_q50:.6f},{mf_q84:.6f},{chif_q16:.6f},{chif_q50:.6f},{chif_q84:.6f},"
            f"{truth_rank:.6f},{int(truth_rank <= 0.9)},{int(healthy)}"
        )
        if args.samples_prefix is not None:
            out_path = args.samples_prefix.with_name(
                f"{args.samples_prefix.stem}_N{n}_{format_dt0_tag(realized_ms)}_emcee.npz"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_path, chain=chain, flat=flat)

        print(
            f"{item['label']}: realized_dt0_ms={realized_ms:.3f} snr={snr_window:.3f} "
            f"acc={acc:.4f} tau=({tau_mf:.1f},{tau_chif:.1f}) "
            f"mf_q50={mf_q50:.3f} chif_q50={chif_q50:.3f} truth_rank={truth_rank:.3f}"
        )

    fig = plt.figure(figsize=(9.2, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(4, 4, hspace=0.05, wspace=0.05)
    ax_top = fig.add_subplot(gs[0, 0:3])
    ax_main = fig.add_subplot(gs[1:4, 0:3], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

    handles: list[mlines.Line2D] = []
    for item in series:
        label = str(item["label"])
        ax_main.contour(
            mf_centers,
            chif_centers,
            pdf2d_by_label[label],
            levels=[level90_by_label[label]],
            colors=[item["color"]],
            linestyles=[item["ls"]],
            linewidths=2.0,
        )
        ax_top.plot(mf_centers, post_mf_by_label[label], color=item["color"], ls=item["ls"], lw=1.8)
        ax_right.plot(post_chif_by_label[label], chif_centers, color=item["color"], ls=item["ls"], lw=1.8)
        handles.append(mlines.Line2D([], [], color=item["color"], ls=item["ls"], lw=2.0, label=label))

    ax_main.axvline(true_mf, color="k", ls=":", lw=1.2)
    ax_main.axhline(true_chif, color="k", ls=":", lw=1.2)
    ax_main.plot(true_mf, true_chif, marker="+", color="k", ms=10, mew=1.5)
    ax_main.set_xlabel(r"$M_f\ [M_\odot]$")
    ax_main.set_ylabel(r"$\chi_f$")
    ax_main.set_xlim(args.mf_min_msun, args.mf_max_msun)
    ax_main.set_ylim(args.chif_min, chif_bounds[1])
    ax_main.grid(True, alpha=0.15)
    ax_main.legend(handles=handles, loc="upper left", fontsize=8)

    ax_top.set_ylabel("Posterior")
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.grid(True, alpha=0.15)
    ax_top.set_title("Fig.11-style posteriors with shared injection and noise")

    ax_right.set_xlabel("Posterior")
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.grid(True, alpha=0.15)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)

    args.diag_csv.parent.mkdir(parents=True, exist_ok=True)
    args.diag_csv.write_text("\n".join(diag_rows) + "\n", encoding="utf-8")

    print(f"true_mf_msun={true_mf:.6f}, true_chif={true_chif:.6f}")
    print(f"t_hpeak_shift_m={t_hpeak:.6f}")
    print(f"snr_before={snr_before:.3f}, snr_scale={snr_scale:.6f}, snr_after={snr_after:.3f}")
    print(f"output={args.output}")
    print(f"diag_csv={args.diag_csv}")


if __name__ == "__main__":
    main()
