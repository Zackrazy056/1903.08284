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
    optimal_snr,
    real_ringdown_mode_tilde,
)
from ringdown.frequencies import kerr_qnm_omegas_22n
from ringdown.sxs_io import load_sxs_waveform22


MSUN_SEC = 4.92549095e-6


def parse_int_list(text: str) -> list[int]:
    vals = [int(s.strip()) for s in text.split(",") if s.strip()]
    if not vals:
        raise ValueError("empty integer list")
    return sorted(set(vals))


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


def detector_strain_from_mode22(h22: np.ndarray) -> np.ndarray:
    if h22.ndim != 1:
        raise ValueError("h22 must be 1D")
    i_ref = int(np.argmax(np.abs(h22)))
    href = h22[i_ref]
    # 用峰值点相位做全局旋转，把 h22 对齐到“plus-like”实通道。
    # 这一步决定后续 t_h-peak 与振幅归一化口径。
    phase_ref = float(np.angle(href)) if np.abs(href) > 0 else 0.0
    return np.real(h22 * np.exp(-1j * phase_ref))


def peak_time_from_detector_strain(t: np.ndarray, h_det: np.ndarray) -> float:
    if t.ndim != 1 or h_det.ndim != 1 or t.size != h_det.size:
        raise ValueError("t and h_det must be 1D arrays with identical length")
    idx = int(np.argmax(np.abs(h_det)))
    return float(t[idx])


def health_failed(acc: float, tau_mf: float, tau_chif: float, min_acceptance: float) -> tuple[bool, str]:
    if not np.isfinite(acc):
        return True, "acceptance is not finite"
    # 采样健康门禁：拒绝低接受率或 tau 非法的结果，避免假收敛出图。
    if acc < min_acceptance:
        return True, f"acceptance={acc:.6f} < {min_acceptance:.6f}"
    if not np.isfinite(tau_mf) or not np.isfinite(tau_chif):
        return True, f"tau invalid (tau_mf={tau_mf}, tau_chif={tau_chif})"
    return False, ""


class RingdownPosterior:
    def __init__(
        self,
        *,
        freqs_hz: np.ndarray,
        d_tilde: np.ndarray,
        psd: np.ndarray,
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
        self.m_total_msun = float(m_total_msun)
        self.m_sec = MSUN_SEC * self.m_total_msun
        self.duration_sec = float(duration_sec)
        self.h_peak = float(h_peak)
        self.mf_bounds = mf_bounds
        self.chif_bounds = chif_bounds
        self.amp_bounds_rel = amp_bounds_rel
        self.phi_bounds = phi_bounds

        valid = (psd > 0.0) & np.isfinite(psd) & (freqs_hz > 0.0)
        if np.count_nonzero(valid) < 16:
            raise ValueError("too few valid frequency bins")
        # 仅在有效频段计算似然，减少无信息/病态频点对后验的污染。
        self.f_calc = freqs_hz[valid]
        self.psd_calc = psd[valid]
        self.d_calc = d_tilde[valid]
        self.df = float(self.f_calc[1] - self.f_calc[0])
        self.d_weighted = self.d_calc / self.psd_calc

        chi_lo = max(self.chif_bounds[0], 0.0)
        chi_hi = min(self.chif_bounds[1], 0.999)
        if qnm_chi_grid_size < 64:
            raise ValueError("qnm chi grid too small")
        self.chi_grid = np.linspace(chi_lo, chi_hi, qnm_chi_grid_size)
        self.qnm_re = np.empty((self.n_modes, self.chi_grid.size), dtype=float)
        self.qnm_im = np.empty((self.n_modes, self.chi_grid.size), dtype=float)
        for j, c in enumerate(self.chi_grid):
            vals = kerr_qnm_omegas_22n(mf=1.0, chif=float(c), n_max=self.n_overtones)
            self.qnm_re[:, j] = vals.real
            self.qnm_im[:, j] = vals.imag

    def sample_prior(self, rng: np.random.Generator, nwalkers: int) -> np.ndarray:
        p = np.empty((nwalkers, self.ndim), dtype=float)
        p[:, 0] = rng.uniform(self.mf_bounds[0], self.mf_bounds[1], size=nwalkers)
        p[:, 1] = rng.uniform(self.chif_bounds[0], self.chif_bounds[1], size=nwalkers)
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

    def log_likelihood(self, theta: np.ndarray) -> float:
        mf = float(theta[0])
        chif = float(theta[1])
        mf_frac = mf / self.m_total_msun
        if mf_frac <= 0.0:
            return -np.inf

        amps = theta[2::2] * self.h_peak
        phis = theta[3::2]
        # 由 (Mf, chi_f) 插值出 QNM 频率，是 remnant 参数进入模型的主通道。
        omegas_m = np.empty(self.n_modes, dtype=complex)
        for k in range(self.n_modes):
            wr = np.interp(chif, self.chi_grid, self.qnm_re[k])
            wi = np.interp(chif, self.chi_grid, self.qnm_im[k])
            omegas_m[k] = (wr + 1j * wi) / mf_frac
        omegas_rad_s = omegas_m / self.m_sec

        h_tilde = real_ringdown_mode_tilde(
            self.f_calc,
            omegas_rad_s,
            amps,
            phis,
            duration_sec=self.duration_sec,
        )
        # PSD 加权高斯似然：lnL = <d,h> - 1/2<h,h>。
        d_h = 4.0 * self.df * np.sum(np.real(self.d_weighted * np.conjugate(h_tilde)))
        h_h = 4.0 * self.df * np.sum((np.abs(h_tilde) ** 2) / self.psd_calc)
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
    posterior: RingdownPosterior,
    p0: np.ndarray,
    burnin_steps: int,
    prod_steps: int,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    # burn-in 后 reset，再统计生产链，避免把预热段混入后验统计。
    em = emcee.EnsembleSampler(p0.shape[0], posterior.ndim, posterior.log_posterior)
    state = em.run_mcmc(p0, burnin_steps, progress=False)
    em.reset()
    em.run_mcmc(state, prod_steps, progress=False)
    chain = em.get_chain()
    flat = em.get_chain(flat=True)
    acc = float(np.mean(em.acceptance_fraction))
    tau_mf, tau_ch, ess = robust_tau_ess(chain)
    return chain, flat, acc, tau_mf, tau_ch, ess


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sxs-location", type=str, default="SXS:BBH:0305v2.0/Lev6")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--n-values", type=str, default="0,1,2,3")
    p.add_argument("--m-total-msun", type=float, default=72.0)
    p.add_argument("--target-hpeak", type=float, default=2e-21)
    p.add_argument("--target-snr-postpeak", type=float, default=42.3)
    p.add_argument("--delta-t0-ms", type=float, default=0.0)
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
    p.add_argument("--nwalkers", type=int, default=128)
    p.add_argument("--emcee-burnin", type=int, default=3000)
    p.add_argument("--emcee-steps", type=int, default=14000)
    p.add_argument("--emcee-alt-burnin", type=int, default=3000)
    p.add_argument("--emcee-alt-steps", type=int, default=14000)
    p.add_argument("--n3-alt-init", choices=["from_primary", "prior"], default="from_primary")
    p.add_argument("--n3-max-mf-diff", type=float, default=2.0)
    p.add_argument("--n3-max-chif-diff", type=float, default=0.06)
    p.add_argument("--min-acceptance", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--output", type=Path, default=Path("results/fig10_emcee_full_strict.png"))
    p.add_argument("--diag-csv", type=Path, default=Path("results/fig10_emcee_full_strict_diag.csv"))
    p.add_argument("--samples-prefix", type=Path, default=Path("results/fig10_emcee_full_strict_samples.npz"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n_values = parse_int_list(args.n_values)
    if any(n < 0 for n in n_values):
        raise ValueError("n-values must be nonnegative")
    ndim_max = 2 + 2 * (max(n_values) + 1)
    if args.nwalkers <= ndim_max:
        raise ValueError(f"nwalkers must be > max ndim ({ndim_max})")

    rng = np.random.default_rng(args.seed)
    wf, info = load_sxs_waveform22(location=args.sxs_location, download=not args.no_download)
    if info.remnant_mass is None or info.remnant_chif_z is None:
        raise ValueError("missing remnant metadata")

    t_all = wf.t
    h22 = wf.h
    h_det_raw = detector_strain_from_mode22(h22)
    # 以 detector-strain 峰值定义 t_h-peak，把时间轴平移到该峰值为 0。
    t_hpeak = peak_time_from_detector_strain(t_all, h_det_raw)
    t_all = t_all - t_hpeak
    h_peak_raw = float(np.max(np.abs(h_det_raw)))
    if h_peak_raw <= 0:
        raise RuntimeError("detector strain has non-positive peak amplitude")
    h = h_det_raw * (args.target_hpeak / h_peak_raw)

    m_sec = MSUN_SEC * args.m_total_msun
    ms_per_m = m_sec * 1e3
    t0 = float(args.delta_t0_ms / ms_per_m)
    mask = (t_all >= t0) & (t_all <= args.t_end)
    t = t_all[mask]
    h = h[mask]
    if t.size < 64:
        raise ValueError("analysis window too short")
    tau = t - t0
    dt = float(np.median(np.diff(tau)))
    tau_u = np.arange(0.0, float(tau[-1]) + 0.5 * dt, dt)
    h_u = np.interp(tau_u, tau, h)
    tau_u_sec = tau_u * m_sec

    freqs = np.arange(args.f_min_hz, args.f_max_hz + 0.5 * args.df_hz, args.df_hz)
    psd = aligo_zero_det_high_power_psd(freqs, f_low_hz=10.0)
    valid = (freqs >= args.f_min_hz) & (freqs <= args.f_max_hz) & np.isfinite(psd) & (psd > 0.0)
    signal_tilde = continuous_ft_from_time_series(tau_u_sec, h_u, freqs)
    snr_before = optimal_snr(signal_tilde, psd, args.df_hz, valid_mask=valid)
    # 先统一注入信号到目标 post-peak SNR（论文常用 ~42.3）。
    snr_scale = float(args.target_snr_postpeak / snr_before)
    h_u = h_u * snr_scale
    signal_tilde = continuous_ft_from_time_series(tau_u_sec, h_u, freqs)
    snr_after = optimal_snr(signal_tilde, psd, args.df_hz, valid_mask=valid)
    # 噪声按同一 PSD 生成，构造 d(f)=h(f)+n(f) 的观测数据。
    noise = draw_colored_noise_rfft(rng, freqs.size, psd, args.df_hz, enforce_real_endpoints=True)
    d_tilde = signal_tilde + noise
    h_peak = float(np.max(np.abs(h))) * snr_scale

    true_mf = float(info.remnant_mass * args.m_total_msun)
    true_chif = float(info.remnant_chif_z)

    colors = {0: "#1f77b4", 1: "#6f2da8", 2: "#d4a017", 3: "#d62728"}
    linestyles = {0: "-", 1: "--", 2: "--", 3: "-"}
    mf_grid = np.linspace(args.mf_min_msun, args.mf_max_msun, 220)
    chif_grid = np.linspace(args.chif_min, min(args.chif_max, 0.999), 220)

    pdf2d_by_n: dict[int, np.ndarray] = {}
    post_mf_by_n: dict[int, np.ndarray] = {}
    post_chif_by_n: dict[int, np.ndarray] = {}
    level90_by_n: dict[int, float] = {}

    diag_rows = [
        "N,sampler,nwalkers,burnin_or_init,steps,acceptance,tau_mf,tau_chif,ess,mf_q16,mf_q50,mf_q84,chif_q16,chif_q50,chif_q84,snr_after"
    ]

    for n in n_values:
        posterior = RingdownPosterior(
            freqs_hz=freqs,
            d_tilde=d_tilde,
            psd=psd,
            duration_sec=float(tau_u_sec[-1]),
            h_peak=h_peak,
            n_overtones=n,
            m_total_msun=args.m_total_msun,
            mf_bounds=(args.mf_min_msun, args.mf_max_msun),
            chif_bounds=(args.chif_min, min(args.chif_max, 0.999)),
            amp_bounds_rel=(args.amp_min_rel, args.amp_max_rel),
            phi_bounds=(0.0, 2.0 * np.pi),
            qnm_chi_grid_size=args.qnm_chi_grid_size,
        )
        p0 = posterior.sample_prior(rng, args.nwalkers)
        chain, flat, acc, tau_mf, tau_ch, ess = run_emcee_chain(
            posterior=posterior,
            p0=p0,
            burnin_steps=args.emcee_burnin,
            prod_steps=args.emcee_steps,
        )
        bad, reason = health_failed(acc, tau_mf, tau_ch, args.min_acceptance)
        if bad:
            raise RuntimeError(f"N={n} emcee health check failed: {reason}")

        mf = flat[:, 0]
        ch = flat[:, 1]
        mf_q16, mf_q50, mf_q84 = [float(v) for v in np.quantile(mf, [0.16, 0.5, 0.84])]
        ch_q16, ch_q50, ch_q84 = [float(v) for v in np.quantile(ch, [0.16, 0.5, 0.84])]
        diag_rows.append(
            f"{n},emcee,{args.nwalkers},{args.emcee_burnin},{args.emcee_steps},{acc:.6f},"
            f"{tau_mf:.3f},{tau_ch:.3f},{ess:.2f},{mf_q16:.6f},{mf_q50:.6f},{mf_q84:.6f},"
            f"{ch_q16:.6f},{ch_q50:.6f},{ch_q84:.6f},{snr_after:.6f}"
        )

        if args.samples_prefix is not None:
            out_e = args.samples_prefix.with_name(args.samples_prefix.stem + f"_N{n}_emcee.npz")
            np.savez_compressed(out_e, chain=chain, flat=flat)

        # N=3 额外做第二条链一致性检查，避免单链偶然落入局部模态。
        if n == 3 and args.emcee_alt_steps > 0:
            rng_alt = np.random.default_rng(args.seed + 100000)
            if args.n3_alt_init == "from_primary":
                p0_alt = np.asarray(chain[-1, :, :], dtype=float)
                p0_alt = p0_alt + rng_alt.normal(0.0, 1e-4, size=p0_alt.shape)
                p0_alt[:, 0] = np.clip(p0_alt[:, 0], args.mf_min_msun, args.mf_max_msun)
                p0_alt[:, 1] = np.clip(p0_alt[:, 1], args.chif_min + 1e-6, min(args.chif_max, 0.999) - 1e-6)
                for k in range(posterior.n_modes):
                    ia = 2 + 2 * k
                    ip = 3 + 2 * k
                    p0_alt[:, ia] = np.clip(p0_alt[:, ia], args.amp_min_rel, args.amp_max_rel)
                    p0_alt[:, ip] = np.mod(p0_alt[:, ip], 2.0 * np.pi)
            else:
                p0_alt = posterior.sample_prior(rng_alt, args.nwalkers)
            chain_a, flat_a, acc_a, tau_mf_a, tau_ch_a, ess_a = run_emcee_chain(
                posterior=posterior,
                p0=p0_alt,
                burnin_steps=args.emcee_alt_burnin,
                prod_steps=args.emcee_alt_steps,
            )
            bad, reason = health_failed(acc_a, tau_mf_a, tau_ch_a, args.min_acceptance)
            if bad:
                raise RuntimeError(f"N=3 emcee_alt health check failed: {reason}")
            mf_a = flat_a[:, 0]
            ch_a = flat_a[:, 1]
            mf_a_q16, mf_a_q50, mf_a_q84 = [float(v) for v in np.quantile(mf_a, [0.16, 0.5, 0.84])]
            ch_a_q16, ch_a_q50, ch_a_q84 = [float(v) for v in np.quantile(ch_a, [0.16, 0.5, 0.84])]
            diag_rows.append(
                f"3,emcee_alt,{args.nwalkers},{args.emcee_alt_burnin},{args.emcee_alt_steps},{acc_a:.6f},"
                f"{tau_mf_a:.3f},{tau_ch_a:.3f},{ess_a:.2f},{mf_a_q16:.6f},{mf_a_q50:.6f},{mf_a_q84:.6f},"
                f"{ch_a_q16:.6f},{ch_a_q50:.6f},{ch_a_q84:.6f},{snr_after:.6f}"
            )
            if args.samples_prefix is not None:
                out_ea = args.samples_prefix.with_name(args.samples_prefix.stem + "_N3_emcee_alt.npz")
                np.savez_compressed(out_ea, chain=chain_a, flat=flat_a)
            mf_diff = float(abs(mf_q50 - mf_a_q50))
            ch_diff = float(abs(ch_q50 - ch_a_q50))
            if mf_diff > args.n3_max_mf_diff or ch_diff > args.n3_max_chif_diff:
                raise RuntimeError(
                    "N=3 consistency check failed: "
                    f"|mf_q50(e-e_alt)|={mf_diff:.4f} (max {args.n3_max_mf_diff:.4f}), "
                    f"|chif_q50(e-e_alt)|={ch_diff:.4f} (max {args.n3_max_chif_diff:.4f})"
                )

        hist2d, _, _ = np.histogram2d(mf, ch, bins=[mf_grid, chif_grid], density=False)
        pdf2d = hist2d.T
        if np.sum(pdf2d) <= 0:
            raise RuntimeError(f"invalid posterior histogram for N={n}")
        pdf2d /= np.sum(pdf2d)
        pdf2d_by_n[n] = pdf2d
        level90_by_n[n] = credible_level_2d(pdf2d, 0.9)
        mf_centers = 0.5 * (mf_grid[:-1] + mf_grid[1:])
        ch_centers = 0.5 * (chif_grid[:-1] + chif_grid[1:])
        post_mf_by_n[n] = normalize_1d_pdf(mf_centers, np.histogram(mf, bins=mf_grid, density=False)[0].astype(float))
        post_chif_by_n[n] = normalize_1d_pdf(ch_centers, np.histogram(ch, bins=chif_grid, density=False)[0].astype(float))

        print(
            f"N={n} emcee_acc={acc:.4f} tau=({tau_mf:.1f},{tau_ch:.1f}) "
            f"mf_q50={mf_q50:.3f} chif_q50={ch_q50:.3f}"
        )

    fig = plt.figure(figsize=(9, 8))
    gs = fig.add_gridspec(4, 4, hspace=0.05, wspace=0.05)
    ax_top = fig.add_subplot(gs[0, 0:3])
    ax_main = fig.add_subplot(gs[1:4, 0:3], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
    mf_centers = 0.5 * (mf_grid[:-1] + mf_grid[1:])
    chif_centers = 0.5 * (chif_grid[:-1] + chif_grid[1:])
    handles: list[mlines.Line2D] = []
    for n in n_values:
        c = colors.get(n, None)
        ls = linestyles.get(n, "-")
        ax_main.contour(mf_centers, chif_centers, pdf2d_by_n[n], levels=[level90_by_n[n]], colors=[c], linestyles=[ls], linewidths=2.0)
        ax_top.plot(mf_centers, post_mf_by_n[n], color=c, ls=ls, lw=1.8)
        ax_right.plot(post_chif_by_n[n], chif_centers, color=c, ls=ls, lw=1.8)
        handles.append(mlines.Line2D([], [], color=c, ls=ls, lw=2.0, label=f"N={n}"))

    ax_main.axvline(true_mf, color="k", ls=":", lw=1.2)
    ax_main.axhline(true_chif, color="k", ls=":", lw=1.2)
    ax_main.set_xlabel(r"$M_f\ [M_\odot]$")
    ax_main.set_ylabel(r"$\chi_f$")
    ax_main.set_xlim(args.mf_min_msun, args.mf_max_msun)
    ax_main.set_ylim(args.chif_min, min(args.chif_max, 0.999))
    ax_main.grid(True, alpha=0.15)
    ax_main.legend(handles=handles, loc="upper left", fontsize=9)
    ax_top.set_ylabel("Posterior")
    ax_top.grid(True, alpha=0.15)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_right.set_xlabel("Posterior")
    ax_right.grid(True, alpha=0.15)
    ax_right.tick_params(axis="y", labelleft=False)
    ax_top.set_title(rf"Fig.10-style posteriors with emcee ($\Delta t_0={args.delta_t0_ms:.3f}$ ms)")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)

    args.diag_csv.parent.mkdir(parents=True, exist_ok=True)
    args.diag_csv.write_text("\n".join(diag_rows) + "\n", encoding="utf-8")
    print(f"true_mf_msun={true_mf:.6f}, true_chif={true_chif:.6f}")
    print(f"t_hpeak_shift_m={t_hpeak:.6f}")
    print(f"n_values={n_values}")
    print(f"snr_before={snr_before:.3f}, snr_scale={snr_scale:.6f}, snr_after={snr_after:.3f}")
    print(f"diag_csv={args.diag_csv}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
