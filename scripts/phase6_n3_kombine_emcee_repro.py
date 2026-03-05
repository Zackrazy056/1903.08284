from __future__ import annotations

import argparse
from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np
from kombine import Sampler as KombineSampler

from ringdown.fd_likelihood import (
    aligo_zero_det_high_power_psd,
    continuous_ft_from_time_series,
    draw_colored_noise_rfft,
    optimal_snr,
)
from ringdown.frequencies import kerr_qnm_omegas_22n
from ringdown.sxs_io import load_sxs_waveform22


MSUN_SEC = 4.92549095e-6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sxs-location", type=str, default="SXS:BBH:0305v2.0/Lev6")
    p.add_argument("--no-download", action="store_true")
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
    p.add_argument("--n-overtones", type=int, default=3)
    p.add_argument("--qnm-chi-grid-size", type=int, default=240)
    p.add_argument("--nwalkers", type=int, default=64)
    p.add_argument("--kombine-burnin", type=int, default=400)
    p.add_argument("--kombine-steps", type=int, default=1200)
    p.add_argument("--kombine-segments", type=int, default=1)
    p.add_argument("--kombine-segment-steps", type=int, default=1200)
    p.add_argument("--tail-steps", type=int, default=2400)
    p.add_argument("--emcee-steps", type=int, default=1200)
    p.add_argument("--emcee-segment-steps", type=int, default=1200)
    p.add_argument("--emcee-tail-steps", type=int, default=6000)
    p.add_argument("--emcee-init", choices=["kombine_last", "best_ball"], default="kombine_last")
    p.add_argument("--emcee-jitter-scale", type=float, default=0.15)
    p.add_argument("--emcee-move", choices=["stretch", "de"], default="stretch")
    p.add_argument("--emcee-stretch-a", type=float, default=2.0)
    p.add_argument("--init-mode", choices=["prior", "truth_ball"], default="prior")
    p.add_argument("--init-mf-sigma", type=float, default=1.5)
    p.add_argument("--init-chif-sigma", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--resume-state", type=Path, default=None)
    p.add_argument("--save-state", type=Path, default=None)
    p.add_argument("--emcee-resume-state", type=Path, default=None)
    p.add_argument("--emcee-save-state", type=Path, default=None)
    p.add_argument("--min-acceptance", type=float, default=0.01)
    p.add_argument("--output", type=Path, default=Path("results/fig10_n3_kombine_emcee_compare.png"))
    p.add_argument("--diag-csv", type=Path, default=Path("results/fig10_n3_kombine_emcee_compare_diag.csv"))
    p.add_argument("--samples-prefix", type=Path, default=Path("results/fig10_n3_kombine_emcee_compare_samples.npz"))
    return p.parse_args()


def credible_level_2d(pdf: np.ndarray, cred: float = 0.9) -> float:
    flat = pdf.ravel()
    order = np.argsort(flat)[::-1]
    cdf = np.cumsum(flat[order])
    idx = int(np.searchsorted(cdf, cred, side="left"))
    idx = min(max(idx, 0), order.size - 1)
    return float(flat[order[idx]])


class N3Posterior:
    def __init__(
        self,
        *,
        freqs_hz: np.ndarray,
        d_tilde: np.ndarray,
        psd: np.ndarray,
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
        self.h_peak = float(h_peak)
        self.mf_bounds = mf_bounds
        self.chif_bounds = chif_bounds
        self.amp_bounds_rel = amp_bounds_rel
        self.phi_bounds = phi_bounds

        valid = (psd > 0.0) & np.isfinite(psd) & (freqs_hz > 0.0)
        if np.count_nonzero(valid) < 16:
            raise ValueError("too few valid frequency bins")
        self.f_calc = freqs_hz[valid]
        self.psd_calc = psd[valid]
        self.d_calc = d_tilde[valid]
        self.df = float(self.f_calc[1] - self.f_calc[0])
        self.d_weighted = self.d_calc / self.psd_calc

        if qnm_chi_grid_size < 64:
            raise ValueError("qnm chi grid size too small")
        chi_lo = max(self.chif_bounds[0], 0.0)
        chi_hi = min(self.chif_bounds[1], 0.999)
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
        omegas_m = np.empty(self.n_modes, dtype=complex)
        for k in range(self.n_modes):
            wr = np.interp(chif, self.chi_grid, self.qnm_re[k])
            wi = np.interp(chif, self.chi_grid, self.qnm_im[k])
            omegas_m[k] = (wr + 1j * wi) / mf_frac
        omegas_rad_s = omegas_m / self.m_sec

        c_n = amps * np.exp(-1j * phis)
        freq_diff = 2.0 * np.pi * self.f_calc[None, :] - omegas_rad_s[:, None]
        h_modes = (1j * c_n[:, None]) / freq_diff
        h_tilde = np.sum(h_modes, axis=0)
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


def init_walkers_truth_ball(
    posterior: N3Posterior,
    rng: np.random.Generator,
    nwalkers: int,
    true_mf: float,
    true_chif: float,
    mf_sigma: float,
    chif_sigma: float,
) -> np.ndarray:
    p = posterior.sample_prior(rng, nwalkers)
    p[:, 0] = np.clip(rng.normal(true_mf, mf_sigma, size=nwalkers), posterior.mf_bounds[0], posterior.mf_bounds[1])
    p[:, 1] = np.clip(
        rng.normal(true_chif, chif_sigma, size=nwalkers),
        posterior.chif_bounds[0] + 1e-6,
        posterior.chif_bounds[1] - 1e-6,
    )
    return p


def robust_tau_ess(chain: np.ndarray) -> tuple[float, float, float]:
    # chain: [steps, walkers, ndim]
    try:
        tau = emcee.autocorr.integrated_time(chain, tol=0, quiet=True)
        tau_mf = float(tau[0])
        tau_chi = float(tau[1])
        ess = float(chain.shape[0] * chain.shape[1] / max(np.max(tau[:2]), 1.0))
        return tau_mf, tau_chi, ess
    except Exception:
        return float("nan"), float("nan"), float("nan")


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


def health_failed(acc: float, tau_mf: float, tau_chif: float, min_acceptance: float) -> tuple[bool, str]:
    if not np.isfinite(acc):
        return True, "acceptance is not finite"
    if acc < min_acceptance:
        return True, f"acceptance={acc:.6f} < {min_acceptance:.6f}"
    if not np.isfinite(tau_mf) or not np.isfinite(tau_chif):
        return True, f"tau invalid (tau_mf={tau_mf}, tau_chif={tau_chif})"
    return False, ""


def update_tail_chain(prev_tail: np.ndarray | None, seg_chain: np.ndarray, keep_steps: int) -> np.ndarray:
    if prev_tail is None or prev_tail.size == 0:
        out = seg_chain
    else:
        out = np.concatenate([prev_tail, seg_chain], axis=0)
    if keep_steps > 0 and out.shape[0] > keep_steps:
        out = out[-keep_steps:, :, :]
    return out


def init_emcee_from_best_ball(
    posterior: N3Posterior,
    rng: np.random.Generator,
    nwalkers: int,
    base_pos: np.ndarray,
    base_lnpost: np.ndarray,
    jitter_scale: float,
) -> np.ndarray:
    if base_pos.shape[0] < 4:
        return np.asarray(base_pos, dtype=float)
    order = np.argsort(base_lnpost)
    n_top = max(4, nwalkers // 4)
    top = np.asarray(base_pos[order[-n_top:]], dtype=float)
    draw_idx = rng.integers(0, top.shape[0], size=nwalkers)
    p = top[draw_idx].copy()

    amp_span = posterior.amp_bounds_rel[1] - posterior.amp_bounds_rel[0]
    p[:, 0] += rng.normal(0.0, 0.35 * jitter_scale, size=nwalkers)  # Mf [Msun]
    p[:, 1] += rng.normal(0.0, 0.015 * jitter_scale, size=nwalkers)  # chif
    for k in range(posterior.n_modes):
        ia = 2 + 2 * k
        ip = 3 + 2 * k
        p[:, ia] += rng.normal(0.0, 0.06 * amp_span * jitter_scale, size=nwalkers)
        p[:, ip] += rng.normal(0.0, 0.25 * jitter_scale, size=nwalkers)

    p[:, 0] = np.clip(p[:, 0], posterior.mf_bounds[0], posterior.mf_bounds[1])
    p[:, 1] = np.clip(p[:, 1], posterior.chif_bounds[0] + 1e-6, posterior.chif_bounds[1] - 1e-6)
    for k in range(posterior.n_modes):
        ia = 2 + 2 * k
        ip = 3 + 2 * k
        p[:, ia] = np.clip(p[:, ia], posterior.amp_bounds_rel[0], posterior.amp_bounds_rel[1])
        p[:, ip] = np.mod(p[:, ip], 2.0 * np.pi)
    return p


def main() -> None:
    args = parse_args()
    if args.n_overtones != 3:
        raise ValueError("this script is dedicated to N=3 reproduction; set --n-overtones 3")
    if args.nwalkers <= 2 + 2 * (args.n_overtones + 1):
        raise ValueError("nwalkers must be larger than ndim")
    if args.init_mode != "prior":
        raise ValueError("init_mode is forced to 'prior' for unbiased N=3 reproduction")
    if args.kombine_segments < 1:
        raise ValueError("kombine-segments must be >= 1")
    if args.kombine_segment_steps < 1:
        raise ValueError("kombine-segment-steps must be >= 1")
    if args.tail_steps < 1:
        raise ValueError("tail-steps must be >= 1")
    if args.emcee_segment_steps < 1:
        raise ValueError("emcee-segment-steps must be >= 1")
    if args.emcee_tail_steps < 1:
        raise ValueError("emcee-tail-steps must be >= 1")
    rng = np.random.default_rng(args.seed)

    wf, info = load_sxs_waveform22(location=args.sxs_location, download=not args.no_download)
    if info.remnant_mass is None or info.remnant_chif_z is None:
        raise ValueError("missing remnant metadata")

    t_all = wf.t
    h22 = wf.h
    h_det_raw = detector_strain_from_mode22(h22)
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
    snr_scale = float(args.target_snr_postpeak / snr_before)
    h_u = h_u * snr_scale
    signal_tilde = continuous_ft_from_time_series(tau_u_sec, h_u, freqs)
    snr_after = optimal_snr(signal_tilde, psd, args.df_hz, valid_mask=valid)

    noise = draw_colored_noise_rfft(rng, freqs.size, psd, args.df_hz, enforce_real_endpoints=True)
    d_tilde = signal_tilde + noise
    h_peak = float(np.max(np.abs(h))) * snr_scale

    true_mf = float(info.remnant_mass * args.m_total_msun)
    true_chif = float(info.remnant_chif_z)
    posterior = N3Posterior(
        freqs_hz=freqs,
        d_tilde=d_tilde,
        psd=psd,
        h_peak=h_peak,
        n_overtones=args.n_overtones,
        m_total_msun=args.m_total_msun,
        mf_bounds=(args.mf_min_msun, args.mf_max_msun),
        chif_bounds=(args.chif_min, min(args.chif_max, 0.999)),
        amp_bounds_rel=(args.amp_min_rel, args.amp_max_rel),
        phi_bounds=(0.0, 2.0 * np.pi),
        qnm_chi_grid_size=args.qnm_chi_grid_size,
    )

    p0 = posterior.sample_prior(rng, args.nwalkers)

    # kombine main run (paper-style KDE ensemble sampler), with optional segmented continuation
    total_prod_steps = args.kombine_steps
    if args.kombine_segments > 1:
        total_prod_steps = args.kombine_segments * args.kombine_segment_steps
    save_state = args.save_state
    if save_state is None:
        save_state = args.diag_csv.with_name(args.diag_csv.stem + "_state.npz")

    tail_chain = None
    resume_loaded = False
    completed_steps = 0
    burnin_done = False
    pos = None
    lnpost = None
    if args.resume_state is not None and args.resume_state.exists():
        rs = np.load(args.resume_state, allow_pickle=False)
        pos = np.asarray(rs["pos"], dtype=float)
        lnpost = np.asarray(rs["lnpost"], dtype=float)
        completed_steps = int(rs["completed_steps"])
        burnin_done = bool(int(rs["burnin_done"]))
        if "tail_chain" in rs.files:
            tail_chain = np.asarray(rs["tail_chain"], dtype=float)
        resume_loaded = True
        if pos.shape != (args.nwalkers, posterior.ndim):
            raise ValueError("resume-state shape mismatch for pos")
        if lnpost.shape != (args.nwalkers,):
            raise ValueError("resume-state shape mismatch for lnpost")

    kombine = KombineSampler(args.nwalkers, posterior.ndim, posterior.log_posterior, processes=1)
    if not burnin_done:
        pos, lnpost, _ = kombine.run_mcmc(args.kombine_burnin, p0)
        burnin_done = True

    remaining = max(total_prod_steps - completed_steps, 0)
    while remaining > 0:
        seg_steps = min(args.kombine_segment_steps, remaining)
        pos, lnpost, _ = kombine.run_mcmc(seg_steps, pos, lnpost0=lnpost)
        seg_chain = np.asarray(kombine.chain[-seg_steps:, :, :], dtype=float)
        tail_chain = update_tail_chain(tail_chain, seg_chain, args.tail_steps)
        completed_steps += seg_steps
        remaining -= seg_steps
        save_state.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_state,
            pos=pos,
            lnpost=lnpost,
            completed_steps=np.asarray(completed_steps, dtype=np.int64),
            burnin_done=np.asarray(1 if burnin_done else 0, dtype=np.int64),
            tail_chain=tail_chain,
        )
        print(f"[kombine] completed_steps={completed_steps}/{total_prod_steps}")

    if tail_chain is None:
        raise RuntimeError("empty kombine chain")
    k_chain = np.asarray(tail_chain, dtype=float)
    k_flat = k_chain.reshape(-1, posterior.ndim)
    k_acc = float(np.mean(kombine.acceptance_fraction)) if np.size(kombine.acceptance_fraction) > 0 else float("nan")

    # emcee verification run (same posterior), with true checkpoint continuation support
    e_chain = None
    e_flat = None
    e_acc = float("nan")
    e_completed_steps = 0
    e_resume_loaded = False
    if args.emcee_steps > 0:
        emcee_save_state = args.emcee_save_state
        if emcee_save_state is None:
            emcee_save_state = args.diag_csv.with_name(args.diag_csv.stem + "_emcee_state.npz")
        emcee_resume_state = args.emcee_resume_state
        if emcee_resume_state is None:
            emcee_resume_state = emcee_save_state

        e_tail_chain = None
        e_pos = None
        e_acc_sum = 0.0
        e_acc_steps = 0
        if emcee_resume_state is not None and emcee_resume_state.exists():
            es = np.load(emcee_resume_state, allow_pickle=False)
            e_pos = np.asarray(es["emcee_pos"], dtype=float)
            e_completed_steps = int(es["emcee_completed_steps"])
            if "emcee_tail_chain" in es.files:
                e_tail_chain = np.asarray(es["emcee_tail_chain"], dtype=float)
            if "emcee_acc_sum" in es.files:
                e_acc_sum = float(es["emcee_acc_sum"])
            if "emcee_acc_steps" in es.files:
                e_acc_steps = int(es["emcee_acc_steps"])
            if e_pos.shape != (args.nwalkers, posterior.ndim):
                raise ValueError("emcee-resume-state shape mismatch for emcee_pos")
            e_resume_loaded = True

        if e_pos is None:
            e_pos = np.asarray(pos, dtype=float)
            if args.emcee_init == "best_ball":
                e_pos = init_emcee_from_best_ball(
                    posterior,
                    rng,
                    args.nwalkers,
                    np.asarray(pos, dtype=float),
                    np.asarray(lnpost, dtype=float),
                    jitter_scale=max(args.emcee_jitter_scale, 1e-6),
                )

        e_remaining = max(args.emcee_steps - e_completed_steps, 0)
        while e_remaining > 0:
            e_seg = min(args.emcee_segment_steps, e_remaining)
            if args.emcee_move == "de":
                move = emcee.moves.DEMove()
            else:
                move = emcee.moves.StretchMove(a=float(args.emcee_stretch_a))
            em = emcee.EnsembleSampler(
                args.nwalkers,
                posterior.ndim,
                posterior.log_posterior,
                moves=move,
            )
            state = em.run_mcmc(e_pos, e_seg, progress=False)
            seg_chain = em.get_chain()
            e_tail_chain = update_tail_chain(e_tail_chain, seg_chain, args.emcee_tail_steps)
            e_pos = np.asarray(state.coords, dtype=float)
            seg_acc = float(np.mean(em.acceptance_fraction))
            e_acc_sum += seg_acc * e_seg
            e_acc_steps += e_seg
            e_completed_steps += e_seg
            e_remaining -= e_seg
            emcee_save_state.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                emcee_save_state,
                emcee_pos=e_pos,
                emcee_completed_steps=np.asarray(e_completed_steps, dtype=np.int64),
                emcee_tail_chain=e_tail_chain,
                emcee_acc_sum=np.asarray(e_acc_sum, dtype=float),
                emcee_acc_steps=np.asarray(e_acc_steps, dtype=np.int64),
            )
            print(f"[emcee] completed_steps={e_completed_steps}/{args.emcee_steps}")

        if e_tail_chain is not None:
            e_chain = np.asarray(e_tail_chain, dtype=float)
            e_flat = e_chain.reshape(-1, posterior.ndim)
            e_acc = e_acc_sum / max(e_acc_steps, 1)

    # basic diagnostics
    k_mf = k_flat[:, 0]
    k_chi = k_flat[:, 1]
    k_tau_mf, k_tau_chi, k_ess = robust_tau_ess(k_chain)
    k_q = np.quantile(k_mf, [0.16, 0.5, 0.84]).tolist()
    chi_q = np.quantile(k_chi, [0.16, 0.5, 0.84]).tolist()
    e_tau_mf = float("nan")
    e_tau_chi = float("nan")
    e_ess = float("nan")
    e_q = [float("nan"), float("nan"), float("nan")]
    e_chi_q = [float("nan"), float("nan"), float("nan")]
    if e_chain is not None and e_flat is not None:
        e_mf = e_flat[:, 0]
        e_chi = e_flat[:, 1]
        e_tau_mf, e_tau_chi, e_ess = robust_tau_ess(e_chain)
        e_q = np.quantile(e_mf, [0.16, 0.5, 0.84]).tolist()
        e_chi_q = np.quantile(e_chi, [0.16, 0.5, 0.84]).tolist()

    bad, reason = health_failed(k_acc, k_tau_mf, k_tau_chi, args.min_acceptance)
    if bad:
        raise RuntimeError(f"kombine health check failed: {reason}")
    if args.emcee_steps > 0 and e_chain is not None:
        bad, reason = health_failed(e_acc, e_tau_mf, e_tau_chi, args.min_acceptance)
        if bad:
            raise RuntimeError(f"emcee health check failed: {reason}")

    # contour comparison figure
    mf_grid = np.linspace(args.mf_min_msun, args.mf_max_msun, 180)
    chi_grid = np.linspace(args.chif_min, min(args.chif_max, 0.999), 180)
    k_h2, _, _ = np.histogram2d(k_mf, k_chi, bins=[mf_grid, chi_grid], density=False)
    k_pdf = (k_h2.T / np.sum(k_h2)) if np.sum(k_h2) > 0 else np.zeros_like(k_h2.T)
    k_level = credible_level_2d(k_pdf, 0.9)
    e_pdf = None
    e_level = None
    if e_chain is not None and e_flat is not None:
        e_h2, _, _ = np.histogram2d(e_flat[:, 0], e_flat[:, 1], bins=[mf_grid, chi_grid], density=False)
        e_pdf = (e_h2.T / np.sum(e_h2)) if np.sum(e_h2) > 0 else np.zeros_like(k_h2.T)
        e_level = credible_level_2d(e_pdf, 0.9)

    mf_c = 0.5 * (mf_grid[:-1] + mf_grid[1:])
    chi_c = 0.5 * (chi_grid[:-1] + chi_grid[1:])
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.contour(mf_c, chi_c, k_pdf, levels=[k_level], colors=["#d62728"], linewidths=2.0, linestyles=["-"])
    if e_pdf is not None and e_level is not None:
        ax.contour(mf_c, chi_c, e_pdf, levels=[e_level], colors=["#1f77b4"], linewidths=2.0, linestyles=["--"])
    ax.axvline(true_mf, color="k", ls=":", lw=1.2)
    ax.axhline(true_chif, color="k", ls=":", lw=1.2)
    ax.set_xlabel(r"$M_f\ [M_\odot]$")
    ax.set_ylabel(r"$\chi_f$")
    ax.set_xlim(args.mf_min_msun, args.mf_max_msun)
    ax.set_ylim(args.chif_min, min(args.chif_max, 0.999))
    ax.grid(True, alpha=0.2)
    if e_pdf is not None:
        ax.legend(
            [
                plt.Line2D([], [], color="#d62728", lw=2.0, ls="-"),
                plt.Line2D([], [], color="#1f77b4", lw=2.0, ls="--"),
            ],
            ["kombine (90%)", "emcee verify (90%)"],
            loc="upper left",
        )
    else:
        ax.legend(
            [plt.Line2D([], [], color="#d62728", lw=2.0, ls="-")],
            ["kombine (90%)"],
            loc="upper left",
        )
    ax.set_title("N=3 posterior: kombine vs emcee")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)

    args.diag_csv.parent.mkdir(parents=True, exist_ok=True)
    args.diag_csv.write_text(
        "sampler,nwalkers,burnin_or_init,steps,acceptance,tau_mf,tau_chif,ess,mf_q16,mf_q50,mf_q84,chif_q16,chif_q50,chif_q84,snr_after\n"
        f"kombine,{args.nwalkers},{args.kombine_burnin},{completed_steps},{k_acc:.6f},{k_tau_mf:.3f},{k_tau_chi:.3f},{k_ess:.2f},"
        f"{k_q[0]:.6f},{k_q[1]:.6f},{k_q[2]:.6f},{chi_q[0]:.6f},{chi_q[1]:.6f},{chi_q[2]:.6f},{snr_after:.6f}\n"
        + (
            f"emcee,{args.nwalkers},0,{e_completed_steps},{e_acc:.6f},{e_tau_mf:.3f},{e_tau_chi:.3f},{e_ess:.2f},"
            f"{e_q[0]:.6f},{e_q[1]:.6f},{e_q[2]:.6f},{e_chi_q[0]:.6f},{e_chi_q[1]:.6f},{e_chi_q[2]:.6f},{snr_after:.6f}\n"
            if args.emcee_steps > 0 and e_chain is not None
            else ""
        ),
        encoding="utf-8",
    )

    if args.samples_prefix is not None:
        out_k = args.samples_prefix.with_name(args.samples_prefix.stem + "_kombine.npz")
        np.savez_compressed(out_k, chain=k_chain, flat=k_flat)
        if e_chain is not None and e_flat is not None:
            out_e = args.samples_prefix.with_name(args.samples_prefix.stem + "_emcee.npz")
            np.savez_compressed(out_e, chain=e_chain, flat=e_flat)

    print(f"true_mf_msun={true_mf:.6f}, true_chif={true_chif:.6f}")
    print(f"t_hpeak_shift_m={t_hpeak:.6f}")
    print(f"snr_before={snr_before:.3f}, snr_scale={snr_scale:.6f}, snr_after={snr_after:.3f}")
    print(f"kombine_acc={k_acc:.4f}, emcee_acc={e_acc:.4f}")
    print(f"resume_loaded={resume_loaded}, completed_steps={completed_steps}, state={save_state}")
    print(f"emcee_resume_loaded={e_resume_loaded}, emcee_completed_steps={e_completed_steps}")
    print(f"diag_csv={args.diag_csv}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
