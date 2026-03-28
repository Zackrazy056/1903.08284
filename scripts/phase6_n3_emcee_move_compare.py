from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import emcee
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import phase6_fig10_emcee_full_strict as strict  # noqa: E402
import phase6_n3_posterior_geometry_diagnose as geom  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare N=3 truth-mode occupancy across emcee move choices.")
    p.add_argument("--moves", type=str, default="stretch,de")
    p.add_argument("--run-seeds", type=str, default="101,202,303")
    p.add_argument("--nwalkers", type=int, default=64)
    p.add_argument("--burnin", type=int, default=1000)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--fit-subsample-per-chain", type=int, default=6000)
    p.add_argument("--score-subsample-per-run", type=int, default=6000)
    p.add_argument("--max-components", type=int, default=4)
    p.add_argument("--qnm-chi-grid-size", type=int, default=160)
    p.add_argument("--seed", type=int, default=12345, help="Data/noise seed, kept fixed across move comparisons.")
    p.add_argument("--sxs-location", type=str, default="SXS:BBH:0305v2.0/Lev6")
    p.add_argument("--no-download", action="store_true", default=True)
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
    p.add_argument(
        "--primary-samples",
        type=Path,
        default=Path("results/fig10_n3_strict_long_fix1_samples_N3_emcee.npz"),
    )
    p.add_argument(
        "--alt-samples",
        type=Path,
        default=Path("results/fig10_n3_strict_long_fix1_samples_N3_emcee_alt.npz"),
    )
    p.add_argument("--csv-output", type=Path, default=Path("results/fig10_n3_move_compare_fix1.csv"))
    p.add_argument("--markdown-output", type=Path, default=Path("results/fig10_n3_move_compare_fix1.md"))
    return p.parse_args()


def _parse_int_list(text: str) -> list[int]:
    vals = [int(s.strip()) for s in text.split(",") if s.strip()]
    if not vals:
        raise ValueError("empty integer list")
    return vals


def _parse_moves(text: str) -> list[str]:
    vals = [s.strip().lower() for s in text.split(",") if s.strip()]
    if not vals:
        raise ValueError("empty move list")
    for move in vals:
        if move not in {"stretch", "de"}:
            raise ValueError(f"unsupported move: {move}")
    return vals


def _build_posterior(args: argparse.Namespace) -> strict.RingdownPosterior:
    rng = np.random.default_rng(args.seed)
    wf, info = strict.load_sxs_waveform22(location=args.sxs_location, download=not args.no_download)
    if info.remnant_mass is None or info.remnant_chif_z is None:
        raise ValueError("missing remnant metadata")

    t_all = wf.t
    h22 = wf.h
    h_det_raw = strict.detector_strain_from_mode22(h22)
    t_hpeak = strict.peak_time_from_detector_strain(t_all, h_det_raw)
    t_all = t_all - t_hpeak
    h_peak_raw = float(np.max(np.abs(h_det_raw)))
    h = h_det_raw * (args.target_hpeak / h_peak_raw)

    m_sec = strict.MSUN_SEC * args.m_total_msun
    ms_per_m = m_sec * 1e3
    t0 = float(args.delta_t0_ms / ms_per_m)
    mask = (t_all >= t0) & (t_all <= args.t_end)
    t = t_all[mask]
    h = h[mask]
    tau = t - t0
    dt = float(np.median(np.diff(tau)))
    tau_u = np.arange(0.0, float(tau[-1]) + 0.5 * dt, dt)
    h_u = np.interp(tau_u, tau, h)
    tau_u_sec = tau_u * m_sec

    freqs = np.arange(args.f_min_hz, args.f_max_hz + 0.5 * args.df_hz, args.df_hz)
    psd = strict.aligo_zero_det_high_power_psd(freqs, f_low_hz=10.0)
    valid = (freqs >= args.f_min_hz) & (freqs <= args.f_max_hz) & np.isfinite(psd) & (psd > 0.0)
    signal_tilde = strict.continuous_ft_from_time_series(tau_u_sec, h_u, freqs)
    snr_before = strict.optimal_snr(signal_tilde, psd, args.df_hz, valid_mask=valid)
    snr_scale = float(args.target_snr_postpeak / snr_before)
    h_u = h_u * snr_scale
    signal_tilde = strict.continuous_ft_from_time_series(tau_u_sec, h_u, freqs)
    noise = strict.draw_colored_noise_rfft(rng, freqs.size, psd, args.df_hz, enforce_real_endpoints=True)
    d_tilde = signal_tilde + noise
    h_peak = float(np.max(np.abs(h))) * snr_scale

    return strict.RingdownPosterior(
        freqs_hz=freqs,
        d_tilde=d_tilde,
        psd=psd,
        duration_sec=float(tau_u_sec[-1]),
        h_peak=h_peak,
        n_overtones=3,
        m_total_msun=args.m_total_msun,
        mf_bounds=(args.mf_min_msun, args.mf_max_msun),
        chif_bounds=(args.chif_min, min(args.chif_max, 0.999)),
        amp_bounds_rel=(args.amp_min_rel, args.amp_max_rel),
        phi_bounds=(0.0, 2.0 * np.pi),
        qnm_chi_grid_size=args.qnm_chi_grid_size,
    )


def _fit_reference_gmm(args: argparse.Namespace) -> tuple[object, np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(args.seed)
    primary_chain = geom._load_post_burn_chain(args.primary_samples, 0.5)
    alt_chain = geom._load_post_burn_chain(args.alt_samples, 0.5)
    primary_flat = primary_chain[:, :, :2].reshape(-1, 2)
    alt_flat = alt_chain[:, :, :2].reshape(-1, 2)
    x_fit = np.vstack(
        [
            geom._random_subsample(primary_flat, args.fit_subsample_per_chain, rng),
            geom._random_subsample(alt_flat, args.fit_subsample_per_chain, rng),
        ]
    )
    model, mu, sigma = geom._fit_gmm(x_fit, max_components=args.max_components, rng_seed=args.seed)
    comp_means = np.asarray(model.means_ * sigma + mu, dtype=float)
    truth_dist = ((comp_means[:, 0] - geom.TRUTH_MF) / 72.0) ** 2 + (comp_means[:, 1] - geom.TRUTH_CHIF) ** 2
    truth_component = int(np.argmin(truth_dist))
    return model, mu, sigma, truth_component


def _run_chain(
    posterior: strict.RingdownPosterior,
    *,
    move_name: str,
    run_seed: int,
    nwalkers: int,
    burnin: int,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(run_seed)
    p0 = posterior.sample_prior(rng, nwalkers)
    np.random.seed(run_seed)
    if move_name == "de":
        move = emcee.moves.DEMove()
    else:
        move = emcee.moves.StretchMove(a=2.0)
    sampler = emcee.EnsembleSampler(nwalkers, posterior.ndim, posterior.log_posterior, moves=move)
    state = sampler.run_mcmc(p0, burnin, progress=False)
    sampler.reset()
    sampler.run_mcmc(state, steps, progress=False)
    return (
        np.asarray(sampler.get_chain(), dtype=float),
        np.asarray(sampler.get_chain(flat=True), dtype=float),
        np.asarray(sampler.get_log_prob(flat=True), dtype=float),
        float(np.mean(sampler.acceptance_fraction)),
    )


def main() -> None:
    args = parse_args()
    moves = _parse_moves(args.moves)
    run_seeds = _parse_int_list(args.run_seeds)
    posterior = _build_posterior(args)
    model, mu, sigma, truth_component = _fit_reference_gmm(args)
    rng = np.random.default_rng(args.seed + 77)

    rows: list[dict[str, float | int | str]] = []
    for move_name in moves:
        for run_seed in run_seeds:
            chain, flat, logp, acceptance = _run_chain(
                posterior,
                move_name=move_name,
                run_seed=run_seed,
                nwalkers=args.nwalkers,
                burnin=args.burnin,
                steps=args.steps,
            )
            score_x = geom._random_subsample(flat, args.score_subsample_per_run, rng)
            comp = geom._predict_component(model, score_x[:, :2], mu, sigma)
            truth_frac = float(np.mean(comp == truth_component))
            qmf = np.quantile(flat[:, 0], [0.16, 0.5, 0.84])
            qchi = np.quantile(flat[:, 1], [0.16, 0.5, 0.84])
            rows.append(
                {
                    "move": move_name,
                    "run_seed": run_seed,
                    "truth_component_fraction": truth_frac,
                    "mf_q50": float(qmf[1]),
                    "chif_q50": float(qchi[1]),
                    "logpost_median": float(np.median(logp)),
                    "logpost_p95": float(np.quantile(logp, 0.95)),
                    "logpost_max": float(np.max(logp)),
                    "acceptance": acceptance,
                }
            )
            print(
                f"move={move_name} seed={run_seed} truth_frac={truth_frac:.4f} "
                f"mf_q50={qmf[1]:.3f} chif_q50={qchi[1]:.3f} logpost_max={np.max(logp):.3f}"
            )

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "move",
                "run_seed",
                "truth_component_fraction",
                "mf_q50",
                "chif_q50",
                "logpost_median",
                "logpost_p95",
                "logpost_max",
                "acceptance",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# N=3 Move Compare", "", f"Truth-adjacent component id: `{truth_component}`", ""]
    for move_name in moves:
        sub = [row for row in rows if row["move"] == move_name]
        mean_truth = float(np.mean([float(row["truth_component_fraction"]) for row in sub]))
        mean_mf = float(np.mean([float(row["mf_q50"]) for row in sub]))
        mean_chi = float(np.mean([float(row["chif_q50"]) for row in sub]))
        mean_max = float(np.mean([float(row["logpost_max"]) for row in sub]))
        lines.append(f"## {move_name}")
        lines.append("")
        lines.append(
            f"- mean truth-component fraction: `{mean_truth:.4f}`; "
            f"mean q50 `(Mf, chif)=({mean_mf:.3f}, {mean_chi:.3f})`; mean max logpost `{mean_max:.3f}`"
        )
        lines.append("")
        lines.append("| seed | truth frac | Mf q50 | chif q50 | max logpost |")
        lines.append("|---:|---:|---:|---:|---:|")
        for row in sub:
            lines.append(
                f"| {row['run_seed']} | {float(row['truth_component_fraction']):.4f} | "
                f"{float(row['mf_q50']):.3f} | {float(row['chif_q50']):.3f} | {float(row['logpost_max']):.3f} |"
            )
        lines.append("")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"csv_output={args.csv_output}")
    print(f"markdown_output={args.markdown_output}")
if __name__ == "__main__":
    main()
