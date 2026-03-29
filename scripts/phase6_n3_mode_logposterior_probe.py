"""Legacy N=3 posterior-probe diagnostic.

This script predates the shared paper-faithful forward model and should not be
mixed with the current Fig.10 baseline without migration.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import phase6_fig10_emcee_full_strict as strict  # noqa: E402


TRUTH_MF = 68.546372
TRUTH_CHIF = 0.692085


@dataclass(frozen=True)
class ComponentProbe:
    component: int
    is_truth_adjacent: bool
    fraction: float
    sample_count: int
    mf_q50: float
    chif_q50: float
    logpost_mean: float
    logpost_median: float
    logpost_p95: float
    logpost_max: float
    best_mf: float
    best_chif: float
    best_logpost: float


@dataclass(frozen=True)
class ChainProbe:
    label: str
    overall_logpost_mean: float
    overall_logpost_median: float
    overall_logpost_p95: float
    overall_logpost_max: float
    best_mf: float
    best_chif: float
    components: list[ComponentProbe]


@dataclass(frozen=True)
class ProbeSummary:
    n_components: int
    component_means_mf_chif: list[list[float]]
    truth_component: int
    primary: ChainProbe
    alt: ChainProbe
    dominant_component_by_max_logpost: int
    truth_component_logpost_gap_to_best: float
    diagnosis: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe mode competitiveness by recomputing N=3 log posterior values.")
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
    p.add_argument("--burn-fraction", type=float, default=0.5)
    p.add_argument("--fit-subsample-per-chain", type=int, default=6000)
    p.add_argument("--score-subsample-per-chain", type=int, default=6000)
    p.add_argument("--max-components", type=int, default=4)
    p.add_argument("--qnm-chi-grid-size", type=int, default=160)
    p.add_argument("--seed", type=int, default=12345)
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
    p.add_argument("--json-output", type=Path, default=Path("results/fig10_n3_mode_logpost_fix1.json"))
    p.add_argument("--markdown-output", type=Path, default=Path("results/fig10_n3_mode_logpost_fix1.md"))
    return p.parse_args()


def _load_flat(path: Path, burn_fraction: float) -> np.ndarray:
    z = np.load(path, allow_pickle=False)
    chain = np.asarray(z["chain"], dtype=float)
    start = int(np.floor(chain.shape[0] * burn_fraction))
    start = min(max(start, 0), chain.shape[0] - 1)
    return chain[start:, :, :].reshape(-1, chain.shape[-1])


def _random_subsample(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if x.shape[0] <= n:
        return np.asarray(x, dtype=float)
    idx = rng.choice(x.shape[0], size=n, replace=False)
    return np.asarray(x[idx], dtype=float)


def _fit_gmm(x_fit_2d: np.ndarray, max_components: int, seed: int) -> tuple[GaussianMixture, np.ndarray, np.ndarray]:
    mu = np.mean(x_fit_2d, axis=0)
    sigma = np.std(x_fit_2d, axis=0)
    sigma = np.where(sigma > 0.0, sigma, 1.0)
    z = (x_fit_2d - mu) / sigma
    best_model: GaussianMixture | None = None
    best_bic = float("inf")
    for k in range(1, max_components + 1):
        model = GaussianMixture(n_components=k, covariance_type="full", random_state=seed, n_init=10)
        model.fit(z)
        bic = float(model.bic(z))
        if bic < best_bic:
            best_bic = bic
            best_model = model
    if best_model is None:
        raise RuntimeError("failed to fit GaussianMixture")
    return best_model, mu, sigma


def _predict(model: GaussianMixture, x2d: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict((x2d - mu) / sigma), dtype=int)


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


def _score_samples(posterior: strict.RingdownPosterior, x: np.ndarray) -> np.ndarray:
    vals = np.empty(x.shape[0], dtype=float)
    for i, theta in enumerate(x):
        vals[i] = float(posterior.log_posterior(theta))
    return vals


def _component_probe(x: np.ndarray, logp: np.ndarray, component: int, truth_component: int) -> ComponentProbe:
    mf_q50 = float(np.quantile(x[:, 0], 0.5))
    chif_q50 = float(np.quantile(x[:, 1], 0.5))
    i_best = int(np.argmax(logp))
    return ComponentProbe(
        component=component,
        is_truth_adjacent=(component == truth_component),
        fraction=float(x.shape[0]),
        sample_count=int(x.shape[0]),
        mf_q50=mf_q50,
        chif_q50=chif_q50,
        logpost_mean=float(np.mean(logp)),
        logpost_median=float(np.median(logp)),
        logpost_p95=float(np.quantile(logp, 0.95)),
        logpost_max=float(np.max(logp)),
        best_mf=float(x[i_best, 0]),
        best_chif=float(x[i_best, 1]),
        best_logpost=float(logp[i_best]),
    )


def _build_chain_probe(
    *,
    label: str,
    x: np.ndarray,
    comp: np.ndarray,
    logp: np.ndarray,
    truth_component: int,
    n_components: int,
) -> ChainProbe:
    i_best = int(np.argmax(logp))
    components: list[ComponentProbe] = []
    for c in range(n_components):
        mask = comp == c
        if not np.any(mask):
            continue
        probe = _component_probe(x[mask], logp[mask], c, truth_component)
        probe = ComponentProbe(**{**asdict(probe), "fraction": float(np.mean(mask))})
        components.append(probe)
    return ChainProbe(
        label=label,
        overall_logpost_mean=float(np.mean(logp)),
        overall_logpost_median=float(np.median(logp)),
        overall_logpost_p95=float(np.quantile(logp, 0.95)),
        overall_logpost_max=float(np.max(logp)),
        best_mf=float(x[i_best, 0]),
        best_chif=float(x[i_best, 1]),
        components=components,
    )


def _build_markdown(summary: ProbeSummary) -> str:
    lines: list[str] = []
    lines.append("# N=3 Mode Log-Posterior Probe")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(summary.diagnosis)
    lines.append("")
    lines.append("## Global Metrics")
    lines.append("")
    lines.append(f"- GMM components: `{summary.n_components}`")
    lines.append(f"- Truth-adjacent component: `{summary.truth_component}`")
    lines.append(f"- Component means `(Mf, chif)`: `{summary.component_means_mf_chif}`")
    lines.append(f"- Truth-component max-logpost gap to best component: `{summary.truth_component_logpost_gap_to_best:.4f}`")
    lines.append("")
    for chain in [summary.primary, summary.alt]:
        lines.append(f"## {chain.label.capitalize()} Chain")
        lines.append("")
        lines.append(
            f"- Overall logpost: mean `{chain.overall_logpost_mean:.3f}`, median `{chain.overall_logpost_median:.3f}`, "
            f"p95 `{chain.overall_logpost_p95:.3f}`, max `{chain.overall_logpost_max:.3f}`"
        )
        lines.append(f"- Best sample `(Mf, chif)`: `({chain.best_mf:.3f}, {chain.best_chif:.3f})`")
        lines.append("")
        lines.append("| Component | Truth-adjacent | Weight | Mf q50 | chif q50 | logpost mean | logpost p95 | logpost max |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for comp in chain.components:
            lines.append(
                f"| {comp.component} | {comp.is_truth_adjacent} | {comp.fraction:.4f} | "
                f"{comp.mf_q50:.3f} | {comp.chif_q50:.3f} | {comp.logpost_mean:.3f} | "
                f"{comp.logpost_p95:.3f} | {comp.logpost_max:.3f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    primary_flat = _load_flat(args.primary_samples, args.burn_fraction)
    alt_flat = _load_flat(args.alt_samples, args.burn_fraction)

    fit_x = np.vstack(
        [
            _random_subsample(primary_flat[:, :2], args.fit_subsample_per_chain, rng),
            _random_subsample(alt_flat[:, :2], args.fit_subsample_per_chain, rng),
        ]
    )
    model, mu, sigma = _fit_gmm(fit_x, args.max_components, args.seed)
    comp_means = np.asarray(model.means_ * sigma + mu, dtype=float)
    truth_dist = ((comp_means[:, 0] - TRUTH_MF) / 72.0) ** 2 + (comp_means[:, 1] - TRUTH_CHIF) ** 2
    truth_component = int(np.argmin(truth_dist))

    primary_score = _random_subsample(primary_flat, args.score_subsample_per_chain, rng)
    alt_score = _random_subsample(alt_flat, args.score_subsample_per_chain, rng)
    primary_comp = _predict(model, primary_score[:, :2], mu, sigma)
    alt_comp = _predict(model, alt_score[:, :2], mu, sigma)

    posterior = _build_posterior(args)
    primary_logp = _score_samples(posterior, primary_score)
    alt_logp = _score_samples(posterior, alt_score)

    primary_probe = _build_chain_probe(
        label="primary",
        x=primary_score,
        comp=primary_comp,
        logp=primary_logp,
        truth_component=truth_component,
        n_components=model.n_components,
    )
    alt_probe = _build_chain_probe(
        label="alt",
        x=alt_score,
        comp=alt_comp,
        logp=alt_logp,
        truth_component=truth_component,
        n_components=model.n_components,
    )

    comp_best: dict[int, float] = {}
    for probe in primary_probe.components + alt_probe.components:
        comp_best[probe.component] = max(comp_best.get(probe.component, -np.inf), probe.logpost_max)
    dominant_component_by_max_logpost = int(max(comp_best.items(), key=lambda kv: kv[1])[0])
    truth_component_logpost_gap_to_best = float(comp_best[dominant_component_by_max_logpost] - comp_best[truth_component])

    if truth_component_logpost_gap_to_best < 5.0:
        diagnosis = (
            "The truth-adjacent mode is log-posterior competitive with the best false mode. "
            "This points more strongly to posterior-volume allocation and inter-mode mixing than to a clearly dominant wrong model."
        )
    else:
        diagnosis = (
            "The best false mode has a materially higher sampled log posterior than the truth-adjacent mode. "
            "That suggests the current likelihood / data product still favors the wrong solution, not just poor mixing."
        )

    summary = ProbeSummary(
        n_components=int(model.n_components),
        component_means_mf_chif=[[float(v) for v in row] for row in comp_means],
        truth_component=truth_component,
        primary=primary_probe,
        alt=alt_probe,
        dominant_component_by_max_logpost=dominant_component_by_max_logpost,
        truth_component_logpost_gap_to_best=truth_component_logpost_gap_to_best,
        diagnosis=diagnosis,
    )

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(_build_markdown(summary), encoding="utf-8")

    print(f"n_components={summary.n_components}")
    print(f"truth_component={summary.truth_component}")
    print(f"dominant_component_by_max_logpost={summary.dominant_component_by_max_logpost}")
    print(f"truth_component_logpost_gap_to_best={summary.truth_component_logpost_gap_to_best:.6f}")
    print(f"diagnosis={summary.diagnosis}")
    print(f"json_output={args.json_output}")
    print(f"markdown_output={args.markdown_output}")


if __name__ == "__main__":
    main()
