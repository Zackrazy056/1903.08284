from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import emcee
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from config_io import load_yaml, project_root_from_config
from eval.phase_b_assess import AssessmentThresholds, assess_phase_b
from noise import build_psd_interpolator_from_asd_file
from qnm_kerr import KerrQNMInterpolator
from ringdown_eq1 import ringdown_plus_eq1


def _resolve_path(project_root: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase B MCMC smoke check with strict assessment and bias A/B test")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ringdown_fig10_snpe/configs/mcmc_smoke.yaml"),
        help="Path to mcmc smoke yaml",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Override sampler n_steps from config",
    )
    parser.add_argument(
        "--target-N",
        type=int,
        default=None,
        help="Override target overtone order N from config",
    )
    parser.add_argument(
        "--with-bias",
        action="store_true",
        help="Force run only bias model",
    )
    parser.add_argument(
        "--without-bias",
        action="store_true",
        help="Force run only baseline model",
    )
    parser.add_argument(
        "--ab-on-fail",
        dest="ab_on_fail",
        action="store_true",
        help="Enable A/B run when baseline fails strict checks",
    )
    parser.add_argument(
        "--no-ab-on-fail",
        dest="ab_on_fail",
        action="store_false",
        help="Disable A/B run when baseline fails strict checks",
    )
    parser.add_argument(
        "--assessment-only",
        action="store_true",
        help="Read existing chains and only compute strict assessment",
    )
    parser.set_defaults(ab_on_fail=None)
    return parser.parse_args()


def _append_suffix(path: Path, suffix: str | None) -> Path:
    if not suffix:
        return path
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _compare_models(models: list[dict[str, Any]], metric_order: list[str]) -> list[dict[str, Any]]:
    def sort_key(item: dict[str, Any]):
        m = item["assessment"]["metrics"]
        c = item["assessment"]["checks"]
        key = []
        for k in metric_order:
            if k == "strict_pass":
                key.append(0 if c.get("strict_pass", False) else 1)
            elif k == "truth_hpd_rank":
                key.append(float(m.get("truth_hpd_rank", 1.0)))
            elif k == "edge_occupancy":
                key.append(float(m.get("edge_occupancy", np.inf)))
            elif k == "acceptance_mean":
                key.append(-float(m.get("acceptance_mean", 0.0)))
            else:
                key.append(np.inf)
        return tuple(key)

    return sorted(models, key=sort_key)


def _credible_thresholds(hist2d: np.ndarray, probs: list[float]) -> list[float]:
    flat = np.asarray(hist2d, dtype=float).ravel()
    total = float(np.sum(flat))
    if total <= 0:
        return [0.0 for _ in probs]
    idx = np.argsort(flat)[::-1]
    sorted_vals = flat[idx]
    cdf = np.cumsum(sorted_vals) / total
    out = []
    for p in probs:
        j = int(np.searchsorted(cdf, p, side="left"))
        j = min(max(j, 0), len(sorted_vals) - 1)
        out.append(float(sorted_vals[j]))
    return out


def _make_model_paths(project_root: Path, outputs_cfg: dict[str, Any], model_tag: str, suffix: str | None) -> dict[str, Path]:
    if model_tag == "A_baseline":
        fig = _resolve_path(project_root, outputs_cfg["figure_baseline"])
        tr = _resolve_path(project_root, outputs_cfg["traceplots_baseline"])
    elif model_tag == "B_bias":
        fig = _resolve_path(project_root, outputs_cfg["figure_bias"])
        tr = _resolve_path(project_root, outputs_cfg["traceplots_bias"])
    else:
        raise ValueError(f"Unknown model tag: {model_tag}")

    fig = _append_suffix(fig, suffix)
    tr = _append_suffix(tr, suffix)
    diag = project_root / "outputs" / "diagnostics"
    chain = diag / f"mcmc_chain_smoke_{model_tag}{'_' + suffix if suffix else ''}.npz"
    accept = diag / f"mcmc_acceptance_{model_tag}{'_' + suffix if suffix else ''}.json"
    return {"figure": fig, "traceplots": tr, "chain": chain, "acceptance": accept}


def _summarize_posterior(samples: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples, ddof=1)),
        "p05": float(np.quantile(samples, 0.05)),
        "p50": float(np.quantile(samples, 0.50)),
        "p95": float(np.quantile(samples, 0.95)),
    }


def _run_single_model(
    *,
    project_root: Path,
    fig10_cfg: dict[str, Any],
    mcmc_cfg: dict[str, Any],
    N: int,
    use_bias: bool,
    n_steps_override: int | None,
    suffix: str | None,
) -> dict[str, Any]:
    sampler_cfg = mcmc_cfg["sampler"]
    outputs_cfg = mcmc_cfg["outputs"]
    model_tag = "B_bias" if use_bias else "A_baseline"
    out = _make_model_paths(project_root, outputs_cfg, model_tag, suffix)

    seed_base = int(fig10_cfg["reproducibility"]["random_seed_global"]) + 17 + 1000 * int(N) + (31 if use_bias else 0)
    rng = np.random.default_rng(seed_base)

    priors = fig10_cfg["priors"]
    mf_min = float(priors["Mf_msun"]["min"])
    mf_max = float(priors["Mf_msun"]["max"])
    chi_min = float(priors["chi_f"]["min"])
    chi_max = float(priors["chi_f"]["max"])
    phi_min = float(priors["phase_phi_n"]["min"])
    phi_max = float(priors["phase_phi_n"]["max"])
    h_peak = float(priors["amplitude_A_n"]["h_peak"])
    amp_min = float(priors["amplitude_A_n"]["min_in_h_peak_units"]) * h_peak
    amp_max = float(priors["amplitude_A_n"]["max_in_h_peak_units"]) * h_peak

    bias_cfg = mcmc_cfg.get("model_bias_test", {})
    bias_low_hp, bias_high_hp = bias_cfg.get("bias_prior_in_h_peak_units", [-0.2, 0.2])
    bias_min = float(bias_low_hp) * h_peak
    bias_max = float(bias_high_hp) * h_peak

    truth_mf = float(fig10_cfg["truth_parameters"]["Mf_msun"])
    truth_chi = float(fig10_cfg["truth_parameters"]["chi_f"])

    d_obs_path = project_root / "data" / "injection" / "d_obs.npz"
    obs = np.load(d_obs_path)
    t_sec = np.asarray(obs["t_sec"], dtype=float)
    d_obs = np.asarray(obs["d_obs"], dtype=float)
    dt = float(obs["dt"])

    psd_rel = fig10_cfg["injection"]["detector"]["noise"]["psd_file"]
    psd_path = _resolve_path(project_root, psd_rel)
    psd_fn = build_psd_interpolator_from_asd_file(psd_path)

    fmin_hz = float(fig10_cfg["injection"]["snr_definition"]["integration_fmin_hz"])
    freqs = np.fft.rfftfreq(len(d_obs), d=dt)
    df = float(freqs[1] - freqs[0])
    psd = psd_fn(freqs)
    mask = (freqs >= fmin_hz) & (freqs > 0.0) & np.isfinite(psd)
    if not np.any(mask):
        raise RuntimeError("No frequency bins satisfy f >= fmin and finite PSD")
    weight = np.zeros_like(freqs, dtype=float)
    weight[mask] = 4.0 * df / psd[mask]

    qnm_interp = KerrQNMInterpolator(n_max=N)
    n_modes = N + 1
    ndim = 2 + 2 * n_modes + (1 if use_bias else 0)
    n_walkers = max(int(sampler_cfg["n_walkers_min"]), 4 * ndim)
    n_steps = int(n_steps_override) if n_steps_override is not None else int(sampler_cfg["n_steps"])
    burn_in = int(sampler_cfg["burn_in"])
    thin = int(sampler_cfg["thin"])

    param_names = ["Mf_msun", "chi_f"] + [f"A_{n}" for n in range(n_modes)] + [f"phi_{n}" for n in range(n_modes)]
    if use_bias:
        param_names = param_names + ["bias"]

    def unpack(theta: np.ndarray):
        mf = float(theta[0])
        chi = float(theta[1])
        amps = np.asarray(theta[2 : 2 + n_modes], dtype=float)
        phis = np.asarray(theta[2 + n_modes : 2 + 2 * n_modes], dtype=float)
        bias = float(theta[-1]) if use_bias else 0.0
        return mf, chi, amps, phis, bias

    def log_prior(theta: np.ndarray) -> float:
        mf, chi, amps, phis, bias = unpack(theta)
        if not (mf_min <= mf <= mf_max):
            return -np.inf
        if not (chi_min <= chi <= chi_max):
            return -np.inf
        if np.any((amps < amp_min) | (amps > amp_max)):
            return -np.inf
        if np.any((phis < phi_min) | (phis > phi_max)):
            return -np.inf
        if use_bias and not (bias_min <= bias <= bias_max):
            return -np.inf
        return 0.0

    def log_likelihood(theta: np.ndarray) -> float:
        mf, chi, amps, phis, bias = unpack(theta)
        h_model = ringdown_plus_eq1(
            t_sec=t_sec,
            mf_msun=mf,
            chi_f=chi,
            amplitudes=amps,
            phases=phis,
            qnm_interp=qnm_interp,
            bias=bias,
        )
        resid = d_obs - h_model
        r_tilde = dt * np.fft.rfft(resid)
        quad = float(np.sum(weight * (np.real(r_tilde) ** 2 + np.imag(r_tilde) ** 2)))
        if not np.isfinite(quad):
            return -np.inf
        return -0.5 * quad

    def log_prob(theta: np.ndarray) -> float:
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    def sample_prior(n: int) -> np.ndarray:
        arr = np.empty((n, ndim), dtype=float)
        arr[:, 0] = rng.uniform(mf_min, mf_max, size=n)
        arr[:, 1] = rng.uniform(chi_min, chi_max, size=n)
        arr[:, 2 : 2 + n_modes] = rng.uniform(amp_min, amp_max, size=(n, n_modes))
        arr[:, 2 + n_modes : 2 + 2 * n_modes] = rng.uniform(phi_min, phi_max, size=(n, n_modes))
        if use_bias:
            arr[:, -1] = rng.uniform(bias_min, bias_max, size=n)
        return arr

    p0 = sample_prior(n_walkers)
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob)
    sampler.run_mcmc(p0, n_steps, progress=True)

    chain = sampler.get_chain()
    flat = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, thin=thin, flat=True)
    accept = np.asarray(sampler.acceptance_fraction, dtype=float)
    if len(flat) == 0:
        raise RuntimeError("No post-burnin samples. Adjust burn_in/thin/n_steps.")

    out["traceplots"].parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out["traceplots"]) as pdf:
        fig, axes = plt.subplots(ndim, 1, figsize=(11.0, max(8.0, 1.5 * ndim)), sharex=True)
        if ndim == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.plot(chain[:, :, i], color="black", alpha=0.08, lw=0.4)
            ax.axvline(burn_in, color="tab:red", ls="--", lw=1.0)
            ax.set_ylabel(param_names[i], fontsize=8)
            ax.grid(alpha=0.2)
        axes[-1].set_xlabel("Step")
        fig.suptitle(f"MCMC trace plots ({model_tag}, N={N})", fontsize=12)
        fig.tight_layout()
        pdf.savefig(fig, dpi=180)
        plt.close(fig)

    x = flat[:, 0]
    y = flat[:, 1]
    h2d, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=90,
        range=[[mf_min, mf_max], [chi_min, chi_max]],
    )
    h2d = h2d.T
    xcent = 0.5 * (xedges[:-1] + xedges[1:])
    ycent = 0.5 * (yedges[:-1] + yedges[1:])
    xx, yy = np.meshgrid(xcent, ycent)
    levels = _credible_thresholds(h2d, [0.5, 0.9])
    level_plot = sorted(set([lv for lv in levels if lv > 0]))

    out["figure"].parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.2, 8.2))
    gs = fig.add_gridspec(4, 4, hspace=0.06, wspace=0.06)
    ax_top = fig.add_subplot(gs[0, 0:3])
    ax_joint = fig.add_subplot(gs[1:4, 0:3], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_joint)

    take = min(6000, len(x))
    idx = rng.choice(len(x), size=take, replace=False)
    ax_joint.scatter(x[idx], y[idx], s=2.0, alpha=0.08, color="tab:blue")
    if level_plot:
        ax_joint.contour(xx, yy, h2d, levels=level_plot, colors=["tab:green", "tab:orange"], linewidths=1.6)

    ax_top.hist(x, bins=80, color="tab:blue", alpha=0.8)
    ax_right.hist(y, bins=80, orientation="horizontal", color="tab:blue", alpha=0.8)
    ax_joint.axvline(truth_mf, color="black", ls="--", lw=1.2)
    ax_joint.axhline(truth_chi, color="black", ls="--", lw=1.2)
    ax_joint.plot([truth_mf], [truth_chi], marker="x", color="black", ms=7)
    ax_top.axvline(truth_mf, color="black", ls="--", lw=1.1)
    ax_right.axhline(truth_chi, color="black", ls="--", lw=1.1)

    ax_joint.set_xlabel(r"$M_f\ [M_\odot]$")
    ax_joint.set_ylabel(r"$\chi_f$")
    ax_top.set_ylabel("count")
    ax_right.set_xlabel("count")
    ax_joint.grid(alpha=0.2)
    ax_top.grid(alpha=0.2)
    ax_right.grid(alpha=0.2)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    title = f"MCMC smoke posterior (N={N}, {'+bias' if use_bias else 'baseline'}, delta_t0=0)"
    ax_joint.set_title(title)
    fig.tight_layout()
    fig.savefig(out["figure"], dpi=180)
    plt.close(fig)

    out["chain"].parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out["chain"],
        chain=chain,
        flat_samples=flat,
        param_names=np.asarray(param_names, dtype=object),
        burn_in=burn_in,
        thin=thin,
        log_prob_samples=log_probs,
        acceptance_fraction=accept,
        use_bias=np.asarray(use_bias),
        target_N=np.asarray(N),
    )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_tag": model_tag,
        "use_bias": use_bias,
        "target_N": int(N),
        "n_modes": int(n_modes),
        "ndim": int(ndim),
        "n_walkers": int(n_walkers),
        "n_steps": int(n_steps),
        "burn_in": int(burn_in),
        "thin": int(thin),
        "n_posterior_samples": int(len(flat)),
        "delta_t0_ms": float(fig10_cfg["inference_model"]["start_time"]["delta_t0_ms"]),
        "fmin_hz": float(fmin_hz),
        "truth": {"Mf_msun": truth_mf, "chi_f": truth_chi},
        "acceptance_fraction": {
            "mean": float(np.mean(accept)),
            "min": float(np.min(accept)),
            "max": float(np.max(accept)),
        },
        "posterior_Mf": _summarize_posterior(x),
        "posterior_chi_f": _summarize_posterior(y),
        "files": {
            "figure": str(out["figure"].relative_to(project_root).as_posix()),
            "traceplots": str(out["traceplots"].relative_to(project_root).as_posix()),
            "chain_npz": str(out["chain"].relative_to(project_root).as_posix()),
        },
    }
    if use_bias:
        summary["posterior_bias"] = _summarize_posterior(flat[:, -1])
    _save_json(out["acceptance"], summary)

    return {
        "model_tag": model_tag,
        "use_bias": use_bias,
        "N": int(N),
        "flat_samples": flat,
        "acceptance_fraction": accept,
        "summary": summary,
        "paths": out,
    }


def _assessment_from_chain(
    *,
    chain_npz: Path,
    acceptance_json: Path,
    fig10_cfg: dict[str, Any],
    thresholds: AssessmentThresholds,
) -> dict[str, Any]:
    if not chain_npz.exists() or not acceptance_json.exists():
        raise FileNotFoundError(f"Missing files for assessment-only: {chain_npz} / {acceptance_json}")
    d = np.load(chain_npz, allow_pickle=True)
    samples = np.asarray(d["flat_samples"], dtype=float)
    if "acceptance_fraction" in d:
        accept = np.asarray(d["acceptance_fraction"], dtype=float)
    else:
        s = _load_json(acceptance_json)
        acc = s.get("acceptance_fraction", {})
        mn = float(acc.get("mean", 0.0))
        mi = float(acc.get("min", mn))
        mx = float(acc.get("max", mn))
        accept = np.asarray([mn, mi, mx], dtype=float)

    priors = fig10_cfg["priors"]
    mf_range = (float(priors["Mf_msun"]["min"]), float(priors["Mf_msun"]["max"]))
    chi_range = (float(priors["chi_f"]["min"]), float(priors["chi_f"]["max"]))
    truth_mf = float(fig10_cfg["truth_parameters"]["Mf_msun"])
    truth_chi = float(fig10_cfg["truth_parameters"]["chi_f"])
    assess = assess_phase_b(
        flat_samples=samples,
        acceptance_fraction=accept,
        truth_mf=truth_mf,
        truth_chi=truth_chi,
        mf_range=mf_range,
        chi_range=chi_range,
        thresholds=thresholds,
    )
    assess["checks"]["strict_pass"] = assess["pass"]
    return assess


def _assess_run(
    run: dict[str, Any],
    fig10_cfg: dict[str, Any],
    thresholds: AssessmentThresholds,
) -> dict[str, Any]:
    priors = fig10_cfg["priors"]
    mf_range = (float(priors["Mf_msun"]["min"]), float(priors["Mf_msun"]["max"]))
    chi_range = (float(priors["chi_f"]["min"]), float(priors["chi_f"]["max"]))
    truth_mf = float(fig10_cfg["truth_parameters"]["Mf_msun"])
    truth_chi = float(fig10_cfg["truth_parameters"]["chi_f"])
    assess = assess_phase_b(
        flat_samples=run["flat_samples"],
        acceptance_fraction=run["acceptance_fraction"],
        truth_mf=truth_mf,
        truth_chi=truth_chi,
        mf_range=mf_range,
        chi_range=chi_range,
        thresholds=thresholds,
    )
    assess["checks"]["strict_pass"] = assess["pass"]
    return assess


def _run_for_single_N(
    *,
    project_root: Path,
    fig10_cfg: dict[str, Any],
    mcmc_cfg: dict[str, Any],
    N: int,
    n_steps_override: int | None,
    run_baseline: bool,
    run_bias: bool,
    ab_on_fail: bool,
    suffix: str | None,
    assessment_only: bool,
    thresholds: AssessmentThresholds,
    metric_order: list[str],
) -> dict[str, Any]:
    model_runs: list[dict[str, Any]] = []

    outputs_cfg = mcmc_cfg["outputs"]
    for use_bias in [False, True]:
        if use_bias and not run_bias:
            continue
        if (not use_bias) and not run_baseline:
            continue

        tag = "B_bias" if use_bias else "A_baseline"
        paths = _make_model_paths(project_root, outputs_cfg, tag, suffix)

        if assessment_only:
            accept_path = paths["acceptance"]
            assess = _assessment_from_chain(
                chain_npz=paths["chain"],
                acceptance_json=accept_path,
                fig10_cfg=fig10_cfg,
                thresholds=thresholds,
            )
            summary = _load_json(accept_path)
            model_runs.append(
                {
                    "model_tag": tag,
                    "use_bias": use_bias,
                    "N": int(N),
                    "summary": summary,
                    "assessment": assess,
                    "paths": paths,
                }
            )
        else:
            run = _run_single_model(
                project_root=project_root,
                fig10_cfg=fig10_cfg,
                mcmc_cfg=mcmc_cfg,
                N=N,
                use_bias=use_bias,
                n_steps_override=n_steps_override,
                suffix=suffix,
            )
            run["assessment"] = _assess_run(run, fig10_cfg=fig10_cfg, thresholds=thresholds)
            model_runs.append(run)

    # Auto AB on baseline fail.
    if (not assessment_only) and run_baseline and (not run_bias) and ab_on_fail:
        baseline = model_runs[0] if model_runs else None
        if baseline is not None and not baseline["assessment"]["pass"]:
            bias_run = _run_single_model(
                project_root=project_root,
                fig10_cfg=fig10_cfg,
                mcmc_cfg=mcmc_cfg,
                N=N,
                use_bias=True,
                n_steps_override=n_steps_override,
                suffix=suffix,
            )
            bias_run["assessment"] = _assess_run(bias_run, fig10_cfg=fig10_cfg, thresholds=thresholds)
            model_runs.append(bias_run)

    if not model_runs:
        raise RuntimeError("No models selected to run/assess.")

    ranked = _compare_models(model_runs, metric_order=metric_order)
    selected = ranked[0]
    pass_any = any(m["assessment"]["pass"] for m in model_runs)

    run_payload = {
        "target_N": int(N),
        "model_runs": [
            {
                "model_tag": m["model_tag"],
                "use_bias": bool(m["use_bias"]),
                "N": int(m["N"]),
                "assessment": m["assessment"],
                "summary_file": str(m["paths"]["acceptance"].relative_to(project_root).as_posix()),
                "chain_file": str(m["paths"]["chain"].relative_to(project_root).as_posix()),
                "figure_file": str(m["paths"]["figure"].relative_to(project_root).as_posix()),
            }
            for m in model_runs
        ],
        "ranked_model_order": [m["model_tag"] for m in ranked],
        "selected_model": ranked[0]["model_tag"],
        "strict_pass_any": bool(pass_any),
        "ab_performed": len(model_runs) > 1,
    }

    # A/B compare artifact only when both baseline and bias are available.
    tags = {m["model_tag"] for m in model_runs}
    if {"A_baseline", "B_bias"}.issubset(tags):
        ab_path = project_root / "outputs" / "diagnostics" / f"mcmc_ab_compare{'_' + suffix if suffix else ''}.json"
        ab_payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "target_N": int(N),
            "metric_order": metric_order,
            "models": run_payload["model_runs"],
            "recommended_model": run_payload["selected_model"],
        }
        _save_json(ab_path, ab_payload)
        run_payload["ab_compare_file"] = str(ab_path.relative_to(project_root).as_posix())

    return run_payload


def main() -> None:
    args = _parse_args()
    if args.with_bias and args.without_bias:
        raise ValueError("Cannot set both --with-bias and --without-bias")

    cfg_path = args.config.resolve()
    project_root = project_root_from_config(cfg_path)
    cfg = load_yaml(cfg_path)
    mcmc_cfg = cfg["mcmc"]
    fig10_cfg = load_yaml(_resolve_path(project_root, cfg["use_fig10_spec_from"]))

    eval_cfg = mcmc_cfg.get("evaluation", {})
    thresholds_cfg = eval_cfg.get("thresholds", {})
    thresholds = AssessmentThresholds(
        truth_in_90_hpd_required=bool(thresholds_cfg.get("truth_in_90_hpd_required", True)),
        acceptance_mean_min=float(thresholds_cfg.get("acceptance_mean_min", 0.12)),
        acceptance_min_per_walker_min=float(thresholds_cfg.get("acceptance_min_per_walker_min", 0.02)),
        mf_edge_1pct_max=float(thresholds_cfg.get("mf_edge_1pct_max", 0.10)),
        chi_edge_1pct_max=float(thresholds_cfg.get("chi_edge_1pct_max", 0.05)),
        mf_iqr_fraction_of_prior_max=float(thresholds_cfg.get("mf_iqr_fraction_of_prior_max", 0.70)),
        chi_iqr_fraction_of_prior_max=float(thresholds_cfg.get("chi_iqr_fraction_of_prior_max", 0.70)),
    )

    bias_cfg = mcmc_cfg.get("model_bias_test", {})
    metric_order = list(bias_cfg.get("ab_compare_metric_order", ["strict_pass", "truth_hpd_rank", "edge_occupancy", "acceptance_mean"]))
    ab_on_fail_default = bool(bias_cfg.get("enable_ab_on_fail", True))
    ab_on_fail = ab_on_fail_default if args.ab_on_fail is None else bool(args.ab_on_fail)

    target_N = int(args.target_N) if args.target_N is not None else int(mcmc_cfg["target_overtone_order_N"])
    run_baseline = not args.with_bias
    run_bias = args.with_bias
    if args.without_bias:
        run_baseline = True
        run_bias = False

    primary = _run_for_single_N(
        project_root=project_root,
        fig10_cfg=fig10_cfg,
        mcmc_cfg={"sampler": mcmc_cfg["sampler"], "outputs": cfg["outputs"], "model_bias_test": bias_cfg},
        N=target_N,
        n_steps_override=args.n_steps,
        run_baseline=run_baseline,
        run_bias=run_bias,
        ab_on_fail=ab_on_fail,
        suffix=None,
        assessment_only=args.assessment_only,
        thresholds=thresholds,
        metric_order=metric_order,
    )

    fallback_payload = None
    final_pass = bool(primary["strict_pass_any"])
    selected_N = int(primary["target_N"])
    selected_model = str(primary["selected_model"])

    # Secondary fallback: if strict still fails on N=1, repeat same flow with N=2.
    should_fallback = (
        (not args.assessment_only)
        and (not args.with_bias)
        and (not args.without_bias)
        and (target_N == 1)
        and (not final_pass)
    )
    if should_fallback:
        fallback_payload = _run_for_single_N(
            project_root=project_root,
            fig10_cfg=fig10_cfg,
            mcmc_cfg={"sampler": mcmc_cfg["sampler"], "outputs": cfg["outputs"], "model_bias_test": bias_cfg},
            N=2,
            n_steps_override=args.n_steps,
            run_baseline=True,
            run_bias=False,
            ab_on_fail=ab_on_fail,
            suffix="N2",
            assessment_only=False,
            thresholds=thresholds,
            metric_order=metric_order,
        )
        if fallback_payload["strict_pass_any"]:
            final_pass = True
            selected_N = int(fallback_payload["target_N"])
            selected_model = str(fallback_payload["selected_model"])

    assessment_report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_mode": str(eval_cfg.get("mode", "strict_near_paper")),
        "thresholds": {
            "truth_in_90_hpd_required": thresholds.truth_in_90_hpd_required,
            "acceptance_mean_min": thresholds.acceptance_mean_min,
            "acceptance_min_per_walker_min": thresholds.acceptance_min_per_walker_min,
            "mf_edge_1pct_max": thresholds.mf_edge_1pct_max,
            "chi_edge_1pct_max": thresholds.chi_edge_1pct_max,
            "mf_iqr_fraction_of_prior_max": thresholds.mf_iqr_fraction_of_prior_max,
            "chi_iqr_fraction_of_prior_max": thresholds.chi_iqr_fraction_of_prior_max,
        },
        "target_N": int(target_N),
        "ab_on_fail": bool(ab_on_fail),
        "primary_run": primary,
        "fallback_run": fallback_payload,
        "final_decision": {
            "phase_b_pass": bool(final_pass),
            "selected_N": int(selected_N),
            "selected_model": selected_model,
        },
    }

    assessment_path = project_root / "outputs" / "diagnostics" / "mcmc_phaseb_assessment.json"
    _save_json(assessment_path, assessment_report)

    print("Phase B strict assessment complete")
    print(f"Primary N={primary['target_N']} strict_pass_any={primary['strict_pass_any']}")
    if fallback_payload is not None:
        print(f"Fallback N={fallback_payload['target_N']} strict_pass_any={fallback_payload['strict_pass_any']}")
    print(f"Final pass={final_pass}, selected N={selected_N}, model={selected_model}")
    print(f"Wrote: {assessment_path}")


if __name__ == "__main__":
    main()

