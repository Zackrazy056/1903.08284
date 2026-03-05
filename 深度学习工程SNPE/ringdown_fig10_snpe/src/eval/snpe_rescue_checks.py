from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform
import torch

# Allow running this file directly from src/eval/.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config_io import load_yaml, project_root_from_config
from noise import build_psd_interpolator_from_asd_file, generate_colored_gaussian_noise, optimal_snr
from qnm_kerr import KerrQNMInterpolator
from ringdown_eq1 import ringdown_plus_eq1
from snpe_embedding import build_embedding_net
from summarize import build_fixed_fft_feature_extractor


def _resolve_path(project_root: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SNPE rescue checks: tensor-scale, label-scale, SNR, and micro-overfit")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ringdown_fig10_snpe/configs/snpe.yaml"),
        help="Path to snpe.yaml",
    )
    parser.add_argument("--N", type=int, default=3, help="Overtone order for diagnostics")
    parser.add_argument("--batch-size", type=int, default=512, help="Synthetic batch size for tensor/label sanity checks")
    parser.add_argument("--snr-samples", type=int, default=64, help="Number of simulated clean signals for SNR distribution check")
    parser.add_argument("--overfit-n", type=int, default=10, help="Number of repeated samples for micro-overfit test")
    parser.add_argument("--overfit-epochs", type=int, default=300, help="Epochs for micro-overfit")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Optional output prefix under outputs/diagnostics. Default: snpe_rescue_N{N}",
    )
    return parser.parse_args()


def _stats_1d(x: np.ndarray) -> dict[str, float]:
    arr = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q05": float(np.quantile(arr, 0.05)),
        "q50": float(np.quantile(arr, 0.50)),
        "q95": float(np.quantile(arr, 0.95)),
    }


def _draw_target_snr(rng: np.random.Generator, snr_cfg: dict, default_target: float) -> float:
    mode = str(snr_cfg.get("mode", "fixed")).lower()
    if mode == "fixed":
        return float(snr_cfg.get("target", default_target))
    if mode == "uniform":
        lo = float(snr_cfg.get("low", 0.9 * default_target))
        hi = float(snr_cfg.get("high", 1.1 * default_target))
        if hi <= lo:
            hi = lo + 1e-6
        return float(rng.uniform(lo, hi))
    if mode == "normal":
        mu = float(snr_cfg.get("target", default_target))
        sigma = float(snr_cfg.get("normal_sigma", 0.05 * default_target))
        return float(max(1e-9, rng.normal(mu, sigma)))
    raise ValueError(f"Unsupported simulator.snr_conditioning.mode={mode!r}")


def main() -> None:
    args = _parse_args()
    snpe_cfg_path = args.config.resolve()
    project_root = project_root_from_config(snpe_cfg_path)
    snpe_cfg = load_yaml(snpe_cfg_path)
    fig10_cfg = load_yaml(_resolve_path(project_root, snpe_cfg["use_fig10_spec_from"]))

    seed = int(fig10_cfg["reproducibility"]["random_seed_global"]) + 909 + int(args.N)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    obs_npz = project_root / "data" / "injection" / "d_obs.npz"
    if not obs_npz.exists():
        raise FileNotFoundError(f"{obs_npz} not found. Run Phase A first.")
    obs = np.load(obs_npz)
    t_sec = np.asarray(obs["t_sec"], dtype=float)
    d_obs = np.asarray(obs["d_obs"], dtype=float)
    d_signal = np.asarray(obs["signal"], dtype=float) if "signal" in obs.files else None
    dt = float(obs["dt"])

    psd_path = _resolve_path(project_root, fig10_cfg["injection"]["detector"]["noise"]["psd_file"])
    psd_fn = build_psd_interpolator_from_asd_file(psd_path)
    snr_fmin_hz = float(fig10_cfg["injection"]["snr_definition"]["integration_fmin_hz"])
    snr_target = float(fig10_cfg["injection"]["snr_definition"]["target_value"])

    snpe = snpe_cfg["snpe"]
    feat_cfg = snpe.get("features", {})
    feat_fmin_hz = float(feat_cfg.get("fmin_hz", snr_fmin_hz))
    n_bins_cfg = feat_cfg.get("n_bins", None)
    n_freq_points_cfg = feat_cfg.get("n_freq_points", None)
    n_fft_cfg = feat_cfg.get("n_fft", None)
    feat_extractor = build_fixed_fft_feature_extractor(
        n_time=len(d_obs),
        dt=dt,
        psd_fn=psd_fn,
        fmin_hz=feat_fmin_hz,
        n_bins=int(n_bins_cfg) if n_bins_cfg is not None else None,
        n_freq_points=int(n_freq_points_cfg) if n_freq_points_cfg is not None else None,
        fmax_hz=float(feat_cfg.get("fmax_hz", 1024.0)),
        n_fft=int(n_fft_cfg) if n_fft_cfg is not None else None,
    )

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
    sim_cfg = snpe.get("simulator", {})
    amp_sampling = str(sim_cfg.get("amplitude_sampling", "uniform")).lower()
    if amp_sampling not in {"uniform", "log_uniform"}:
        raise ValueError(f"Unsupported snpe.simulator.amplitude_sampling={amp_sampling!r}")
    snr_cond_cfg = dict(sim_cfg.get("snr_conditioning", {}))
    snr_cond_enabled = bool(snr_cond_cfg.get("enabled", False))
    snr_min = float(snr_cond_cfg.get("min_signal_snr", 1e-9))
    scale_min = float(snr_cond_cfg.get("scale_min", 1e-3))
    scale_max = float(snr_cond_cfg.get("scale_max", 1e3))

    n_modes = int(args.N) + 1
    qnm_interp = KerrQNMInterpolator(n_max=int(args.N))

    n_batch = int(args.batch_size)
    theta = np.empty((n_batch, 2), dtype=np.float32)
    amps_all = np.empty((n_batch, n_modes), dtype=float)
    x_batch = np.empty((n_batch, feat_extractor.feature_dim), dtype=np.float32)
    n_snr = int(max(1, min(int(args.snr_samples), n_batch)))
    snr_batch_before = np.empty(n_snr, dtype=float)
    snr_batch_after = np.empty(n_snr, dtype=float)
    snr_targets = np.empty(n_snr, dtype=float)
    snr_scales = np.empty(n_snr, dtype=float)

    for i in range(n_batch):
        mf = rng.uniform(mf_min, mf_max)
        chi = rng.uniform(chi_min, chi_max)
        if amp_sampling == "log_uniform":
            amps = np.power(10.0, rng.uniform(np.log10(max(amp_min, 1e-300)), np.log10(max(amp_max, 1e-300)), size=n_modes))
        else:
            amps = rng.uniform(amp_min, amp_max, size=n_modes)
        phis = rng.uniform(phi_min, phi_max, size=n_modes)
        signal = ringdown_plus_eq1(
            t_sec=t_sec,
            mf_msun=float(mf),
            chi_f=float(chi),
            amplitudes=amps,
            phases=phis,
            qnm_interp=qnm_interp,
        )
        rho0 = float(optimal_snr(signal, dt=dt, fmin_hz=snr_fmin_hz, psd_fn=psd_fn))
        target_snr = _draw_target_snr(rng=rng, snr_cfg=snr_cond_cfg, default_target=snr_target)
        scale = 1.0
        rho1 = rho0
        if snr_cond_enabled and np.isfinite(rho0) and rho0 > snr_min and np.isfinite(target_snr) and target_snr > 0.0:
            scale = float(np.clip(target_snr / rho0, scale_min, scale_max))
            signal = signal * scale
            rho1 = rho0 * scale
        noise = generate_colored_gaussian_noise(len(t_sec), dt=dt, rng=rng, psd_fn=psd_fn)
        theta[i, 0] = mf
        theta[i, 1] = chi
        amps_all[i, :] = amps
        x_batch[i, :] = feat_extractor.transform(signal + noise)
        if i < n_snr:
            snr_batch_before[i] = rho0
            snr_batch_after[i] = rho1
            snr_targets[i] = target_snr
            snr_scales[i] = scale

    x_mu = np.mean(x_batch, axis=0, keepdims=True)
    x_sigma = np.std(x_batch, axis=0, keepdims=True)
    x_sigma_safe = x_sigma.copy()
    x_sigma_safe[x_sigma_safe < 1e-8] = 1.0
    x_std = (x_batch - x_mu) / x_sigma_safe

    overfit_report: dict[str, float | bool | str | None] = {}
    loss_curve: list[float] = []
    try:
        truth_mf = float(fig10_cfg["truth_parameters"]["Mf_msun"])
        truth_chi = float(fig10_cfg["truth_parameters"]["chi_f"])
        fixed_amps = rng.uniform(amp_min, amp_max, size=n_modes)
        fixed_phis = rng.uniform(phi_min, phi_max, size=n_modes)
        fixed_signal = ringdown_plus_eq1(
            t_sec=t_sec,
            mf_msun=truth_mf,
            chi_f=truth_chi,
            amplitudes=fixed_amps,
            phases=fixed_phis,
            qnm_interp=qnm_interp,
        )
        rho0 = float(optimal_snr(fixed_signal, dt=dt, fmin_hz=snr_fmin_hz, psd_fn=psd_fn))
        target_snr = _draw_target_snr(rng=rng, snr_cfg=snr_cond_cfg, default_target=snr_target)
        if snr_cond_enabled and np.isfinite(rho0) and rho0 > snr_min and np.isfinite(target_snr) and target_snr > 0.0:
            scale = float(np.clip(target_snr / rho0, scale_min, scale_max))
            fixed_signal = fixed_signal * scale
        fixed_noise = generate_colored_gaussian_noise(len(t_sec), dt=dt, rng=rng, psd_fn=psd_fn)
        fixed_feature = feat_extractor.transform(fixed_signal + fixed_noise)

        overfit_n = int(args.overfit_n)
        theta_overfit = np.repeat(np.array([[truth_mf, truth_chi]], dtype=np.float32), repeats=overfit_n, axis=0)
        x_overfit = np.repeat(fixed_feature[None, :], repeats=overfit_n, axis=0).astype(np.float32)

        prior = BoxUniform(
            low=torch.tensor([mf_min, chi_min], dtype=torch.float32),
            high=torch.tensor([mf_max, chi_max], dtype=torch.float32),
        )
        est_cfg = snpe.get("estimator", {})
        model_name = str(est_cfg.get("type", "maf")).lower()
        if model_name in {"normalizing_flow", "flow"}:
            model_name = "maf"
        embedding_net = build_embedding_net(input_dim=feat_extractor.feature_dim, estimator_cfg=est_cfg)
        density_builder = posterior_nn(
            model=model_name,
            z_score_theta=est_cfg.get("z_score_theta", "independent"),
            z_score_x=est_cfg.get("z_score_x", "independent"),
            hidden_features=int(est_cfg.get("hidden_features", 128)),
            num_transforms=int(est_cfg.get("num_transforms", 5)),
            embedding_net=embedding_net,
        )
        inference = NPE(prior=prior, density_estimator=density_builder, device="cpu", show_progress_bars=False)

        theta_t = torch.from_numpy(theta_overfit)
        x_t = torch.from_numpy(x_overfit)
        inference.append_simulations(theta_t, x_t)
        estimator = inference.train(
            training_batch_size=min(16, overfit_n),
            learning_rate=5e-4,
            max_num_epochs=int(args.overfit_epochs),
            stop_after_epochs=int(args.overfit_epochs),
            validation_fraction=0.2,
            show_train_summary=False,
        )
        loss_curve = list(map(float, inference.summary.get("training_loss", [])))

        posterior = inference.build_posterior(estimator).set_default_x(x_t[0])
        samples = posterior.sample((2000,), show_progress_bars=False).detach().cpu().numpy()
        post_std_mf = float(np.std(samples[:, 0]))
        post_std_chi = float(np.std(samples[:, 1]))
        overfit_report = {
            "ok": True,
            "train_loss_first": float(loss_curve[0]) if loss_curve else None,
            "train_loss_last": float(loss_curve[-1]) if loss_curve else None,
            "posterior_std_mf": post_std_mf,
            "posterior_std_chi": post_std_chi,
        }
    except Exception as exc:  # pragma: no cover - diagnostic path
        overfit_report = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "train_loss_first": None,
            "train_loss_last": None,
            "posterior_std_mf": None,
            "posterior_std_chi": None,
        }

    snr_obs = float(optimal_snr(d_signal, dt=dt, fmin_hz=snr_fmin_hz, psd_fn=psd_fn)) if d_signal is not None else None
    snr_report_path = project_root / "outputs" / "diagnostics" / "snr_report.json"
    snr_report_json = None
    if snr_report_path.exists():
        snr_report_json = json.loads(snr_report_path.read_text(encoding="utf-8"))

    pass_flags = {
        "x_not_collapsed": float(np.std(x_batch)) > 1e-10,
        "theta_not_collapsed": float(np.std(theta[:, 0])) > 1e-3 and float(np.std(theta[:, 1])) > 1e-4,
        "x_standardized_close_to_unit": abs(float(np.mean(x_std))) < 0.1 and 0.8 < float(np.std(x_std)) < 1.2,
        "injection_snr_close_to_target": (snr_obs is not None) and abs(snr_obs - snr_target) / snr_target < 0.05,
        "simulated_snr_scale_reasonable": (
            float(np.quantile(snr_batch_after, 0.5)) < 2.0 * snr_target
            and float(np.quantile(snr_batch_after, 0.05)) > 0.5 * snr_target
        ),
        "micro_overfit_ok": bool(overfit_report.get("ok", False)),
        "micro_overfit_loss_decrease": (
            overfit_report.get("train_loss_first") is not None
            and overfit_report.get("train_loss_last") is not None
            and float(overfit_report["train_loss_last"]) < float(overfit_report["train_loss_first"])
        ),
        "micro_overfit_posterior_contract": (
            overfit_report.get("posterior_std_mf") is not None
            and overfit_report.get("posterior_std_chi") is not None
            and float(overfit_report["posterior_std_mf"]) < 10.0
            and float(overfit_report["posterior_std_chi"]) < 0.15
        ),
    }

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "N": int(args.N),
        "feature_dim": int(feat_extractor.feature_dim),
        "batch_size": n_batch,
        "tensor_checks": {
            "x_raw": {
                "global_mean": float(np.mean(x_batch)),
                "global_std": float(np.std(x_batch)),
                "per_dim_std_min": float(np.min(np.std(x_batch, axis=0))),
                "per_dim_std_max": float(np.max(np.std(x_batch, axis=0))),
            },
            "x_standardized": {
                "global_mean": float(np.mean(x_std)),
                "global_std": float(np.std(x_std)),
                "per_dim_std_min": float(np.min(np.std(x_std, axis=0))),
                "per_dim_std_max": float(np.max(np.std(x_std, axis=0))),
            },
        },
        "label_checks": {
            "theta_mf": _stats_1d(theta[:, 0]),
            "theta_chi": _stats_1d(theta[:, 1]),
            "amp_linear": _stats_1d(amps_all.reshape(-1)),
            "amp_log10": _stats_1d(np.log10(np.clip(amps_all.reshape(-1), 1e-300, None))),
        },
        "snr_checks": {
            "snr_estimator": "noise.optimal_snr",
            "snr_fmin_hz": float(snr_fmin_hz),
            "feature_fmin_hz": float(feat_fmin_hz),
            "target": snr_target,
            "observation_signal_snr": snr_obs,
            "amplitude_sampling": amp_sampling,
            "snr_conditioning_enabled": bool(snr_cond_enabled),
            "snr_batch_size": int(n_snr),
            "simulated_signal_snr_before": _stats_1d(snr_batch_before),
            "simulated_signal_snr_after": _stats_1d(snr_batch_after),
            "snr_target_samples": _stats_1d(snr_targets),
            "snr_scale_samples": _stats_1d(snr_scales),
            "snr_report_json": snr_report_json,
        },
        "micro_overfit": overfit_report,
        "pass_flags": pass_flags,
        "pass_all": bool(all(pass_flags.values())),
    }

    out_dir = project_root / "outputs" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix if args.output_prefix else f"snpe_rescue_N{int(args.N)}"
    report_path = out_dir / f"{prefix}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if loss_curve:
        fig_path = out_dir / f"{prefix}_loss.png"
        plt.figure(figsize=(7.2, 4.2))
        plt.plot(loss_curve, lw=1.3)
        plt.xlabel("Epoch")
        plt.ylabel("Training NLL")
        plt.title(f"SNPE micro-overfit loss (N={int(args.N)})")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=180)
        plt.close()
        print(f"Wrote: {fig_path}")

    print(f"Wrote: {report_path}")
    print(
        "Summary: "
        f"x_std={report['tensor_checks']['x_raw']['global_std']:.3e}, "
        f"x_stdz={report['tensor_checks']['x_standardized']['global_std']:.3f}, "
        f"snr_obs={snr_obs if snr_obs is not None else float('nan'):.3f}, "
        f"pass_all={report['pass_all']}"
    )


if __name__ == "__main__":
    main()
