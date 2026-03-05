from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform
import torch

from config_io import load_yaml, project_root_from_config
from noise import build_psd_interpolator_from_asd_file, generate_colored_gaussian_noise
from qnm_kerr import KerrQNMInterpolator
from ringdown_eq1 import ringdown_plus_eq1
from summarize import build_fixed_fft_feature_extractor


NON_PRODUCTION_REASONS = [
    "Uses micro-dataset and aggressive overfit settings for debugging only.",
    "Uses legacy n_bins=64 fallback instead of production raw-FFT feature settings.",
    "Uses log-uniform amplitude sampling without production snr_conditioning.",
]


def _resolve_path(project_root: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def _scale_unit_interval_to_minus1_plus1(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return 2.0 * (x - lo) / (hi - lo) - 1.0


def _scale_minus1_plus1_to_unit_interval(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return lo + 0.5 * (x + 1.0) * (hi - lo)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round-3 cold-start try #1: extreme data standardization + micro overfitting check"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ringdown_fig10_snpe/configs/snpe.yaml"),
        help="Path to snpe.yaml",
    )
    parser.add_argument("--N", type=int, default=1, help="Overtone order for cold-start micro run")
    parser.add_argument("--n-samples", type=int, default=100, help="Micro dataset size for sanity overfitting check")
    parser.add_argument("--max-epochs", type=int, default=220, help="Max epochs for cold-start run")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    snpe_cfg_path = args.config.resolve()
    project_root = project_root_from_config(snpe_cfg_path)
    snpe_cfg = load_yaml(snpe_cfg_path)
    fig10_cfg = load_yaml(_resolve_path(project_root, snpe_cfg["use_fig10_spec_from"]))

    seed = int(fig10_cfg["reproducibility"]["random_seed_global"]) + 301
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    obs_npz = project_root / "data" / "injection" / "d_obs.npz"
    if not obs_npz.exists():
        raise FileNotFoundError(f"{obs_npz} not found. Run Phase A first.")
    obs = np.load(obs_npz)
    t_sec = np.asarray(obs["t_sec"], dtype=float)
    dt = float(obs["dt"])

    psd_path = _resolve_path(project_root, fig10_cfg["injection"]["detector"]["noise"]["psd_file"])
    psd_fn = build_psd_interpolator_from_asd_file(psd_path)
    fmin_hz = float(fig10_cfg["injection"]["snr_definition"]["integration_fmin_hz"])

    snpe = snpe_cfg["snpe"]
    feat_cfg = snpe.get("features", {})
    feat_extractor = build_fixed_fft_feature_extractor(
        n_time=len(t_sec),
        dt=dt,
        psd_fn=psd_fn,
        fmin_hz=fmin_hz,
        n_bins=int(feat_cfg.get("n_bins", 64)),
        fmax_hz=float(feat_cfg.get("fmax_hz", 1024.0)),
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
    log_amp_min = float(np.log10(amp_min))
    log_amp_max = float(np.log10(amp_max))

    n_modes = int(args.N) + 1
    qnm_interp = KerrQNMInterpolator(n_max=int(args.N))
    n_samples = int(args.n_samples)

    # Build fixed micro dataset (for sanity overfitting).
    theta_phys = np.empty((n_samples, 2), dtype=np.float32)
    x_raw = np.empty((n_samples, feat_extractor.feature_dim), dtype=np.float32)
    for i in range(n_samples):
        mf = rng.uniform(mf_min, mf_max)
        chi = rng.uniform(chi_min, chi_max)
        theta_phys[i, 0] = mf
        theta_phys[i, 1] = chi

        # Sample amplitudes in log-space, equivalent to predicting log10(A_n) in stabilized regime.
        log_amps = rng.uniform(log_amp_min, log_amp_max, size=n_modes)
        amps = np.power(10.0, log_amps)
        phis = rng.uniform(phi_min, phi_max, size=n_modes)

        signal = ringdown_plus_eq1(
            t_sec=t_sec,
            mf_msun=float(mf),
            chi_f=float(chi),
            amplitudes=amps,
            phases=phis,
            qnm_interp=qnm_interp,
        )
        noise = generate_colored_gaussian_noise(len(t_sec), dt=dt, rng=rng, psd_fn=psd_fn)
        x_raw[i, :] = feat_extractor.transform(signal + noise)

    # Suggestion #1 implementation: strict standardization.
    x_mu = x_raw.mean(axis=0, keepdims=True)
    x_sigma = x_raw.std(axis=0, keepdims=True)
    x_sigma[x_sigma < 1e-8] = 1.0
    x_std = (x_raw - x_mu) / x_sigma

    mf_scaled = _scale_unit_interval_to_minus1_plus1(theta_phys[:, 0], mf_min, mf_max)
    chi_scaled = _scale_unit_interval_to_minus1_plus1(theta_phys[:, 1], chi_min, chi_max)
    theta_scaled = np.column_stack([mf_scaled, chi_scaled]).astype(np.float32)

    prior = BoxUniform(
        low=torch.tensor([-1.0, -1.0], dtype=torch.float32),
        high=torch.tensor([1.0, 1.0], dtype=torch.float32),
    )
    density_builder = posterior_nn(model="maf", hidden_features=64, num_transforms=5)
    inference = NPE(prior=prior, density_estimator=density_builder, device="cpu")

    theta_t = torch.from_numpy(theta_scaled)
    x_t = torch.from_numpy(x_std.astype(np.float32))
    inference.append_simulations(theta_t, x_t)
    density_estimator = inference.train(
        training_batch_size=int(args.batch_size),
        learning_rate=float(args.lr),
        max_num_epochs=int(args.max_epochs),
        stop_after_epochs=int(args.max_epochs),  # avoid early stop in micro sanity run
        validation_fraction=0.2,
        show_train_summary=False,
    )
    posterior = inference.build_posterior(density_estimator)

    # Evaluate train NLL and posterior collapse on a training point.
    lp = density_estimator.log_prob(theta_t.unsqueeze(0), x_t).detach().cpu().numpy().reshape(-1)
    mean_nll = float(-np.mean(lp))
    min_nll = float(-np.max(lp))

    x0 = x_t[0]
    posterior_samples_scaled = posterior.set_default_x(x0).sample((2000,), show_progress_bars=False).detach().cpu().numpy()
    post_std_scaled = posterior_samples_scaled.std(axis=0, ddof=1)
    post_std_phys = np.array(
        [
            0.5 * (mf_max - mf_min) * post_std_scaled[0],
            0.5 * (chi_max - chi_min) * post_std_scaled[1],
        ],
        dtype=float,
    )

    train_loss = list(map(float, inference.summary.get("training_loss", [])))
    val_loss = list(map(float, inference.summary.get("validation_loss", [])))
    epochs = int(inference.summary.get("epochs_trained", [len(train_loss)])[0]) if train_loss else 0

    # Decode example posterior mean back to physical units.
    post_mean_scaled = posterior_samples_scaled.mean(axis=0)
    post_mean_phys = np.array(
        [
            _scale_minus1_plus1_to_unit_interval(np.array([post_mean_scaled[0]]), mf_min, mf_max)[0],
            _scale_minus1_plus1_to_unit_interval(np.array([post_mean_scaled[1]]), chi_min, chi_max)[0],
        ],
        dtype=float,
    )

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "round3_coldstart_try1_data_standardization",
        "production_parity": False,
        "non_production_reasons": NON_PRODUCTION_REASONS,
        "N": int(args.N),
        "n_samples": n_samples,
        "feature_dim": int(feat_extractor.feature_dim),
        "standardization_checks": {
            "x_raw_global_mean": float(np.mean(x_raw)),
            "x_raw_global_std": float(np.std(x_raw)),
            "x_std_global_mean": float(np.mean(x_std)),
            "x_std_global_std": float(np.std(x_std)),
            "x_std_mean_abs_max_per_dim": float(np.max(np.abs(np.mean(x_std, axis=0)))),
            "x_std_std_min_per_dim": float(np.min(np.std(x_std, axis=0))),
            "x_std_std_max_per_dim": float(np.max(np.std(x_std, axis=0))),
        },
        "parameter_scaling_checks": {
            "mf_scaled_minmax": [float(np.min(mf_scaled)), float(np.max(mf_scaled))],
            "chi_scaled_minmax": [float(np.min(chi_scaled)), float(np.max(chi_scaled))],
        },
        "micro_overfit": {
            "epochs_trained": epochs,
            "training_loss_first": float(train_loss[0]) if train_loss else None,
            "training_loss_last": float(train_loss[-1]) if train_loss else None,
            "training_loss_min": float(np.min(train_loss)) if train_loss else None,
            "validation_loss_last": float(val_loss[-1]) if val_loss else None,
            "mean_nll_on_train": mean_nll,
            "min_nll_on_train": min_nll,
            "posterior_std_scaled": [float(post_std_scaled[0]), float(post_std_scaled[1])],
            "posterior_std_phys": [float(post_std_phys[0]), float(post_std_phys[1])],
            "posterior_mean_phys_for_first_x": [float(post_mean_phys[0]), float(post_mean_phys[1])],
            "first_train_theta_phys": [float(theta_phys[0, 0]), float(theta_phys[0, 1])],
        },
        "pass_flags": {
            "standardization_close_to_unit_gaussian": bool(abs(np.mean(x_std)) < 0.1 and 0.8 < np.std(x_std) < 1.2),
            "theta_scaled_inside_minus1_plus1": bool(np.min(theta_scaled) >= -1.0001 and np.max(theta_scaled) <= 1.0001),
            "nll_below_zero_on_train": bool(mean_nll < 0.0),
            "posterior_concentrates_for_first_x": bool(float(np.max(post_std_scaled)) < 0.2),
        },
    }

    out_diag = project_root / "outputs" / "diagnostics"
    out_diag.mkdir(parents=True, exist_ok=True)
    report_path = out_diag / "npe_coldstart_try1_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    loss_plot_path = out_diag / "npe_coldstart_try1_loss.png"
    plt.figure(figsize=(7.2, 4.4))
    if train_loss:
        plt.plot(train_loss, label="train_nll")
    if val_loss:
        plt.plot(val_loss, label="val_nll")
    plt.xlabel("Epoch")
    plt.ylabel("NLL")
    plt.title("Cold-start micro overfit (100 samples)")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=180)
    plt.close()

    print("Round-3 cold-start try #1 complete")
    print("NOTE: This script is a non-production debug path and should not be used as SNPE production quality evidence.")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {loss_plot_path}")
    print(
        "Summary: "
        f"x_std_mean={report['standardization_checks']['x_std_global_mean']:.4f}, "
        f"x_std_std={report['standardization_checks']['x_std_global_std']:.4f}, "
        f"mean_train_nll={mean_nll:.4f}, "
        f"posterior_std_scaled_max={float(np.max(post_std_scaled)):.4f}"
    )


if __name__ == "__main__":
    main()
