from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np
from sbi.neural_nets import posterior_nn
import torch
from torch.utils.data import DataLoader, Dataset

from config_io import load_yaml, project_root_from_config
from noise import build_psd_interpolator_from_asd_file, generate_colored_gaussian_noise
from qnm_kerr import KerrQNMInterpolator
from ringdown_eq1 import ringdown_plus_eq1
from summarize import build_fixed_fft_feature_extractor


def _resolve_path(project_root: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def _scale_to_minus1_plus1(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return 2.0 * (x - lo) / (hi - lo) - 1.0


def _scale_to_physical(x_scaled: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return lo + 0.5 * (x_scaled + 1.0) * (hi - lo)


@dataclass
class CleanSample:
    theta_scaled: np.ndarray
    clean_signal: np.ndarray


class CleanWaveformDataset(Dataset):
    def __init__(self, samples: list[CleanSample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        item = self._samples[idx]
        return item.theta_scaled, item.clean_signal


class DynamicNoiseCollator:
    def __init__(self, dt: float, psd_fn, rng: np.random.Generator, noise_scale: float = 1.0) -> None:
        self._dt = float(dt)
        self._psd_fn = psd_fn
        self._rng = rng
        self.noise_scale = float(noise_scale)

    def __call__(self, batch: list[tuple[np.ndarray, np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor]:
        theta, clean = zip(*batch)
        theta_arr = np.asarray(theta, dtype=np.float32)
        clean_arr = np.asarray(clean, dtype=float)
        noisy = np.empty_like(clean_arr)
        for i in range(clean_arr.shape[0]):
            n = generate_colored_gaussian_noise(
                n_samples=clean_arr.shape[1],
                dt=self._dt,
                rng=self._rng,
                psd_fn=self._psd_fn,
            )
            noisy[i, :] = clean_arr[i, :] + self.noise_scale * n
        return torch.from_numpy(theta_arr), torch.from_numpy(noisy.astype(np.float32))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round-3 cold-start try #2: dynamic noise injection + 100-sample strong overfit tuning"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ringdown_fig10_snpe/configs/snpe.yaml"),
        help="Path to snpe.yaml",
    )
    parser.add_argument("--N", type=int, default=1, help="Overtone order for cold-start micro run")
    parser.add_argument("--n-samples", type=int, default=100, help="Micro dataset size for sanity overfitting check")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--epochs-dynamic", type=int, default=80, help="Epochs with dynamic noise injection")
    parser.add_argument("--epochs-overfit", type=int, default=260, help="Epochs with frozen clean-wave overfit")
    parser.add_argument("--noise-floor-scale", type=float, default=0.0, help="Noise scale in overfit stage")
    parser.add_argument("--hidden-features", type=int, default=192, help="Hidden features for MAF")
    parser.add_argument("--num-transforms", type=int, default=8, help="MAF transform count")
    parser.add_argument(
        "--renorm-on-overfit-stage",
        action="store_true",
        help="Recompute feature normalization stats at overfit-stage noise level before strong overfit",
    )
    parser.add_argument("--seed-offset", type=int, default=302, help="Seed offset from fig10 global seed")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="npe_coldstart_try2",
        help="Output artifact prefix under outputs/diagnostics",
    )
    return parser.parse_args()


def _iter_epoch_losses(
    estimator,
    loader: DataLoader,
    feat_extractor,
    feature_mu: np.ndarray,
    feature_sigma: np.ndarray,
    optimizer: torch.optim.Optimizer,
) -> Iterator[float]:
    for theta_batch, noisy_batch in loader:
        noisy_np = noisy_batch.detach().cpu().numpy()
        feats = np.empty((noisy_np.shape[0], feat_extractor.feature_dim), dtype=np.float32)
        for i in range(noisy_np.shape[0]):
            feats[i, :] = feat_extractor.transform(noisy_np[i, :])
        x_std = (feats - feature_mu) / feature_sigma
        x_t = torch.from_numpy(x_std.astype(np.float32))
        theta_t = theta_batch.to(dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        # Use estimator-native loss API: returns per-sample negative log-probability.
        neg_lp = estimator.loss(theta_t, x_t).mean()
        neg_lp.backward()
        optimizer.step()
        yield float(neg_lp.detach().cpu().item())


def main() -> None:
    args = _parse_args()
    snpe_cfg_path = args.config.resolve()
    project_root = project_root_from_config(snpe_cfg_path)
    snpe_cfg = load_yaml(snpe_cfg_path)
    fig10_cfg = load_yaml(_resolve_path(project_root, snpe_cfg["use_fig10_spec_from"]))

    seed = int(fig10_cfg["reproducibility"]["random_seed_global"]) + int(args.seed_offset)
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

    n_modes = int(args.N) + 1
    qnm_interp = KerrQNMInterpolator(n_max=int(args.N))

    clean_samples: list[CleanSample] = []
    theta_scaled_rows = []
    calib_features = np.empty((int(args.n_samples), feat_extractor.feature_dim), dtype=np.float32)
    for i in range(int(args.n_samples)):
        mf = rng.uniform(mf_min, mf_max)
        chi = rng.uniform(chi_min, chi_max)
        amps = np.power(10.0, rng.uniform(np.log10(amp_min), np.log10(amp_max), size=n_modes))
        phis = rng.uniform(phi_min, phi_max, size=n_modes)
        clean = ringdown_plus_eq1(
            t_sec=t_sec,
            mf_msun=float(mf),
            chi_f=float(chi),
            amplitudes=amps,
            phases=phis,
            qnm_interp=qnm_interp,
        ).astype(np.float32)
        theta_scaled = np.array(
            [
                _scale_to_minus1_plus1(np.array([mf]), mf_min, mf_max)[0],
                _scale_to_minus1_plus1(np.array([chi]), chi_min, chi_max)[0],
            ],
            dtype=np.float32,
        )
        clean_samples.append(CleanSample(theta_scaled=theta_scaled, clean_signal=clean))
        theta_scaled_rows.append(theta_scaled)

        calib_noise = generate_colored_gaussian_noise(len(t_sec), dt=dt, rng=rng, psd_fn=psd_fn).astype(np.float32)
        calib_features[i, :] = feat_extractor.transform(clean + calib_noise)

    theta_scaled_arr = np.asarray(theta_scaled_rows, dtype=np.float32)
    feature_mu = calib_features.mean(axis=0, keepdims=True)
    feature_sigma = calib_features.std(axis=0, keepdims=True)
    feature_sigma[feature_sigma < 1e-8] = 1.0

    dataset = CleanWaveformDataset(clean_samples)
    collate = DynamicNoiseCollator(dt=dt, psd_fn=psd_fn, rng=rng, noise_scale=1.0)
    loader = DataLoader(
        dataset=dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        drop_last=False,
        collate_fn=collate,
    )

    density_builder = posterior_nn(
        model="maf",
        hidden_features=int(args.hidden_features),
        num_transforms=int(args.num_transforms),
    )
    # Instantiate estimator using one synthetic batch to infer dimensions.
    init_theta, init_noisy = next(iter(loader))
    init_feats = np.empty((init_noisy.shape[0], feat_extractor.feature_dim), dtype=np.float32)
    for i in range(init_noisy.shape[0]):
        init_feats[i, :] = feat_extractor.transform(init_noisy[i, :].detach().cpu().numpy())
    init_x = torch.from_numpy(((init_feats - feature_mu) / feature_sigma).astype(np.float32))
    estimator = density_builder(init_theta.to(dtype=torch.float32), init_x)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=float(args.lr), weight_decay=0.0)

    loss_dynamic: list[float] = []
    for _ in range(int(args.epochs_dynamic)):
        epoch_losses = list(
            _iter_epoch_losses(
                estimator=estimator,
                loader=loader,
                feat_extractor=feat_extractor,
                feature_mu=feature_mu,
                feature_sigma=feature_sigma,
                optimizer=optimizer,
            )
        )
        loss_dynamic.append(float(np.mean(epoch_losses)))

    collate.noise_scale = float(args.noise_floor_scale)
    if args.renorm_on_overfit_stage:
        overfit_calib = np.empty((len(dataset), feat_extractor.feature_dim), dtype=np.float32)
        for i, s in enumerate(clean_samples):
            n = generate_colored_gaussian_noise(len(t_sec), dt=dt, rng=rng, psd_fn=psd_fn) * float(args.noise_floor_scale)
            overfit_calib[i, :] = feat_extractor.transform(s.clean_signal + n.astype(np.float32))
        feature_mu = overfit_calib.mean(axis=0, keepdims=True)
        feature_sigma = overfit_calib.std(axis=0, keepdims=True)
        feature_sigma[feature_sigma < 1e-8] = 1.0

    loss_overfit: list[float] = []
    for _ in range(int(args.epochs_overfit)):
        epoch_losses = list(
            _iter_epoch_losses(
                estimator=estimator,
                loader=loader,
                feat_extractor=feat_extractor,
                feature_mu=feature_mu,
                feature_sigma=feature_sigma,
                optimizer=optimizer,
            )
        )
        loss_overfit.append(float(np.mean(epoch_losses)))

    # Evaluate final train-set NLL using noise_floor stage.
    eval_feats = np.empty((len(dataset), feat_extractor.feature_dim), dtype=np.float32)
    eval_theta = np.empty((len(dataset), 2), dtype=np.float32)
    for i, s in enumerate(clean_samples):
        eval_theta[i, :] = s.theta_scaled
        eval_noise = generate_colored_gaussian_noise(len(t_sec), dt=dt, rng=rng, psd_fn=psd_fn) * float(args.noise_floor_scale)
        eval_feats[i, :] = feat_extractor.transform(s.clean_signal + eval_noise.astype(np.float32))
    eval_x = ((eval_feats - feature_mu) / feature_sigma).astype(np.float32)
    eval_theta_t = torch.from_numpy(eval_theta)
    eval_x_t = torch.from_numpy(eval_x)

    logp = estimator.log_prob(eval_theta_t.unsqueeze(0), eval_x_t).detach().cpu().numpy().reshape(-1)
    mean_nll = float(-np.mean(logp))
    min_nll = float(-np.max(logp))

    x0 = eval_x_t[0].unsqueeze(0)
    s0 = estimator.sample((2000,), condition=x0).squeeze(1).detach().cpu().numpy()
    std_scaled = s0.std(axis=0, ddof=1)
    std_phys = np.array(
        [
            0.5 * (mf_max - mf_min) * std_scaled[0],
            0.5 * (chi_max - chi_min) * std_scaled[1],
        ],
        dtype=float,
    )
    mean_scaled = s0.mean(axis=0)
    mean_phys = np.array(
        [
            _scale_to_physical(np.array([mean_scaled[0]]), mf_min, mf_max)[0],
            _scale_to_physical(np.array([mean_scaled[1]]), chi_min, chi_max)[0],
        ],
        dtype=float,
    )
    theta0_phys = np.array(
        [
            _scale_to_physical(np.array([theta_scaled_arr[0, 0]]), mf_min, mf_max)[0],
            _scale_to_physical(np.array([theta_scaled_arr[0, 1]]), chi_min, chi_max)[0],
        ],
        dtype=float,
    )

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "round3_coldstart_try2_dynamic_noise_plus_strong_overfit",
        "N": int(args.N),
        "n_samples": int(args.n_samples),
        "feature_dim": int(feat_extractor.feature_dim),
        "optimizer": {
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "epochs_dynamic": int(args.epochs_dynamic),
            "epochs_overfit": int(args.epochs_overfit),
            "noise_floor_scale": float(args.noise_floor_scale),
            "maf_hidden_features": int(args.hidden_features),
            "maf_num_transforms": int(args.num_transforms),
            "renorm_on_overfit_stage": bool(args.renorm_on_overfit_stage),
        },
        "standardization_checks": {
            "x_calib_global_mean": float(np.mean(calib_features)),
            "x_calib_global_std": float(np.std(calib_features)),
            "x_std_global_mean": float(np.mean((calib_features - feature_mu) / feature_sigma)),
            "x_std_global_std": float(np.std((calib_features - feature_mu) / feature_sigma)),
            "x_std_mean_abs_max_per_dim": float(np.max(np.abs(np.mean((calib_features - feature_mu) / feature_sigma, axis=0)))),
            "x_std_std_min_per_dim": float(np.min(np.std((calib_features - feature_mu) / feature_sigma, axis=0))),
            "x_std_std_max_per_dim": float(np.max(np.std((calib_features - feature_mu) / feature_sigma, axis=0))),
        },
        "micro_overfit": {
            "loss_dynamic_first": float(loss_dynamic[0]) if loss_dynamic else None,
            "loss_dynamic_last": float(loss_dynamic[-1]) if loss_dynamic else None,
            "loss_overfit_first": float(loss_overfit[0]) if loss_overfit else None,
            "loss_overfit_last": float(loss_overfit[-1]) if loss_overfit else None,
            "loss_overfit_min": float(np.min(loss_overfit)) if loss_overfit else None,
            "mean_nll_on_train": mean_nll,
            "min_nll_on_train": min_nll,
            "posterior_std_scaled": [float(std_scaled[0]), float(std_scaled[1])],
            "posterior_std_phys": [float(std_phys[0]), float(std_phys[1])],
            "posterior_mean_phys_for_first_x": [float(mean_phys[0]), float(mean_phys[1])],
            "first_train_theta_phys": [float(theta0_phys[0]), float(theta0_phys[1])],
        },
        "pass_flags": {
            "standardization_close_to_unit_gaussian": bool(abs(float(np.mean((calib_features - feature_mu) / feature_sigma))) < 0.1),
            "theta_scaled_inside_minus1_plus1": bool(np.min(theta_scaled_arr) >= -1.0001 and np.max(theta_scaled_arr) <= 1.0001),
            "nll_below_zero_on_train": bool(mean_nll < 0.0),
            "posterior_concentrates_for_first_x": bool(float(np.max(std_scaled)) < 0.2),
        },
    }

    out_diag = project_root / "outputs" / "diagnostics"
    out_diag.mkdir(parents=True, exist_ok=True)
    prefix = str(args.output_prefix).strip()
    report_path = out_diag / f"{prefix}_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    loss_plot_path = out_diag / f"{prefix}_loss.png"
    plt.figure(figsize=(8.0, 4.6))
    if loss_dynamic:
        plt.plot(loss_dynamic, label="dynamic_noise_stage")
    if loss_overfit:
        x = np.arange(len(loss_overfit)) + len(loss_dynamic)
        plt.plot(x, loss_overfit, label="strong_overfit_stage")
    plt.axvline(len(loss_dynamic), color="k", linestyle="--", linewidth=1.0, alpha=0.6)
    plt.xlabel("Epoch")
    plt.ylabel("NLL")
    plt.title("Cold-start try2: dynamic noise + strong overfit tuning")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=180)
    plt.close()

    print("Round-3 cold-start try #2 complete")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {loss_plot_path}")
    print(
        "Summary: "
        f"mean_nll={mean_nll:.4f}, "
        f"min_nll={min_nll:.4f}, "
        f"posterior_std_scaled_max={float(np.max(std_scaled)):.4f}"
    )


if __name__ == "__main__":
    main()
