from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import pickle
from pathlib import Path

import numpy as np
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform
import torch

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


def _parse_n_list(text: str) -> list[int]:
    vals = []
    for tok in text.split(","):
        tok = tok.strip()
        if tok:
            vals.append(int(tok))
    if not vals:
        raise ValueError("Empty N list")
    return vals


def _parse_int_list(text: str) -> list[int]:
    vals: list[int] = []
    for tok in text.split(","):
        tok = tok.strip()
        if tok:
            vals.append(int(tok))
    if not vals:
        raise ValueError("Empty integer list")
    return vals


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase C SNPE training for Fig.10 setup")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ringdown_fig10_snpe/configs/snpe.yaml"),
        help="Path to snpe.yaml",
    )
    parser.add_argument(
        "--N",
        type=str,
        default=None,
        help="Optional comma-separated N list to train, e.g. 0 or 0,1,2,3",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: cap per-round simulations and default to N=0 if --N not provided",
    )
    parser.add_argument(
        "--sim-scale",
        type=float,
        default=1.0,
        help="Scale factor for n_simulations_per_N in each round",
    )
    parser.add_argument(
        "--round-cap",
        type=int,
        default=None,
        help="Optional hard cap of simulations per round",
    )
    parser.add_argument(
        "--posterior-samples",
        type=int,
        default=None,
        help="Override posterior_num_samples from config",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Enable smoke budget override for n_simulations_per_round and faster end-to-end checks.",
    )
    parser.add_argument(
        "--smoke-sims",
        type=str,
        default="5000,5000,2000",
        help="Comma-separated smoke simulations per round, e.g. 5000,5000,2000",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional runtime device override, e.g. cpu/cuda/mps",
    )
    return parser.parse_args()


def _sample_from_proposal(proposal, n: int) -> torch.Tensor:
    try:
        return proposal.sample((n,), show_progress_bars=False)
    except TypeError:
        return proposal.sample((n,))


def _sample_from_proposal_with_bounds(
    proposal,
    n: int,
    low: np.ndarray,
    high: np.ndarray,
    batch_size: int = 4096,
    max_batches: int = 1000,
) -> torch.Tensor:
    accepted: list[torch.Tensor] = []
    accepted_count = 0
    total_drawn = 0
    low_arr = np.asarray(low, dtype=float)[None, :]
    high_arr = np.asarray(high, dtype=float)[None, :]

    for _ in range(max_batches):
        if accepted_count >= n:
            break
        draw_n = max(int(batch_size), int(n - accepted_count))
        cand = _sample_from_proposal(proposal, draw_n).to(dtype=torch.float32)
        cand_np = np.asarray(cand.detach().cpu(), dtype=float)
        mask = np.all((cand_np >= low_arr) & (cand_np <= high_arr), axis=1)
        total_drawn += len(cand_np)
        if np.any(mask):
            keep = cand[torch.from_numpy(mask.astype(np.bool_)).to(device=cand.device)]
            accepted.append(keep)
            accepted_count += int(keep.shape[0])

    if accepted_count < n:
        rate = accepted_count / max(total_drawn, 1)
        raise RuntimeError(
            f"Failed truncated proposal sampling: requested={n}, accepted={accepted_count}, "
            f"drawn={total_drawn}, acceptance_rate={rate:.4e}"
        )

    return torch.cat(accepted, dim=0)[:n]


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


def _summarize_array(arr: np.ndarray) -> dict[str, float]:
    x = np.asarray(arr, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "q05": float(np.quantile(x, 0.05)),
        "q50": float(np.quantile(x, 0.50)),
        "q95": float(np.quantile(x, 0.95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _compute_sanity_stats(theta_np: np.ndarray, x_np: np.ndarray) -> dict[str, float]:
    if theta_np.ndim != 2 or theta_np.shape[1] < 2:
        raise ValueError(f"theta must have shape (n, >=2), got {theta_np.shape}")
    if x_np.ndim != 2:
        raise ValueError(f"x must have shape (n, d), got {x_np.shape}")

    x_dim_mu = np.mean(x_np, axis=0, keepdims=True)
    x_dim_std = np.std(x_np, axis=0, keepdims=True)
    x_dim_std_safe = x_dim_std.copy()
    x_dim_std_safe[x_dim_std_safe < 1e-12] = 1.0
    x_std = (x_np - x_dim_mu) / x_dim_std_safe

    return {
        "x_global_mean": float(np.mean(x_np)),
        "x_global_std": float(np.std(x_np)),
        "x_min": float(np.min(x_np)),
        "x_max": float(np.max(x_np)),
        "x_dim_std_min": float(np.min(x_dim_std)),
        "x_dim_std_max": float(np.max(x_dim_std)),
        "x_standardized_global_mean": float(np.mean(x_std)),
        "x_standardized_global_std": float(np.std(x_std)),
        "theta_mf_mean": float(np.mean(theta_np[:, 0])),
        "theta_mf_std": float(np.std(theta_np[:, 0])),
        "theta_chi_mean": float(np.mean(theta_np[:, 1])),
        "theta_chi_std": float(np.std(theta_np[:, 1])),
    }


def _apply_sanity_gate(stats: dict[str, float], gate_cfg: dict, N: int, round_idx: int) -> None:
    if not bool(gate_cfg.get("enabled", True)):
        return

    failed: list[str] = []
    x_std = float(stats["x_global_std"])
    if (not np.isfinite(x_std)) or x_std < float(gate_cfg.get("x_global_std_min", 1e-12)):
        failed.append("x_global_std")

    theta_mf_std = float(stats["theta_mf_std"])
    if (not np.isfinite(theta_mf_std)) or theta_mf_std < float(gate_cfg.get("theta_mf_std_min", 1e-6)):
        failed.append("theta_mf_std")

    theta_chi_std = float(stats["theta_chi_std"])
    if (not np.isfinite(theta_chi_std)) or theta_chi_std < float(gate_cfg.get("theta_chi_std_min", 1e-6)):
        failed.append("theta_chi_std")

    x_std_mu_abs = abs(float(stats["x_standardized_global_mean"]))
    if x_std_mu_abs > float(gate_cfg.get("x_standardized_mean_abs_max", 0.2)):
        failed.append("x_standardized_global_mean")

    x_std_std = float(stats["x_standardized_global_std"])
    if x_std_std < float(gate_cfg.get("x_standardized_std_min", 0.5)) or x_std_std > float(
        gate_cfg.get("x_standardized_std_max", 1.5)
    ):
        failed.append("x_standardized_global_std")

    if failed:
        raise RuntimeError(
            "SNPE sanity gate failed for "
            f"N={N}, round={round_idx}, failed={failed}, stats={json.dumps(stats, ensure_ascii=False)}"
        )


def main() -> None:
    args = _parse_args()
    snpe_cfg_path = args.config.resolve()
    project_root = project_root_from_config(snpe_cfg_path)

    cfg = load_yaml(snpe_cfg_path)
    fig10_cfg = load_yaml(_resolve_path(project_root, cfg["use_fig10_spec_from"]))
    snpe = cfg["snpe"]
    obs_repr = str(snpe.get("observation_representation", "whitened_fft_features")).strip().lower()
    if obs_repr not in {"whitened_fft_features", "whitened_fft_raw"}:
        raise ValueError(f"Unsupported observation_representation={obs_repr!r}")

    if args.N:
        n_list = _parse_n_list(args.N)
    else:
        n_list = list(map(int, snpe["N_list"]))
        if args.fast:
            n_list = [0]

    seed = int(fig10_cfg["reproducibility"]["random_seed_global"]) + 101
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_npz = project_root / "data" / "injection" / "d_obs.npz"
    if not obs_npz.exists():
        raise FileNotFoundError(f"{obs_npz} not found. Run Phase A first.")
    # 固定观测 d_obs：SNPE 训练最终要学习 p(theta | x_obs)。
    obs = np.load(obs_npz)
    t_sec = np.asarray(obs["t_sec"], dtype=float)
    d_obs = np.asarray(obs["d_obs"], dtype=float)
    dt = float(obs["dt"])

    psd_path = _resolve_path(project_root, fig10_cfg["injection"]["detector"]["noise"]["psd_file"])
    psd_fn = build_psd_interpolator_from_asd_file(psd_path)

    feat_cfg = snpe.get("features", {})
    snr_fmin_hz = float(fig10_cfg["injection"]["snr_definition"]["integration_fmin_hz"])
    feat_fmin_hz = float(feat_cfg.get("fmin_hz", snr_fmin_hz))
    fmax_hz = feat_cfg.get("fmax_hz", None)
    fmax_hz = float(fmax_hz) if fmax_hz is not None else None
    n_fft_cfg = feat_cfg.get("n_fft", None)
    n_fft = int(n_fft_cfg) if n_fft_cfg is not None else None
    n_bins_cfg = feat_cfg.get("n_bins", None)
    n_freq_points_cfg = feat_cfg.get("n_freq_points", None)
    if obs_repr == "whitened_fft_raw":
        # keep full in-band FFT bins by default for raw representation
        if n_bins_cfg is None and n_freq_points_cfg is None:
            n_freq_points_cfg = 0
    n_bins = int(n_bins_cfg) if n_bins_cfg is not None else None
    n_freq_points = int(n_freq_points_cfg) if n_freq_points_cfg is not None else None
    # 频域特征提取：把时序压缩为固定维度 x，减少网络训练难度。
    feat_extractor = build_fixed_fft_feature_extractor(
        n_time=len(d_obs),
        dt=dt,
        psd_fn=psd_fn,
        fmin_hz=feat_fmin_hz,
        n_bins=n_bins,
        n_freq_points=n_freq_points,
        fmax_hz=fmax_hz,
        n_fft=n_fft,
    )
    x_obs = feat_extractor.transform(d_obs)
    x_obs_t = torch.from_numpy(x_obs).to(dtype=torch.float32)

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
    amp_min_log10 = float(np.log10(max(amp_min, 1e-300)))
    amp_max_log10 = float(np.log10(max(amp_max, 1e-300)))

    # 仅对 (Mf, chi_f) 学后验；A_n/phi_n 在模拟器内部抽样并边缘化。
    prior_low = np.array([mf_min, chi_min], dtype=np.float32)
    prior_high = np.array([mf_max, chi_max], dtype=np.float32)

    rounds_cfg = list(snpe["rounds"])
    est_cfg = snpe["estimator"]
    opt_cfg = snpe["optimizer"]
    sim_cfg = snpe.get("simulator", {})
    trunc_cfg = snpe.get("proposal_truncation", {})
    amp_sampling = str(sim_cfg.get("amplitude_sampling", "uniform")).lower()
    if amp_sampling not in {"uniform", "log_uniform"}:
        raise ValueError(f"Unsupported snpe.simulator.amplitude_sampling={amp_sampling!r}")
    snr_cond_cfg = dict(sim_cfg.get("snr_conditioning", {}))
    snr_cond_enabled = bool(snr_cond_cfg.get("enabled", False))
    snr_target_default = float(fig10_cfg["injection"]["snr_definition"]["target_value"])
    snr_min = float(snr_cond_cfg.get("min_signal_snr", 1e-9))
    scale_min = float(snr_cond_cfg.get("scale_min", 1e-3))
    scale_max = float(snr_cond_cfg.get("scale_max", 1e3))
    sanity_cfg = snpe.get("sanity_gate", {})
    runtime_cfg = snpe.get("runtime", {})
    device = str(args.device if args.device else runtime_cfg.get("device", "cpu")).lower()
    simulation_data_device = str(runtime_cfg.get("simulation_data_device", "cpu" if device == "cuda" else device)).lower()
    allow_fallback_cpu = bool(runtime_cfg.get("allow_fallback_cpu", True))
    if device == "cuda" and not torch.cuda.is_available():
        if allow_fallback_cpu:
            print("[runtime] cuda unavailable, fallback to cpu.")
            device = "cpu"
        else:
            raise RuntimeError("runtime.device=cuda but CUDA is not available.")
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    if device == "mps" and not mps_available:
        if allow_fallback_cpu:
            print("[runtime] mps unavailable, fallback to cpu.")
            device = "cpu"
        else:
            raise RuntimeError("runtime.device=mps but MPS backend is not available.")

    if simulation_data_device == "cuda" and not torch.cuda.is_available():
        if allow_fallback_cpu:
            print("[runtime] simulation_data_device=cuda unavailable, fallback to cpu.")
            simulation_data_device = "cpu"
        else:
            raise RuntimeError("runtime.simulation_data_device=cuda but CUDA is not available.")
    if simulation_data_device == "mps" and not mps_available:
        if allow_fallback_cpu:
            print("[runtime] simulation_data_device=mps unavailable, fallback to cpu.")
            simulation_data_device = "cpu"
        else:
            raise RuntimeError("runtime.simulation_data_device=mps but MPS backend is not available.")
    if simulation_data_device not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported runtime.simulation_data_device={simulation_data_device!r}")

    torch_device = torch.device(device)
    simulation_data_torch_device = torch.device(simulation_data_device)
    x_obs_t = x_obs_t.to(device=torch_device)
    prior = BoxUniform(
        low=torch.tensor(prior_low, dtype=torch.float32, device=torch_device),
        high=torch.tensor(prior_high, dtype=torch.float32, device=torch_device),
    )

    posterior_num_samples = int(args.posterior_samples) if args.posterior_samples else int(snpe.get("posterior_num_samples", 40000))
    smoke_sims = _parse_int_list(args.smoke_sims)
    if args.smoke_test and args.posterior_samples is None:
        posterior_num_samples = min(posterior_num_samples, 10000)

    ckpt_dir = _resolve_path(project_root, cfg["outputs"]["checkpoints_dir"])
    post_dir = _resolve_path(project_root, cfg["outputs"]["posterior_samples_dir"])
    diag_dir = project_root / "outputs" / "diagnostics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    post_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    for N in n_list:
        n_modes = int(N) + 1
        qnm_interp = KerrQNMInterpolator(n_max=int(N))
        rng = np.random.default_rng(seed + 1000 * int(N))

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
        inference = NPE(prior=prior, density_estimator=density_builder, device=device)

        round_summaries: list[dict] = []
        proposal = prior
        round_meta_paths: list[str] = []

        def simulate_features(theta_tensor: torch.Tensor) -> tuple[torch.Tensor, dict]:
            theta_np = np.asarray(theta_tensor.detach().cpu(), dtype=float)
            x = np.empty((len(theta_np), feat_extractor.feature_dim), dtype=np.float32)
            rho_before = np.empty(len(theta_np), dtype=float)
            rho_after = np.empty(len(theta_np), dtype=float)
            snr_targets = np.empty(len(theta_np), dtype=float)
            scales = np.empty(len(theta_np), dtype=float)
            for i, (mf, chi) in enumerate(theta_np):
                # nuisance 参数 (A_n, phi_n) 每次重采样，相当于对其做数值边缘化。
                if amp_sampling == "log_uniform":
                    amps = np.power(10.0, rng.uniform(amp_min_log10, amp_max_log10, size=n_modes))
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

                # SNR-conditioned sampling: enforce train simulator SNR regime near observation target.
                rho0 = float(optimal_snr(signal, dt=dt, fmin_hz=snr_fmin_hz, psd_fn=psd_fn))
                target_snr = _draw_target_snr(rng=rng, snr_cfg=snr_cond_cfg, default_target=snr_target_default)
                scale = 1.0
                rho1 = rho0
                if snr_cond_enabled and np.isfinite(rho0) and rho0 > snr_min and np.isfinite(target_snr) and target_snr > 0.0:
                    scale = float(np.clip(target_snr / rho0, scale_min, scale_max))
                    signal = signal * scale
                    rho1 = rho0 * scale

                # 训练数据使用 colored Gaussian noise + PSD，与观测口径保持一致。
                noise = generate_colored_gaussian_noise(len(t_sec), dt=dt, rng=rng, psd_fn=psd_fn)
                x[i, :] = feat_extractor.transform(signal + noise)
                rho_before[i] = rho0
                rho_after[i] = rho1
                scales[i] = scale
                snr_targets[i] = target_snr

            snr_diag = {
                "enabled": bool(snr_cond_enabled),
                "mode": str(snr_cond_cfg.get("mode", "fixed")),
                "snr_estimator": "noise.optimal_snr",
                "snr_fmin_hz": float(snr_fmin_hz),
                "feature_fmin_hz": float(feat_fmin_hz),
                "target_default": float(snr_target_default),
                "target_stats": _summarize_array(snr_targets),
                "rho_before_stats": _summarize_array(rho_before),
                "rho_after_stats": _summarize_array(rho_after),
                "scale_stats": _summarize_array(scales),
            }
            return torch.from_numpy(x).to(dtype=torch.float32, device=theta_tensor.device), snr_diag

        for r_i, r_cfg in enumerate(rounds_cfg):
            if args.smoke_test:
                n_sim = int(smoke_sims[min(r_i, len(smoke_sims) - 1)])
            else:
                n_sim = int(max(200, round(float(r_cfg["n_simulations_per_N"]) * float(args.sim_scale))))
                if args.fast:
                    n_sim = min(n_sim, 2500 if r_i < 2 else 1500)
            if args.round_cap is not None:
                n_sim = min(n_sim, int(args.round_cap))

            truncation_applied = False
            bounds_low = None
            bounds_high = None
            meta_rel = None
            if r_i > 0 and bool(trunc_cfg.get("enabled", True)):
                q_low = float(trunc_cfg.get("quantile_low", 0.005))
                q_high = float(trunc_cfg.get("quantile_high", 0.995))
                n_probe = int(trunc_cfg.get("proposal_probe_samples", 12000))
                probe = _sample_from_proposal(proposal, n_probe).to(dtype=torch.float32)
                probe_np = np.asarray(probe.detach().cpu(), dtype=float)
                bounds_low = np.quantile(probe_np, q_low, axis=0)
                bounds_high = np.quantile(probe_np, q_high, axis=0)
                theta = _sample_from_proposal_with_bounds(
                    proposal=proposal,
                    n=n_sim,
                    low=bounds_low,
                    high=bounds_high,
                    batch_size=int(trunc_cfg.get("draw_batch_size", max(4096, n_sim))),
                    max_batches=int(trunc_cfg.get("max_batches", 1000)),
                ).to(dtype=torch.float32, device=simulation_data_torch_device)
                truncation_applied = True

                meta = {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "N": int(N),
                    "round_id": int(r_cfg.get("id", r_i)),
                    "seed_global": int(seed),
                    "proposal_type": str(r_cfg.get("proposal", "unknown")),
                    "quantile_low": q_low,
                    "quantile_high": q_high,
                    "proposal_probe_samples": int(n_probe),
                    "bounds": {
                        "Mf_msun": [float(bounds_low[0]), float(bounds_high[0])],
                        "chi_f": [float(bounds_low[1]), float(bounds_high[1])],
                    },
                }
                meta_path = post_dir / f"snpe_N{N}_meta_round_{int(r_cfg.get('id', r_i))}.json"
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                meta_rel = str(meta_path.relative_to(project_root).as_posix())
                round_meta_paths.append(meta_rel)
            else:
                theta = _sample_from_proposal(proposal, n_sim).to(dtype=torch.float32, device=simulation_data_torch_device)

            x, snr_diag = simulate_features(theta)
            x = x.to(dtype=torch.float32)
            stats = _compute_sanity_stats(
                theta_np=np.asarray(theta.detach().cpu(), dtype=float),
                x_np=np.asarray(x.detach().cpu(), dtype=float),
            )
            _apply_sanity_gate(stats=stats, gate_cfg=sanity_cfg, N=int(N), round_idx=int(r_i))

            # 多轮 SNPE：首轮用先验采样，后续轮用上一轮后验作为 proposal。
            if r_i == 0:
                inference.append_simulations(theta, x, data_device=simulation_data_device)
            else:
                inference.append_simulations(theta, x, proposal=proposal, data_device=simulation_data_device)

            density_estimator = inference.train(
                training_batch_size=int(opt_cfg.get("batch_size", 512)),
                learning_rate=float(opt_cfg.get("learning_rate", 1e-3)),
                max_num_epochs=int(opt_cfg.get("max_epochs", 200)),
                stop_after_epochs=int(opt_cfg.get("early_stopping_patience", 20)),
                show_train_summary=False,
            )
            posterior = inference.build_posterior(density_estimator)
            proposal = posterior.set_default_x(x_obs_t)

            round_summaries.append(
                {
                    "round_id": int(r_cfg.get("id", r_i)),
                    "n_simulations": int(n_sim),
                    "proposal_type": str(r_cfg.get("proposal", "unknown")),
                    "sanity_stats": stats,
                    "truncation_applied": bool(truncation_applied),
                    "proposal_bounds_mf_chi": (
                        [[float(bounds_low[0]), float(bounds_high[0])], [float(bounds_low[1]), float(bounds_high[1])]]
                        if truncation_applied
                        else None
                    ),
                    "proposal_meta_file": meta_rel,
                    "snr_conditioning": snr_diag,
                }
            )
            print(
                f"[SNPE sanity] N={N} round={r_i} "
                f"x_std={stats['x_global_std']:.3e} "
                f"theta_std=(mf={stats['theta_mf_std']:.3e}, chi={stats['theta_chi_std']:.3e}) "
                f"x_stdz={stats['x_standardized_global_std']:.3f} "
                f"rho_before_q50={snr_diag['rho_before_stats']['q50']:.2f} "
                f"rho_after_q50={snr_diag['rho_after_stats']['q50']:.2f}"
            )

        posterior_samples = _sample_from_proposal(proposal, posterior_num_samples).detach().cpu().numpy()

        pkl_path = ckpt_dir / f"snpe_N{N}_posterior.pkl"
        with pkl_path.open("wb") as f:
            pickle.dump(proposal, f)

        npz_path = post_dir / f"snpe_N{N}_posterior_samples.npz"
        np.savez(
            npz_path,
            samples=posterior_samples,
            x_obs_feature=x_obs,
            selected_freqs_hz=feat_extractor.selected_freqs_hz,
            N=int(N),
        )

        train_summary = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "target_N": int(N),
            "n_modes": int(n_modes),
            "observation_representation": obs_repr,
            "feature_dim": int(feat_extractor.feature_dim),
            "n_bins": int(feat_extractor.n_bins),
            "device": device,
            "simulation_data_device": simulation_data_device,
            "posterior_num_samples": int(posterior_num_samples),
            "amplitude_sampling": amp_sampling,
            "snr_conditioning": {
                "enabled": bool(snr_cond_enabled),
                "mode": str(snr_cond_cfg.get("mode", "fixed")),
                "snr_estimator": "noise.optimal_snr",
                "snr_fmin_hz": float(snr_fmin_hz),
                "feature_fmin_hz": float(feat_fmin_hz),
                "target_default": float(snr_target_default),
            },
            "proposal_truncation_enabled": bool(trunc_cfg.get("enabled", True)),
            "proposal_meta_files": round_meta_paths,
            "rounds": round_summaries,
            "files": {
                "posterior_pickle": str(pkl_path.relative_to(project_root).as_posix()),
                "posterior_samples_npz": str(npz_path.relative_to(project_root).as_posix()),
            },
        }
        summary_path = diag_dir / f"snpe_N{N}_train.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(train_summary, f, ensure_ascii=False, indent=2)

        print(f"SNPE training complete for N={N}")
        print(f"Wrote: {pkl_path}")
        print(f"Wrote: {npz_path}")
        print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
