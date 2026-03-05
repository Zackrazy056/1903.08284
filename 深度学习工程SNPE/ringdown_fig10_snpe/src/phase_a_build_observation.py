from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request

import numpy as np

from config_io import load_yaml, project_root_from_config
from noise import build_psd_interpolator_from_asd_file, generate_colored_gaussian_noise, optimal_snr
from peak_alignment import compute_peak_alignment, save_peak_alignment_plot
from sxs_io import load_sxs_mode22
from units_scaling import scale_mode22_to_detector_strain


def _resample_uniform_real(t_src: np.ndarray, x_src: np.ndarray, fs_hz: float, t_start: float, t_end: float) -> tuple[np.ndarray, np.ndarray]:
    dt = 1.0 / fs_hz
    n = int(np.floor((t_end - t_start) / dt)) + 1
    n = max(n, 2)
    t_new = t_start + np.arange(n, dtype=float) * dt
    x_new = np.interp(t_new, t_src, x_src)
    return t_new, x_new


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase A: Build d_obs and diagnostics for Fig.10 setup")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ringdown_fig10_snpe/configs/fig10.yaml"),
        help="Path to fig10 config yaml",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        help="Uniform resampling rate in Hz for d_obs and SNR calculation (overrides config)",
    )
    parser.add_argument(
        "--sxs-lev",
        type=str,
        default="highest",
        help="SXS resolution level, e.g. highest, 6, Lev6",
    )
    return parser.parse_args()


def _ensure_psd_file(psd_path: Path, psd_url: str) -> None:
    if psd_path.exists():
        return
    psd_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(psd_url, timeout=60) as r:
        data = r.read()
    psd_path.write_bytes(data)


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    cfg = load_yaml(config_path)
    project_root = project_root_from_config(config_path)

    injection_cfg = cfg["injection"]
    sim_id = injection_cfg["nr_source"]["id"]
    total_mass = float(injection_cfg["scaling"]["total_initial_mass_msun_detector_frame"])
    distance_mpc = float(injection_cfg["scaling"]["luminosity_distance_mpc"])
    antenna = injection_cfg["detector"]["geometry"]["antenna_pattern"]
    f_plus = float(antenna["F_plus"])
    f_cross = float(antenna["F_cross"])
    fmin_hz = float(injection_cfg["snr_definition"]["integration_fmin_hz"])
    snr_target = float(injection_cfg["snr_definition"]["target_value"])
    seed = int(cfg["reproducibility"]["random_seed_global"])
    noise_cfg = injection_cfg["detector"]["noise"]
    psd_rel_path = Path(noise_cfg.get("psd_file", "data/psd/ZERO_DET_high_P.txt"))
    psd_url = str(noise_cfg.get("psd_url", "https://dcc.ligo.org/public/0002/T0900288/003/ZERO_DET_high_P.txt"))
    psd_path = (project_root / psd_rel_path).resolve()

    _ensure_psd_file(psd_path=psd_path, psd_url=psd_url)
    psd_fn = build_psd_interpolator_from_asd_file(psd_path)

    numerics = cfg.get("numerics", {})
    sample_rate_hz = float(args.sample_rate) if args.sample_rate is not None else float(numerics.get("sample_rate_hz", 4096.0))

    sxs_data = load_sxs_mode22(sim_id=sim_id, lev=args.sxs_lev)
    scaled = scale_mode22_to_detector_strain(
        t_M=sxs_data.t_M,
        h22=sxs_data.h22,
        total_mass_msun=total_mass,
        distance_mpc=distance_mpc,
        f_plus=f_plus,
        f_cross=f_cross,
        align_polarization_at_peak=False,
    )

    peak = compute_peak_alignment(
        t_sec=scaled.t_sec,
        h_complex=scaled.h_complex,
        h_detector=scaled.h_detector,
    )

    peak_plot = project_root / "outputs" / "diagnostics" / "peak_alignment.png"
    save_peak_alignment_plot(
        t_sec=scaled.t_sec,
        h_complex=scaled.h_complex,
        h_detector=scaled.h_detector,
        result=peak,
        out_path=peak_plot,
    )

    post_mask = scaled.t_sec >= peak.t_h_peak_sec
    t_post = scaled.t_sec[post_mask]
    signal_post = scaled.h_detector[post_mask]
    if len(t_post) < 2:
        raise RuntimeError("Post-peak segment has fewer than 2 points")

    t_uniform, signal_uniform = _resample_uniform_real(
        t_src=t_post,
        x_src=signal_post,
        fs_hz=sample_rate_hz,
        t_start=float(t_post[0]),
        t_end=float(t_post[-1]),
    )

    dt = float(t_uniform[1] - t_uniform[0])
    rng = np.random.default_rng(seed)
    noise = generate_colored_gaussian_noise(n_samples=len(t_uniform), dt=dt, rng=rng, psd_fn=psd_fn)
    d_obs = signal_uniform + noise

    snr_postpeak = optimal_snr(signal_uniform, dt=dt, fmin_hz=fmin_hz, psd_fn=psd_fn)

    # Shift observation time to enforce delta_t0 = 0 at detector-strain peak.
    t_obs = t_uniform - peak.t_h_peak_sec

    injection_out = project_root / "data" / "injection" / "d_obs.npz"
    injection_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        injection_out,
        t_sec=t_obs,
        d_obs=d_obs,
        signal=signal_uniform,
        noise=noise,
        dt=dt,
        sample_rate_hz=sample_rate_hz,
        t_h_peak_sec=0.0,
        t_peak_minus_t_h_peak_ms=(peak.t_peak_sec - peak.t_h_peak_sec) * 1.0e3,
    )

    snr_report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "simulation": sim_id,
        "sxs_lev": sxs_data.lev,
        "snr_postpeak": snr_postpeak,
        "snr_target": snr_target,
        "snr_abs_error": abs(snr_postpeak - snr_target),
        "fmin": fmin_hz,
        "Mf_total": total_mass,
        "D": f"{distance_mpc}Mpc",
        "psd": noise_cfg["psd_name"],
        "psd_file": str(psd_rel_path.as_posix()),
        "delta_t0_ms": float(cfg["inference_model"]["start_time"]["delta_t0_ms"]),
        "peak_times": {
            "t_peak_sec": peak.t_peak_sec,
            "t_h_peak_sec": peak.t_h_peak_sec,
            "t_h_minus_t_peak_ms": peak.delta_t_h_minus_peak_ms,
        },
        "postpeak_duration_ms": float(1.0e3 * (t_uniform[-1] - t_uniform[0])),
        "sample_rate_hz": sample_rate_hz,
        "n_samples": int(len(t_uniform)),
        "truth_parameters": cfg["truth_parameters"],
        "remnant_from_sxs_metadata": {
            "remnant_mass": sxs_data.metadata.get("remnant_mass"),
            "remnant_dimensionless_spin": sxs_data.metadata.get("remnant_dimensionless_spin"),
        },
        "files": {
            "injection_npz": str(injection_out.relative_to(project_root).as_posix()),
            "peak_plot": str(peak_plot.relative_to(project_root).as_posix()),
            "psd_file": str(psd_path.relative_to(project_root).as_posix()),
        },
    }

    report_out = project_root / "outputs" / "diagnostics" / "snr_report.json"
    report_out.parent.mkdir(parents=True, exist_ok=True)
    with report_out.open("w", encoding="utf-8") as f:
        json.dump(snr_report, f, ensure_ascii=False, indent=2)

    print("Phase A complete")
    print(f"SXS level: Lev{sxs_data.lev}")
    print(f"t_h-peak - t_peak [ms]: {peak.delta_t_h_minus_peak_ms:.4f}")
    print(f"Post-peak optimal SNR (f>{fmin_hz} Hz): {snr_postpeak:.3f} (target {snr_target})")
    print(f"Wrote: {injection_out}")
    print(f"Wrote: {peak_plot}")
    print(f"Wrote: {report_out}")


if __name__ == "__main__":
    main()
