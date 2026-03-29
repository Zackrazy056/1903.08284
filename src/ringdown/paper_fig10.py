from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

import numpy as np

from .conventions import (
    FACE_ON_PLUS_FACTOR_22,
    MPC_METERS,
    MSUN_METERS,
    MSUN_SEC,
    finite_duration_from_uniform_time_samples,
    detector_real_plus_from_mode22_face_on,
    paper_fig10_convention_summary,
)
from .fd_likelihood import (
    continuous_ft_from_time_series,
    draw_colored_noise_rfft,
    optimal_snr,
)
from .sxs_io import SXSRemnantInfo, load_sxs_waveform22


@dataclass(frozen=True)
class PaperFigure10Priors:
    mf_bounds_msun: tuple[float, float] = (10.0, 100.0)
    chif_bounds: tuple[float, float] = (0.0, 1.0)
    amp_bounds_rel: tuple[float, float] = (0.01, 250.0)
    phi_bounds: tuple[float, float] = (0.0, 2.0 * pi)


@dataclass(frozen=True)
class PaperFigure10Config:
    sxs_location: str = "SXS:BBH:0305v2.0/Lev6"
    total_mass_msun: float = 72.0
    distance_mpc: float = 400.0
    delta_t0_ms: float = 0.0
    t_end_m: float = 90.0
    f_min_hz: float = 20.0
    f_max_hz: float = 1024.0
    df_hz: float = 1.0
    priors: PaperFigure10Priors = field(default_factory=PaperFigure10Priors)
    download: bool = True


@dataclass(frozen=True)
class PaperFigure10Signal:
    config: PaperFigure10Config
    info: SXSRemnantInfo
    t_mode_m: np.ndarray
    h22_mode: np.ndarray
    detector_strain_full: np.ndarray
    t_peak_complex_m: float
    t_hpeak_m: float
    delta_hpeak_minus_complex_peak_m: float
    delta_hpeak_minus_complex_peak_ms: float
    amplitude_scale: float
    h_peak: float
    t_rel_m: np.ndarray
    t0_m: float
    delta_t0_m: float
    delta_t0_ms: float
    t_window_m: np.ndarray
    tau_window_m: np.ndarray
    tau_window_sec: np.ndarray
    signal_window: np.ndarray
    freqs_hz: np.ndarray
    psd: np.ndarray
    valid_mask: np.ndarray
    signal_tilde: np.ndarray
    duration_sec: float
    postpeak_optimal_snr: float
    true_mf_msun: float
    true_chif: float
    psd_source: str


@dataclass(frozen=True)
class PaperFigure10Observation:
    signal: PaperFigure10Signal
    noise_tilde: np.ndarray
    d_tilde: np.ndarray


def physical_strain_scale(total_mass_msun: float, distance_mpc: float) -> float:
    if total_mass_msun <= 0.0:
        raise ValueError("total_mass_msun must be positive")
    if distance_mpc <= 0.0:
        raise ValueError("distance_mpc must be positive")
    return float((total_mass_msun * MSUN_METERS) / (distance_mpc * MPC_METERS))


def face_on_plus_only_detector_channel_from_mode22(h22: np.ndarray) -> np.ndarray:
    """
    Paper-faithful detector channel for Fig.10.

    This is not a generic detector response. It is the specific real,
    plus-only, face-on projection used when injecting only the dominant
    ``(l=m=2)`` mode in the paper-faithful detector study.
    """
    return detector_real_plus_from_mode22_face_on(h22)


def detector_plus_from_mode22(h22: np.ndarray) -> np.ndarray:
    """Backwards-compatible alias for `face_on_plus_only_detector_channel_from_mode22`."""
    return face_on_plus_only_detector_channel_from_mode22(h22)


def build_aligo_design_psd(freqs_hz: np.ndarray) -> tuple[np.ndarray, str]:
    freqs = np.asarray(freqs_hz, dtype=float)
    if freqs.ndim != 1:
        raise ValueError("freqs_hz must be 1D")
    if np.any(freqs < 0.0):
        raise ValueError("freqs_hz must be nonnegative")
    try:
        from bilby.gw.detector import PowerSpectralDensity
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "bilby is required for the paper-faithful Advanced LIGO design PSD"
        ) from exc
    psd = PowerSpectralDensity.from_aligo().get_power_spectral_density_array(freqs)
    return np.asarray(psd, dtype=float), "bilby_aLIGOZeroDetHighPower"


def build_paper_fig10_signal(config: PaperFigure10Config) -> PaperFigure10Signal:
    wf, info = load_sxs_waveform22(location=config.sxs_location, download=config.download)
    if info.remnant_mass is None or info.remnant_chif_z is None:
        raise ValueError("missing remnant metadata")

    t_mode_m = np.asarray(wf.t, dtype=float)
    h22_mode = np.asarray(wf.h, dtype=complex)
    detector_strain_full = face_on_plus_only_detector_channel_from_mode22(h22_mode) * physical_strain_scale(
        config.total_mass_msun,
        config.distance_mpc,
    )

    idx_complex_peak = int(np.argmax(np.abs(h22_mode)))
    idx_hpeak = int(np.argmax(np.abs(detector_strain_full)))
    t_peak_complex_m = float(t_mode_m[idx_complex_peak])
    t_hpeak_m = float(t_mode_m[idx_hpeak])
    m_sec = MSUN_SEC * config.total_mass_msun
    delta_hpeak_minus_complex_peak_m = float(t_hpeak_m - t_peak_complex_m)
    delta_hpeak_minus_complex_peak_ms = float(delta_hpeak_minus_complex_peak_m * m_sec * 1e3)

    t_rel_m = t_mode_m - t_hpeak_m
    delta_t0_m = float(config.delta_t0_ms / (m_sec * 1e3))
    t0_m = delta_t0_m

    mask = (t_rel_m >= t0_m) & (t_rel_m <= config.t_end_m)
    t_window_m = t_rel_m[mask]
    signal_window = detector_strain_full[mask]
    if t_window_m.size < 64:
        raise ValueError("analysis window too short for paper Fig.10 signal")

    tau_window_m = t_window_m - t0_m
    dt_m = float(np.median(np.diff(tau_window_m)))
    tau_u_m = np.arange(0.0, float(tau_window_m[-1]) + 0.5 * dt_m, dt_m)
    signal_u = np.interp(tau_u_m, tau_window_m, signal_window)
    tau_u_sec = tau_u_m * m_sec

    freqs_hz = np.arange(config.f_min_hz, config.f_max_hz + 0.5 * config.df_hz, config.df_hz)
    if freqs_hz.size < 16:
        raise ValueError("frequency grid too small")
    psd, psd_source = build_aligo_design_psd(freqs_hz)
    valid_mask = (
        (freqs_hz >= config.f_min_hz)
        & (freqs_hz <= config.f_max_hz)
        & np.isfinite(psd)
        & (psd > 0.0)
        & (freqs_hz > 0.0)
    )
    if np.count_nonzero(valid_mask) < 16:
        raise ValueError("too few valid PSD bins for paper Fig.10 signal")

    signal_tilde = continuous_ft_from_time_series(tau_u_sec, signal_u, freqs_hz)
    postpeak_optimal_snr = optimal_snr(signal_tilde, psd, config.df_hz, valid_mask=valid_mask)

    return PaperFigure10Signal(
        config=config,
        info=info,
        t_mode_m=t_mode_m,
        h22_mode=h22_mode,
        detector_strain_full=detector_strain_full,
        t_peak_complex_m=t_peak_complex_m,
        t_hpeak_m=t_hpeak_m,
        delta_hpeak_minus_complex_peak_m=delta_hpeak_minus_complex_peak_m,
        delta_hpeak_minus_complex_peak_ms=delta_hpeak_minus_complex_peak_ms,
        amplitude_scale=physical_strain_scale(config.total_mass_msun, config.distance_mpc),
        h_peak=float(np.max(np.abs(detector_strain_full))),
        t_rel_m=t_rel_m,
        t0_m=t0_m,
        delta_t0_m=delta_t0_m,
        delta_t0_ms=float(config.delta_t0_ms),
        t_window_m=t_window_m,
        tau_window_m=tau_u_m,
        tau_window_sec=tau_u_sec,
        signal_window=signal_u,
        freqs_hz=freqs_hz,
        psd=psd,
        valid_mask=valid_mask,
        signal_tilde=signal_tilde,
        duration_sec=finite_duration_from_uniform_time_samples(tau_u_sec),
        postpeak_optimal_snr=float(postpeak_optimal_snr),
        true_mf_msun=float(info.remnant_mass * config.total_mass_msun),
        true_chif=float(info.remnant_chif_z),
        psd_source=psd_source,
    )


def inject_paper_fig10_noise(
    signal: PaperFigure10Signal,
    rng: np.random.Generator,
) -> PaperFigure10Observation:
    noise_tilde = draw_colored_noise_rfft(
        rng,
        signal.freqs_hz.size,
        signal.psd,
        signal.config.df_hz,
        enforce_real_endpoints=True,
    )
    return PaperFigure10Observation(
        signal=signal,
        noise_tilde=noise_tilde,
        d_tilde=signal.signal_tilde + noise_tilde,
    )


def paper_fig10_signal_diagnostics(signal: PaperFigure10Signal) -> dict[str, float | str]:
    diagnostics: dict[str, float | str] = {
        "sxs_location": signal.config.sxs_location,
        "total_mass_msun": signal.config.total_mass_msun,
        "distance_mpc": signal.config.distance_mpc,
        "true_mf_msun": signal.true_mf_msun,
        "true_chif": signal.true_chif,
        "h_peak": signal.h_peak,
        "postpeak_optimal_snr": signal.postpeak_optimal_snr,
        "t_peak_complex_m": signal.t_peak_complex_m,
        "t_hpeak_m": signal.t_hpeak_m,
        "delta_hpeak_minus_complex_peak_m": signal.delta_hpeak_minus_complex_peak_m,
        "delta_hpeak_minus_complex_peak_ms": signal.delta_hpeak_minus_complex_peak_ms,
        "t0_minus_hpeak_m": signal.delta_t0_m,
        "t0_minus_hpeak_ms": signal.delta_t0_ms,
        "analysis_duration_sec": signal.duration_sec,
        "psd_source": signal.psd_source,
    }
    diagnostics.update(paper_fig10_convention_summary())
    return diagnostics
