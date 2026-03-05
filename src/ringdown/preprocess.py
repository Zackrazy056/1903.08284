from __future__ import annotations

from dataclasses import replace

import numpy as np

from .types import Waveform22


def peak_time_from_strain(wf: Waveform22) -> float:
    """Return t_peak where |h22| is maximal."""
    # t_peak 的定义会影响后续 t0 网格和拟合窗口，是关键系统量。
    idx = int(np.argmax(np.abs(wf.h)))
    return float(wf.t[idx])


def shift_time(wf: Waveform22, dt: float) -> Waveform22:
    """Return waveform with shifted time t -> t - dt."""
    return replace(wf, t=wf.t - dt)


def align_to_peak(wf: Waveform22) -> tuple[Waveform22, float]:
    """Shift waveform so peak strain is at t=0."""
    t_peak = peak_time_from_strain(wf)
    return shift_time(wf, t_peak), t_peak


def crop_time(wf: Waveform22, t_min: float, t_max: float) -> Waveform22:
    """Keep samples within [t_min, t_max]."""
    mask = (wf.t >= t_min) & (wf.t <= t_max)
    if np.count_nonzero(mask) < 3:
        raise ValueError("cropped waveform has fewer than 3 samples")
    return replace(wf, t=wf.t[mask], h=wf.h[mask])


def resample_uniform(wf: Waveform22, dt: float) -> Waveform22:
    """Resample complex waveform onto uniform grid via linear interpolation."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    t0, t1 = float(wf.t[0]), float(wf.t[-1])
    n = int(np.floor((t1 - t0) / dt)) + 1
    t_new = t0 + np.arange(n) * dt
    h_re = np.interp(t_new, wf.t, wf.h.real)
    h_im = np.interp(t_new, wf.t, wf.h.imag)
    return replace(wf, t=t_new, h=h_re + 1j * h_im)


def build_start_time_grid(
    t_peak: float,
    m_total: float = 1.0,
    rel_start_m: float = -25.0,
    rel_end_m: float = 60.0,
    step_m: float = 1.0,
) -> np.ndarray:
    """
    Build start-time grid:
    t0 in [t_peak + rel_start_m*M, t_peak + rel_end_m*M], inclusive if exact.
    """
    if m_total <= 0 or step_m <= 0:
        raise ValueError("m_total and step_m must be positive")
    # 以 M 为单位构造等间距 t0 网格，便于与论文中的 Δt0/M 对齐。
    start = t_peak + rel_start_m * m_total
    end = t_peak + rel_end_m * m_total
    n = int(np.floor((end - start) / (step_m * m_total))) + 1
    return start + np.arange(n) * (step_m * m_total)
