from __future__ import annotations

import numpy as np

from .types import Waveform22


def window_waveform(
    wf: Waveform22, t0: float, t_end: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return time and strain samples in [t0, t_end] (or [t0, +inf) if t_end is None)."""
    if t_end is None:
        mask = wf.t >= t0
    else:
        mask = (wf.t >= t0) & (wf.t <= t_end)
    return wf.t[mask], wf.h[mask]


def interp_complex(
    t_src: np.ndarray, h_src: np.ndarray, t_target: np.ndarray
) -> np.ndarray:
    """Linear interpolation for complex time series."""
    h_re = np.interp(t_target, t_src, h_src.real)
    h_im = np.interp(t_target, t_src, h_src.imag)
    return h_re + 1j * h_im


def phase_align_to_reference_at_tref(
    t_ref: float,
    t_reference: np.ndarray,
    h_reference: np.ndarray,
    t_target: np.ndarray,
    h_target: np.ndarray,
) -> np.ndarray:
    """
    Apply a constant phase rotation to target so phases match reference at t_ref.
    """
    h_ref = interp_complex(t_reference, h_reference, np.array([t_ref], dtype=float))[0]
    h_tgt = interp_complex(t_target, h_target, np.array([t_ref], dtype=float))[0]
    if np.abs(h_tgt) == 0:
        return h_target
    phase_shift = np.angle(h_ref) - np.angle(h_tgt)
    return h_target * np.exp(1j * phase_shift)


def align_time_and_phase_by_window(
    t_reference: np.ndarray,
    h_reference: np.ndarray,
    t_target: np.ndarray,
    h_target: np.ndarray,
    *,
    t_start: float,
    t_end: float,
    dt_search_half_width: float = 2.0,
    dt_step: float = 0.002,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Find constant time shift and phase rotation for target that best match reference
    on [t_start, t_end] in least-squares sense.

    Returns:
    - shifted target time array
    - phase-rotated target strain
    - best dt (target time is shifted by -dt)
    - best phase shift (target multiplied by exp(i*phase))
    """
    if t_end <= t_start:
        raise ValueError("t_end must be greater than t_start")
    if dt_step <= 0:
        raise ValueError("dt_step must be positive")

    dt_candidates = np.arange(-dt_search_half_width, dt_search_half_width + 0.5 * dt_step, dt_step)
    best_obj = np.inf
    best_dt = 0.0
    best_phase = 0.0

    for dt in dt_candidates:
        t_shifted = t_target - dt
        left = max(t_start, t_reference[0], t_shifted[0])
        right = min(t_end, t_reference[-1], t_shifted[-1])
        if right - left <= 0:
            continue
        mask = (t_reference >= left) & (t_reference <= right)
        if np.count_nonzero(mask) < 5:
            continue

        t_eval = t_reference[mask]
        h_ref_eval = h_reference[mask]
        h_tgt_eval = interp_complex(t_shifted, h_target, t_eval)

        c = np.trapezoid(np.conjugate(h_tgt_eval) * h_ref_eval, t_eval)
        phase = 0.0 if np.abs(c) == 0 else np.angle(c)
        h_tgt_aligned = h_tgt_eval * np.exp(1j * phase)
        obj = float(np.trapezoid(np.abs(h_ref_eval - h_tgt_aligned) ** 2, t_eval))
        if obj < best_obj:
            best_obj = obj
            best_dt = float(dt)
            best_phase = float(phase)

    t_best = t_target - best_dt
    h_best = h_target * np.exp(1j * best_phase)
    return t_best, h_best, best_dt, best_phase
