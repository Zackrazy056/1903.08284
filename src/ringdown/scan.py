from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .fit import LinearFitResult, solve_complex_lstsq
from .metrics import mismatch
from .types import Waveform22


@dataclass(frozen=True)
class StartTimeFitResult:
    t0: float
    mismatch: float
    residual_norm: float
    coeffs: np.ndarray
    n_samples: int
    condition_number: float


@dataclass(frozen=True)
class RemnantGridResult:
    best_mf: float
    best_chif: float
    best_mismatch: float
    best_coeffs: np.ndarray
    mismatch_grid: np.ndarray
    valid_mask: np.ndarray


def _window_waveform(
    wf: Waveform22, t0: float, t_end: float | None
) -> tuple[np.ndarray, np.ndarray]:
    if t_end is None:
        mask = wf.t >= t0
    else:
        mask = (wf.t >= t0) & (wf.t <= t_end)
    t_win = wf.t[mask]
    h_win = wf.h[mask]
    return t_win, h_win


def _relative_overtone_amplitude_excess(
    coeffs: np.ndarray, max_overtone_to_fund_ratio: float
) -> bool:
    """
    Return True if any overtone amplitude is implausibly larger than the fundamental.
    """
    if coeffs.size <= 1:
        return False
    a0 = float(np.abs(coeffs[0]))
    overtone_max = float(np.max(np.abs(coeffs[1:])))
    if a0 == 0:
        return overtone_max > 0
    return (overtone_max / a0) > max_overtone_to_fund_ratio


def fit_at_start_time(
    wf: Waveform22,
    omegas: np.ndarray,
    t0: float,
    t_end: float | None = None,
    *,
    lstsq_rcond: float = 1e-12,
    max_condition_number: float | None = 1e12,
    max_overtone_to_fund_ratio: float | None = 1e4,
    min_signal_norm: float = 1e-14,
) -> tuple[StartTimeFitResult, LinearFitResult]:
    t_win, h_win = _window_waveform(wf, t0=t0, t_end=t_end)
    if t_win.size <= omegas.size:
        raise ValueError("insufficient samples in fitting window")
    signal_norm = float(np.trapezoid(np.abs(h_win) ** 2, t_win))
    if signal_norm <= min_signal_norm:
        raise ValueError("signal norm too small in fitting window")

    lin = solve_complex_lstsq(
        t=t_win,
        h=h_win,
        omegas=omegas,
        t0=t0,
        lstsq_rcond=lstsq_rcond,
    )
    if max_condition_number is not None and lin.condition_number > max_condition_number:
        raise ValueError("design matrix ill-conditioned")
    if max_overtone_to_fund_ratio is not None and _relative_overtone_amplitude_excess(
        lin.coeffs, max_overtone_to_fund_ratio=max_overtone_to_fund_ratio
    ):
        raise ValueError("overtone amplitudes are unphysical at this t0")

    mm = mismatch(h_nr=h_win, h_model=lin.model, t=t_win)
    return (
        StartTimeFitResult(
            t0=float(t0),
            mismatch=float(mm),
            residual_norm=lin.residual_norm,
            coeffs=lin.coeffs,
            n_samples=int(t_win.size),
            condition_number=lin.condition_number,
        ),
        lin,
    )


def scan_start_times_fixed_omegas(
    wf: Waveform22,
    omegas: np.ndarray,
    t0_grid: np.ndarray,
    t_end: float | None = None,
    *,
    lstsq_rcond: float = 1e-12,
    max_condition_number: float | None = 1e12,
    max_overtone_to_fund_ratio: float | None = 1e4,
    min_signal_norm: float = 1e-14,
) -> list[StartTimeFitResult]:
    results: list[StartTimeFitResult] = []
    for t0 in t0_grid:
        t_win, _ = _window_waveform(wf, t0=float(t0), t_end=t_end)
        if t_win.size <= omegas.size:
            continue
        try:
            fit_result, _ = fit_at_start_time(
                wf=wf,
                omegas=omegas,
                t0=float(t0),
                t_end=t_end,
                lstsq_rcond=lstsq_rcond,
                max_condition_number=max_condition_number,
                max_overtone_to_fund_ratio=max_overtone_to_fund_ratio,
                min_signal_norm=min_signal_norm,
            )
        except ValueError:
            continue
        results.append(fit_result)
    return results


def grid_search_remnant(
    wf: Waveform22,
    n_overtones: int,
    t0: float,
    mf_grid: np.ndarray,
    chif_grid: np.ndarray,
    omega_provider: Callable[[float, float, int], np.ndarray],
    t_end: float | None = None,
    *,
    lstsq_rcond: float = 1e-12,
    max_condition_number: float | None = 1e12,
    max_overtone_to_fund_ratio: float | None = 1e4,
    min_signal_norm: float = 1e-14,
) -> RemnantGridResult:
    """
    Evaluate mismatch over (M_f, chi_f) grid.

    omega_provider(mf, chif, n_overtones) must return complex array of length n_overtones+1.
    """
    mismatch_map = np.full((mf_grid.size, chif_grid.size), np.nan, dtype=float)
    valid_mask = np.zeros((mf_grid.size, chif_grid.size), dtype=bool)
    best_mm = np.inf
    best_mf = float("nan")
    best_chif = float("nan")
    best_coeffs = np.array([], dtype=complex)

    for i, mf in enumerate(mf_grid):
        for j, chif in enumerate(chif_grid):
            omegas = omega_provider(float(mf), float(chif), int(n_overtones))
            if omegas.ndim != 1 or omegas.size != n_overtones + 1:
                raise ValueError("omega_provider returned invalid shape")
            try:
                fit_result, _ = fit_at_start_time(
                    wf=wf,
                    omegas=omegas,
                    t0=t0,
                    t_end=t_end,
                    lstsq_rcond=lstsq_rcond,
                    max_condition_number=max_condition_number,
                    max_overtone_to_fund_ratio=max_overtone_to_fund_ratio,
                    min_signal_norm=min_signal_norm,
                )
            except ValueError:
                continue
            mismatch_map[i, j] = fit_result.mismatch
            valid_mask[i, j] = True
            if fit_result.mismatch < best_mm:
                best_mm = fit_result.mismatch
                best_mf = float(mf)
                best_chif = float(chif)
                best_coeffs = fit_result.coeffs

    return RemnantGridResult(
        best_mf=best_mf,
        best_chif=best_chif,
        best_mismatch=float(best_mm),
        best_coeffs=best_coeffs,
        mismatch_grid=mismatch_map,
        valid_mask=valid_mask,
    )
