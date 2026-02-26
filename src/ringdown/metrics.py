from __future__ import annotations

import numpy as np


def inner_product(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> complex:
    """Compute <x, y> = integral x(t) * conj(y(t)) dt using trapezoidal rule."""
    if x.shape != y.shape or x.shape != t.shape:
        raise ValueError("x, y, t must have identical shapes")
    if x.ndim != 1:
        raise ValueError("x, y, t must be 1D arrays")
    return np.trapezoid(x * np.conjugate(y), t)


def mismatch(h_nr: np.ndarray, h_model: np.ndarray, t: np.ndarray) -> float:
    """
    Compute mismatch:
    M = 1 - <h_nr, h_model> / sqrt(<h_nr,h_nr><h_model,h_model>)
    Returns real scalar in [0, 2] in normal use.
    """
    num = inner_product(h_nr, h_model, t)
    den = np.sqrt(inner_product(h_nr, h_nr, t) * inner_product(h_model, h_model, t))
    if np.abs(den) == 0:
        raise ValueError("zero norm encountered in mismatch denominator")
    m = 1.0 - np.real(num / den)
    # Numerical safety: small floating deviations can move outside range slightly.
    return float(np.clip(m, 0.0, 2.0))


def remnant_error_epsilon(
    mf_fit: float, mf_true: float, chi_fit: float, chi_true: float, total_mass: float
) -> float:
    """Compute epsilon = sqrt((delta Mf / M)^2 + (delta chi)^2)."""
    if total_mass <= 0:
        raise ValueError("total_mass must be positive")
    dmf = mf_fit - mf_true
    dchi = chi_fit - chi_true
    return float(np.sqrt((dmf / total_mass) ** 2 + dchi**2))
