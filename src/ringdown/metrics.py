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
    mf_fit: float,
    mf_true: float,
    chi_fit: float,
    chi_true: float,
    mass_scale_msun: float | None = None,
    *,
    total_mass: float | None = None,
) -> float:
    """
    Compute epsilon = sqrt((delta Mf / M)^2 + (delta chi)^2).

    The preferred argument name is ``mass_scale_msun`` to make the normalization
    explicit. For the current paper-faithful convention this mass scale is the
    initial total binary mass ``M`` from Eq.(4), not the remnant mass.

    ``total_mass`` is retained as a backwards-compatible alias.
    """
    if mass_scale_msun is None:
        mass_scale_msun = total_mass
    if mass_scale_msun is None or mass_scale_msun <= 0:
        raise ValueError("mass_scale_msun must be positive")
    dmf = mf_fit - mf_true
    dchi = chi_fit - chi_true
    return float(np.sqrt((dmf / mass_scale_msun) ** 2 + dchi**2))
