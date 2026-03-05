from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearFitResult:
    # QNM coefficients C_n (constant offset, if enabled, is reported separately).
    coeffs: np.ndarray
    model: np.ndarray
    residual_norm: float
    rank: int
    singular_values: np.ndarray
    condition_number: float
    constant_offset: complex | None


def build_design_matrix(t: np.ndarray, omegas: np.ndarray, t0: float) -> np.ndarray:
    """Build A_{kn} = exp(-i omega_n (t_k - t0))."""
    if t.ndim != 1 or omegas.ndim != 1:
        raise ValueError("t and omegas must be 1D arrays")
    if omegas.size < 1:
        raise ValueError("at least one mode is required")
    dt = t[:, None] - t0
    return np.exp(-1j * dt * omegas[None, :])


def solve_complex_lstsq(
    t: np.ndarray,
    h: np.ndarray,
    omegas: np.ndarray,
    t0: float,
    *,
    lstsq_rcond: float = 1e-12,
    include_constant_offset: bool = True,
) -> LinearFitResult:
    """
    Solve complex linear least squares for QNM amplitudes C_n.

    If include_constant_offset=True, augment basis with a constant complex term b.
    """
    if t.shape != h.shape:
        raise ValueError("t and h must have same shape")
    if t.ndim != 1:
        raise ValueError("t and h must be 1D arrays")
    if t.size <= omegas.size:
        raise ValueError("need more data samples than number of modes")

    a_base = build_design_matrix(t, omegas, t0)
    if include_constant_offset:
        a = np.column_stack([a_base, np.ones((t.size, 1), dtype=complex)])
    else:
        a = a_base

    coeffs_full, residuals, rank, svals = np.linalg.lstsq(a, h, rcond=lstsq_rcond)
    model = a @ coeffs_full

    if include_constant_offset:
        coeffs = np.asarray(coeffs_full[: omegas.size], dtype=complex)
        constant_offset = complex(coeffs_full[-1])
    else:
        coeffs = np.asarray(coeffs_full, dtype=complex)
        constant_offset = None
    if svals.size == 0:
        cond = float("inf")
    elif svals[-1] <= 0:
        cond = float("inf")
    else:
        cond = float(np.abs(svals[0] / svals[-1]))

    if residuals.size > 0:
        residual_norm = float(np.sqrt(np.real(residuals[0])))
    else:
        residual_norm = float(np.linalg.norm(h - model))

    return LinearFitResult(
        coeffs=coeffs,
        model=model,
        residual_norm=residual_norm,
        rank=int(rank),
        singular_values=svals,
        condition_number=cond,
        constant_offset=constant_offset,
    )
