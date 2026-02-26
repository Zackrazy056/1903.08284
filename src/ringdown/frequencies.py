from __future__ import annotations

from functools import lru_cache

import numpy as np
import qnm


@lru_cache(maxsize=256)
def _cached_mode_function(s: int, ell: int, m: int, n: int):
    return qnm.modes_cache(s=s, l=ell, m=m, n=n)


def kerr_qnm_omega_lmn(
    mf: float, chif: float, ell: int, m: int, n: int, s: int = -2
) -> complex:
    """
    Return complex QNM frequency omega_lmn(Mf, chif) in time units of total mass M.

    qnm package returns dimensionless M_f * omega(chif); divide by M_f to get omega.
    """
    if mf <= 0:
        raise ValueError("mf must be positive")
    if not (-0.999 <= chif <= 0.999):
        raise ValueError("chif must satisfy -0.999 <= chif <= 0.999")
    if ell < 2:
        raise ValueError("ell must be >= 2")
    if abs(m) > ell:
        raise ValueError("must satisfy |m| <= ell")
    if n < 0:
        raise ValueError("n must be nonnegative")

    if chif >= 0:
        mode = _cached_mode_function(s=s, ell=ell, m=m, n=n)
        w_bar, _, _ = mode(a=chif)
        return complex(w_bar) / mf

    # Kerr symmetry for negative spin:
    # omega_{lmn}(a, m) = -conj(omega_{l,-m,n}(-a))
    # Here qnm returns dimensionless M_f * omega.
    mode_neg_m = _cached_mode_function(s=s, ell=ell, m=-m, n=n)
    w_bar_neg_m, _, _ = mode_neg_m(a=abs(chif))
    return -np.conjugate(complex(w_bar_neg_m)) / mf


def kerr_qnm_omegas_22n(mf: float, chif: float, n_max: int, s: int = -2) -> np.ndarray:
    """Return [omega_220, ..., omega_22n_max] as complex ndarray."""
    if n_max < 0:
        raise ValueError("n_max must be nonnegative")
    return np.array(
        [kerr_qnm_omega_lmn(mf=mf, chif=chif, ell=2, m=2, n=n, s=s) for n in range(n_max + 1)],
        dtype=complex,
    )


def make_omega_provider_22(s: int = -2):
    """
    Factory for grid_search_remnant:
    provider(mf, chif, n_overtones) -> omegas[0..n_overtones]
    """

    def provider(mf: float, chif: float, n_overtones: int) -> np.ndarray:
        return kerr_qnm_omegas_22n(mf=mf, chif=chif, n_max=n_overtones, s=s)

    return provider
