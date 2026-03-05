from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MTSUN_SI = 4.925490947e-6
MRSUN_SI = 1476.6250614046494
MPC_SI = 3.085677581491367e22


@dataclass
class ScaledStrain:
    t_sec: np.ndarray
    h_complex: np.ndarray
    h_plus: np.ndarray
    h_cross: np.ndarray
    h_detector: np.ndarray
    y22_iota0: float
    polarization_psi_rad: float
    distance_scale: float


def scale_mode22_to_detector_strain(
    t_M: np.ndarray,
    h22: np.ndarray,
    total_mass_msun: float,
    distance_mpc: float,
    f_plus: float,
    f_cross: float,
    align_polarization_at_peak: bool = False,
) -> ScaledStrain:
    # For iota=0, only the (2,2) contribution is used in this Phase A setup.
    y22_iota0 = float(np.sqrt(5.0 / (4.0 * np.pi)))
    h_complex = h22 * y22_iota0

    psi = 0.0
    if align_polarization_at_peak and h_complex.size:
        peak_idx = int(np.argmax(np.abs(h_complex)))
        psi = -0.5 * float(np.angle(h_complex[peak_idx]))
        h_complex = h_complex * np.exp(2.0j * psi)

    distance_scale = (MRSUN_SI * total_mass_msun) / (distance_mpc * MPC_SI)
    h_complex = h_complex * distance_scale

    h_plus = np.real(h_complex)
    h_cross = -np.imag(h_complex)  # h = h_plus - i h_cross
    h_detector = f_plus * h_plus + f_cross * h_cross

    t_sec = np.asarray(t_M, dtype=float) * (MTSUN_SI * total_mass_msun)

    return ScaledStrain(
        t_sec=t_sec,
        h_complex=h_complex,
        h_plus=h_plus,
        h_cross=h_cross,
        h_detector=h_detector,
        y22_iota0=y22_iota0,
        polarization_psi_rad=psi,
        distance_scale=distance_scale,
    )
