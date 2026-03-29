from __future__ import annotations

from dataclasses import asdict, dataclass
from math import pi, sqrt

import numpy as np


MSUN_SEC = 4.92549095e-6
MSUN_METERS = 1476.6250614046494
MPC_METERS = 3.085677581491367e22
FACE_ON_PLUS_FACTOR_22 = sqrt(5.0 / (4.0 * pi))


@dataclass(frozen=True)
class PhysicalConventions:
    detector_channel: str
    peak_definition: str
    psd_source: str
    finite_duration_convention: str
    mass_unit_convention: str
    epsilon_mass_scale: str


PAPER_FIG10_CONVENTIONS = PhysicalConventions(
    detector_channel="plus-only real detector channel from face-on (l=m=2) mode",
    peak_definition="detector-strain peak t_h-peak",
    psd_source="bilby aLIGO design PSD",
    finite_duration_convention="duration_sec = tau_window_sec[-1] = (N-1)dt on the uniform post-t0 grid",
    mass_unit_convention="NR times in total-mass geometric units; Fourier model evaluated in SI seconds",
    epsilon_mass_scale="initial total binary mass M, matching paper Eq.(4)",
)


def detector_real_plus_from_mode22_face_on(h22: np.ndarray) -> np.ndarray:
    h22_arr = np.asarray(h22, dtype=complex)
    if h22_arr.ndim != 1:
        raise ValueError("h22 must be a 1D complex array")
    return np.real(FACE_ON_PLUS_FACTOR_22 * h22_arr)


def finite_duration_from_uniform_time_samples(t_sec: np.ndarray) -> float:
    """
    Return the finite-duration window length used by the analytic FD templates.

    The sealed convention is:
        duration_sec = t_sec[-1] - t_sec[0]
    on the uniform post-t0 sampling grid. For grids starting at zero, this is
    equal to ``(N-1)dt`` rather than ``N dt``.
    """
    t = np.asarray(t_sec, dtype=float)
    if t.ndim != 1:
        raise ValueError("t_sec must be 1D")
    if t.size < 2:
        raise ValueError("need at least two time samples")
    if np.any(np.diff(t) < 0.0):
        raise ValueError("t_sec must be nondecreasing")
    return float(t[-1] - t[0])


def paper_fig10_convention_summary() -> dict[str, str]:
    return {key: str(value) for key, value in asdict(PAPER_FIG10_CONVENTIONS).items()}
