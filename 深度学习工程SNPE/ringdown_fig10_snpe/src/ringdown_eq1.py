from __future__ import annotations

import numpy as np

from qnm_kerr import KerrQNMInterpolator


def ringdown_plus_eq1(
    t_sec: np.ndarray,
    mf_msun: float,
    chi_f: float,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    qnm_interp: KerrQNMInterpolator,
    bias: float = 0.0,
) -> np.ndarray:
    """Eq.(1)-style overtone model projected to plus polarization for face-on setup."""
    t = np.asarray(t_sec, dtype=float)
    amps = np.asarray(amplitudes, dtype=float)
    phs = np.asarray(phases, dtype=float)
    if amps.shape != phs.shape:
        raise ValueError("amplitudes and phases must have same shape")

    # 物理模型：h_+(t) = sum_n A_n exp(Im(omega_n)t) cos(Re(omega_n)t + phi_n) + bias
    # 其中 Im(omega_n)<0 时对应阻尼衰减。
    h_plus = np.zeros_like(t, dtype=float)
    for n, (amp, phi) in enumerate(zip(amps, phs)):
        # (Mf, chi_f) 通过 QNM 频率进入时域波形，直接控制振荡频率和衰减率。
        omega = qnm_interp.omega_22n(mf_msun=mf_msun, chi_f=chi_f, n=n).omega_rad_s
        wr = float(np.real(omega))
        wi = float(np.imag(omega))
        # qnm uses exp[-i*omega*t], with Im(omega)<0 for damped modes.
        h_plus += amp * np.exp(wi * t) * np.cos(wr * t + float(phi))
    return h_plus + float(bias)
