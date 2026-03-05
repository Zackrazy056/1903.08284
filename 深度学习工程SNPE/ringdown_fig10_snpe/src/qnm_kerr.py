from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import qnm

from units_scaling import MTSUN_SI


@dataclass
class QNMOmega:
    omega_rad_s: complex
    omega_geom: complex


class KerrQNMInterpolator:
    """Fast interpolator for Kerr QNM omega_22n(Mf, chi) with n in [0, n_max]."""

    def __init__(self, n_max: int = 3, n_grid: int = 1201, chi_min: float = 0.0, chi_max: float = 0.999):
        if n_max < 0:
            raise ValueError("n_max must be >= 0")
        if n_grid < 8:
            raise ValueError("n_grid must be >= 8")
        if not (0.0 <= chi_min < chi_max < 1.0):
            raise ValueError("Require 0 <= chi_min < chi_max < 1")

        self.n_max = int(n_max)
        self.chi_min = float(chi_min)
        self.chi_max = float(chi_max)
        self.chi_grid = np.linspace(self.chi_min, self.chi_max, int(n_grid), dtype=float)

        self._re = np.zeros((self.n_max + 1, len(self.chi_grid)), dtype=float)
        self._im = np.zeros((self.n_max + 1, len(self.chi_grid)), dtype=float)
        self._build_tables()

    def _build_tables(self) -> None:
        # 预计算 (n, chi) 网格上的复频率，训练/推断阶段只做插值，减少开销。
        for n in range(self.n_max + 1):
            seq = qnm.modes_cache(s=-2, l=2, m=2, n=n)
            for i, chi in enumerate(self.chi_grid):
                omega_geom = complex(seq(a=float(chi))[0])
                self._re[n, i] = float(np.real(omega_geom))
                self._im[n, i] = float(np.imag(omega_geom))

    def _omega_geom_22n(self, chi_f: float, n: int) -> complex:
        if n < 0 or n > self.n_max:
            raise ValueError(f"n={n} out of range [0, {self.n_max}]")
        chi = float(np.clip(chi_f, self.chi_min, self.chi_max))
        re = float(np.interp(chi, self.chi_grid, self._re[n]))
        im = float(np.interp(chi, self.chi_grid, self._im[n]))
        return complex(re, im)

    def omega_22n(self, mf_msun: float, chi_f: float, n: int) -> QNMOmega:
        omega_geom = self._omega_geom_22n(chi_f=chi_f, n=n)
        # 几何单位频率 omega_geom -> SI 角频率 omega_si:
        # omega_si = omega_geom / (Mf * MTSUN_SI)
        m_sec = float(mf_msun) * MTSUN_SI
        omega_si = omega_geom / m_sec
        return QNMOmega(omega_rad_s=omega_si, omega_geom=omega_geom)
