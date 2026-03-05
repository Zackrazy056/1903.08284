from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ringdown_eq1 import ringdown_plus_eq1


@dataclass
class _OmegaObj:
    omega_rad_s: complex


class _DummyQNM:
    def omega_22n(self, mf_msun: float, chi_f: float, n: int) -> _OmegaObj:
        _ = (mf_msun, chi_f)
        # Keep all modes with same damped oscillation for deterministic test.
        return _OmegaObj(omega_rad_s=complex(1000.0 + 20.0 * n, -120.0))


def test_ringdown_bias_zero_matches_original_shape() -> None:
    t = np.linspace(0.0, 0.02, 100)
    amps = np.array([1.0e-21, 0.3e-21])
    phs = np.array([0.0, 1.2])
    qnm = _DummyQNM()

    h0 = ringdown_plus_eq1(t, 68.5, 0.69, amps, phs, qnm_interp=qnm, bias=0.0)
    h1 = ringdown_plus_eq1(t, 68.5, 0.69, amps, phs, qnm_interp=qnm)
    assert np.allclose(h0, h1)


def test_ringdown_bias_adds_constant_offset() -> None:
    t = np.linspace(0.0, 0.02, 100)
    amps = np.array([1.0e-21, 0.3e-21])
    phs = np.array([0.5, -0.2])
    qnm = _DummyQNM()
    bias = 2.5e-22

    h_base = ringdown_plus_eq1(t, 68.5, 0.69, amps, phs, qnm_interp=qnm, bias=0.0)
    h_bias = ringdown_plus_eq1(t, 68.5, 0.69, amps, phs, qnm_interp=qnm, bias=bias)
    assert np.allclose(h_bias - h_base, bias)

