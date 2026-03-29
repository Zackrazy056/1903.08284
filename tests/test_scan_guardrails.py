from __future__ import annotations

import numpy as np
import pytest

from ringdown.scan import PhysicsHeuristics, fit_at_start_time
from ringdown.types import Waveform22


def _build_large_overtone_waveform() -> tuple[Waveform22, np.ndarray]:
    t = np.linspace(0.0, 0.08, 400)
    omegas = np.array([340.0 + 35.0j, 520.0 + 80.0j], dtype=complex)
    coeffs = np.array([1.0, 2.0e4], dtype=complex)
    h = np.sum(coeffs[None, :] * np.exp(-1j * t[:, None] * omegas[None, :]), axis=1)
    return Waveform22(t=t, h=h, source="unit-test"), omegas


def test_fit_at_start_time_default_keeps_physics_heuristics_disabled() -> None:
    wf, omegas = _build_large_overtone_waveform()

    fit_result, lin = fit_at_start_time(
        wf=wf,
        omegas=omegas,
        t0=0.0,
        include_constant_offset=False,
    )

    assert fit_result.mismatch < 1e-10
    assert np.isclose(np.abs(lin.coeffs[1]) / np.abs(lin.coeffs[0]), 2.0e4, rtol=1e-6)


def test_fit_at_start_time_can_opt_in_to_physics_heuristics() -> None:
    wf, omegas = _build_large_overtone_waveform()

    with pytest.raises(ValueError, match="overtone amplitudes are unphysical"):
        fit_at_start_time(
            wf=wf,
            omegas=omegas,
            t0=0.0,
            include_constant_offset=False,
            physics_heuristics=PhysicsHeuristics(max_overtone_to_fund_ratio=1.0e4),
        )
