from __future__ import annotations

from math import pi, sqrt

import numpy as np

from ringdown.conventions import finite_duration_from_uniform_time_samples
from ringdown.paper_fig10 import (
    FACE_ON_PLUS_FACTOR_22,
    detector_plus_from_mode22,
    face_on_plus_only_detector_channel_from_mode22,
    physical_strain_scale,
)


def test_face_on_plus_only_detector_channel_from_mode22_is_face_on_real_projection() -> None:
    h22 = np.array([1.0 + 2.0j, -3.0 + 4.0j], dtype=complex)
    projected = face_on_plus_only_detector_channel_from_mode22(h22)
    expected = np.real(h22) * sqrt(5.0 / (4.0 * pi))
    np.testing.assert_allclose(projected, expected)
    assert FACE_ON_PLUS_FACTOR_22 == sqrt(5.0 / (4.0 * pi))
    np.testing.assert_allclose(detector_plus_from_mode22(h22), expected)


def test_finite_duration_from_uniform_time_samples_returns_n_minus_1_dt() -> None:
    t_sec = np.array([0.0, 0.1, 0.2, 0.3])
    assert np.isclose(finite_duration_from_uniform_time_samples(t_sec), 0.3)


def test_physical_strain_scale_matches_mass_over_distance() -> None:
    scale_72_400 = physical_strain_scale(72.0, 400.0)
    scale_36_400 = physical_strain_scale(36.0, 400.0)
    scale_72_800 = physical_strain_scale(72.0, 800.0)
    assert scale_72_400 > 0.0
    assert np.isclose(scale_36_400, 0.5 * scale_72_400)
    assert np.isclose(scale_72_800, 0.5 * scale_72_400)
