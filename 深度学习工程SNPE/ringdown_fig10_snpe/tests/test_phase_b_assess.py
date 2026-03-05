from __future__ import annotations

import numpy as np

from eval.phase_b_assess import AssessmentThresholds, assess_phase_b


def _thresholds() -> AssessmentThresholds:
    return AssessmentThresholds(
        truth_in_90_hpd_required=True,
        acceptance_mean_min=0.12,
        acceptance_min_per_walker_min=0.02,
        mf_edge_1pct_max=0.10,
        chi_edge_1pct_max=0.05,
        mf_iqr_fraction_of_prior_max=0.70,
        chi_iqr_fraction_of_prior_max=0.70,
    )


def test_assess_phase_b_passes_for_concentrated_truth_centered_samples() -> None:
    rng = np.random.default_rng(7)
    mf = rng.normal(68.5, 2.0, size=8000)
    chi = rng.normal(0.69, 0.03, size=8000)
    mf = np.clip(mf, 10.0, 100.0)
    chi = np.clip(chi, 0.0, 1.0)
    samples = np.column_stack([mf, chi])
    accept = rng.uniform(0.15, 0.25, size=48)

    out = assess_phase_b(
        flat_samples=samples,
        acceptance_fraction=accept,
        truth_mf=68.5,
        truth_chi=0.69,
        mf_range=(10.0, 100.0),
        chi_range=(0.0, 1.0),
        thresholds=_thresholds(),
    )
    assert out["pass"] is True
    assert out["checks"]["truth_in_90_hpd"] is True


def test_assess_phase_b_fails_for_boundary_stuck_samples() -> None:
    rng = np.random.default_rng(11)
    # Force edge occupancy violation near lower Mf bound and low acceptance.
    mf = 10.0 + rng.uniform(0.0, 0.5, size=5000)
    chi = rng.uniform(0.10, 0.95, size=5000)
    samples = np.column_stack([mf, chi])
    accept = np.full(48, 0.01)

    out = assess_phase_b(
        flat_samples=samples,
        acceptance_fraction=accept,
        truth_mf=68.5,
        truth_chi=0.69,
        mf_range=(10.0, 100.0),
        chi_range=(0.0, 1.0),
        thresholds=_thresholds(),
    )
    assert out["pass"] is False
    assert "mf_edge_1pct" in out["failed_checks"]
    assert "acceptance_mean" in out["failed_checks"]

