"""Ringdown reproduction package."""

from .compare import (
    align_time_and_phase_by_window,
    interp_complex,
    phase_align_to_reference_at_tref,
    window_waveform,
)
from .fit import LinearFitResult, build_design_matrix, solve_complex_lstsq
from .fd_likelihood import (
    FrequencyDomainRingdownLikelihood,
    aligo_zero_det_high_power_psd,
    continuous_ft_from_time_series,
    draw_colored_noise_rfft,
    one_sided_inner_product,
    optimal_snr,
    rfft_continuous,
)
from .frequencies import kerr_qnm_omega_lmn, kerr_qnm_omegas_22n, make_omega_provider_22
from .io import load_waveform_csv, load_waveform_npz
from .metrics import inner_product, mismatch, remnant_error_epsilon
from .preprocess import (
    align_to_peak,
    build_start_time_grid,
    crop_time,
    peak_time_from_strain,
    resample_uniform,
    shift_time,
)
from .scan import (
    RemnantGridResult,
    StartTimeFitResult,
    fit_at_start_time,
    grid_search_remnant,
    scan_start_times_fixed_omegas,
)
from .sxs_io import SXSRemnantInfo, load_sxs_waveform22
from .types import Waveform22

__all__ = [
    "LinearFitResult",
    "FrequencyDomainRingdownLikelihood",
    "RemnantGridResult",
    "SXSRemnantInfo",
    "StartTimeFitResult",
    "Waveform22",
    "align_time_and_phase_by_window",
    "align_to_peak",
    "build_design_matrix",
    "build_start_time_grid",
    "crop_time",
    "fit_at_start_time",
    "grid_search_remnant",
    "interp_complex",
    "inner_product",
    "kerr_qnm_omega_lmn",
    "kerr_qnm_omegas_22n",
    "load_waveform_csv",
    "load_waveform_npz",
    "load_sxs_waveform22",
    "make_omega_provider_22",
    "mismatch",
    "peak_time_from_strain",
    "phase_align_to_reference_at_tref",
    "remnant_error_epsilon",
    "resample_uniform",
    "scan_start_times_fixed_omegas",
    "shift_time",
    "solve_complex_lstsq",
    "aligo_zero_det_high_power_psd",
    "continuous_ft_from_time_series",
    "draw_colored_noise_rfft",
    "one_sided_inner_product",
    "optimal_snr",
    "rfft_continuous",
    "window_waveform",
]
