from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import qnm

from ringdown.compare import align_time_and_phase_by_window, interp_complex, window_waveform
from ringdown.fit import build_design_matrix
from ringdown.frequencies import kerr_qnm_omegas_22n
from ringdown.preprocess import align_to_peak, build_start_time_grid
from ringdown.scan import fit_at_start_time
from ringdown.sxs_io import load_sxs_waveform22
from ringdown.types import Waveform22


@dataclass(frozen=True)
class AuditThresholds:
    max_dt_jitter_ratio_warn: float
    min_rank_coverage_pass: float
    cond_warn: float
    coeff_growth_warn: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sxs-id", type=str, default="SXS:BBH:0305v2.0")
    parser.add_argument("--lev", type=int, default=6)
    parser.add_argument("--sxs-location", type=str, default=None)
    parser.add_argument("--reference-lev", type=int, default=None)
    parser.add_argument("--no-download", action="store_true")

    parser.add_argument("--mf", type=float, default=None)
    parser.add_argument("--chif", type=float, default=None)

    parser.add_argument("--t0-min", type=float, default=-25.0)
    parser.add_argument("--t0-max", type=float, default=60.0)
    parser.add_argument("--t0-step", type=float, default=1.0)
    parser.add_argument("--t-end", type=float, default=90.0)
    parser.add_argument("--n-max", type=int, default=7)
    parser.add_argument("--lstsq-rcond", type=float, default=None)
    parser.add_argument(
        "--no-constant-offset",
        action="store_true",
        help="Disable default complex constant-offset basis term b in baseline fitting.",
    )
    parser.add_argument(
        "--diagnostic-constant-b",
        action="store_true",
        help="Enable diagnostic branch: add optional complex constant term b to model basis.",
    )
    parser.add_argument(
        "--plot-only-diagnostic-constant-b",
        action="store_true",
        help="When diagnostic branch is enabled, plot only mismatch curves with (+b).",
    )

    parser.add_argument("--max-dt-jitter-ratio-warn", type=float, default=1e-4)
    parser.add_argument("--min-rank-coverage-pass", type=float, default=0.95)
    parser.add_argument("--cond-warn", type=float, default=1e6)
    parser.add_argument("--coeff-growth-warn", type=float, default=1e3)

    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("results/fig1_contract_audit"),
    )
    return parser.parse_args()


def _resolve_location(args: argparse.Namespace) -> str:
    if args.sxs_location is not None:
        return args.sxs_location
    return f"{args.sxs_id}/Lev{args.lev}"


def _resolve_reference_location(args: argparse.Namespace) -> str | None:
    if args.reference_lev is None:
        return None
    return f"{args.sxs_id}/Lev{args.reference_lev}"


def _output_paths(prefix: Path) -> tuple[Path, Path, Path]:
    base = prefix.with_suffix("") if prefix.suffix else prefix
    return base.with_suffix(".json"), base.with_suffix(".md"), base.with_suffix(".png")


def _safe_git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return out.stdout.strip() or None


def _compute_mismatch_from_inner(
    h_nr: np.ndarray, h_model: np.ndarray, t: np.ndarray, *, method: str
) -> float:
    if method == "trapezoid":
        inner = lambda x, y: np.trapezoid(x * np.conjugate(y), t)
    elif method == "left_riemann":
        dt = np.diff(t)

        def inner(x: np.ndarray, y: np.ndarray) -> complex:
            return np.sum((x[:-1] * np.conjugate(y[:-1])) * dt)
    else:
        raise ValueError(f"unsupported integration method: {method}")

    num = inner(h_nr, h_model)
    den = np.sqrt(inner(h_nr, h_nr) * inner(h_model, h_model))
    if np.abs(den) == 0:
        return float("nan")
    return float(1.0 - np.real(num / den))


def _load_waveform(location: str, no_download: bool) -> tuple[Waveform22, float | None, float | None]:
    wf, info = load_sxs_waveform22(location=location, download=not no_download)
    return wf, info.remnant_mass, info.remnant_chif_z


def _fit_with_constant_b(
    t: np.ndarray,
    h: np.ndarray,
    omegas: np.ndarray,
    t0: float,
    *,
    lstsq_rcond: float | None,
) -> dict[str, float]:
    a_base = build_design_matrix(t=t, omegas=omegas, t0=t0)
    a = np.column_stack([a_base, np.ones((t.size, 1), dtype=complex)])
    coeffs, residuals, rank, svals = np.linalg.lstsq(a, h, rcond=lstsq_rcond)
    model = a @ coeffs
    mm = _compute_mismatch_from_inner(h, model, t, method="trapezoid")
    h_norm = float(np.linalg.norm(h))
    b = complex(coeffs[-1])

    if residuals.size > 0:
        residual_norm = float(np.sqrt(np.real(residuals[0])))
    else:
        residual_norm = float(np.linalg.norm(h - model))
    if svals.size == 0 or svals[-1] <= 0:
        cond = float("inf")
    else:
        cond = float(np.abs(svals[0] / svals[-1]))

    return {
        "mismatch": float(mm),
        "residual_norm": residual_norm,
        "rank": float(rank),
        "condition_number": cond,
        "b_abs": float(np.abs(b)),
        "b_over_signal_l2": float(np.abs(b) / h_norm) if h_norm > 0 else float("nan"),
    }


def _rank_coverage(points: list[dict[str, Any]], mode_count: int) -> float:
    if not points:
        return 0.0
    hits = sum(1 for p in points if int(p["rank"]) == mode_count)
    return hits / len(points)


def _compute_reference_floor(
    wf_main: Waveform22,
    wf_ref: Waveform22,
    *,
    t0: float,
    t_end: float,
    align_window: float = 20.0,
) -> dict[str, float]:
    t_main, h_main = window_waveform(wf_main, t0=t0, t_end=t_end)
    align_end = min(t_end, t0 + align_window)
    t_ref_aligned, h_ref_aligned, dt_best, phase_best = align_time_and_phase_by_window(
        t_reference=t_main,
        h_reference=h_main,
        t_target=wf_ref.t,
        h_target=wf_ref.h,
        t_start=t0,
        t_end=align_end,
    )

    valid = (t_main >= t_ref_aligned[0]) & (t_main <= t_ref_aligned[-1])
    t_eval = t_main[valid]
    h_main_eval = h_main[valid]
    h_ref_eval = interp_complex(t_ref_aligned, h_ref_aligned, t_eval)

    rel_noise_l2 = float(np.linalg.norm(h_main_eval - h_ref_eval) / np.linalg.norm(h_main_eval))
    floor_mm = _compute_mismatch_from_inner(h_main_eval, h_ref_eval, t_eval, method="trapezoid")
    return {
        "ref_best_dt": float(dt_best),
        "ref_best_phase": float(phase_best),
        "relative_noise_l2": rel_noise_l2,
        "mismatch_floor": float(floor_mm),
        "n_samples": int(t_eval.size),
    }


def _build_markdown(report: dict[str, Any]) -> str:
    cfg = report["config"]
    step1 = report["step1_window_contract"]
    verdicts = report["verdicts"]
    summary = report["fig1_curve_summary"]
    diagnostic_enabled = bool(cfg.get("diagnostic_constant_b", False))

    lines: list[str] = []
    lines.append(f"# Fig.1 Contract Audit ({report['run_metadata']['timestamp_utc'][:10]})")
    lines.append("")
    lines.append("## Scope")
    lines.append(
        f"- Location: `{cfg['location']}`"
    )
    lines.append(
        f"- Fixed remnant: `mf={cfg['fixed_remnant']['mf']:.12f}`, `chif={cfg['fixed_remnant']['chif']:.12f}`"
    )
    lines.append(
        f"- Scan: `t0 in [{cfg['t0_scan']['min']:.1f}, {cfg['t0_scan']['max']:.1f}]` step `{cfg['t0_scan']['step']:.1f}`, `N<= {cfg['n_max']}`"
    )
    lines.append(f"- Integration: `[t0, t_peak+{cfg['t_end']:.1f}M]`")
    lines.append("")
    lines.append("## Verdicts")
    for v in verdicts:
        lines.append(f"- {v['status']}: {v['name']} - {v['message']}")
    lines.append("")
    lines.append("## Key Checks")
    lines.append(
        f"- Peak alignment: `t_peak_original={step1['t_peak_original_M']:.6f}M`, `aligned_peak_time={step1['aligned_peak_time_M']:.6f}M`"
    )
    lines.append(
        f"- dt stats: `min={step1['dt_min_M']:.9f}`, `max={step1['dt_max_M']:.9f}`, "
        f"`mean={step1['dt_mean_M']:.9f}`, `std={step1['dt_std_M']:.9e}`"
    )
    lines.append("")
    lines.append("## Fig.1 Summary")
    lines.append("| N | best_t0 [M] | best_mm | mm@60M | turnover_ratio | rank_coverage | max_cond |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary:
        lines.append(
            f"| {row['N']} | {row['best_t0_M']:.1f} | {row['best_mismatch']:.3e} | "
            f"{row['mismatch_at_t0_max']:.3e} | {row['turnover_ratio']:.1f} | "
            f"{row['rank_coverage']:.3f} | {row['max_condition_number']:.3e} |"
        )
    if diagnostic_enabled:
        lines.append("")
        lines.append("## Diagnostic Constant-b Summary")
        lines.append("| N | best_mm(+b) | mm@60M(+b) | median b_abs | median improvement(>=20M) |")
        lines.append("|---:|---:|---:|---:|---:|")
        for row in summary:
            lines.append(
                f"| {row['N']} | {row.get('diag_best_mismatch_with_b', float('nan')):.3e} | "
                f"{row.get('diag_mismatch_at_t0_max_with_b', float('nan')):.3e} | "
                f"{row.get('diag_median_b_abs', float('nan')):.3e} | "
                f"{row.get('diag_median_improvement_factor_t0_ge20', float('nan')):.2e} |"
            )
    lines.append("")
    lines.append("## Note")
    lines.append(
        "- `N=0` at `~+47M` is not a hard Fig.1 acceptance threshold; Fig.1 checks trend consistency under fixed-remnant mismatch scan."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.plot_only_diagnostic_constant_b and not args.diagnostic_constant_b:
        raise ValueError("--plot-only-diagnostic-constant-b requires --diagnostic-constant-b")
    thresholds = AuditThresholds(
        max_dt_jitter_ratio_warn=args.max_dt_jitter_ratio_warn,
        min_rank_coverage_pass=args.min_rank_coverage_pass,
        cond_warn=args.cond_warn,
        coeff_growth_warn=args.coeff_growth_warn,
    )

    location = _resolve_location(args)
    ref_location = _resolve_reference_location(args)

    wf_raw, mf_data, chif_data = _load_waveform(location, no_download=args.no_download)
    wf_main, t_peak_original = align_to_peak(wf_raw)
    wf_ref = None
    if ref_location is not None:
        wf_ref_raw, _, _ = _load_waveform(ref_location, no_download=args.no_download)
        wf_ref, _ = align_to_peak(wf_ref_raw)

    mf = args.mf if args.mf is not None else mf_data
    chif = args.chif if args.chif is not None else chif_data
    if mf is None or chif is None:
        raise ValueError("mf/chif not available from input. Provide --mf and --chif.")

    t0_grid = build_start_time_grid(
        t_peak=0.0,
        m_total=1.0,
        rel_start_m=args.t0_min,
        rel_end_m=args.t0_max,
        step_m=args.t0_step,
    )
    t0_max = float(t0_grid[-1])

    curve_records: list[dict[str, Any]] = []
    detail_records: dict[str, list[dict[str, Any]]] = {}
    omega_rows: list[dict[str, float]] = []

    for n in range(args.n_max + 1):
        mode = qnm.modes_cache(s=-2, l=2, m=2, n=n)
        w_bar, _, _ = mode(a=float(chif))
        w = kerr_qnm_omegas_22n(mf=float(mf), chif=float(chif), n_max=n)[-1]
        omega_rows.append(
            {
                "n": n,
                "Re_omega_1_over_M": float(np.real(w)),
                "minus_Im_omega_1_over_M": float(-np.imag(w)),
                "Re_Mf_omega_qnm": float(np.real(w_bar)),
                "minus_Im_Mf_omega_qnm": float(-np.imag(w_bar)),
                "scaling_abs_error": float(np.abs((float(mf) * w) - w_bar)),
            }
        )
    for n in range(args.n_max + 1):
        omegas = kerr_qnm_omegas_22n(mf=float(mf), chif=float(chif), n_max=n)
        rows: list[dict[str, Any]] = []
        for t0 in t0_grid:
            t_win, h_win = window_waveform(wf_main, t0=float(t0), t_end=args.t_end)
            if t_win.size <= omegas.size:
                continue
            try:
                fit_result, lin = fit_at_start_time(
                    wf=wf_main,
                    omegas=omegas,
                    t0=float(t0),
                    t_end=args.t_end,
                    lstsq_rcond=args.lstsq_rcond,
                    include_constant_offset=not args.no_constant_offset,
                    max_condition_number=None,
                    max_overtone_to_fund_ratio=None,
                    min_signal_norm=0.0,
                )
            except ValueError:
                continue

            h_norm = float(np.linalg.norm(h_win))
            rel_res = float(np.linalg.norm(h_win - lin.model) / h_norm) if h_norm > 0 else float("nan")
            coeff_norm = float(np.linalg.norm(lin.coeffs))
            coeff_to_signal = coeff_norm / h_norm if h_norm > 0 else float("nan")
            svals = lin.singular_values
            row: dict[str, Any] = {
                "t0_M": float(t0),
                "mismatch": float(fit_result.mismatch),
                "residual_norm": float(fit_result.residual_norm),
                "relative_residual_l2": rel_res,
                "rank": int(lin.rank),
                "condition_number": float(lin.condition_number),
                "sigma_max": float(svals[0]) if svals.size else float("nan"),
                "sigma_min": float(svals[-1]) if svals.size else float("nan"),
                "coeff_l2": coeff_norm,
                "coeff_to_signal_l2": float(coeff_to_signal),
                "n_samples": int(t_win.size),
            }

            if args.diagnostic_constant_b:
                diag = _fit_with_constant_b(
                    t=t_win,
                    h=h_win,
                    omegas=omegas,
                    t0=float(t0),
                    lstsq_rcond=args.lstsq_rcond,
                )
                diag_mm = float(diag["mismatch"])
                row.update(
                    {
                        "diag_mismatch_with_b": diag_mm,
                        "diag_residual_norm_with_b": float(diag["residual_norm"]),
                        "diag_rank_with_b": int(diag["rank"]),
                        "diag_condition_number_with_b": float(diag["condition_number"]),
                        "diag_b_abs": float(diag["b_abs"]),
                        "diag_b_over_signal_l2": float(diag["b_over_signal_l2"]),
                        "diag_improvement_factor": (
                            float(fit_result.mismatch / diag_mm) if diag_mm > 0 else float("inf")
                        ),
                    }
                )

            rows.append(row)

        if not rows:
            continue

        rows = sorted(rows, key=lambda x: x["t0_M"])
        detail_records[f"N{n}"] = rows

        mm = np.array([r["mismatch"] for r in rows], dtype=float)
        t0 = np.array([r["t0_M"] for r in rows], dtype=float)
        cond = np.array([r["condition_number"] for r in rows], dtype=float)
        coeff_ratio = np.array([r["coeff_to_signal_l2"] for r in rows], dtype=float)

        best_i = int(np.argmin(mm))
        idx_at_max = int(np.argmin(np.abs(t0 - t0_max)))
        idx_at_zero = int(np.argmin(np.abs(t0 - 0.0)))
        coeff_growth = float(
            coeff_ratio[idx_at_max] / coeff_ratio[idx_at_zero]
            if np.isfinite(coeff_ratio[idx_at_zero]) and coeff_ratio[idx_at_zero] > 0
            else float("nan")
        )

        summary_row: dict[str, Any] = {
            "N": n,
            "valid_points": int(len(rows)),
            "best_t0_M": float(t0[best_i]),
            "best_mismatch": float(mm[best_i]),
            "mismatch_at_t0_max": float(mm[idx_at_max]),
            "turnover_ratio": float(mm[idx_at_max] / mm[best_i]),
            "rank_coverage": float(
                _rank_coverage(rows, n + 1 + (0 if args.no_constant_offset else 1))
            ),
            "max_condition_number": float(np.nanmax(cond)),
            "coeff_growth_t0max_over_t00": float(coeff_growth),
        }
        if args.diagnostic_constant_b:
            diag_mm = np.array([r["diag_mismatch_with_b"] for r in rows], dtype=float)
            diag_b_abs = np.array([r["diag_b_abs"] for r in rows], dtype=float)
            diag_improve = np.array([r["diag_improvement_factor"] for r in rows], dtype=float)
            late = t0 >= 20.0
            summary_row.update(
                {
                    "diag_best_mismatch_with_b": float(np.nanmin(diag_mm)),
                    "diag_mismatch_at_t0_max_with_b": float(diag_mm[idx_at_max]),
                    "diag_turnover_ratio_with_b": float(
                        diag_mm[idx_at_max] / np.nanmin(diag_mm) if np.nanmin(diag_mm) > 0 else float("nan")
                    ),
                    "diag_median_b_abs": float(np.nanmedian(diag_b_abs)),
                    "diag_median_improvement_factor_t0_ge20": float(np.nanmedian(diag_improve[late]))
                    if np.any(late)
                    else float(np.nanmedian(diag_improve)),
                }
            )

        curve_records.append(summary_row)

    if not curve_records:
        raise RuntimeError("No valid fits produced; check waveform and scan config.")

    # Step-1 dt stats in canonical fig1 window [0, T].
    t_win0, _ = window_waveform(wf_main, t0=0.0, t_end=args.t_end)
    dt = np.diff(t_win0)
    dt_mean = float(np.mean(dt))
    dt_std = float(np.std(dt))
    dt_ratio = float(dt_std / dt_mean) if dt_mean > 0 else float("inf")

    # Integration cross-check at t0=0 for N in {0,3,Nmax}
    cross_ns = sorted({0, min(3, args.n_max), args.n_max})
    integration_rows: list[dict[str, float]] = []
    for n in cross_ns:
        key = f"N{n}"
        if key not in detail_records:
            continue
        rows = detail_records[key]
        row0 = min(rows, key=lambda x: abs(x["t0_M"] - 0.0))
        t0 = float(row0["t0_M"])
        omegas = kerr_qnm_omegas_22n(mf=float(mf), chif=float(chif), n_max=n)
        _, lin = fit_at_start_time(
            wf=wf_main,
            omegas=omegas,
            t0=t0,
            t_end=args.t_end,
            lstsq_rcond=args.lstsq_rcond,
            include_constant_offset=not args.no_constant_offset,
            max_condition_number=None,
            max_overtone_to_fund_ratio=None,
            min_signal_norm=0.0,
        )
        t_eval, h_eval = window_waveform(wf_main, t0=t0, t_end=args.t_end)
        mm_trap = _compute_mismatch_from_inner(h_eval, lin.model, t_eval, method="trapezoid")
        mm_riem = _compute_mismatch_from_inner(h_eval, lin.model, t_eval, method="left_riemann")
        integration_rows.append(
            {
                "N": n,
                "t0_M": t0,
                "mismatch_trapezoid": float(mm_trap),
                "mismatch_left_riemann": float(mm_riem),
                "abs_diff": float(abs(mm_trap - mm_riem)),
            }
        )

    reference_rows: list[dict[str, Any]] = []
    if wf_ref is not None:
        for c in curve_records:
            n = int(c["N"])
            best_t0 = float(c["best_t0_M"])
            floor = _compute_reference_floor(wf_main, wf_ref, t0=best_t0, t_end=args.t_end)
            reference_rows.append(
                {
                    "N": n,
                    "best_t0_M": best_t0,
                    "best_mismatch": float(c["best_mismatch"]),
                    **floor,
                    "best_mm_over_floor_mm": float(c["best_mismatch"] / floor["mismatch_floor"])
                    if floor["mismatch_floor"] > 0
                    else float("nan"),
                }
            )

    verdicts: list[dict[str, str]] = []
    verdicts.append(
        {
            "name": "Window contract",
            "status": "PASS"
            if np.isclose(float(t0_grid[0]), args.t0_min)
            and np.isclose(float(t0_grid[-1]), args.t0_max)
            and args.t_end > 0
            else "FAIL",
            "message": f"t0=[{float(t0_grid[0]):.1f},{float(t0_grid[-1]):.1f}], T=t_peak+{args.t_end:.1f}M",
        }
    )
    verdicts.append(
        {
            "name": "dt uniformity",
            "status": "PASS" if dt_ratio <= thresholds.max_dt_jitter_ratio_warn else "WARN",
            "message": f"dt_std/dt_mean={dt_ratio:.3e}, threshold={thresholds.max_dt_jitter_ratio_warn:.1e}",
        }
    )

    min_rank_cov = min(float(c["rank_coverage"]) for c in curve_records)
    verdicts.append(
        {
            "name": "Rank coverage",
            "status": "PASS" if min_rank_cov >= thresholds.min_rank_coverage_pass else "WARN",
            "message": f"min rank coverage={min_rank_cov:.3f}, threshold={thresholds.min_rank_coverage_pass:.3f}",
        }
    )

    max_cond = max(float(c["max_condition_number"]) for c in curve_records)
    verdicts.append(
        {
            "name": "Condition number",
            "status": "PASS" if max_cond <= thresholds.cond_warn else "WARN",
            "message": f"max cond(A)={max_cond:.3e}, warn>{thresholds.cond_warn:.1e}",
        }
    )

    max_coeff_growth = max(float(c["coeff_growth_t0max_over_t00"]) for c in curve_records)
    verdicts.append(
        {
            "name": "Coefficient growth",
            "status": "PASS" if max_coeff_growth <= thresholds.coeff_growth_warn else "WARN",
            "message": (
                f"max ||C|| growth(t0_max/t0=0)={max_coeff_growth:.3e}, "
                f"warn>{thresholds.coeff_growth_warn:.1e}"
            ),
        }
    )

    if args.diagnostic_constant_b:
        improve_values = [
            float(c.get("diag_median_improvement_factor_t0_ge20", float("nan"))) for c in curve_records
        ]
        max_improve = float(np.nanmax(np.asarray(improve_values, dtype=float)))
        verdicts.append(
            {
                "name": "Diagnostic constant-b",
                "status": "WARN" if max_improve > 10.0 else "PASS",
                "message": (
                    f"max median improvement(>=20M)={max_improve:.2e}; "
                    "large improvement suggests unresolved constant bias."
                ),
            }
        )

    if reference_rows:
        min_floor_ratio = min(float(r["best_mm_over_floor_mm"]) for r in reference_rows)
        verdicts.append(
            {
                "name": "Multi-resolution floor",
                "status": "WARN" if min_floor_ratio < 1.0 else "PASS",
                "message": f"min(best_mm/floor_mm)={min_floor_ratio:.3e}",
            }
        )

    report: dict[str, Any] = {
        "run_metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit_short": _safe_git_commit(),
        },
        "config": {
            "location": location,
            "reference_location": ref_location,
            "fixed_remnant": {"mf": float(mf), "chif": float(chif)},
            "t0_scan": {"min": args.t0_min, "max": args.t0_max, "step": args.t0_step},
            "t_end": args.t_end,
            "n_max": args.n_max,
            "lstsq_rcond": args.lstsq_rcond,
            "include_constant_offset": bool(not args.no_constant_offset),
            "diagnostic_constant_b": bool(args.diagnostic_constant_b),
            "plot_only_diagnostic_constant_b": bool(args.plot_only_diagnostic_constant_b),
            "thresholds": asdict(thresholds),
        },
        "step1_window_contract": {
            "t_peak_original_M": float(t_peak_original),
            "aligned_peak_time_M": float(wf_main.t[np.argmax(np.abs(wf_main.h))]),
            "dt_min_M": float(np.min(dt)),
            "dt_max_M": float(np.max(dt)),
            "dt_mean_M": dt_mean,
            "dt_std_M": dt_std,
            "dt_std_over_mean": dt_ratio,
            "samples_in_t0_0_to_tend": int(t_win0.size),
        },
        "step2_omega_contract": omega_rows,
        "step3_fit_detail_by_N": detail_records,
        "step4_integral_crosscheck": integration_rows,
        "fig1_curve_summary": sorted(curve_records, key=lambda x: int(x["N"])),
        "reference_floor_summary": reference_rows,
        "verdicts": verdicts,
        "notes": [
            "Fig.1 acceptance is trend consistency under fixed remnant mismatch scan.",
            "N=0 near +47M is not a hard acceptance threshold for this figure.",
        ],
    }

    json_path, md_path, png_path = _output_paths(args.out_prefix)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_build_markdown(report), encoding="utf-8")

    # Plot: mismatch and cond(A) vs t0.
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for n in range(args.n_max + 1):
        key = f"N{n}"
        if key not in detail_records:
            continue
        rows = detail_records[key]
        t0 = np.array([r["t0_M"] for r in rows], dtype=float)
        mm = np.array([r["mismatch"] for r in rows], dtype=float)
        cond = np.array([r["condition_number"] for r in rows], dtype=float)
        if args.diagnostic_constant_b and args.plot_only_diagnostic_constant_b:
            mm_b = np.array([r["diag_mismatch_with_b"] for r in rows], dtype=float)
            axes[0].semilogy(t0, mm_b, label=f"N={n} (+b)")
        else:
            axes[0].semilogy(t0, mm, label=f"N={n}")
            if args.diagnostic_constant_b:
                mm_b = np.array([r["diag_mismatch_with_b"] for r in rows], dtype=float)
                axes[0].semilogy(t0, mm_b, ls="--", alpha=0.8, label=f"N={n} (+b)")
        axes[1].semilogy(t0, cond, label=f"N={n}")

    axes[0].set_ylabel("Mismatch")
    axes[0].grid(True, which="both", alpha=0.2)
    axes[0].legend(ncol=2, fontsize=9)
    axes[0].set_title(f"Fig.1 Contract Audit ({location})")

    axes[1].set_ylabel("cond(A)")
    axes[1].set_xlabel(r"$t_0 - t_{\mathrm{peak}} \ [M]$")
    axes[1].grid(True, which="both", alpha=0.2)
    axes[1].axhline(args.cond_warn, color="r", ls="--", lw=1.0, alpha=0.7, label="cond warn")
    axes[1].legend(ncol=3, fontsize=8)

    fig.tight_layout()
    fig.savefig(png_path, dpi=180)

    print(f"location={location}")
    print(f"reference_location={ref_location}")
    print(f"mf={float(mf):.12f}, chif={float(chif):.12f}")
    print(f"json={json_path}")
    print(f"md={md_path}")
    print(f"png={png_path}")
    for v in verdicts:
        print(f"[{v['status']}] {v['name']}: {v['message']}")


if __name__ == "__main__":
    main()
