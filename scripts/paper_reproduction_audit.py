from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


TRUTH_MF = 68.546372
TRUTH_CHIF = 0.692085
TOTAL_MASS_MSUN = 72.0


@dataclass(frozen=True)
class Fig10RunSpec:
    key: str
    label: str
    diag_csv: Path
    samples_prefix: Path
    figure_path: Path
    base_sampler: str
    n3_samplers: tuple[str, str]


@dataclass(frozen=True)
class Fig10SamplerMetrics:
    sampler: str
    acceptance: float
    tau_mf: float
    tau_chif: float
    ess: float
    healthy_minimum: bool
    health_reason: str


@dataclass(frozen=True)
class Fig10NMetrics:
    n: int
    samplers: list[Fig10SamplerMetrics]
    sample_count: int
    mf_q16: float
    mf_q50: float
    mf_q84: float
    chif_q16: float
    chif_q50: float
    chif_q84: float
    delta_mf_q50: float
    delta_chif_q50: float
    epsilon72_q50: float
    truth_hpd_rank: float
    truth_in_90_hpd: bool
    truth_in_40_hpd: bool
    n3_consistency_mf_diff: float | None
    n3_consistency_chif_diff: float | None


@dataclass(frozen=True)
class Fig10RunAudit:
    spec: Fig10RunSpec
    per_n: list[Fig10NMetrics]
    health_minimum_pass: bool
    n3_best_by_truth_rank: bool
    n3_best_by_epsilon72_q50: bool
    n3_truth_in_top40: bool
    n0_not_in_top40: bool
    paper_parity_pass: bool
    summary: str


@dataclass(frozen=True)
class Phase5NMetrics:
    n: int
    count: int
    epsilon_median: float
    epsilon_mean: float
    epsilon_p90: float
    epsilon_max: float
    boundary_rate: float
    mean_grid_expansions: float


@dataclass(frozen=True)
class Phase5Audit:
    csv_path: Path
    per_n: list[Phase5NMetrics]
    complete_counts_pass: bool
    median_monotonic_pass: bool
    p90_monotonic_pass: bool
    boundary_free_pass: bool
    paper_parity_pass: bool
    summary: str


def parse_args() -> argparse.Namespace:
    today = datetime.now().strftime("%Y%m%d")
    p = argparse.ArgumentParser(
        description="Audit current paper-reproduction status from existing fig10 and phase5 artifacts."
    )
    p.add_argument(
        "--strict-diag-csv",
        type=Path,
        default=Path("results/fig10_emcee_full_strict_prod1_diag.csv"),
    )
    p.add_argument(
        "--strict-samples-prefix",
        type=Path,
        default=Path("results/fig10_emcee_full_strict_prod1_samples.npz"),
    )
    p.add_argument(
        "--strict-figure",
        type=Path,
        default=Path("results/fig10_emcee_full_strict_prod1.png"),
    )
    p.add_argument(
        "--legacy-diag-csv",
        type=Path,
        default=Path("results/fig10_kombine_emcee_full_prod2_w128_diag.csv"),
    )
    p.add_argument(
        "--legacy-samples-prefix",
        type=Path,
        default=Path("results/fig10_kombine_emcee_full_prod2_w128_samples.npz"),
    )
    p.add_argument(
        "--legacy-figure",
        type=Path,
        default=Path("results/fig10_kombine_emcee_full_prod2_w128.png"),
    )
    p.add_argument(
        "--phase5-csv",
        type=Path,
        default=Path("results/phase5_sweep_12_adaptive4.csv"),
    )
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(f"results/paper_reproduction_audit_{today}"),
    )
    p.add_argument("--hist-bins", type=int, default=90)
    p.add_argument("--mf-min", type=float, default=50.0)
    p.add_argument("--mf-max", type=float, default=100.0)
    p.add_argument("--chi-min", type=float, default=0.0)
    p.add_argument("--chi-max", type=float, default=1.0)
    p.add_argument("--min-acceptance", type=float, default=0.01)
    return p.parse_args()


def _sample_path(prefix: Path, n: int, sampler: str) -> Path:
    return prefix.with_name(prefix.stem + f"_N{n}_{sampler}.npz")


def _load_flat(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"sample file not found: {path}")
    return np.load(path, allow_pickle=False)["flat"]


def _parse_diag_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"diagnostics csv not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _get_diag_row(rows: list[dict[str, str]], n: int, sampler: str) -> dict[str, str]:
    for row in rows:
        if row.get("N", "").strip() == str(n) and row.get("sampler", "").strip().lower() == sampler.lower():
            return row
    raise KeyError(f"missing diagnostics row for N={n}, sampler={sampler}")


def _finite_float(row: dict[str, str], key: str) -> float:
    val = row.get(key, "")
    try:
        return float(val)
    except Exception as exc:
        raise ValueError(f"invalid float for {key}: {val!r}") from exc


def _health_reason(acceptance: float, tau_mf: float, tau_chif: float, min_acceptance: float) -> tuple[bool, str]:
    if not np.isfinite(acceptance):
        return False, "acceptance is not finite"
    if acceptance < min_acceptance:
        return False, f"acceptance={acceptance:.6f} < {min_acceptance:.6f}"
    if not np.isfinite(tau_mf) or not np.isfinite(tau_chif):
        return False, f"tau invalid (tau_mf={tau_mf}, tau_chif={tau_chif})"
    return True, ""


def _credible_threshold(hist2d: np.ndarray, prob: float) -> float:
    flat = np.asarray(hist2d, dtype=float).ravel()
    total = float(np.sum(flat))
    if total <= 0.0:
        return 0.0
    order = np.argsort(flat)[::-1]
    sorted_vals = flat[order]
    cdf = np.cumsum(sorted_vals) / total
    idx = int(np.searchsorted(cdf, prob, side="left"))
    idx = min(max(idx, 0), len(sorted_vals) - 1)
    return float(sorted_vals[idx])


def _truth_hpd_rank(
    hist2d: np.ndarray,
    truth_mf: float,
    truth_chif: float,
    mf_range: tuple[float, float],
    chi_range: tuple[float, float],
) -> float:
    h = np.asarray(hist2d, dtype=float)
    total = float(np.sum(h))
    if total <= 0.0:
        return 1.0
    mf_min, mf_max = mf_range
    chi_min, chi_max = chi_range
    if not (mf_min <= truth_mf <= mf_max and chi_min <= truth_chif <= chi_max):
        return 1.0
    n_chi, n_mf = h.shape
    mf_bin = int(np.floor((truth_mf - mf_min) / (mf_max - mf_min) * n_mf))
    chi_bin = int(np.floor((truth_chif - chi_min) / (chi_max - chi_min) * n_chi))
    mf_bin = min(max(mf_bin, 0), n_mf - 1)
    chi_bin = min(max(chi_bin, 0), n_chi - 1)
    truth_density = float(h[chi_bin, mf_bin])
    return float(np.sum(h[h >= truth_density])) / total


def _epsilon72_q50(mf_q50: float, chif_q50: float) -> float:
    delta_mf = float(mf_q50 - TRUTH_MF)
    delta_chif = float(chif_q50 - TRUTH_CHIF)
    return float(np.sqrt((delta_mf / TOTAL_MASS_MSUN) ** 2 + delta_chif**2))


def _build_fig10_run_spec(args: argparse.Namespace) -> list[Fig10RunSpec]:
    return [
        Fig10RunSpec(
            key="strict_emcee_v2",
            label="Strict emcee production run",
            diag_csv=args.strict_diag_csv,
            samples_prefix=args.strict_samples_prefix,
            figure_path=args.strict_figure,
            base_sampler="emcee",
            n3_samplers=("emcee", "emcee_alt"),
        ),
        Fig10RunSpec(
            key="legacy_kombine_v1",
            label="Legacy kombine/emcee publication run",
            diag_csv=args.legacy_diag_csv,
            samples_prefix=args.legacy_samples_prefix,
            figure_path=args.legacy_figure,
            base_sampler="kombine",
            n3_samplers=("kombine", "emcee"),
        ),
    ]


def _audit_fig10_run(spec: Fig10RunSpec, args: argparse.Namespace) -> Fig10RunAudit:
    rows = _parse_diag_rows(spec.diag_csv)
    per_n: list[Fig10NMetrics] = []

    for n in range(4):
        if n < 3:
            sampler_names = [spec.base_sampler]
        else:
            sampler_names = list(spec.n3_samplers)

        sampler_metrics: list[Fig10SamplerMetrics] = []
        sample_parts: list[np.ndarray] = []
        component_q50: list[np.ndarray] = []

        for sampler_name in sampler_names:
            row = _get_diag_row(rows, n, sampler_name)
            acceptance = _finite_float(row, "acceptance")
            tau_mf = _finite_float(row, "tau_mf")
            tau_chif = _finite_float(row, "tau_chif")
            ess = _finite_float(row, "ess")
            healthy, reason = _health_reason(acceptance, tau_mf, tau_chif, args.min_acceptance)
            sampler_metrics.append(
                Fig10SamplerMetrics(
                    sampler=sampler_name,
                    acceptance=acceptance,
                    tau_mf=tau_mf,
                    tau_chif=tau_chif,
                    ess=ess,
                    healthy_minimum=healthy,
                    health_reason=reason,
                )
            )

            flat = _load_flat(_sample_path(spec.samples_prefix, n, sampler_name))
            core = np.asarray(flat[:, :2], dtype=float)
            core = core[np.isfinite(core).all(axis=1)]
            sample_parts.append(core)
            component_q50.append(np.quantile(core, 0.5, axis=0))

        core = np.vstack(sample_parts)
        mf = core[:, 0]
        chi = core[:, 1]
        q16, q50, q84 = np.quantile(core, [0.16, 0.5, 0.84], axis=0)

        hist2d, _, _ = np.histogram2d(
            mf,
            chi,
            bins=args.hist_bins,
            range=[[args.mf_min, args.mf_max], [args.chi_min, args.chi_max]],
        )
        hist2d = hist2d.T
        truth_rank = _truth_hpd_rank(
            hist2d=hist2d,
            truth_mf=TRUTH_MF,
            truth_chif=TRUTH_CHIF,
            mf_range=(args.mf_min, args.mf_max),
            chi_range=(args.chi_min, args.chi_max),
        )
        n3_consistency_mf_diff = None
        n3_consistency_chif_diff = None
        if n == 3 and len(component_q50) == 2:
            n3_consistency_mf_diff = float(abs(component_q50[0][0] - component_q50[1][0]))
            n3_consistency_chif_diff = float(abs(component_q50[0][1] - component_q50[1][1]))

        per_n.append(
            Fig10NMetrics(
                n=n,
                samplers=sampler_metrics,
                sample_count=int(core.shape[0]),
                mf_q16=float(q16[0]),
                mf_q50=float(q50[0]),
                mf_q84=float(q84[0]),
                chif_q16=float(q16[1]),
                chif_q50=float(q50[1]),
                chif_q84=float(q84[1]),
                delta_mf_q50=float(q50[0] - TRUTH_MF),
                delta_chif_q50=float(q50[1] - TRUTH_CHIF),
                epsilon72_q50=_epsilon72_q50(float(q50[0]), float(q50[1])),
                truth_hpd_rank=float(truth_rank),
                truth_in_90_hpd=bool(truth_rank <= 0.9),
                truth_in_40_hpd=bool(truth_rank <= 0.4),
                n3_consistency_mf_diff=n3_consistency_mf_diff,
                n3_consistency_chif_diff=n3_consistency_chif_diff,
            )
        )

    health_minimum_pass = all(
        sampler.healthy_minimum
        for metric in per_n
        for sampler in metric.samplers
    )
    best_rank_n = min(per_n, key=lambda item: item.truth_hpd_rank).n
    best_eps_n = min(per_n, key=lambda item: item.epsilon72_q50).n
    n3_metric = next(metric for metric in per_n if metric.n == 3)
    n0_metric = next(metric for metric in per_n if metric.n == 0)
    n3_best_by_truth_rank = best_rank_n == 3
    n3_best_by_epsilon72_q50 = best_eps_n == 3
    n3_truth_in_top40 = n3_metric.truth_in_40_hpd
    n0_not_in_top40 = not n0_metric.truth_in_40_hpd
    paper_parity_pass = (
        health_minimum_pass
        and n3_best_by_truth_rank
        and n3_best_by_epsilon72_q50
        and n3_truth_in_top40
        and n0_not_in_top40
    )

    if paper_parity_pass:
        summary = "Current run passes both minimum sampler health and the key Fig.10 paper-parity checks."
    elif health_minimum_pass:
        summary = "Sampler health is minimally acceptable, but the recovered posteriors still do not match the paper trend."
    else:
        summary = "This run should not be treated as final: sampler health already fails before paper-parity is even considered."

    return Fig10RunAudit(
        spec=spec,
        per_n=per_n,
        health_minimum_pass=health_minimum_pass,
        n3_best_by_truth_rank=n3_best_by_truth_rank,
        n3_best_by_epsilon72_q50=n3_best_by_epsilon72_q50,
        n3_truth_in_top40=n3_truth_in_top40,
        n0_not_in_top40=n0_not_in_top40,
        paper_parity_pass=paper_parity_pass,
        summary=summary,
    )


def _audit_phase5(csv_path: Path) -> Phase5Audit:
    if not csv_path.exists():
        raise FileNotFoundError(f"phase5 csv not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("phase5 csv has no data rows")

    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(int(row["n_overtones"]), []).append(row)

    per_n: list[Phase5NMetrics] = []
    for n in sorted(grouped):
        entries = grouped[n]
        eps = np.array([float(row["epsilon"]) for row in entries], dtype=float)
        boundary = np.array(
            [
                str(row["best_at_grid_boundary"]).strip().lower() in {"true", "1", "yes"}
                for row in entries
            ],
            dtype=float,
        )
        expansions = np.array([float(row["grid_expansions_used"]) for row in entries], dtype=float)
        per_n.append(
            Phase5NMetrics(
                n=n,
                count=int(eps.size),
                epsilon_median=float(np.median(eps)),
                epsilon_mean=float(np.mean(eps)),
                epsilon_p90=float(np.percentile(eps, 90)),
                epsilon_max=float(np.max(eps)),
                boundary_rate=float(np.mean(boundary)),
                mean_grid_expansions=float(np.mean(expansions)),
            )
        )

    counts = [item.count for item in per_n]
    medians = {item.n: item.epsilon_median for item in per_n}
    p90s = {item.n: item.epsilon_p90 for item in per_n}
    complete_counts_pass = len(set(counts)) == 1 and counts[0] > 0
    median_monotonic_pass = medians[0] > medians[3] > medians[7]
    p90_monotonic_pass = p90s[0] > p90s[3] > p90s[7]
    boundary_free_pass = all(item.boundary_rate == 0.0 for item in per_n)
    paper_parity_pass = complete_counts_pass and median_monotonic_pass and p90_monotonic_pass and boundary_free_pass

    if paper_parity_pass:
        summary = "Phase5 sweep cleanly reproduces the paper trend: epsilon drops sharply as overtones are added."
    else:
        summary = "Phase5 sweep is still missing at least one of the paper-trend checks."

    return Phase5Audit(
        csv_path=csv_path,
        per_n=per_n,
        complete_counts_pass=complete_counts_pass,
        median_monotonic_pass=median_monotonic_pass,
        p90_monotonic_pass=p90_monotonic_pass,
        boundary_free_pass=boundary_free_pass,
        paper_parity_pass=paper_parity_pass,
        summary=summary,
    )


def _bool_flag(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _build_markdown(fig10_runs: list[Fig10RunAudit], phase5: Phase5Audit, output_json: Path) -> str:
    strict_run = next(run for run in fig10_runs if run.spec.key == "strict_emcee_v2")
    legacy_run = next(run for run in fig10_runs if run.spec.key == "legacy_kombine_v1")

    fig11_scripts = sorted(Path("scripts").glob("*fig11*"))
    fig11_outputs = sorted(Path("results").glob("*fig11*"))
    fig11_missing = len(fig11_scripts) == 0 and len(fig11_outputs) == 0

    lines: list[str] = []
    lines.append(f"# Paper Reproduction Audit ({datetime.now().strftime('%Y-%m-%d')})")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(f"- Fig.10 strict emcee rerun: {_bool_flag(strict_run.paper_parity_pass)}")
    lines.append(f"- Fig.10 legacy kombine/emcee publication run: {_bool_flag(legacy_run.paper_parity_pass)}")
    lines.append(f"- Phase5 multi-simulation epsilon sweep: {_bool_flag(phase5.paper_parity_pass)}")
    lines.append(f"- Fig.11 implementation status: {'MISSING' if fig11_missing else 'PRESENT'}")
    lines.append("")
    lines.append("Current reading:")
    lines.append("- The project now has a strong non-Bayesian / fixed-remnant / grid-search reproduction backbone.")
    lines.append("- The current Fig.10 Bayesian artifacts are still not paper-faithful enough to be called a successful reproduction.")
    lines.append("- The Phase5 sweep is currently the strongest paper-parity result among the unfinished items.")
    lines.append("")

    lines.append("## Fig.10 Re-Audit")
    lines.append(
        f"- Truth point used for audit: `Mf={TRUTH_MF:.6f} Msun`, `chif={TRUTH_CHIF:.6f}`, "
        f"`M_total={TOTAL_MASS_MSUN:.1f} Msun`."
    )
    lines.append(
        "- Paper target: `N=0` should be clearly offset near peak, while `N=3` should be the best case and include the truth within the top `40%` credible region."
    )
    lines.append("")

    for run in fig10_runs:
        lines.append(f"### {run.spec.label}")
        lines.append(f"- Figure: `{run.spec.figure_path}`")
        lines.append(f"- Diagnostics: `{run.spec.diag_csv}`")
        lines.append(f"- Samples prefix: `{run.spec.samples_prefix}`")
        lines.append(f"- Minimum sampler health: {_bool_flag(run.health_minimum_pass)}")
        lines.append(f"- Paper parity: {_bool_flag(run.paper_parity_pass)}")
        lines.append(f"- Summary: {run.summary}")
        lines.append("")
        lines.append("| N | sample source | eps72(q50) | Mf q50 | chif q50 | truth rank | in 90% HPD | in 40% HPD | health |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---|")
        for metric in run.per_n:
            sampler_names = "+".join(item.sampler for item in metric.samplers)
            health = "; ".join(
                f"{item.sampler}:{'ok' if item.healthy_minimum else item.health_reason}"
                for item in metric.samplers
            )
            lines.append(
                f"| {metric.n} | {sampler_names} | {metric.epsilon72_q50:.4f} | "
                f"{metric.mf_q50:.3f} | {metric.chif_q50:.3f} | {metric.truth_hpd_rank:.3f} | "
                f"{int(metric.truth_in_90_hpd)} | {int(metric.truth_in_40_hpd)} | {health} |"
            )
            if metric.n == 3 and metric.n3_consistency_mf_diff is not None:
                lines.append(
                    f"N=3 consistency deltas: `|dMf_q50|={metric.n3_consistency_mf_diff:.4f}`, "
                    f"`|dchif_q50|={metric.n3_consistency_chif_diff:.4f}`."
                )
        lines.append("")
        lines.append(
            f"Verdicts: health={_bool_flag(run.health_minimum_pass)}, "
            f"N3_best_by_truth_rank={_bool_flag(run.n3_best_by_truth_rank)}, "
            f"N3_best_by_eps72={_bool_flag(run.n3_best_by_epsilon72_q50)}, "
            f"N3_truth_top40={_bool_flag(run.n3_truth_in_top40)}, "
            f"N0_not_top40={_bool_flag(run.n0_not_in_top40)}"
        )
        lines.append("")

    lines.append("### Fig.10 Interpretation")
    lines.append(
        "- The strict emcee rerun is healthier numerically than the legacy kombine result, but it fails the physics-facing parity test: none of the `N=0..3` posteriors place the truth inside the 90% region, and `N=3` is not the best case."
    )
    lines.append(
        "- The legacy kombine/emcee publication run looks visually plausible, but it fails a more basic audit: `kombine` rows have effectively zero acceptance and undefined autocorrelation, and it also breaks the paper trend because `N=0` already places the truth deep inside the posterior."
    )
    lines.append(
        "- Conclusion: current Fig.10 figures should be treated as exploratory artifacts, not as a successful reproduction of the paper claim."
    )
    lines.append("")

    lines.append("## Phase5 Audit")
    lines.append(f"- Source CSV: `{phase5.csv_path}`")
    lines.append(f"- Summary: {phase5.summary}")
    lines.append("")
    lines.append("| N | count | median epsilon | p90 epsilon | max epsilon | boundary rate | mean grid expansions |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for metric in phase5.per_n:
        lines.append(
            f"| {metric.n} | {metric.count} | {metric.epsilon_median:.6f} | {metric.epsilon_p90:.6f} | "
            f"{metric.epsilon_max:.6f} | {metric.boundary_rate:.3f} | {metric.mean_grid_expansions:.3f} |"
        )
    lines.append("")
    lines.append(
        f"Verdicts: counts={_bool_flag(phase5.complete_counts_pass)}, "
        f"median_monotonic={_bool_flag(phase5.median_monotonic_pass)}, "
        f"p90_monotonic={_bool_flag(phase5.p90_monotonic_pass)}, "
        f"boundary_free={_bool_flag(phase5.boundary_free_pass)}"
    )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- This is the clearest currently-successful paper trend in the repository.")
    lines.append("- `N=0` requires repeated grid expansion and remains much more biased; `N=3` and especially `N=7` recover the remnant very accurately at `t_peak`.")
    lines.append("")

    lines.append("## Next Reproduction Targets")
    lines.append("- Promote Phase5 into a polished paper-facing result: export the missing publication-grade figure and add a short methods note anchored to this audit.")
    lines.append("- Do not spend more time post-processing current Fig.10 images before the sampler/likelihood issue is resolved; the audit shows the mismatch is in the posterior itself.")
    if fig11_missing:
        lines.append("- The largest missing paper item outside Fig.10 is Fig.11: `N=0` posteriors at multiple positive `dt0` values (3/6/10 ms) compared against the `N=3, dt0=0` contour.")
        lines.append("- Fig.11 is not yet implemented in `scripts/` or present in `results/`, so it is the cleanest next coding target.")
    else:
        lines.append("- Fig.11 is now implemented at the script level; the next step is a production-length run to test the paper claim that only sufficiently late `N=0` start times recover the truth within the 90% region.")
    lines.append("")
    lines.append("## Machine-Readable Report")
    lines.append(f"- JSON: `{output_json}`")
    return "\n".join(lines) + "\n"


def _to_serializable(fig10_runs: list[Fig10RunAudit], phase5: Phase5Audit) -> dict[str, Any]:
    return {
        "truth": {
            "mf_msun": TRUTH_MF,
            "chif": TRUTH_CHIF,
            "total_mass_msun": TOTAL_MASS_MSUN,
        },
        "fig10_runs": [
            {
                "spec": {
                    "key": run.spec.key,
                    "label": run.spec.label,
                    "diag_csv": str(run.spec.diag_csv),
                    "samples_prefix": str(run.spec.samples_prefix),
                    "figure_path": str(run.spec.figure_path),
                    "base_sampler": run.spec.base_sampler,
                    "n3_samplers": list(run.spec.n3_samplers),
                },
                "per_n": [asdict(metric) for metric in run.per_n],
                "health_minimum_pass": run.health_minimum_pass,
                "n3_best_by_truth_rank": run.n3_best_by_truth_rank,
                "n3_best_by_epsilon72_q50": run.n3_best_by_epsilon72_q50,
                "n3_truth_in_top40": run.n3_truth_in_top40,
                "n0_not_in_top40": run.n0_not_in_top40,
                "paper_parity_pass": run.paper_parity_pass,
                "summary": run.summary,
            }
            for run in fig10_runs
        ],
        "phase5": {
            "csv_path": str(phase5.csv_path),
            "per_n": [asdict(metric) for metric in phase5.per_n],
            "complete_counts_pass": phase5.complete_counts_pass,
            "median_monotonic_pass": phase5.median_monotonic_pass,
            "p90_monotonic_pass": phase5.p90_monotonic_pass,
            "boundary_free_pass": phase5.boundary_free_pass,
            "paper_parity_pass": phase5.paper_parity_pass,
            "summary": phase5.summary,
        },
    }


def main() -> None:
    args = parse_args()
    output_prefix = args.output_prefix.with_suffix("") if args.output_prefix.suffix else args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_md = output_prefix.with_suffix(".md")
    output_json = output_prefix.with_suffix(".json")

    fig10_runs = [_audit_fig10_run(spec, args) for spec in _build_fig10_run_spec(args)]
    phase5 = _audit_phase5(args.phase5_csv)

    payload = _to_serializable(fig10_runs, phase5)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown = _build_markdown(fig10_runs, phase5, output_json)
    output_md.write_text(markdown, encoding="utf-8")

    print(f"markdown={output_md}")
    print(f"json={output_json}")
    for run in fig10_runs:
        print(f"fig10[{run.spec.key}] health={run.health_minimum_pass} paper_parity={run.paper_parity_pass}")
    print(f"phase5 paper_parity={phase5.paper_parity_pass}")


if __name__ == "__main__":
    main()
