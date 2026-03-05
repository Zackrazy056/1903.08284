from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ringdown.frequencies import make_omega_provider_22
from ringdown.metrics import remnant_error_epsilon
from ringdown.preprocess import align_to_peak
from ringdown.scan import grid_search_remnant
from ringdown.sxs_io import load_sxs_waveform22


def parse_n_list(text: str) -> list[int]:
    vals = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(int(p))
    if not vals:
        raise ValueError("empty N list")
    return sorted(set(vals))


def select_simulations(
    *,
    max_sims: int,
    q_max: float,
    spin_abs_max: float,
    aligned_tol: float,
    no_download: bool,
    sample_mode: str,
) -> pd.DataFrame:
    import sxs

    df = sxs.load("dataframe", download=not no_download)
    # Focus on BBH systems and use available reference metadata.
    mask = df.index.to_series().str.startswith("SXS:BBH:")
    if "deprecated" in df.columns:
        mask &= ~df["deprecated"].fillna(False)
    if "reference_mass_ratio" in df.columns:
        mask &= df["reference_mass_ratio"].fillna(np.inf) <= q_max

    # Require nearly aligned component spins.
    for c in [
        "reference_dimensionless_spin1_x",
        "reference_dimensionless_spin1_y",
        "reference_dimensionless_spin2_x",
        "reference_dimensionless_spin2_y",
    ]:
        if c in df.columns:
            mask &= df[c].fillna(0.0).abs() <= aligned_tol

    # Magnitude constraint close to paper setup.
    for c in ["reference_dimensionless_spin1_mag", "reference_dimensionless_spin2_mag"]:
        if c in df.columns:
            mask &= df[c].fillna(np.inf) <= spin_abs_max

    out = df.loc[mask].copy()
    if "reference_mass_ratio" in out.columns:
        out = out.sort_values("reference_mass_ratio")
    if max_sims > 0:
        if sample_mode == "head" or len(out) <= max_sims:
            out = out.head(max_sims)
        elif sample_mode == "stratified_q":
            idx = np.linspace(0, len(out) - 1, max_sims, dtype=int)
            out = out.iloc[idx]
        else:
            raise ValueError(f"unknown sample_mode: {sample_mode}")
    return out


def run_single_sim(
    sim_id: str,
    n_values: Sequence[int],
    *,
    t0: float,
    t_end: float,
    mf_half_width: float,
    chif_half_width: float,
    mf_points: int,
    chif_points: int,
    lstsq_rcond: float,
    include_constant_offset: bool,
    max_condition_number: float | None,
    max_overtone_ratio: float | None,
    min_signal_norm: float,
    no_download: bool,
    max_grid_expansions: int,
    grid_expand_factor: float,
) -> list[dict]:
    wf_raw, info = load_sxs_waveform22(sim_id, download=not no_download)
    wf, _ = align_to_peak(wf_raw)
    if info.remnant_mass is None or info.remnant_chif_z is None:
        return []

    mf_true = float(info.remnant_mass)
    chif_true = float(info.remnant_chif_z)
    total_mass = float(info.initial_total_mass) if info.initial_total_mass is not None else 1.0

    omega_provider = make_omega_provider_22()
    rows: list[dict] = []
    for n in n_values:
        current_mf_half = mf_half_width
        current_chi_half = chif_half_width
        res = None
        boundary_hit = False
        expansions_used = 0
        for expand_idx in range(max_grid_expansions + 1):
            mf_grid = np.linspace(
                max(1e-6, mf_true - current_mf_half),
                mf_true + current_mf_half,
                mf_points,
            )
            chif_grid = np.linspace(
                max(-0.999, chif_true - current_chi_half),
                min(0.999, chif_true + current_chi_half),
                chif_points,
            )
            candidate = grid_search_remnant(
                wf=wf,
                n_overtones=n,
                t0=t0,
                mf_grid=mf_grid,
                chif_grid=chif_grid,
                omega_provider=omega_provider,
                t_end=t_end,
                lstsq_rcond=lstsq_rcond,
                include_constant_offset=include_constant_offset,
                max_condition_number=max_condition_number,
                max_overtone_to_fund_ratio=max_overtone_ratio,
                min_signal_norm=min_signal_norm,
            )
            if not np.any(candidate.valid_mask):
                res = candidate
                break
            valid_mm = candidate.mismatch_grid.copy()
            valid_mm[~candidate.valid_mask] = np.nan
            min_flat = int(np.nanargmin(valid_mm))
            i_best, j_best = np.unravel_index(min_flat, valid_mm.shape)
            at_boundary = (
                i_best == 0
                or i_best == mf_points - 1
                or j_best == 0
                or j_best == chif_points - 1
            )
            res = candidate
            boundary_hit = at_boundary
            expansions_used = expand_idx
            if not at_boundary:
                break
            if expand_idx < max_grid_expansions:
                current_mf_half *= grid_expand_factor
                current_chi_half *= grid_expand_factor

        if res is None:
            continue
        if not np.isfinite(res.best_mismatch):
            continue
        eps = remnant_error_epsilon(
            mf_fit=res.best_mf,
            mf_true=mf_true,
            chi_fit=res.best_chif,
            chi_true=chif_true,
            total_mass=total_mass,
        )
        rows.append(
            {
                "simulation": sim_id,
                "n_overtones": n,
                "mf_true": mf_true,
                "chif_true": chif_true,
                "mf_fit": res.best_mf,
                "chif_fit": res.best_chif,
                "best_mismatch": res.best_mismatch,
                "epsilon": eps,
                "valid_grid_points": int(np.count_nonzero(res.valid_mask)),
                "grid_points_total": int(res.valid_mask.size),
                "grid_expansions_used": expansions_used,
                "best_at_grid_boundary": boundary_hit,
                "mf_half_width_final": current_mf_half,
                "chif_half_width_final": current_chi_half,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", type=str, default="0,3,7")
    parser.add_argument("--max-sims", type=int, default=12)
    parser.add_argument("--sample-mode", choices=["stratified_q", "head"], default="stratified_q")
    parser.add_argument("--q-max", type=float, default=8.0)
    parser.add_argument("--spin-abs-max", type=float, default=0.8)
    parser.add_argument("--aligned-tol", type=float, default=5e-3)
    parser.add_argument("--t0", type=float, default=0.0)
    parser.add_argument("--t-end", type=float, default=90.0)

    parser.add_argument("--mf-half-width", type=float, default=0.03)
    parser.add_argument("--chif-half-width", type=float, default=0.06)
    parser.add_argument("--mf-points", type=int, default=41)
    parser.add_argument("--chif-points", type=int, default=41)

    parser.add_argument("--lstsq-rcond", type=float, default=1e-12)
    parser.add_argument(
        "--no-constant-offset",
        action="store_true",
        help="Disable default complex constant-offset basis term b in fitting.",
    )
    parser.add_argument("--max-condition-number", type=float, default=1e12)
    parser.add_argument("--max-overtone-ratio", type=float, default=1e4)
    parser.add_argument("--min-signal-norm", type=float, default=1e-14)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--max-grid-expansions", type=int, default=4)
    parser.add_argument("--grid-expand-factor", type=float, default=2.0)

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/phase5_sxs_error_distribution.csv"),
    )
    parser.add_argument(
        "--output-fig",
        type=Path,
        default=Path("results/phase5_sxs_error_distribution.png"),
    )
    args = parser.parse_args()

    n_values = parse_n_list(args.n_values)
    sims = select_simulations(
        max_sims=args.max_sims,
        q_max=args.q_max,
        spin_abs_max=args.spin_abs_max,
        aligned_tol=args.aligned_tol,
        no_download=args.no_download,
        sample_mode=args.sample_mode,
    )
    sim_ids = list(sims.index)
    print(f"selected_simulations={len(sim_ids)}")

    rows: list[dict] = []
    failures: list[dict] = []
    for i, sim_id in enumerate(sim_ids, start=1):
        print(f"[{i}/{len(sim_ids)}] {sim_id}")
        try:
            sim_rows = run_single_sim(
                sim_id=sim_id,
                n_values=n_values,
                t0=args.t0,
                t_end=args.t_end,
                mf_half_width=args.mf_half_width,
                chif_half_width=args.chif_half_width,
                mf_points=args.mf_points,
                chif_points=args.chif_points,
                lstsq_rcond=args.lstsq_rcond,
                include_constant_offset=not args.no_constant_offset,
                max_condition_number=args.max_condition_number,
                max_overtone_ratio=args.max_overtone_ratio,
                min_signal_norm=args.min_signal_norm,
                no_download=args.no_download,
                max_grid_expansions=args.max_grid_expansions,
                grid_expand_factor=args.grid_expand_factor,
            )
            rows.extend(sim_rows)
        except Exception as exc:
            failures.append({"simulation": sim_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  failed: {type(exc).__name__}")

    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"rows={len(df)} csv={args.output_csv}")
    print(f"include_constant_offset={not args.no_constant_offset}")

    if failures:
        fail_csv = args.output_csv.with_name(args.output_csv.stem + "_failures.csv")
        pd.DataFrame(failures).to_csv(fail_csv, index=False)
        print(f"failures={len(failures)} failure_csv={fail_csv}")

    if len(df) == 0:
        print("no successful fits; skip figure")
        return

    args.output_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.6, 6.2))
    bins = np.logspace(-4.5, 0.0, 36)
    for n in n_values:
        vals = df.loc[df["n_overtones"] == n, "epsilon"].to_numpy()
        if vals.size == 0:
            continue
        plt.hist(vals, bins=bins, alpha=0.45, label=f"N={n}")
        print(
            f"N={n}: count={vals.size} median={np.median(vals):.3e} "
            f"p90={np.percentile(vals, 90):.3e} max={np.max(vals):.3e}"
        )
    plt.xscale("log")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel("Number of simulations")
    plt.title("Remnant-parameter error distribution at $t_0=t_{peak}$")
    plt.grid(True, which="both", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_fig, dpi=180)
    print(f"figure={args.output_fig}")


if __name__ == "__main__":
    main()
