from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ringdown.frequencies import kerr_qnm_omegas_22n
from ringdown.metrics import mismatch, remnant_error_epsilon
from ringdown.paper_fig10 import (
    PaperFigure10Config,
    build_paper_fig10_signal,
    paper_fig10_signal_diagnostics,
)


def parse_int_list(text: str) -> list[int]:
    vals = [int(s.strip()) for s in text.split(",") if s.strip()]
    if not vals:
        raise ValueError("empty integer list")
    return sorted(set(vals))


def real_profile_fit(
    tau_m: np.ndarray,
    data: np.ndarray,
    omegas_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.exp(-1j * tau_m[:, None] * omegas_m[None, :])
    design = np.column_stack([q.real, -q.imag])
    coeffs, *_ = np.linalg.lstsq(design, data, rcond=1e-12)
    model = design @ coeffs
    return coeffs, model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sxs-location", type=str, default="SXS:BBH:0305v2.0/Lev6")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--n-values", type=str, default="0,3")
    p.add_argument("--m-total-msun", type=float, default=72.0)
    p.add_argument("--distance-mpc", type=float, default=400.0)
    p.add_argument("--delta-t0-ms", type=float, default=0.0)
    p.add_argument("--t-end", type=float, default=90.0)
    p.add_argument("--f-min-hz", type=float, default=20.0)
    p.add_argument("--f-max-hz", type=float, default=1024.0)
    p.add_argument("--df-hz", type=float, default=1.0)
    p.add_argument("--mf-min-msun", type=float, default=58.0)
    p.add_argument("--mf-max-msun", type=float, default=78.0)
    p.add_argument("--chif-min", type=float, default=0.45)
    p.add_argument("--chif-max", type=float, default=0.85)
    p.add_argument("--mf-points", type=int, default=60)
    p.add_argument("--chif-points", type=int, default=60)
    p.add_argument("--output", type=Path, default=Path("results/fig10_paper_forward_sanity.png"))
    p.add_argument("--summary-json", type=Path, default=Path("results/fig10_paper_forward_sanity.json"))
    p.add_argument("--summary-md", type=Path, default=Path("results/fig10_paper_forward_sanity.md"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n_values = parse_int_list(args.n_values)
    signal = build_paper_fig10_signal(
        PaperFigure10Config(
            sxs_location=args.sxs_location,
            total_mass_msun=args.m_total_msun,
            distance_mpc=args.distance_mpc,
            delta_t0_ms=args.delta_t0_ms,
            t_end_m=args.t_end,
            f_min_hz=args.f_min_hz,
            f_max_hz=args.f_max_hz,
            df_hz=args.df_hz,
            download=not args.no_download,
        )
    )

    mf_grid = np.linspace(args.mf_min_msun, args.mf_max_msun, args.mf_points)
    chif_grid = np.linspace(args.chif_min, args.chif_max, args.chif_points)
    tau_m = signal.t_window_m
    data = signal.signal_window

    result_payload: dict[str, object] = {
        "signal_diagnostics": paper_fig10_signal_diagnostics(signal),
        "grid": {
            "mf_min_msun": float(args.mf_min_msun),
            "mf_max_msun": float(args.mf_max_msun),
            "mf_points": int(args.mf_points),
            "chif_min": float(args.chif_min),
            "chif_max": float(args.chif_max),
            "chif_points": int(args.chif_points),
        },
        "per_n": {},
    }

    fig, axes = plt.subplots(1, len(n_values), figsize=(6.2 * len(n_values), 4.9), constrained_layout=True)
    if len(n_values) == 1:
        axes = np.array([axes])

    for ax, n in zip(axes, n_values):
        mismatch_grid = np.full((mf_grid.size, chif_grid.size), np.nan, dtype=float)
        epsilon_grid = np.full_like(mismatch_grid, np.nan)
        best = None
        best_model = None
        for i, mf_msun in enumerate(mf_grid):
            mf_frac = float(mf_msun / signal.config.total_mass_msun)
            for j, chif in enumerate(chif_grid):
                omegas_m = kerr_qnm_omegas_22n(mf=mf_frac, chif=float(chif), n_max=n)
                _, model = real_profile_fit(tau_m, data, omegas_m)
                mm = mismatch(data, model, tau_m)
                eps = remnant_error_epsilon(
                    float(mf_msun),
                    signal.true_mf_msun,
                    float(chif),
                    signal.true_chif,
                    signal.config.total_mass_msun,
                )
                mismatch_grid[i, j] = mm
                epsilon_grid[i, j] = eps
                if best is None or mm < best["best_mismatch"]:
                    best = {
                        "best_mismatch": float(mm),
                        "best_mf_msun": float(mf_msun),
                        "best_chif": float(chif),
                        "best_epsilon": float(eps),
                    }
                    best_model = model

        if best is None or best_model is None:
            raise RuntimeError(f"no valid grid points found for N={n}")

        heat = np.log10(np.clip(mismatch_grid.T, 1e-12, None))
        im = ax.imshow(
            heat,
            origin="lower",
            aspect="auto",
            extent=[mf_grid[0], mf_grid[-1], chif_grid[0], chif_grid[-1]],
            cmap="gist_heat_r",
            vmin=float(np.nanmin(heat)),
            vmax=float(np.nanpercentile(heat, 99.5)),
        )
        ax.axvline(signal.true_mf_msun, color="white", lw=1.0, alpha=0.85)
        ax.axhline(signal.true_chif, color="white", lw=1.0, alpha=0.85)
        ax.plot(best["best_mf_msun"], best["best_chif"], marker="o", ms=4.5, color="cyan")
        ax.set_xlabel(r"$M_f\ [M_\odot]$")
        ax.set_ylabel(r"$\chi_f$")
        ax.set_title(
            rf"$N={n}$, best=({best['best_mf_msun']:.2f}, {best['best_chif']:.3f})"
            "\n"
            rf"$\mathcal{{M}}_{{\rm min}}={best['best_mismatch']:.2e}$"
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(r"$\log_{10}\mathcal{M}$")

        result_payload["per_n"][str(n)] = {
            **best,
            "mismatch_grid_shape": [int(mismatch_grid.shape[0]), int(mismatch_grid.shape[1])],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")

    md_lines = [
        "# Fig.10 Paper-Forward Sanity",
        "",
        "## Signal Diagnostics",
    ]
    for key, value in paper_fig10_signal_diagnostics(signal).items():
        md_lines.append(f"- `{key}`: {value}")
    md_lines.extend(["", "## Grid Results"])
    for n in n_values:
        best = result_payload["per_n"][str(n)]
        md_lines.append(
            f"- `N={n}`: best `(Mf, chif)=({best['best_mf_msun']:.4f}, {best['best_chif']:.4f})`, "
            f"`mismatch={best['best_mismatch']:.6e}`, `epsilon={best['best_epsilon']:.6e}`"
        )
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"output={args.output}")
    print(f"summary_json={args.summary_json}")
    print(f"summary_md={args.summary_md}")


if __name__ == "__main__":
    main()
