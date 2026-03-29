from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ringdown.frequencies import make_omega_provider_22
from ringdown.metrics import remnant_error_epsilon
from ringdown.preprocess import align_to_peak
from ringdown.scan import grid_search_remnant
from ringdown.sxs_io import load_sxs_waveform22


@dataclass
class PanelConfig:
    fig_label: str
    n_overtones: int
    t0: float
    mf_min: float
    mf_max: float
    chif_min: float
    chif_max: float
    note_lines: tuple[str, str]
    vmin: float
    vmax: float
    cbar_ticks: tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce paper-style Fig. 4/5/6 remnant mismatch landscapes."
    )
    parser.add_argument("--sxs-location", type=str, default="SXS:BBH:0305/Lev6")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--t-end", type=float, default=90.0)
    parser.add_argument("--mf-points", type=int, default=121)
    parser.add_argument("--chif-points", type=int, default=121)
    parser.add_argument("--lstsq-rcond", type=float, default=1e-12)
    parser.add_argument("--max-condition-number", type=float, default=1e12)
    parser.add_argument("--max-overtone-ratio", type=float, default=1e4)
    parser.add_argument("--min-signal-norm", type=float, default=1e-14)
    parser.add_argument(
        "--include-constant-offset",
        action="store_true",
        help="Use the engineering baseline with a complex constant offset. Off by default for paper Eq.(1).",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/fig456_paper_repro"),
    )
    return parser.parse_args()


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "savefig.facecolor": "white",
        }
    )


def _panel_configs() -> list[PanelConfig]:
    return [
        PanelConfig(
            fig_label="fig4",
            n_overtones=7,
            t0=0.0,
            mf_min=0.875,
            mf_max=1.000,
            chif_min=0.5,
            chif_max=0.9,
            note_lines=(r"$N=7$", r"$t_0=t_{\rm peak}$"),
            vmin=-6.2,
            vmax=-1.2,
            cbar_ticks=tuple(np.arange(-6.0, -1.19, 0.6)),
        ),
        PanelConfig(
            fig_label="fig5",
            n_overtones=0,
            t0=0.0,
            mf_min=0.900,
            mf_max=1.450,
            chif_min=0.5,
            chif_max=0.9,
            note_lines=(r"$N=0$", r"$t_0=t_{\rm peak}$"),
            vmin=-1.35,
            vmax=-0.15,
            cbar_ticks=tuple(np.arange(-1.35, -0.149, 0.15)),
        ),
        PanelConfig(
            fig_label="fig6",
            n_overtones=0,
            t0=47.0,
            mf_min=0.875,
            mf_max=1.000,
            chif_min=0.5,
            chif_max=0.9,
            note_lines=(r"$N=0$", r"$t_0=t_{\rm peak}+47M$"),
            vmin=-6.2,
            vmax=-1.2,
            cbar_ticks=tuple(np.arange(-6.0, -1.19, 0.6)),
        ),
    ]


def _compute_panel(
    cfg: PanelConfig,
    *,
    wf,
    omega_provider,
    include_constant_offset: bool,
    t_end: float,
    mf_points: int,
    chif_points: int,
    lstsq_rcond: float,
    max_condition_number: float | None,
    max_overtone_ratio: float | None,
    min_signal_norm: float,
    mf_true: float,
    chif_true: float,
    total_mass: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    mf_grid = np.linspace(cfg.mf_min, cfg.mf_max, mf_points)
    chif_grid = np.linspace(cfg.chif_min, cfg.chif_max, chif_points)
    result = grid_search_remnant(
        wf=wf,
        n_overtones=cfg.n_overtones,
        t0=cfg.t0,
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
    if not np.any(result.valid_mask):
        raise RuntimeError(f"no valid grid points for {cfg.fig_label}")

    mismatch_grid = result.mismatch_grid.copy()
    mismatch_grid[result.valid_mask] = np.maximum(mismatch_grid[result.valid_mask], 1e-20)
    log_mismatch = np.full_like(mismatch_grid, np.nan, dtype=float)
    log_mismatch[result.valid_mask] = np.log10(mismatch_grid[result.valid_mask])
    eps = remnant_error_epsilon(
        mf_fit=result.best_mf,
        mf_true=mf_true,
        chi_fit=result.best_chif,
        chi_true=chif_true,
        total_mass=total_mass,
    )
    meta = {
        "n_overtones": float(cfg.n_overtones),
        "t0": float(cfg.t0),
        "best_mf": float(result.best_mf),
        "best_chif": float(result.best_chif),
        "best_mismatch": float(result.best_mismatch),
        "epsilon": float(eps),
        "log10_mismatch_min": float(np.nanmin(log_mismatch)),
        "log10_mismatch_max": float(np.nanmax(log_mismatch)),
        "valid_grid_points": float(np.count_nonzero(result.valid_mask)),
        "grid_points_total": float(result.valid_mask.size),
    }
    return mf_grid, chif_grid, log_mismatch, meta


def _draw_panel(
    *,
    fig,
    ax,
    mf_grid: np.ndarray,
    chif_grid: np.ndarray,
    log_mismatch: np.ndarray,
    cfg: PanelConfig,
    mf_true: float,
    chif_true: float,
) -> None:
    mesh = ax.imshow(
        log_mismatch,
        origin="lower",
        interpolation="nearest",
        extent=[float(chif_grid[0]), float(chif_grid[-1]), float(mf_grid[0]), float(mf_grid[-1])],
        aspect="auto",
        cmap="gist_heat_r",
        vmin=cfg.vmin,
        vmax=cfg.vmax,
    )
    ax.axvline(chif_true, color="white", lw=0.65, alpha=0.95)
    ax.axhline(mf_true, color="white", lw=0.65, alpha=0.95)
    ax.set_xlabel(r"$\chi_f$")
    ax.set_ylabel(r"$M_f\ [M]$")
    ax.set_xlim(cfg.chif_min, cfg.chif_max)
    ax.set_ylim(cfg.mf_min, cfg.mf_max)
    ax.set_box_aspect(1.0)
    ax.text(
        0.93,
        0.11,
        cfg.note_lines[0] + "\n" + cfg.note_lines[1],
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="white",
        fontsize=10.5,
    )
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label(r"$\log_{10}\,\mathcal{M}$")
    cbar.set_ticks(list(cfg.cbar_ticks))


def _write_summary(path: Path, *, args: argparse.Namespace, metas: dict[str, dict[str, float]], mf_true: float, chif_true: float) -> None:
    lines = [
        "# Fig. 4/5/6 Reproduction Audit",
        "",
        f"- Source waveform: `{args.sxs_location}`",
        f"- Model: {'Eq.(1) + constant offset b' if args.include_constant_offset else 'pure Eq.(1) QNM model'}",
        f"- Time window end: `{args.t_end}`",
        f"- Grid: `{args.mf_points} x {args.chif_points}`",
        f"- NR remnant: `Mf={mf_true:.9f}`, `chif={chif_true:.9f}`",
        "",
        "| Figure | N | t0 | best Mf | best chif | epsilon | min log10 mismatch | max log10 mismatch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in ["fig4", "fig5", "fig6"]:
        meta = metas[label]
        lines.append(
            f"| {label} | {int(meta['n_overtones'])} | {meta['t0']:.1f} | "
            f"{meta['best_mf']:.9f} | {meta['best_chif']:.9f} | {meta['epsilon']:.6e} | "
            f"{meta['log10_mismatch_min']:.3f} | {meta['log10_mismatch_max']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    _configure_style()

    wf_raw, info = load_sxs_waveform22(args.sxs_location, download=not args.no_download)
    wf, _ = align_to_peak(wf_raw)
    mf_true = float(info.remnant_mass)
    chif_true = float(info.remnant_chif_z)
    total_mass = float(info.initial_total_mass) if info.initial_total_mass is not None else 1.0

    provider = make_omega_provider_22()
    configs = _panel_configs()

    results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    metas: dict[str, dict[str, float]] = {}
    for cfg in configs:
        mf_grid, chif_grid, log_mismatch, meta = _compute_panel(
            cfg,
            wf=wf,
            omega_provider=provider,
            include_constant_offset=args.include_constant_offset,
            t_end=args.t_end,
            mf_points=args.mf_points,
            chif_points=args.chif_points,
            lstsq_rcond=args.lstsq_rcond,
            max_condition_number=args.max_condition_number,
            max_overtone_ratio=args.max_overtone_ratio,
            min_signal_norm=args.min_signal_norm,
            mf_true=mf_true,
            chif_true=chif_true,
            total_mass=total_mass,
        )
        results[cfg.fig_label] = (mf_grid, chif_grid, log_mismatch)
        metas[cfg.fig_label] = meta

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(10.1, 3.45), constrained_layout=True)
    for ax, cfg in zip(axes, configs):
        mf_grid, chif_grid, log_mismatch = results[cfg.fig_label]
        _draw_panel(
            fig=fig,
            ax=ax,
            mf_grid=mf_grid,
            chif_grid=chif_grid,
            log_mismatch=log_mismatch,
            cfg=cfg,
            mf_true=mf_true,
            chif_true=chif_true,
        )
    combined_png = args.output_prefix.with_suffix(".png")
    combined_pdf = args.output_prefix.with_suffix(".pdf")
    fig.savefig(combined_png, dpi=320)
    fig.savefig(combined_pdf)
    plt.close(fig)

    for cfg in configs:
        one_fig, one_ax = plt.subplots(1, 1, figsize=(3.3, 3.1), constrained_layout=True)
        mf_grid, chif_grid, log_mismatch = results[cfg.fig_label]
        _draw_panel(
            fig=one_fig,
            ax=one_ax,
            mf_grid=mf_grid,
            chif_grid=chif_grid,
            log_mismatch=log_mismatch,
            cfg=cfg,
            mf_true=mf_true,
            chif_true=chif_true,
        )
        out_png = args.output_prefix.with_name(f"{args.output_prefix.stem}_{cfg.fig_label}.png")
        out_pdf = args.output_prefix.with_name(f"{args.output_prefix.stem}_{cfg.fig_label}.pdf")
        one_fig.savefig(out_png, dpi=320)
        one_fig.savefig(out_pdf)
        plt.close(one_fig)

    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    payload = {
        "source": args.sxs_location,
        "include_constant_offset": args.include_constant_offset,
        "t_end": args.t_end,
        "mf_points": args.mf_points,
        "chif_points": args.chif_points,
        "nr_remnant": {"mf": mf_true, "chif": chif_true},
        "panels": {cfg.fig_label: {**asdict(cfg), **metas[cfg.fig_label]} for cfg in configs},
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_summary(md_path, args=args, metas=metas, mf_true=mf_true, chif_true=chif_true)

    print(f"source={args.sxs_location}")
    print(f"include_constant_offset={args.include_constant_offset}")
    print(f"output_combined_png={combined_png}")
    print(f"output_combined_pdf={combined_pdf}")
    for cfg in configs:
        meta = metas[cfg.fig_label]
        print(
            f"{cfg.fig_label}: best_mf={meta['best_mf']:.9f}, "
            f"best_chif={meta['best_chif']:.9f}, epsilon={meta['epsilon']:.6e}, "
            f"log10_min={meta['log10_mismatch_min']:.3f}"
        )


if __name__ == "__main__":
    main()
