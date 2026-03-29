from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a paper-style Fig. 3 epsilon histogram from an existing Phase5 CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="CSV produced by scripts/phase5_sxs_error_distribution.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output image path, typically a PNG.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=None,
        help="Optional vector PDF export. Defaults to the PNG stem with .pdf suffix.",
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=None,
        help="Optional markdown summary path.",
    )
    parser.add_argument("--bins", type=int, default=28)
    parser.add_argument("--xmin", type=float, default=1e-4)
    parser.add_argument("--xmax", type=float, default=1.0)
    parser.add_argument("--fig-width", type=float, default=3.35)
    parser.add_argument("--fig-height", type=float, default=2.7)
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
            "axes.titlesize": 8.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": False,
            "savefig.facecolor": "white",
        }
    )


def _load_grouped_eps(path: Path) -> dict[int, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"input csv not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"input csv has no rows: {path}")
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["n_overtones"]), []).append(float(row["epsilon"]))
    return {n: np.asarray(values, dtype=float) for n, values in grouped.items()}


def _prepare_for_log_plot(values: np.ndarray, xmin: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    positive = arr[arr > 0.0]
    if positive.size == 0:
        arr[:] = xmin
        return arr
    floor = max(float(np.min(positive)) * 0.5, xmin)
    arr[arr <= 0.0] = floor
    return arr


def _write_summary(summary_path: Path, grouped_raw: dict[int, np.ndarray], plotted: dict[int, np.ndarray]) -> None:
    lines = [
        "# Fig. 3 Reproduction Summary",
        "",
        "| N | count | median epsilon | p90 epsilon | max epsilon | plotted floor |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for n in sorted(grouped_raw):
        vals = grouped_raw[n]
        plot_vals = plotted[n]
        plotted_floor = float(np.min(plot_vals))
        lines.append(
            f"| {n} | {vals.size} | {np.median(vals):.6e} | "
            f"{np.percentile(vals, 90):.6e} | {np.max(vals):.6e} | {plotted_floor:.6e} |"
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    _configure_style()

    grouped_raw = _load_grouped_eps(args.input_csv)
    grouped_plot = {
        n: _prepare_for_log_plot(values, xmin=args.xmin) for n, values in grouped_raw.items()
    }

    colors = {
        0: "#202020",
        3: "#d97706",
        7: "#4f79c6",
    }
    bins = np.logspace(np.log10(args.xmin), np.log10(args.xmax), args.bins)

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    peak_count = 0
    for n in [0, 3, 7]:
        values = grouped_plot.get(n)
        if values is None or values.size == 0:
            continue
        counts, _, _ = ax.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=1.35,
            color=colors.get(n, None),
            label=rf"$N = {n}$",
        )
        peak_count = max(peak_count, int(np.max(counts)))

    ax.set_xscale("log")
    ax.set_xlim(args.xmin, args.xmax)
    if peak_count > 0:
        ymax = 5 * int(np.ceil((peak_count + 1) / 5.0))
        ax.set_ylim(0, ymax)
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel("Number of simulations")
    ax.legend(loc="upper left", handlelength=1.8, borderaxespad=0.3)

    fig.tight_layout(pad=0.35)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)

    output_pdf = args.output_pdf or args.output.with_suffix(".pdf")
    fig.savefig(output_pdf)

    if args.summary_md is not None:
        _write_summary(args.summary_md, grouped_raw, grouped_plot)

    print(f"input_csv={args.input_csv}")
    print(f"output={args.output}")
    print(f"output_pdf={output_pdf}")
    for n in sorted(grouped_raw):
        vals = grouped_raw[n]
        print(
            f"N={n}: count={vals.size} median={np.median(vals):.6e} "
            f"p90={np.percentile(vals, 90):.6e} max={np.max(vals):.6e}"
        )


if __name__ == "__main__":
    main()
