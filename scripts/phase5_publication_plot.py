from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a publication-style Phase5 epsilon summary plot from an existing sweep CSV."
    )
    p.add_argument(
        "--input-csv",
        type=Path,
        default=Path("results/phase5_sweep_12_adaptive4.csv"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase5_sweep_12_adaptive4_publication.png"),
    )
    p.add_argument("--bins", type=int, default=28)
    p.add_argument("--xmin", type=float, default=1e-4)
    p.add_argument("--xmax", type=float, default=1.0)
    return p.parse_args()


def _load_grouped_eps(path: Path) -> dict[int, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"input csv not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("input csv has no rows")
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["n_overtones"]), []).append(float(row["epsilon"]))
    return {n: np.asarray(vals, dtype=float) for n, vals in grouped.items()}


def _ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(np.asarray(x, dtype=float))
    ys = np.arange(1, xs.size + 1, dtype=float) / xs.size
    return xs, ys


def _prepare_for_log_plot(x: np.ndarray, xmin: float) -> np.ndarray:
    vals = np.asarray(x, dtype=float).copy()
    positive = vals[vals > 0.0]
    if positive.size == 0:
        vals[:] = xmin
        return vals
    floor = max(float(np.min(positive)) * 0.5, xmin)
    vals[vals <= 0.0] = floor
    return vals


def main() -> None:
    args = parse_args()
    grouped = _load_grouped_eps(args.input_csv)
    styles = {
        0: {"color": "#1f77b4", "label": "N=0"},
        3: {"color": "#d62728", "label": "N=3"},
        7: {"color": "#d4a017", "label": "N=7"},
    }
    bins = np.logspace(np.log10(args.xmin), np.log10(args.xmax), args.bins)

    fig = plt.figure(figsize=(10.2, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.22)
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_cdf = fig.add_subplot(gs[0, 1])

    for n in sorted(grouped):
        vals = grouped[n]
        vals_plot = _prepare_for_log_plot(vals, args.xmin)
        st = styles.get(n, {"color": None, "label": f"N={n}"})
        ax_hist.hist(
            vals_plot,
            bins=bins,
            histtype="stepfilled",
            alpha=0.28,
            lw=1.8,
            color=st["color"],
            edgecolor=st["color"],
            label=f"{st['label']} (median={np.median(vals):.3e})",
        )
        ax_hist.axvline(max(float(np.median(vals_plot)), args.xmin), color=st["color"], lw=1.6, ls="--", alpha=0.9)

        xs, ys = _ecdf(vals_plot)
        ax_cdf.step(xs, ys, where="post", color=st["color"], lw=2.1, label=st["label"])
        ax_cdf.scatter(
            [max(float(np.median(vals_plot)), args.xmin)],
            [0.5],
            color=st["color"],
            s=28,
            zorder=3,
        )

    ax_hist.set_xscale("log")
    ax_hist.set_xlim(args.xmin, args.xmax)
    ax_hist.set_xlabel(r"$\epsilon$")
    ax_hist.set_ylabel("Simulation count")
    ax_hist.set_title(r"Remnant-error distribution at $t_0=t_{peak}$")
    ax_hist.grid(True, which="both", alpha=0.18)
    ax_hist.legend(frameon=False, fontsize=9, loc="upper left")

    ax_cdf.set_xscale("log")
    ax_cdf.set_xlim(args.xmin, args.xmax)
    ax_cdf.set_ylim(0.0, 1.02)
    ax_cdf.set_xlabel(r"$\epsilon$")
    ax_cdf.set_ylabel("Empirical CDF")
    ax_cdf.set_title("Empirical CDF by overtone count")
    ax_cdf.grid(True, which="both", alpha=0.18)
    ax_cdf.legend(frameon=False, fontsize=9, loc="lower right")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
