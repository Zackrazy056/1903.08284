from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config_io import load_yaml, project_root_from_config


def _resolve_path(project_root: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def _credible_thresholds(hist2d: np.ndarray, probs: list[float]) -> list[float]:
    flat = np.asarray(hist2d, dtype=float).ravel()
    total = float(np.sum(flat))
    if total <= 0:
        return [0.0 for _ in probs]
    # 按密度从高到低累计，得到给定 credible level 的等高线阈值。
    idx = np.argsort(flat)[::-1]
    sorted_vals = flat[idx]
    cdf = np.cumsum(sorted_vals) / total
    out = []
    for p in probs:
        j = int(np.searchsorted(cdf, p, side="left"))
        j = min(max(j, 0), len(sorted_vals) - 1)
        out.append(float(sorted_vals[j]))
    return out


def _parse_n_list(text: str) -> list[int]:
    vals = []
    for tok in text.split(","):
        tok = tok.strip()
        if tok:
            vals.append(int(tok))
    if not vals:
        raise ValueError("Empty N list")
    return vals


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SNPE posterior contours for N=0..3")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ringdown_fig10_snpe/configs/snpe.yaml"),
        help="Path to snpe.yaml",
    )
    parser.add_argument(
        "--N",
        type=str,
        default=None,
        help="Optional comma-separated N list to plot",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    snpe_cfg_path = args.config.resolve()
    project_root = project_root_from_config(snpe_cfg_path)
    cfg = load_yaml(snpe_cfg_path)
    fig10_cfg = load_yaml(_resolve_path(project_root, cfg["use_fig10_spec_from"]))
    snpe = cfg["snpe"]

    if args.N:
        n_list = _parse_n_list(args.N)
    else:
        n_list = list(map(int, snpe["N_list"]))

    post_dir = _resolve_path(project_root, cfg["outputs"]["posterior_samples_dir"])
    out_path = _resolve_path(project_root, cfg["outputs"]["figure"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mf_min = float(fig10_cfg["priors"]["Mf_msun"]["min"])
    mf_max = float(fig10_cfg["priors"]["Mf_msun"]["max"])
    chi_min = float(fig10_cfg["priors"]["chi_f"]["min"])
    chi_max = float(fig10_cfg["priors"]["chi_f"]["max"])
    truth_mf = float(fig10_cfg["truth_parameters"]["Mf_msun"])
    truth_chi = float(fig10_cfg["truth_parameters"]["chi_f"])

    colors = {0: "#1f77b4", 1: "#2ca02c", 2: "#ff7f0e", 3: "#d62728"}

    fig = plt.figure(figsize=(8.4, 8.4))
    gs = fig.add_gridspec(4, 4, hspace=0.06, wspace=0.06)
    ax_top = fig.add_subplot(gs[0, 0:3])
    ax_joint = fig.add_subplot(gs[1:4, 0:3], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_joint)

    for N in n_list:
        npz = post_dir / f"snpe_N{N}_posterior_samples.npz"
        if not npz.exists():
            raise FileNotFoundError(f"Missing posterior samples for N={N}: {npz}")
        data = np.load(npz)
        samples = np.asarray(data["samples"], dtype=float)
        x = samples[:, 0]
        y = samples[:, 1]

        # 这里用样本直方图近似后验密度并绘制 90% credible contour。
        h2d, xedges, yedges = np.histogram2d(
            x,
            y,
            bins=90,
            range=[[mf_min, mf_max], [chi_min, chi_max]],
        )
        h2d = h2d.T
        xcent = 0.5 * (xedges[:-1] + xedges[1:])
        ycent = 0.5 * (yedges[:-1] + yedges[1:])
        xx, yy = np.meshgrid(xcent, ycent)
        lv90 = _credible_thresholds(h2d, [0.9])[0]
        if lv90 > 0:
            ax_joint.contour(xx, yy, h2d, levels=[lv90], colors=[colors.get(N, "black")], linewidths=1.8)

        ax_top.hist(x, bins=70, density=True, histtype="step", color=colors.get(N, "black"), lw=1.4, label=f"N={N}")
        ax_right.hist(y, bins=70, density=True, histtype="step", orientation="horizontal", color=colors.get(N, "black"), lw=1.4)

    ax_joint.axvline(truth_mf, color="black", ls="--", lw=1.2)
    ax_joint.axhline(truth_chi, color="black", ls="--", lw=1.2)
    ax_joint.plot([truth_mf], [truth_chi], marker="x", color="black", ms=7)
    ax_top.axvline(truth_mf, color="black", ls="--", lw=1.1)
    ax_right.axhline(truth_chi, color="black", ls="--", lw=1.1)

    ax_joint.set_xlabel(r"$M_f\ [M_\odot]$")
    ax_joint.set_ylabel(r"$\chi_f$")
    ax_top.set_ylabel("density")
    ax_right.set_xlabel("density")
    ax_top.legend(loc="upper left", fontsize=9)
    ax_joint.grid(alpha=0.2)
    ax_top.grid(alpha=0.2)
    ax_right.grid(alpha=0.2)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_joint.set_title("SNPE posterior comparison (90% contours, delta_t0=0)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
