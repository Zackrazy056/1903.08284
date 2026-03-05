from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def credible_level_2d(pdf: np.ndarray, cred: float = 0.9) -> float:
    flat = pdf.ravel()
    order = np.argsort(flat)[::-1]
    cdf = np.cumsum(flat[order])
    idx = int(np.searchsorted(cdf, cred, side="left"))
    idx = min(max(idx, 0), order.size - 1)
    return float(flat[order[idx]])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--samples-prefix",
        type=Path,
        default=Path("results/fig10_kombine_emcee_full_prod2_w128_samples.npz"),
    )
    p.add_argument(
        "--diag-csv",
        type=Path,
        default=Path("results/fig10_kombine_emcee_full_prod2_w128_diag.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/publication_figures/fig10"),
    )
    p.add_argument(
        "--output-stem",
        type=str,
        default="fig10_papergrade_final_v1_20260301",
    )
    p.add_argument("--chi-min", type=float, default=0.5)
    p.add_argument("--chi-max", type=float, default=1.0)
    p.add_argument("--mf-min", type=float, default=50.0)
    p.add_argument("--mf-max", type=float, default=100.0)
    p.add_argument("--kde-grid", type=int, default=260)
    p.add_argument("--max-kde-samples", type=int, default=120000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--min-acceptance", type=float, default=0.01)
    p.add_argument("--n3-source", choices=["consistency", "kombine", "emcee"], default="consistency")
    p.add_argument("--n3-max-mf-diff", type=float, default=2.0)
    p.add_argument("--n3-max-chif-diff", type=float, default=0.06)
    return p.parse_args()


def load_samples(prefix: Path, n: int, n3_source: str) -> np.ndarray:
    if n == 3 and n3_source == "emcee":
        target = prefix.with_name(prefix.stem + f"_N{n}_emcee.npz")
        if target.exists():
            return np.load(target, allow_pickle=False)["flat"]
        raise FileNotFoundError(f"missing N=3 emcee samples: {target}")

    # Prefer kombine when available for legacy runs; otherwise fallback to emcee.
    target_k = prefix.with_name(prefix.stem + f"_N{n}_kombine.npz")
    if target_k.exists():
        return np.load(target_k, allow_pickle=False)["flat"]
    target_e = prefix.with_name(prefix.stem + f"_N{n}_emcee.npz")
    if target_e.exists():
        return np.load(target_e, allow_pickle=False)["flat"]
    raise FileNotFoundError(f"missing sample file for N={n}: {target_k} or {target_e}")


def maybe_subsample(x: np.ndarray, nmax: int, rng: np.random.Generator) -> np.ndarray:
    if x.shape[0] <= nmax:
        return x
    idx = rng.choice(x.shape[0], size=nmax, replace=False)
    return x[idx]


def parse_diag_rows(diag_csv: Path) -> list[dict[str, str]]:
    if not diag_csv.exists():
        raise FileNotFoundError(f"diagnostics csv not found: {diag_csv}")
    with diag_csv.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_row(rows: list[dict[str, str]], n: int, sampler: str) -> dict[str, str] | None:
    for r in rows:
        if r.get("N", "").strip() == str(n) and r.get("sampler", "").strip().lower() == sampler.lower():
            return r
    return None


def get_preferred_row(rows: list[dict[str, str]], n: int) -> dict[str, str] | None:
    # Prefer emcee diagnostics for strict-gated production runs.
    row_e = get_row(rows, n, "emcee")
    if row_e is not None:
        return row_e
    return get_row(rows, n, "kombine")


def check_row_health(row: dict[str, str], min_acceptance: float) -> tuple[bool, str]:
    try:
        acc = float(row["acceptance"])
        tau_mf = float(row["tau_mf"])
        tau_ch = float(row["tau_chif"])
    except Exception as exc:
        return False, f"invalid diagnostics row format: {exc}"
    if not np.isfinite(acc):
        return False, "acceptance is not finite"
    if acc < min_acceptance:
        return False, f"acceptance={acc:.6f} < {min_acceptance:.6f}"
    if not np.isfinite(tau_mf) or not np.isfinite(tau_ch):
        return False, f"tau invalid (tau_mf={tau_mf}, tau_chif={tau_ch})"
    return True, ""


def load_n3_with_consistency(prefix: Path, max_mf_diff: float, max_chif_diff: float) -> tuple[np.ndarray, float, float]:
    # Preferred strict mode: emcee vs emcee_alt.
    e_path = prefix.with_name(prefix.stem + "_N3_emcee.npz")
    ea_path = prefix.with_name(prefix.stem + "_N3_emcee_alt.npz")
    if e_path.exists() and ea_path.exists():
        e = np.load(e_path, allow_pickle=False)["flat"]
        ea = np.load(ea_path, allow_pickle=False)["flat"]
        e_q50 = np.quantile(e[:, :2], 0.5, axis=0)
        ea_q50 = np.quantile(ea[:, :2], 0.5, axis=0)
        mf_diff = float(abs(e_q50[0] - ea_q50[0]))
        ch_diff = float(abs(e_q50[1] - ea_q50[1]))
        if mf_diff > max_mf_diff or ch_diff > max_chif_diff:
            raise RuntimeError(
                "N=3 consistency check failed: "
                f"|mf_q50(e-e_alt)|={mf_diff:.4f} (max {max_mf_diff:.4f}), "
                f"|chif_q50(e-e_alt)|={ch_diff:.4f} (max {max_chif_diff:.4f})"
            )
        return np.vstack([e, ea]), mf_diff, ch_diff

    # Legacy fallback: kombine vs emcee.
    k_path = prefix.with_name(prefix.stem + "_N3_kombine.npz")
    if not k_path.exists() or not e_path.exists():
        raise FileNotFoundError("N=3 consistency mode requires emcee+emcee_alt or kombine+emcee sample files")
    k = np.load(k_path, allow_pickle=False)["flat"]
    e = np.load(e_path, allow_pickle=False)["flat"]
    k_q50 = np.quantile(k[:, :2], 0.5, axis=0)
    e_q50 = np.quantile(e[:, :2], 0.5, axis=0)
    mf_diff = float(abs(k_q50[0] - e_q50[0]))
    ch_diff = float(abs(k_q50[1] - e_q50[1]))
    if mf_diff > max_mf_diff or ch_diff > max_chif_diff:
        raise RuntimeError(
            "N=3 consistency check failed: "
            f"|mf_q50(k-e)|={mf_diff:.4f} (max {max_mf_diff:.4f}), "
            f"|chif_q50(k-e)|={ch_diff:.4f} (max {max_chif_diff:.4f})"
        )
    return np.vstack([k, e]), mf_diff, ch_diff


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    rows = parse_diag_rows(args.diag_csv)
    for n in (0, 1, 2, 3):
        row = get_preferred_row(rows, n)
        if row is None:
            raise RuntimeError(f"missing diagnostics row for N={n} (need emcee or kombine)")
        ok, reason = check_row_health(row, args.min_acceptance)
        if not ok:
            raise RuntimeError(f"sampler health check failed for N={n} {row.get('sampler','unknown')}: {reason}")
    if args.n3_source == "consistency":
        row_e = get_row(rows, 3, "emcee")
        if row_e is None:
            raise RuntimeError("N=3 consistency mode requires emcee diagnostics row")
        ok, reason = check_row_health(row_e, args.min_acceptance)
        if not ok:
            raise RuntimeError(f"sampler health check failed for N=3 emcee: {reason}")

    styles = {
        0: {"color": "#1f77b4", "ls": "-", "label": "N=0"},
        1: {"color": "#6f2da8", "ls": "--", "label": "N=1"},
        2: {"color": "#d4a017", "ls": "--", "label": "N=2"},
        3: {"color": "#d62728", "ls": "-", "label": "N=3"},
    }

    truth_mf = 68.546372
    truth_chif = 0.692085

    mf_grid = np.linspace(args.mf_min, args.mf_max, args.kde_grid)
    ch_grid = np.linspace(args.chi_min, args.chi_max, args.kde_grid)
    mf_mesh, ch_mesh = np.meshgrid(mf_grid, ch_grid, indexing="xy")
    eval_xy = np.vstack([mf_mesh.ravel(), ch_mesh.ravel()])

    pdf2d_by_n: dict[int, np.ndarray] = {}
    post_mf_by_n: dict[int, np.ndarray] = {}
    post_ch_by_n: dict[int, np.ndarray] = {}
    level90_by_n: dict[int, float] = {}

    n3_consistency_mf_diff = float("nan")
    n3_consistency_chif_diff = float("nan")
    for n in (0, 1, 2, 3):
        if n == 3 and args.n3_source == "consistency":
            flat, n3_consistency_mf_diff, n3_consistency_chif_diff = load_n3_with_consistency(
                args.samples_prefix, args.n3_max_mf_diff, args.n3_max_chif_diff
            )
        else:
            flat = load_samples(args.samples_prefix, n, args.n3_source)
        core = flat[:, :2]
        core = core[np.isfinite(core).all(axis=1)]
        core = core[(core[:, 0] >= args.mf_min) & (core[:, 0] <= args.mf_max)]
        core = core[(core[:, 1] >= args.chi_min) & (core[:, 1] <= args.chi_max)]
        if core.shape[0] < 500:
            raise RuntimeError(f"too few filtered samples for N={n}")
        core = maybe_subsample(core, args.max_kde_samples, rng)

        kde2 = gaussian_kde(core.T, bw_method="scott")
        # 仅用于可视化平滑；不改变采样结果本身，也不参与参数推断。
        pdf2d = kde2(eval_xy).reshape(args.kde_grid, args.kde_grid)
        pdf2d = np.clip(pdf2d, 0.0, None)
        pdf2d /= np.sum(pdf2d)
        pdf2d_by_n[n] = pdf2d
        level90_by_n[n] = credible_level_2d(pdf2d, cred=0.9)

        kde_mf = gaussian_kde(core[:, 0], bw_method="scott")
        p_mf = kde_mf(mf_grid)
        p_mf = np.clip(p_mf, 0.0, None)
        p_mf /= np.trapezoid(p_mf, mf_grid)
        post_mf_by_n[n] = p_mf

        kde_ch = gaussian_kde(core[:, 1], bw_method="scott")
        p_ch = kde_ch(ch_grid)
        p_ch = np.clip(p_ch, 0.0, None)
        p_ch /= np.trapezoid(p_ch, ch_grid)
        post_ch_by_n[n] = p_ch

    fig = plt.figure(figsize=(9.2, 8.4))
    gs = fig.add_gridspec(4, 4, hspace=0.05, wspace=0.05)
    ax_top = fig.add_subplot(gs[0, 0:3])
    ax_main = fig.add_subplot(gs[1:4, 0:3], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

    handles: list[mlines.Line2D] = []
    for n in (0, 1, 2, 3):
        st = styles[n]
        ax_main.contour(
            mf_grid,
            ch_grid,
            pdf2d_by_n[n],
            levels=[level90_by_n[n]],
            colors=[st["color"]],
            linestyles=[st["ls"]],
            linewidths=2.2,
        )
        ax_top.plot(mf_grid, post_mf_by_n[n], color=st["color"], ls=st["ls"], lw=2.0)
        ax_right.plot(post_ch_by_n[n], ch_grid, color=st["color"], ls=st["ls"], lw=2.0)
        handles.append(mlines.Line2D([], [], color=st["color"], ls=st["ls"], lw=2.2, label=st["label"]))

    ax_main.axvline(truth_mf, color="k", ls=":", lw=1.2)
    ax_main.axhline(truth_chif, color="k", ls=":", lw=1.2)
    ax_main.plot(truth_mf, truth_chif, marker="+", color="k", ms=10, mew=1.4)
    ax_main.set_xlabel(r"$M_f\ [M_\odot]$")
    ax_main.set_ylabel(r"$\chi_f$")
    ax_main.set_xlim(args.mf_min, args.mf_max)
    ax_main.set_ylim(args.chi_min, args.chi_max)
    ax_main.grid(True, alpha=0.18)
    ax_main.legend(handles=handles, loc="upper left", fontsize=10, frameon=True)

    ax_top.set_ylabel("Posterior")
    ax_top.grid(True, alpha=0.18)
    ax_top.tick_params(axis="x", labelbottom=False)

    ax_right.set_xlabel("Posterior")
    ax_right.grid(True, alpha=0.18)
    ax_right.tick_params(axis="y", labelleft=False)
    ax_top.set_title(r"Fig.10-style posteriors (smoothed KDE, $\Delta t_0=0$ ms)")

    fig.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.output_dir / f"{args.output_stem}.png"
    out_pdf = args.output_dir / f"{args.output_stem}.pdf"
    out_meta = args.output_dir / f"{args.output_stem}_meta.txt"
    out_diag = args.output_dir / f"{args.output_stem}_diag.csv"

    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf)
    if args.diag_csv.exists():
        shutil.copyfile(args.diag_csv, out_diag)
    out_meta.write_text(
        "\n".join(
            [
                f"samples_prefix={args.samples_prefix}",
                f"diag_csv={args.diag_csv}",
                f"n3_source={args.n3_source}",
                f"min_acceptance={args.min_acceptance}",
                f"n3_max_mf_diff={args.n3_max_mf_diff}",
                f"n3_max_chif_diff={args.n3_max_chif_diff}",
                f"n3_consistency_mf_diff={n3_consistency_mf_diff}",
                f"n3_consistency_chif_diff={n3_consistency_chif_diff}",
                f"mf_range=[{args.mf_min}, {args.mf_max}]",
                f"chi_range=[{args.chi_min}, {args.chi_max}]",
                f"kde_grid={args.kde_grid}",
                f"max_kde_samples={args.max_kde_samples}",
                "contour_credible=0.9",
                f"truth_mf={truth_mf}",
                f"truth_chif={truth_chif}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"output_png={out_png}")
    print(f"output_pdf={out_pdf}")
    print(f"output_diag_copy={out_diag if args.diag_csv.exists() else 'skipped'}")
    print(f"output_meta={out_meta}")


if __name__ == "__main__":
    main()
