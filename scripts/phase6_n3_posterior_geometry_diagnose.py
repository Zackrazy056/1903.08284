from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.mixture import GaussianMixture


TRUTH_MF = 68.546372
TRUTH_CHIF = 0.692085


@dataclass(frozen=True)
class ComponentSummary:
    component: int
    is_truth_adjacent: bool
    fraction: float
    sample_count: int
    mf_q16: float
    mf_q50: float
    mf_q84: float
    chif_q16: float
    chif_q50: float
    chif_q84: float
    truth_hpd_rank: float
    truth_in_90_hpd: bool


@dataclass(frozen=True)
class ChainSummary:
    label: str
    sample_count: int
    mf_q16: float
    mf_q50: float
    mf_q84: float
    chif_q16: float
    chif_q50: float
    chif_q84: float
    truth_hpd_rank: float
    truth_in_90_hpd: bool
    truth_component_fraction: float
    chunk_truth_component_fractions: list[float]
    components: list[ComponentSummary]


@dataclass(frozen=True)
class GeometrySummary:
    n_components: int
    component_means_mf_chif: list[list[float]]
    truth_component: int
    primary: ChainSummary
    alt: ChainSummary
    js_distance: float
    bhattacharyya: float
    dominant_component_same: bool
    stable_within_chain: bool
    primary_truth_fraction_gap: float
    diagnosis: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose N=3 posterior geometry from two saved long chains.")
    p.add_argument(
        "--primary-samples",
        type=Path,
        default=Path("results/fig10_n3_strict_long_fix1_samples_N3_emcee.npz"),
    )
    p.add_argument(
        "--alt-samples",
        type=Path,
        default=Path("results/fig10_n3_strict_long_fix1_samples_N3_emcee_alt.npz"),
    )
    p.add_argument("--burn-fraction", type=float, default=0.5)
    p.add_argument("--fit-subsample-per-chain", type=int, default=6000)
    p.add_argument("--scatter-subsample-per-chain", type=int, default=3000)
    p.add_argument("--max-components", type=int, default=4)
    p.add_argument("--time-chunks", type=int, default=6)
    p.add_argument("--mf-min", type=float, default=50.0)
    p.add_argument("--mf-max", type=float, default=100.0)
    p.add_argument("--chif-min", type=float, default=0.0)
    p.add_argument("--chif-max", type=float, default=1.0)
    p.add_argument("--hist-bins", type=int, default=220)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--output", type=Path, default=Path("results/fig10_n3_geometry_fix1.png"))
    p.add_argument("--json-output", type=Path, default=Path("results/fig10_n3_geometry_fix1.json"))
    p.add_argument("--markdown-output", type=Path, default=Path("results/fig10_n3_geometry_fix1.md"))
    return p.parse_args()


def _load_post_burn_chain(path: Path, burn_fraction: float) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"sample file not found: {path}")
    z = np.load(path, allow_pickle=False)
    chain = np.asarray(z["chain"], dtype=float)
    if chain.ndim != 3 or chain.shape[-1] < 2:
        raise ValueError(f"unexpected chain shape in {path}: {chain.shape}")
    start = int(np.floor(chain.shape[0] * burn_fraction))
    start = min(max(start, 0), chain.shape[0] - 1)
    return np.asarray(chain[start:, :, :], dtype=float)


def _random_subsample(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if x.shape[0] <= n:
        return np.asarray(x, dtype=float)
    idx = rng.choice(x.shape[0], size=n, replace=False)
    return np.asarray(x[idx], dtype=float)


def _truth_hpd_rank(
    samples_2d: np.ndarray,
    *,
    mf_range: tuple[float, float],
    chif_range: tuple[float, float],
    hist_bins: int,
) -> float:
    mf = np.asarray(samples_2d[:, 0], dtype=float)
    ch = np.asarray(samples_2d[:, 1], dtype=float)
    h, mf_edges, ch_edges = np.histogram2d(
        mf,
        ch,
        bins=[hist_bins, hist_bins],
        range=[mf_range, chif_range],
        density=False,
    )
    total = float(np.sum(h))
    if total <= 0.0:
        return 1.0
    mf_bin = int(np.searchsorted(mf_edges, TRUTH_MF, side="right") - 1)
    ch_bin = int(np.searchsorted(ch_edges, TRUTH_CHIF, side="right") - 1)
    mf_bin = min(max(mf_bin, 0), h.shape[0] - 1)
    ch_bin = min(max(ch_bin, 0), h.shape[1] - 1)
    truth_density = float(h[mf_bin, ch_bin])
    return float(np.sum(h[h >= truth_density])) / total


def _summarize_quantiles(samples_2d: np.ndarray) -> tuple[float, float, float, float, float, float]:
    mf_q16, mf_q50, mf_q84 = [float(v) for v in np.quantile(samples_2d[:, 0], [0.16, 0.5, 0.84])]
    ch_q16, ch_q50, ch_q84 = [float(v) for v in np.quantile(samples_2d[:, 1], [0.16, 0.5, 0.84])]
    return mf_q16, mf_q50, mf_q84, ch_q16, ch_q50, ch_q84


def _fit_gmm(
    x_fit: np.ndarray,
    *,
    max_components: int,
    rng_seed: int,
) -> tuple[GaussianMixture, np.ndarray, np.ndarray]:
    mu = np.mean(x_fit, axis=0)
    sigma = np.std(x_fit, axis=0)
    sigma = np.where(sigma > 0.0, sigma, 1.0)
    z_fit = (x_fit - mu) / sigma

    best_model: GaussianMixture | None = None
    best_bic = float("inf")
    for k in range(1, max_components + 1):
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=rng_seed,
            n_init=10,
        )
        model.fit(z_fit)
        bic = float(model.bic(z_fit))
        if bic < best_bic:
            best_bic = bic
            best_model = model
    if best_model is None:
        raise RuntimeError("failed to fit any GaussianMixture model")
    return best_model, mu, sigma


def _predict_component(model: GaussianMixture, x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict((x - mu) / sigma), dtype=int)


def _chain_summary(
    *,
    label: str,
    chain_2d: np.ndarray,
    component_ids: np.ndarray,
    truth_component: int,
    n_components: int,
    mf_range: tuple[float, float],
    chif_range: tuple[float, float],
    hist_bins: int,
    time_chunks: int,
) -> ChainSummary:
    flat = chain_2d.reshape(-1, chain_2d.shape[-1])
    flat_comp = component_ids.reshape(-1)
    overall_rank = _truth_hpd_rank(flat, mf_range=mf_range, chif_range=chif_range, hist_bins=hist_bins)
    mf_q16, mf_q50, mf_q84, ch_q16, ch_q50, ch_q84 = _summarize_quantiles(flat)

    components: list[ComponentSummary] = []
    for comp in range(n_components):
        mask = flat_comp == comp
        x = flat[mask]
        if x.size == 0:
            continue
        rank = _truth_hpd_rank(x, mf_range=mf_range, chif_range=chif_range, hist_bins=hist_bins)
        c_mf_q16, c_mf_q50, c_mf_q84, c_ch_q16, c_ch_q50, c_ch_q84 = _summarize_quantiles(x)
        components.append(
            ComponentSummary(
                component=comp,
                is_truth_adjacent=(comp == truth_component),
                fraction=float(np.mean(mask)),
                sample_count=int(x.shape[0]),
                mf_q16=c_mf_q16,
                mf_q50=c_mf_q50,
                mf_q84=c_mf_q84,
                chif_q16=c_ch_q16,
                chif_q50=c_ch_q50,
                chif_q84=c_ch_q84,
                truth_hpd_rank=rank,
                truth_in_90_hpd=bool(rank <= 0.9),
            )
        )

    steps = chain_2d.shape[0]
    chunk_truth_fracs: list[float] = []
    for i in range(time_chunks):
        s0 = i * steps // time_chunks
        s1 = (i + 1) * steps // time_chunks
        sub = component_ids[s0:s1].reshape(-1)
        chunk_truth_fracs.append(float(np.mean(sub == truth_component)))

    return ChainSummary(
        label=label,
        sample_count=int(flat.shape[0]),
        mf_q16=mf_q16,
        mf_q50=mf_q50,
        mf_q84=mf_q84,
        chif_q16=ch_q16,
        chif_q50=ch_q50,
        chif_q84=ch_q84,
        truth_hpd_rank=overall_rank,
        truth_in_90_hpd=bool(overall_rank <= 0.9),
        truth_component_fraction=float(np.mean(flat_comp == truth_component)),
        chunk_truth_component_fractions=chunk_truth_fracs,
        components=components,
    )


def _hist_pdf(
    samples_2d: np.ndarray,
    *,
    mf_range: tuple[float, float],
    chif_range: tuple[float, float],
    hist_bins: int,
) -> np.ndarray:
    h, _, _ = np.histogram2d(
        samples_2d[:, 0],
        samples_2d[:, 1],
        bins=[hist_bins, hist_bins],
        range=[mf_range, chif_range],
        density=False,
    )
    p = h.astype(float).ravel()
    total = float(np.sum(p))
    if total <= 0.0:
        return p
    return p / total


def _credible_level_2d(pdf: np.ndarray, cred: float = 0.9) -> float:
    flat = pdf.ravel()
    order = np.argsort(flat)[::-1]
    cdf = np.cumsum(flat[order])
    idx = int(np.searchsorted(cdf, cred, side="left"))
    idx = min(max(idx, 0), order.size - 1)
    return float(flat[order[idx]])


def _diagnosis_text(summary: GeometrySummary) -> str:
    primary_dominant = max(summary.primary.components, key=lambda c: c.fraction).component
    alt_dominant = max(summary.alt.components, key=lambda c: c.fraction).component
    if (
        summary.js_distance > 0.5
        and not summary.dominant_component_same
        and summary.stable_within_chain
        and summary.primary_truth_fraction_gap > 0.2
    ):
        return (
            "The main blocker is posterior multimodality plus poor inter-mode mixing. "
            "Each long chain is internally stable, but they allocate very different weight to the "
            f"truth-adjacent component (primary dominant={primary_dominant}, alt dominant={alt_dominant}). "
            "This is stronger evidence for sampler / posterior-geometry trouble than for a simple single-mode "
            "signal-definition bug."
        )
    if summary.js_distance > 0.5:
        return (
            "The two chains still disagree strongly in remnant-posterior geometry. "
            "There may be both sampling and modeling issues remaining."
        )
    return "The two chains are geometrically similar enough that mode-mixing is not the dominant remaining issue."


def _build_markdown(summary: GeometrySummary) -> str:
    lines: list[str] = []
    lines.append("# N=3 Posterior Geometry Diagnosis")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(summary.diagnosis)
    lines.append("")
    lines.append("## Global Metrics")
    lines.append("")
    lines.append(f"- Chosen GMM components: `{summary.n_components}`")
    lines.append(f"- Truth-adjacent component id: `{summary.truth_component}`")
    lines.append(f"- Component means `(Mf, chif)`: `{summary.component_means_mf_chif}`")
    lines.append(f"- Chain JS distance: `{summary.js_distance:.4f}`")
    lines.append(f"- Chain Bhattacharyya coefficient: `{summary.bhattacharyya:.4f}`")
    lines.append(f"- Dominant component same across chains: `{summary.dominant_component_same}`")
    lines.append(f"- Within-chain chunk stability pass: `{summary.stable_within_chain}`")
    lines.append("")
    for chain in [summary.primary, summary.alt]:
        lines.append(f"## {chain.label.capitalize()} Chain")
        lines.append("")
        lines.append(
            f"- Overall q50: `Mf={chain.mf_q50:.3f}`, `chif={chain.chif_q50:.3f}`, "
            f"`truth_hpd_rank={chain.truth_hpd_rank:.4f}`"
        )
        lines.append(
            f"- Truth-adjacent component weight: `{chain.truth_component_fraction:.4f}`; "
            f"chunk fractions: `{[round(v, 3) for v in chain.chunk_truth_component_fractions]}`"
        )
        lines.append("")
        lines.append("| Component | Truth-adjacent | Weight | Mf q50 | chif q50 | Truth HPD rank |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for comp in chain.components:
            lines.append(
                f"| {comp.component} | {comp.is_truth_adjacent} | {comp.fraction:.4f} | "
                f"{comp.mf_q50:.3f} | {comp.chif_q50:.3f} | {comp.truth_hpd_rank:.4f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _plot_summary(
    *,
    output: Path,
    primary_flat: np.ndarray,
    alt_flat: np.ndarray,
    primary_chunk_fracs: list[float],
    alt_chunk_fracs: list[float],
    primary_comp_fracs: list[float],
    alt_comp_fracs: list[float],
    comp_means: np.ndarray,
    truth_component: int,
    scatter_primary: np.ndarray,
    scatter_alt: np.ndarray,
    scatter_primary_comp: np.ndarray,
    scatter_alt_comp: np.ndarray,
    mf_range: tuple[float, float],
    chif_range: tuple[float, float],
) -> None:
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    truth_color = colors[truth_component % len(colors)]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    ax = axes[0, 0]
    for comp in np.unique(scatter_primary_comp):
        m = scatter_primary_comp == comp
        ax.scatter(
            scatter_primary[m, 0],
            scatter_primary[m, 1],
            s=6,
            alpha=0.22,
            color=colors[int(comp) % len(colors)],
            label=f"primary c{int(comp)}",
        )
    for comp in np.unique(scatter_alt_comp):
        m = scatter_alt_comp == comp
        ax.scatter(
            scatter_alt[m, 0],
            scatter_alt[m, 1],
            s=6,
            alpha=0.22,
            marker="x",
            color=colors[int(comp) % len(colors)],
            label=f"alt c{int(comp)}",
        )
    ax.scatter(TRUTH_MF, TRUTH_CHIF, s=80, marker="*", color="k", label="truth")
    ax.scatter(comp_means[:, 0], comp_means[:, 1], s=60, marker="D", color="none", edgecolor="k")
    ax.set_title("Second-half Samples By GMM Component")
    ax.set_xlabel(r"$M_f\ [M_\odot]$")
    ax.set_ylabel(r"$\chi_f$")
    ax.set_xlim(*mf_range)
    ax.set_ylim(*chif_range)
    ax.grid(True, alpha=0.15)

    ax = axes[0, 1]
    chunks = np.arange(1, len(primary_chunk_fracs) + 1)
    ax.plot(chunks, primary_chunk_fracs, "o-", color="#d62728", lw=2.0, label="primary")
    ax.plot(chunks, alt_chunk_fracs, "s--", color="#1f77b4", lw=2.0, label="alt")
    ax.axhline(0.5, color="k", ls=":", lw=1.0)
    ax.set_title("Truth-adjacent Mode Fraction By Time Chunk")
    ax.set_xlabel("Chunk")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.15)
    ax.legend()

    ax = axes[1, 0]
    idx = np.arange(len(primary_comp_fracs))
    width = 0.36
    ax.bar(idx - width / 2, primary_comp_fracs, width=width, color="#d62728", alpha=0.8, label="primary")
    ax.bar(idx + width / 2, alt_comp_fracs, width=width, color="#1f77b4", alpha=0.8, label="alt")
    ax.bar(truth_component, max(primary_comp_fracs[truth_component], alt_comp_fracs[truth_component]), width=0.9, fill=False, edgecolor=truth_color, linewidth=2.0)
    ax.set_title("Mode Weights")
    ax.set_xlabel("Component")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.15)
    ax.legend()

    ax = axes[1, 1]
    h_primary, xedges, yedges = np.histogram2d(
        primary_flat[:, 0], primary_flat[:, 1], bins=[180, 180], range=[mf_range, chif_range], density=False
    )
    h_alt, _, _ = np.histogram2d(
        alt_flat[:, 0], alt_flat[:, 1], bins=[180, 180], range=[mf_range, chif_range], density=False
    )
    pdf_primary = h_primary.T / np.sum(h_primary)
    pdf_alt = h_alt.T / np.sum(h_alt)
    xcent = 0.5 * (xedges[:-1] + xedges[1:])
    ycent = 0.5 * (yedges[:-1] + yedges[1:])
    ax.contour(xcent, ycent, pdf_primary, levels=[_credible_level_2d(pdf_primary, 0.9)], colors=["#d62728"], linewidths=2.0)
    ax.contour(xcent, ycent, pdf_alt, levels=[_credible_level_2d(pdf_alt, 0.9)], colors=["#1f77b4"], linestyles=["--"], linewidths=2.0)
    ax.scatter(TRUTH_MF, TRUTH_CHIF, s=80, marker="*", color="k")
    ax.set_title("Second-half 90% Contours")
    ax.set_xlabel(r"$M_f\ [M_\odot]$")
    ax.set_ylabel(r"$\chi_f$")
    ax.set_xlim(*mf_range)
    ax.set_ylim(*chif_range)
    ax.grid(True, alpha=0.15)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.burn_fraction < 1.0):
        raise ValueError("--burn-fraction must be in [0, 1)")
    if args.max_components < 1:
        raise ValueError("--max-components must be >= 1")
    if args.time_chunks < 2:
        raise ValueError("--time-chunks must be >= 2")

    rng = np.random.default_rng(args.seed)
    mf_range = (args.mf_min, args.mf_max)
    chif_range = (args.chif_min, args.chif_max)

    primary_chain = _load_post_burn_chain(args.primary_samples, args.burn_fraction)
    alt_chain = _load_post_burn_chain(args.alt_samples, args.burn_fraction)
    primary_flat = primary_chain[:, :, :2].reshape(-1, 2)
    alt_flat = alt_chain[:, :, :2].reshape(-1, 2)

    x_fit = np.vstack(
        [
            _random_subsample(primary_flat, args.fit_subsample_per_chain, rng),
            _random_subsample(alt_flat, args.fit_subsample_per_chain, rng),
        ]
    )
    model, mu, sigma = _fit_gmm(x_fit, max_components=args.max_components, rng_seed=args.seed)
    comp_means = np.asarray(model.means_ * sigma + mu, dtype=float)
    truth_dist = ((comp_means[:, 0] - TRUTH_MF) / 72.0) ** 2 + (comp_means[:, 1] - TRUTH_CHIF) ** 2
    truth_component = int(np.argmin(truth_dist))

    primary_comp = _predict_component(model, primary_flat, mu, sigma).reshape(primary_chain.shape[0], primary_chain.shape[1])
    alt_comp = _predict_component(model, alt_flat, mu, sigma).reshape(alt_chain.shape[0], alt_chain.shape[1])

    primary_summary = _chain_summary(
        label="primary",
        chain_2d=primary_chain[:, :, :2],
        component_ids=primary_comp,
        truth_component=truth_component,
        n_components=model.n_components,
        mf_range=mf_range,
        chif_range=chif_range,
        hist_bins=args.hist_bins,
        time_chunks=args.time_chunks,
    )
    alt_summary = _chain_summary(
        label="alt",
        chain_2d=alt_chain[:, :, :2],
        component_ids=alt_comp,
        truth_component=truth_component,
        n_components=model.n_components,
        mf_range=mf_range,
        chif_range=chif_range,
        hist_bins=args.hist_bins,
        time_chunks=args.time_chunks,
    )

    p1 = _hist_pdf(primary_flat, mf_range=mf_range, chif_range=chif_range, hist_bins=180)
    p2 = _hist_pdf(alt_flat, mf_range=mf_range, chif_range=chif_range, hist_bins=180)
    js_distance = float(jensenshannon(p1, p2))
    bhattacharyya = float(np.sum(np.sqrt(p1 * p2)))

    primary_comp_fracs = [0.0] * model.n_components
    alt_comp_fracs = [0.0] * model.n_components
    for comp in primary_summary.components:
        primary_comp_fracs[comp.component] = comp.fraction
    for comp in alt_summary.components:
        alt_comp_fracs[comp.component] = comp.fraction

    primary_dominant = int(np.argmax(primary_comp_fracs))
    alt_dominant = int(np.argmax(alt_comp_fracs))
    stable_within_chain = (
        (max(primary_summary.chunk_truth_component_fractions) - min(primary_summary.chunk_truth_component_fractions) < 0.18)
        and (max(alt_summary.chunk_truth_component_fractions) - min(alt_summary.chunk_truth_component_fractions) < 0.18)
    )

    summary = GeometrySummary(
        n_components=int(model.n_components),
        component_means_mf_chif=[[float(v) for v in row] for row in comp_means],
        truth_component=truth_component,
        primary=primary_summary,
        alt=alt_summary,
        js_distance=js_distance,
        bhattacharyya=bhattacharyya,
        dominant_component_same=bool(primary_dominant == alt_dominant),
        stable_within_chain=bool(stable_within_chain),
        primary_truth_fraction_gap=float(abs(primary_summary.truth_component_fraction - alt_summary.truth_component_fraction)),
        diagnosis="",
    )
    summary = replace(summary, diagnosis=_diagnosis_text(summary))

    scatter_primary = _random_subsample(primary_flat, args.scatter_subsample_per_chain, rng)
    scatter_alt = _random_subsample(alt_flat, args.scatter_subsample_per_chain, rng)
    scatter_primary_comp = _predict_component(model, scatter_primary, mu, sigma)
    scatter_alt_comp = _predict_component(model, scatter_alt, mu, sigma)

    _plot_summary(
        output=args.output,
        primary_flat=primary_flat,
        alt_flat=alt_flat,
        primary_chunk_fracs=primary_summary.chunk_truth_component_fractions,
        alt_chunk_fracs=alt_summary.chunk_truth_component_fractions,
        primary_comp_fracs=primary_comp_fracs,
        alt_comp_fracs=alt_comp_fracs,
        comp_means=comp_means,
        truth_component=truth_component,
        scatter_primary=scatter_primary,
        scatter_alt=scatter_alt,
        scatter_primary_comp=scatter_primary_comp,
        scatter_alt_comp=scatter_alt_comp,
        mf_range=mf_range,
        chif_range=chif_range,
    )

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(_build_markdown(summary), encoding="utf-8")

    print(f"n_components={summary.n_components}")
    print(f"truth_component={summary.truth_component}")
    print(f"component_means={summary.component_means_mf_chif}")
    print(f"primary_truth_component_fraction={summary.primary.truth_component_fraction:.6f}")
    print(f"alt_truth_component_fraction={summary.alt.truth_component_fraction:.6f}")
    print(f"js_distance={summary.js_distance:.6f}")
    print(f"bhattacharyya={summary.bhattacharyya:.6f}")
    print(f"diagnosis={summary.diagnosis}")
    print(f"output={args.output}")
    print(f"json_output={args.json_output}")
    print(f"markdown_output={args.markdown_output}")


if __name__ == "__main__":
    main()
