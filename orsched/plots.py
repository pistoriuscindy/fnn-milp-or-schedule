# -*- coding: utf-8 -*-
"""
Comparison plots. Every function takes a DataFrame with one row per
(dataset, method, runtime, alpha), as produced by orsched.results.results_table().

"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from orsched import config

sns.set_style("whitegrid")
PALETTE = dict(zip(config.METHODS, sns.color_palette("tab10", len(config.METHODS))))


def _color(method):
    return PALETTE.get(method, "gray")


def _grid(n):
    """Figure with n panels, laid out as square as possible."""
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.5 * nrows), squeeze=False)
    axes = axes.flatten()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes


def _methods(df):
    present = set(df["method"])
    return [m for m in config.METHODS if m in present] + sorted(present - set(config.METHODS))


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)


def grouped_bars(df, value, ylabel, title, path, mark_best=None, target_line=None):
    """One panel per dataset, bars per method grouped by alpha.
    mark_best: function(group DataFrame) -> index of the best row (gets a gold star).
    target_line: function(alpha) -> y value of a red dashed target line."""
    datasets = sorted(df["dataset"].unique())
    methods = _methods(df)
    alphas = sorted(df["alpha"].unique())
    width = 0.8 / len(methods)
    x = np.arange(len(alphas))
    fig, axes = _grid(len(datasets))

    for ax, dataset in zip(axes, datasets):
        sub = df[df["dataset"] == dataset]
        for j, method in enumerate(methods):
            s = sub[sub["method"] == method].set_index("alpha").reindex(alphas)
            ax.bar(x + j * width, s[value], width=width, color=_color(method))
        if target_line is not None:
            for k, a in enumerate(alphas):
                ax.plot([x[k] - 0.05, x[k] + width * len(methods) + 0.05], [target_line(a)] * 2,
                        color="red", linestyle="--", linewidth=1.2)
        if mark_best is not None:
            for k, a in enumerate(alphas):
                group = sub[(sub["alpha"] == a) & sub[value].notna()]
                if group.empty:
                    continue
                best = mark_best(group)
                j = methods.index(group.loc[best, "method"])
                ax.annotate("*", (x[k] + j * width, group.loc[best, value]), fontsize=22,
                            color="gold", ha="center", va="bottom", fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels([str(a) for a in alphas])
        ax.set_title(dataset)
        ax.set_xlabel("alpha")
        ax.set_ylabel(ylabel)

    handles = [plt.Rectangle((0, 0), 1, 1, color=_color(m)) for m in methods]
    labels = list(methods)
    if target_line is not None:
        handles.append(plt.Line2D([0], [0], color="red", linestyle="--"))
        labels.append("target (= alpha)")
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.05),
               frameon=False)
    fig.suptitle(title, y=1.02)
    _save(fig, path)


def objective_vs_alpha(df, path):
    """Objective vs alpha per dataset; one line style per runtime."""
    datasets = sorted(df["dataset"].unique())
    methods = _methods(df)
    runtimes = sorted(df["runtime"].unique())
    styles = ["--", "-", "-.", ":"]
    fig, axes = _grid(len(datasets))
    for ax, dataset in zip(axes, datasets):
        sub = df[df["dataset"] == dataset]
        for method in methods:
            for j, rt in enumerate(runtimes):
                s = sub[(sub["method"] == method) & (sub["runtime"] == rt)].sort_values("alpha")
                ax.plot(s["alpha"], s["objective"], linestyle=styles[j % 4], marker="o", color=_color(method))
        ax.set_title(dataset)
        ax.set_xlabel("alpha")
        ax.set_ylabel("Objective value")
    handles = [plt.Line2D([0], [0], color=_color(m), lw=2) for m in methods]
    handles += [plt.Line2D([0], [0], color="gray", lw=2, linestyle=styles[j % 4]) for j in range(len(runtimes))]
    labels = methods + [f"{rt // 60} min" for rt in runtimes]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.05), frameon=False)
    fig.suptitle("Objective value vs. alpha", y=1.02)
    _save(fig, path)


def gap_heatmap(df, path, runtime):
    methods = _methods(df)
    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 4), sharey=True, squeeze=False)
    vmax = max(df["gap"].max() * 100, 1e-6) if df["gap"].notna().any() else 1.0
    for i, (ax, method) in enumerate(zip(axes[0], methods)):
        sub = df[(df["method"] == method) & (df["runtime"] == runtime)]
        if sub.empty:
            ax.set_title(f"{method}\n(no data)")
            ax.axis("off")
            continue
        pivot = sub.pivot_table(index="dataset", columns="alpha", values="gap").reindex(datasets) * 100
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Reds", vmin=0, vmax=vmax, ax=ax,
                    cbar=(i == len(methods) - 1), cbar_kws={"label": "MIP gap (%)"})
        ax.set_title(method)
    fig.suptitle(f"Final MIP gap (%) at {runtime // 60}-min runtime (0 = optimal)", y=1.05)
    _save(fig, path)


def build_time(df, path, runtime, alpha):
    methods = _methods(df)
    datasets = sorted(df["dataset"].unique())
    sub = df[(df["runtime"] == runtime) & np.isclose(df["alpha"], alpha)]
    width = 0.8 / len(methods)
    x = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(max(11, 2.5 * len(datasets)), 6))
    for j, method in enumerate(methods):
        s = sub[sub["method"] == method].set_index("dataset").reindex(datasets)
        ax.bar(x + j * width, s["build_time"], width=width, color=_color(method), label=method)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("Build time (s)")
    ax.set_title(f"Model build time (runtime {runtime // 60} min, alpha {alpha})")
    ax.legend()
    _save(fig, path)


def all_plots(df, output_dir):
    """Generates every comparison plot into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = df[df["has_solution"].fillna(False).astype(bool)]
    if df.empty:
        print("No solved runs to plot.")
        return
    runtimes = sorted(df["runtime"].unique())
    alphas = sorted(df["alpha"].unique())
    alpha_mid = alphas[len(alphas) // 2]

    objective_vs_alpha(df, output_dir / "objective_vs_alpha.png")
    for rt in runtimes:
        sub = df[df["runtime"] == rt]
        tag = f"runtime{rt}"
        gap_heatmap(df, output_dir / f"gap_heatmap_{tag}.png", rt)
        grouped_bars(sub, "objective", "Objective value", f"Objective at {rt // 60}-min runtime",
                     output_dir / f"objective_{tag}.png")
        grouped_bars(sub, "utilization_pct", "OR utilisation (%)",
                     f"OR utilisation (scheduled sample mean / capacity) at {rt // 60}-min runtime",
                     output_dir / f"utilization_{tag}.png")
        grouped_bars(sub, "overtime_component", "Expected overtime (min)",
                     f"Expected overtime (unweighted objective term) at {rt // 60}-min runtime",
                     output_dir / f"expected_overtime_{tag}.png")
        build_time(df, output_dir / f"build_time_{tag}.png", rt, alpha_mid)
    print(f"Plots saved to {output_dir}")


def example_results(seed=config.RANDOM_SEED):
    """Synthetic results table (random numbers) to try the plots without any runs."""
    rng = np.random.default_rng(seed)
    rows = []
    for dataset in config.DATASETS:
        for i, method in enumerate(config.METHODS):
            for rt in config.RUNTIMES:
                for a in config.ALPHAS:
                    duration = 1000 * (0.7 + 1.5 * a) * rng.uniform(0.9, 1.0)
                    overtime = rng.uniform(0, 50) * (0.25 / a)
                    build = rng.uniform(5, 30) * (1 + i)
                    rows.append(dict(
                        dataset=dataset, method=method, runtime=rt, alpha=a, has_solution=True,
                        duration_component=duration, priority_component=0.03 * duration,
                        overtime_component=overtime, overtime_weight=config.OVERTIME_PENALTY,
                        objective=1.03 * duration - config.OVERTIME_PENALTY * overtime,
                        build_time=build, solve_time=rt - build, total_time=rt,
                        gap=rng.uniform(0, 0.08), utilization_pct=min(100, 55 + 150 * a), infeasible=False))
    return pd.DataFrame(rows)
