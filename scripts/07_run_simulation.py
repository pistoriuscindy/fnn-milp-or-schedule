# -*- coding: utf-8 -*-
"""
Validates every solved schedule by simulation: the duration of each
scheduled surgery is drawn (with replacement) from the historical durations
of its surgery type, N_SIMULATIONS times. Per OT-day this gives
    p_overtime         fraction of simulations above capacity
    expected_overtime  mean overtime over the simulations above capacity, E[k - C | k > C]
which are compared with each method's own estimates.

Output in results/simulation/:
    <method>/<dataset>/runtime_<s>/alpha_<code>/   per-run CSVs, log and boxplot
    simulation_summary.csv / .xlsx                 one row per run (+ Detail sheet per OT-day)
    overview_*.png                                 overview figures per runtime

    python scripts/07_run_simulation.py

"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from orsched import config, plots
from orsched.data import duration_pools, load_types_per_slot
from orsched.excel import write_workbook
from orsched.paths import alpha_code, iter_runs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-simulations", type=int, default=config.N_SIMULATIONS)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


def simulate(types_per_slot, pools, n_sims, rng):
    """Simulated total duration per OT-day, shape (n_sims, nr_OTs, horizon)."""
    nr_OTs, horizon = len(types_per_slot), len(types_per_slot[0])
    missing = {t for row in types_per_slot for cell in row for t in cell if t not in pools}
    if missing:
        raise KeyError(f"No historical durations for surgery type(s) {sorted(missing)[:5]}.")
    totals = np.zeros((n_sims, nr_OTs, horizon))
    for o in range(nr_OTs):
        for d in range(horizon):
            for t in types_per_slot[o][d]:
                totals[:, o, d] += rng.choice(pools[t], size=n_sims)
    return totals


def statistics(totals, capacity, alpha):
    over = totals - capacity[None]
    is_over = over > 0
    n_over = is_over.sum(axis=0)
    expected = np.divide(np.where(is_over, over, 0).sum(axis=0), n_over,
                         out=np.zeros(capacity.shape), where=n_over > 0)
    return {"mean": totals.mean(axis=0), "max": totals.max(axis=0),
            "percentile": np.percentile(totals, 100 * (1 - alpha), axis=0),
            "p_overtime": is_over.mean(axis=0), "expected_overtime": expected}


def boxplot(totals, capacity, model_estimate, sim_estimate, title, path):
    """Distribution of (duration - capacity) per OT-day, with the simulated
    and the model's expected overtime as markers."""
    nr_OTs, horizon = capacity.shape
    labels = [f"OT{o + 1}\nD{d + 1}" for o in range(nr_OTs) for d in range(horizon)]
    data = [totals[:, o, d] - capacity[o, d] for o in range(nr_OTs) for d in range(horizon)]
    x = np.arange(1, len(labels) + 1)
    fig, ax = plt.subplots(figsize=(max(10, 0.9 * len(labels)), 6))
    ax.boxplot(data, tick_labels=labels, patch_artist=True, boxprops=dict(facecolor="#a6cee3", alpha=0.7))
    ax.axhline(0, color="black", linestyle="--", linewidth=1, label="capacity")
    ax.scatter(x, sim_estimate.ravel(), marker="D", color="green", zorder=5,
               label="simulated expected overtime")
    if model_estimate is not None:
        ax.scatter(x, model_estimate.ravel(), marker="D", color="red", zorder=5,
                   label="model's expected overtime")
    ax.set_ylabel("duration - capacity (min)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def load_optional(path):
    return np.loadtxt(path, delimiter=",", ndmin=2) if path.exists() else None


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    pools_cache, summary_rows, detail_rows = {}, [], []

    for method, dataset, runtime, alpha, folder in iter_runs():
        if not (folder / "surgery_types_ot.csv").exists():
            continue  # run without a solution
        specialty = config.specialty_of(dataset)
        if specialty not in pools_cache:
            pools_cache[specialty] = duration_pools(specialty)
        capacity = config.ot_capacity(specialty)
        open_mask = capacity > 0

        totals = simulate(load_types_per_slot(folder / "surgery_types_ot.csv"),
                          pools_cache[specialty], args.n_simulations, rng)
        sim = statistics(totals, capacity, alpha)
        model_eo = load_optional(folder / "ot_expected_overtime_model.csv")
        model_p = load_optional(folder / "ot_overtime_prob.csv")

        out = config.SIMULATION_DIR / method / dataset / f"runtime_{runtime}" / f"alpha_{alpha_code(alpha)}"
        out.mkdir(parents=True, exist_ok=True)
        for name, values in sim.items():
            np.savetxt(out / f"ot_{name}_simulated.csv", values, delimiter=",")
        boxplot(totals, capacity, model_eo, sim["expected_overtime"],
                f"{method}, {dataset}, alpha={alpha}, runtime={runtime}s", out / "boxplot.png")

        violating = sim["expected_overtime"][sim["expected_overtime"] > 0]
        summary_rows.append({
            "dataset": dataset, "method": method, "runtime": runtime, "alpha": alpha,
            "nr_open_slots": int(open_mask.sum()),
            "p_overtime_simulated_mean_open": sim["p_overtime"][open_mask].mean(),
            "nr_slots_exceeding_alpha": int((sim["p_overtime"] > alpha).sum()),
            "expected_overtime_simulated_mean_violating": violating.mean() if violating.size else 0.0,
            "total_expected_overtime_simulated": sim["expected_overtime"].sum(),
            "total_expected_overtime_model": model_eo.sum() if model_eo is not None else np.nan,
        })
        nr_OTs, horizon = capacity.shape
        for o in range(nr_OTs):
            for d in range(horizon):
                detail_rows.append({
                    "dataset": dataset, "method": method, "runtime": runtime, "alpha": alpha,
                    "OT": o + 1, "day": d + 1, "capacity": capacity[o, d],
                    "mean_duration_simulated": sim["mean"][o, d],
                    "percentile_duration_simulated": sim["percentile"][o, d],
                    "p_overtime_simulated": sim["p_overtime"][o, d],
                    "p_overtime_model": model_p[o, d] if model_p is not None else np.nan,
                    "expected_overtime_simulated": sim["expected_overtime"][o, d],
                    "expected_overtime_model": model_eo[o, d] if model_eo is not None else np.nan,
                })
        print(f"{method:16s} {dataset} runtime={runtime} alpha={alpha}: "
              f"{summary_rows[-1]['nr_slots_exceeding_alpha']}/{open_mask.sum()} open OT-days above alpha")

    if not summary_rows:
        print("No solved schedules found. Run scripts/05_run_milp.py first.")
        return

    summary = pd.DataFrame(summary_rows)
    summary["total_overtime_difference_sim_minus_model"] = (
        summary["total_expected_overtime_simulated"] - summary["total_expected_overtime_model"])
    detail = pd.DataFrame(detail_rows)
    detail["p_overtime_difference"] = detail["p_overtime_simulated"] - detail["p_overtime_model"]
    detail["expected_overtime_difference"] = detail["expected_overtime_simulated"] - detail["expected_overtime_model"]

    summary.to_csv(config.SIMULATION_DIR / "simulation_summary.csv", index=False)
    write_workbook(config.SIMULATION_DIR / "simulation_summary.xlsx", {"Summary": summary, "Detail": detail},
                   percent_columns={"p_overtime_simulated_mean_open", "p_overtime_simulated",
                                    "p_overtime_model", "p_overtime_difference"})

    summary["p_overtime_pct"] = 100 * summary["p_overtime_simulated_mean_open"]
    summary["calibration_gap"] = summary["total_overtime_difference_sim_minus_model"]
    for rt in sorted(summary["runtime"].unique()):
        sub = summary[summary["runtime"] == rt]
        plots.grouped_bars(
            sub, "p_overtime_pct", "Simulated overtime probability (%)",
            f"Simulated overtime probability, open OT-days (runtime {rt}s); * = closest to alpha",
            config.SIMULATION_DIR / f"overview_overtime_probability_runtime{rt}.png",
            mark_best=lambda g: (g["p_overtime_pct"] - 100 * g["alpha"]).abs().idxmin(),
            target_line=lambda a: 100 * a)
        plots.grouped_bars(
            sub.dropna(subset=["calibration_gap"]), "calibration_gap", "Simulated - model (min)",
            f"Total expected overtime: simulated - model (runtime {rt}s); * = smallest difference",
            config.SIMULATION_DIR / f"overview_calibration_gap_runtime{rt}.png",
            mark_best=lambda g: g["calibration_gap"].abs().idxmin())
    print(f"Results saved to {config.SIMULATION_DIR}")


if __name__ == "__main__":
    main()
