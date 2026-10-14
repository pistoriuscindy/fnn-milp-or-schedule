# -*- coding: utf-8 -*-
"""
Collects all solved runs into results/analysis/results.csv, checks them for
inconsistencies, makes the comparison plots and an overtime summary per
method and alpha.

    python scripts/06_analyze_results.py
    python scripts/06_analyze_results.py --demo     (random example numbers, no runs needed)

"""
import argparse

import numpy as np
import pandas as pd

from orsched import config, plots
from orsched.results import results_table


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", action="store_true", help="Use random example results.")
    return p.parse_args()


def check_consistency(df):
    """Warn about runs that make a comparison between methods misleading."""
    no_solution = df[~df["has_solution"].fillna(False).astype(bool)]
    if not no_solution.empty:
        print(f"Warning: {len(no_solution)} run(s) have no solution (infeasible or no "
              f"solution within the time limit):")
        print(no_solution[["dataset", "method", "runtime", "alpha"]].to_string(index=False))
    weights = df["overtime_weight"].dropna().unique()
    if len(weights) > 1:
        print(f"Warning: different overtime weights across runs: {sorted(weights)}.")
    not_optimal = df[df["gap"] > 0]
    if not not_optimal.empty:
        print(f"Note: {len(not_optimal)} run(s) stopped with a gap > 0 (time limit reached).")


def overtime_summary(df):
    """Mean expected overtime (unweighted and weighted) per method and alpha,
    and the weighted overtime as % of duration + priority."""
    valid = df.dropna(subset=["overtime_component"]).copy()
    valid["overtime_cost"] = valid["overtime_weight"] * valid["overtime_component"]
    valid["positive_value"] = valid["duration_component"] + valid["priority_component"]
    out = valid.groupby(["method", "alpha"]).agg(
        overtime_component_mean=("overtime_component", "mean"),
        overtime_cost_mean=("overtime_cost", "mean"),
        positive_value_mean=("positive_value", "mean"),
        n_runs=("overtime_component", "size")).reset_index()
    out["overtime_share_of_obj_pct"] = np.where(
        out["positive_value_mean"] > 0, 100 * out["overtime_cost_mean"] / out["positive_value_mean"], np.nan)
    return out.drop(columns="positive_value_mean")


def main():
    args = parse_args()
    out = config.ANALYSIS_DIR / ("demo" if args.demo else "")
    out.mkdir(parents=True, exist_ok=True)

    df = plots.example_results() if args.demo else results_table()
    if df.empty:
        print(f"No runs found under {config.MILP_RESULTS_DIR}. Run scripts/05_run_milp.py first.")
        return
    df.to_csv(out / "results.csv", index=False)
    print(f"{len(df)} runs saved to {out / 'results.csv'}")

    check_consistency(df)
    plots.all_plots(df, out / "plots")

    summary = overtime_summary(df)
    summary.to_csv(out / "overtime_summary.csv", index=False)
    with pd.option_context("display.width", 200):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
