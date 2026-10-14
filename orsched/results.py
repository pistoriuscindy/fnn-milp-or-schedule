# -*- coding: utf-8 -*-
"""
Collects the summary.json of every solved run into one table.

"""
import json

import numpy as np
import pandas as pd

from orsched import config
from orsched.paths import iter_runs

COLUMNS = ["dataset", "method", "runtime", "alpha", "objective", "duration_component",
           "priority_component", "overtime_component", "overtime_weight", "infeasible",
           "has_solution", "build_time", "solve_time", "total_time", "gap", "utilization_pct"]


def results_table(methods=None):
    """One row per (method, dataset, runtime, alpha). Runs without a solution
    have NaN objective terms. utilization_pct = 100 * scheduled sample mean
    duration / total OT capacity."""
    rows = []
    for method, dataset, runtime, alpha, folder in iter_runs(methods):
        with open(folder / "summary.json") as f:
            summary = json.load(f)
        utilization = np.nan
        if (folder / "ot_mean.csv").exists():
            capacity = config.ot_capacity(config.specialty_of(dataset))
            utilization = 100 * np.loadtxt(folder / "ot_mean.csv", delimiter=",").sum() / capacity.sum()
        rows.append({**summary, "dataset": dataset, "method": method, "runtime": runtime,
                     "alpha": alpha, "utilization_pct": utilization})
    df = pd.DataFrame(rows)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[COLUMNS + [c for c in df.columns if c not in COLUMNS]]
