# -*- coding: utf-8 -*-
"""
Judges every solved schedule with the overtime-probability criterion of
every method: an open OT-day is "accepted" by a method if that method's
diagnostic gives p_overtime <= alpha.

    PLF  normal approximation, exact sqrt
    SBM  fraction of the scenarios with overtime
    FNN  probability derived from the network's percentile (heuristic,
         see orsched.milp_fnn.overtime_diagnostics); both FNN variants use
         the same network, so they share this criterion

Output: results/cross_method/cross_method_feasibility.xlsx with sheets
    Summary            per specialty, runtime, alpha, source and judging method
    Acceptance_Counts  "<accepted>/<open> (xx%)" per dataset and source method
    Avg_Overtime_Prob  mean p_overtime over the open OT-days
    <source>_by_OR     per OT-day detail for each source method

    python scripts/08_check_cross_method_feasibility.py

"""
import numpy as np
import pandas as pd

from orsched import config, milp_fnn, milp_plf, milp_sbm
from orsched.data import load_instance, load_scenarios
from orsched.excel import write_workbook
from orsched.fnn_model import load_model
from orsched.paths import iter_runs

JUDGES = ["PLF", "SBM", "FNN"]


def judge(name, x, surgeries, durations, capacity, alpha, fnn):
    if name == "PLF":
        return milp_plf.overtime_diagnostics(x, surgeries, capacity, alpha)["p_overtime"]
    if name == "SBM":
        return milp_sbm.overtime_diagnostics(x, durations, capacity)["p_overtime"]
    return milp_fnn.overtime_diagnostics(x, surgeries, capacity, alpha, fnn)["p_overtime"]


def main():
    fnn = load_model()
    data_cache = {}
    summary_rows, detail_rows = [], {}

    for source, dataset, runtime, alpha, folder in iter_runs():
        if not (folder / "x_schedule.csv").exists():
            continue
        if dataset not in data_cache:
            surgeries = load_instance(dataset)
            scen_surgeries, durations = load_scenarios(dataset)
            if len(scen_surgeries) != len(surgeries):
                raise ValueError(f"{dataset}: instance and scenario file have a different number of surgeries.")
            data_cache[dataset] = (surgeries, durations)
        surgeries, durations = data_cache[dataset]
        specialty = config.specialty_of(dataset)
        capacity = config.ot_capacity(specialty)
        nr_OTs, horizon = capacity.shape
        open_mask = capacity > 0
        x = np.loadtxt(folder / "x_schedule.csv", delimiter=",", ndmin=2).reshape(len(surgeries), horizon, nr_OTs)

        per_slot = {}
        for name in JUDGES:
            p = judge(name, x, surgeries, durations, capacity, alpha, fnn)
            accepted = (p <= alpha) & open_mask
            summary_rows.append({
                "dataset": dataset, "specialty": specialty, "runtime": runtime, "alpha": alpha,
                "source_method": source, "judged_by": name,
                "nr_open_slots": int(open_mask.sum()), "nr_accepted_open": int(accepted.sum()),
                "avg_p_overtime_open": p[open_mask].mean(),
            })
            for o in range(nr_OTs):
                for d in range(horizon):
                    row = per_slot.setdefault((o, d), {
                        "dataset": dataset, "runtime": runtime, "alpha": alpha, "OT": o + 1, "day": d + 1,
                        "capacity": capacity[o, d], "open_slot": bool(open_mask[o, d])})
                    row[f"p_overtime_{name}"] = p[o, d]
                    row[f"accepted_{name}"] = bool(p[o, d] <= alpha)
        detail_rows.setdefault(source, []).extend(per_slot.values())
        print(f"{source:16s} {dataset} runtime={runtime} alpha={alpha}: " + ", ".join(
            f"{r['judged_by']} {r['nr_accepted_open']}/{r['nr_open_slots']}" for r in summary_rows[-len(JUDGES):]))

    if not summary_rows:
        print("No solved schedules found. Run scripts/05_run_milp.py first.")
        return

    df = pd.DataFrame(summary_rows)
    keys = ["specialty", "runtime", "alpha", "source_method", "judged_by"]
    df["p_weighted"] = df["avg_p_overtime_open"] * df["nr_open_slots"]
    grouped = df.groupby(keys).agg(
        datasets=("dataset", lambda s: ", ".join(sorted(s.unique()))),
        nr_open_slots=("nr_open_slots", "sum"), nr_accepted_open=("nr_accepted_open", "sum"),
        p_weighted=("p_weighted", "sum")).reset_index()
    grouped["frac_accepted_open"] = grouped["nr_accepted_open"] / grouped["nr_open_slots"]
    grouped["avg_p_overtime_open"] = grouped.pop("p_weighted") / grouped["nr_open_slots"]

    idx = ["dataset", "runtime", "alpha", "source_method"]
    df["accepted"] = [f"{a}/{n} ({100 * a / n:.0f}%)" for a, n in zip(df["nr_accepted_open"], df["nr_open_slots"])]
    counts = df.pivot_table(index=idx, columns="judged_by", values="accepted", aggfunc="first").reset_index()
    probs = df.pivot_table(index=idx, columns="judged_by", values="avg_p_overtime_open").reset_index()

    sheets = {"Summary": grouped, "Acceptance_Counts": counts, "Avg_Overtime_Prob": probs}
    sheets.update({f"{src}_by_OR": pd.DataFrame(rows) for src, rows in detail_rows.items()})
    path = config.CROSS_METHOD_DIR / "cross_method_feasibility.xlsx"
    write_workbook(path, sheets, percent_columns={"frac_accepted_open", "avg_p_overtime_open", *JUDGES,
                                                  *[f"p_overtime_{j}" for j in JUDGES]})
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
