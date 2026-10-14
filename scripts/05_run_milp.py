# -*- coding: utf-8 -*-
"""
Solves every combination of method x dataset x runtime x alpha (defaults in
orsched/config.py) and saves each run to
    results/milp/<method>/<dataset>/runtime_<s>/alpha_<code>/

Files per run:
    summary.json                     objective terms, times, gap, status
    gurobi.log                       solver log
    x_schedule.csv                   x[s, day, ot], reshaped to n_surgeries x (days * OTs)
    surgery_types_ot.csv             surgery types per OT (rows) and day (columns)
    ot_mean.csv                      total sample mean duration per OT-day
    ot_expected_overtime_model.csv   the method's own expected-overtime term per OT-day
    ot_overtime_prob.csv             the method's own overtime probability per OT-day
    + method-specific extras

    python scripts/05_run_milp.py
    python scripts/05_run_milp.py --methods PLF SBM --datasets dataset_ENT_1 --runtimes 300 --alphas 0.1

"""
import argparse
import itertools

import numpy as np

from orsched import config, milp_fnn, milp_plf, milp_sbm
from orsched.data import load_instance, load_scenarios, save_types_per_slot
from orsched.milp_common import write_summary
from orsched.paths import run_dir


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", default=config.METHODS, choices=config.METHODS)
    p.add_argument("--datasets", nargs="+", default=config.DATASETS)
    p.add_argument("--runtimes", nargs="+", type=int, default=config.RUNTIMES)
    p.add_argument("--alphas", nargs="+", type=float, default=config.ALPHAS)
    return p.parse_args()


def solve_one(method, surgeries, durations, capacity, alpha, runtime, folder, shared):
    log = folder / "gurobi.log"
    if method in ("FNN_implicit", "FNN_explicit_lp"):
        embedding = "implicit" if method == "FNN_implicit" else "explicit_lp"
        result = milp_fnn.solve(surgeries, capacity, alpha, shared["fnn"], embedding, runtime, log)
        diag = lambda xs: milp_fnn.overtime_diagnostics(xs, surgeries, capacity, alpha, shared["fnn"])
    elif method == "PLF":
        result = milp_plf.solve(surgeries, capacity, alpha, shared["breakpoints"], runtime, log)
        diag = lambda xs: milp_plf.overtime_diagnostics(xs, surgeries, capacity, alpha)
    else:
        result = milp_sbm.solve(surgeries, durations, capacity, alpha, runtime, log)
        diag = lambda xs: milp_sbm.overtime_diagnostics(xs, durations, capacity)
    return result, diag


def save_run(result, diag, surgeries, capacity, folder):
    """Writes all per-run outputs; returns a one-line description."""
    write_summary(result.summary, folder / "summary.json")
    if result.x is None:
        return f"no solution (Gurobi status {result.summary['status']})"

    x = result.x
    nr_OTs, horizon = capacity.shape
    sample_mean = surgeries["sample_mean"].to_numpy(dtype=float)
    types = surgeries["surgery_type"].to_numpy()

    np.savetxt(folder / "x_schedule.csv", x.reshape(len(x), -1), delimiter=",", fmt="%d")
    ot_mean = np.array([[sample_mean @ x[:, d, o] for d in range(horizon)] for o in range(nr_OTs)])
    np.savetxt(folder / "ot_mean.csv", ot_mean, delimiter=",")
    save_types_per_slot([[list(types[x[:, d, o] > 0.5]) for d in range(horizon)] for o in range(nr_OTs)],
                        folder / "surgery_types_ot.csv")
    for name, values in result.outputs.items():
        np.savetxt(folder / f"{name}.csv", values, delimiter=",")

    diagnostics = diag(x)
    np.savetxt(folder / "ot_overtime_prob.csv", diagnostics["p_overtime"], delimiter=",")
    if "var" in diagnostics:
        np.savetxt(folder / "ot_var.csv", diagnostics["var"], delimiter=",")
    if "expected_overtime" in diagnostics and "ot_expected_overtime_model" in result.outputs:
        # recomputed from the schedule (exact sqrt for PLF, scenarios for SBM)
        np.savetxt(folder / "ot_expected_overtime_recomputed.csv", diagnostics["expected_overtime"],
                   delimiter=",")

    s = result.summary
    return (f"objective {s['objective']:.2f}, gap {100 * s['gap']:.2f}%, "
            f"{s['n_scheduled']}/{s['n_surgeries']} scheduled, {s['total_time']:.0f}s")


def main():
    args = parse_args()
    shared = {}
    if any(m.startswith("FNN") for m in args.methods):
        from orsched.fnn_model import load_model
        shared["fnn"] = load_model()
    if "PLF" in args.methods:
        shared["breakpoints"] = milp_plf.load_breakpoints()

    for dataset in args.datasets:
        capacity = config.ot_capacity(config.specialty_of(dataset))
        for method in args.methods:
            if method == "SBM":
                surgeries, durations = load_scenarios(dataset)
            else:
                surgeries, durations = load_instance(dataset), None

            for runtime, alpha in itertools.product(args.runtimes, args.alphas):
                folder = run_dir(method, dataset, runtime, alpha)
                folder.mkdir(parents=True, exist_ok=True)
                print(f"\n===== {method} | {dataset} | runtime {runtime}s | alpha {alpha} =====")
                result, diag = solve_one(method, surgeries, durations, capacity, alpha,
                                         runtime, folder, shared)
                print(f"--> {save_run(result, diag, surgeries, capacity, folder)}")


if __name__ == "__main__":
    main()
