# -*- coding: utf-8 -*-
"""
Would each method have accepted the real-life overtime cases? Every case
(one OT-day, capacity OVERTIME_CASE_CAPACITY) is solved with each method and
alpha; the case is "accepted" if the solver proves optimality and every
surgery of the case is scheduled. All surgeries get release_date 0,
due_date 1 and must_schedule 0.

If 09_simulate_overtime_cases.py was run first, acceptance is compared with
the simulated overtime probability (eligible = probability <= alpha):
    TP accepted & eligible, FP accepted & not eligible,
    FN rejected & eligible, TN rejected & not eligible

Mean, standard deviation and sample mean per surgery type come from the
historical file (columns mu and sigma; sample_mean is computed from the
durations if absent). SBM scenarios are bootstrapped from the historical
durations.

Output: results/overtime_cases/overtime_case_summary.xlsx

    python scripts/10_check_overtime_cases.py

"""
import argparse

import gurobipy as grb
import numpy as np
import pandas as pd

from orsched import config, milp_fnn, milp_plf, milp_sbm
from orsched.data import load_historical, overtime_cases
from orsched.excel import write_workbook
from orsched.fnn_model import load_model
from orsched.paths import alpha_code


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", default=config.METHODS, choices=config.METHODS)
    p.add_argument("--alphas", nargs="+", type=float, default=config.ALPHAS)
    p.add_argument("--runtime", type=int, default=300, help="Time budget per case (s).")
    p.add_argument("--min-solve-time", type=float, default=5)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


def type_lookup(specialty):
    """Per surgery type: sample_mean, mu, sigma, and the historical durations."""
    df = load_historical(specialty)
    missing = [c for c in ("mu", "sigma") if c not in df.columns]
    if missing:
        raise ValueError(f"Historical file for {specialty} needs columns "
                         f"{[config.COLUMNS[c] for c in missing]} for this analysis.")
    lookup = df.groupby("surgery_type")[["mu", "sigma"]].first()
    if "sample_mean" in df.columns:
        lookup["sample_mean"] = df.groupby("surgery_type")["sample_mean"].first()
    else:
        lookup["sample_mean"] = df.groupby("surgery_type")["duration"].mean()
    pools = {k: g.to_numpy(dtype=float) for k, g in df.groupby("surgery_type")["duration"]}
    return lookup, pools


def solve_case(method, surgeries, durations, capacity, alpha, args, fnn, breakpoints, log):
    common = dict(runtime=args.runtime, log_path=log, min_solve_time=args.min_solve_time)
    if method == "SBM":
        return milp_sbm.solve(surgeries, durations, capacity, alpha, **common)
    if method == "PLF":
        return milp_plf.solve(surgeries, capacity, alpha, breakpoints, **common)
    embedding = "implicit" if method == "FNN_implicit" else "explicit_lp"
    return milp_fnn.solve(surgeries, capacity, alpha, fnn, embedding, **common)


def confusion(group):
    known = group.dropna(subset=["eligible"])
    acc, eli = known["accepted"].astype(bool), known["eligible"].astype(bool)
    tp, fp = int((acc & eli).sum()), int((acc & ~eli).sum())
    fn, tn = int((~acc & eli).sum()), int((~acc & ~eli).sum())
    total = tp + fp + fn + tn
    return {"n_eligible": tp + fn, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "accuracy_vs_sim_pct": 100 * (tp + tn) / total if total else np.nan}


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    capacity = np.array([[config.OVERTIME_CASE_CAPACITY]])
    fnn = load_model() if any(m.startswith("FNN") for m in args.methods) else None
    breakpoints = milp_plf.load_breakpoints() if "PLF" in args.methods else None
    out_dir = config.OVERTIME_CASES_RESULTS_DIR
    rows = []

    for specialty in config.CAPACITY_HOURS:
        lookup, pools = type_lookup(specialty)
        for case_id, types in overtime_cases(specialty):
            missing = sorted({t for t in types if t not in lookup.index})
            if missing:
                print(f"skipping {specialty}/{case_id}: unknown surgery type(s) {missing}")
                continue
            n = len(types)
            surgeries = pd.DataFrame({
                "surgery_type": types,
                "sample_mean": lookup.loc[types, "sample_mean"].to_numpy(),
                "mu": lookup.loc[types, "mu"].to_numpy(),
                "sigma": lookup.loc[types, "sigma"].to_numpy(),
                "release_date": np.zeros(n, int), "due_date": np.ones(n, int), "must_schedule": np.zeros(n, int)})
            n_scen = config.NR_SCENARIOS[specialty]
            durations = np.array([rng.choice(pools[t], size=n_scen) for t in types])

            for alpha in args.alphas:
                for method in args.methods:
                    folder = out_dir / "runs" / specialty / method / f"alpha_{alpha_code(alpha)}"
                    folder.mkdir(parents=True, exist_ok=True)
                    row = {"specialty": specialty, "case_id": case_id, "alpha": alpha, "method": method,
                           "nr_surgeries": n}
                    try:
                        res = solve_case(method, surgeries, durations, capacity, alpha, args, fnn,
                                         breakpoints, folder / f"{case_id}.log")
                        s = res.summary
                        row.update(gurobi_status=s["status"], gap=s.get("gap"),
                                   nr_scheduled=s.get("n_scheduled", 0))
                        row["accepted"] = bool(s["status"] == grb.GRB.OPTIMAL and row["nr_scheduled"] == n)
                    except Exception as e:  # keep going; failed runs are listed in the Errors sheet
                        row.update(accepted=None, error=str(e))
                    rows.append(row)
                    print(f"{specialty}/{case_id} {method} alpha={alpha}: accepted={row['accepted']}")

    df = pd.DataFrame(rows)
    sim_path = out_dir / "simulation_overtime_probabilities.csv"
    if sim_path.exists():
        sim = pd.read_csv(sim_path)[["specialty", "case_id", "overtime_probability"]]
        df = df.merge(sim, on=["specialty", "case_id"], how="left")
        df["eligible"] = np.where(df["overtime_probability"].isna(), np.nan,
                                  df["overtime_probability"] <= df["alpha"])
    else:
        print(f"{sim_path} not found: run 09_simulate_overtime_cases.py for the TP/FP/FN/TN comparison.")
        df["overtime_probability"] = np.nan
        df["eligible"] = np.nan

    summary_rows = []
    for (specialty, method, alpha), g in df.groupby(["specialty", "method", "alpha"]):
        valid = g[g["accepted"].notna()]
        n_acc = int(valid["accepted"].astype(bool).sum())
        summary_rows.append({"specialty": specialty, "method": method, "alpha": alpha,
                             "n_cases": len(valid), "n_accepted": n_acc, "n_rejected": len(valid) - n_acc,
                             "n_errors": int(g["accepted"].isna().sum()),
                             "accepted_pct": 100 * n_acc / len(valid) if len(valid) else np.nan,
                             **confusion(valid)})
    summary = pd.DataFrame(summary_rows)

    error_rows = df[df["accepted"].isna()] if "error" in df.columns else df.iloc[0:0]
    df.to_csv(out_dir / "overtime_case_checks_detailed.csv", index=False)
    write_workbook(out_dir / "overtime_case_summary.xlsx", {
        "Summary": summary,
        "Accepted_Count": summary.pivot_table(index=["specialty", "alpha"], columns="method",
                                              values="n_accepted").reset_index(),
        "Accepted_Pct": summary.pivot_table(index=["specialty", "alpha"], columns="method",
                                            values="accepted_pct").reset_index(),
        "Errors": error_rows,
        "Detailed": df,
    })
    print(summary.to_string(index=False))
    print(f"Saved to {out_dir / 'overtime_case_summary.xlsx'}")


if __name__ == "__main__":
    main()
