# -*- coding: utf-8 -*-
"""
Simulated overtime probability of each real-life overtime case (one OT-day
with capacity OVERTIME_CASE_CAPACITY): the durations of its surgeries are
drawn from the historical durations of their types.

Output: results/overtime_cases/simulation_overtime_probabilities.csv

    python scripts/09_simulate_overtime_cases.py

"""
import argparse

import numpy as np
import pandas as pd

from orsched import config
from orsched.data import duration_pools, overtime_cases


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-simulations", type=int, default=config.N_SIMULATIONS)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for specialty in config.CAPACITY_HOURS:
        pools = duration_pools(specialty)
        cases = overtime_cases(specialty)
        print(f"{specialty}: {len(cases)} overtime cases")
        for case_id, types in cases:
            missing = sorted({t for t in types if t not in pools})
            if missing:
                print(f"  skipping {case_id}: no historical durations for {missing}")
                continue
            totals = sum(rng.choice(pools[t], size=args.n_simulations) for t in types)
            p = float((totals > config.OVERTIME_CASE_CAPACITY).mean())
            rows.append({"specialty": specialty, "case_id": case_id, "nr_surgeries": len(types),
                         "overtime_probability": p})

    df = pd.DataFrame(rows)
    df["sim_group"] = np.where(df["overtime_probability"] <= config.OVERTIME_CASE_THRESHOLD,
                               f"<={config.OVERTIME_CASE_THRESHOLD}", f">{config.OVERTIME_CASE_THRESHOLD}")
    path = config.OVERTIME_CASES_RESULTS_DIR / "simulation_overtime_probabilities.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} cases to {path}")


if __name__ == "__main__":
    main()
