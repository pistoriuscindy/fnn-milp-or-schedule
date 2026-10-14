# -*- coding: utf-8 -*-
"""
Creates a small SYNTHETIC dataset in the same format as the real input
files, so the whole pipeline can be tried without the (confidential)
hospital data. The numbers are random and mean nothing.

    python scripts/00_make_synthetic_data.py
    then set ORSCHED_DATA_DIR=data_synthetic before running the other scripts.

"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from orsched import config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(config.ROOT / "data_synthetic"), help="Output folder.")
    p.add_argument("--n-types", type=int, default=20, help="Surgery types per specialty.")
    p.add_argument("--n-surgeries", type=int, nargs=2, default=[80, 40],
                   metavar=("CAR", "ENT"), help="Surgeries per instance.")
    p.add_argument("--n-cases", type=int, default=10, help="Overtime cases per specialty.")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


def save(df, path):
    """Write with the column names from config.COLUMNS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.rename(columns={k: v for k, v in config.COLUMNS.items() if k in df.columns}).to_csv(path, index=False)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    sizes = {"CAR": args.n_surgeries[0], "ENT": args.n_surgeries[1]}
    fnn_rows = []

    for specialty in ("CAR", "ENT"):
        # historical durations: lognormal per surgery type
        types = [f"{specialty}{i:03d}" for i in range(args.n_types)]
        hist = []
        for t in types:
            median = rng.uniform(40, 200)
            durations = np.round(median * rng.lognormal(0.0, rng.uniform(0.15, 0.4), size=rng.integers(31, 80)))
            hist += [{"surgery_type": t, "duration": d} for d in durations]
        hist = pd.DataFrame(hist)
        stats = hist.groupby("surgery_type")["duration"].agg(["mean", "std", "var"])
        stats["mu"], stats["sigma"], stats["sample_mean"] = stats["mean"], stats["std"], stats["mean"]
        hist = hist.join(stats[["mu", "sigma", "sample_mean"]], on="surgery_type")
        save(hist, out / "historical" / f"historical_durations_{specialty}.csv")
        fnn_rows.append(pd.DataFrame({"surgery_type": stats.index, "mean_duration": stats["mean"].values,
                                      "var_duration": stats["var"].values}))
        pools = {t: g.to_numpy() for t, g in hist.groupby("surgery_type")["duration"]}

        # two scheduling instances per specialty
        for k in (1, 2):
            n = sizes[specialty]
            chosen = rng.choice(types, size=n)
            release = rng.integers(0, 3, size=n)
            due = release + rng.integers(0, 8, size=n)
            must = ((due < config.PLAN_HORIZON) & (rng.random(n) < 0.2)).astype(int)
            inst = pd.DataFrame({
                "surgery_type": chosen,
                "sample_mean": stats.loc[chosen, "sample_mean"].values,
                "mu": stats.loc[chosen, "mu"].values,
                "sigma": stats.loc[chosen, "sigma"].values,
                "release_date": release, "due_date": due, "must_schedule": must,
            })
            dataset = f"dataset_{specialty}_{k}"
            save(inst, out / "instances" / f"{dataset}.csv")

            n_scen = config.NR_SCENARIOS[specialty]
            scen = np.array([rng.choice(pools[t], size=n_scen) for t in chosen])
            scen_df = pd.concat([inst.drop(columns=["mu", "sigma"]),
                                 pd.DataFrame(scen, columns=[f"scenario_{i + 1}" for i in range(n_scen)])], axis=1)
            save(scen_df, out / "scenarios" / f"{dataset}_scenarios_{n_scen}.csv")

        # real-life overtime cases: one OT-day each
        for c in range(args.n_cases):
            case = pd.DataFrame({"surgery_type": rng.choice(types, size=rng.integers(3, 7))})
            save(case, out / "overtime_cases" / specialty / f"case_{c + 1:03d}.csv")

    save(pd.concat(fnn_rows, ignore_index=True), out / "fnn" / "data_fnn.csv")
    print(f"Synthetic data written to {out}")


if __name__ == "__main__":
    main()
