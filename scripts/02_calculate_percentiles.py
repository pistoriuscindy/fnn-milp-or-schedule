# -*- coding: utf-8 -*-
"""
Step 2 of the FNN training data: for every data point (total mean, total
variance) and every alpha in {0.05, 0.10, 0.15, 0.20}, compute the two
network targets under a lognormal distribution with that mean and variance
(Fenton-Wilkinson moment matching):

    P<level>  = (1 - alpha)-percentile                      (output h1)
    CE<level> = E[k | k > (1 - alpha)-percentile]          (output h2)

The file is processed in chunks, so it never has to fit in memory.

    python scripts/02_calculate_percentiles.py

"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

from orsched import config

LEVELS = [0.95, 0.90, 0.85, 0.80]  # 1 - alpha


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-file", default=str(config.FNN_TRAINING_DIR / "generated_datapoints.csv"))
    p.add_argument("--output-file", default=str(config.FNN_TRAINING_DIR / "data_percentiles.csv"))
    p.add_argument("--chunksize", type=int, default=2_000_000)
    return p.parse_args()


def lognormal_targets(mean, var, level):
    """(1 - alpha)-percentile and tail expectation of a lognormal with the
    given mean and variance; level = 1 - alpha."""
    sigma = np.sqrt(np.log(1 + var / mean ** 2))
    mu = np.log(mean) - 0.5 * sigma ** 2
    z = norm.ppf(level)
    percentile = np.exp(mu + z * sigma)
    tail_expectation = mean * norm.cdf(sigma - z) / (1.0 - level)
    return percentile, tail_expectation


def main():
    args = parse_args()
    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)

    reader = pd.read_csv(args.input_file, usecols=["average", "var"], dtype=np.float32,
                         chunksize=args.chunksize)
    n_rows = 0
    for chunk in tqdm(reader, desc="chunks"):
        mean = chunk["average"].to_numpy()
        var = chunk["var"].to_numpy()
        result = {"Average": mean, "Variance": var}
        for level in LEVELS:
            p, ce = lognormal_targets(mean, var, level)
            result[f"P{round(level * 100)}"] = p.astype(np.float32)
            result[f"CE{round(level * 100)}"] = ce.astype(np.float32)
        pd.DataFrame(result).to_csv(out, mode="a", header=not out.exists(), index=False)
        n_rows += len(chunk)

    print(f"{n_rows:,} rows saved to {out}")


if __name__ == "__main__":
    main()
