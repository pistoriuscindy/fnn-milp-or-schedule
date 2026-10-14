# -*- coding: utf-8 -*-
"""
Step 1 of the FNN training data: every multiset (combination with
replacement) of ..max_group_size surgery types becomes one data point, with
the summed mean and summed variance of its surgery types. The single
surgery types themselves are included as well.

    python scripts/01_create_fnn_datapoints.py --max-group-size 6

"""
import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from orsched import config
from orsched.data import load_fnn_data


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-file", default=str(config.FNN_DATA_FILE))
    p.add_argument("--output-file", default=str(config.FNN_TRAINING_DIR / "generated_datapoints.csv"))
    p.add_argument("--max-group-size", type=int, default=6)
    p.add_argument("--include-ids", action="store_true",
                   help="Also store which surgery types make up each data point (larger file).")
    p.add_argument("--chunk-size", type=int, default=5_000_000,
                   help="Combinations processed (and written) at a time.")
    return p.parse_args()


def combination_chunks(n, r, chunk_size):
    """Yields arrays of index combinations (chunk x r)."""
    combos = itertools.combinations_with_replacement(range(n), r)
    remaining = math.comb(n + r - 1, r)
    while remaining > 0:
        size = min(chunk_size, remaining)
        idx = np.fromiter(itertools.chain.from_iterable(itertools.islice(combos, size)),
                          dtype=np.uint16, count=size * r).reshape(size, r)
        remaining -= size
        yield idx


def main():
    args = parse_args()
    df = load_fnn_data(args.input_file)
    n = len(df)
    ids = df["surgery_type"].to_numpy()
    means = df["mean_duration"].to_numpy(dtype=np.float32)
    variances = df["var_duration"].to_numpy(dtype=np.float32)
    columns = ["ids", "average", "var"] if args.include_ids else ["average", "var"]

    total = sum(math.comb(n + r - 1, r) for r in range(2, args.max_group_size + 1))
    print(f"{n} surgery types -> {total:,} combinations of size 2..{args.max_group_size}.")

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    first = pd.DataFrame({"ids": [[i] for i in ids], "average": means, "var": variances})
    first[columns].to_csv(out, index=False)

    with tqdm(total=total, unit="combo", unit_scale=True) as bar:
        for r in range(2, args.max_group_size + 1):
            for idx in combination_chunks(n, r, args.chunk_size):
                chunk = pd.DataFrame({"average": means[idx].sum(axis=1), "var": variances[idx].sum(axis=1)})
                if args.include_ids:
                    chunk.insert(0, "ids", [row.tolist() for row in ids[idx]])
                chunk.to_csv(out, mode="a", header=False, index=False)
                bar.update(len(idx))

    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
