# -*- coding: utf-8 -*-
"""
Computes the breakpoints of the piecewise-linear sqrt approximation used by
the PLF method and saves them to models/plf_breakpoints.csv (plus a plot).

    python scripts/04_determine_breakpoints.py

"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from orsched import config
from orsched.milp_plf import sqrt_breakpoints


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--x-max", type=float, default=config.PLF_X_MAX,
                   help="Upper end of the variance range (must cover the largest OT-day variance).")
    p.add_argument("--delta-max", type=float, default=config.PLF_DELTA_MAX,
                   help="Maximum deviation from sqrt per segment.")
    return p.parse_args()


def main():
    args = parse_args()
    points = sqrt_breakpoints(args.x_max, args.delta_max)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.savetxt(config.PLF_BREAKPOINTS_FILE, points, delimiter=",", header="x,y", comments="", fmt="%.10f")
    print(f"{len(points)} breakpoints saved to {config.PLF_BREAKPOINTS_FILE}")

    x = np.linspace(0, args.x_max, 1000)
    plt.figure(figsize=(10, 6))
    plt.plot(points[:, 0], points[:, 1], marker="o", label="piecewise linear")
    plt.plot(x, np.sqrt(x), "--", color="gray", label="sqrt(x)")
    plt.title(f"Piecewise-linear approximation of sqrt(x) (max deviation {args.delta_max})")
    plt.xlabel("variance")
    plt.ylabel("standard deviation")
    plt.legend()
    plt.grid(True)
    plt.savefig(config.MODELS_DIR / "plf_breakpoints.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
