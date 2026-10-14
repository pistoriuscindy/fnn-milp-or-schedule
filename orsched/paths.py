# -*- coding: utf-8 -*-
"""
Folder layout of the MILP results, shared by the solver and all analyses:

    results/milp/<method>/<dataset>/runtime_<s>/alpha_<code>/

"""
from orsched import config


def alpha_code(alpha):
    """0.05 -> '005'."""
    return f"{alpha:.2f}".replace(".", "")


def alpha_from_code(code):
    """'005' -> 0.05."""
    return int(code) / 100.0


def run_dir(method, dataset, runtime, alpha):
    return config.MILP_RESULTS_DIR / method / dataset / f"runtime_{runtime}" / f"alpha_{alpha_code(alpha)}"


def iter_runs(methods=None):
    """Yields (method, dataset, runtime, alpha, folder) for every solved run on disk."""
    for summary in sorted(config.MILP_RESULTS_DIR.glob("*/*/runtime_*/alpha_*/summary.json")):
        folder = summary.parent
        method, dataset = folder.parts[-4], folder.parts[-3]
        if methods is not None and method not in methods:
            continue
        runtime = int(folder.parts[-2].split("_")[1])
        alpha = alpha_from_code(folder.parts[-1].split("_")[1])
        yield method, dataset, runtime, alpha, folder
