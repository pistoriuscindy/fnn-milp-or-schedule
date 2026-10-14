# -*- coding: utf-8 -*-
"""
Central settings for the whole project.

Everything that used to be repeated across scripts (column names, folder
locations, OT capacities, alphas, runtimes, the overtime weight, ...) lives
here, so a change only has to be made once.

"""
import os
from pathlib import Path

import numpy as np

# =============================================================================
# Folders
# =============================================================================
ROOT = Path(__file__).resolve().parents[1]

# Set the environment variable ORSCHED_DATA_DIR / ORSCHED_RESULTS_DIR to use
# another location (e.g. data_synthetic/ to try the code without real data).
DATA_DIR = Path(os.environ.get("ORSCHED_DATA_DIR", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("ORSCHED_RESULTS_DIR", ROOT / "results"))

INSTANCES_DIR = DATA_DIR / "instances"            # <dataset>.csv
SCENARIOS_DIR = DATA_DIR / "scenarios"            # <dataset>_scenarios_<N>.csv
HISTORICAL_DIR = DATA_DIR / "historical"          # historical_durations_<SPECIALTY>.csv
OVERTIME_CASES_DIR = DATA_DIR / "overtime_cases"  # <SPECIALTY>/<case_id>.csv
FNN_DATA_FILE = DATA_DIR / "fnn" / "data_fnn.csv"

MODELS_DIR = ROOT / "models"
FNN_MODEL_PATH = MODELS_DIR / "final_model.pt"
PLF_BREAKPOINTS_FILE = MODELS_DIR / "plf_breakpoints.csv"

FNN_TRAINING_DIR = RESULTS_DIR / "fnn_training"
MILP_RESULTS_DIR = RESULTS_DIR / "milp"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
SIMULATION_DIR = RESULTS_DIR / "simulation"
CROSS_METHOD_DIR = RESULTS_DIR / "cross_method"
OVERTIME_CASES_RESULTS_DIR = RESULTS_DIR / "overtime_cases"

# =============================================================================
# Column names
# =============================================================================
# Left: the generic name used everywhere in the code.
# Right: the column name in YOUR csv files. If your files use other column
# names, only change the right-hand side here; the code renames them on load.
COLUMNS = {
    # instance / scenario / historical files
    "surgery_type": "surgery_type",    # code of the surgery type
    "sample_mean": "sample_mean",      # sample mean duration (min)
    "mu": "mu",                        # mean duration used in the stochastic models (min)
    "sigma": "sigma",                  # standard deviation of the duration (min)
    "release_date": "release_date",    # first day (0-based) the surgery may be scheduled
    "due_date": "due_date",            # due day (0-based)
    "must_schedule": "must_schedule",  # 1 = must be scheduled on or before due_date
    "duration": "duration",            # realised duration of a historical surgery (min)
    # FNN training file (one row per surgery type)
    "mean_duration": "mean_duration",  # mean duration of the surgery type (min)
    "var_duration": "var_duration",    # variance of the duration (min^2)
}

# =============================================================================
# Experiment settings
# =============================================================================
DATASETS = ["dataset_CAR_1", "dataset_CAR_2", "dataset_ENT_1", "dataset_ENT_2"]
METHODS = ["FNN_implicit", "FNN_explicit_lp", "PLF", "SBM"]
RUNTIMES = [5 * 60, 60 * 60]          # total time budget per run (s): build + solve
ALPHAS = [0.05, 0.10, 0.15, 0.20]     # allowed overtime probability
OVERTIME_PENALTY = 0.1                # weight of expected overtime in the objective
MIP_GAP = 0.0
MIN_SOLVE_TIME = 10                   # seconds always given to the solver, even if building took long
PLAN_HORIZON = 5                      # days

# Number of scenarios in the SBM scenario files, per specialty
NR_SCENARIOS = {"CAR": 360, "ENT": 290}

# OT capacity in hours, one row per OT, one column per day (0 = OT closed)
CAPACITY_HOURS = {
    "CAR": [
        [8.5, 8.5, 8.5, 8.5, 8.5],
        [8.5, 8.5, 8.5, 8.5, 8.5],
        [8.5, 8.5, 8.5, 8.5, 8.5],
        [8.5, 8.5, 8.5, 8.5, 8.5],
        [0.0, 8.5, 0.0, 8.5, 8.5],
    ],
    "ENT": [
        [8.5, 8.5, 8.5, 8.5, 8.5],
        [8.5, 8.5, 8.5, 8.5, 12.0],  # extended session on day 5
    ],
}

# =============================================================================
# Method-specific settings
# =============================================================================
# FNN input normalisation (input / divisor).
FNN_NORM_MEAN_DIVISOR = 1267.06795143
FNN_NORM_VAR_DIVISOR = 21124.59835006
FNN_EXPLICIT_RELAX_BINARY = True      # LP (True) or MIP (False) bound tightening

# PLF: sqrt(variance) is approximated on [0, PLF_X_MAX] with max error PLF_DELTA_MAX.
# PLF_X_MAX must be at least the largest total variance of one OT-day.
PLF_X_MAX = 432280
PLF_DELTA_MAX = 1.0

SBM_MIP_FOCUS = 3

# =============================================================================
# Simulation / overtime cases
# =============================================================================
RANDOM_SEED = 42
N_SIMULATIONS = 10000
OVERTIME_CASE_CAPACITY = 8.5 * 60     # minutes, one OT-day
OVERTIME_CASE_THRESHOLD = 0.15        # for the "<=0.15 / >0.15" grouping


# =============================================================================
# Helpers
# =============================================================================
def specialty_of(dataset):
    """'dataset_ENT_2' -> 'ENT'."""
    return dataset.split("_")[1]


def ot_capacity(specialty):
    """OT capacity in minutes, shape (nr_OTs, plan_horizon)."""
    if specialty not in CAPACITY_HOURS:
        raise ValueError(f"Unknown specialty {specialty!r}; add it to CAPACITY_HOURS.")
    return np.array(CAPACITY_HOURS[specialty], dtype=float) * 60
