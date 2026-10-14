# -*- coding: utf-8 -*-
"""
Loading the input files. Every loader renames the columns of the file to the
generic names in config.COLUMNS, so the rest of the code never depends on
the column names used in the original data.

"""
import json
from pathlib import Path

import pandas as pd

from orsched import config

INSTANCE_COLUMNS = ["surgery_type", "sample_mean", "mu", "sigma",
                    "release_date", "due_date", "must_schedule"]
SCENARIO_COLUMNS = ["surgery_type", "sample_mean", "release_date", "due_date", "must_schedule"]


def to_generic(df, required, source="file"):
    """Rename the columns of df to their generic names and check that all
    `required` generic columns are present."""
    file_to_generic = {file_col: generic for generic, file_col in config.COLUMNS.items()}
    df = df.rename(columns=file_to_generic)
    missing = [c for c in required if c not in df.columns]
    if missing:
        expected = [config.COLUMNS[c] for c in missing]
        raise ValueError(
            f"{source} is missing column(s) {expected}. Either rename them in the "
            f"file or update config.COLUMNS."
        )
    return df


def _read(path, required):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return to_generic(pd.read_csv(path), required, source=str(path))


# ----------------------------------------------------------------------------
# Scheduling instances
# ----------------------------------------------------------------------------
def load_instance(dataset):
    """Surgeries of one instance (used by FNN and PLF)."""
    df = _read(config.INSTANCES_DIR / f"{dataset}.csv", INSTANCE_COLUMNS)
    return df[INSTANCE_COLUMNS].reset_index(drop=True)


def scenario_file(dataset):
    n = config.NR_SCENARIOS[config.specialty_of(dataset)]
    return config.SCENARIOS_DIR / f"{dataset}_scenarios_{n}.csv", n


def load_scenarios(dataset):
    """Surgeries + scenario durations of one instance (used by SBM).

    The last N columns of the file are the sampled durations, one column per
    scenario. Returns (surgeries DataFrame, durations array n_surgeries x N).
    """
    path, n_scenarios = scenario_file(dataset)
    df = _read(path, SCENARIO_COLUMNS)
    durations = df.iloc[:, -n_scenarios:].to_numpy(dtype=float)
    return df[SCENARIO_COLUMNS].reset_index(drop=True), durations


# ----------------------------------------------------------------------------
# Historical durations and overtime cases
# ----------------------------------------------------------------------------
def load_historical(specialty):
    """One row per historical surgery: surgery_type, duration (+ optionally
    mu, sigma, sample_mean per type)."""
    path = config.HISTORICAL_DIR / f"historical_durations_{specialty}.csv"
    return _read(path, ["surgery_type", "duration"])


def duration_pools(specialty):
    """{surgery_type: array of historical durations} for bootstrapping."""
    df = load_historical(specialty)
    return {k: g.to_numpy(dtype=float) for k, g in df.groupby("surgery_type")["duration"]}


def overtime_cases(specialty):
    """List of (case_id, [surgery types]) for the real-life overtime cases."""
    folder = config.OVERTIME_CASES_DIR / specialty
    cases = []
    for path in sorted(folder.glob("*.csv")):
        df = _read(path, ["surgery_type"])
        cases.append((path.stem, df["surgery_type"].tolist()))
    return cases


# ----------------------------------------------------------------------------
# FNN training data
# ----------------------------------------------------------------------------
def load_fnn_data(path=None):
    """One row per surgery type: surgery_type, mean_duration, var_duration."""
    path = path or config.FNN_DATA_FILE
    df = _read(path, ["surgery_type", "mean_duration", "var_duration"])
    return df[["surgery_type", "mean_duration", "var_duration"]]


# ----------------------------------------------------------------------------
# Saving / reading the surgery types per OT-day
# ----------------------------------------------------------------------------
def _plain(value):
    return value.item() if hasattr(value, "item") else value


def save_types_per_slot(types_per_slot, path):
    """types_per_slot[o][d] is a list of surgery types; saved as JSON lists."""
    rows = [[json.dumps([_plain(v) for v in cell]) for cell in row] for row in types_per_slot]
    pd.DataFrame(rows).to_csv(path, index=False, header=False)


def load_types_per_slot(path):
    df = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    return [[json.loads(cell) if cell else [] for cell in row] for row in df.to_numpy()]
