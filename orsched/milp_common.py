# -*- coding: utf-8 -*-
"""
Parts shared by all four MILPs: assignment variables, scheduling
constraints, objective, solving within a time budget, and the run summary.

"""
import json
import time
from dataclasses import dataclass, field

import gurobipy as grb
import numpy as np


@dataclass
class SolveResult:
    """What every method returns."""
    model: grb.Model
    summary: dict                      # written to summary.json
    x: np.ndarray = None               # (n_surgeries, plan_horizon, nr_OTs), None if no solution
    outputs: dict = field(default_factory=dict)  # extra per-OT/day arrays, (nr_OTs, plan_horizon)


def slots(capacity):
    """Lists of open and closed (ot, day) slots."""
    nr_OTs, horizon = capacity.shape
    all_slots = [(o, d) for o in range(nr_OTs) for d in range(horizon)]
    open_slots = [(o, d) for (o, d) in all_slots if capacity[o, d] > 0]
    closed_slots = [(o, d) for (o, d) in all_slots if capacity[o, d] <= 0]
    return open_slots, closed_slots


def add_schedule(model, surgeries, capacity):
    """Adds x[s, day, ot] and the constraints shared by all methods:
      - each surgery is scheduled at most once
      - no surgeries in closed OT-days
      - total sample mean duration per OT-day <= capacity
      - not before the release date
      - surgeries with must_schedule == 1 are scheduled on or before the due date
    """
    n = len(surgeries)
    nr_OTs, horizon = capacity.shape
    sample_mean = surgeries["sample_mean"].to_numpy(dtype=float)
    release = surgeries["release_date"].to_numpy().astype(int)
    due = surgeries["due_date"].to_numpy().astype(int)
    must = surgeries["must_schedule"].to_numpy().astype(int)

    bad = np.flatnonzero(due < release)
    if bad.size:
        raise ValueError(f"Surgeries {bad.tolist()} have due_date before release_date.")

    x = model.addVars(n, horizon, nr_OTs, vtype=grb.GRB.BINARY, name="x")

    model.addConstrs(
        (x.sum(s, "*", "*") <= 1 for s in range(n)), name="schedule_once")

    _, closed_slots = slots(capacity)
    for (o, d) in closed_slots:
        model.addConstrs((x[s, d, o] == 0 for s in range(n)), name=f"closed_{o}_{d}")

    for o in range(nr_OTs):
        for d in range(horizon):
            model.addConstr(
                grb.quicksum(sample_mean[s] * x[s, d, o] for s in range(n)) <= capacity[o, d],
                name=f"capacity_{o}_{d}")

    for s in range(n):
        if release[s] > 0:
            model.addConstr(
                grb.quicksum(x[s, d, o] for d in range(release[s]) for o in range(nr_OTs)) == 0,
                name=f"release_{s}")
        if must[s] == 1:
            model.addConstr(
                grb.quicksum(x[s, d, o] for d in range(due[s] + 1) for o in range(nr_OTs)) == 1,
                name=f"due_{s}")
    return x


def set_objective(model, x, surgeries, expected_overtime, overtime_penalty):
    """Maximise scheduled sample mean duration + due-date priority
    - overtime_penalty * expected overtime."""
    sample_mean = surgeries["sample_mean"].to_numpy(dtype=float)
    priority = 1.0 / (surgeries["due_date"].to_numpy(dtype=float) + 1.0)
    total_duration = grb.quicksum(sample_mean[s] * x[s, d, o] for (s, d, o) in x.keys())
    total_priority = grb.quicksum(priority[s] * x[s, d, o] for (s, d, o) in x.keys())
    model.setObjective(total_duration + total_priority - overtime_penalty * expected_overtime,
                       grb.GRB.MAXIMIZE)
    model._terms = {"duration": total_duration, "priority": total_priority,
                    "expected_overtime": expected_overtime}


def optimize_within_budget(model, t_start, runtime_budget, min_solve_time, mip_gap, log_path,
                           extra_params=None):
    """Gives the solver whatever is left of the total budget after building
    the model (at least min_solve_time seconds)."""
    build_time = time.time() - t_start
    remaining = runtime_budget - build_time
    if remaining < min_solve_time:
        print(f"Warning: building took {build_time:.1f}s; the solver still gets "
              f"{min_solve_time}s, so this run exceeds its {runtime_budget}s budget.")
    model.setParam("LogFile", str(log_path))
    model.setParam("MIPGap", mip_gap)
    model.setParam("TimeLimit", max(remaining, min_solve_time))
    for name, value in (extra_params or {}).items():
        model.setParam(name, value)
    model.optimize()
    return build_time


def has_solution(model):
    return model.SolCount > 0


def x_values(x, n, horizon, nr_OTs):
    arr = np.zeros((n, horizon, nr_OTs))
    for (s, d, o), var in x.items():
        arr[s, d, o] = var.X
    return np.round(arr)


def var_values(var_dict, nr_OTs, horizon):
    return np.array([[var_dict[o, d].X for d in range(horizon)] for o in range(nr_OTs)])


def build_summary(model, x, n, build_time, t_start, overtime_penalty):
    """Run statistics; objective terms are only filled in if a solution exists."""
    summary = {
        "status": model.Status,
        "has_solution": has_solution(model),
        "infeasible": model.Status == grb.GRB.INFEASIBLE,
        "build_time": build_time,
        "solve_time": model.Runtime,
        "total_time": time.time() - t_start,
        "overtime_weight": overtime_penalty,
        "n_surgeries": n,
        "num_constraints": model.NumConstrs,
        "num_variables": model.NumVars,
    }
    if summary["has_solution"]:
        summary.update({
            "objective": model.ObjVal,
            "duration_component": model._terms["duration"].getValue(),
            "priority_component": model._terms["priority"].getValue(),
            "overtime_component": model._terms["expected_overtime"].getValue(),
            "gap": model.MIPGap,
            "n_scheduled": int(round(sum(v.X for v in x.values()))),
        })
    return summary


def write_summary(summary, path):
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
