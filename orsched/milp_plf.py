# -*- coding: utf-8 -*-
"""
Piecewise-linear (PLF) MILP. Normal approximation of the total OT-day
duration:
    chance constraint:  mu_od + z_alpha * sqrt(var_od) <= C_od
    expected overtime:  max(0, mu_od + kappa_alpha * sqrt(var_od) - C_od),
                        kappa_alpha = phi(z_alpha) / alpha,
i.e. the tail expectation E[k | k > (1 - alpha)-percentile] minus capacity.
sqrt(var_od) is replaced by a piecewise-linear function (SOS2 weights).

"""
import time

import gurobipy as grb
import numpy as np
from scipy.stats import norm

from orsched import config
from orsched.milp_common import (SolveResult, add_schedule, build_summary, has_solution,
                                 optimize_within_budget, set_objective, slots, var_values,
                                 x_values)


# =============================================================================
# Breakpoints of the sqrt approximation
# =============================================================================
def sqrt_breakpoints(x_max, delta_max):
    """Breakpoints (x, y) of a piecewise-linear approximation of sqrt(x) on
    [0, x_max] with maximum deviation delta_max per segment, following the
    construction in Schneider et al. ("Scheduling surgery groups considering
    multiple downstream resources"). Returns an array of shape (n_points, 2)."""
    a = np.zeros(100)
    a[0] = 4
    for k in range(99):
        a[k + 1] = (2 * a[k] - np.sqrt(a[k])) ** 2 / a[k] ** 2

    delta = np.sqrt(x_max) * (1 / (2 * np.sqrt(a[:-1])) + np.sqrt(a[:-1]) / 2 - 1)
    M = int(np.argmax(delta < delta_max))
    if M == 0:
        raise ValueError("No valid segments found for the given delta_max.")

    x_vals = np.zeros(M + 1)
    t_vals = np.zeros(M + 1)
    x_vals[M] = x_max
    t_vals[M] = x_max / a[M]
    for i in range(M, 0, -1):
        t_vals[i - 1] = t_vals[i] / a[i - 1] ** 2
        x_vals[i - 1] = a[i - 1] * t_vals[i - 1]

    # each segment is the tangent-based line of its own t, evaluated at its start
    starts = [(x_vals[i], 0.5 * np.sqrt(t_vals[i]) + 0.5 * np.sqrt(1 / t_vals[i]) * x_vals[i])
              for i in range(M)]
    # last point: same offset from sqrt as the first segment start
    offset = starts[0][1] - np.sqrt(starts[0][0])
    points = [(0.0, 0.0)] + starts + [(x_max, np.sqrt(x_max) + offset)]
    return np.array(points)


def load_breakpoints(path=None):
    return np.loadtxt(path or config.PLF_BREAKPOINTS_FILE, delimiter=",", skiprows=1)


# =============================================================================
# MILP
# =============================================================================
def solve(surgeries, capacity, alpha, breakpoints, runtime, log_path,
          overtime_penalty=config.OVERTIME_PENALTY, mip_gap=config.MIP_GAP,
          min_solve_time=config.MIN_SOLVE_TIME):
    t_start = time.time()
    n = len(surgeries)
    nr_OTs, horizon = capacity.shape
    open_slots, closed_slots = slots(capacity)
    mu = surgeries["mu"].to_numpy(dtype=float)
    var = surgeries["sigma"].to_numpy(dtype=float) ** 2
    bx, by = breakpoints[:, 0], breakpoints[:, 1]
    n_points = len(bx)
    z_alpha = norm.ppf(1 - alpha)
    kappa_alpha = norm.pdf(z_alpha) / alpha

    model = grb.Model()
    x = add_schedule(model, surgeries, capacity)
    lam = model.addVars(nr_OTs, horizon, n_points, lb=0, ub=1, name="lambda")
    overtime = model.addVars(nr_OTs, horizon, lb=0, name="overtime")

    for (o, d) in open_slots:
        mu_od = grb.quicksum(mu[s] * x[s, d, o] for s in range(n))
        sqrt_var_od = grb.quicksum(by[i] * lam[o, d, i] for i in range(n_points))
        model.addConstr(lam.sum(o, d, "*") == 1, name=f"lambda_sum_{o}_{d}")
        model.addConstr(grb.quicksum(bx[i] * lam[o, d, i] for i in range(n_points))
                        == grb.quicksum(var[s] * x[s, d, o] for s in range(n)),
                        name=f"variance_{o}_{d}")
        model.addSOS(grb.GRB.SOS_TYPE2, [lam[o, d, i] for i in range(n_points)])
        model.addConstr(mu_od + z_alpha * sqrt_var_od <= capacity[o, d], name=f"chance_{o}_{d}")
        model.addConstr(overtime[o, d] >= mu_od + kappa_alpha * sqrt_var_od - capacity[o, d],
                        name=f"overtime_{o}_{d}")

    for (o, d) in closed_slots:
        lam[o, d, 0].LB = 1.0
        for i in range(1, n_points):
            lam[o, d, i].UB = 0.0
        overtime[o, d].UB = 0.0

    set_objective(model, x, surgeries, overtime.sum(), overtime_penalty)
    build_time = optimize_within_budget(model, t_start, runtime, min_solve_time, mip_gap, log_path)

    result = SolveResult(model, build_summary(model, x, n, build_time, t_start, overtime_penalty))
    if has_solution(model):
        result.x = x_values(x, n, horizon, nr_OTs)
        result.outputs = {"ot_expected_overtime_model": var_values(overtime, nr_OTs, horizon)}
    return result


# =============================================================================
# Diagnostics for a given schedule
# =============================================================================
def overtime_diagnostics(x_schedule, surgeries, capacity, alpha):
    """Per OT-day, under the normal approximation (exact sqrt):
    p_overtime = P(k > C) and the clipped expected-overtime term
    max(0, mu + kappa_alpha * sigma - C)."""
    nr_OTs, horizon = capacity.shape
    mu = surgeries["mu"].to_numpy(dtype=float)
    var = surgeries["sigma"].to_numpy(dtype=float) ** 2
    kappa_alpha = norm.pdf(norm.ppf(1 - alpha)) / alpha
    out = {k: np.zeros((nr_OTs, horizon)) for k in ("mu", "var", "p_overtime", "expected_overtime")}
    for o in range(nr_OTs):
        for d in range(horizon):
            m = float(mu @ x_schedule[:, d, o])
            v = float(var @ x_schedule[:, d, o])
            out["mu"][o, d], out["var"][o, d] = m, v
            if v > 0:
                sigma = np.sqrt(v)
                out["p_overtime"][o, d] = 1.0 - norm.cdf((capacity[o, d] - m) / sigma)
                out["expected_overtime"][o, d] = max(0.0, m + kappa_alpha * sigma - capacity[o, d])
    return out
