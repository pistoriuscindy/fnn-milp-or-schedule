# -*- coding: utf-8 -*-
"""
Scenario-based (SBM) MILP. For each OT-day and scenario l:
    y_odl = 1 if the scenario duration exceeds capacity (big-M constraint)
    sum_l y_odl <= alpha * N                          (chance constraint)
    O_odl >= duration_odl - C_od, O_odl <= M_odl y_odl  (overtime per scenario)
    g_od = sum_l O_odl / sum_l y_odl                   (mean overtime over violating scenarios)
The product g_od * sum_l y_odl is linearised with a binary expansion of the
violation count. g_od, i.e. E[k - C | k > C], is the expected-overtime term
in the objective.

"""
import time

import gurobipy as grb
import numpy as np

from orsched import config
from orsched.milp_common import (SolveResult, add_schedule, build_summary, has_solution,
                                 optimize_within_budget, set_objective, var_values, x_values)


def big_m_values(surgeries, durations, capacity):
    """M_odl = (largest possible duration of OT-day (o, d) in scenario l) - C_od,
    found by maximising that duration subject to the scheduling constraints.
    Two OT-days on the same day with the same capacity are interchangeable,
    so their values are computed once. Values below 0 are set to 0 (that
    OT-day can never exceed capacity in that scenario)."""
    n, n_scen = durations.shape
    nr_OTs, horizon = capacity.shape
    aux = grb.Model()
    aux.setParam("OutputFlag", 0)
    x = add_schedule(aux, surgeries, capacity)

    big_m = np.zeros((nr_OTs, horizon, n_scen))
    cache = {}
    for d in range(horizon):
        for o in range(nr_OTs):
            key = (d, capacity[o, d])
            if key in cache:
                big_m[o, d, :] = cache[key]
                continue
            for scen in range(n_scen):
                aux.setObjective(grb.quicksum(durations[s, scen] * x[s, d, o] for s in range(n)),
                                 grb.GRB.MAXIMIZE)
                aux.optimize()
                big_m[o, d, scen] = aux.ObjVal - capacity[o, d]
            cache[key] = big_m[o, d, :].copy()
    return np.maximum(big_m, 0.0)


def solve(surgeries, durations, capacity, alpha, runtime, log_path,
          overtime_penalty=config.OVERTIME_PENALTY, mip_gap=config.MIP_GAP,
          min_solve_time=config.MIN_SOLVE_TIME):
    """overtime_penalty must be > 0: g_od is only pushed down to its true
    value through the objective."""
    t_start = time.time()
    n, n_scen = durations.shape
    nr_OTs, horizon = capacity.shape

    big_m = big_m_values(surgeries, durations, capacity)
    m_max = big_m.max(axis=2)  # upper bound on g_od

    # binary expansion of the violation count 0 .. floor(alpha * N)
    max_viol = int(np.floor(alpha * n_scen))
    H = max(int(max_viol).bit_length(), 1)

    model = grb.Model()
    x = add_schedule(model, surgeries, capacity)
    y = model.addVars(nr_OTs, horizon, n_scen, vtype=grb.GRB.BINARY, name="y")
    O = model.addVars(nr_OTs, horizon, n_scen, lb=0, name="O")
    g = model.addVars(nr_OTs, horizon, lb=0, name="g")
    z = model.addVars(nr_OTs, horizon, H, vtype=grb.GRB.BINARY, name="z")
    tau = model.addVars(nr_OTs, horizon, H, lb=0, name="tau")

    for o in range(nr_OTs):
        for d in range(horizon):
            g[o, d].UB = m_max[o, d]
            for l in range(n_scen):
                O[o, d, l].UB = big_m[o, d, l]
                duration = grb.quicksum(durations[s, l] * x[s, d, o] for s in range(n))
                model.addConstr(duration <= capacity[o, d] + big_m[o, d, l] * y[o, d, l],
                                name=f"bigM_{o}_{d}_{l}")
                model.addConstr(duration <= capacity[o, d] + O[o, d, l],
                                name=f"overtime_{o}_{d}_{l}")
                model.addConstr(O[o, d, l] <= big_m[o, d, l] * y[o, d, l],
                                name=f"overtime_if_violated_{o}_{d}_{l}")

            n_violations = y.sum(o, d, "*")
            model.addConstr(n_violations <= alpha * n_scen, name=f"chance_{o}_{d}")
            model.addConstr(n_violations == grb.quicksum(2 ** h * z[o, d, h] for h in range(H)),
                            name=f"violation_count_{o}_{d}")
            for h in range(H):
                # tau_odh = g_od * z_odh (McCormick)
                tau[o, d, h].UB = m_max[o, d]
                model.addConstr(tau[o, d, h] <= m_max[o, d] * z[o, d, h], name=f"tau_z_{o}_{d}_{h}")
                model.addConstr(tau[o, d, h] <= g[o, d], name=f"tau_g_{o}_{d}_{h}")
                model.addConstr(tau[o, d, h] >= g[o, d] - m_max[o, d] * (1 - z[o, d, h]),
                                name=f"tau_lb_{o}_{d}_{h}")
            # g_od * (number of violations) = total overtime
            model.addConstr(grb.quicksum(2 ** h * tau[o, d, h] for h in range(H)) == O.sum(o, d, "*"),
                            name=f"g_def_{o}_{d}")

    set_objective(model, x, surgeries, g.sum(), overtime_penalty)
    build_time = optimize_within_budget(model, t_start, runtime, min_solve_time, mip_gap, log_path,
                                        extra_params={"MIPFocus": config.SBM_MIP_FOCUS})

    summary = build_summary(model, x, n, build_time, t_start, overtime_penalty)
    summary["n_scenarios"] = n_scen
    result = SolveResult(model, summary)
    if has_solution(model):
        result.x = x_values(x, n, horizon, nr_OTs)
        result.outputs = {"ot_expected_overtime_model": var_values(g, nr_OTs, horizon)}
    return result


def overtime_diagnostics(x_schedule, durations, capacity, alpha=None):
    """Per OT-day, over the scenarios: fraction of scenarios with overtime
    and the mean overtime over those scenarios (0 if none)."""
    nr_OTs, horizon = capacity.shape
    out = {k: np.zeros((nr_OTs, horizon)) for k in ("p_overtime", "expected_overtime")}
    for o in range(nr_OTs):
        for d in range(horizon):
            totals = durations.T @ x_schedule[:, d, o]
            over = totals - capacity[o, d]
            violating = over[over > 0]
            out["p_overtime"][o, d] = violating.size / totals.size
            out["expected_overtime"][o, d] = violating.mean() if violating.size else 0.0
    return out
