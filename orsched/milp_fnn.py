# -*- coding: utf-8 -*-
"""
FNN-based MILP. For every open OT-day the trained network maps
[total mean, total variance, Phi^-1(1 - alpha)] to
    h1 = (1 - alpha)-percentile, constrained to h1 <= capacity
    h2 = tail expectation E[k | k > h1]; max(0, h2 - capacity) is the
         expected-overtime term in the objective.

Two ways to embed the network:
    "implicit"    - gurobi-machinelearning (gurobi_ml.torch)
    "explicit_lp" - explicit big-M ReLU constraints with LP-based bound tightening

"""
import time

import gurobipy as grb
import numpy as np
import torch.nn as nn

from orsched import config
from orsched.fnn_model import alpha_to_z, predict
from orsched.milp_common import (SolveResult, add_schedule, build_summary, has_solution,
                                 optimize_within_budget, set_objective, slots, var_values,
                                 x_values)


# =============================================================================
# Explicit embedding
# =============================================================================
def network_weights(fnn):
    """Weights (n_in x n_out) and biases per Linear layer."""
    linear = [m for m in fnn.net if isinstance(m, nn.Linear)]
    weights = [layer.weight.detach().cpu().numpy().T.astype(float) for layer in linear]
    biases = [layer.bias.detach().cpu().numpy().astype(float) for layer in linear]
    return weights, biases


def _interval_bounds(W, b, a_lb, a_ub):
    """Interval arithmetic for z = W^T a + b over the box [a_lb, a_ub]."""
    W_pos, W_neg = np.maximum(W, 0), np.minimum(W, 0)
    z_lb = a_lb @ W_pos + a_ub @ W_neg + b
    z_ub = a_ub @ W_pos + a_lb @ W_neg + b
    return z_lb, z_ub


def _add_relu_layer(model, W, b, a_prev, z_lb, z_ub, binary, prefix):
    """z = W^T a_prev + b and a = max(0, z) with big-M constraints:
    a >= z, a >= 0, a <= z - LB (1 - u), a <= UB u."""
    n_in, n_out = W.shape
    z_vars, a_vars, u_vars = [], [], []
    for j in range(n_out):
        z = model.addVar(lb=z_lb[j], ub=z_ub[j], name=f"{prefix}_z_{j}")
        model.addConstr(z == grb.quicksum(W[i, j] * a_prev[i] for i in range(n_in)) + b[j],
                        name=f"{prefix}_zdef_{j}")
        a = model.addVar(lb=max(z_lb[j], 0.0), ub=max(z_ub[j], 0.0), name=f"{prefix}_a_{j}")
        u = model.addVar(vtype=grb.GRB.BINARY if binary else grb.GRB.CONTINUOUS,
                         lb=0.0, ub=1.0, name=f"{prefix}_u_{j}")
        model.addConstr(a >= z, name=f"{prefix}_relu_a_{j}")
        model.addConstr(a <= z - z_lb[j] * (1 - u), name=f"{prefix}_relu_c_{j}")
        model.addConstr(a <= z_ub[j] * u, name=f"{prefix}_relu_d_{j}")
        z_vars.append(z)
        a_vars.append(a)
        u_vars.append(u)
    return z_vars, a_vars, u_vars


def lp_tightened_bounds(weights, biases, input_lb, input_ub, relax_binary=True):
    """Bounds (z_lb, z_ub) for every layer. With
    relax_binary=True the auxiliary problems are LPs, otherwise MIPs."""
    z_lb, z_ub = _interval_bounds(weights[0], biases[0], np.asarray(input_lb, float),
                                  np.asarray(input_ub, float))
    bounds = [(z_lb, z_ub)]
    if len(weights) == 1:
        return bounds

    aux = grb.Model()
    aux.setParam("OutputFlag", 0)
    a_in = [aux.addVar(lb=lo, ub=hi) for lo, hi in zip(input_lb, input_ub)]
    _, a_prev, _ = _add_relu_layer(aux, weights[0], biases[0], a_in, z_lb, z_ub,
                                   not relax_binary, "l0")

    for layer in range(1, len(weights)):
        W, b = weights[layer], biases[layer]
        a_lb = np.maximum(bounds[-1][0], 0.0)
        a_ub = np.maximum(bounds[-1][1], 0.0)
        ia_lb, ia_ub = _interval_bounds(W, b, a_lb, a_ub)
        z_lb, z_ub = np.empty(W.shape[1]), np.empty(W.shape[1])
        for j in range(W.shape[1]):
            expr = grb.quicksum(W[i, j] * a_prev[i] for i in range(W.shape[0])) + b[j]
            aux.setObjective(expr, grb.GRB.MINIMIZE)
            aux.optimize()
            lo = aux.ObjVal if aux.Status == grb.GRB.OPTIMAL else ia_lb[j]
            aux.setObjective(expr, grb.GRB.MAXIMIZE)
            aux.optimize()
            hi = aux.ObjVal if aux.Status == grb.GRB.OPTIMAL else ia_ub[j]
            # intersect with interval bounds to stay valid under solver tolerances
            z_lb[j], z_ub[j] = max(lo, ia_lb[j]), min(hi, ia_ub[j])
        bounds.append((z_lb, z_ub))
        if layer < len(weights) - 1:
            _, a_prev, _ = _add_relu_layer(aux, W, b, a_prev, z_lb, z_ub,
                                           not relax_binary, f"l{layer}")
    return bounds


def add_explicit_network(model, weights, biases, bounds, inputs, outputs, prefix):
    """One copy of the network (one OT-day) with big-M ReLU constraints on
    every layer, including the output layer."""
    a_prev = list(inputs)
    for layer, (W, b) in enumerate(zip(weights, biases)):
        z_lb, z_ub = bounds[layer]
        _, a_prev, _ = _add_relu_layer(model, W, b, a_prev, z_lb, z_ub, True, f"{prefix}_l{layer}")
    for k, out in enumerate(outputs):
        model.addConstr(out == a_prev[k], name=f"{prefix}_output_{k}")


# =============================================================================
# MILP
# =============================================================================
def solve(surgeries, capacity, alpha, fnn, embedding, runtime, log_path,
          overtime_penalty=config.OVERTIME_PENALTY, mip_gap=config.MIP_GAP,
          min_solve_time=config.MIN_SOLVE_TIME,
          relax_binary=config.FNN_EXPLICIT_RELAX_BINARY):
    """Builds and solves the FNN MILP. embedding is "implicit" or "explicit_lp"."""
    if embedding not in ("implicit", "explicit_lp"):
        raise ValueError(f"Unknown embedding {embedding!r}.")
    t_start = time.time()
    n = len(surgeries)
    nr_OTs, horizon = capacity.shape
    open_slots, closed_slots = slots(capacity)
    mu = surgeries["mu"].to_numpy(dtype=float)
    var = surgeries["sigma"].to_numpy(dtype=float) ** 2
    norm_mean = 1.0 / fnn.norm_mean_divisor
    norm_var = 1.0 / fnn.norm_var_divisor
    z_alpha = alpha_to_z(alpha)

    model = grb.Model()
    x = add_schedule(model, surgeries, capacity)

    # Bounds on the (normalised) network inputs, as used in the paper. The
    # explicit embedding needs a finite bound on var_OT: the sum of all
    # variances, as if every surgery were in one OT-day.
    mean_ub = 1440
    var_ub = norm_var * var.sum() if embedding == "explicit_lp" else grb.GRB.INFINITY

    mean_OT = model.addVars(nr_OTs, horizon, lb=0, ub=mean_ub, name="mean_OT")  # normalised
    var_OT = model.addVars(nr_OTs, horizon, lb=0, ub=var_ub, name="var_OT")     # normalised
    perc_OT = model.addVars(nr_OTs, horizon, lb=-grb.GRB.INFINITY, ub=1440, name="perc_OT")  # h1 (min)
    condexp_OT = model.addVars(nr_OTs, horizon, lb=0, name="condexp_OT")         # h2 (min)
    eo_OT = model.addVars(nr_OTs, horizon, lb=0, name="eo_OT")                   # max(0, h2 - C)
    z_var = model.addVar(lb=z_alpha, ub=z_alpha, name="z_alpha")

    for o in range(nr_OTs):
        for d in range(horizon):
            model.addConstr(mean_OT[o, d] == norm_mean * grb.quicksum(mu[s] * x[s, d, o] for s in range(n)),
                            name=f"mean_def_{o}_{d}")
            model.addConstr(var_OT[o, d] == norm_var * grb.quicksum(var[s] * x[s, d, o] for s in range(n)),
                            name=f"var_def_{o}_{d}")
            model.addConstr(perc_OT[o, d] <= capacity[o, d], name=f"chance_{o}_{d}")

    for (o, d) in closed_slots:
        for v in (mean_OT, var_OT, perc_OT, condexp_OT, eo_OT):
            v[o, d].LB = v[o, d].UB = 0.0
    for (o, d) in open_slots:
        model.addConstr(eo_OT[o, d] >= condexp_OT[o, d] - capacity[o, d], name=f"eo_{o}_{d}")

    if embedding == "implicit":
        import gurobi_ml.torch
        for (o, d) in open_slots:
            gurobi_ml.torch.add_sequential_constr(
                model, fnn.net, [mean_OT[o, d], var_OT[o, d], z_var],
                [perc_OT[o, d], condexp_OT[o, d]])
    else:
        weights, biases = network_weights(fnn)
        bounds = lp_tightened_bounds(weights, biases, [0.0, 0.0, z_alpha],
                                     [mean_ub, var_ub, z_alpha], relax_binary)
        for (o, d) in open_slots:
            add_explicit_network(model, weights, biases, bounds,
                                 [mean_OT[o, d], var_OT[o, d], z_var],
                                 [perc_OT[o, d], condexp_OT[o, d]], prefix=f"fnn_{o}_{d}")

    set_objective(model, x, surgeries, eo_OT.sum(), overtime_penalty)
    build_time = optimize_within_budget(model, t_start, runtime, min_solve_time, mip_gap, log_path)

    result = SolveResult(model, build_summary(model, x, n, build_time, t_start, overtime_penalty))
    if has_solution(model):
        result.x = x_values(x, n, horizon, nr_OTs)
        result.outputs = {
            "ot_mu": var_values(mean_OT, nr_OTs, horizon) / norm_mean,
            "ot_var": var_values(var_OT, nr_OTs, horizon) / norm_var,
            "ot_percentile": var_values(perc_OT, nr_OTs, horizon),
            "ot_condexp": var_values(condexp_OT, nr_OTs, horizon),
            "ot_expected_overtime_model": var_values(eo_OT, nr_OTs, horizon),
        }
    return result


# =============================================================================
# Diagnostics for a given schedule
# =============================================================================
def overtime_diagnostics(x_schedule, surgeries, capacity, alpha, fnn):
    """Per OT-day: the network's predictions and an overtime probability
    derived from them.

    The network only predicts the (1 - alpha)-percentile, not a full
    distribution, so p_overtime is a heuristic: (1 - alpha) * C / h1 is taken
    as P(k <= C) when C <= h1, with linear interpolation up to 1 above h1.
    """
    nr_OTs, horizon = capacity.shape
    mu = surgeries["mu"].to_numpy(dtype=float)
    var = surgeries["sigma"].to_numpy(dtype=float) ** 2
    level = 1.0 - alpha
    out = {k: np.zeros((nr_OTs, horizon)) for k in
           ("mu", "var", "percentile", "condexp", "p_overtime", "expected_overtime")}

    for o in range(nr_OTs):
        for d in range(horizon):
            cap = capacity[o, d]
            m = float(mu @ x_schedule[:, d, o])
            v = float(var @ x_schedule[:, d, o])
            if cap <= 0 or m <= 0 or v <= 0:
                continue
            h1, h2 = predict(fnn, m, v, alpha)
            if h1 > 0:
                if cap <= h1:
                    p_leq = level * cap / h1
                else:
                    p_leq = level + (1.0 - level) * (cap - h1) / h1
                p_over = 1.0 - min(max(p_leq, 0.0), 1.0)
            else:
                p_over = 0.0
            out["mu"][o, d], out["var"][o, d] = m, v
            out["percentile"][o, d], out["condexp"][o, d] = h1, h2
            out["p_overtime"][o, d] = p_over
            out["expected_overtime"][o, d] = max(0.0, h2 - cap)
    return out
