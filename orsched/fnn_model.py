# -*- coding: utf-8 -*-
"""
The feedforward network (FNN) that predicts, for an OT-day with total mean
`mean` and total variance `var` and a risk level alpha:
    h1: the (1 - alpha)-percentile of the total duration
    h2: the tail expectation E[k | k > h1]
Inputs are [mean / norm_mean_divisor, var / norm_var_divisor, Phi^-1(1 - alpha)].
Every layer, including the output layer, has a ReLU activation, so both
outputs are non-negative.

Checkpoints store the architecture, the weights and the normalisation
divisors (not the pickled model object).
"""
from statistics import NormalDist

import torch
import torch.nn as nn

from orsched import config


class FNNReLUOutput(nn.Module):
    def __init__(self, n_inputs=3, n_outputs=2, hidden_layers=2, neurons=10):
        super().__init__()
        layers = []
        in_dim = n_inputs
        for _ in range(hidden_layers):
            layers += [nn.Linear(in_dim, neurons), nn.ReLU()]
            in_dim = neurons
        layers += [nn.Linear(in_dim, n_outputs), nn.ReLU()]
        self.net = nn.Sequential(*layers)
        self.norm_mean_divisor = config.FNN_NORM_MEAN_DIVISOR
        self.norm_var_divisor = config.FNN_NORM_VAR_DIVISOR

    def forward(self, x):
        return self.net(x)


def alpha_to_z(alpha):
    """Phi^-1(1 - alpha), the third network input."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}.")
    return NormalDist().inv_cdf(1.0 - alpha)


def save_model(model, path, hidden_layers, neurons, norm_mean_divisor, norm_var_divisor):
    torch.save({
        "hidden_layers": hidden_layers,
        "neurons": neurons,
        "n_inputs": 3,
        "n_outputs": 2,
        "norm_mean_divisor": float(norm_mean_divisor),
        "norm_var_divisor": float(norm_var_divisor),
        "state_dict": model.state_dict(),
    }, path)


def load_model(path=None):
    """Loads a checkpoint and checks that it has the expected shape:
    3 inputs, 2 outputs, only Linear/ReLU layers, ending in a ReLU."""
    path = path or config.FNN_MODEL_PATH
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = FNNReLUOutput(ckpt["n_inputs"], ckpt["n_outputs"], ckpt["hidden_layers"], ckpt["neurons"])
    model.load_state_dict(ckpt["state_dict"])
    model.norm_mean_divisor = ckpt.get("norm_mean_divisor", config.FNN_NORM_MEAN_DIVISOR)
    model.norm_var_divisor = ckpt.get("norm_var_divisor", config.FNN_NORM_VAR_DIVISOR)
    model.eval()

    linear = [m for m in model.net if isinstance(m, nn.Linear)]
    if linear[0].in_features != 3 or linear[-1].out_features != 2:
        raise ValueError("Expected a network with 3 inputs and 2 outputs.")
    if any(not isinstance(m, (nn.Linear, nn.ReLU)) for m in model.net) or not isinstance(model.net[-1], nn.ReLU):
        raise ValueError("Expected only Linear/ReLU layers, with a ReLU on the output layer.")
    return model


def predict(model, mean, var, alpha):
    """Network prediction for one OT-day: returns (h1 percentile, h2 tail expectation)."""
    x = torch.tensor([[mean / model.norm_mean_divisor, var / model.norm_var_divisor,
                       alpha_to_z(alpha)]], dtype=torch.float32)
    with torch.no_grad():
        h1, h2 = model(x).numpy()[0]
    return float(h1), float(h2)
