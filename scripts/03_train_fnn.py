# -*- coding: utf-8 -*-
"""
Step 3: train the FNN on the output of 02_calculate_percentiles.py.

Inputs  X = [mean, variance, Phi^-1(1 - alpha)]  (mean and variance scaled by their maximum)
Outputs y = [percentile h1, tail expectation h2]

Rows with |z-score| >= 3 on mean or variance are removed, 1% all-zero rows
are added (an empty OT-day has zero duration), and the data is split
70/15/15 into train/validation/test. A grid search over the given
hyperparameters is run; the model with the lowest validation MAE is saved
as best_model.pt. Copy it to models/final_model.pt to use it in the MILPs.

    python scripts/03_train_fnn.py --neurons 10 --hidden-layers 2 --learning-rates 0.01 --batch-sizes 256

"""
import argparse
import copy
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.model_selection import ParameterGrid, train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from orsched import config
from orsched.fnn_model import FNNReLUOutput, save_model

LEVELS = [95, 90, 85, 80]                                   # (1 - alpha) * 100
Z_INPUTS = norm.ppf(np.array(LEVELS) / 100).astype(np.float32)
NEEDED_COLS = ["Average", "Variance"] + [f"P{l}" for l in LEVELS] + [f"CE{l}" for l in LEVELS]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-file", default=str(config.FNN_TRAINING_DIR / "data_percentiles.csv"))
    p.add_argument("--output-dir", default=str(config.FNN_TRAINING_DIR))
    p.add_argument("--chunksize", type=int, default=5_000_000)
    p.add_argument("--sample-frac", type=float, default=1.0, help="Use a random fraction of the rows.")
    p.add_argument("--neurons", type=int, nargs="+", default=[10])
    p.add_argument("--hidden-layers", type=int, nargs="+", default=[2])
    p.add_argument("--learning-rates", type=float, nargs="+", default=[0.01])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[256])
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=75, help="Early stopping patience (epochs).")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


# =============================================================================
# Data
# =============================================================================
def load_data(path, chunksize):
    """Two passes over the file: first mean/std of Average and Variance,
    then load the rows within 3 standard deviations."""
    n, s1, s2 = 0, np.zeros(2), np.zeros(2)
    for chunk in pd.read_csv(path, usecols=["Average", "Variance"], dtype=np.float64, chunksize=chunksize):
        v = chunk.to_numpy()
        n += len(v)
        s1 += v.sum(axis=0)
        s2 += (v ** 2).sum(axis=0)
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12))

    kept = []
    for chunk in tqdm(pd.read_csv(path, usecols=NEEDED_COLS, dtype=np.float32, chunksize=chunksize),
                      desc="loading"):
        v = chunk[NEEDED_COLS].to_numpy()
        mask = (np.abs((v[:, :2] - mean) / std) < 3).all(axis=1)
        kept.append(v[mask])
    wide = np.vstack(kept)
    print(f"Rows: {n:,} loaded, {len(wide):,} after removing outliers.")
    return wide


def to_long(wide):
    """One row per (data point, alpha): X = [mean, var, z], y = [P, CE]."""
    n = len(wide)
    X = np.column_stack([np.tile(wide[:, 0], 4), np.tile(wide[:, 1], 4), np.repeat(Z_INPUTS, n)])
    y = np.column_stack([wide[:, 2:6].T.reshape(-1), wide[:, 6:10].T.reshape(-1)])
    return X.astype(np.float32), y.astype(np.float32)


def split(X, y, seed):
    """70/15/15 train/validation/test split."""
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, test_size=0.30, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_rest, y_rest, test_size=0.50, random_state=seed)
    return [X_tr, X_va, X_te], [y_tr, y_va, y_te]


def prepare(X, y, rng, seed):
    """Scale mean/variance by their maximum, split, and add 1% zero rows
    (spread over the alphas and the three splits)."""
    divisors = X[:, :2].max(axis=0)
    X[:, :2] /= divisors

    n_zero = max(1, int(0.01 * len(X)))
    Xs, ys = split(X, y, seed)
    zero_counts = np.array_split(np.arange(n_zero), len(Z_INPUTS))
    for z, idx in zip(Z_INPUTS, zero_counts):
        Xz = np.zeros((len(idx), 3), np.float32)
        Xz[:, 2] = z
        yz = np.zeros((len(idx), 2), np.float32)
        if len(idx) < 3:  # too few to split: all to train
            Xzs, yzs = [Xz, Xz[:0], Xz[:0]], [yz, yz[:0], yz[:0]]
        else:
            Xzs, yzs = split(Xz, yz, seed)
        Xs = [np.vstack([a, b]) for a, b in zip(Xs, Xzs)]
        ys = [np.vstack([a, b]) for a, b in zip(ys, yzs)]

    for i in range(3):
        perm = rng.permutation(len(Xs[i]))
        Xs[i], ys[i] = Xs[i][perm], ys[i][perm]
    return Xs, ys, divisors


# =============================================================================
# Training
# =============================================================================
def train(params, Xs, ys, epochs, patience):
    model = FNNReLUOutput(hidden_layers=params["hidden_layers"], neurons=params["neurons"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    loss_fn = nn.MSELoss()
    X_tr, X_va = (torch.from_numpy(a) for a in Xs[:2])
    y_tr, y_va = (torch.from_numpy(a) for a in ys[:2])
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=params["batch_size"], shuffle=True)

    history = {"train": [], "val": []}
    best_loss, best_state, waited = np.inf, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            history["train"].append(loss_fn(model(X_tr), y_tr).item())
            history["val"].append(loss_fn(model(X_va), y_va).item())
        if history["val"][-1] < best_loss:
            best_loss, best_state, waited = history["val"][-1], copy.deepcopy(model.state_dict()), 0
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def errors(model, X, y):
    with torch.no_grad():
        err = np.abs(model(torch.from_numpy(X)).numpy() - y)
    return {"mae_h1": err[:, 0].mean(), "maxae_h1": err[:, 0].max(),
            "mae_h2": err[:, 1].mean(), "maxae_h2": err[:, 1].max(),
            "mae": err.mean(), "maxae": err.max()}


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    models_dir = out_dir / "trained_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    X, y = to_long(load_data(args.input_file, args.chunksize))
    if args.sample_frac < 1.0:
        keep = rng.choice(len(X), size=int(len(X) * args.sample_frac), replace=False)
        X, y = X[keep], y[keep]
    Xs, ys, divisors = prepare(X, y, rng, args.seed)
    print(f"Train/val/test: {len(Xs[0]):,} / {len(Xs[1]):,} / {len(Xs[2]):,}")
    print(f"Normalisation divisors: mean {divisors[0]:.8f}, variance {divisors[1]:.8f}")

    grid = ParameterGrid({"neurons": args.neurons, "hidden_layers": args.hidden_layers,
                          "learning_rate": args.learning_rates, "batch_size": args.batch_sizes})
    rows, best = [], None
    for i, params in enumerate(tqdm(list(grid), desc="grid search")):
        model, history = train(params, Xs, ys, args.epochs, args.patience)
        tag = (f"layers{params['hidden_layers']}_neurons{params['neurons']}_batch{params['batch_size']}"
               f"_lr{str(params['learning_rate']).replace('.', 'p')}_{i}")

        plt.figure(figsize=(10, 5))
        plt.plot(history["train"], label="Train loss")
        plt.plot(history["val"], label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title(tag)
        plt.legend()
        plt.grid(True)
        plt.savefig(models_dir / f"loss_curve_{tag}.png")
        plt.close()

        row = dict(params, epochs_run=len(history["train"]), model_file=f"model_{tag}.pt")
        for name, X_s, y_s in zip(("train", "val", "test"), Xs, ys):
            row.update({f"{name}_{k}": v for k, v in errors(model, X_s, y_s).items()})
        rows.append(row)
        save_model(model, models_dir / row["model_file"], params["hidden_layers"], params["neurons"],
                   divisors[0], divisors[1])
        if best is None or row["val_mae"] < best["val_mae"]:
            best = row

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "hyperparameter_results.csv", index=False)
    shutil.copy(models_dir / best["model_file"], out_dir / "best_model.pt")

    print("\nBest model (lowest validation MAE):", best["model_file"])
    for name in ("train", "val", "test"):
        print(f"  {name:5s}  h1 MAE {best[f'{name}_mae_h1']:.4f}  h2 MAE {best[f'{name}_mae_h2']:.4f}  "
              f"max AE {best[f'{name}_maxae']:.4f}")
    print(f"Saved to {out_dir / 'best_model.pt'}; copy it to {config.FNN_MODEL_PATH} to use it in the MILPs.")


if __name__ == "__main__":
    main()
