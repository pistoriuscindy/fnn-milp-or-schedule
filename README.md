# Operating room scheduling: FNN vs. piecewise-linear vs. scenario-based MILPs

> Handling Overtime Constraints in Mixed Integer Linear Programming for Surgical
Scheduling: A Comparison of Neural Network and Classical Linearization
Techniques

This repository contains the code for the paper *"Handling Overtime Constraints in Mixed Integer Linear Programming for Surgical
Scheduling: A Comparison of Neural Network and Classical Linearization
Techniques"*. It
compares four ways of handling uncertain surgery durations in a
overtime probability-constrained, multi-day operating room (OR) scheduling MILP:

|Method|Module|Overtime probability constraint and expected overtime|
|-|-|-|
|`FNN\_implicit`|`orsched/milp\_fnn.py`|A trained ReLU network predicts the (1-α)-percentile and the tail expectation of the total OT-day duration (lognormal, Fenton–Wilkinson). Embedded with [gurobi-machinelearning](https://github.com/Gurobi/gurobi-machinelearning).|
|`FNN\_explicit\_lp`|`orsched/milp\_fnn.py`|The same network, embedded with explicit big-M ReLU constraints and LP-based bound tightening.|
|`PLF`|`orsched/milp\_plf.py`|Normal approximation μ + z<sub>α</sub>·√σ² ≤ C, with √· replaced by a piecewise-linear function (SOS2).|
|`SBM`|`orsched/milp\_sbm.py`|Sample-based chance constraint over duration scenarios (big-M), with the mean overtime over the violating scenarios.|

All methods maximise *scheduled mean duration + due-date priority − λ · expected overtime*
subject to the same assignment, capacity, release-date and due-date constraints
(`orsched/milp\_common.py`).

> \*\*Data availability.\*\* The hospital data cannot be shared because of
> patient-privacy restrictions. The expected file formats are described in
> \[`data/README.md`](data/README.md), and
> `scripts/00\_make\_synthetic\_data.py` creates a random dataset in that format,
> so the whole pipeline can be run without the real data.

## Repository structure

```
├── orsched/                     # the methods (Python package)
│   ├── config.py                # ALL settings: paths, column names, capacities, alphas, runtimes, λ
│   ├── data.py                  # loading input files (renames columns to generic names)
│   ├── paths.py                 # folder layout of the results
│   ├── milp\_common.py           # constraints/objective/solving shared by all methods
│   ├── milp\_fnn.py              # FNN MILP (implicit and explicit embedding)
│   ├── milp\_plf.py              # PLF MILP + breakpoints of the sqrt approximation
│   ├── milp\_sbm.py              # SBM MILP
│   ├── fnn\_model.py             # network definition, saving/loading
│   ├── results.py, plots.py, excel.py   # analysis helpers
├── scripts/                     # run these, in this order
│   ├── 00\_make\_synthetic\_data.py
│   ├── 01\_create\_fnn\_datapoints.py
│   ├── 02\_calculate\_percentiles.py
│   ├── 03\_train\_fnn.py
│   ├── 04\_determine\_breakpoints.py
│   ├── 05\_run\_milp.py
│   ├── 06\_analyze\_results.py
│   ├── 07\_run\_simulation.py
│   ├── 08\_check\_cross\_method\_feasibility.py
│   ├── 09\_simulate\_overtime\_cases.py
│   └── 10\_check\_overtime\_cases.py
├── models/
│   ├── final\_model.pt           # trained network used in the paper (2 hidden layers x 10 neurons)
│   └── plf\_breakpoints.csv      # PLF breakpoints used in the paper
├── data/                        # input data (not included, see data/README.md)
└── results/                     # created by the scripts
```

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt  # also installs the orsched package
```

**Gurobi.** All models are solved with [Gurobi](https://www.gurobi.com/). The
licence that comes with `pip install gurobipy` is limited to small models; the
experiments need a full licence (free for academic use).

## Quick start with synthetic data

```bash
python scripts/00\_make\_synthetic\_data.py         # writes data\_synthetic/
export ORSCHED\_DATA\_DIR=data\_synthetic           # Windows: set ORSCHED\_DATA\_DIR=data\_synthetic
python scripts/05\_run\_milp.py --datasets dataset\_ENT\_1 --runtimes 60 --alphas 0.1
python scripts/06\_analyze\_results.py
```

`python scripts/06\_analyze\_results.py --demo` makes the plots from random
example numbers, without running anything.

## Reproducing the experiments

All settings (datasets, methods, runtimes, alphas, λ, OT capacities, number of
scenarios, random seed) are in `orsched/config.py`. Every script can be run
from any folder and shows its options with `--help`.

**1. Train the FNN** (optional, `models/final\_model.pt` is included)

```bash
python scripts/01\_create\_fnn\_datapoints.py --max-group-size 6
python scripts/02\_calculate\_percentiles.py
python scripts/03\_train\_fnn.py --neurons 10 --hidden-layers 2 --learning-rates 0.01 --batch-sizes 256
```

Step 1 sums the means and variances of every combination (with replacement)
of 2 to 6 surgery types. Step 2 computes the network targets for
α ∈ {0.05, 0.10, 0.15, 0.20}. Step 3 runs a grid search over the given
hyperparameters, selects the model with the lowest validation error and saves
it as `results/fnn\_training/best\_model.pt`, including its input normalisation.
Copy it to `models/final\_model.pt` to use it.

**2. PLF breakpoints** (optional, `models/plf\_breakpoints.csv` is included)

```bash
python scripts/04\_determine\_breakpoints.py
```

**3. Solve all instances**

```bash
python scripts/05\_run\_milp.py
```

Every combination of method × dataset × runtime × α is solved. The runtime
is the total budget per run. Each run is saved to
`results/milp/<method>/<dataset>/runtime\_<s>/alpha\_<code>/`:

|file|content|
|-|-|
|`summary.json`|objective and its terms, build/solve time, MIP gap, status|
|`gurobi.log`|solver log|
|`x\_schedule.csv`|assignment x\[s, day, OT], one row per surgery|
|`surgery\_types\_ot.csv`|surgery types per OT (rows) and day (columns)|
|`ot\_mean.csv`|scheduled sample mean duration per OT-day|
|`ot\_expected\_overtime\_model.csv`|the method's own expected-overtime term per OT-day|
|`ot\_overtime\_prob.csv`|the method's own overtime probability per OT-day|

**4. Analyse**

```bash
python scripts/06\_analyze\_results.py                  # results table, plots, overtime summary
python scripts/07\_run\_simulation.py                   # bootstrap simulation of every schedule
python scripts/08\_check\_cross\_method\_feasibility.py   # every schedule judged by every method
python scripts/09\_simulate\_overtime\_cases.py          # simulated overtime probability of real cases
python scripts/10\_check\_overtime\_cases.py             # would each method accept those cases?
```

All output goes to `results/`.

## License

Copyright (C) 2026 Cindy Pistorius

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. 

Note: the code depends on Gurobi, which is proprietary software and is not
covered by this license. A separate Gurobi licence is required to run the models.

## Contact

Cindy Pistorius — c.pistorius@erasmusmc.nl

