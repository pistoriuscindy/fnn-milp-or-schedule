# Input data

The hospital data used in the paper cannot be shared (patient privacy), so
this folder is empty on GitHub. To run the code, put
your own files here in the format below, or create a synthetic dataset with
`python scripts/00\_make\_synthetic\_data.py`.

The column names below are the defaults. If your files use other names,
change the right-hand side of `COLUMNS` in `orsched/config.py`; nothing else
needs to change.

```
data/
├── instances/dataset\_<SPEC>\_<n>.csv                 # FNN and PLF
├── scenarios/dataset\_<SPEC>\_<n>\_scenarios\_<N>.csv   # SBM
├── historical/historical\_durations\_<SPEC>.csv       # simulation, overtime cases
├── overtime\_cases/<SPEC>/<case\_id>.csv              # overtime cases
└── fnn/data\_fnn.csv                                 # FNN training
```

`<SPEC>` is the specialty (`CAR` or `ENT`); it selects the OT capacity in
`config.CAPACITY\_HOURS`. Durations are in minutes, days are 0-based.

### instances/dataset\_<SPEC>\_<n>.csv

One row per surgery on the waiting list.

|column|meaning|
|-|-|
|`surgery\_type`|code of the surgery type|
|`sample\_mean`|sample mean duration (deterministic capacity constraint and objective)|
|`mu`|mean duration (stochastic models)|
|`sigma`|standard deviation of the duration|
|`release\_date`|first day the surgery may be scheduled|
|`due\_date`|due day|
|`must\_schedule`|1 if the surgery must be scheduled on or before `due\_date`, else 0|

### scenarios/dataset\_<SPEC>\_<n>*scenarios*<N>.csv

The same surgeries in the same order (at least `surgery\_type, sample\_mean, release\_date, due\_date, must\_schedule`), followed by **N** columns with one
sampled duration per scenario. These must be the **last** N columns. N is
set per specialty in `config.NR\_SCENARIOS` (360 for CAR, 290 for ENT in the
paper).

### historical/historical\_durations\_<SPEC>.csv

One row per historical surgery: `surgery\_type`, `duration` (realised
duration). `scripts/10\_check\_overtime\_cases.py` also needs `mu` and `sigma`
per surgery type (repeated on every row of that type), and optionally
`sample\_mean`.

### overtime\_cases/<SPEC>/<case\_id>.csv

One file per real-life OT-day that ran into overtime, with a `surgery\_type`
column listing the surgeries performed that day.

### fnn/data\_fnn.csv

One row per surgery type (only types with more than 30 observations):
`surgery\_type`, `mean\_duration`, `var\_duration` (variance, min²).

