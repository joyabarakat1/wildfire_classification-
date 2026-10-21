# Can Location and Season Help Identify Wildfire Causes?

This project tests whether location improves classification of recorded California wildfire causes (Human or Natural), compared with discovery month and weekday alone. It does not predict whether a wildfire will occur.

## Dataset

Karen C. Short's FPA FOD, sixth edition (1992-2020), US Forest Service. DOI: https://doi.org/10.2737/RDS-2013-0009.6

[Download the SQLite archive](https://www.datalumos.org/datalumos/project/249023/version/V2/view) and extract `FPA_FOD_20221014.sqlite`. The raw database is not included. Cleaning retains 213,463 California records with known broad cause.

## Models and evaluation

NumPy logistic regression is implemented from scratch. Calendar-only and calendar-plus-location models are compared with and without balanced training weights, against a majority baseline. A random forest is planned.

Training: 1992-2014. Validation: 2015-2017. Test: 2018-2020, not yet scored. All initial thresholds are 0.5. Validation macro-F1 is 0.453 vs 0.672 without weights, and 0.540 vs 0.695 with balanced weights. These are initial comparisons, not optimally tuned models.

## Files

- `code/`: loading, figures, baseline models, and dependencies.
- `results/`: reproduced metrics, weights, optimization histories, and data checks.
- `figures/`: exploration plots.

Keep the code files together. `baseline.py` contains the shared loader and a separate optional scikit-learn workflow; it is not the source of reported NumPy results.

## Setup and running

Use Python 3.11 or newer. From the repository root:

```bash
python -m pip install -r code/requirements.txt
python code/make_figures.py --database "C:/path/to/FPA_FOD_20221014.sqlite" --output figures
python code/train_numpy_baselines.py --database "C:/path/to/FPA_FOD_20221014.sqlite" --output results
```

Replace the example path with your extracted database path. These commands regenerate outputs, so choose other output folders if you want to preserve existing results. No GPU is needed. Scikit-learn and matplotlib are optional dependencies only for running `baseline.py` directly.

## Limitations

Reported causes may contain errors, and excluding unknown causes can introduce bias. Geographic patterns may partly reflect agency reporting practices. Class proportions change over time, so validation performance may not transfer to test. Results cannot assign responsibility for an individual fire.
