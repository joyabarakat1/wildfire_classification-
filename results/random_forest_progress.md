# Random forest progress results

Train: 1992-2014. Validation: 2015-2017. The 2018-2020 test set remains unscored.

From the repository root, with the SQLite file described in the main README:

    python -m pip install -r code/requirements.txt "scikit-learn>=1.4" "matplotlib>=3.8"
    python code/train_random_forests.py --database "C:/path/to/FPA_FOD_20221014.sqlite" --output results_reproduced

Both feature sets use 200 trees, maximum depth 15, minimum leaf size 20, and seed 42. Both selected balanced training weights and a 0.60 Natural-class threshold on validation. Macro-F1 was 0.593 with calendar features and 0.792 with calendar and location.

- [Validation metrics](forest_validation.csv)
- [Seasonal and geographic error breakdown](forest_validation_errors.csv)

Confusion matrices use Natural, then Human, with actual labels in rows.
