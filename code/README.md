# Code

What each file does. No data is included in this repository.

| File | Function |
|---|---|
| build_dataset.py | Applies the scope and data rules and builds the modeling dataset |
| check_beyond_range.py | Sets how Logistic Regression handles values outside the training range |
| check_models.py | Diagnostic checks on the final models |
| check_raw.py | Checks that the raw files loaded completely and correctly |
| count_over_limit.py | Counts loans above the conforming loan limit |
| credit_box.py | Q3: finds, scores and tests the approval cutoffs |
| explore.py | Summarizes delinquency rates by variable on the training loans |
| fit_models.py | Trains Logistic Regression and LightGBM (Optuna search) and calibrates their percentages |
| inventory.py | Profiles every raw field: counts, blanks and unusual values |
| load_raw.py | Loads the raw Freddie Mac and Fannie Mae files into a DuckDB database |
| loan_path.py | Follows each loan's monthly records to find whether it became seriously delinquent within its first 24 payments |
| model_core.py | Shared pieces used across the scripts: variable definitions, model inputs and calibration |
| requirements.txt | The Python packages the code needs |
| score_test.py | Scores the test loans: Q1 results and Q2's SHAP analysis |
