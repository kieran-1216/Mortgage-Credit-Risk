"""Fit the two Q1 models on training loans and adjust them on validation loans.
Summaries only.

What it does (loan_dataset.duckdb opened READ-ONLY; the test set is never scored and its outcome is never read)
  * Model variables: 17. Loan amount in dollars is NOT a model variable: it is replaced by
    loan amount as a percent of the loan's conforming loan limit (FHFA basic limit for the loan's number of units and
    the year of its origination quarter; 100 x amount / limit, 2 decimals) plus an at-the-limit flag (amount rounded
    to the nearest $1,000 = limit rounded to the nearest $1,000). Both are calculated in the model_loans view; the
    dataset is unchanged. The dollars stay available as the column `dollars` (Q3's weight).
  * Logistic Regression: the 7 numbers as straight lines, the 10 categories as yes/no columns
    (reference = most common TRAINING value), blanks filled with the training lower median / most common value, one
    "incomplete application" column, no penalty. Fitted by Newton's method (IRLS) in chunks (sklearn has no
    exact no-penalty Newton fit that stays inside a laptop's memory on 21 million loans).
  * LightGBM: numbers as stored; native categorical features with fixed codes
    (documented category order); a blank category is missing (NaN); a value absent from training (special program R, or an
    undocumented value) gets the reference category's code. Optuna's LightGBMTuner unchanged (6 steps, up to 68
    setups); trees by early stopping on the validation raw log loss (patience 100, at most 1000 trees); fixed seeds;
    the Optuna study in SQLite and every setup's booster in model_dir so a rerun resumes.
  * Adjustment: isotonic, for both models. scikit-learn's isotonic regression of the outcome on the
    raw score, tied scores pooled; neighbouring steps merged from the low-risk end until each holds at least 11 bad
    and 11 good loans; flat inside a step, a straight line between steps, the nearest step's value outside the
    fitted range. Setups are compared on the 5-fold cross-fitted validation log loss (fold = MD5 of agency and loan
    ID, modulo 5), computed ONCE per setup at its early-stopped tree count (LightGBM metric name "custom"). The
    adjustment that is saved is fitted on all validation loans.
  * Report outputs 1-7 as masked tables (counts 1-10 show as "1-10"; no rate or average from a masked count),
    the saved model files, and the shape check on validation loans: Logistic Regression's log-odds
    contribution and LightGBM's average SHAP value (pred_contrib) per distinct value of DTI, combined LTV and MI %.

Files (full mode; --smoke writes the *_smoke variants)
  outputs/models/logistic_regression.json   coefficients (original units and standardized), fills, references,
                                            columns, convergence
  outputs/models/fill_values.csv            the fill value of each of the 17 variables
  outputs/models/lightgbm_model.txt         the chosen booster (best iteration)
  outputs/models/lightgbm_settings.json     chosen settings, trees, seed, threads, feature order, category codes
  outputs/models/isotonic_calibration.json  both models' isotonic steps and the validation measures
  outputs/models/lightgbm_trials.csv        all setups (output 3)
  outputs/models/shape_check_<variable>.csv output 4 (dti, cltv, mi_pct)
  outputs/models/beyond_range.csv, unseen_categories.csv   outputs 5 and 6
  outputs/models/optuna_study.sqlite, outputs/models/lightgbm_trials/<n>.pkl + <n>.json   resume files
  logs/fit_models_report.txt, logs/fit_models_progress.log, logs/fit_models_error_detail.txt (stack frames only)

Usage (from the repository root):
    python code/fit_models.py [--smoke] [--threads 8] [--memory-limit 2GB]
  --smoke: the real dataset, a fixed 1-in-50 hash sample of training loans, validation loans and test inputs.
  A rerun of the same command continues the LightGBM setup search at the next unfinished setup.
Exit codes: 0 SUCCESS; 1 FAILURE; 130 interrupted (rerun to resume).
"""

import argparse
import json
import math
import pickle
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import optuna
import scipy
from scipy.stats import rankdata
from sklearn.isotonic import isotonic_regression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset as bd  # noqa: E402  (reused: check lines, report sections, error detail)
import check_raw  # noqa: E402
import explore as ex  # noqa: E402  (reused: dataset connection, variables, masking helpers, CSV writing)
import inventory as inv  # noqa: E402  (reused: progress log, report file, 1-10 masking)
import load_raw  # noqa: E402
from model_core import (CAT_COLS, CATEGORIES, CHUNK, DOLLARS, FLAG, INCOMPLETE, LABEL, LEVELS, LIMITS, N_FOLDS,  # noqa: E402,F401
                        NUM_COLS, NUMBERS, RATIO, SOURCE, VARIABLES, build_design, design_columns, expit, fold_sql,
                        isotonic_apply, lgb_matrix, limit_sql, lr_logodds, model_view_sql)
# (from model_core.py: the 17 variables, limits, constants, the model
#  view's SQL text, and the design / LightGBM matrix / isotonic-apply / log-odds routines)
from check_raw import out, out_table, section  # noqa: E402
from inventory import StopRun, lit, m, mc, progress  # noqa: E402
from load_raw import LOG_DIR, PROJECT_ROOT, hms  # noqa: E402

SCRIPT_VERSION = "2"                    # 2: isotonic adjustment; loan amount as a percent of its conforming limit plus an at-the-limit flag
MASK = ex.MASK
MISSING = ex.MISSING
SEED = 1                                # LightGBM seed and Optuna's optuna_seed
NUM_BOOST_ROUND = 1000                  # LightGBMTuner's own default maximum
EARLY_STOPPING_ROUNDS = 100             # optuna-examples lightgbm_tuner_simple.py
TUNER_STEPS = [("feature_fraction", 7), ("num_leaves", 20), ("bagging", 10), ("feature_fraction_stage2", 6),
               ("regularization_factors", 20), ("min_child_samples", 5)]     # the tuner's own step names
N_SETUPS = sum(n for _, n in TUNER_STEPS)   # 68
SMOKE_MOD = 50                          # --smoke: hash(agency, loan_id) % 50 = 0
LR_MAX_ITER = 50
LR_GRAD_TOL = 1e-6                      # largest absolute gradient of the log-likelihood
LR_STEP_TOL = 1e-8                      # largest |step_j| / max(1, |beta_j|)
MIN_SIDE = 11                           # isotonic merge rule: at least this many bad and this many good loans per step
SHAPE_VARS = ["dti", "cltv", "mi_pct"]  # the shape check's variables (report output 4)



WORK_ROOT = PROJECT_ROOT    # tests re-point these two before calling main()
LOGS = LOG_DIR
MODE = "full"
P = {}
FAILS = bd.FAILS
WRITTEN = []
_TRIAL_LIMIT = None         # tests only: stop (exit 130) after this many setups finished in this run
_NUM_BOOST_ROUND = None     # tests only: override NUM_BOOST_ROUND


def configure(mode):
    """Paths of this mode. explore.py's connection helpers are reused by re-pointing their module paths."""
    global MODE
    MODE = mode
    sfx = "_smoke" if mode == "smoke" else ""
    ex.WORK_ROOT = WORK_ROOT
    ex.configure(mode)
    P.clear()
    P.update(db=WORK_ROOT / "loan_dataset.duckdb", out=WORK_ROOT / "outputs" / f"models{sfx}", spill=WORK_ROOT / f"fit_models_work{sfx}.tmp",
             report=LOGS / f"fit_models_report{sfx}.txt", progress=LOGS / "fit_models_progress.log",
             error=LOGS / "fit_models_error_detail.txt")
    P.update(study=P["out"] / "optuna_study.sqlite", model_dir=P["out"] / "lightgbm_trials")
    load_raw.DB_PATH = P["db"]                          # both modes read the real dataset (smoke = a hash sample of it)
    ex.P["spill"] = P["spill"]                          # explore.open_dataset spills here
    inv.REPORT_PATH = P["report"]
    inv.PROGRESS_PATH = P["progress"]
    bd.P["error"] = P["error"]


# ---------------------------------------------------------------- views and data

def sample_sql():
    return f" AND hash(agency || '|' || loan_id) % {SMOKE_MOD} = 0" if MODE == "smoke" else ""


def var_sql():
    cols = []
    for c, _, kind, _ in VARIABLES:
        cols.append(f"CAST({c} AS VARCHAR) AS {c}" if kind == "category" else f"CAST({c} AS DOUBLE) AS {c}")
    return ", ".join(cols)


def create_model_view(con):
    """model_loans = kept_loans plus the two calculated model variables and the cross-fitting fold.
    Percent of limit = 100 x original UPB / limit, 2 decimals. At-the-limit flag = Y when original UPB rounded to the
    nearest $1,000 equals the limit rounded to the nearest $1,000, else N. Both blank when the limit is (units blank)."""
    con.execute(model_view_sql())


def create_views(con):
    """train and validation: bad + the 17 variables + dollars (Q3's weight) + fold. test_inputs: the 17 variables
    ONLY (no outcome column)."""
    create_model_view(con)
    for name, label in (("train", "train"), ("validation", "validation")):
        con.execute(f"CREATE OR REPLACE TEMP VIEW {name} AS SELECT (outcome = 'bad')::TINYINT AS bad, {var_sql()}, "
                    f"CAST({DOLLARS} AS DOUBLE) AS dollars, fold FROM {SOURCE} WHERE label = {lit(label)}{sample_sql()}")
    con.execute(f"CREATE OR REPLACE TEMP VIEW test_inputs AS SELECT {var_sql()} FROM {SOURCE} "
                f"WHERE label = 'test'{sample_sql()}")


def training_categories(con):
    """Per category: {value: training loans} for every value present in training (NULL excluded)."""
    res = {}
    for c in CAT_COLS:
        got = dict(con.execute(f"SELECT {c}, count(*) FROM train WHERE {c} IS NOT NULL GROUP BY 1").fetchall())
        res[c] = {str(k): int(v) for k, v in got.items()}
    return res


def separated_values(con):
    """Category values whose training loans are all good or all bad: their Logistic Regression coefficient has no
    finite maximum-likelihood value (separation), so Newton's method cannot converge. Returns [(col, value, loans, bad)]."""
    res = []
    for c in CAT_COLS:
        for v, n, b in con.execute(f"SELECT {c}, count(*), sum(bad) FROM train WHERE {c} IS NOT NULL GROUP BY 1 ORDER BY 1").fetchall():
            if b == 0 or b == n:
                res.append((c, str(v), int(n), int(b)))
    return res


def level_codes(train_cats):
    """Fixed integer codes per category: documented category order first, then any training value outside it (sorted).
    Values seen only outside training that are not documented values get no code (-2 in SQL) and score as the reference."""
    codes, refs, present = {}, {}, {}
    for c in CAT_COLS:
        counts = train_cats[c]
        order = list(LEVELS[c]) + sorted(v for v in counts if v not in LEVELS[c])
        codes[c] = {v: i for i, v in enumerate(order)}
        present[c] = [v for v in order if counts.get(v, 0) > 0]
        refs[c] = max(present[c], key=lambda v: counts[v]) if present[c] else order[0]   # ties: documented category order
    return codes, refs, present


def code_sql(c, codes):
    whens = " ".join(f"WHEN {c} = {lit(v)} THEN {i}" for v, i in codes[c].items())
    return f"CAST(CASE WHEN {c} IS NULL THEN -1 {whens} ELSE -2 END AS TINYINT) AS {c}"


def fetch_rows(con, view, codes, with_bad, extra=()):
    """One streamed query -> numbers (n x 7 float32, NaN = blank), codes (n x 10 int8: -1 blank, -2 undocumented),
    bad (n int8 or None). Column order is NUM_COLS then CAT_COLS; the row order is the query's, used only inside
    this run. Never called for test_inputs rows (outputs 5 and 6 count test inputs in SQL).
    extra: names of further number columns of the view (e.g. "dollars", "fold") read in the same query, so they share
    the row order; when given, a 4th value is returned: {name: float64 array}."""
    n = con.execute(f"SELECT count(*) FROM {view}").fetchall()[0][0]     # fetchall: DuckDB 2.0 needs the result finished
    sel = ([f"CAST({c} AS DOUBLE) AS {c}" for c in NUM_COLS] + [code_sql(c, codes) for c in CAT_COLS]
           + [f"CAST({c} AS DOUBLE) AS {c}" for c in extra] + (["bad"] if with_bad else []))
    num = np.empty((n, len(NUM_COLS)), dtype=np.float32)
    cat = np.empty((n, len(CAT_COLS)), dtype=np.int8)
    bad = np.empty(n, dtype=np.int8) if with_bad else None
    more = {c: np.empty(n, dtype=np.float64) for c in extra}
    reader = con.sql(f"SELECT {', '.join(sel)} FROM {view}").to_arrow_reader(CHUNK)   # streamed, no extra copy
    pos = 0
    for batch in reader:
        k = batch.num_rows
        for j in range(len(NUM_COLS)):
            num[pos:pos + k, j] = batch.column(j).to_numpy(zero_copy_only=False)
        for j in range(len(CAT_COLS)):
            cat[pos:pos + k, j] = batch.column(len(NUM_COLS) + j).to_numpy(zero_copy_only=False)
        for j, c in enumerate(extra):
            more[c][pos:pos + k] = batch.column(len(NUM_COLS) + len(CAT_COLS) + j).to_numpy(zero_copy_only=False)
        if with_bad:
            bad[pos:pos + k] = batch.column(len(sel) - 1).to_numpy(zero_copy_only=False)
        pos += k
    if pos != n:
        raise RuntimeError(f"{view}: streamed {pos} rows, counted {n}")
    return (num, cat, bad, more) if extra else (num, cat, bad)


# ---------------------------------------------------------------- Logistic Regression

def log_loss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def auc(y, score):
    """Mann-Whitney AUC with average ranks for ties; None if one class is absent."""
    y = np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------------------------------------------------------- isotonic adjustment

def _isotonic_sorted(zs, ys):
    """The adjustment fitted on loans already sorted by raw score (zs ascending, ys 0/1). Returns the steps as a
    dict of arrays, lowest risk first: score_low, score_high (the lowest and highest fitted score of the step),
    value (its bad rate), loans, bad."""
    n = len(zs)
    if n == 0:
        raise ValueError("isotonic fit on no loans")
    starts = np.flatnonzero(np.r_[True, zs[1:] != zs[:-1]])              # (a) loans with the same score are pooled
    ends = np.r_[starts[1:], n]
    cum_bad = np.r_[0, np.cumsum(ys, dtype=np.int64)]
    cnt = (ends - starts).astype(np.float64)
    bad = (cum_bad[ends] - cum_bad[starts]).astype(np.float64)
    fitted = isotonic_regression(bad / cnt, sample_weight=cnt, increasing=True)
    first = np.flatnonzero(np.r_[True, fitted[1:] != fitted[:-1]])       # a step = one run of equal fitted values
    lo = starts[first]                                                   # row range of each step
    hi = np.r_[lo[1:], n]
    s_bad = cum_bad[hi] - cum_bad[lo]
    s_loans = hi - lo
    # The steps once more in whole numbers (bad x loans cross-products): two neighbours whose bad rates are equal, or
    # out of order by float rounding inside the library, are one step. The result is the exact isotonic fit.
    steps = []                                                           # [bad, loans, first row, row after the last]
    for k in range(len(lo)):
        cur = [int(s_bad[k]), int(s_loans[k]), int(lo[k]), int(hi[k])]
        while steps and steps[-1][0] * cur[1] >= cur[0] * steps[-1][1]:
            top = steps.pop()
            cur = [top[0] + cur[0], top[1] + cur[1], top[2], cur[3]]
        steps.append(cur)
    # (b) merge rule: from the lowest-risk step, merge neighbours until the step holds at least MIN_SIDE bad and
    # MIN_SIDE good loans, then start the next step; a remainder at the high-risk end that falls short joins the
    # step below it.
    cuts, acc_bad, acc_loans, begin = [], 0, 0, 0
    for s_b, s_n, _, s_end in steps:
        acc_bad += s_b
        acc_loans += s_n
        if acc_bad >= MIN_SIDE and acc_loans - acc_bad >= MIN_SIDE:
            cuts.append((begin, s_end))
            begin, acc_bad, acc_loans = s_end, 0, 0
    if begin < n:
        if cuts:
            cuts[-1] = (cuts[-1][0], n)
        else:
            cuts.append((0, n))
    a = np.array([c[0] for c in cuts], dtype=np.int64)
    b = np.array([c[1] for c in cuts], dtype=np.int64)
    step_bad = cum_bad[b] - cum_bad[a]
    step_loans = b - a
    return {"score_low": zs[a].astype(np.float64), "score_high": zs[b - 1].astype(np.float64),
            "value": step_bad / step_loans, "loans": step_loans, "bad": step_bad}


def isotonic_fit(z, y):
    """Isotonic adjustment of the outcome y (0/1) on the raw score z, with the merge rule."""
    z = np.asarray(z, dtype=np.float64)
    order = np.argsort(z, kind="stable")
    return _isotonic_sorted(z[order], np.asarray(y)[order])


def cross_fitted_risk(z, y, fold):
    """Each loan's adjusted risk from an isotonic fit on the loans of the OTHER folds."""
    z = np.asarray(z, dtype=np.float64)
    order = np.argsort(z, kind="stable")
    zs, ys, fs = z[order], np.asarray(y)[order], np.asarray(fold)[order]
    res = np.empty(len(z))
    for k in range(N_FOLDS):
        hold = fs == k
        if hold.any():
            res[order[hold]] = isotonic_apply(_isotonic_sorted(zs[~hold], ys[~hold]), zs[hold])
    return res


def cross_fitted_logloss(z, y, fold):
    return log_loss(np.asarray(y, dtype=np.float64), cross_fitted_risk(z, y, fold))


def adjust(z, y, fold):
    """The saved adjustment (fitted on all the loans given) and its measures. Returns (cal, measures, risk)."""
    yf = np.asarray(y, dtype=np.float64)
    cal = isotonic_fit(z, y)
    risk = isotonic_apply(cal, z)
    good = cal["loans"] - cal["bad"]
    measures = {"logloss_before": log_loss(yf, expit(z)), "logloss_after": log_loss(yf, risk),
                "logloss_cross_fitted": cross_fitted_logloss(z, y, fold), "auc": auc(y, z), "steps": int(len(cal["value"])),
                "min_bad_in_a_step": int(cal["bad"].min()), "min_good_in_a_step": int(good.min()),
                "lowest_value": float(cal["value"][0]), "highest_value": float(cal["value"][-1])}
    return cal, measures, risk


def cal_json(cal):
    """The adjustment as saved: scores and values only (every step holds at least 11 bad and 11 good loans)."""
    return {k: np.asarray(cal[k], dtype=np.float64).tolist() for k in ("score_low", "score_high", "value")}


CAL_RULE = ("risk = the step's value when score_low <= raw score <= score_high; between two steps, the straight line "
            "from (score_high, value) of the lower step to (score_low, value) of the upper step; below the first or "
            "above the last step, that step's value. Raw score = log-odds. Fitted on validation loans.")


def fill_values(con):
    """Training fills: lower median of each number (quantile_disc), most common value of each category."""
    sel = ", ".join(f"quantile_disc({c}, 0.5)" for c in NUM_COLS)
    nums = con.execute(f"SELECT {sel} FROM train").fetchone()
    return {c: (None if v is None else float(v)) for c, v in zip(NUM_COLS, nums)}


def constant_columns(X):
    """Indices of columns (other than the intercept) with no variation: they get no coefficient (fixed at 0)."""
    return [j for j in range(1, X.shape[1]) if float(X[:, j].min()) == float(X[:, j].max())]


def newton_logistic(X, y, max_iter=LR_MAX_ITER, dropped=()):
    """No-penalty logistic regression by Newton's method (IRLS), chunked, float64 sums. Columns in `dropped` are
    left out of the fit (coefficient 0). Returns a dict with the coefficients (standardized columns, one per column
    of X), iterations, final gradient and step sizes, converged, log-likelihood."""
    n, k_all = X.shape
    keep = [j for j in range(k_all) if j not in set(dropped)]
    k = len(keep)
    y = y.astype(np.float64)
    rate = float(y.mean())
    beta = np.zeros(k)
    beta[0] = math.log(rate / (1 - rate)) if 0 < rate < 1 else 0.0

    def pass_(beta, need_hessian=True):
        g = np.zeros(k)
        H = np.zeros((k, k))
        ll = 0.0
        for i in range(0, n, CHUNK):
            Xc = X[i:i + CHUNK][:, keep].astype(np.float64) if k < k_all else X[i:i + CHUNK].astype(np.float64)
            yc = y[i:i + CHUNK]
            eta = Xc @ beta
            p = expit(eta)
            ll += float(np.sum(yc * eta - np.logaddexp(0.0, eta)))
            g += Xc.T @ (yc - p)
            if need_hessian:
                H += Xc.T @ (Xc * (p * (1 - p))[:, None])
        return ll, g, H

    info = {"iterations": 0, "converged": False, "halvings": 0}
    ll, g, H = pass_(beta)
    for it in range(1, max_iter + 1):
        info["iterations"] = it
        step = np.linalg.solve(H, g)
        rel = float(np.max(np.abs(step) / np.maximum(1.0, np.abs(beta))))
        grad = float(np.max(np.abs(g)))
        info.update(max_abs_gradient=grad, max_rel_step=rel, log_likelihood=ll)
        if grad < LR_GRAD_TOL and rel < LR_STEP_TOL:
            info["converged"] = True
            break
        new_beta = beta + step
        ll_new, g_new, H_new = pass_(new_beta)
        halvings = 0
        while ll_new < ll - 1e-9 and halvings < 20:      # safeguard: never accept a worse log-likelihood
            halvings += 1
            step = step / 2
            new_beta = beta + step
            ll_new, g_new, H_new = pass_(new_beta)
        info["halvings"] += halvings
        beta, ll, g, H = new_beta, ll_new, g_new, H_new
    else:
        grad = float(np.max(np.abs(g)))
        step = np.linalg.solve(H, g)
        info.update(max_abs_gradient=grad, max_rel_step=float(np.max(np.abs(step) / np.maximum(1.0, np.abs(beta)))),
                    log_likelihood=ll)
    full = np.zeros(k_all)
    full[keep] = beta
    info["beta"] = full
    info["dropped_columns"] = list(dropped)
    return info


def to_original(beta, means, sds):
    """Standardized coefficients -> original units (numbers at columns 1..7)."""
    orig = beta.copy()
    nn = len(NUM_COLS)
    orig[1:1 + nn] = beta[1:1 + nn] / sds
    orig[0] = beta[0] - float(np.sum(beta[1:1 + nn] * means / sds))
    return orig


# ---------------------------------------------------------------- LightGBM

class SetupScore:
    """How setups are compared: the 5-fold cross-fitted isotonic validation log loss at the
    setup's early-stopped tree count, computed ONCE per setup.

    LightGBM's own early stopping (patience on the validation raw log loss, the first metric) runs inside this
    callback, unchanged. Every training run ends with its EarlyStopException (patience reached, or the last round),
    which names the best round. Only then are the validation loans scored with the trees up to that round and the
    cross-fitted loss calculated; it replaces the placeholder of the metric "custom" in the best round's scores,
    which is where the tuner reads a setup's value. The result is the same number as calculating the loss after
    every tree and reading it at the best round; only the runtime differs."""
    before_iteration = False

    def __init__(self, X_v, y, fold):
        self.X_v, self.y, self.fold = X_v, np.asarray(y), np.asarray(fold)
        self.stopper = lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False)
        self.order = 30                 # lgb.early_stopping's own order
        self.rounds, self.value, self.score_seconds = 0, math.nan, 0.0

    @staticmethod
    def feval(preds, dataset):
        """The metric "custom": a placeholder on every round (never used for early stopping: first metric only)."""
        return "custom", math.nan, False

    def __call__(self, env):
        self.rounds = env.iteration - env.begin_iteration + 1
        try:
            self.stopper(env)
        except lgb.callback.EarlyStopException as stop:
            t0 = time.time()
            z = env.model.predict(self.X_v, raw_score=True, num_iteration=stop.best_iteration + 1)
            self.value = cross_fitted_logloss(z, self.y, self.fold)
            self.score_seconds = time.time() - t0
            scores = [(r.dataset_name, r.metric_name, self.value if r.metric_name == "custom" else r.metric_value, r.maximize)
                      for r in stop.best_score]
            raise lgb.callback.EarlyStopException(stop.best_iteration, scores) from None


def base_params(threads):
    return {"objective": "binary", "metric": ["binary_logloss", "custom"], "verbosity": -1, "seed": SEED,
            "deterministic": True, "num_threads": int(threads), "feature_pre_filter": False}


def num_boost_round():
    return _NUM_BOOST_ROUND or NUM_BOOST_ROUND


def trial_sidecar(n):
    return P["model_dir"] / f"{n}.json"


def trial_params(trial):
    for key, val in trial.system_attrs.items():
        if key.endswith("lgbm_params"):
            return json.loads(val)
    return {}


def record_trial(study, trial, scorer, started, counter):
    """Optuna callback after each setup: read LightGBM's own answer from the pickled booster, store a sidecar JSON
    (trees, raw and cross-fitted loss, cap flag), log progress, honor the test-only trial limit."""
    if trial.state != optuna.trial.TrialState.COMPLETE:
        raise RuntimeError(f"setup {trial.number} did not finish with a value (state {trial.state.name})")
    with open(P["model_dir"] / f"{trial.number}.pkl", "rb") as f:
        booster = pickle.load(f)
    best = int(booster.best_iteration)
    rounds = scorer.rounds                   # rounds trained; the returned booster is cut at the best round
    scores = booster.best_score["validation"]
    value = float(scores["custom"])
    if not math.isfinite(value) or value != scorer.value or trial.value != value:
        raise RuntimeError(f"setup {trial.number}: the cross-fitted loss was not recorded as the setup's value")
    side = {"trial": trial.number, "step": next((v for k, v in trial.system_attrs.items() if k.endswith("step_name")), ""),
            "params": trial_params(trial), "trees": best, "rounds_trained": rounds, "hit_cap": rounds >= num_boost_round(),
            "raw_logloss": float(scores["binary_logloss"]), "cross_fitted_logloss": value,
            "scoring_seconds": scorer.score_seconds,
            "seconds": (trial.datetime_complete - trial.datetime_start).total_seconds() if trial.datetime_complete else None}
    trial_sidecar(trial.number).write_text(json.dumps(side, indent=1), encoding="utf-8")
    counter["done"] += 1
    n_done_all = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    per = (time.time() - started) / counter["done"]
    remaining = per * (N_SETUPS - n_done_all)
    progress(f"  setup {n_done_all}/{N_SETUPS} ({side['step']}): {best} trees{' (hit the cap)' if side['hit_cap'] else ''}, "
             f"raw {side['raw_logloss']:.6f}, cross-fitted {side['cross_fitted_logloss']:.6f}, {hms(side['seconds'] or 0)} | "
             f"remaining ~{hms(remaining)} (finish ~{datetime.fromtimestamp(time.time() + remaining):%m-%d %H:%M})")
    if _TRIAL_LIMIT is not None and counter["done"] >= _TRIAL_LIMIT:
        progress(f"STOP (test hook): {_TRIAL_LIMIT} setup(s) finished in this run; rerun to resume.")
        raise StopRun(130)


def run_tuner(dtrain, dvalid, X_v, y_valid, fold, threads):
    """Optuna's LightGBMTuner, unchanged, on the validation set only; resumable through the SQLite study + model_dir.
    A study made by another version of this script is never continued."""
    import optuna_integration.lightgbm as olgb
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    P["model_dir"].mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name="lightgbm_tuner", storage="sqlite:///" + P["study"].as_posix(),
                                direction="minimize", load_if_exists=True)
    signature = {"script_version": SCRIPT_VERSION, "features": NUM_COLS + CAT_COLS}
    if study.trials and study.user_attrs.get("fit_models") != signature:
        progress(f"STOP: {P['study']} holds a setup search made by another version of fit_models.py or on other "
                 f"variables. Move {P['out']} away (do not mix two searches), then rerun.")
        raise StopRun(1)
    study.set_user_attr("fit_models", signature)
    done_before = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    progress(f"LightGBM setup search: {done_before} of {N_SETUPS} setups already finished (study {P['study'].name})")
    scorer = SetupScore(X_v, y_valid, fold)
    counter = {"done": 0}
    started = time.time()
    tuner = olgb.LightGBMTuner(
        base_params(threads), dtrain, num_boost_round=num_boost_round(), valid_sets=[dvalid], valid_names=["validation"],
        feval=scorer.feval, callbacks=[scorer],
        study=study, optuna_callbacks=[lambda s, t: record_trial(study, t, scorer, started, counter)],
        model_dir=str(P["model_dir"]), show_progress_bar=False, optuna_seed=SEED)
    tuner.run()
    return tuner, study


def trials_table(study):
    """One row per finished setup from the sidecars (a missing sidecar is rebuilt from the pickle)."""
    rows = []
    for t in sorted(study.trials, key=lambda t: t.number):
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        side_path = trial_sidecar(t.number)
        if side_path.exists():
            rows.append(json.loads(side_path.read_text(encoding="utf-8")))
        else:
            rows.append({"trial": t.number, "step": "", "params": trial_params(t), "trees": None, "rounds_trained": None,
                         "hit_cap": None, "raw_logloss": None, "cross_fitted_logloss": t.value, "scoring_seconds": None,
                         "seconds": None, "sidecar_missing": True})
    return rows


TUNED = ["feature_fraction", "num_leaves", "bagging_fraction", "bagging_freq", "lambda_l1", "lambda_l2", "min_child_samples"]


def expected_setups(rows):
    """Setups per step as the tuner defines them: fixed counts, except feature_fraction_stage2, whose 6-point grid
    (best feature_fraction +- 0.08) keeps only the values within 0.4-1.0. Returns {step: expected count}."""
    want = dict(TUNER_STEPS)
    before = [r for r in rows if r["step"] in ("feature_fraction", "num_leaves", "bagging") and r["cross_fitted_logloss"] is not None]
    if before:
        best = min(before, key=lambda r: (r["cross_fitted_logloss"], r["trial"]))["params"].get("feature_fraction", 1.0)
        grid = np.linspace(best - 0.08, best + 0.08, want["feature_fraction_stage2"]).tolist()
        want["feature_fraction_stage2"] = len([v for v in grid if 0.4 <= v <= 1.0])
    return want


# ---------------------------------------------------------------- report sections

def rep_header(args, meta):
    out(f"fit_models.py v{SCRIPT_VERSION}  {datetime.now():%Y-%m-%d %H:%M}  mode {MODE}  dataset {P['db'].name} (read-only)")
    out("Summaries only: no loan IDs, no row-level values. Counts from 1 to 10 show as '1-10'; no rate or average is "
        "shown for a masked count. The test set is not scored here; its outcome is never read (test inputs are counted "
        "in outputs 5 and 6 only).")
    out(f"Dataset built by build_dataset.py v{meta.get('script_version', '?')} at {meta.get('built_at', '?')} "
        f"(mode {meta.get('mode', '?')}); specification file SHA-256 {meta.get('decision_log_sha256', '?')}.")
    import optuna_integration
    out(f"DuckDB memory limit {args.memory_limit}, threads {args.threads} (DuckDB and LightGBM), seed {SEED} (LightGBM seed and "
        f"Optuna's optuna_seed), LightGBM deterministic = true. numpy {np.__version__}, scipy {scipy.__version__}, duckdb "
        f"{duckdb.__version__}, lightgbm {lgb.__version__}, optuna {optuna.__version__}, optuna-integration "
        f"{optuna_integration.__version__}.")
    out(f"Early stopping: patience {EARLY_STOPPING_ROUNDS} rounds on the validation raw log loss, at most {num_boost_round()} "
        f"trees. Setups compared on the {N_FOLDS}-fold cross-fitted isotonic validation log loss, calculated "
        f"once per setup at its early-stopped tree count.")
    out(f"Adjustment: isotonic on the raw score, tied scores pooled; steps merged from the low-risk end "
        f"until each holds at least {MIN_SIDE} bad and {MIN_SIDE} good loans; flat inside a step, a straight line between "
        f"steps, the nearest step's value outside the fitted range. Fold = MD5 of agency and loan ID, modulo {N_FOLDS}.")
    out("Validation loans do three jobs (early stopping, the isotonic adjustment, the choice of setup). The cross-fitted "
        "loss adjusts each loan with a fit that did not use it; the 'after adjustment' loss of the saved adjustment "
        "(fitted on all validation loans) is in-sample.")
    out(f"Model variables: {len(VARIABLES)}. Loan amount in dollars is replaced by {LABEL[RATIO]} "
        f"(100 x amount / FHFA basic conforming limit for the loan's units and origination year, 2 decimals) and "
        f"{LABEL[FLAG]} (amount = limit, both rounded to the nearest $1,000).")
    out(f"Outputs: {P['out']}.")
    if MODE == "smoke":
        out(f"SMOKE MODE: the real dataset, a fixed 1-in-{SMOKE_MOD} hash sample of training loans, validation loans and test "
            "inputs; counts are not the full populations.")


def rep_population(con):
    section("1. Run info: training and validation loans and bad rates (both agencies pooled)")
    table, counts = [], {}
    for name in ("train", "validation"):
        n, b = con.execute(f"SELECT count(*), coalesce(sum(bad), 0) FROM {name}").fetchone()
        counts[name] = (int(n), int(b))
        table.append([name, m(n), m(b), ex.pct_text(n, b)])
    nt = con.execute("SELECT count(*) FROM test_inputs").fetchone()[0]
    out_table(["loans", "count", "bad loans", "bad rate"], table, left={0})
    out(f"  Test inputs (counted only, never scored): {mc(nt)} loans.")
    cols = [r[0] for r in con.execute("DESCRIBE test_inputs").fetchall()]
    bd.check_line("the test_inputs view has no outcome column", "outcome" not in cols and "bad" not in cols)
    no_limit = con.execute(f"SELECT count(*) FROM {SOURCE} WHERE label IN ('train', 'validation', 'test'){sample_sql()} "
                           f"AND num_units IS NOT NULL AND {DOLLARS} IS NOT NULL AND {RATIO} IS NULL").fetchone()[0]
    bd.check_line("every loan with a number of units has a conforming loan limit (its origination year and units are in "
                  "the limits table)", no_limit == 0)
    flags = [con.execute(f"SELECT count(*) FILTER ({FLAG} = 'Y'), count(*) FILTER ({FLAG} IS NULL) FROM {v}").fetchone()
             for v in ("train", "validation", "test_inputs")]
    out(f"  {LABEL[FLAG]} = Y: training {mc(flags[0][0])}, validation {mc(flags[1][0])}, test inputs {mc(flags[2][0])}. "
        f"{LABEL[RATIO]} and {LABEL[FLAG]} blank (number of units blank): training {mc(flags[0][1])}, validation "
        f"{mc(flags[1][1])}, test inputs {mc(flags[2][1])}.")
    bd.check_line("there are training loans with both outcomes", counts["train"][0] > counts["train"][1] > 0)
    bd.check_line("there are validation loans with both outcomes", counts["validation"][0] > counts["validation"][1] > 0)
    if FAILS:
        raise StopRun(1)
    return counts


def measure_rows(ms):
    """The validation measures of one model as report rows. A step's value is shown only when every step holds at
    least 11 bad and 11 good loans (the merge rule; the check line reports it)."""
    ok = ms["min_bad_in_a_step"] >= MIN_SIDE and ms["min_good_in_a_step"] >= MIN_SIDE
    return [["log loss before the isotonic adjustment", f"{ms['logloss_before']:.6f}"],
            ["log loss after the isotonic adjustment (fitted on all validation loans: in-sample)", f"{ms['logloss_after']:.6f}"],
            [f"log loss after a {N_FOLDS}-fold cross-fitted isotonic adjustment", f"{ms['logloss_cross_fitted']:.6f}"],
            ["AUC (on the raw score)", "-" if ms["auc"] is None else f"{ms['auc']:.6f}"],
            ["isotonic steps", str(ms["steps"])],
            ["fewest bad / fewest good loans in a step", f"{m(ms['min_bad_in_a_step'])} / {m(ms['min_good_in_a_step'])}"],
            ["lowest / highest step value", f"{100 * ms['lowest_value']:.4f}% / {100 * ms['highest_value']:.4f}%" if ok else "-"]]


def check_steps(name, ms):
    bd.check_line(f"{name}: every isotonic step holds at least {MIN_SIDE} bad and {MIN_SIDE} good validation loans",
                  ms["min_bad_in_a_step"] >= MIN_SIDE and ms["min_good_in_a_step"] >= MIN_SIDE)


def rep_logistic(lr, measures, fills_num, fills_cat, refs, train_cats, n_inc_train, n_inc_val):
    section("2. Logistic Regression: fit, fills, incomplete applications, validation measures, isotonic adjustment")
    conv = lr["converged"]
    out(f"  Newton (IRLS) iterations {lr['iterations']}, step halvings {lr['halvings']}; largest |gradient| "
        f"{lr['max_abs_gradient']:.3e} (tolerance {LR_GRAD_TOL:g}); largest relative step {lr['max_rel_step']:.3e} "
        f"(tolerance {LR_STEP_TOL:g}); log-likelihood {lr['log_likelihood']:.4f}.")
    bd.check_line("Logistic Regression converged", conv)
    sep = lr.get("separated", [])
    bd.check_line("no category value has all-good or all-bad training loans (separation)", not sep)
    for c, v, n, b in sep:
        out(f"  Separation: {LABEL[c]} = {v} has {mc(n)} training loans, {mc(b)} bad: its coefficient has no finite value, so the "
            f"fit cannot converge (the dummy is kept; the coefficient reported is where Newton's method stopped).")
    if lr.get("dropped_columns"):
        out(f"  Columns constant in training (no variation), coefficient fixed at 0: " + "; ".join(lr["dropped_names"]))
    out()
    out("  Fill values (training lower median / most common training value) and reference categories:")
    rows = [[LABEL[c], "number", inv.fmt_value(fills_num[c]), "-"] for c in NUM_COLS]
    rows += [[LABEL[c], "category", fills_cat[c], refs[c]] for c in CAT_COLS]
    out_table(["variable", "kind", "fill value", "reference"], rows, left={0, 1, 2, 3})
    for c in CAT_COLS:
        extra = sorted(v for v in train_cats[c] if v not in LEVELS[c])
        bd.check_line(f"{LABEL[c]}: every training value is a documented category value", not extra)
    out()
    out(f"  Loans with incomplete application = 1 (any of the {len(VARIABLES)} variables blank): training {mc(n_inc_train)}, "
        f"validation {mc(n_inc_val)}.")
    out()
    out_table(["validation measure", "value"], measure_rows(measures), left={0})
    check_steps("Logistic Regression", measures)
    out(f"  Coefficients (original units) are in {P['out'].name}/logistic_regression.json; the isotonic steps are in "
        f"{P['out'].name}/isotonic_calibration.json. check_beyond_range.py decides the edge rule and "
        f"saves the adjustment the test uses.")


def rep_lightgbm(rows, chosen, measures, counts_hit):
    section("3. LightGBM: all setups (settings, trees, validation raw and cross-fitted log loss); the chosen setup")
    want = expected_setups(rows)
    got = {s: sum(1 for r in rows if r["step"] == s) for s, _ in TUNER_STEPS}
    out(f"  {len(rows)} setups finished of {sum(want.values())} expected (up to {N_SETUPS}). Steps in order, finished / expected: "
        + ", ".join(f"{s} {got[s]} / {want[s]}" for s, _ in TUNER_STEPS)
        + f". The feature_fraction_stage2 grid is the best feature_fraction +- 0.08 in 6 steps, keeping only values within "
          f"0.4-1.0 (the tuner's rule), so it can run fewer than 6 setups.")
    out("  learning rate = LightGBM's default; objective binary; num_leaves up to 256 (default tree depth).")
    table = []
    for r in rows:
        p = r["params"]
        table.append([r["trial"], r["step"]] + [inv.fmt_value(p.get(k)) if p.get(k) is not None else "-" for k in TUNED]
                     + [r["trees"] if r["trees"] is not None else "-", "yes" if r["hit_cap"] else "no" if r["hit_cap"] is not None else "-",
                        f"{r['raw_logloss']:.6f}" if r["raw_logloss"] is not None else "-",
                        f"{r['cross_fitted_logloss']:.6f}" if r["cross_fitted_logloss"] is not None else "-",
                        "chosen" if r["trial"] == chosen["trial"] else ""])
    out_table(["setup", "step"] + TUNED + ["trees", f"hit {num_boost_round()}", "raw log loss", "cross-fitted log loss", ""],
              table, left={1, 13})
    bd.check_line("every step ran its full number of setups", all(got[s] == want[s] for s, _ in TUNER_STEPS))
    bd.check_line("every setup has its tree count (sidecar present)", all(r.get("trees") is not None for r in rows))
    if counts_hit:
        out(f"  Setups that hit the {num_boost_round()}-tree cap (early stopping did not trigger): {counts_hit}.")
    out()
    out(f"  Chosen setup: {chosen['trial']} (lowest cross-fitted log loss; step {chosen['step']}); {chosen['trees']} trees.")
    out_table(["setting", "value"], [[k, inv.fmt_value(chosen["params"].get(k))] for k in TUNED]
              + [["learning_rate", inv.fmt_value(chosen["params"].get("learning_rate", 0.1))]], left={0})
    out_table(["validation measure", "value"], measure_rows(measures), left={0})
    check_steps("LightGBM", measures)


def shape_rows(values, lr_coef, fill, shap_col):
    """Per distinct validation value: loans, LR log-odds contribution (coef x value), mean SHAP (masked)."""
    nan = np.isnan(values)
    vals, inv_idx = np.unique(values[~nan], return_inverse=True)
    counts = np.bincount(inv_idx)
    sums = np.bincount(inv_idx, weights=shap_col[~nan])
    rows = []
    for v, n, s in zip(vals, counts, sums):
        rows.append(_shape_row(ex.num_text(v), int(n), lr_coef * float(v), s / n if n else None))
    n_miss = int(nan.sum())
    rows.append(_shape_row(MISSING, n_miss, lr_coef * fill, float(shap_col[nan].mean()) if n_miss else None))
    return rows


def _shape_row(label, n, contrib, mean_shap):
    if 1 <= n <= 10:
        return [label, MASK, f"{contrib:.6f}", ""]
    return [label, str(n), f"{contrib:.6f}", "" if n == 0 else f"{mean_shap:.6f}"]


def rep_shape(num_v, shap, coef_orig, fills_num):
    section("4. Shape check on validation loans: per distinct value, Logistic Regression's log-odds "
            "contribution (coefficient x value) and LightGBM's average SHAP value (log-odds scale)")
    out("  The (missing) row shows the contribution at the fill value; the incomplete-application coefficient is separate. "
        "Rows with 1-10 loans show no SHAP average.")
    for c in SHAPE_VARS:
        j = NUM_COLS.index(c)
        rows = shape_rows(num_v[:, j].astype(np.float64), float(coef_orig[1 + j]), fills_num[c], shap[:, j])
        path = P["out"] / f"shape_check_{c}.csv"
        ex.write_csv(path, ["value", "loans", "lr_logodds_contribution", "lgbm_mean_shap"], rows)
        WRITTEN.append(path)
        out()
        out(f"--- {LABEL[c]}: {len(rows) - 1} distinct values + missing row; coefficient {coef_orig[1 + j]:.6g} per unit")
        out_table(["value", "loans", "LR log-odds contribution", "LightGBM mean SHAP"],
                  [[r[0], ex.commas(r[1]), r[2], r[3] or "-"] for r in rows], left={0})


def rep_beyond_range(con, num_v, risk_lr, risk_gbm):
    section("5. Values beyond the training range (left as they are): training min and max, validation and "
            "test-input loans below the min or above the max, and the adjusted predicted risk of those validation loans")
    sel = ", ".join(f"min({c}), max({c})" for c in NUM_COLS)
    mm = con.execute(f"SELECT {sel} FROM train").fetchone()
    rows, csv_rows = [], []
    summary = [["all validation loans", "-", f"{len(risk_lr):,}"] + _risk_cells(risk_lr) + _risk_cells(risk_gbm)]
    for j, c in enumerate(NUM_COLS):
        lo, hi = mm[2 * j], mm[2 * j + 1]
        if lo is None:
            rows.append([LABEL[c], "-", "-", "-", "-", "-", "-"])
            continue
        counts = {}
        for view in ("validation", "test_inputs"):
            counts[view] = con.execute(f"SELECT count(*) FILTER ({c} < {lo}), count(*) FILTER ({c} > {hi}) FROM {view}").fetchone()
        rows.append([LABEL[c], inv.fmt_value(lo), inv.fmt_value(hi), m(counts["validation"][0]), m(counts["validation"][1]),
                     m(counts["test_inputs"][0]), m(counts["test_inputs"][1])])
        csv_rows.append([LABEL[c], ex.num_text(lo), ex.num_text(hi)] + [str(m(x)) for x in counts["validation"] + counts["test_inputs"]])
        x = num_v[:, j]
        for side, mask, k in (("below min", x < np.float32(lo), counts["validation"][0]), ("above max", x > np.float32(hi), counts["validation"][1])):
            n = int(mask.sum())
            bd.check_line(f"{LABEL[c]} {side}: SQL count equals the in-memory count", n == k)
            if n == 0:
                continue
            if 1 <= n <= 10:
                summary.append([LABEL[c], side, MASK] + [""] * 6)
            else:
                summary.append([LABEL[c], side, f"{n:,}"] + _risk_cells(risk_lr[mask]) + _risk_cells(risk_gbm[mask]))
    out_table(["variable", "training min", "training max", "validation below", "validation above", "test below", "test above"],
              rows, left={0})
    path = P["out"] / "beyond_range.csv"
    ex.write_csv(path, ["variable", "training_min", "training_max", "validation_below", "validation_above", "test_below", "test_above"], csv_rows)
    WRITTEN.append(path)
    out()
    out("  Adjusted predicted risk (probability) of the validation loans beyond the range, vs all validation loans:")
    out_table(["variable", "side", "loans", "LR mean", "LR min", "LR max", "LightGBM mean", "LightGBM min", "LightGBM max"],
              summary, left={0, 1})


def _risk_cells(r):
    return [f"{r.mean():.6f}", f"{r.min():.6f}", f"{r.max():.6f}"]


def rep_unseen(con, train_cats):
    section("6. Category values in validation and test inputs that never appear in training (scored as the "
            "reference category in both models; the dataset is unchanged)")
    rows = []
    for c in CAT_COLS:
        seen = ", ".join(lit(v) for v in train_cats[c]) or "''"
        for view in ("validation", "test_inputs"):
            got = con.execute(f"SELECT {c}, count(*) FROM {view} WHERE {c} IS NOT NULL AND {c} NOT IN ({seen}) "
                              f"GROUP BY 1 ORDER BY 1").fetchall()
            rows += [[LABEL[c], view, str(v), m(k)] for v, k in got]
    out_table(["variable", "loans", "value", "count"], rows or [["(none)", "-", "-", "-"]], left={0, 1, 2})
    path = P["out"] / "unseen_categories.csv"
    ex.write_csv(path, ["variable", "set", "value", "loans"], [[a, b, c, str(d)] for a, b, c, d in rows])
    WRITTEN.append(path)


def rep_files():
    section("7. Files written")
    for p in WRITTEN:
        out(f"  {p.relative_to(WORK_ROOT)}")
    out(f"  {P['study'].relative_to(WORK_ROOT)} and {P['model_dir'].relative_to(WORK_ROOT)}/<setup>.pkl + .json (resume files; "
        f"{shutil.disk_usage(WORK_ROOT).free / 1e9:,.1f} GB free on the drive)")


def save_json(name, obj):
    path = P["out"] / name
    path.write_text(json.dumps(obj, indent=1, default=_json_default), encoding="utf-8")
    WRITTEN.append(path)
    return path


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


# ---------------------------------------------------------------- main

def parse_args(argv):
    """Same options as explore.py, with defaults for model fitting."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--smoke", action="store_true", help=f"1-in-{SMOKE_MOD} hash sample of every set; writes the *_smoke outputs")
    p.add_argument("--memory-limit", default="2GB", help="DuckDB memory limit (default 2GB)")
    p.add_argument("--threads", type=int, default=8, help="threads for DuckDB and LightGBM (default 8)")
    args = p.parse_args(argv)
    if not re.fullmatch(r"\d+(\.\d+)?\s*[KMGT]i?B", args.memory_limit, re.IGNORECASE):
        p.error("--memory-limit must look like 2GB or 2048MB")
    return args


def fit_logistic(con, num, cat, y, num_v, cat_v, y_v, fold_v, codes, refs, present, train_cats):
    t0 = time.time()
    fills_num = fill_values(con)
    fills_cat = {c: max(present[c], key=lambda v: train_cats[c][v]) for c in CAT_COLS}
    fills_cat_code = {c: codes[c][fills_cat[c]] for c in CAT_COLS}
    X, means, sds, inc = build_design(num, cat, fills_num, fills_cat_code, codes, refs, present)
    n_inc_train = int(inc.sum())
    progress(f"Logistic Regression matrix: {X.shape[0]:,} loans x {X.shape[1]} columns built in {hms(time.time() - t0)}")
    t0 = time.time()
    cols, dummies = design_columns(refs, present)
    dropped = constant_columns(X)
    if dropped:
        progress(f"Logistic Regression: {len(dropped)} column(s) constant in training get no coefficient: "
                 + "; ".join(cols[j] for j in dropped))
    lr = newton_logistic(X, y, dropped=dropped)
    lr["dropped_names"] = [cols[j] for j in dropped]
    lr["separated"] = separated_values(con) if not lr["converged"] else []
    progress(f"Logistic Regression fitted in {hms(time.time() - t0)}: {lr['iterations']} iterations, converged {lr['converged']}")
    del X
    X_v, _, _, inc_v = build_design(num_v, cat_v, fills_num, fills_cat_code, codes, refs, present, means, sds)
    z_v = lr_logodds(X_v, lr["beta"])
    del X_v
    cal, measures, risk = adjust(z_v, y_v, fold_v)
    coef_orig = to_original(lr["beta"], means, sds)
    model = {"columns": cols, "coefficients_original_units": dict(zip(cols, coef_orig.tolist())),
             "coefficients_standardized": dict(zip(cols, lr["beta"].tolist())),
             "columns_constant_in_training": [cols[j] for j in dropped],
             "standardization": {LABEL[c]: {"mean": float(means[j]), "sd": float(sds[j])} for j, c in enumerate(NUM_COLS)},
             "fill_values": {**{LABEL[c]: fills_num[c] for c in NUM_COLS}, **{LABEL[c]: fills_cat[c] for c in CAT_COLS}},
             "reference_categories": {LABEL[c]: refs[c] for c in CAT_COLS},
             "dummy_columns": [{"variable": LABEL[c], "value": v} for c, v in dummies],
             "incomplete_application": {"rule": f"1 if any of the {len(VARIABLES)} model variables is blank", "training_loans": n_inc_train,
                                        "validation_loans": int(inc_v.sum())},
             "fit": {k: v for k, v in lr.items() if k not in ("beta", "dropped_columns")}, "validation": measures,
             "unseen_category_rule": "a value absent from training has no column and scores as the reference"}
    save_json("logistic_regression.json", model)
    path = P["out"] / "fill_values.csv"
    ex.write_csv(path, ["variable", "kind", "fill_value", "reference_category"],
                 [[LABEL[c], "number", ex.num_text(fills_num[c]), ""] for c in NUM_COLS] + [[LABEL[c], "category", fills_cat[c], refs[c]] for c in CAT_COLS])
    WRITTEN.append(path)
    return lr, cal, measures, fills_num, fills_cat, coef_orig, n_inc_train, int(inc_v.sum()), risk


def lgb_datasets(num, cat, y, num_v, cat_v, y_v, codes, refs, present, threads):
    """The LightGBM datasets, NOT constructed here: the tuner shallow-copies them for every setup and each copy bins
    the raw array itself (a copy of an already-constructed Dataset would free the shared handle when it dies and
    crash the next setup). The raw training array lives inside dtrain until the search ends."""
    X = lgb_matrix(num, cat, codes, refs, present)
    names = NUM_COLS + CAT_COLS
    cat_idx = list(range(len(NUM_COLS), len(names)))
    dparams = {k: v for k, v in base_params(threads).items() if k != "metric"}
    dtrain = lgb.Dataset(X, label=y, feature_name=names, categorical_feature=cat_idx, params=dparams, free_raw_data=True)
    X_v = lgb_matrix(num_v, cat_v, codes, refs, present)
    dvalid = lgb.Dataset(X_v, label=y_v, reference=dtrain, params=dparams, free_raw_data=False)
    progress(f"LightGBM arrays ready: training {X.shape[0]:,} x {X.shape[1]}, validation {X_v.shape[0]:,} (binned inside each setup)")
    return dtrain, dvalid, X_v


def fit_lightgbm(dtrain, dvalid, X_v, y_v, fold_v, codes, refs, threads):
    names = NUM_COLS + CAT_COLS
    t0 = time.time()
    tuner, study = run_tuner(dtrain, dvalid, X_v, y_v, fold_v, threads)
    progress(f"LightGBM setup search finished in {hms(time.time() - t0)}")
    rows = trials_table(study)
    booster = tuner.get_best_booster()
    best_trial = study.best_trial.number
    chosen = next(r for r in rows if r["trial"] == best_trial)
    bd.check_line("the chosen setup has the lowest cross-fitted log loss of all finished setups",
                  all(r["cross_fitted_logloss"] is None or r["cross_fitted_logloss"] >= chosen["cross_fitted_logloss"] for r in rows))
    z_v = booster.predict(X_v, raw_score=True, num_iteration=booster.best_iteration)
    cal, measures, risk = adjust(z_v, y_v, fold_v)
    bd.check_line("the chosen setup's cross-fitted log loss recalculated from the saved booster equals the search's (1e-12)",
                  abs(measures["logloss_cross_fitted"] - chosen["cross_fitted_logloss"]) < 1e-12)
    t0 = time.time()
    shap = booster.predict(X_v, pred_contrib=True, num_iteration=booster.best_iteration)
    progress(f"SHAP values (pred_contrib) for {X_v.shape[0]:,} validation loans in {hms(time.time() - t0)}")
    bd.check_line("SHAP values plus the expected value equal the raw log-odds (1e-6)", float(np.max(np.abs(shap.sum(axis=1) - z_v))) < 1e-6)
    path = P["out"] / "lightgbm_model.txt"
    booster.save_model(str(path), num_iteration=booster.best_iteration)
    WRITTEN.append(path)
    settings = {"chosen_setup": chosen["trial"], "step": chosen["step"], "params": chosen["params"], "trees": int(booster.best_iteration),
                "rounds_trained": chosen.get("rounds_trained"), "hit_cap": chosen.get("hit_cap"), "seed": SEED, "optuna_seed": SEED,
                "threads": threads, "deterministic": True, "num_boost_round": num_boost_round(), "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                "feature_order": names, "categorical_features": CAT_COLS,
                "category_codes": {LABEL[c]: codes[c] for c in CAT_COLS}, "reference_categories": {LABEL[c]: refs[c] for c in CAT_COLS},
                "unseen_category_rule": "a value absent from training gets the reference category's code; a blank category is missing (NaN)",
                "setup_comparison": f"{N_FOLDS}-fold cross-fitted isotonic validation log loss at the early-stopped tree count",
                "validation": measures}
    save_json("lightgbm_settings.json", settings)
    tpath = P["out"] / "lightgbm_trials.csv"
    ex.write_csv(tpath, ["setup", "step"] + TUNED + ["trees", "rounds_trained", "hit_cap", "raw_logloss", "cross_fitted_logloss", "scoring_seconds", "seconds"],
                 [[r["trial"], r["step"]] + [r["params"].get(k, "") for k in TUNED]
                  + [r["trees"], r["rounds_trained"], r["hit_cap"], r["raw_logloss"], r["cross_fitted_logloss"], r.get("scoring_seconds"), r["seconds"]] for r in rows])
    WRITTEN.append(tpath)
    hits = [r["trial"] for r in rows if r.get("hit_cap")]
    return rows, chosen, measures, cal, shap, risk, hits


def main(argv=None):
    args = parse_args(argv)
    configure("smoke" if args.smoke else "full")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    LOGS.mkdir(parents=True, exist_ok=True)
    check_raw.report.clear()
    FAILS.clear()
    WRITTEN.clear()
    started = time.time()
    progress("=" * 78)
    progress(f"fit_models.py v{SCRIPT_VERSION} started ({MODE} mode). Dataset {P['db']} (read-only); outputs {P['out']}.")
    con, code = None, 0
    try:
        if not P["db"].exists():
            progress(f"STOP: no dataset at {P['db']}; run build_dataset.py first.")
            raise StopRun(1)
        progress(f"DuckDB memory limit {args.memory_limit}, threads {args.threads}, seed {SEED}.")
        con = ex.open_dataset(args)
        P["out"].mkdir(parents=True, exist_ok=True)
        create_views(con)
        rep_header(args, ex.read_meta(con))
        counts = rep_population(con)
        progress(f"Training loans {mc(counts['train'][0])}, validation loans {mc(counts['validation'][0])}")
        train_cats = training_categories(con)
        codes, refs, present = level_codes(train_cats)
        t0 = time.time()
        num, cat, y = fetch_rows(con, "train", codes, True)
        num_v, cat_v, y_v, more_v = fetch_rows(con, "validation", codes, True, extra=("fold",))
        fold_v = more_v["fold"].astype(np.int8)
        del more_v
        progress(f"Rows read into memory in {hms(time.time() - t0)}")
        res = []
        bd.rsection("logistic regression", lambda: res.append(fit_logistic(con, num, cat, y, num_v, cat_v, y_v, fold_v, codes, refs, present, train_cats)))
        if not res:
            raise StopRun(1)
        lr, cal_lr, meas_lr, fills_num, fills_cat, coef_orig, n_inc_train, n_inc_val, risk_lr = res[0]
        bd.rsection("2 logistic regression", rep_logistic, lr, meas_lr, fills_num, fills_cat, refs, train_cats, n_inc_train, n_inc_val)
        dtrain, dvalid, X_v = lgb_datasets(num, cat, y, num_v, cat_v, y_v, codes, refs, present, args.threads)
        del num, cat, y                                 # dtrain's raw array holds the training loans from here on
        res2 = []
        bd.rsection("lightgbm", lambda: res2.append(fit_lightgbm(dtrain, dvalid, X_v, y_v, fold_v, codes, refs, args.threads)))
        if not res2:
            raise StopRun(1)
        rows, chosen, meas_gbm, cal_gbm, shap, risk_gbm, hits = res2[0]
        bd.rsection("3 lightgbm", rep_lightgbm, rows, chosen, meas_gbm, hits)
        save_json("isotonic_calibration.json", {"logistic_regression": {**cal_json(cal_lr), "validation": meas_lr},
                                                "lightgbm": {**cal_json(cal_gbm), "validation": meas_gbm},
                                                "rule": CAL_RULE, "min_bad_and_good_per_step": MIN_SIDE, "folds": N_FOLDS})
        bd.rsection("4 shape check", rep_shape, num_v, shap, coef_orig, fills_num)
        bd.rsection("5 beyond range", rep_beyond_range, con, num_v, risk_lr, risk_gbm)
        bd.rsection("6 unseen categories", rep_unseen, con, train_cats)
        rep_files()
        section(f"SUMMARY: {len(FAILS)} problem(s)")
        for f in FAILS:
            out(f"  FAIL: {f}")
        out("  SUCCESS: every output written; every check passed." if not FAILS else "  FAILURE: see the problems above.")
        out(f"(fit_models took {hms(time.time() - started)})")
        code = 1 if FAILS else 0
    except StopRun as stop:
        code = stop.code
    except KeyboardInterrupt:
        progress("Interrupted. Rerun the same command: finished LightGBM setups are kept and the search resumes.")
        code = 130
    except Exception as exc:
        bd.write_error_detail(exc, progress)
        code = 1
    finally:
        if con is not None:
            con.close()
        if P["spill"].exists() and not any(P["spill"].iterdir()):
            P["spill"].rmdir()
        if check_raw.report:
            inv.save_report()
            progress(f"Report saved to {P['report']}")
        word = "SUCCESS" if code == 0 else "INTERRUPTED (rerun to resume)" if code == 130 else "FAILURE"
        progress(f"{word}: exit code {code}, elapsed {hms(time.time() - started)}.")
    return code


if __name__ == "__main__":
    sys.exit(main())
