"""Shared model helpers: the variable definitions, the conforming limits, the SQL text of the two calculated variables
and the loan-term product groups, and the scoring / SHAP routines of the final models. numpy only: this module never opens a
database, imports no DuckDB and needs no dataset, so the live tool can use it.

Used by (the names each script imports from here):
  explore.py           BASE_VARIABLES (explore.VARIABLES)
  fit_models.py        LIMITS, SOURCE, DOLLARS, RATIO, FLAG, VARIABLES, NUMBERS, CATEGORIES, NUM_COLS, CAT_COLS, LEVELS,
                       LABEL, INCOMPLETE, CHUNK, N_FOLDS, limit_sql, fold_sql, model_view_sql (the SQL text of
                       create_model_view), expit, isotonic_apply, design_columns, build_design, lr_logodds, lgb_matrix
  check_beyond_range.py hold_at_edges, lr_risk (lr_design + lr_logodds), lr_inputs
  score_test.py        lr_shap, lr_by_variable
  credit_box.py        raw_scores, adjust_scores, adjusted_risks
  build_dataset.py     PRODUCT_SQL (the loan-term product groups)
"""

import numpy as np

CHUNK = 1_000_000                       # rows per Newton chunk and per DuckDB record batch
N_FOLDS = 5                             # cross-fitting folds

# Each loan term gets exactly one product (the shortest whose range contains it; 84 months or less: 10-year).
PRODUCT_SQL = ("CASE WHEN loan_term <= 120 THEN '10-year' WHEN loan_term <= 180 THEN '15-year' "
               "WHEN loan_term <= 240 THEN '20-year' WHEN loan_term <= 360 THEN '30-year' END")

# explore.py's variables and their documented category values.
BASE_VARIABLES = [
    ("credit_score", "credit score", "number", None),
    ("ltv", "LTV", "number", None),
    ("cltv", "combined LTV", "number", None),
    ("dti", "DTI", "number", None),
    ("rate_spread", "rate spread", "number", None),
    ("mi_pct", "mortgage insurance %", "number", None),
    ("original_upb", "loan amount", "number", None),
    ("loan_term", "loan term", "number", None),
    ("loan_purpose", "loan purpose", "category", ["purchase", "cash-out refinance", "no-cash-out refinance"]),
    ("num_borrowers", "number of borrowers", "category", ["1", "2 or more"]),
    ("occupancy", "occupancy", "category", ["primary", "second home", "investor"]),
    ("first_time_homebuyer", "first-time homebuyer", "category", ["Y", "N"]),
    ("channel", "channel", "category", ["retail", "broker", "correspondent"]),
    ("property_type", "property type", "category", ["SF", "PU", "CO", "CP", "MH"]),
    ("num_units", "number of units", "category", ["1", "2", "3", "4"]),
    ("special_program", "special program", "category", ["none", "F", "H", "R"]),
    ("high_balance", "high-balance flag", "category", ["Y", "N"]),
]

# FHFA basic conforming loan limits (contiguous US) by origination year, for 1 / 2 / 3 / 4 units.
LIMITS = {2014: (417_000, 533_850, 645_300, 801_950), 2015: (417_000, 533_850, 645_300, 801_950),
          2016: (417_000, 533_850, 645_300, 801_950), 2017: (424_100, 543_000, 656_350, 815_650),
          2018: (453_100, 580_150, 701_250, 871_450), 2019: (484_350, 620_200, 749_650, 931_600),
          2020: (510_400, 653_550, 789_950, 981_700), 2022: (647_200, 828_700, 1_001_650, 1_244_850),
          2023: (726_200, 929_850, 1_123_900, 1_396_800)}
SOURCE = "model_loans"                  # the view every script reads: kept_loans + the two calculated variables
DOLLARS = "original_upb"                # loan amount in dollars: Q3's weight, never a model variable
RATIO = "pct_of_limit"
FLAG = "at_limit"

# The 17 model variables: explore.py's 17 without LTV, with loan amount replaced in its
# place by the percent of limit, and the at-the-limit flag last.
VARIABLES = [(RATIO, "loan amount % of limit", "number", None) if v[0] == DOLLARS else v for v in BASE_VARIABLES if v[0] != "ltv"]
VARIABLES.append((FLAG, "at-the-limit flag", "category", ["Y", "N"]))
NUMBERS = [v for v in VARIABLES if v[2] == "number"]
CATEGORIES = [v for v in VARIABLES if v[2] == "category"]
NUM_COLS = [c for c, *_ in NUMBERS]
CAT_COLS = [c for c, *_ in CATEGORIES]
LEVELS = {c: lv for c, _, _, lv in CATEGORIES}
LABEL = {c: lab for c, lab, _, _ in VARIABLES}
INCOMPLETE = "incomplete application"


# ---------------------------------------------------------------- SQL text (run by the caller's DuckDB connection)

def limit_sql(scale=1, year_shift=0):
    """The loan's conforming loan limit as a SQL CASE: by the year of its origination quarter and
    its number of units; NULL when the units are blank or the year is not in LIMITS. scale=1000: the limit rounded
    to the nearest $1,000, in thousands. year_shift=1: the following year's limit instead (NULL when that year is
    not in LIMITS); used only by count_over_limit.py's diagnostic column."""
    whens = []
    for year in LIMITS:
        lims = LIMITS.get(year + year_shift)
        if lims is None:
            continue
        for units, lim in enumerate(lims, start=1):
            value = lim if scale == 1 else (lim + scale // 2) // scale
            whens.append(f"WHEN substr(origination_quarter, 1, 4) = '{year}' AND CAST(num_units AS VARCHAR) = '{units}' THEN {value}")
    return "CASE " + " ".join(whens) + " END"


def fold_sql():
    """Cross-fitting fold 0-4: MD5 of agency and loan ID, as a number, modulo 5. Calculated inside
    the database; the loan ID itself is never fetched."""
    return f"CAST(CAST('0x' || md5(agency || '|' || CAST(loan_id AS VARCHAR)) AS UHUGEINT) % {N_FOLDS} AS TINYINT)"


def model_view_sql():
    """model_loans = kept_loans plus the two calculated model variables and the cross-fitting fold.
    Percent of limit = 100 x original UPB / limit, 2 decimals. At-the-limit flag = Y when original UPB rounded to the
    nearest $1,000 equals the limit rounded to the nearest $1,000, else N. Both blank when the limit is (units blank)."""
    return f"""CREATE OR REPLACE TEMP VIEW {SOURCE} AS
SELECT k.*, round(100.0 * {DOLLARS} / conforming_limit, 2) AS {RATIO},
       CASE WHEN conforming_limit IS NULL OR {DOLLARS} IS NULL THEN NULL
            WHEN floor(({DOLLARS} + 500) / 1000) = conforming_limit_k THEN 'Y' ELSE 'N' END AS {FLAG},
       {fold_sql()} AS fold
FROM (SELECT *, {limit_sql()} AS conforming_limit, {limit_sql(1000)} AS conforming_limit_k FROM kept_loans) k"""


# ---------------------------------------------------------------- Logistic Regression

def expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def design_columns(refs, present):
    cols = ["intercept"] + [LABEL[c] for c in NUM_COLS] + [INCOMPLETE]
    dummies = []
    for c in CAT_COLS:
        for v in present[c]:
            if v != refs[c]:
                dummies.append((c, v))
    return cols + [f"{LABEL[c]} = {v}" for c, v in dummies], dummies


def build_design(num, cat, fills_num, fills_cat_code, codes, refs, present, means=None, sds=None):
    """The Logistic Regression matrix (float32): intercept, standardized filled numbers, incomplete flag, dummies.
    Blank categories get the fill value's dummy; unseen values (-2) get no dummy (= reference). Returns
    (X, means, sds, incomplete) with the training means / sds of the filled numbers when not given."""
    n = num.shape[0]
    cols, dummies = design_columns(refs, present)
    X = np.empty((n, len(cols)), dtype=np.float32)
    X[:, 0] = 1.0
    blank_num = np.isnan(num)
    incomplete = blank_num.any(axis=1) | (cat == -1).any(axis=1)
    filled = num.copy()
    for j, c in enumerate(NUM_COLS):
        filled[blank_num[:, j], j] = np.float32(fills_num[c])
    if means is None:
        means = filled.astype(np.float64).mean(axis=0)
        sds = filled.astype(np.float64).std(axis=0)
        sds = np.where(sds > 0, sds, 1.0)
    for j in range(len(NUM_COLS)):
        X[:, 1 + j] = (filled[:, j].astype(np.float64) - means[j]) / sds[j]
    del filled
    X[:, 1 + len(NUM_COLS)] = incomplete
    base = 2 + len(NUM_COLS)
    for k, (c, v) in enumerate(dummies):
        j = CAT_COLS.index(c)
        col = cat[:, j]
        code = codes[c][v]
        hit = col == code
        if fills_cat_code[c] == code:
            hit |= col == -1
        X[:, base + k] = hit
    return X, means, sds, incomplete


def lr_logodds(X, beta):
    z = np.empty(X.shape[0])
    for i in range(0, X.shape[0], CHUNK):
        z[i:i + CHUNK] = X[i:i + CHUNK].astype(np.float64) @ beta
    return z


# ---------------------------------------------------------------- LightGBM

def lgb_matrix(num, cat, codes, refs, present):
    """n x 17 float32 for LightGBM: numbers as stored (NaN = blank); category codes, NaN for a blank (code -1), and
    the reference's code for any value absent from training (a documented value with no training loans, e.g. R, or
    an undocumented value, code -2)."""
    X = np.empty((num.shape[0], len(NUM_COLS) + len(CAT_COLS)), dtype=np.float32)
    X[:, :len(NUM_COLS)] = num
    for j, c in enumerate(CAT_COLS):
        lookup = np.full(len(codes[c]) + 2, np.float32(codes[c][refs[c]]), dtype=np.float32)   # index = code + 2
        lookup[1] = np.nan                                                                   # -1: blank; -2 keeps the reference
        for v in present[c]:
            lookup[codes[c][v] + 2] = codes[c][v]
        X[:, len(NUM_COLS) + j] = lookup[cat[:, j].astype(np.int64) + 2]
    return X


# ---------------------------------------------------------------- isotonic adjustment

def isotonic_apply(cal, z):
    """Adjusted risk of raw scores z: the step's value inside a step, the straight line between two
    neighbouring steps, the nearest step's value outside the fitted range."""
    lo, hi, v = (np.asarray(cal[k], dtype=np.float64) for k in ("score_low", "score_high", "value"))
    xp = np.column_stack([lo, hi]).ravel()
    fp = np.repeat(v, 2)
    keep = np.r_[True, xp[1:] != xp[:-1]]                # a step with one distinct score: one knot, not two
    return np.interp(np.asarray(z, dtype=np.float64), xp[keep], fp[keep])


# ---------------------------------------------------------------- the edge rule and the saved Logistic Regression

def hold_at_edges(num, mins, maxes):
    """Version B's inputs: each number below its training min set to the min, above its max set to the max.
    Blanks (NaN) stay blank; the comparison is in float32, as in fit_models' output 5."""
    held = num.copy()
    for j, c in enumerate(NUM_COLS):
        lo, hi = np.float32(mins[c]), np.float32(maxes[c])
        col = held[:, j]
        col[col < lo] = lo
        col[col > hi] = hi
    return held


def lr_design(lr, num, cat, codes, refs, present):
    """The saved Logistic Regression's design matrix (fills and standardization as saved)."""
    fills_num = {c: lr["fill_values"][LABEL[c]] for c in NUM_COLS}
    fills_code = {c: codes[c][lr["fill_values"][LABEL[c]]] for c in CAT_COLS}
    means = np.array([lr["standardization"][LABEL[c]]["mean"] for c in NUM_COLS])
    sds = np.array([lr["standardization"][LABEL[c]]["sd"] for c in NUM_COLS])
    X, _, _, _ = build_design(num, cat, fills_num, fills_code, codes, refs, present, means, sds)
    return X


def lr_risk(lr, num, cat, codes, refs, present):
    """The saved Logistic Regression's raw log-odds (fills, standardization, coefficients as saved)."""
    beta = np.array([lr["coefficients_standardized"][c] for c in lr["columns"]])
    return lr_logodds(lr_design(lr, num, cat, codes, refs, present), beta)


def lr_inputs(rule, num):
    """The Logistic Regression's number inputs under the edge rule: held at the training min / max if the result is B."""
    if rule["result"] != "B":
        return num
    mins = {c: rule["training_range"][LABEL[c]]["training_min"] for c in NUM_COLS}
    maxes = {c: rule["training_range"][LABEL[c]]["training_max"] for c in NUM_COLS}
    return hold_at_edges(num, mins, maxes)


# ---------------------------------------------------------------- SHAP values of the Logistic Regression

def lr_shap(lr, X, shares):
    """Logistic Regression SHAP values per design column: beta_std x (column - its training mean), and the baseline
    (training mean log-odds). Column means: 1 for the intercept, 0 for standardized numbers, training shares for the
    incomplete flag and the dummies."""
    beta = np.array([lr["coefficients_standardized"][c] for c in lr["columns"]])
    mu = np.zeros(len(beta))
    mu[0] = 1.0
    for j, c in enumerate(lr["columns"]):
        if c in shares:
            mu[j] = shares[c]
    baseline = float(beta @ mu)
    phi = np.empty((X.shape[0], len(beta)))
    for i in range(0, X.shape[0], CHUNK):
        phi[i:i + CHUNK] = (X[i:i + CHUNK].astype(np.float64) - mu) * beta
    return phi, baseline


def lr_by_variable(lr, phi):
    """Per-variable Logistic Regression SHAP values: numbers as they are, dummies summed per variable, the incomplete
    column on its own. Returns {label: array}."""
    res = {}
    cols = lr["columns"]
    for j, c in enumerate(NUM_COLS):
        res[LABEL[c]] = phi[:, cols.index(LABEL[c])]
    res[INCOMPLETE] = phi[:, cols.index(INCOMPLETE)]
    for c in CAT_COLS:
        idx = [j for j, name in enumerate(cols) if name.startswith(LABEL[c] + " = ")]
        res[LABEL[c]] = phi[:, idx].sum(axis=1) if idx else np.zeros(phi.shape[0])
    return res


# ---------------------------------------------------------------- both final models (as score_test.py scores the test)

def raw_scores(num, cat, saved):
    """Both models' raw scores (log-odds), in chunks, exactly as score_test.py scores the test loans: Logistic
    Regression with lr_edge_rule.json's inputs, LightGBM on the stored values."""
    n = num.shape[0]
    z_lr, z_g = np.empty(n), np.empty(n)
    for i in range(0, n, CHUNK):
        nc, cc = num[i:i + CHUNK], cat[i:i + CHUNK]
        z_lr[i:i + CHUNK] = lr_risk(saved["lr"], lr_inputs(saved["rule"], nc), cc, saved["codes"], saved["refs"], saved["present"])
        z_g[i:i + CHUNK] = saved["booster"].predict(lgb_matrix(nc, cc, saved["codes"], saved["refs"], saved["present"]), raw_score=True)
    return z_lr, z_g


def adjust_scores(z_lr, z_g, saved):
    """Raw scores -> adjusted (isotonic) predicted risks with the saved steps."""
    return isotonic_apply(saved["rule"]["calibration"], z_lr), isotonic_apply(saved["cal"]["lightgbm"], z_g)


def adjusted_risks(num, cat, saved):
    """Both models' adjusted (isotonic) predicted risk, exactly as score_test.py scores the test loans."""
    return adjust_scores(*raw_scores(num, cat, saved), saved)
