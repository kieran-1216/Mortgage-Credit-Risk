"""Score the TEST loans once with the final models and produce Q1 and Q2.
Summaries only.

The models are final: the saved Logistic Regression with lr_edge_rule.json applied (its result: under B every number
is held at its training min / max when outside it) and the winner's isotonic adjustment saved there, the saved fills
and the incomplete-application column; the saved chosen LightGBM model with its isotonic adjustment
(isotonic_calibration.json), a blank category as missing. Unseen category values score as the reference category in
both models. 17 model variables (loan amount as a percent of its conforming loan limit and an
at-the-limit flag; loan amount in dollars is not a model variable). Nothing is refitted, chosen or tuned here.

TEST ONCE: the script refuses to start if its report or its output folder already exists. The override
--rescore-test renames both with a timestamp (nothing is deleted), prints a banner, and the report header says RESCORED.
It works ONCE: when the existing report is already a rescore that finished with SUCCESS, the script
refuses even with --rescore-test. (A rescore that did not finish SUCCESS can be rerun with --rescore-test; its
results are renamed, never deleted.) Every report carries the RESCORED line. There is no smoke
run: every number this script writes is a test result. Before anything is renamed or read, the script checks that the
LightGBM model's trees are small enough for the exact interaction values (Q2c) and stops if not.

Q1 (each model; all test loans, then each agency alone): AUC on the raw score (log-odds; the isotonic
adjustment ties many loans and never changes their order); calibration in ten equal-sized risk groups ranked by the
raw score (ties by row position), with loans, mean adjusted predicted risk and actual bad rate per group; the
population's loans and bad rate.
Q2 (all test loans, log-odds scale, before the isotonic adjustment):
  importance: mean absolute SHAP value per variable, each model's own ranking side by side (Logistic Regression's
              incomplete-application column is its own row);
  shape: mean SHAP value per distinct value or category of each of the 17 variables, both models (CSV; long tables
         summarized in the report);
  compounding (LightGBM): exact SHAP interaction values per tree from the model dump, mean absolute full pair effect
         (both halves) per variable pair, ranked; checked against LightGBM's pred_contrib (row sums = SHAP values);
  odds ratios: Logistic Regression's weights with 95% Wald intervals from the observed information (one Hessian
         pass over the training loans with the saved coefficients), per 1 unit as stored and per 1 training SD.
Logistic Regression SHAP value of a column = coefficient x (value the model sees - training mean); the baseline is
the training mean log-odds; a category's value is the sum of its dummy columns; the values sum to the raw log-odds.

Reuses fit_models.py (paths, read-only connection, views, row reader, column builders, AUC, isotonic adjustment,
masking) and check_beyond_range.py (saved-model loading, edge holding, saved Logistic Regression scoring). New here: the per-tree
interaction routine (no installed package does it for LightGBM categorical splits), the
Logistic Regression SHAP and Hessian passes, the rank groups, and the two-model shape / importance tables.

Files: outputs/test/ (q1_auc.csv, q1_calibration.csv, q2_importance.csv, q2_shape_<variable>.csv, q2_interactions.csv,
q2_odds_ratios.csv), logs/score_test_report.txt, logs/score_test_progress.log.

Usage (from the repository root, after fit_models.py and check_beyond_range.py):
    python code/score_test.py --rescore-test [--threads 8] [--memory-limit 2GB]
Exit codes: 0 SUCCESS; 1 FAILURE or refused (test results already exist, already rescored, or --smoke); 130 interrupted.
"""

import itertools
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset as bd  # noqa: E402  (reused: check lines, report sections, error detail)
import check_beyond_range as cb  # noqa: E402  (reused: saved-model loading, edge holding, saved LR scoring)
import check_raw  # noqa: E402
import explore as ex  # noqa: E402  (reused: dataset connection, masking helpers, CSV writing)
import fit_models as fm  # noqa: E402  (reused: views, readers, column builders, AUC, isotonic, masking)
import inventory as inv  # noqa: E402  (reused: progress log, report file, 1-10 masking)
from model_core import lr_by_variable, lr_shap  # noqa: E402,F401
from check_raw import out, out_table, section  # noqa: E402
from inventory import StopRun, lit, m, mc, progress  # noqa: E402
from load_raw import LOG_DIR, PROJECT_ROOT, hms  # noqa: E402

SCRIPT_VERSION = "2"                    # 2: isotonic adjustment; loan amount as a percent of its conforming limit plus an at-the-limit flag
MASK, MISSING = ex.MASK, ex.MISSING
N_GROUPS = 10
Z95 = 1.959963984540054
PRINT_ROWS = 150
INTER_CHUNK = 100_000
MAX_TREE_FEATURES = 10          # 2^10 subsets per tree at most
RESCORED_LINE = ("RESCORED once after a review of first test results "
                 "(calibration and loan-size handling).")
AGENCIES = [("Freddie Mac", "test_fre"), ("Fannie Mae", "test_fan")]
LR, GBM = "Logistic Regression", "LightGBM"

WORK_ROOT = PROJECT_ROOT    # tests re-point these two before calling main()
LOGS = LOG_DIR
P = {}
FAILS = bd.FAILS
WRITTEN = []


def configure(mode):
    cb.WORK_ROOT = WORK_ROOT
    cb.LOGS = LOGS
    cb.configure(mode)                                  # fit_models' paths (saved models in cb.P['models']) and spill
    sfx = "_smoke" if mode == "smoke" else ""
    P.clear()
    P.update(models=cb.P["models"], out=WORK_ROOT / "outputs" / f"test{sfx}", report=LOGS / f"score_test_report{sfx}.txt",
             progress=LOGS / "score_test_progress.log", error=LOGS / "score_test_error_detail.txt")
    inv.REPORT_PATH = P["report"]
    inv.PROGRESS_PATH = P["progress"]
    bd.P["error"] = P["error"]


# ---------------------------------------------------------------- test once

def finished_redo_rescore(report_path):
    """True when the report is a rescore (it carries RESCORED_LINE) that finished with SUCCESS."""
    if not report_path.exists():
        return False
    text = report_path.read_text(encoding="utf-8", errors="replace")
    return RESCORED_LINE in text and "SUCCESS: every output written; every check passed." in text


def guard_test_once(rescore):
    """Refuse when results exist; with --rescore-test rename them with a timestamp (nothing deleted). The
    rescore is done once: a finished rescore is never replaced."""
    existing = [p for p in (P["report"], P["out"]) if p.exists()]
    if not existing:
        return None
    if finished_redo_rescore(P["report"]):
        progress(f"REFUSED: {P['report']} is a rescore of the test and it finished with SUCCESS. The test is rescored "
                 "exactly once; --rescore-test does not apply again.")
        raise StopRun(1)
    if not rescore:
        progress("REFUSED: test results already exist: " + "; ".join(str(p) for p in existing)
                 + ". The test is scored once. To score it again anyway, run with --rescore-test (the existing results are "
                   "renamed with a timestamp, never deleted).")
        raise StopRun(1)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    moved = []
    for p in existing:
        target = p.with_name(f"{p.stem}_before_{stamp}{p.suffix}")
        p.rename(target)
        moved.append(f"{p.name} -> {target.name}")
    banner = "#" * 78
    progress(banner)
    progress("RESCORING THE TEST SET: --rescore-test given. The test was already scored; this run scores it again. Nothing was "
             "refitted or re-chosen; the previous results were renamed: " + "; ".join(moved))
    progress(banner)
    return stamp


# ---------------------------------------------------------------- data

def create_test_views(con):
    """The only two views that read the test outcome: one per agency, the 17 variables of fit_models' model_loans view
    (fm.create_views must run first) and the loan amount in dollars."""
    for agency, view in AGENCIES:
        con.execute(f"CREATE OR REPLACE TEMP VIEW {view} AS SELECT (outcome = 'bad')::TINYINT AS bad, {fm.var_sql()}, "
                    f"CAST({fm.DOLLARS} AS DOUBLE) AS dollars FROM {fm.SOURCE} "
                    f"WHERE label = 'test' AND agency = {lit(agency)}{fm.sample_sql()}")


# ---------------------------------------------------------------- Q1

def rank_groups(risk, n_groups=N_GROUPS):
    """Ten equal-sized groups by rank of the risk (stable sort: ties keep row position). Returns (group index per
    loan, number of loans in a tie that straddles a group boundary)."""
    order = np.argsort(risk, kind="stable")
    group = np.empty(len(risk), dtype=np.int8)
    bounds = np.linspace(0, len(risk), n_groups + 1).round().astype(int)
    straddle = 0
    sorted_risk = risk[order]
    for g in range(n_groups):
        group[order[bounds[g]:bounds[g + 1]]] = g
        if 0 < g and bounds[g] < len(risk) and sorted_risk[bounds[g] - 1] == sorted_risk[bounds[g]]:
            straddle += int(np.sum(risk == sorted_risk[bounds[g]]))
    return group, straddle


def rate_cell(n, b):
    return f"{100 * b / n:.4f}%" if ex.rate_shown(n, b) else "-"


def q1_population(name, y, risks, scores):
    """AUC and the ten risk groups for both models on one population: AUC and the groups by the raw
    score; each group's mean ADJUSTED predicted risk. Returns (auc rows, calibration rows)."""
    n, b = len(y), int(y.sum())
    auc_rows, cal_rows = [], []
    for model, risk in risks.items():
        score = scores[model]
        a = fm.auc(y, score)
        group, straddle = rank_groups(score)
        auc_rows.append([name, model, mc(n), mc(b), rate_cell(n, b), "-" if a is None else f"{a:.6f}", mc(straddle)])
        for g in range(N_GROUPS):
            mask = group == g
            k, kb = int(mask.sum()), int(y[mask].sum())
            cal_rows.append([name, model, g + 1, m(k), "" if 1 <= k <= 10 or k == 0 else f"{100 * risk[mask].mean():.4f}%", m(kb), rate_cell(k, kb)])
    return auc_rows, cal_rows


# ---------------------------------------------------------------- Q2: SHAP values

def column_shares(con, lr):
    """Training means of the incomplete flag and the dummy columns (shares of training loans)."""
    n_train = con.execute("SELECT count(*) FROM train").fetchall()[0][0]
    cats = fm.training_categories(con)
    shares = {fm.INCOMPLETE: lr["incomplete_application"]["training_loans"] / n_train}
    for d in lr["dummy_columns"]:
        col = next(c for c in fm.CAT_COLS if fm.LABEL[c] == d["variable"])
        shares[f"{d['variable']} = {d['value']}"] = cats[col].get(d["value"], 0) / n_train
    return shares, n_train


# ---------------------------------------------------------------- Q2: exact SHAP interaction values per tree

def flatten_tree(root):
    """Splits and leaves of one dumped tree. splits: (feature, decision_type, threshold, categories, default_left,
    missing_type, left share of the training count). leaves: (value, [(split index, goes left)])."""
    feats, splits, leaves = set(), [], []

    def count(node):
        return node.get("internal_count", node.get("leaf_count"))

    def rec(node, path):
        if "leaf_value" in node:
            leaves.append((float(node["leaf_value"]), list(path)))
            return
        f = int(node["split_feature"])
        feats.add(f)
        cats = None
        if node["decision_type"] == "==":
            cats = np.array([int(v) for v in str(node["threshold"]).split("||")], dtype=np.float64)
        left_n, right_n = count(node["left_child"]), count(node["right_child"])
        s = len(splits)
        splits.append((f, node["decision_type"], node["threshold"], cats, bool(node["default_left"]), node["missing_type"],
                       left_n / (left_n + right_n)))
        rec(node["left_child"], path + [(s, True)])
        rec(node["right_child"], path + [(s, False)])
    rec(root, [])
    return sorted(feats), splits, leaves


def goes_left(split, X):
    """LightGBM's decision rule (tree.h NumericalDecision / CategoricalDecision), including its missing-value rules."""
    f, dtype, thr, cats, default_left, missing_type, _ = split
    v = X[:, f].astype(np.float64)
    nan = np.isnan(v)
    if dtype == "==":
        # a blank (NaN) or negative category always goes right, whatever the split's missing type (checked against
        # LightGBM 4.7's predictions: a blank at a split learned with no training blanks goes right too)
        iv = np.where(nan, -1.0, np.floor(v))
        left = np.isin(iv, cats)
        left[iv < 0] = False
        return left
    fval = np.where(nan & (missing_type != "NaN"), 0.0, v)
    left = fval <= float(thr)
    if missing_type == "Zero":
        left[np.abs(fval) <= 1e-35] = default_left
    elif missing_type == "NaN":
        left[nan] = default_left
    return left


def tree_interactions(root, X, n_features, out_phi, out_pairs):
    """Add one tree's exact Shapley interaction values to out_pairs (n x F x F, full matrix) and its SHAP values to
    out_phi (n x F). Conditional expectations follow LightGBM's path-dependent rule (training counts at each split a
    subset does not fix). Variables the tree does not use are dummy players (zero). Returns the tree's expected value."""
    feats, splits, leaves = flatten_tree(root)
    n = X.shape[0]
    if not splits:
        out_phi[:, :] += 0.0
        return leaves[0][0] if leaves else 0.0
    if len(feats) > MAX_TREE_FEATURES:
        raise StopRun(1)
    L = [goes_left(s, X) for s in splits]
    share = [s[6] for s in splits]
    E = {}
    for r in range(len(feats) + 1):
        for S in itertools.combinations(feats, r):
            Sset = set(S)
            total = np.zeros(n)
            for value, path in leaves:
                p = np.ones(n)
                for s_idx, left in path:
                    if splits[s_idx][0] in Sset:
                        p = p * (L[s_idx] if left else ~L[s_idx])
                    else:
                        p = p * (share[s_idx] if left else 1 - share[s_idx])
                total += value * p
            E[S] = total
    M = len(feats)
    for i in feats:
        rest = [f for f in feats if f != i]
        acc = np.zeros(n)
        for r in range(len(rest) + 1):
            w = math.factorial(r) * math.factorial(M - r - 1) / math.factorial(M)
            for S in itertools.combinations(rest, r):
                acc += w * (E[tuple(sorted(S + (i,)))] - E[S])
        out_phi[:, i] += acc
    for i, j in itertools.combinations(feats, 2):
        rest = [f for f in feats if f not in (i, j)]
        acc = np.zeros(n)
        for r in range(len(rest) + 1):
            w = math.factorial(r) * math.factorial(M - r - 2) / (2 * math.factorial(M - 1))
            for S in itertools.combinations(rest, r):
                acc += w * (E[tuple(sorted(S + (i, j)))] - E[tuple(sorted(S + (i,)))] - E[tuple(sorted(S + (j,)))] + E[S])
        out_pairs[:, i, j] += acc
        out_pairs[:, j, i] += acc
    return float(E[()][0])


def tree_feature_counts(booster):
    """Distinct variables used by each tree of the saved model (num_iteration = its best iteration)."""
    dump = booster.dump_model(num_iteration=booster.best_iteration or None)
    return [len(flatten_tree(t["tree_structure"])[0]) for t in dump["tree_info"]]


def check_tree_sizes(booster):
    """Q2c's exact per-tree interaction values enumerate every subset of a tree's variables, so a tree with more than
    MAX_TREE_FEATURES variables cannot be done in time. Checked before anything is renamed or any test loan is read."""
    counts = tree_feature_counts(booster)
    big = sum(1 for k in counts if k > MAX_TREE_FEATURES)
    cost = sum(2 ** k for k in counts)
    progress(f"LightGBM model: {len(counts)} trees; variables per tree: at most {max(counts) if counts else 0}; "
             f"subsets to evaluate per loan: {cost:,}")
    if big:
        progress(f"STOP (nothing renamed, no test loan read): {big} tree(s) use more than {MAX_TREE_FEATURES} variables, so the "
                 "exact interaction values (Q2c) would not finish; "
                 "the script needs a change before the test is scored.")
        raise StopRun(1)
    return counts, cost


def interaction_means(booster, X, shap_check):
    """Mean |full pair effect| (both halves) per variable pair over all rows, in chunks; also the largest difference
    between the trees' summed SHAP values and LightGBM's pred_contrib (identity check) and the expected value."""
    dump = booster.dump_model(num_iteration=booster.best_iteration or None)
    trees = [t["tree_structure"] for t in dump["tree_info"]]
    F = X.shape[1]
    sums = np.zeros((F, F))
    max_diff, max_base_diff = 0.0, 0.0
    t0 = time.time()
    for start in range(0, X.shape[0], INTER_CHUNK):
        Xc = X[start:start + INTER_CHUNK]
        phi = np.zeros((Xc.shape[0], F))
        pairs = np.zeros((Xc.shape[0], F, F))
        base = 0.0
        for tree in trees:
            base += tree_interactions(tree, Xc, F, phi, pairs)
        sums += np.abs(2 * pairs).sum(axis=0)
        sc = shap_check[start:start + INTER_CHUNK]
        max_diff = max(max_diff, float(np.max(np.abs(phi - sc[:, :F]))))     # the trees' Shapley values vs LightGBM's pred_contrib
        max_base_diff = max(max_base_diff, float(np.max(np.abs(base - sc[:, F]))))
        done = min(start + INTER_CHUNK, X.shape[0])
        rate = done / max(time.time() - t0, 1e-9)
        progress(f"  interaction values: {done:,} of {X.shape[0]:,} loans; remaining ~{hms((X.shape[0] - done) / rate)}")
    return sums / X.shape[0], max_diff, max_base_diff, len(trees)


# ---------------------------------------------------------------- Q2: odds ratios

def hessian_pass(X, beta, keep):
    """Observed information at the saved coefficients: sum over loans of p(1-p) x xᵀx, on the kept columns."""
    k = len(keep)
    H = np.zeros((k, k))
    for i in range(0, X.shape[0], fm.CHUNK):
        Xc = X[i:i + fm.CHUNK][:, keep].astype(np.float64)
        p = fm.expit(Xc @ beta[keep])
        H += Xc.T @ (Xc * (p * (1 - p))[:, None])
    return H


def odds_ratio_rows(lr, se_std):
    """Per column (no intercept): unit, coefficient, odds ratio and 95% Wald interval. Numbers get two rows: per 1 unit
    as stored and per 1 training SD. se_std: standard error per column in standardized units (None if not estimable)."""
    rows = []
    cols = lr["columns"]
    sd = {fm.LABEL[c]: lr["standardization"][fm.LABEL[c]]["sd"] for c in fm.NUM_COLS}
    refs = lr["reference_categories"]

    def fmt(beta, se):
        if se is None:
            return [f"{beta:.6f}", f"{math.exp(beta):.6f}", "-", "-"]
        return [f"{beta:.6f}", f"{math.exp(beta):.6f}", f"{math.exp(beta - Z95 * se):.6f}", f"{math.exp(beta + Z95 * se):.6f}"]
    for j, name in enumerate(cols):
        if name == "intercept":
            continue
        b_std, se = lr["coefficients_standardized"][name], se_std[j]
        if name in sd:
            b_orig = lr["coefficients_original_units"][name]
            se_orig = None if se is None else se / sd[name]
            rows.append([name, "per 1 unit as stored"] + fmt(b_orig, se_orig))
            rows.append([name, f"per 1 training SD ({inv.fmt_value(sd[name])})"] + fmt(b_std, se))
        elif name == fm.INCOMPLETE:
            rows.append([name, "yes vs no"] + fmt(b_std, se))
        else:
            var = name.split(" = ")[0]
            rows.append([name, f"vs {refs[var]}"] + fmt(b_std, se))
    return rows


# ---------------------------------------------------------------- Q2: tables

def shape_table(values, phi_lr, phi_gbm, categorical, levels=None, codes=None):
    """Mean SHAP value of both models per distinct value (numbers) or per category level (incl. unseen), then the
    missing row. Rows with 1-10 loans show only '1-10'."""
    rows = []
    if categorical:
        inv_codes = {v: k for k, v in codes.items()}
        present = sorted({int(v) for v in values if v >= 0})
        order = [codes[lv] for lv in levels if codes[lv] in present] + [c for c in present if inv_codes.get(c) not in levels]
        for code in order:
            mask = values == code
            rows.append(_shape_row(inv_codes.get(code, str(code)), mask, phi_lr, phi_gbm))
        rows.append(_shape_row(MISSING, values == -1, phi_lr, phi_gbm))
    else:
        nan = np.isnan(values)
        for v in np.unique(values[~nan]):
            rows.append(_shape_row(ex.num_text(v), values == v, phi_lr, phi_gbm))
        rows.append(_shape_row(MISSING, nan, phi_lr, phi_gbm))
    return rows


def _shape_row(label, mask, phi_lr, phi_gbm):
    n = int(mask.sum())
    if 1 <= n <= 10:
        return [label, MASK, "", ""]
    if n == 0:
        return [label, "0", "", ""]
    return [label, str(n), f"{phi_lr[mask].mean():.6f}", f"{phi_gbm[mask].mean():.6f}"]


# ---------------------------------------------------------------- report

def rep_header(args, meta, stamp, rule):
    out(f"score_test.py v{SCRIPT_VERSION}  {datetime.now():%Y-%m-%d %H:%M}  dataset {fm.P['db'].name} (read-only)  saved models "
        f"{P['models'].relative_to(WORK_ROOT)}" + ("  RESCORED (previous results renamed with the stamp " + stamp + ")" if stamp else ""))
    out(RESCORED_LINE)
    out("Summaries only: no loan IDs, no row-level values. Counts from 1 to 10 show as '1-10'; no rate or average is shown for a "
        "masked count. The test loans are scored once with the final models; nothing is refitted, chosen or tuned here.")
    out(f"Dataset built by build_dataset.py v{meta.get('script_version', '?')} at {meta.get('built_at', '?')}; specification file SHA-256 "
        f"{meta.get('decision_log_sha256', '?')}. DuckDB memory limit {args.memory_limit}, threads {args.threads}.")
    out(f"Logistic Regression inputs: lr_edge_rule.json result {rule['result']} "
        + ("(every number held at its training min / max when outside it)" if rule["result"] == "B" else "(inputs as stored)")
        + f", adjusted by that version's isotonic steps; saved fills and the incomplete-application column; unseen category "
        "values score as the reference. LightGBM: the saved chosen model and its isotonic steps (isotonic_calibration.json); "
        "numbers as stored; a blank category is missing; unseen category values get the reference category's code. "
        f"{len(fm.VARIABLES)} model variables.")
    out("Q2 SHAP values are on the log-odds scale before the isotonic adjustment: Logistic Regression's are coefficient x (value "
        "the model sees - training mean) with the training mean log-odds as baseline; LightGBM's are pred_contrib (training-based "
        "expected value). Interaction values are exact per-tree Shapley interaction values from the model dump.")


def rep_q1(auc_rows, cal_rows):
    section("Q1. Order and level on the test loans: AUC and ten equal-sized risk groups, each model, all loans then each agency")
    out("  AUC and groups on the raw score (log-odds): loans ranked by the model's raw score and cut into ten "
        "equal groups (sizes differ by at most one loan); ties keep row position. 'tied at a boundary' = loans whose raw "
        "score is shared across two groups. 'mean predicted risk' = the mean ADJUSTED (isotonic) predicted risk.")
    out_table(["population", "model", "loans", "bad loans", "actual bad rate", "AUC", "tied at a boundary"], auc_rows, left={0, 1})
    for pop in dict.fromkeys(r[0] for r in cal_rows):
        for model in (LR, GBM):
            out()
            out(f"--- {pop}, {model}: calibration by risk group (1 = lowest predicted risk)")
            out_table(["group", "loans", "mean predicted risk", "bad loans", "actual bad rate"],
                      [r[2:] for r in cal_rows if r[0] == pop and r[1] == model], left={0})
    path = P["out"] / "q1_auc.csv"
    ex.write_csv(path, ["population", "model", "loans", "bad_loans", "actual_bad_rate", "auc", "tied_at_boundary"], [[str(x) for x in r] for r in auc_rows])
    WRITTEN.append(path)
    path = P["out"] / "q1_calibration.csv"
    ex.write_csv(path, ["population", "model", "group", "loans", "mean_predicted_risk", "bad_loans", "actual_bad_rate"], [[str(x) for x in r] for r in cal_rows])
    WRITTEN.append(path)


def rep_importance(imp_lr, imp_gbm):
    section("Q2a. Importance: mean absolute SHAP value per variable (log-odds), each model's own ranking")
    out("  A category's SHAP value is the sum of its columns per loan. 'incomplete application' is a Logistic Regression column only.")
    rl = sorted(imp_lr.items(), key=lambda t: -t[1])
    rg = sorted(imp_gbm.items(), key=lambda t: -t[1])
    rows = []
    for k in range(max(len(rl), len(rg))):
        a = rl[k] if k < len(rl) else ("", None)
        g = rg[k] if k < len(rg) else ("", None)
        rows.append([k + 1, a[0] + (" (Logistic Regression only)" if a[0] == fm.INCOMPLETE else ""), "" if a[1] is None else f"{a[1]:.6f}",
                     g[0], "" if g[1] is None else f"{g[1]:.6f}"])
    out_table(["rank", "Logistic Regression: variable", "mean |SHAP|", "LightGBM: variable", "mean |SHAP|"], rows, left={1, 3})
    path = P["out"] / "q2_importance.csv"
    ex.write_csv(path, ["rank", "lr_variable", "lr_mean_abs_shap", "lgbm_variable", "lgbm_mean_abs_shap"], [[str(x) for x in r] for r in rows])
    WRITTEN.append(path)


def rep_shape(num_t, cat_t, by_var_lr, phi_gbm, codes):
    section("Q2b. Shape: mean SHAP value per distinct value or category, both models (log-odds)")
    out(f"  Every table is in a CSV; number tables longer than {PRINT_ROWS} rows show only their range and missing row here.")
    for k, (c, label, kind, levels) in enumerate(fm.VARIABLES):
        if kind == "number":
            j = fm.NUM_COLS.index(c)
            rows = shape_table(num_t[:, j].astype(np.float64), by_var_lr[label], phi_gbm[:, j], False)
        else:
            j = fm.CAT_COLS.index(c)
            rows = shape_table(cat_t[:, j].astype(np.int64), by_var_lr[label], phi_gbm[:, len(fm.NUM_COLS) + j], True, levels, codes[c])
        path = P["out"] / f"q2_shape_{c}.csv"
        ex.write_csv(path, ["value", "loans", "lr_mean_shap", "lgbm_mean_shap"], rows)
        WRITTEN.append(path)
        out()
        out(f"--- {label} ({kind}): {len(rows) - 1} {'values' if kind == 'number' else 'categories'} + missing row; "
            f"rows with 1-10 loans: {sum(1 for r in rows if r[1] == MASK)}   ({path.relative_to(WORK_ROOT)})")
        show = [[r[0], ex.commas(r[1]), r[2] or "-", r[3] or "-"] for r in rows]
        if len(rows) <= PRINT_ROWS:
            out_table(["value", "loans", "LR mean SHAP", "LightGBM mean SHAP"], show, left={0})
        else:
            vals = [float(r[0]) for r in rows if r[0] != MISSING]
            out(f"  {len(vals):,} distinct values from {inv.fmt_value(min(vals))} to {inv.fmt_value(max(vals))} (see the CSV)")
            out_table(["value", "loans", "LR mean SHAP", "LightGBM mean SHAP"], show[-1:], left={0})


def rep_interactions(means, max_diff, max_base_diff, n_trees, n_loans):
    section("Q2c. Compounding (LightGBM): mean absolute SHAP interaction value per variable pair, ranked (log-odds)")
    out(f"  Exact Shapley interaction values per tree ({n_trees} trees) from the model dump, all {n_loans:,} test loans; a pair's "
        "value is the full pair effect (both halves of the symmetric matrix added).")
    bd.check_line("the trees' summed SHAP values equal LightGBM's pred_contrib on every test loan (1e-9)", max_diff < 1e-9)
    bd.check_line("the trees' summed expected value equals LightGBM's (1e-9)", max_base_diff < 1e-9)
    out(f"  Largest difference from pred_contrib: {max_diff:.2e}; expected value: {max_base_diff:.2e}.")
    names = fm.NUM_COLS + fm.CAT_COLS
    pairs = sorted(((means[i, j], names[i], names[j]) for i, j in itertools.combinations(range(len(names)), 2)), key=lambda t: -t[0])
    rows = [[k + 1, fm.LABEL[a], fm.LABEL[b], f"{v:.6f}"] for k, (v, a, b) in enumerate(pairs)]
    out_table(["rank", "variable 1", "variable 2", "mean |interaction|"], rows, left={1, 2})
    path = P["out"] / "q2_interactions.csv"
    ex.write_csv(path, ["rank", "variable_1", "variable_2", "mean_abs_interaction"], [[str(x) for x in r] for r in rows])
    WRITTEN.append(path)


def rep_odds(rows, n_train):
    section("Q2d. Odds ratios: Logistic Regression weights with 95% Wald intervals (observed information from the training fit)")
    out(f"  Standard errors from the inverse of the Hessian of the training log-likelihood at the saved coefficients "
        f"({mc(n_train)} training loans, one pass). Numbers: per 1 unit as stored and per 1 training standard deviation; "
        "categories: value vs the reference; incomplete application: yes vs no. No odds ratio for the intercept.")
    out_table(["column", "unit", "coefficient", "odds ratio", "95% low", "95% high"], rows, left={0, 1})
    path = P["out"] / "q2_odds_ratios.csv"
    ex.write_csv(path, ["column", "unit", "coefficient", "odds_ratio", "ci_low_95", "ci_high_95"], rows)
    WRITTEN.append(path)


def rep_files():
    section("Files written")
    for p in WRITTEN:
        out(f"  {p.relative_to(WORK_ROOT)}")


# ---------------------------------------------------------------- main

def parse_args(argv):
    argv = list(sys.argv[1:] if argv is None else argv)
    rescore = "--rescore-test" in argv
    args = fm.parse_args([a for a in argv if a != "--rescore-test"])
    args.rescore_test = rescore
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        print("REFUSED: score_test.py has no smoke run: every number it writes is a test result, and the test is rescored "
              "exactly once. Run check_beyond_range.py --smoke and credit_box.py --smoke instead.")
        return 1
    configure("full")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    LOGS.mkdir(parents=True, exist_ok=True)
    check_raw.report.clear()
    FAILS.clear()
    WRITTEN.clear()
    started = time.time()
    progress("=" * 78)
    progress(f"score_test.py v{SCRIPT_VERSION} started ({fm.MODE} mode). Dataset {fm.P['db']} (read-only); saved models {P['models']}; "
             f"outputs {P['out']}.")
    con, code = None, 0
    try:
        if not fm.P["db"].exists():
            progress(f"STOP: no dataset at {fm.P['db']}.")
            raise StopRun(1)
        lr, cal_saved, gbm_settings, _ = cb.load_saved()
        if not cb.P["rule"].exists():
            progress(f"STOP: {cb.P['rule']} is missing; run check_beyond_range.py first.")
            raise StopRun(1)
        rule = json.loads(cb.P["rule"].read_text(encoding="utf-8"))
        if "calibration" not in rule:
            progress(f"STOP: {cb.P['rule']} has no isotonic calibration (made by an earlier check_beyond_range.py); rerun it.")
            raise StopRun(1)
        booster = lgb.Booster(model_file=str(P["models"] / "lightgbm_model.txt"))
        check_tree_sizes(booster)
        stamp = guard_test_once(args.rescore_test)
        con = ex.open_dataset(args)
        fm.create_views(con)
        create_test_views(con)
        P["out"].mkdir(parents=True, exist_ok=True)
        rep_header(args, ex.read_meta(con), stamp, rule)
        cats = fm.training_categories(con)
        codes, refs, present = fm.level_codes(cats)
        cols, _ = fm.design_columns(refs, present)
        bd.check_line("the training categories give the saved model's columns", cols == lr["columns"])
        bd.check_line("the training categories give the saved LightGBM category codes",
                      all(gbm_settings["category_codes"][fm.LABEL[c]] == codes[c] for c in fm.CAT_COLS))
        if FAILS:
            raise StopRun(1)

        # odds-ratio standard errors: one Hessian pass over the training loans with the saved coefficients
        t0 = time.time()
        num, cat, y = fm.fetch_rows(con, "train", codes, True)
        fills_num = {c: lr["fill_values"][fm.LABEL[c]] for c in fm.NUM_COLS}
        fills_code = {c: codes[c][lr["fill_values"][fm.LABEL[c]]] for c in fm.CAT_COLS}
        means = np.array([lr["standardization"][fm.LABEL[c]]["mean"] for c in fm.NUM_COLS])
        sds = np.array([lr["standardization"][fm.LABEL[c]]["sd"] for c in fm.NUM_COLS])
        X, _, _, _ = fm.build_design(num, cat, fills_num, fills_code, codes, refs, present, means, sds)
        del num, cat, y
        beta = np.array([lr["coefficients_standardized"][c] for c in lr["columns"]])
        constant = set(lr.get("columns_constant_in_training", []))
        keep = [j for j, c in enumerate(lr["columns"]) if c not in constant]
        H = hessian_pass(X, beta, keep)
        del X
        cov = np.linalg.inv(H)
        se_std = [None] * len(lr["columns"])
        for pos, j in enumerate(keep):
            se_std[j] = math.sqrt(cov[pos, pos])
        shares, n_train = column_shares(con, lr)
        progress(f"Hessian pass over {n_train:,} training loans in {hms(time.time() - t0)}")

        # test loans: one read per agency (the only reads of the test outcome)
        t0 = time.time()
        parts = {agency: fm.fetch_rows(con, view, codes, True) for agency, view in AGENCIES}
        num_t = np.concatenate([parts[a][0] for a, _ in AGENCIES])
        cat_t = np.concatenate([parts[a][1] for a, _ in AGENCIES])
        y_t = np.concatenate([parts[a][2] for a, _ in AGENCIES]).astype(np.int8)
        agency_of = np.concatenate([np.full(len(parts[a][2]), k, dtype=np.int8) for k, (a, _) in enumerate(AGENCIES)])
        del parts
        progress(f"Test loans read: {mc(len(y_t))} in {hms(time.time() - t0)}")
        bd.check_line("there are test loans of both agencies with both outcomes",
                      all((agency_of == k).sum() > y_t[agency_of == k].sum() > 0 for k in range(len(AGENCIES))))

        # Logistic Regression: edge rule, raw log-odds, SHAP, adjusted risk
        num_lr = cb.lr_inputs(rule, num_t)
        X_t, _, _, _ = fm.build_design(num_lr, cat_t, fills_num, fills_code, codes, refs, present, means, sds)
        z_lr = fm.lr_logodds(X_t, beta)
        phi_lr, baseline_lr = lr_shap(lr, X_t, shares)
        del X_t
        bd.check_line("Logistic Regression SHAP values plus the baseline equal the raw log-odds on every test loan (1e-9)",
                      float(np.max(np.abs(phi_lr.sum(axis=1) + baseline_lr - z_lr))) < 1e-9)
        risk_lr = fm.isotonic_apply(rule["calibration"], z_lr)
        by_var_lr = lr_by_variable(lr, phi_lr)
        del phi_lr

        # LightGBM: raw score, SHAP, adjusted risk
        X_g = fm.lgb_matrix(num_t, cat_t, codes, refs, present)
        z_g = booster.predict(X_g, raw_score=True)
        t0 = time.time()
        contrib = booster.predict(X_g, pred_contrib=True)
        progress(f"LightGBM pred_contrib for {len(y_t):,} test loans in {hms(time.time() - t0)}")
        bd.check_line("LightGBM SHAP values plus the expected value equal the raw log-odds on every test loan (1e-6)",
                      float(np.max(np.abs(contrib.sum(axis=1) - z_g))) < 1e-6)
        risk_g = fm.isotonic_apply(cal_saved["lightgbm"], z_g)
        phi_g = contrib[:, :len(fm.NUM_COLS) + len(fm.CAT_COLS)]

        # Q1
        auc_rows, cal_rows = [], []
        for name, mask in [("all test loans", np.ones(len(y_t), dtype=bool))] + [(a, agency_of == k) for k, (a, _) in enumerate(AGENCIES)]:
            a_rows, c_rows = q1_population(name, y_t[mask], {LR: risk_lr[mask], GBM: risk_g[mask]}, {LR: z_lr[mask], GBM: z_g[mask]})
            auc_rows += a_rows
            cal_rows += c_rows
        bd.rsection("Q1", rep_q1, auc_rows, cal_rows)

        # Q2
        imp_lr = {k: float(np.abs(v).mean()) for k, v in by_var_lr.items()}
        imp_gbm = {fm.LABEL[c]: float(np.abs(phi_g[:, j]).mean()) for j, c in enumerate(fm.NUM_COLS + fm.CAT_COLS)}
        bd.rsection("Q2a importance", rep_importance, imp_lr, imp_gbm)
        bd.rsection("Q2b shape", rep_shape, num_t, cat_t, by_var_lr, phi_g, codes)
        progress("Interaction values (exact, per tree) ...")
        res = []
        bd.rsection("Q2c interactions", lambda: res.append(interaction_means(booster, X_g, contrib)))
        if res:
            means_ij, max_diff, max_base_diff, n_trees = res[0]
            bd.rsection("Q2c interactions report", rep_interactions, means_ij, max_diff, max_base_diff, n_trees, len(y_t))
        bd.rsection("Q2d odds ratios", rep_odds, odds_ratio_rows(lr, se_std), n_train)
        rep_files()
        section(f"SUMMARY: {len(FAILS)} problem(s)")
        for f in FAILS:
            out(f"  FAIL: {f}")
        out("  SUCCESS: every output written; every check passed." if not FAILS else "  FAILURE: see the problems above.")
        out(f"(score_test took {hms(time.time() - started)})")
        code = 1 if FAILS else 0
    except StopRun as stop:
        code = stop.code
    except KeyboardInterrupt:
        progress("Interrupted. Nothing to resume: delete nothing; rerun with --rescore-test only if results were written.")
        code = 130
    except Exception as exc:
        bd.write_error_detail(exc, progress)
        code = 1
    finally:
        if con is not None:
            con.close()
        if cb.P.get("spill") and cb.P["spill"].exists() and not any(cb.P["spill"].iterdir()):
            cb.P["spill"].rmdir()
        if check_raw.report:
            inv.save_report()
            progress(f"Report saved to {P['report']}")
        word = "SUCCESS" if code == 0 else "INTERRUPTED" if code == 130 else "FAILURE"
        progress(f"{word}: exit code {code}, elapsed {hms(time.time() - started)}.")
    return code


if __name__ == "__main__":
    sys.exit(main())
