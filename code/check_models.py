"""Five diagnostic checks on the saved models (C1-C5). Summaries only. Diagnostic only: nothing is changed after it.

Diagnostic checks: reads test outcomes after the test was scored; nothing is refit, chosen or tuned here. The saved
models, their isotonic adjustments and the six chosen Q3 rules are used exactly as saved.

Checks
  C1  VALIDATION calibration: ten equal-sized groups per model ranked by the raw score (score_test.py's Q1 groups), all
      validation loans, then each agency. The mean predicted risk is the 5-fold CROSS-FITTED isotonic risk
      (each loan adjusted by a fit on the other four folds), so the overall mean is not equal to the actual bad
      rate by construction (stated in the report).
  C2  TEST (and validation, side by side) predicted vs actual for every value of the 10 category variables (incl. the
      at-the-limit flag): documented category order, other training values, an undocumented-value row when present, then blank.
      Special program R is its own row.
  C3  TEST by origination quarter, both models: loans, bad loans, actual bad rate, mean predicted risk, AUC on the raw
      score (quarter totals), plus ten equal-sized groups by raw score within each quarter.
  C4  TEST (and validation, side by side) by bands of the 7 model number variables (LTV is not a model variable),
      with rows for blanks, loans below the training min and loans above the training max; plus, beside the percent of
      limit, loan-amount DOLLAR bands (loan amount in dollars is not a model variable; its training
      range is the training loans' min and max) and the at-the-limit flag's rows (Y, N, blank). Bands cover loans
      inside the training range and are set from TEST loans:
        loan amount ($): the 14 conforming limits, each rounded to the nearest $1,000 (half up), open bands between them and a
                     row for the loans whose stored amount equals each rounded limit; loans below the first rounded limit
                     ($417,000) in tenths of the test loans below it;
        mortgage insurance %: one row per distinct value (test and validation values), or, when the test has more than 40
                     distinct values, a row for 0 and tenths of the test loans above 0;
        loan term: the product term groups (<= 120, 121-180, 181-240, 241-359) with 360 as its own row;
        the others: tenths of test loans (cut p = the value at position ceil(p x N / 10); equal cuts merge; a band is
                     (previous cut, cut]; the first band has no lower cut and the last band is above the last cut).
  C5  On all TEST loans: is a gap bigger than chance? LightGBM AUC - Logistic Regression AUC, and the Q3 score of each of
      the 15 pairs of the six chosen rules (outputs/q3/chosen_rules.json, applied unchanged). Paired bootstrap of loans
      (plain resampling with replacement), 2,000 resamples, seed 1, 95% percentile intervals (numpy linear
      interpolation); scores with each resample's own bad and good dollar totals; two-sided bootstrap p-value
      min(1, 2 x min(share <= 0, share >= 0)); Holm adjustment over the 16 comparisons; verdict 'bigger than chance' when
      the interval excludes 0 AND the Holm p-value is below 0.05. DeLong's paired test is a cross-check for the AUC gap.

Every C1-C4 row: loans, bad loans, actual bad rate, its 95% Wilson interval, mean predicted risk per model and (mean
predicted risk) / (actual bad rate) per model to 2 decimals. Predicted risk = the isotonic-adjusted risk as scored (the
validation columns of C2 and C4 use the adjustment fitted on all validation loans, in-sample; C1 uses cross-fitted
risks). AUCs (C3, C5) are on the raw score, as in score_test.py. Masking: counts 1-10 show '1-10'; a row whose loan, bad
or good count is 1-10 shows no rate, interval, mean, ratio or AUC; rows keep a fixed order (never sorted by a rate);
validation and test cells are masked separately.

Before any check is written the run reproduces what was published and stops (exit 1) if it cannot:
  - the six rules: each chosen_rules.json rule mapped onto credit_box's training ladders gives the JSON's validation
    counts and dollars (a category boundary next to a hidden group is the first matching one in credit_box's order);
  - (full mode) Q1: score_test's AUC table (outputs/test/q1_auc.csv) and, where no tie straddles a group boundary, its
    calibration table; Q3: outputs/q3/test_results.csv, every cell.
And it reports, per agency and set, how many loan amounts are not a multiple of $1,000 (the rounding premise of C4).

Reuses fit_models.py (paths, views, row reader, AUC, cross-fitted isotonic risks), check_beyond_range.py (training
ranges), score_test.py (rank groups, Q1 tables, the RESCORED statement), credit_box.py (saved models, raw scores and
adjusted risks, ladders, rule application, score, dollars),
explore.py / inventory.py / build_dataset.py / check_raw.py (connection, masking, report, checks, error detail).
New here: the agency x quarter views (score_test's views are test-only and have no quarter), the JSON-to-ladder rule
mapping (the JSON hides masked placements), the band tables, the Wilson interval, the weighted AUC and the bootstrap,
DeLong, Holm.

Files: outputs/checks/ (--smoke: outputs/checks_smoke/), nothing anywhere else:
  check_models_report.txt, check_models_progress.log, check_models_error_detail.txt (stack frames only, on error),
  c1_validation_calibration.csv, c2_categories.csv, c3_quarters.csv, c3_quarter_groups.csv, c4_number_bands.csv,
  c5_auc_gap.csv, c5_q3_gaps.csv.
outputs/test/, outputs/q3/, logs/ and the test-once locks are read only (never written). A rerun overwrites the
outputs/checks files (deterministic: fixed seed).

Usage (from the repository root, after score_test.py and credit_box.py):
    python code/check_models.py [--smoke] [--threads 8] [--memory-limit 2GB]
  --smoke: a 1-in-50 hash sample of every set, models from outputs/models_smoke, rules from outputs/q3_smoke; the
  Q1/Q3 reproduction against outputs/test and outputs/q3/test_results.csv is skipped (smoke runs never wrote them).
Exit codes: 0 SUCCESS; 1 FAILURE (missing files, a reproduction that fails, a failed check); 130 interrupted (rerun).
"""

import csv
import itertools
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset as bd  # noqa: E402  (reused: check lines, report sections, error detail)
import check_beyond_range as cb  # noqa: E402  (reused: training ranges)
import check_raw  # noqa: E402
import credit_box as cx  # noqa: E402  (reused: saved models, adjusted risks, ladders, rule application, score)
import explore as ex  # noqa: E402  (reused: dataset connection, masking helpers, CSV writing)
import fit_models as fm  # noqa: E402  (reused: views, row reader, AUC, cross-fitting)
import inventory as inv  # noqa: E402  (reused: progress log, report file, 1-10 masking)
import score_test as st  # noqa: E402  (reused: rank groups, Q1 tables)
from check_raw import out, out_table, section  # noqa: E402
from inventory import StopRun, lit, m, mc, progress  # noqa: E402
from load_raw import LOG_DIR, PROJECT_ROOT, hms  # noqa: E402

SCRIPT_VERSION = "2"                    # 2: 17 variables (percent of limit and at-the-limit flag); cross-fitted isotonic risks in C1
HEADER = "Diagnostic checks: reads test outcomes after the test was scored; nothing is refit, chosen or tuned here."
MASK, MISSING = ex.MASK, ex.MISSING
LR, GBM = st.LR, st.GBM
MODELS = (LR, GBM)
AGENCIES = st.AGENCIES
METHODS = cx.METHODS
N_GROUPS = st.N_GROUPS
Z95 = st.Z95
RESAMPLES = 2000            # tests may lower this before calling main()
SEED = 1                    # the shared seed (fit_models.SEED)
ALPHA = 0.05
TENTHS = 10
MI_MAX_DISTINCT = 40
LIMITS = [417_000, 424_100, 453_100, 484_350, 510_400, 625_500, 636_150, 647_200, 679_650, 726_200, 726_525, 765_600, 970_800, 1_089_300]
TERM_GROUPS = [("<= 120", None, False, 120, True), ("121-180", 120, False, 180, True), ("181-240", 180, False, 240, True),
               ("241-359", 240, False, 360, False), ("360", 360, True, 360, True)]
SET_COLS = ["loans", "bad_loans", "actual_bad_rate", "wilson_low_95", "wilson_high_95", "mean_predicted_lr", "mean_predicted_lgbm",
            "predicted_over_actual_lr", "predicted_over_actual_lgbm"]
SET_HEADS = ["loans", "bad", "bad rate", "Wilson low", "Wilson high", "mean LR", "mean LightGBM", "LR / actual", "LightGBM / actual"]

WORK_ROOT = PROJECT_ROOT    # tests re-point these two before calling main()
LOGS = LOG_DIR
MODE = "full"
P = {}
FAILS = bd.FAILS
WRITTEN = []
LAST = {}                   # tests only: {work root: the arrays, risks, rules and masks of the last run}


def configure(mode):
    """credit_box's configuration (which sets score_test's, check_beyond_range's and fit_models' paths); then this
    script's own folder, which holds every file it writes (report, progress log, error detail, DuckDB spill)."""
    global MODE
    MODE = mode
    cx.WORK_ROOT = WORK_ROOT
    cx.LOGS = LOGS
    cx.configure(mode)
    sfx = "_smoke" if mode == "smoke" else ""
    P.clear()
    P.update(models=cx.P["models"], q3=cx.P["out"], test=WORK_ROOT / "outputs" / f"test{sfx}", out=WORK_ROOT / "outputs" / f"checks{sfx}")
    P.update(rules=P["q3"] / "chosen_rules.json", q3_test=P["q3"] / "test_results.csv", q1_auc=P["test"] / "q1_auc.csv",
             q1_cal=P["test"] / "q1_calibration.csv", report=P["out"] / "check_models_report.txt",
             progress=P["out"] / "check_models_progress.log", error=P["out"] / "check_models_error_detail.txt",
             spill=P["out"] / "duckdb_spill.tmp")
    inv.REPORT_PATH = P["report"]
    inv.PROGRESS_PATH = P["progress"]
    bd.P["error"] = P["error"]
    ex.P["spill"] = P["spill"]
    cb.P["spill"] = P["spill"]


# ---------------------------------------------------------------- small helpers

def pct(x):
    return f"{100 * x:.4f}%"


def wilson(b, n):
    """95% Wilson score interval for b bad of n loans."""
    p, z2 = b / n, Z95 * Z95
    den = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / den
    half = Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def cells(n, b, means):
    """One set's cells for a row: loans, bad, rate, Wilson low / high, mean predicted risk per model, mean / actual per
    model. A row whose loan, bad or good count is 1-10 shows the masked counts only."""
    n, b = int(n), int(b)
    if n == 0:
        return [0, 0] + ["-"] * (3 + 2 * len(means))
    if not ex.rate_shown(n, b):
        return [m(n), "-" if 1 <= n <= 10 else m(b)] + ["-"] * (3 + 2 * len(means))
    rate = b / n
    lo, hi = wilson(b, n)
    return ([m(n), m(b), pct(rate), pct(lo), pct(hi)] + [pct(mu) for mu in means]
            + ["-" if b == 0 else f"{mu / rate:.2f}" for mu in means])


def grouped(idx, n_rows, y, risks):
    """Per row index 0..n_rows-1: loans, bad, mean of each risk (bincount; idx -1 = no row)."""
    ok = idx >= 0
    i = idx[ok]
    n = np.bincount(i, minlength=n_rows)
    b = np.bincount(i, weights=y[ok].astype(np.float64), minlength=n_rows).round().astype(np.int64)
    sums = [np.bincount(i, weights=r[ok], minlength=n_rows) for r in risks]
    means = [[s[k] / n[k] if n[k] else 0.0 for k in range(n_rows)] for s in sums]
    return n, b, means


def write(path, header, rows):
    ex.write_csv(path, header, [[str(x) for x in r] for r in rows])
    WRITTEN.append(path)


def fmt(v):
    return "-" if v is None else f"{v:,}" if isinstance(v, int) else str(v)


# ---------------------------------------------------------------- data: one view per agency x quarter

def read_set(con, label, codes):
    """Every loan of a label, read through one view per (agency, origination quarter) of fit_models' model_loans view
    with fit_models' row reader. Returns a dict of arrays: num, cat, y, dollars (loan amount in dollars), fold (the
    cross-fitting fold), agency (index into AGENCIES), quarter (index into quarters), quarters."""
    quarters = [r[0] for r in con.execute(f"SELECT DISTINCT origination_quarter FROM kept_loans WHERE label = {lit(label)}"
                                          f"{fm.sample_sql()} ORDER BY 1").fetchall()]
    nums, cats, ys, ags, qs, dols, folds = [], [], [], [], [], [], []
    for k, (agency, _) in enumerate(AGENCIES):
        for qi, q in enumerate(quarters):
            view = f"chk_{label}_{k}_{qi}"
            con.execute(f"CREATE OR REPLACE TEMP VIEW {view} AS SELECT (outcome = 'bad')::TINYINT AS bad, {fm.var_sql()}, "
                        f"CAST({fm.DOLLARS} AS DOUBLE) AS dollars, fold FROM {fm.SOURCE} "
                        f"WHERE label = {lit(label)} AND agency = {lit(agency)} AND origination_quarter = {lit(q)}{fm.sample_sql()}")
            num, cat, y, more = fm.fetch_rows(con, view, codes, True, extra=("dollars", "fold"))
            dols.append(more["dollars"])
            folds.append(more["fold"].astype(np.int8))
            nums.append(num)
            cats.append(cat)
            ys.append(y)
            ags.append(np.full(len(y), k, dtype=np.int8))
            qs.append(np.full(len(y), qi, dtype=np.int8))
    total = con.execute(f"SELECT count(*) FROM kept_loans WHERE label = {lit(label)}{fm.sample_sql()}").fetchall()[0][0]
    d = {"num": np.concatenate(nums), "cat": np.concatenate(cats), "y": np.concatenate(ys).astype(np.int8), "agency": np.concatenate(ags),
         "quarter": np.concatenate(qs), "quarters": quarters, "dollars": np.concatenate(dols), "fold": np.concatenate(folds)}
    bd.check_line(f"{label}: the agency x quarter views hold every {label} loan once ({mc(total)})", len(d["y"]) == total)
    return d


# ---------------------------------------------------------------- the six chosen rules, mapped onto the ladders

def term_candidates(lad, jt):
    """Ladder cutoffs consistent with one JSON term: same value / declined groups, same direction, same blank treatment
    when it is not masked. Returns (kind, ascending cutoffs = credit_box's enumeration order from the risky end)."""
    kind = "value" if jt["cutoff_basis"].startswith("every distinct") else "step"
    allowed = lad.allowed_values() if kind == "value" else lad.allowed_steps()
    found = []
    for c in (int(c) for c in np.flatnonzero(allowed)):
        if lad.kind == "number":
            v = lad.rv[c] if kind == "value" else lad.mark_value(c)
            if np.float32(v) != np.float32(jt["cutoff"]) or ("high" if lad.high else "low") != jt["direction"].split()[0]:
                continue
        elif lad.declined_names(c) != jt["declined_groups"]:
            continue
        term = cx.term_of(lad, kind, c)
        if term["text"] != jt["cutoff_text"]:
            continue
        if jt["blank"] != "masked" and (jt["blank"] == "declined") != term["blank_declined"]:
            continue
        found.append(c)
    return kind, found


def validation_matches(res, jv):
    """The JSON's validation counts (masked as written) and dollars (None when masked) against a plain-mask result."""
    bn, bdol = res["bad_declined"]
    gn, gdol = res["good_declined"]
    want = {"declined_loans": m(bn + gn), "bad_declined_loans": m(bn), "good_declined_loans": m(gn),
            "declined_dollars": None if cx.masked(bn + gn) else bdol + gdol, "bad_declined_dollars": None if cx.masked(bn) else bdol,
            "good_declined_dollars": None if cx.masked(gn) else gdol}
    return all(jv.get(k) == v for k, v in want.items())


def rules_from_json(chosen, ladders, v, risk_v, upb_v):
    """Internal rules (credit_box's structure) from chosen_rules.json. A rule must reproduce the JSON's validation result;
    when several ladder cutoffs do (a boundary next to a hidden group), the first in credit_box's order is used.
    Returns (rules, notes); a rule that cannot be reproduced stops the run."""
    forms = dict(cx.PAIR_FORMS + cx.TRIO_FORMS)
    names = [j["method"] for j in chosen["methods"]]
    bd.check_line("chosen_rules.json lists the six methods in their fixed order", names == METHODS)
    if names != METHODS:
        raise StopRun(1)
    col_of = {lad.col: lad for lad in ladders}
    rules, notes = [], []
    for k, j in enumerate(chosen["methods"]):
        if j["type"] == "none":
            combos = [cx.make_rule(k)]
        elif j["type"] == "model":
            combos = [cx.make_rule(k, type="model", model=j["model"], cutoff=float(j["cutoff_adjusted_risk"]))]
        else:
            per_term = [(col_of[t["column"]], *term_candidates(col_of[t["column"]], t)) for t in j["terms"]]
            combos = [cx.make_rule(k, type="vars", form_text=j["form"], form_fn=forms.get(j["form"]),
                                   terms=[cx.term_of(lad, kind, c) for (lad, kind, _), c in zip(per_term, combo)])
                      for combo in itertools.product(*[c for _, _, c in per_term])]
        hits = []
        for rule in combos:
            res = cx.outcome_of(cx.declined_mask(rule, v["num"], v["cat"], risk_v[LR], risk_v[GBM], ladders), v["y"], upb_v)
            if validation_matches(res, j["validation"]):
                hits.append(rule)
        ok = len(hits) > 0
        bd.check_line(f"{METHODS[k]}: the rule in chosen_rules.json, applied to the validation loans, gives its validation counts and dollars", ok)
        if not ok:
            progress(f"STOP: {METHODS[k]}: no ladder cutoff reproduces chosen_rules.json's validation result ({len(combos)} tried).")
            raise StopRun(1)
        if len(combos) > 1:
            notes.append(f"{METHODS[k]}: {len(combos)} ladder cutoff combinations fit the JSON's wording, {len(hits)} reproduce its validation "
                         "result; the first in credit_box's order is used")
        rules.append(hits[0])
    return rules, notes


# ---------------------------------------------------------------- reproduction of the published results

def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.reader(f)][1:]


def reproduce_q1(t, risk_t, score_t):
    """score_test's Q1 tables from this run's risks: AUC table exactly; calibration table exactly for each
    (population, model) with no tie across a group boundary (otherwise the row order decides, which is not stored)."""
    pops = [("all test loans", np.ones(len(t["y"]), dtype=bool))] + [(a, t["agency"] == k) for k, (a, _) in enumerate(AGENCIES)]
    auc_rows, cal_rows = [], []
    for name, sel in pops:
        a, c = st.q1_population(name, t["y"][sel], {LR: risk_t[LR][sel], GBM: risk_t[GBM][sel]}, {LR: score_t[LR][sel], GBM: score_t[GBM][sel]})
        auc_rows += a
        cal_rows += c
    auc_rows = [[str(x) for x in r] for r in auc_rows]
    cal_rows = [[str(x) for x in r] for r in cal_rows]
    ok_auc = auc_rows == read_rows(P["q1_auc"])
    bd.check_line("Q1 AUC table (loans, bad loans, bad rate, AUC to 6 decimals, ties) equals outputs/test/q1_auc.csv", ok_auc)
    published = read_rows(P["q1_cal"])
    clean = {(r[0], r[1]) for r in auc_rows if r[6] == "0"}
    mine = [r for r in cal_rows if (r[0], r[1]) in clean]
    theirs = [r for r in published if (r[0], r[1]) in clean]
    ok_cal = mine == theirs and len(mine) == N_GROUPS * len(clean)
    bd.check_line(f"Q1 calibration table equals outputs/test/q1_calibration.csv ({len(clean)} of {len(auc_rows)} population x model blocks "
                  "have no tie across a group boundary and are compared)", ok_cal)
    return ok_auc and ok_cal


def test_results(rules, ladders, t, risk_t, upb_t):
    """credit_box's test cells and formatted rows for all test loans and each agency; also the declined masks."""
    masks = {r["method"]: cx.declined_mask(r, t["num"], t["cat"], risk_t[LR], risk_t[GBM], ladders) for r in rules}
    rows = []
    for name, sel in [("all test loans", np.ones(len(t["y"]), dtype=bool))] + [(a, t["agency"] == k) for k, (a, _) in enumerate(AGENCIES)]:
        tot = cx.totals_of(t["y"][sel], upb_t[sel])
        res = {k: cx.outcome_of(mk[sel], t["y"][sel], upb_t[sel]) for k, mk in masks.items()}
        rows += cx.result_rows(name, rules, ladders, res, tot)
    return masks, [[str(x) for x in r] for r in rows]


def rep_reproduce(notes, results):
    section("1. Reproduction of the published results (run stops if any fails)")
    for note in notes:
        out(f"  {note}.")
    for what, ok in results:
        out(f"  {what}: {'reproduced' if ok else 'NOT reproduced'}")


def rep_amounts(counts):
    section("2. Rounding premise of C4: loan amounts that are not a multiple of $1,000 (information, not a failure condition)")
    out_table(["set", "agency", "loans", "not a multiple of $1,000"], counts, left={0, 1})


# ---------------------------------------------------------------- C1 / C3: ten equal-sized risk groups

def group_rows(prefix, y, risk, score):
    """Ten equal-sized groups by the model's raw score (score_test.rank_groups, as Q1) with the mean of `risk` per
    group, plus the population row."""
    rows = []
    if len(y) == 0:
        return rows, 0
    g, straddle = st.rank_groups(score)
    n, b, means = grouped(g.astype(np.int64), N_GROUPS, y, [risk])
    for k in range(N_GROUPS):
        rows.append(prefix + [k + 1] + cells(n[k], b[k], [means[0][k]]))
    rows.append(prefix + ["all"] + cells(len(y), int(y.sum()), [float(risk.mean())]))
    return rows, straddle


def rep_c1(v, cross_v, score_v):
    section("C1. VALIDATION calibration: ten equal-sized groups per model (1 = lowest raw score), all loans then each agency, "
            "cross-fitted risks")
    out(f"  Groups exactly as score_test.py's Q1 groups (rank of the model's raw score, ten groups differing by at most one loan, "
        f"ties by row position). Mean predicted risk = the {fm.N_FOLDS}-fold cross-fitted isotonic risk: each "
        "validation loan is adjusted by a fit on the other four folds, so the overall mean is NOT equal to the actual bad rate by "
        "construction. The validation loans still did other jobs (LightGBM's early stopping and the choice of its setup; the "
        "edge rule), so these rows are not a fully independent check.")
    rows, straddles = [], []
    for name, sel in [("all validation loans", np.ones(len(v["y"]), dtype=bool))] + [(a, v["agency"] == k) for k, (a, _) in enumerate(AGENCIES)]:
        for model in MODELS:
            r, s = group_rows([name, model], v["y"][sel], cross_v[model][sel], score_v[model][sel])
            rows += r
            straddles.append(f"{name} / {model} {mc(s)}")
    heads = ["group", "loans", "bad", "bad rate", "Wilson low", "Wilson high", "mean predicted", "predicted / actual"]
    for name in dict.fromkeys(r[0] for r in rows):
        for model in MODELS:
            out()
            out(f"--- {name}, {model}")
            out_table(heads, [[fmt(x) for x in r[2:]] for r in rows if r[0] == name and r[1] == model], left={0})
    out()
    out("  Loans tied across a group boundary: " + "; ".join(straddles) + ".")
    write(P["out"] / "c1_validation_calibration.csv", ["population", "model", "group", "loans", "bad_loans", "actual_bad_rate", "wilson_low_95",
                                                       "wilson_high_95", "mean_predicted_risk", "predicted_over_actual"], rows)


def rep_c3(t, risk_t, score_t):
    section("C3. TEST by origination quarter, both models: quarter totals with AUC, then ten equal-sized risk groups within each quarter")
    tot_rows, grp_rows, straddles = [], [], []
    for qi, q in enumerate(t["quarters"]):
        sel = t["quarter"] == qi
        y = t["y"][sel]
        n, b = len(y), int(y.sum())
        for model in MODELS:
            a = fm.auc(y, score_t[model][sel]) if ex.rate_shown(n, b) else None
            tot_rows.append([q, model] + cells(n, b, [float(risk_t[model][sel].mean()) if n else 0.0]) + ["-" if a is None else f"{a:.6f}"])
            r, s = group_rows([q, model], y, risk_t[model][sel], score_t[model][sel])
            grp_rows += [row for row in r if row[2] != "all"]
            straddles.append(f"{q} / {model} {mc(s)}")
    out_table(["quarter", "model", "loans", "bad", "bad rate", "Wilson low", "Wilson high", "mean predicted", "predicted / actual", "AUC"],
              [[fmt(x) for x in r] for r in tot_rows], left={0, 1})
    heads = ["group", "loans", "bad", "bad rate", "Wilson low", "Wilson high", "mean predicted", "predicted / actual"]
    for q in t["quarters"]:
        for model in MODELS:
            out()
            out(f"--- {q}, {model}: ten groups within the quarter (1 = lowest raw score)")
            out_table(heads, [[fmt(x) for x in r[2:]] for r in grp_rows if r[0] == q and r[1] == model], left={0})
    out()
    out("  Loans tied across a group boundary: " + "; ".join(straddles) + ".")
    write(P["out"] / "c3_quarters.csv", ["quarter", "model", "loans", "bad_loans", "actual_bad_rate", "wilson_low_95", "wilson_high_95",
                                         "mean_predicted_risk", "predicted_over_actual", "auc"], tot_rows)
    write(P["out"] / "c3_quarter_groups.csv", ["quarter", "model", "group", "loans", "bad_loans", "actual_bad_rate", "wilson_low_95",
                                               "wilson_high_95", "mean_predicted_risk", "predicted_over_actual"], grp_rows)


# ---------------------------------------------------------------- C2 / C4: rows with test and validation side by side

def side_rows(labels, idx_t, idx_v, t, v, risk_t, risk_v):
    """Rows: label, then the test cells, then the validation cells (each set masked on its own)."""
    nt, bt, mt = grouped(idx_t, len(labels), t["y"], [risk_t[LR], risk_t[GBM]])
    nv, bv, mv = grouped(idx_v, len(labels), v["y"], [risk_v[LR], risk_v[GBM]])
    return [[lab] + cells(nt[k], bt[k], [mt[0][k], mt[1][k]]) + cells(nv[k], bv[k], [mv[0][k], mv[1][k]]) for k, lab in enumerate(labels)]


def print_side(rows, first):
    out_table([first] + [f"test {h}" for h in SET_HEADS], [[fmt(x) for x in [r[0]] + r[1:10]] for r in rows], left={0})
    out_table([first] + [f"validation {h}" for h in SET_HEADS], [[fmt(x) for x in [r[0]] + r[10:]] for r in rows], left={0})


def category_rows(c, codes):
    """Row labels and the code of each: codes in fit_models' order (documented category order, then other training values), an
    undocumented-value row (-2), the blank row (-1)."""
    names = sorted(codes[c].items(), key=lambda kv: kv[1])
    return [(name, code) for name, code in names] + [("(value not documented and not in training)", -2), (MISSING, -1)]


def rep_c2(t, v, risk_t, risk_v, codes):
    section(f"C2. TEST and VALIDATION predicted vs actual for every value of the {len(fm.CAT_COLS)} category variables")
    out("  Rows in documented category order (then any other training value), an undocumented-value row only when a set has one, then blank. "
        "Special program R (absent from training) is its own row; both models score it as 'none'.")
    csv_rows = []
    for c in fm.CAT_COLS:
        j = fm.CAT_COLS.index(c)
        spec = category_rows(c, codes)
        ct, cv = t["cat"][:, j].astype(np.int64), v["cat"][:, j].astype(np.int64)
        if not ((ct == -2).any() or (cv == -2).any()):
            spec = [s for s in spec if s[1] != -2]
        pos = {code: k for k, (_, code) in enumerate(spec)}
        lut = lambda a: np.array([pos.get(int(x), -1) for x in range(-2, max(codes[c].values()) + 1)], dtype=np.int64)[a + 2]   # noqa: E731
        rows = side_rows([s[0] for s in spec], lut(ct), lut(cv), t, v, risk_t, risk_v)
        out()
        out(f"--- {fm.LABEL[c]}")
        print_side(rows, "value")
        csv_rows += [[fm.LABEL[c]] + r for r in rows]
    write(P["out"] / "c2_categories.csv", ["variable", "value"] + [f"test_{h}" for h in SET_COLS] + [f"validation_{h}" for h in SET_COLS], csv_rows)


class Band:
    """An interval of stored values: lo / hi None = unbounded; closed flags per end. Compared in float32."""
    def __init__(self, label, lo, lo_closed, hi, hi_closed):
        self.label, self.lo, self.lo_closed, self.hi, self.hi_closed = label, lo, lo_closed, hi, hi_closed

    def holds(self, x):
        ok = np.ones(len(x), dtype=bool)
        if self.lo is not None:
            lo = np.float32(self.lo)
            ok &= (x >= lo) if self.lo_closed else (x > lo)
        if self.hi is not None:
            hi = np.float32(self.hi)
            ok &= (x <= hi) if self.hi_closed else (x < hi)
        return ok


def tenth_cuts(values):
    """Cut p (1..9) = the value at position ceil(p x N / 10) of the sorted values; equal cuts merge."""
    s = np.sort(values)
    n = len(s)
    if n == 0:
        return []
    pos = [(p * n + TENTHS - 1) // TENTHS for p in range(1, TENTHS)]
    return [float(x) for x in np.unique(s[np.array(pos) - 1])]


def tenth_bands(values, text, lo=None, hi=None, hi_text=None):
    """Bands from the tenth cuts of `values`: (lo, c1], (c1, c2], ..., (c_last, hi) — the last band is open at hi, or
    unbounded. lo is an open lower bound (None = unbounded)."""
    cuts = tenth_cuts(values)
    bands, prev = [], lo
    for c in cuts:
        label = f"<= {text(c)}" if prev is None else f"({text(prev)}, {text(c)}]"
        bands.append(Band(label, prev, False, c, True))
        prev = c
    if hi is None:
        bands.append(Band(f"> {text(prev)}" if prev is not None else "all values", prev, False, None, False))
    else:
        bands.append(Band(f"< {hi_text}" if prev is None else f"({text(prev)}, {hi_text})", prev, False, hi, False))
    return bands


def dollars_text(x):
    return f"{int(round(x)):,}"


def rounded_limit(v):
    return (int(v) + 500) // 1000 * 1000


def amount_bands(x_in):
    """Tenths of the test loans below the first rounded limit, then for each limit: the open band up to it and the row
    of loans whose stored amount equals it; above the last rounded limit."""
    r = [rounded_limit(v) for v in LIMITS]
    bands = tenth_bands(x_in[x_in < np.float32(r[0])], dollars_text, hi=r[0], hi_text=dollars_text(r[0]))
    for k, (lim, rl) in enumerate(zip(LIMITS, r)):
        if k:
            bands.append(Band(f"({dollars_text(r[k - 1])}, {dollars_text(rl)})", r[k - 1], False, rl, False))
        bands.append(Band(f"= {dollars_text(rl)} (limit ${lim:,}{', rounded' if rl != lim else ''})", rl, True, rl, True))
    bands.append(Band(f"> {dollars_text(r[-1])}", r[-1], False, None, False))
    return bands


def mi_bands(xt_in, xv_in):
    """One row per distinct value (test and validation values) when the test has at most 40; else 0 and tenths above 0."""
    distinct_t = np.unique(xt_in)
    if len(distinct_t) <= MI_MAX_DISTINCT:
        vals = np.unique(np.concatenate([xt_in, xv_in]))
        return [Band(f"= {cx.f32text(v)}", float(v), True, float(v), True) for v in vals], f"one row per distinct value ({len(distinct_t)} in test)"
    bands = [Band("= 0", 0.0, True, 0.0, True)] + tenth_bands(xt_in[xt_in > 0], cx.f32text, lo=0.0)
    return bands, f"a row for 0 and tenths of the test loans above 0 (test has {len(distinct_t)} distinct values, more than {MI_MAX_DISTINCT})"


def assign(x, bands, lo, hi):
    """Row index per loan: 0 below the training min, 1..len(bands) the bands, then above the max, then blank. Returns
    (index, loans in range matched by no band or by more than one)."""
    idx = np.full(len(x), -1, dtype=np.int64)
    nan = np.isnan(x)
    below, above = ~nan & (x < np.float32(lo)), ~nan & (x > np.float32(hi))
    inside = ~nan & ~below & ~above
    hits = np.zeros(len(x), dtype=np.int64)
    for k, band in enumerate(bands):
        h = inside & band.holds(x)
        idx[h] = k + 1
        hits += h
    idx[below], idx[above], idx[nan] = 0, len(bands) + 1, len(bands) + 2
    return idx, int((inside & (hits != 1)).sum())


def flag_rows(t, v, risk_t, risk_v, codes):
    """The at-the-limit flag's rows (Y, N, blank) with test and validation side by side."""
    j = fm.CAT_COLS.index(fm.FLAG)
    labels, order = [], []
    for lv in fm.LEVELS[fm.FLAG]:
        labels.append(lv)
        order.append(codes[fm.FLAG][lv])
    labels.append(MISSING)
    order.append(-1)

    def idx(d):
        x = d["cat"][:, j].astype(np.int64)
        res = np.full(len(x), -1, dtype=np.int64)
        for k, code in enumerate(order):
            res[x == code] = k
        return res
    return side_rows(labels, idx(t), idx(v), t, v, risk_t, risk_v)


def rep_c4(t, v, risk_t, risk_v, mins, maxes, codes, dollar_range):
    section("C4. TEST and VALIDATION predicted vs actual by bands of the 7 model number variables, plus loan amount in dollars "
            "and the at-the-limit flag")
    out("  Bands cover loans inside the training range and are set from TEST loans; validation loans use the same bands. Rows: "
        "below the training min, the bands, above the training max, blank. Logistic Regression scores beyond-range loans with "
        "their values held at the training min / max (lr_edge_rule.json); the rows are by the stored values. Loan amounts: the "
        "14 conforming limits rounded to the nearest $1,000 (half up); the tenths below the first limit exclude the loans stored "
        "at exactly that limit, which have their own row. Loan term: the product term groups with 360 on its own.")
    out(f"  Loan amount in DOLLARS is not a model variable; its rows (bands, with "
        f"its training range = the training loans' min and max) are kept beside {fm.LABEL[fm.RATIO]} (a model variable, tenths "
        f"of test loans) and {fm.LABEL[fm.FLAG]} (Y, N, blank). Logistic Regression's edge rule holds {fm.LABEL[fm.RATIO]}, not "
        "the dollars.")
    csv_rows, problems = [], 0
    blocks = []
    for c in fm.NUM_COLS:
        if c == fm.RATIO:
            blocks.append(("dollars", "loan amount ($, not a model variable)"))
        blocks.append((c, fm.LABEL[c]))
        if c == fm.RATIO:
            blocks.append(("flag", fm.LABEL[fm.FLAG]))
    for c, label in blocks:
        if c == "flag":
            rows = flag_rows(t, v, risk_t, risk_v, codes)
            out()
            out(f"--- {label}: Y, N, blank")
            print_side(rows, "band")
            csv_rows += [[label] + r for r in rows]
            continue
        if c == "dollars":
            xt, xv = t["dollars"].astype(np.float32), v["dollars"].astype(np.float32)
            lo, hi = dollar_range
        else:
            j = fm.NUM_COLS.index(c)
            xt, xv = t["num"][:, j], v["num"][:, j]
            lo, hi = mins[c], maxes[c]
        in_t = xt[~np.isnan(xt) & (xt >= np.float32(lo)) & (xt <= np.float32(hi))]
        in_v = xv[~np.isnan(xv) & (xv >= np.float32(lo)) & (xv <= np.float32(hi))]
        how = "tenths of test loans"
        if c == "dollars":
            bands, how = amount_bands(in_t), "conforming limits; tenths below the first"
        elif c == "mi_pct":
            bands, how = mi_bands(in_t, in_v)
        elif c == "loan_term":
            bands, how = [Band(*g) for g in TERM_GROUPS], "product term groups, 360 on its own"
        else:
            bands = tenth_bands(in_t, cx.f32text)
        idx_t, bad_t = assign(xt, bands, lo, hi)
        idx_v, bad_v = assign(xv, bands, lo, hi)
        problems += bad_t + bad_v
        labels = ([f"below training min (< {cx.f32text(lo)})"] + [b.label for b in bands]
                  + [f"above training max (> {cx.f32text(hi)})", MISSING])
        rows = side_rows(labels, idx_t, idx_v, t, v, risk_t, risk_v)
        out()
        out(f"--- {label}: {how}")
        print_side(rows, "band")
        csv_rows += [[label] + r for r in rows]
    bd.check_line("C4: every loan inside the training range falls in exactly one band (test and validation)", problems == 0)
    write(P["out"] / "c4_number_bands.csv", ["variable", "band"] + [f"test_{h}" for h in SET_COLS] + [f"validation_{h}" for h in SET_COLS], csv_rows)


# ---------------------------------------------------------------- C5: gaps bigger than chance?

def auc_groups(risk):
    """Tie groups of a risk (ascending) for the weighted AUC."""
    _, g = np.unique(risk, return_inverse=True)
    return g.astype(np.int64), int(g.max()) + 1 if len(g) else 0


def weighted_auc(w, yf, g, G):
    """Mann-Whitney AUC with average ranks for ties, each loan counted w times (w = bootstrap counts)."""
    wb = np.bincount(g, weights=w * yf, minlength=G)
    wg = np.bincount(g, weights=w * (1 - yf), minlength=G)
    below = np.cumsum(wg) - wg
    return float((wb * (below + 0.5 * wg)).sum() / (wb.sum() * wg.sum()))


def delong(y, s1, s2):
    """DeLong's paired test of AUC(s2) - AUC(s1) (placement values with average ranks). Returns (diff, se, low, high, p)."""
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    v10, v01 = [], []
    for s in (s1, s2):
        s = s.astype(np.float64)
        r_all, r_pos, r_neg = rankdata(s), rankdata(s[pos]), rankdata(s[~pos])
        v10.append((r_all[pos] - r_pos) / n0)
        v01.append(1 - (r_all[~pos] - r_neg) / n1)
    a1, a2 = v10[0].mean(), v10[1].mean()
    s10, s01 = np.cov(np.vstack(v10)), np.cov(np.vstack(v01))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / n1 + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n0
    diff = a2 - a1
    se = math.sqrt(max(var, 0.0))
    p = float(2 * norm.sf(abs(diff) / se)) if se > 0 else (1.0 if diff == 0 else 0.0)
    return diff, se, diff - Z95 * se, diff + Z95 * se, p


def boot_p(d):
    """Two-sided bootstrap p-value: min(1, 2 x min(share of resamples <= 0, share >= 0))."""
    return min(1.0, 2 * min(float(np.mean(d <= 0)), float(np.mean(d >= 0))))


def holm(ps):
    """Holm step-down adjusted p-values (same order as given)."""
    k = len(ps)
    order = sorted(range(k), key=lambda i: ps[i])
    adj, running = [0.0] * k, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (k - rank) * ps[i]))
        adj[i] = running
    return adj


def bootstrap(y, upb, score_t, masks, resamples, seed):
    """Paired bootstrap of loans: per resample, the two AUCs and the six rules' scores (own totals). Returns
    (auc gaps, scores R x 6)."""
    n = len(y)
    yf = y.astype(np.float64)
    groups = {model: auc_groups(score_t[model]) for model in MODELS}
    bad_up, good_up = upb * yf, upb * (1 - yf)
    D = np.vstack([masks[k].astype(np.float64) for k in range(len(METHODS))])
    rng = np.random.default_rng(seed)
    gaps, scores = np.empty(resamples), np.empty((resamples, len(METHODS)))
    t0 = time.time()
    for r in range(resamples):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        gaps[r] = weighted_auc(w, yf, *groups[GBM]) - weighted_auc(w, yf, *groups[LR])
        bw, gw = w * bad_up, w * good_up
        scores[r] = 100 * ((D @ bw) / bw.sum() - (D @ gw) / gw.sum())
        if (r + 1) % 200 == 0 or r + 1 == resamples:
            rate = (r + 1) / max(time.time() - t0, 1e-9)
            progress(f"  bootstrap: {r + 1:,} of {resamples:,} resamples; remaining ~{hms((resamples - r - 1) / rate)}")
    return gaps, scores


def c5_tables(t, score_t, upb_t, masks, resamples, seed):
    """Point gaps, bootstrap intervals and p-values, Holm, verdicts; DeLong for the AUC gap. Masked comparisons (a
    hidden AUC or a rule whose test score is hidden) are not computed and are outside the Holm family."""
    y = t["y"]
    n, b = len(y), int(y.sum())
    yf, upbf = y.astype(np.float64), upb_t.astype(np.float64)
    auc_shown = ex.rate_shown(n, b)
    B, G = float(upbf[y == 1].sum()), float(upbf[y == 0].sum())
    point, shown = [], []
    for k in range(len(METHODS)):
        mk = masks[k]
        bn, gn = int((mk & (y == 1)).sum()), int((mk & (y == 0)).sum())
        shown.append(k == 0 or (not cx.masked(bn) and not cx.masked(gn)))
        point.append(100 * (float(upbf[mk & (y == 1)].sum()) / B - float(upbf[mk & (y == 0)].sum()) / G))
    gaps, scores = bootstrap(y, upbf, score_t, masks, resamples, seed)
    comps = []          # (label, point, draws) for every shown comparison
    a_lr, a_g = fm.auc(y, score_t[LR]), fm.auc(y, score_t[GBM])
    if auc_shown:
        comps.append(("AUC: LightGBM - Logistic Regression", a_g - a_lr, gaps))
    pairs = list(itertools.combinations(range(len(METHODS)), 2))
    for i, j in pairs:
        if shown[i] and shown[j]:
            comps.append((f"{METHODS[j]} - {METHODS[i]}", point[j] - point[i], scores[:, j] - scores[:, i]))
    ps = [boot_p(d) for _, _, d in comps]
    adj = holm(ps)
    res = {}
    for (label, pt, d), p, h in zip(comps, ps, adj):
        lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
        res[label] = {"point": pt, "low": lo, "high": hi, "p": p, "holm": h,
                      "verdict": "bigger than chance" if (lo > 0 or hi < 0) and h < ALPHA else "not shown to be bigger than chance"}
    dl = delong(y, score_t[LR], score_t[GBM]) if auc_shown else None
    # the bootstrap's own AUC with unit weights equals fit_models' AUC (the weighted routine's identity)
    ones = np.ones(n)
    same = all(abs(weighted_auc(ones, yf, *auc_groups(score_t[mdl])) - fm.auc(y, score_t[mdl])) < 1e-12 for mdl in MODELS)
    bd.check_line("the bootstrap's AUC routine with every weight 1 equals fit_models' AUC (1e-12)", same)
    return {"auc": (a_lr, a_g, auc_shown), "delong": dl, "res": res, "family": len(comps), "point": point, "shown": shown, "pairs": pairs}


def p_text(p, resamples):
    return f"< {1 / resamples:.4f}" if p == 0 else f"{p:.4f}"


def rep_c5(c5, resamples, seconds):
    section("C5. On all TEST loans: are the gaps bigger than chance? (paired bootstrap of loans; DeLong cross-check for the AUC)")
    out(f"  {resamples:,} resamples of the test loans with replacement (plain, not stratified), seed {SEED}; each resample scores both models' "
        "AUC and the six chosen rules (applied unchanged) with its own bad and good dollar totals. 95% percentile intervals (numpy "
        "linear interpolation). Two-sided bootstrap p = min(1, 2 x min(share of gaps <= 0, share >= 0)); 0 means no resample "
        f"crossed 0 (shown as < 1/{resamples:,}). Holm adjustment over the {c5['family']} comparisons shown. Verdict 'bigger than chance' = the "
        f"interval excludes 0 AND the Holm p-value is below {ALPHA}. Score gaps in percentage points. Bootstrap time {hms(seconds)}.")
    out("  Limitation: loans are resampled one by one, as if independent. Loans of the same origination quarter shared the same "
        "economy (rates, prices, unemployment), so their outcomes move together; with only four test quarters there is no usable "
        "resampling by quarter. The intervals are therefore likely too narrow, and 'bigger than chance' means bigger than "
        "loan-to-loan chance within 2023, not across years.")
    a_lr, a_g, auc_shown = c5["auc"]
    auc_row = []
    out()
    if auc_shown:
        r = c5["res"]["AUC: LightGBM - Logistic Regression"]
        diff, se, dlo, dhi, dp = c5["delong"]
        auc_row = [[f"{a_lr:.6f}", f"{a_g:.6f}", f"{r['point']:.6f}", f"{r['low']:.6f}", f"{r['high']:.6f}", p_text(r["p"], resamples),
                    f"{r['holm']:.4f}", r["verdict"], f"{se:.6f}", f"{dlo:.6f}", f"{dhi:.6f}", f"{dp:.4g}"]]
        out_table(["LR AUC", "LightGBM AUC", "gap", "boot 95% low", "boot 95% high", "boot p", "Holm p", "verdict", "DeLong SE",
                   "DeLong 95% low", "DeLong 95% high", "DeLong p"], auc_row, left={7})
        bd.check_line("DeLong's AUC gap equals the point gap (1e-12)", abs(diff - r["point"]) < 1e-12)
    else:
        out("  AUC gap: masked (the test set's bad or good count is 1-10).")
        auc_row = [["-"] * 7 + ["masked"] + ["-"] * 4]
    write(P["out"] / "c5_auc_gap.csv", ["lr_auc", "lgbm_auc", "gap", "boot_low_95", "boot_high_95", "boot_p", "holm_p", "verdict",
                                        "delong_se", "delong_low_95", "delong_high_95", "delong_p"], auc_row)
    rows = []
    for i, j in c5["pairs"]:
        label = f"{METHODS[j]} - {METHODS[i]}"
        r = c5["res"].get(label)
        if r is None:
            rows.append([METHODS[j], METHODS[i], "-", "-", "-", "-", "-", "masked (a test score is hidden)"])
        else:
            rows.append([METHODS[j], METHODS[i], f"{r['point']:.4f}", f"{r['low']:.4f}", f"{r['high']:.4f}", p_text(r["p"], resamples),
                         f"{r['holm']:.4f}", r["verdict"]])
    out()
    out("  Q3 score gaps, method A - method B (each pair once; a gap with 'no cutoff' is the method's own score, since no cutoff scores 0):")
    out_table(["method A", "method B", "gap", "boot 95% low", "boot 95% high", "boot p", "Holm p", "verdict"], rows, left={0, 1, 7})
    write(P["out"] / "c5_q3_gaps.csv", ["method_a", "method_b", "gap_pct_points", "boot_low_95", "boot_high_95", "boot_p", "holm_p", "verdict"], rows)


# ---------------------------------------------------------------- report

def rep_header(args, meta, saved, chosen):
    rule = saved["rule"]
    out(f"check_models.py v{SCRIPT_VERSION}  {datetime.now():%Y-%m-%d %H:%M}  mode {MODE}  dataset {fm.P['db'].name} (read-only)  "
        f"saved models {P['models'].relative_to(WORK_ROOT)}  chosen rules {P['rules'].relative_to(WORK_ROOT)} (written {chosen.get('written_at', '?')})")
    out(HEADER)
    if MODE != "smoke":
        out(st.RESCORED_LINE)
    out("Summaries only: no loan IDs, no row-level values. Counts from 1 to 10 show as '1-10'; a row whose loan, bad or good count "
        "is 1-10 shows no rate, interval, mean, ratio or AUC ('-'); rows keep a fixed order (never sorted by a rate); test and "
        "validation cells are masked separately.")
    out(f"Dataset built by build_dataset.py v{meta.get('script_version', '?')} at {meta.get('built_at', '?')}; specification file SHA-256 "
        f"{meta.get('decision_log_sha256', '?')}. DuckDB memory limit {args.memory_limit}, threads {args.threads}.")
    out(f"Predicted risk = each model's isotonic-adjusted risk exactly as score_test.py: Logistic Regression with lr_edge_rule.json "
        f"result {rule['result']} " + ("(every number held at its training min / max when outside it)" if rule["result"] == "B"
                                        else "(inputs as stored)")
        + " and that version's isotonic steps; LightGBM, the saved chosen model and its isotonic steps (credit_box.adjusted_risks). "
        "AUCs and the ten groups use the raw score (log-odds), as score_test.py.")
    out("Category inputs, exactly as score_test.py scores them : a documented value with no training "
        "loans (special program R) and an undocumented value (neither documented nor in training) score as the reference "
        "category in both models; a blank scores as the most common training value with the incomplete-application column set "
        "in Logistic Regression, and as missing (NaN) in LightGBM.")
    out(f"Model variables: {len(fm.VARIABLES)}: loan amount in dollars is replaced by {fm.LABEL[fm.RATIO]} and "
        f"{fm.LABEL[fm.FLAG]}; the dollars stay the Q3 weight and keep their C4 rows.")
    out("Each row: loans, bad loans, actual bad rate with its 95% Wilson interval, mean predicted risk per model, and mean predicted "
        "/ actual bad rate per model (2 decimals; '-' when the actual rate is 0).")
    out("Check settings: C4 bands, C5 method, validation side by side in C2-C4, ten groups "
        "within each quarter in C3, Wilson intervals, the predicted / actual column, the reproduction checks and the $1,000 count.")
    if MODE == "smoke":
        out(f"SMOKE MODE: a fixed 1-in-{fm.SMOKE_MOD} hash sample of every set; models from outputs/models_smoke; rules from "
            "outputs/q3_smoke; the Q1 / Q3 test reproduction is skipped (smoke runs never wrote those tables).")


def rep_files():
    section("Files written")
    for p in WRITTEN:
        out(f"  {p.relative_to(WORK_ROOT)}")


# ---------------------------------------------------------------- main

def amount_counts(sets):
    rows = []
    for name, d in sets:
        upb = d["dollars"].astype(np.float64)
        for k, (agency, _) in enumerate(AGENCIES):
            x = upb[d["agency"] == k]
            x = x[~np.isnan(x)]
            rows.append([name, agency, mc(len(x)), mc(int((np.mod(x, 1000) != 0).sum()))])
    return rows


def main(argv=None):
    args = fm.parse_args(argv)
    configure("smoke" if args.smoke else "full")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    P["out"].mkdir(parents=True, exist_ok=True)
    check_raw.report.clear()
    FAILS.clear()
    WRITTEN.clear()
    started = time.time()
    progress("=" * 78)
    progress(f"check_models.py v{SCRIPT_VERSION} started ({MODE} mode). Dataset {fm.P['db']} (read-only); saved models {P['models']}; "
             f"rules {P['rules']}; outputs {P['out']}.")
    con, code = None, 0
    try:
        if not fm.P["db"].exists():
            progress(f"STOP: no dataset at {fm.P['db']}.")
            raise StopRun(1)
        need = [P["rules"]] + ([P["q1_auc"], P["q1_cal"], P["q3_test"]] if MODE == "full" else [])
        missing = [str(p) for p in need if not p.exists()]
        if missing:
            progress("STOP: missing " + "; ".join(missing) + ". Run score_test.py and credit_box.py first.")
            raise StopRun(1)
        chosen = json.loads(P["rules"].read_text(encoding="utf-8"))
        saved = cx.load_models()
        con = ex.open_dataset(args)
        fm.create_views(con)
        rep_header(args, ex.read_meta(con), saved, chosen)
        section("0. Setup checks")
        cats = fm.training_categories(con)
        codes, refs, present = fm.level_codes(cats)
        cols, _ = fm.design_columns(refs, present)
        bd.check_line("the training categories give the saved model's columns", cols == saved["lr"]["columns"])
        bd.check_line("the training categories give the saved LightGBM category codes",
                      all(saved["settings"]["category_codes"][fm.LABEL[c]] == codes[c] for c in fm.CAT_COLS))
        mins, maxes = cb.training_ranges(con, saved["ranges"])
        saved.update(codes=codes, refs=refs, present=present, mins=mins, maxes=maxes)
        if FAILS:
            raise StopRun(1)

        # training loans: credit_box's ladders only (to apply the chosen rules unchanged)
        t0 = time.time()
        num, cat, y = fm.fetch_rows(con, "train", codes, True)
        ladders = cx.build_ladders(num, cat, y, codes, refs, present)
        dollar_range = tuple(float(x) for x in con.execute("SELECT min(dollars), max(dollars) FROM train").fetchone())
        del num, cat, y
        undefined = [lad.label for lad in ladders if lad.kind == "number" and lad.high is None]
        bd.check_line("every number variable has a training direction (credit_box's ladders)", not undefined)
        if undefined:
            raise StopRun(1)
        progress(f"Training loans read and ladders built in {hms(time.time() - t0)}")

        # validation and test loans, one view per agency x quarter, both models' adjusted risks
        t0 = time.time()
        v = read_set(con, "validation", codes)
        score_v = dict(zip(MODELS, cx.raw_scores(v["num"], v["cat"], saved)))
        risk_v = dict(zip(MODELS, cx.adjust_scores(score_v[LR], score_v[GBM], saved)))
        cross_v = {model: fm.cross_fitted_risk(score_v[model], v["y"], v["fold"]) for model in MODELS}
        upb_v, _, _ = cx.dollars_of(v["dollars"], v["y"], "validation")
        t = read_set(con, "test", codes)
        score_t = dict(zip(MODELS, cx.raw_scores(t["num"], t["cat"], saved)))
        risk_t = dict(zip(MODELS, cx.adjust_scores(score_t[LR], score_t[GBM], saved)))
        upb_t, _, _ = cx.dollars_of(t["dollars"], t["y"], "test")
        yv = v["y"].astype(np.float64)
        saved_cross = {LR: saved["rule"][saved["rule"]["result"]]["logloss_cross_fitted"],
                       GBM: saved["cal"]["lightgbm"]["validation"]["logloss_cross_fitted"]}
        for model in MODELS:
            bd.check_line(f"{model}: the cross-fitted validation log loss of C1's risks equals the saved one (1e-12)",
                          abs(fm.log_loss(yv, cross_v[model]) - saved_cross[model]) < 1e-12)
        progress(f"Validation {mc(len(v['y']))} and test {mc(len(t['y']))} loans read and scored in {hms(time.time() - t0)}")
        if FAILS:
            raise StopRun(1)

        # reproduction of the published results: the run stops if any fails
        rules, notes = rules_from_json(chosen, ladders, v, risk_v, upb_v)
        masks, rows = test_results(rules, ladders, t, risk_t, upb_t)
        results = [("the six chosen rules' validation results (chosen_rules.json)", True)]
        if MODE == "full":
            ok_q1 = reproduce_q1(t, risk_t, score_t)
            ok_q3 = rows == read_rows(P["q3_test"])
            bd.check_line("Q3 test results (every cell, all test loans and each agency) equal outputs/q3/test_results.csv", ok_q3)
            results += [("Q1 AUC and calibration tables (outputs/test)", ok_q1), ("Q3 test results (outputs/q3/test_results.csv)", ok_q3)]
        else:
            results += [("Q1 and Q3 test tables: skipped in smoke mode", True)]
        LAST[str(WORK_ROOT)] = {"t": t, "v": v, "risk_t": risk_t, "risk_v": risk_v, "score_t": score_t, "score_v": score_v,
                                "cross_v": cross_v, "upb_t": upb_t, "rules": rules, "masks": masks}
        bd.rsection("1 reproduction", rep_reproduce, notes, results)
        if FAILS:
            progress("STOP: a published result was not reproduced; no check is written.")
            raise StopRun(1)
        bd.rsection("2 amounts", rep_amounts, amount_counts([("validation", v), ("test", t)]))

        bd.rsection("C1", rep_c1, v, cross_v, score_v)
        bd.rsection("C2", rep_c2, t, v, risk_t, risk_v, codes)
        bd.rsection("C3", rep_c3, t, risk_t, score_t)
        bd.rsection("C4", rep_c4, t, v, risk_t, risk_v, mins, maxes, codes, dollar_range)
        progress(f"C1-C4 written; bootstrap of {RESAMPLES:,} resamples for C5 ...")
        t0 = time.time()
        c5 = c5_tables(t, score_t, upb_t.astype(np.float64), masks, RESAMPLES, SEED)
        bd.rsection("C5", rep_c5, c5, RESAMPLES, time.time() - t0)
        rep_files()
        section(f"SUMMARY: {len(FAILS)} problem(s)")
        for f in FAILS:
            out(f"  FAIL: {f}")
        out("  SUCCESS: every output written; every check passed." if not FAILS else "  FAILURE: see the problems above.")
        out(f"(check_models took {hms(time.time() - started)})")
        code = 1 if FAILS else 0
    except StopRun as stop:
        code = stop.code
    except KeyboardInterrupt:
        progress("Interrupted. Nothing to resume: rerun the same command (it overwrites its own outputs).")
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
        word = "SUCCESS" if code == 0 else "INTERRUPTED" if code == 130 else "FAILURE"
        progress(f"{word}: exit code {code}, elapsed {hms(time.time() - started)}.")
    return code


if __name__ == "__main__":
    sys.exit(main())
