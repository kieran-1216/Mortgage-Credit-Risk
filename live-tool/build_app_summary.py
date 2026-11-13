"""Build the small AGGREGATE files the live tool needs. Summaries only; run once.

loan_dataset.duckdb is opened READ-ONLY. What is read:
  * training loans: per-category counts (the Logistic Regression SHAP baseline's column shares, score_test.column_shares;
    and which values have training loans, fit_models.level_codes) and the number of blanks per model variable;
  * the 2023 TEST INPUTS only: the 17 model variables of the test_inputs view (no outcome column, no loan ID); every
    loan is scored with the final models exactly as score_test.py / credit_box.py score it (model_core.adjusted_risks);
  * the monthly median-rate table (rate_medians: first payment month x product, loans, median rate, used);
  * the saved files: outputs/models (models), outputs/q3 (chosen rules, curve, test results), outputs/test (Q1).
Nothing is fitted or chosen. The test OUTCOME is never read: the checks against Q1 and Q3 use their saved summaries.

Rules
  rank        = % of 2023 test loans with an adjusted risk strictly below the applicant's, per model.
  cutoff      each LightGBM curve point's cutoff = the lowest adjusted risk among the loans it declines (the app uses
              chosen_rules.json's exact value at Q3's chosen point).
  median rate 2023 median rate per product = the median of the monthly medians of first payment months 2023-01 to
              2023-12 (months whose group has 30 or more loans, "used"); a product with no such month gets no median
              (spread blank).
  blank note  variables blank in fewer than 1 in 1,000 training loans (RARE_BLANK) are flagged.
  mean risk   each model's exact mean adjusted risk over all 2023 test loans (checked against q1_calibration.csv).

Files written (full mode; --smoke writes app/data_smoke, which the app never reads, from a 1-in-50 hash sample of the
test inputs; the checks that need every test loan are skipped in smoke)
  app/data/app_summary.json            training category values and column shares, rare-blank flags, 2023 median
                                       rates, mean adjusted risks
  app/data/rank_lightgbm.csv           LightGBM: adjusted risk -> share of 2023 test loans below / at or below it
  app/data/rank_logistic_regression.csv  the same for Logistic Regression
  app/data/curve_lightgbm.csv          the LightGBM trade-off curve the app shows: safe stops only (declined share, risk
                                       cutoff, delinquent / performing dollars and shares, the optimal stop)
  logs/build_app_summary_report.txt, logs/build_app_summary_progress.log, logs/build_app_summary_error_detail.txt

Privacy (everything in app/ is public): no loan IDs, no loan-level input or outcome rows, no count anywhere (shares and
risks only); no integer-typed value from 1 to 10; consecutive rows of a rank table are 0 or at least 11 loans apart
(a tie of 1-10 loans is never shown); each step between neighbouring curve stops holds at least 11 loans, 11 delinquent
loans and 11 performing loans (a smaller group's dollars would show by subtraction), with the optimal and the 100%
stops always kept. privacy_problems() checks every file in the output folder (the curve's loan steps from its shares);
the report checks the delinquent and performing steps against curve_lines.csv and lists every file with its row count
and what it holds.

Reuses fit_models.py (paths, read-only connection, views, row reader, category codes), check_beyond_range.py (saved
model loading), score_test.py (column shares, paths) and model_core.py (scoring). New here: the rank tables, the curve
cutoffs, the median query, the privacy check and the file list.

Usage: builds the app's data files from the full project (the dataset and the pipeline's outputs); it is not needed
to run the app. Run it from the project root with this file beside the pipeline scripts in code/, after
score_test.py and credit_box.py have finished; it writes app/data, whose files the app reads from live-tool/data:
    python code/build_app_summary.py [--smoke] [--threads 8] [--memory-limit 2GB]
Exit codes: 0 SUCCESS; 1 FAILURE; 130 interrupted (rerun: the outputs are rewritten).
"""

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset as bd  # noqa: E402  (reused: check lines, report sections, error detail)
import check_beyond_range as cb  # noqa: E402  (reused: saved-model loading)
import check_raw  # noqa: E402
import explore as ex  # noqa: E402  (reused: read-only dataset connection, metadata)
import fit_models as fm  # noqa: E402  (reused: views, row reader, category codes)
import inventory as inv  # noqa: E402  (reused: progress log, report file, 1-10 masking)
import model_core as core  # noqa: E402  (reused: the final models' scoring)
import score_test as st  # noqa: E402  (reused: configuration, column shares)
from check_raw import out, out_table, section  # noqa: E402
from inventory import StopRun, mc, progress  # noqa: E402
from load_raw import LOG_DIR, PROJECT_ROOT, hms  # noqa: E402

SCRIPT_VERSION = "3"
LR, GBM = st.LR, st.GBM
MIN_GROUP = 11                  # a group of loans is shown only if it holds at least this many (1-10 masked)
RANK_MARKS = 10_000             # rank table rows at every 0.01% of the test loans (plus the tied step values)
SMOKE_MOD = fm.SMOKE_MOD        # --smoke: hash(agency, loan_id) % 50 = 0, test inputs only
RARE_BLANK = 0.001              # blank in fewer than 1 in 1,000 training loans -> the blank note
MEDIAN_MONTHS = ("2023-01", "2023-12")
PRODUCTS = ["10-year", "15-year", "20-year", "30-year"]
CURVE_LINE = f"{GBM} risk"
RANK_FILES = {GBM: "rank_lightgbm.csv", LR: "rank_logistic_regression.csv"}
FILES = {
    "app_summary.json": "training category values with loans and the Logistic Regression column shares (the SHAP "
                        "baseline), rare-blank flags per variable, the 2023 median rate per term group, each model's mean "
                        "adjusted risk over 2023 test loans",
    RANK_FILES[GBM]: "LightGBM adjusted risk -> share of 2023 test loans with a lower risk / a lower or equal risk",
    RANK_FILES[LR]: "Logistic Regression adjusted risk -> share of 2023 test loans with a lower risk / a lower or equal risk",
    "curve_lightgbm.csv": "the LightGBM trade-off curve on 2023 test loans at safe stops only (each step 11+ loans, "
                          "11+ delinquent and 11+ performing loans): "
                          "declined share, risk cutoff, delinquent / performing dollars and shares, the optimal stop",
}
# Columns / keys allowed in the files (privacy_problems fails on anything else).
CSV_COLUMNS = {"rank": ["adjusted_risk", "share_below", "share_at_or_below"],
               "curve_lightgbm.csv": ["share_declined", "cutoff_adjusted_risk", "delinquent_dollars_caught", "delinquent_dollars_pct",
                                      "performing_dollars_turned_away", "performing_dollars_pct", "optimal"]}
TEXT_COLUMNS = {"optimal": {"yes", "no"}}   # the only non-number cells
DOLLAR_COLUMNS = {"delinquent_dollars_caught", "performing_dollars_turned_away"}   # sums of loan amounts: long digit strings
LEGACY_FILES = ("curve_cutoffs.csv",)      # an older file name; removed so the folder holds only the current files
JSON_KEYS = {"script", "written_at", "mode", "note", "category_values", "lr_column_shares", "rarely_blank_in_training",
             "rare_blank_rule", "median_rate_2023", "median_rule", "mean_adjusted_risk_2023"}
ID_LIKE = re.compile(r"^(F\d{2}Q[1-4]\w*|\d{9,})$")     # Freddie loan IDs; long digit strings (Fannie loan IDs)

WORK_ROOT = PROJECT_ROOT    # can be re-pointed before calling main()
LOGS = LOG_DIR
APP_DIR = None              # can be re-pointed; default WORK_ROOT / "app"
MODE = "full"
P = {}
FAILS = bd.FAILS
WRITTEN = []                # (path, rows, description)


def configure(mode):
    """score_test's configuration in FULL mode always (the app uses the final models in outputs/models); this script's
    own files; smoke only samples the test inputs and writes app/data_smoke."""
    global MODE
    MODE = mode
    st.WORK_ROOT = WORK_ROOT
    st.LOGS = LOGS
    st.configure("full")
    sfx = "_smoke" if mode == "smoke" else ""
    app = APP_DIR or WORK_ROOT / "app"
    P.clear()
    P.update(models=cb.P["models"], q3=WORK_ROOT / "outputs" / "q3", q1=WORK_ROOT / "outputs" / "test", out=app / f"data{sfx}",
             report=LOGS / f"build_app_summary_report{sfx}.txt", progress=LOGS / "build_app_summary_progress.log",
             error=LOGS / "build_app_summary_error_detail.txt", spill=WORK_ROOT / f"build_app_summary_work{sfx}.tmp")
    inv.REPORT_PATH = P["report"]
    inv.PROGRESS_PATH = P["progress"]
    bd.P["error"] = P["error"]
    ex.P["spill"] = P["spill"]


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--smoke", action="store_true", help=f"1-in-{SMOKE_MOD} hash sample of the test inputs; writes app/data_smoke")
    p.add_argument("--memory-limit", default="2GB", help="DuckDB memory limit (default 2GB)")
    p.add_argument("--threads", type=int, default=8, help="threads for DuckDB and LightGBM (default 8)")
    args = p.parse_args(argv)
    if not re.fullmatch(r"\d+(\.\d+)?\s*[KMGT]i?B", args.memory_limit, re.IGNORECASE):
        p.error("--memory-limit must look like 2GB or 2048MB")
    return args


# ---------------------------------------------------------------- saved files

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def count_of(text):
    """A count as written by the project's CSVs ("1,234" or 1234); None when masked ("1-10") or blank."""
    t = str(text).replace(",", "").strip()
    return int(t) if t.isdigit() else None


def load_models():
    """The final models as score_test.py loads them."""
    lr, cal, settings, _ = cb.load_saved()
    rule_path = P["models"] / "lr_edge_rule.json"
    if not rule_path.exists():
        progress(f"STOP: {rule_path} is missing; run check_beyond_range.py first.")
        raise StopRun(1)
    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    if "calibration" not in rule:
        progress(f"STOP: {rule_path} has no isotonic calibration; rerun check_beyond_range.py.")
        raise StopRun(1)
    booster = lgb.Booster(model_file=str(P["models"] / "lightgbm_model.txt"))
    return {"lr": lr, "cal": cal, "settings": settings, "rule": rule, "booster": booster}


def load_results():
    need = [P["q3"] / "chosen_rules.json", P["q3"] / "curve_lines.csv", P["q3"] / "curve_dots.csv", P["q3"] / "test_results.csv",
            P["q1"] / "q1_auc.csv", P["q1"] / "q1_calibration.csv"]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        progress("STOP: saved results missing (run score_test.py and credit_box.py first): " + "; ".join(missing))
        raise StopRun(1)
    rules = json.loads(need[0].read_text(encoding="utf-8"))
    cut = {r["model"]: r["cutoff_adjusted_risk"] for r in rules["methods"] if r.get("type") == "model"}
    tests = {r["method"]: r for r in read_csv(need[3]) if r["population"] == "all test loans"}
    declined = {}
    for model in (LR, GBM):
        r = tests[f"{model} risk"]
        a, b = count_of(r["bad_declined_loans"]), count_of(r["good_declined_loans"])
        declined[model] = None if a is None or b is None else a + b
    cal_rows = [r for r in read_csv(need[5]) if r["population"] == "all test loans"]
    curve = [r for r in read_csv(need[1]) if r["line"] == CURVE_LINE]
    dot = next(r for r in read_csv(need[2]) if r["method"] == CURVE_LINE)
    return {"cutoff": cut, "declined": declined, "calibration": cal_rows, "curve": curve, "optimal_k": count_of(dot["loans_declined"])}


# ---------------------------------------------------------------- the aggregates

def rank_rows(risk, marks=RANK_MARKS, min_group=MIN_GROUP):
    """Rank table rows (value, loans below, loans at or below), ascending. Candidates: the risk at every 1/marks of the
    loans and every value tied by at least min_group loans. A candidate is kept only when the loans between it and the
    last kept row number 0 or at least min_group; a tie of fewer than min_group loans is not shown (at or below =
    below); a candidate that adds no loans and shows no tie is dropped. So consecutive cumulative counts in the table
    differ by 0 or by min_group or more, and every row stands for at least min_group loans (the first row, the
    minimum, can stand for fewer)."""
    r = np.sort(np.asarray(risk, dtype=np.float64))
    n = len(r)
    if n == 0:
        return []
    vals, first, counts = np.unique(r, return_index=True, return_counts=True)
    pos = np.ceil(np.arange(1, marks + 1) * n / marks).astype(np.int64) - 1
    cand = np.union1d(r[np.clip(pos, 0, n - 1)], vals[counts >= min_group])
    rows, last = [], 0
    for v in cand:
        i = int(np.searchsorted(vals, v))
        below, tie = int(first[i]), int(counts[i])
        gap = below - last
        if 1 <= gap < min_group:
            continue
        shown_tie = tie >= min_group
        if rows and gap == 0 and not shown_tie:
            continue
        at_or_below = below + tie if shown_tie else below
        rows.append((float(v), below, at_or_below))
        last = at_or_below
    return rows


def _share(text):
    """A share as written in curve_lines.csv ("76.7125%"), in percent; None when masked."""
    t = str(text).replace("%", "").strip()
    try:
        return float(t)
    except ValueError:
        return None


CURVE_START = (0, 0, 0)          # the no-cutoff stop: 0 loans, 0 delinquent, 0 performing declined


def step_ok(prev, cur, min_group=MIN_GROUP):
    """True when the step between two neighbouring curve stops (tuples starting k, delinquent loans, performing loans;
    the first stop's step is from CURVE_START) holds at least min_group loans, at least min_group delinquent loans and
    at least min_group performing loans: a smaller group's dollars would be readable by subtraction."""
    return all(b - a >= min_group for a, b in zip(prev[:3], cur[:3]))


def safe_curve(curve, optimal_k, min_group=MIN_GROUP):
    """The stops of the LightGBM curve the app shows. One stop per distinct declined count of curve_lines.csv (masked
    points dropped). A stop is kept only when its step from the last kept stop (from 0 declined) passes step_ok: at
    least min_group loans, delinquent loans and performing loans. The optimal stop (Q3's chosen rule, optimal_k loans)
    and the 100% stop are always kept, and a kept stop whose step to either fails is dropped. Returns ascending tuples
    (k, delinquent loans, performing loans, delinquent $, delinquent %, performing $, performing %); the two loan counts
    are used for the report only and never written."""
    pts = {}
    for r in curve:
        k = count_of(r["loans_declined"])
        cells = (count_of(r["bad_loans_caught"]), count_of(r["good_loans_declined"]), count_of(r["bad_dollars_caught"]),
                 _share(r["bad_dollars_pct"]), count_of(r["good_dollars_declined"]), _share(r["good_dollars_pct"]))
        if k is None or k == 0 or any(c is None for c in cells):
            continue
        pts[k] = (k, *cells)
    if not pts:
        return []
    must = {optimal_k, max(pts)} & set(pts)
    kept = []
    for k in sorted(pts):
        if k in must:
            while kept and kept[-1] not in must and not step_ok(pts[kept[-1]], pts[k], min_group):
                kept.pop()
            kept.append(k)
        elif step_ok(pts[kept[-1]] if kept else CURVE_START, pts[k], min_group):
            kept.append(k)
    return [pts[k] for k in kept]


def cutoffs_at(ks, risk_g, n_curve):
    """The cutoff of a stop declining k loans = the lowest adjusted risk among the k riskiest test loans (k scaled
    to the sample in smoke). Returns [(cutoff, loans with risk >= cutoff)]."""
    desc = np.sort(risk_g)[::-1]
    n = len(desc)
    res = []
    for k in ks:
        if n_curve and n != n_curve:
            k = max(1, int(round(k * n / n_curve)))
        c = float(desc[min(k, n) - 1])
        res.append((c, int(np.sum(risk_g >= c))))
    return res


def curve_file_rows(stops, cuts, optimal_k, chosen_cutoff):
    """The rows of curve_lightgbm.csv: share declined (k / all test loans), the cutoff (Q3's exact cutoff at the optimal
    stop; None in smoke, where the sampled cutoffs can lie above it), dollars and shares as published in
    curve_lines.csv, and the optimal flag."""
    n_all = stops[-1][0]
    rows = []
    for (k, _, _, bad_d, bad_pct, good_d, good_pct), (c, _) in zip(stops, cuts):
        opt = k == optimal_k
        rows.append([fnum(k / n_all), fnum(chosen_cutoff if opt and chosen_cutoff is not None else c), bad_d, fnum(bad_pct), good_d, fnum(good_pct),
                     "yes" if opt else "no"])
    return rows


def medians_2023(con):
    """Per product, the median of the used monthly medians of first payment months 2023-01 to 2023-12.
    Returns {product: (median or None, months used, loans in those months)}."""
    got = con.execute("SELECT product, quantile_cont(median_rate, 0.5), count(*), sum(n_loans) FROM rate_medians "
                      f"WHERE used AND first_payment_month BETWEEN '{MEDIAN_MONTHS[0]}' AND '{MEDIAN_MONTHS[1]}' "
                      "GROUP BY product").fetchall()
    found = {p: (float(med), int(k), int(nl)) for p, med, k, nl in got}
    return {p: found.get(p, (None, 0, 0)) for p in PRODUCTS}


def blank_counts(con):
    sel = ", ".join(f"count(*) FILTER (WHERE {c} IS NULL)" for c in fm.NUM_COLS + fm.CAT_COLS)
    got = con.execute(f"SELECT count(*), {sel} FROM train").fetchone()
    return int(got[0]), dict(zip(fm.NUM_COLS + fm.CAT_COLS, map(int, got[1:])))


def test_view(con):
    """Full: fit_models' test_inputs view (the 17 variables, no outcome). Smoke: the same columns for a 1-in-50 hash
    sample (the loan ID is used inside the database only)."""
    if MODE != "smoke":
        return "test_inputs"
    con.execute(f"CREATE OR REPLACE TEMP VIEW app_test_sample AS SELECT {fm.var_sql()} FROM {fm.SOURCE} "
                f"WHERE label = 'test' AND hash(agency || '|' || loan_id) % {SMOKE_MOD} = 0")
    return "app_test_sample"


# ---------------------------------------------------------------- privacy

def _int_cell(text):
    t = str(text).strip()
    return int(t) if re.fullmatch(r"-?\d+", t) else None


def _json_values(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _json_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _json_values(v)
    else:
        yield obj


def privacy_problems(folder, n_test=None, n_curve=None):
    """Every file in the folder: aggregate only. No unexpected file, column or key; no loan-ID-like text; no integer-typed
    value from 1 to 10; rank tables ascending with shares in [0, 1] and (n_test given) consecutive cumulative counts 0 or
    at least MIN_GROUP apart; no more rows than the loans could support; the curve file's stops ascending, ending at
    100%, one optimal stop, cutoffs not rising and (n_curve given: its test loans, default n_test) every stop at least
    MIN_GROUP loans after the one before. Returns a list of problems (empty = OK)."""
    n_curve = n_curve or n_test
    probs = []
    folder = Path(folder)
    for path in sorted(folder.iterdir()):
        name = path.name
        if name not in FILES:
            probs.append(f"{name}: not a file this script writes")
            continue
        if name.endswith(".json"):
            obj = json.loads(path.read_text(encoding="utf-8"))
            extra = set(obj) - JSON_KEYS
            if extra:
                probs.append(f"{name}: unexpected keys {sorted(extra)}")
            for v in _json_values(obj):
                if isinstance(v, bool):
                    continue
                if isinstance(v, int) and 1 <= v <= 10:
                    probs.append(f"{name}: an integer from 1 to 10 ({v})")
                if isinstance(v, str) and ID_LIKE.match(v):
                    probs.append(f"{name}: loan-ID-like text")
            continue
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        header, body = rows[0], rows[1:]
        want = CSV_COLUMNS["rank"] if name.startswith("rank_") else CSV_COLUMNS.get(name)
        if header != want:
            probs.append(f"{name}: columns {header}, expected {want}")
            continue
        text_cols = [j for j, c in enumerate(header) if c in TEXT_COLUMNS]
        for row in body:
            for j in text_cols:
                if j < len(row) and row[j] not in TEXT_COLUMNS[header[j]]:
                    probs.append(f"{name}: an unexpected value in column {header[j]}")
            for j, cell in enumerate(row):
                iv = _int_cell(cell)
                if iv is not None and 1 <= iv <= 10:
                    probs.append(f"{name}: an integer from 1 to 10 ({cell})")
                if ID_LIKE.match(cell.strip()) and not (j < len(header) and header[j] in DOLLAR_COLUMNS and iv is not None):
                    probs.append(f"{name}: loan-ID-like text")
        try:
            numbers = [[None if c == "" or j in text_cols else float(c) for j, c in enumerate(r)] for r in body]
        except ValueError:
            probs.append(f"{name}: a cell that is not a number")
            continue
        if any(len(r) != len(want) for r in body):
            probs.append(f"{name}: a row with the wrong number of cells")
            continue
        if name.startswith("rank_"):
            vals = [r[0] for r in numbers]
            sh = [(r[1], r[2]) for r in numbers]
            if any(b2 <= b1 for b1, b2 in zip(vals, vals[1:])):
                probs.append(f"{name}: risks not strictly ascending")
            if any(not 0 <= a <= 1 or not 0 <= b <= 1 or b < a for a, b in sh):
                probs.append(f"{name}: shares outside [0, 1] or at-or-below < below")
            if n_test:
                seq = []
                for a, b in sh:
                    seq += [round(a * n_test), round(b * n_test)]
                if any(1 <= d < MIN_GROUP for d in np.diff(seq)):
                    probs.append(f"{name}: two neighbouring shares differ by 1-10 loans")
                if len(body) > n_test // MIN_GROUP + 1:
                    probs.append(f"{name}: more rows than groups of {MIN_GROUP} loans")
        if name == "curve_lightgbm.csv":
            share, cut = [r[0] for r in numbers], [r[1] for r in numbers]
            if not body or any(y <= x for x, y in zip(share, share[1:])) or not 0 < share[0] or share[-1] != 1.0:
                probs.append(f"{name}: declined shares not strictly ascending in (0, 1] ending at 1")
            if any(c is None or not 0 <= c <= 1 for c in cut) or any(y > x for x, y in zip(cut, cut[1:])):
                probs.append(f"{name}: a cutoff missing, outside [0, 1] or rising")
            if any(not 0 <= r[j] <= 100 for r in numbers for j in (3, 5)):
                probs.append(f"{name}: a dollar share outside [0, 100]")
            if [r[6] for r in body].count("yes") != 1:
                probs.append(f"{name}: not exactly one optimal stop")
            if n_curve:
                ks = [0] + [round(x * n_curve) for x in share]
                if any(1 <= d < MIN_GROUP for d in np.diff(ks)):
                    probs.append(f"{name}: two neighbouring stops differ by 1-10 loans")
                if len(body) > n_curve // MIN_GROUP + 1:
                    probs.append(f"{name}: more rows than groups of {MIN_GROUP} loans")
    return probs


# ---------------------------------------------------------------- writing

def write_json(name, obj):
    path = P["out"] / name
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    WRITTEN.append((path, 1, FILES[name]))


def write_rows(name, header, rows):
    path = P["out"] / name
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    WRITTEN.append((path, len(rows), FILES[name]))


def fnum(v):
    """Full precision (round-trips exactly), always with a decimal point or exponent."""
    return repr(float(v))


# ---------------------------------------------------------------- main

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
    progress(f"build_app_summary.py v{SCRIPT_VERSION} started ({MODE} mode). Dataset {fm.P['db']} (read-only); saved models "
             f"{P['models']}; outputs {P['out']}.")
    con, code = None, 0
    try:
        if not fm.P["db"].exists():
            progress(f"STOP: no dataset at {fm.P['db']}.")
            raise StopRun(1)
        saved = load_models()
        res = load_results()
        con = ex.open_dataset(args)
        fm.create_views(con)
        meta = ex.read_meta(con)
        section(f"build_app_summary.py v{SCRIPT_VERSION}: aggregate files for the live tool ({MODE} mode)")
        out(f"  Run {datetime.now():%Y-%m-%d %H:%M:%S}. Dataset {fm.P['db']} opened read-only (built {meta.get('built_at', '-')}). "
            f"Saved models {P['models']}.")
        out("  Reads training category and blank counts, the 2023 test INPUTS (never the outcome), the monthly median-rate table and "
            "the saved model, Q1 and Q3 files. Writes shares and risks only (no counts, no loan rows).")
        if MODE == "smoke":
            out(f"  SMOKE: a 1-in-{SMOKE_MOD} hash sample of the test inputs; outputs go to {P['out']} (never read by the app); "
                "the checks that need every test loan are skipped.")

        # 1. training: category values, column shares, blanks
        section("1. Training loans: category values, Logistic Regression column shares, blanks")
        cats = fm.training_categories(con)
        codes, refs, present = fm.level_codes(cats)
        cols, _ = fm.design_columns(refs, present)
        bd.check_line("the training categories give the saved model's columns", cols == saved["lr"]["columns"])
        bd.check_line("the training categories give the saved LightGBM category codes and references",
                      all(saved["settings"]["category_codes"][fm.LABEL[c]] == codes[c]
                          and saved["settings"]["reference_categories"][fm.LABEL[c]] == refs[c] for c in fm.CAT_COLS))
        if FAILS:
            raise StopRun(1)
        shares, n_train = st.column_shares(con, saved["lr"])
        small = [f"{fm.LABEL[c]} = {v}" for c in fm.CAT_COLS for v, k in cats[c].items() if 1 <= k < MIN_GROUP]
        inc = saved["lr"]["incomplete_application"]["training_loans"]
        bd.check_line(f"every category value with training loans has at least {MIN_GROUP} (its share is shown)", not small)
        bd.check_line(f"incomplete applications in training: 0 or at least {MIN_GROUP}", not 1 <= inc < MIN_GROUP)
        if FAILS:
            out("  STOP: a share would reveal a group of 1-10 training loans; nothing is written.")
            raise StopRun(1)
        rows = [[fm.LABEL[c], v, mc(cats[c].get(v, 0)), "yes" if v in present[c] else "no (scores as " + refs[c] + ")"]
                for c in fm.CAT_COLS for v in codes[c]]
        out(f"  Training loans: {mc(n_train)}.")
        out_table(["variable", "value", "training loans", "has training loans"], rows)
        n_blank_train, blanks = blank_counts(con)
        rare = {fm.LABEL[c]: blanks[c] < RARE_BLANK * n_blank_train for c in fm.NUM_COLS + fm.CAT_COLS}
        out()
        out(f"  Blanks (the blank note is shown when blank in fewer than 1 in {round(1 / RARE_BLANK):,} training loans):")
        out_table(["variable", "training blanks", "note shown when blank"], [[fm.LABEL[c], mc(blanks[c]), "yes" if rare[fm.LABEL[c]] else "no"]
                                                                               for c in fm.NUM_COLS + fm.CAT_COLS])

        # 2. medians
        section("2. 2023 median rate per term group")
        meds = medians_2023(con)
        out(f"  Median of the monthly medians, first payment months {MEDIAN_MONTHS[0]} to {MEDIAN_MONTHS[1]}, months with 30 or "
            "more loans only.")
        out_table(["term group", "months used", "loans in those months", "2023 median rate"],
                  [[p, k, mc(nl), "-" if med is None else f"{med:.4f}"] for p, (med, k, nl) in meds.items()])
        missing = [p for p, (med, _, _) in meds.items() if med is None]
        if missing:
            out(f"  No median for {', '.join(missing)}: the app leaves the rate spread blank for those terms.")
        bd.check_line("the 30-year term group has a 2023 median", meds["30-year"][0] is not None)

        # 3. test inputs scored
        section("3. 2023 test inputs scored with the final models")
        t0 = time.time()
        num, cat, _ = fm.fetch_rows(con, test_view(con), codes, False)
        saved.update(codes=codes, refs=refs, present=present)
        risk_lr, risk_g = core.adjusted_risks(num, cat, saved)
        n = len(risk_g)
        del num, cat
        progress(f"Test inputs scored: {mc(n)} in {hms(time.time() - t0)}")
        mean = {LR: float(risk_lr.mean()), GBM: float(risk_g.mean())}
        out(f"  Test loans scored: {mc(n)}.")
        for model, risk in ((LR, risk_lr), (GBM, risk_g)):
            q = [r for r in res["calibration"] if r["model"] == model]
            k = sum(count_of(r["loans"]) for r in q)
            published = sum(count_of(r["loans"]) * float(r["mean_predicted_risk"].rstrip("%")) for r in q) / k
            out(f"  {model}: mean adjusted risk {100 * mean[model]:.6f}% (q1_calibration.csv, from 4-decimal group means: {published:.6f}%).")
            ge = int(np.sum(risk >= res["cutoff"][model]))
            if MODE != "smoke":
                bd.check_line(f"{model}: the mean equals q1_calibration.csv's loan-weighted group means (within their rounding)",
                              k == n and abs(100 * mean[model] - published) <= 0.00005 + 1e-9)
                if res["declined"][model] is None:
                    out(f"  {model}: the Q3 declined count is masked in test_results.csv (1-10 bad or good loans): not compared.")
                else:
                    bd.check_line(f"{model}: test loans with risk >= the Q3 chosen cutoff = the declined loans in test_results.csv ({mc(ge)})",
                                  ge == res["declined"][model])

        # 4. rank tables
        section("4. Rank tables")
        ranks = {model: rank_rows(risk) for model, risk in ((GBM, risk_g), (LR, risk_lr))}
        for model, rr in ranks.items():
            gaps = np.diff([x for _, b, a in rr for x in (b, a)])
            out(f"  {model}: {len(rr):,} rows; distinct risks {len(np.unique(risk_g if model == GBM else risk_lr)):,}; "
                f"largest gap between rows {int(gaps.max()) if len(gaps) else 0:,} loans.")
            bd.check_line(f"{model}: neighbouring cumulative counts differ by 0 or at least {MIN_GROUP}",
                          not any(1 <= d < MIN_GROUP for d in gaps))

        # 5. the curve the app shows
        section("5. LightGBM trade-off curve for the app (safe stops)")
        n_curve = max((count_of(r["loans_declined"]) or 0) for r in res["curve"]) if res["curve"] else 0
        stops = safe_curve(res["curve"], res["optimal_k"])
        ks = [x[0] for x in stops]
        cuts = cutoffs_at(ks, risk_g, n_curve)
        distinct = len({count_of(r["loans_declined"]) for r in res["curve"]} - {None, 0})
        out(f"  curve_lines.csv: {len(res['curve']):,} LightGBM points, {distinct:,} distinct declined counts; kept {len(stops):,} "
            f"stops (each step at least {MIN_GROUP} loans, {MIN_GROUP} delinquent and {MIN_GROUP} performing loans; the "
            "optimal and 100% stops always kept).")
        out(f"  Optimal stop: {mc(res['optimal_k'])} loans declined ({100 * res['optimal_k'] / n_curve:.2f}%), "
            f"cutoff {100 * res['cutoff'][GBM]:.6f}%.")
        bd.check_line("the optimal stop and the 100% stop are kept", res["optimal_k"] in ks and bool(ks) and ks[-1] == n_curve)
        bd.check_line(f"every step from the stop before (from 0) holds at least {MIN_GROUP} loans, {MIN_GROUP} delinquent loans "
                      f"and {MIN_GROUP} performing loans", all(step_ok(a, b) for a, b in zip([CURVE_START] + stops, stops)))
        bd.check_line("the stops' declined shares are distinct at 2 decimals (the slider's labels)",
                      len({round(100 * k / n_curve, 2) for k in ks}) == len(ks))
        if MODE != "smoke":
            bd.check_line("the curve's last point declines every test loan (its count equals the test loans scored)", n_curve == n)
            bd.check_line("at every stop, the loans with risk >= its cutoff are exactly the loans it declines (ties never split)",
                          all(ge == k for k, (_, ge) in zip(ks, cuts)))
            bd.check_line("at the optimal stop, the loans with risk >= Q3's cutoff are exactly its loans",
                          int(np.sum(risk_g >= res["cutoff"][GBM])) == res["optimal_k"])

        # 6. write and check privacy
        P["out"].mkdir(parents=True, exist_ok=True)
        for old in P["out"].iterdir():
            if old.name in FILES or old.name in LEGACY_FILES:
                old.unlink()
        cat_values = {fm.LABEL[c]: present[c] for c in fm.CAT_COLS}
        write_json("app_summary.json", {
            "script": f"build_app_summary.py v{SCRIPT_VERSION}", "written_at": f"{datetime.now():%Y-%m-%dT%H:%M:%S}", "mode": MODE,
            "note": "aggregates only: shares and risks, no counts and no loan rows",
            "category_values": cat_values,
            "lr_column_shares": {k: float(v) for k, v in shares.items()},
            "rarely_blank_in_training": rare,
            "rare_blank_rule": f"blank in fewer than 1 in {round(1 / RARE_BLANK):,} training loans",
            "median_rate_2023": {p: med for p, (med, _, _) in meds.items()},
            "median_rule": f"median of the monthly medians, first payment months {MEDIAN_MONTHS[0]} to {MEDIAN_MONTHS[1]}, "
                           "months with 30 or more loans",
            "mean_adjusted_risk_2023": mean})
        for model, rr in ranks.items():
            write_rows(RANK_FILES[model], CSV_COLUMNS["rank"], [[fnum(v), fnum(b / n), fnum(a / n)] for v, b, a in rr])
        write_rows("curve_lightgbm.csv", CSV_COLUMNS["curve_lightgbm.csv"],
                   curve_file_rows(stops, cuts, res["optimal_k"], res["cutoff"][GBM] if MODE != "smoke" else None))
        section("6. Privacy check of every file in the output folder")
        probs = privacy_problems(P["out"], n, n_curve)
        for pr in probs:
            out(f"  PROBLEM: {pr}")
        if probs:
            for p, _, _ in WRITTEN:
                p.unlink(missing_ok=True)
            out("  The files written by this run were removed: nothing unsafe is left in the output folder.")
            WRITTEN.clear()
        bd.check_line("every file is aggregate only (no IDs, no loan rows, no integer 1-10, rank rows and curve stops 0 or 11+ loans apart)", not probs)

        section("7. FILES WRITTEN")
        out_table(["file", "rows", "what it holds"], [[str(p.relative_to(WORK_ROOT)) if p.is_relative_to(WORK_ROOT) else str(p), r, d]
                                                       for p, r, d in WRITTEN], left={0, 2})
        section(f"SUMMARY: {len(FAILS)} problem(s)")
        for f in FAILS:
            out(f"  FAIL: {f}")
        out("  SUCCESS: every output written; every check passed." if not FAILS else "  FAILURE: see the problems above.")
        out(f"(build_app_summary took {hms(time.time() - started)})")
        code = 1 if FAILS else 0
    except StopRun as stop:
        code = stop.code
    except KeyboardInterrupt:
        progress("Interrupted. Rerun the same command: the outputs are rewritten.")
        code = 130
    except Exception as exc:
        bd.write_error_detail(exc, progress)
        code = 1
    finally:
        if con is not None:
            con.close()
        if P.get("spill") and P["spill"].exists() and not any(P["spill"].iterdir()):
            P["spill"].rmdir()
        if check_raw.report:
            inv.save_report()
            progress(f"Report saved to {P['report']}")
        word = "SUCCESS" if code == 0 else "INTERRUPTED" if code == 130 else "FAILURE"
        progress(f"{word}: exit code {code}, elapsed {hms(time.time() - started)}.")
    return code


if __name__ == "__main__":
    sys.exit(main())
