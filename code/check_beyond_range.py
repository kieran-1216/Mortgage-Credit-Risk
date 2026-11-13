"""Edge rule check: should Logistic Regression hold beyond-range values at the
training edge? Summaries only.

Rule, on VALIDATION loans:
  (A) Logistic Regression as is (the saved model, inputs as stored);
  (B) the same saved model, unchanged (no refit), with every number variable's value held at its training min or max
      when it falls below or above it (all 7 number variables, both ends; the training mins and maxes of fit_models
      output 5).
  Each version is adjusted by isotonic calibration with 5-fold cross-fitting (each loan adjusted by a
  fit on the other four folds). The version with the lower cross-fitted validation log loss wins. If they are equal
  to 6 decimal places, A wins. The winner's adjustment used for scoring is its isotonic calibration fitted on all
  validation loans.

Reuses fit_models.py (paths, read-only connection, views, row reader, Logistic Regression column builder, LightGBM
matrix, isotonic fit / apply / cross-fitting, log loss) and its saved files in outputs/models; nothing is refitted
except the two isotonic adjustments. Only validation rows are read into memory; test loans are never read.

Writes
  logs/check_beyond_range.txt            the report (tables only, 1-10 masked)
  logs/check_beyond_range_progress.log   timestamps and the final SUCCESS / FAILURE line
  outputs/models/lr_edge_rule.json       training min and max per number variable, both versions' isotonic steps and
                                         log losses, the result (A or B) and the winner's isotonic steps ("calibration"),
                                         which score_test.py, credit_box.py and check_models.py use

Usage (from the repository root, after fit_models.py has finished):
    python code/check_beyond_range.py [--smoke] [--threads 8] [--memory-limit 2GB]
  (fit_models.py's options are reused, so --smoke reads outputs/models_smoke and writes the *_smoke report.)
Exit codes: 0 SUCCESS; 1 FAILURE (missing saved files, failed check); 130 interrupted (rerun).
"""

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset as bd  # noqa: E402  (reused: check lines, report sections, error detail)
import check_raw  # noqa: E402
import explore as ex  # noqa: E402  (reused: dataset connection, masking helpers)
import fit_models as fm  # noqa: E402  (reused: views, readers, column builder, isotonic, saved files)
import inventory as inv  # noqa: E402  (reused: progress log, report file, 1-10 masking)
from model_core import hold_at_edges, lr_inputs, lr_risk  # noqa: E402,F401
from check_raw import out, out_table, section  # noqa: E402
from inventory import StopRun, m, mc, progress  # noqa: E402
from load_raw import LOG_DIR, PROJECT_ROOT, hms  # noqa: E402

SCRIPT_VERSION = "2"                    # 2: isotonic adjustment with 5-fold cross-fitting; 17 model variables
DECIMALS = 6
RULE = [
    "Rule: on VALIDATION loans, compare",
    "(A) Logistic Regression as is (current saved model, inputs as stored), and",
    "(B) the same saved model, unchanged (no refit), but with every number variable's value held at its training min or "
    "max when it falls below or above it (all 7 number variables, both ends; the training mins and maxes are those in "
    "fit_models output 5).",
    "Each version is adjusted by isotonic calibration with 5-fold cross-fitting (each validation loan is "
    "adjusted by a fit on the other four folds). The version with the lower cross-fitted validation log loss wins. If they "
    "are equal to 6 decimal places, A wins.",
    "The winner's adjustment used for scoring is its isotonic calibration fitted on all validation loans.",
]

WORK_ROOT = PROJECT_ROOT    # tests re-point these two before calling main()
LOGS = LOG_DIR
P = {}
FAILS = bd.FAILS


def configure(mode):
    """fit_models.py's paths (its saved files are read from P['models']); this script's own report and logs."""
    fm.WORK_ROOT = WORK_ROOT
    fm.LOGS = LOGS
    fm.configure(mode)
    sfx = "_smoke" if mode == "smoke" else ""
    P.clear()
    P.update(models=fm.P["out"], report=LOGS / f"check_beyond_range{sfx}.txt", progress=LOGS / "check_beyond_range_progress.log",
             error=LOGS / "check_beyond_range_error_detail.txt", spill=WORK_ROOT / f"check_beyond_range_work{sfx}.tmp")
    P["rule"] = P["models"] / "lr_edge_rule.json"
    inv.REPORT_PATH = P["report"]
    inv.PROGRESS_PATH = P["progress"]
    bd.P["error"] = P["error"]
    ex.P["spill"] = P["spill"]


# ---------------------------------------------------------------- the two versions

def decide(ll_a, ll_b):
    """B wins only if its cross-fitted log loss is lower at 6 decimal places; equal at 6 decimal places -> A."""
    return "B" if round(ll_b, DECIMALS) < round(ll_a, DECIMALS) else "A"


def load_saved():
    need = ["logistic_regression.json", "isotonic_calibration.json", "lightgbm_model.txt", "lightgbm_settings.json", "beyond_range.csv"]
    missing = [n for n in need if not (P["models"] / n).exists()]
    if missing:
        progress(f"STOP: fit_models.py's saved files are missing from {P['models']}: {', '.join(missing)}. Run fit_models.py first.")
        raise StopRun(1)
    read = lambda n: json.loads((P["models"] / n).read_text(encoding="utf-8"))   # noqa: E731
    with open(P["models"] / "beyond_range.csv", newline="", encoding="utf-8") as f:
        ranges = {r["variable"]: r for r in csv.DictReader(f)}
    return read("logistic_regression.json"), read("isotonic_calibration.json"), read("lightgbm_settings.json"), ranges


def training_ranges(con, saved_ranges):
    """Training min and max per number variable (the SQL of fit_models output 5), checked against its beyond_range.csv."""
    sel = ", ".join(f"min({c}), max({c})" for c in fm.NUM_COLS)
    mm = con.execute(f"SELECT {sel} FROM train").fetchone()
    mins = {c: float(mm[2 * j]) for j, c in enumerate(fm.NUM_COLS)}
    maxes = {c: float(mm[2 * j + 1]) for j, c in enumerate(fm.NUM_COLS)}

    def same(a, b):
        return abs(a - float(b)) <= 1e-9 * max(1.0, abs(a))
    ok = all(same(mins[c], saved_ranges[fm.LABEL[c]]["training_min"]) and same(maxes[c], saved_ranges[fm.LABEL[c]]["training_max"])
             for c in fm.NUM_COLS)
    bd.check_line("training mins and maxes equal fit_models output 5 (beyond_range.csv)", ok)
    return mins, maxes


def adjust(z, y, fold):
    """fit_models' isotonic adjustment: the steps fitted on all the loans given, the measures (incl. the
    5-fold cross-fitted log loss) and the adjusted risks from the steps fitted on all of them."""
    cal, ms, risk = fm.adjust(z, y, fold)
    return {**ms, "calibration": fm.cal_json(cal)}, risk


# ---------------------------------------------------------------- report

def pct4(x):
    return f"{100 * x:.4f}%"


def group_row(label, side, mask, y, risk_a, risk_b, risk_g):
    n = int(mask.sum())
    if 1 <= n <= 10:
        return [label, side, "1-10", "", "", "", "", ""]
    b = int(y[mask].sum())
    rate = pct4(b / n) if ex.rate_shown(n, b) else "-"
    return [label, side, f"{n:,}", mc(b), rate, pct4(risk_a[mask].mean()), pct4(risk_b[mask].mean()), pct4(risk_g[mask].mean())]


def main(argv=None):
    args = fm.parse_args(argv)
    configure("smoke" if args.smoke else "full")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    LOGS.mkdir(parents=True, exist_ok=True)
    check_raw.report.clear()
    FAILS.clear()
    started = time.time()
    progress("=" * 78)
    progress(f"check_beyond_range.py v{SCRIPT_VERSION} started. Dataset {fm.P['db']} (read-only); saved models {P['models']}.")
    con, code = None, 0
    try:
        if not fm.P["db"].exists():
            progress(f"STOP: no dataset at {fm.P['db']}.")
            raise StopRun(1)
        lr, cal_saved, gbm_settings, saved_ranges = load_saved()
        con = ex.open_dataset(args)
        fm.create_views(con)
        out(f"check_beyond_range.py v{SCRIPT_VERSION}  {datetime.now():%Y-%m-%d %H:%M}  dataset {fm.P['db'].name} (read-only)  "
            f"saved models {P['models'].relative_to(WORK_ROOT)}")
        out("Summaries only: no loan IDs, no row-level values. Counts from 1 to 10 show as '1-10'; no rate or average is shown "
            "for a group of 1-10 loans, and no bad rate when its bad or good count is 1-10. Validation loans only; test loans "
            "are never read.")
        out(f"DuckDB memory limit {args.memory_limit}, threads {args.threads}. "
            f"Isotonic adjustment, {len(fm.VARIABLES)} model variables.")
        out("Mean adjusted predicted risk (sections 3 and 4) uses each version's isotonic steps fitted on all validation loans, "
            "as scoring would; the decision uses the cross-fitted log loss.")

        section("1. The rule")
        for line in RULE:
            out("  " + line)

        cats = fm.training_categories(con)
        codes, refs, present = fm.level_codes(cats)
        cols, _ = fm.design_columns(refs, present)
        bd.check_line("the training categories give the saved model's columns", cols == lr["columns"])
        bd.check_line("the training categories give the saved LightGBM category codes",
                      all(gbm_settings["category_codes"][fm.LABEL[c]] == codes[c] for c in fm.CAT_COLS))
        mins, maxes = training_ranges(con, saved_ranges)
        if FAILS:
            raise StopRun(1)
        t0 = time.time()
        num, cat, y, more = fm.fetch_rows(con, "validation", codes, True, extra=("fold",))
        fold = more["fold"].astype(np.int8)
        progress(f"Validation loans read: {mc(len(y))} in {hms(time.time() - t0)}")

        z_a = lr_risk(lr, num, cat, codes, refs, present)
        z_b = lr_risk(lr, hold_at_edges(num, mins, maxes), cat, codes, refs, present)
        res_a, risk_a = adjust(z_a, y, fold)
        res_b, risk_b = adjust(z_b, y, fold)
        result = decide(res_a["logloss_cross_fitted"], res_b["logloss_cross_fitted"])
        saved = cal_saved["logistic_regression"]
        bd.check_line("A reproduces fit_models' saved cross-fitted validation log loss (1e-12)",
                      abs(res_a["logloss_cross_fitted"] - saved["validation"]["logloss_cross_fitted"]) < 1e-12)
        bd.check_line("A's isotonic steps equal fit_models' saved ones",
                      all(res_a["calibration"][k] == saved[k] for k in ("score_low", "score_high", "value")))
        for name, res in (("A", res_a), ("B", res_b)):
            bd.check_line(f"{name}: every isotonic step holds at least {fm.MIN_SIDE} bad and {fm.MIN_SIDE} good validation loans",
                          res["min_bad_in_a_step"] >= fm.MIN_SIDE and res["min_good_in_a_step"] >= fm.MIN_SIDE)

        booster = lgb.Booster(model_file=str(P["models"] / "lightgbm_model.txt"))
        z_g = booster.predict(fm.lgb_matrix(num, cat, codes, refs, present), raw_score=True)
        risk_g = fm.isotonic_apply(cal_saved["lightgbm"], z_g)

        section("2. A and B on all validation loans: isotonic adjustment, log loss, result")
        out_table(["version", "inputs", "isotonic steps", "log loss before", "after (in-sample)",
                   f"cross-fitted ({DECIMALS} dp)", "cross-fitted (full)"],
                  [[v, inputs, str(r["steps"]), f"{r['logloss_before']:.6f}", f"{r['logloss_after']:.6f}",
                    f"{r['logloss_cross_fitted']:.{DECIMALS}f}", repr(r["logloss_cross_fitted"])]
                   for v, inputs, r in (("A", "as stored", res_a), ("B", "held at the training edge", res_b))],
                  left={0, 1})
        out(f"  Result: {result} ({'B has the lower cross-fitted log loss at 6 decimal places' if result == 'B' else 'A: B is not lower at 6 decimal places'}).")

        section("3. Beyond-range groups in validation: loans, bad loans, actual bad rate, mean adjusted predicted risk")
        rows = []
        for j, c in enumerate(fm.NUM_COLS):
            x = num[:, j]
            for side, mask in (("below min", x < np.float32(mins[c])), ("above max", x > np.float32(maxes[c]))):
                if mask.any():
                    rows.append(group_row(fm.LABEL[c], side, mask, y, risk_a, risk_b, risk_g))
        heads = ["variable", "side", "loans", "bad loans", "actual bad rate", "mean risk A", "mean risk B", "mean risk LightGBM"]
        out_table(heads, rows or [["(none)", "-", "0", "-", "-", "-", "-", "-"]], left={0, 1})
        out("  A loan beyond the range on more than one variable appears in each of its groups.")

        section("4. All validation loans")
        out_table(heads, [group_row("all validation loans", "-", np.ones(len(y), dtype=bool), y, risk_a, risk_b, risk_g)], left={0, 1})

        rule = {"rule": " ".join(RULE), "result": result, "decimals": DECIMALS,
                "training_range": {fm.LABEL[c]: {"column": c, "training_min": mins[c], "training_max": maxes[c]} for c in fm.NUM_COLS},
                "A": {"inputs": "as stored", **res_a}, "B": {"inputs": "held at the training min / max", **res_b},
                "calibration": (res_b if result == "B" else res_a)["calibration"], "calibration_rule": fm.CAL_RULE,
                "how_to_apply": ("if result is B: hold each number variable at its training min / max before the saved Logistic "
                                 "Regression; if A: inputs as stored. Then adjust the raw log-odds with 'calibration' (the winner's "
                                 "isotonic steps, fitted on all validation loans)")}
        P["rule"].write_text(json.dumps(rule, indent=1), encoding="utf-8")
        section("Files written")
        out(f"  {P['rule'].relative_to(WORK_ROOT)}")
        section(f"SUMMARY: {len(FAILS)} problem(s)")
        for f in FAILS:
            out(f"  FAIL: {f}")
        out("  SUCCESS: every output written; every check passed." if not FAILS else "  FAILURE: see the problems above.")
        code = 1 if FAILS else 0
    except StopRun as stop:
        code = stop.code
    except KeyboardInterrupt:
        progress("Interrupted. Nothing to resume: rerun the same command.")
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
