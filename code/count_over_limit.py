"""Count loans above 100% of their conforming loan limit with high-balance = N. Summaries only.

Why: each loan's limit is taken from the YEAR OF ITS ORIGINATION QUARTER. Lenders often used the next
year's limit for loans closed after its late-November announcement, so such a loan shows as above 100% of its limit
without the high-balance flag. This count shows how many loans that affects.

What it does (loan_dataset.duckdb opened READ-ONLY; inputs only: no outcome column is read, nothing is fitted)
  One table per set (training, validation, test), one row per origination quarter and agency, in a fixed order:
    loans                      all loans of the row
    above 100%, HB = N         percent of limit above 100.00 with high-balance flag N   <- the main count
    % of loans                 that count as a share of the row's loans (not shown for a masked count)
    of those: flag Y           at the limit: the limit rounds UP when stored to the nearest $1,000 (e.g. $533,850
                               stored as $534,000 = 100.03%), so these are at-the-limit loans, not next-year loans
    of those: flag N           above the limit and not at it
    flag N, <= next year       of the flag N loans: amount (rounded to $1,000) at or below the NEXT year's limit for the
                               same units, rounded to $1,000 ("-" when the next year is not in the limits table)
    above 100%, HB = Y         for comparison: above the basic limit with the high-balance flag
  Counts from 1 to 10 show as "1-10" and carry no percentage.

It reuses
fit_models.py's model_loans view and limit table, explore.py's read-only connection and the shared report helpers.

Files: logs/count_over_limit.txt (report; --smoke: logs/count_over_limit_smoke.txt), logs/count_over_limit_progress.log,
logs/count_over_limit_error_detail.txt (stack frames only). Nothing is written under outputs/.

Usage (from the repository root):
    python code/count_over_limit.py [--smoke] [--threads 8] [--memory-limit 2GB]
Exit codes: 0 SUCCESS; 1 FAILURE.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset as bd  # noqa: E402  (reused: check lines, error detail)
import check_raw  # noqa: E402
import explore as ex  # noqa: E402  (reused: dataset connection, build info)
import fit_models as fm  # noqa: E402  (reused: model_loans view, limits, sample, options)
import inventory as inv  # noqa: E402  (reused: progress log, report file, 1-10 masking)
from check_raw import out, out_table, section  # noqa: E402
from inventory import StopRun, lit, m, progress  # noqa: E402
from load_raw import hms  # noqa: E402

SCRIPT_VERSION = "1"
SETS = [("train", "training"), ("validation", "validation"), ("test", "test")]
FAILS = bd.FAILS
COUNTS = ["loans", "over_n", "over_n_flag_y", "over_n_flag_n", "over_n_next", "over_y"]


def configure(mode):
    fm.configure(mode)                                   # the dataset path, the smoke sample, explore's connection
    sfx = "_smoke" if mode == "smoke" else ""
    fm.P["spill"] = fm.WORK_ROOT / f"count_over_limit_work{sfx}.tmp"      # its own DuckDB spill folder
    ex.P["spill"] = fm.P["spill"]
    inv.REPORT_PATH = fm.LOGS / f"count_over_limit{sfx}.txt"
    inv.PROGRESS_PATH = fm.LOGS / "count_over_limit_progress.log"
    bd.P["error"] = fm.LOGS / "count_over_limit_error_detail.txt"


def count_rows(con):
    """[(label, quarter, agency, has_next, loans, over_n, over_n_flag_y, over_n_flag_n, over_n_next, over_y)]: raw
    counts, one row per set, origination quarter and agency. Inputs only."""
    over = f"{fm.RATIO} > 100"
    return con.execute(f"""
SELECT label, origination_quarter, agency, bool_or(next_k IS NOT NULL) AS has_next, count(*) AS loans,
       count(*) FILTER ({over} AND high_balance = 'N') AS over_n,
       count(*) FILTER ({over} AND high_balance = 'N' AND {fm.FLAG} = 'Y') AS over_n_flag_y,
       count(*) FILTER ({over} AND high_balance = 'N' AND {fm.FLAG} = 'N') AS over_n_flag_n,
       count(*) FILTER ({over} AND high_balance = 'N' AND {fm.FLAG} = 'N' AND floor(({fm.DOLLARS} + 500) / 1000) <= next_k) AS over_n_next,
       count(*) FILTER ({over} AND high_balance = 'Y') AS over_y
FROM (SELECT *, {fm.limit_sql(1000, year_shift=1)} AS next_k FROM {fm.SOURCE}
      WHERE label IN ('train', 'validation', 'test'){fm.sample_sql()})
GROUP BY 1, 2, 3 ORDER BY 2, 3""").fetchall()


def shown(row, has_next):
    """One table row from raw counts [loans, over_n, flag_y, flag_n, next, over_y]: masked counts; the percentage only
    for a count that is not masked."""
    loans, over_n, flag_y, flag_n, nxt, over_y = row
    pct = "-" if loans == 0 or 1 <= over_n <= 10 or 1 <= loans <= 10 else f"{100 * over_n / loans:.3f}%"
    return [m(loans), m(over_n), pct, m(flag_y), m(flag_n), m(nxt) if has_next else "-", m(over_y)]


def rep_set(rows, label, name):
    section(f"{name.capitalize()} loans: above 100% of the conforming loan limit, by origination quarter and agency")
    mine = [r for r in rows if r[0] == label]
    table, total, any_next = [], [0] * len(COUNTS), False
    for _, quarter, agency, has_next, *counts in mine:
        table.append([quarter, agency] + shown(counts, has_next))
        total = [a + b for a, b in zip(total, counts)]
        any_next = any_next or has_next
    table.append([f"all {name}", "both"] + shown(total, any_next))
    out_table(["quarter", "agency", "loans", "above 100%, HB = N", "% of loans", "of those: flag Y", "of those: flag N",
               "flag N, <= next year", "above 100%, HB = Y"], table, left={0, 1})
    return total


def main(argv=None):
    args = fm.parse_args(argv)
    configure("smoke" if args.smoke else "full")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    fm.LOGS.mkdir(parents=True, exist_ok=True)
    check_raw.report.clear()
    FAILS.clear()
    started = time.time()
    progress("=" * 78)
    progress(f"count_over_limit.py v{SCRIPT_VERSION} started ({fm.MODE} mode). Dataset {fm.P['db']} (read-only).")
    con, code = None, 0
    try:
        if not fm.P["db"].exists():
            progress(f"STOP: no dataset at {fm.P['db']}; run build_dataset.py first.")
            raise StopRun(1)
        con = ex.open_dataset(args)
        fm.create_model_view(con)
        meta = ex.read_meta(con)
        out(f"count_over_limit.py v{SCRIPT_VERSION}  {datetime.now():%Y-%m-%d %H:%M}  mode {fm.MODE}  dataset {fm.P['db'].name} (read-only)")
        out("Summaries only: no loan IDs, no row-level values. Counts from 1 to 10 show as '1-10'; no percentage is shown "
            "for a masked count. Inputs only: no outcome is read; nothing is fitted or scored.")
        out(f"Dataset built by build_dataset.py v{meta.get('script_version', '?')} at {meta.get('built_at', '?')} "
            f"(mode {meta.get('mode', '?')}).")
        out("Limit: FHFA basic conforming loan limit for the loan's number of units and the YEAR OF ITS "
            "ORIGINATION QUARTER. Percent of limit = 100 x loan amount / limit, 2 decimals. Flag Y = amount and limit equal "
            "after rounding each to the nearest $1,000. HB = the dataset's high-balance flag.")
        out("'flag N, <= next year': of the loans above their limit and not at it, those at or below the NEXT year's limit "
            "for the same units (both rounded to $1,000). 2014 and 2015 share 2016's limits. '-' = the next year is not in "
            f"the limits table (years in the table: {', '.join(str(y) for y in fm.LIMITS)}).")
        if fm.MODE == "smoke":
            out(f"SMOKE MODE: a fixed 1-in-{fm.SMOKE_MOD} hash sample of the loans; counts are not the full populations.")
        rows = count_rows(con)
        totals = {label: rep_set(rows, label, name) for label, name in SETS}
        section("Checks")
        for label, name in SETS:
            n = con.execute(f"SELECT count(*) FROM kept_loans WHERE label = {lit(label)}{fm.sample_sql()}").fetchone()[0]
            bd.check_line(f"{name}: the rows add up to every {name} loan", totals[label][0] == n)
            bd.check_line(f"{name}: flag Y + flag N = above 100% with HB = N", totals[label][2] + totals[label][3] == totals[label][1])
        blank = con.execute(f"SELECT count(*) FILTER ({fm.RATIO} IS NULL), count(*) FILTER (num_units IS NOT NULL AND {fm.DOLLARS} IS NOT NULL "
                            f"AND {fm.RATIO} IS NULL), count(*) FILTER (high_balance IS NULL) FROM {fm.SOURCE} "
                            f"WHERE label IN ('train', 'validation', 'test'){fm.sample_sql()}").fetchone()
        out(f"  Loans with no percent of limit (number of units blank; in no 'above 100%' column): {m(blank[0])}. "
            f"Loans with a blank high-balance flag (in no 'above 100%' column): {m(blank[2])}.")
        bd.check_line("every loan with a number of units has a conforming loan limit", blank[1] == 0)
        section(f"SUMMARY: {len(FAILS)} problem(s)")
        for f in FAILS:
            out(f"  FAIL: {f}")
        out("  SUCCESS: every check passed." if not FAILS else "  FAILURE: see the problems above.")
        out(f"(count_over_limit took {hms(time.time() - started)})")
        code = 1 if FAILS else 0
    except StopRun as stop:
        code = stop.code
    except KeyboardInterrupt:
        progress("Interrupted.")
        code = 130
    except Exception as exc:
        bd.write_error_detail(exc, progress)
        code = 1
    finally:
        if con is not None:
            con.close()
        if fm.P["spill"].exists() and not any(fm.P["spill"].iterdir()):
            fm.P["spill"].rmdir()
        if check_raw.report:
            inv.save_report()
            progress(f"Report saved to {inv.REPORT_PATH}")
        progress(f"{'SUCCESS' if code == 0 else 'FAILURE'}: exit code {code}, elapsed {hms(time.time() - started)}.")
    return code


if __name__ == "__main__":
    sys.exit(main())
