"""Build the loan dataset: every loan placed by the build rules, with its model variables in common values.

What it produces
  loan_dataset.duckdb (new; the raw database is only read)
    kept_loans        one row per kept loan: agency, loan_id, origination quarter, label, outcome, decision,
                      bad_rules_all, trailing_run_case and the 17 model variables in their common values
                      (missing stays NULL)
    kept_loans_audit  same keys: original_interest_rate, rate_median, rate_product, first_payment_month
                      (never model variables; kept apart from kept_loans on purpose)
    loan_outcomes     one row per loan in both agencies: place (dropped / good / bad), decision (the ID of the
                      first rule that applies, in the order of evaluation) and one flag per rule that applies
    rate_medians      the rate median table (first payment month x product)
    build_meta        provenance: script version, raw-load fingerprint, SHA-256 of the specification file, counts
  logs/build_report.txt   counts only, 1-10 masked: every required report item,
                          final counts per decision / agency / label, check lines, SUCCESS or FAILURE
  logs/build_progress.log timestamps, pieces done / total, ETA, final SUCCESS / FAILURE / INTERRUPTED line
  build_work.duckdb       intermediate per-loan tables (loan IDs inside; never share); deleted by --clean

How it runs
  * freddie_fannie_raw.duckdb is opened READ-ONLY (load_raw.connect); the working database is attached.
  * Stage 1 works one loaded source file at a time (inventory.py's piece scheme): an "ll" piece resolves one
    value per loan-level field (inventory.run_ll: rule 39's agreed value / CONFLICTING / blank on some
    rows) and a "path" piece makes one sorted pass over the file's monthly rows and keeps one row per loan
    (the 48-character status path of payments 1-24, relief flags, ending, dates). Each piece is one
    transaction with a done marker, so a rerun continues where it stopped.
  * Stage 2 (final, each step marked done when finished): values (not-available codes, blanks and impossible values -> missing; common category
    values; the credit score used) and value flags (1, 2, 3, 9, 10, 11, 39, 42 R1), the rate medians and spreads, the monthly path (SQL fast path for
    trivial windows, loan_path.py for the rest), the decision (order of evaluation) generated from loan_path.py's
    order constants, consistency checks, the output tables and the report.
  * The build STOPS (exit 1) if any loan fits no rule, any total does not add up, a value or code has no
    rule, or the data holds a structure the rules do not cover (duplicate months, a termination
    row that is not the last row, ...).
  * A lock file (build_work.lock) stops a second build; a stale lock from a dead process is reported with
    the exact command to clear it. Free disk is checked before starting and before every piece.

Privacy: no raw rows are read into Python, printed or logged; every report and log line is a count, a share
or a range. Counts from 1 to 10 show as "1-10". Seller and servicer names never appear.

Usage (from the repository root):
    python code/build_dataset.py                       full build (both agencies)
    python code/build_dataset.py --smoke               one source file per agency, separate *_smoke files
    python code/build_dataset.py --clean [--smoke]     delete the working and output files of that mode
    options: --smoke-units FREDDIE_Q FANNIE_Q, --agency freddie|fannie, --compare-paths, --full-path,
             --report-only, --memory-limit 5GB, --threads 6, --min-free-gb 55
Exit codes: 0 SUCCESS; 1 FAILURE (stop rule, totals, error); 2 working files from another load or script
version (run --clean); 4 not enough free disk; 5 lock held or stale; 130 interrupted (rerun to resume).
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_raw  # noqa: E402
import inventory as inv  # noqa: E402  (reused: paths are re-pointed in configure())
import load_raw  # noqa: E402
import loan_path as lp  # noqa: E402
from model_core import PRODUCT_SQL  # noqa: E402  (the loan-term product groups)
from check_raw import out, out_table, section  # noqa: E402
from inventory import (  # noqa: E402
    BAP_START, PERF_EXPR, SPILL_CAP_GB, StopRun, check_disk, lit, m, mc, month_idx, pct, progress, unit_q,
    unit_qend, ym,
)
from load_raw import LOG_DIR, PROJECT_ROOT, hms, read_layouts  # noqa: E402

SCRIPT_VERSION = "1"
EXPECTED_LOANS = {"fre": 20_324_566, "fan": 26_344_089}     # checked in full mode
AGENCY_NAME = {"fre": "Freddie Mac", "fan": "Fannie Mae"}
DEFAULT_SMOKE = {"fre": "2019Q1", "fan": "2019Q1"}
PY_BATCH = 200_000          # loans per Python batch (about 100 MB of Python objects)
# Working database (per-loan values, paths and the wide *_decided tables for 46.7 million loans) ~20 GB, output
# ~3 GB, DuckDB spill up to the 30 GB cap: 55 GB free keeps the drive from filling up.
DEFAULT_MIN_FREE_GB = 55
NEED_GB_TEXT = "working database ~20 GB, output database ~3 GB, DuckDB spill up to %d GB" % SPILL_CAP_GB
DECISION_LOG = PROJECT_ROOT / "docs" / "decision_log.md"

WORK_ROOT = PROJECT_ROOT    # tests re-point these two before calling main()
LOGS = LOG_DIR
MODE = "full"
P = {}                      # paths of this mode: work, spill, out, lock, report, progress, error
FAILS = []                  # reasons the build is a FAILURE
_AFTER_PIECE_HOOK = None    # tests only

# scope (rule 1) and train / validation / test labels: quarter ordinal = year * 4 + quarter
SCOPE = [((2014, 1), (2020, 1), "train"), ((2022, 2), (2022, 4), "validation"), ((2023, 1), (2023, 4), "test")]
OUT_OF_SCOPE = "(out of scope)"
UNKNOWN_QUARTER = "(unknown quarter)"
LABELS = ["train", "validation", "test", OUT_OF_SCOPE, UNKNOWN_QUARTER]
BANDS_LA = [("1", 1, 1), ("2", 2, 2), ("3-6", 3, 6), ("7-12", 7, 12), (">12", 13, None)]


def configure(mode):
    """Working files of this mode; inventory.py's module paths are re-pointed so its functions write here."""
    global MODE
    MODE = mode
    sfx = "_smoke" if mode == "smoke" else ""
    P.clear()
    P.update(work=WORK_ROOT / f"build_work{sfx}.duckdb", spill=WORK_ROOT / f"build_work{sfx}.duckdb.tmp",
             out=WORK_ROOT / f"loan_dataset{sfx}.duckdb", lock=WORK_ROOT / f"build_work{sfx}.lock",
             report=LOGS / f"build_report{sfx}.txt", progress=LOGS / "build_progress.log",
             error=LOGS / "build_error_detail.txt")
    inv.WORK_DB = P["work"]
    inv.SPILL_DIR = P["spill"]
    inv.REPORT_PATH = P["report"]
    inv.PROGRESS_PATH = P["progress"]
    inv.ERROR_DETAIL_PATH = P["error"]


# ---------------------------------------------------------------- lock file and errors

def pid_alive(pid):
    if os.name == "nt":
        import ctypes
        kernel = ctypes.windll.kernel32
        handle = kernel.OpenProcess(0x1000, False, int(pid))   # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return kernel.GetLastError() == 5                  # access denied: the process exists
        code = ctypes.c_ulong()
        ok = kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel.CloseHandle(handle)
        return bool(ok) and code.value == 259                  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def acquire_lock():
    path = P["lock"]
    if path.exists():
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            info = {}
        pid = info.get("pid")
        if pid is not None and int(pid) != os.getpid() and pid_alive(pid):
            progress(f"STOP: another build (PID {pid}, started {info.get('started', '?')}) holds {path}. "
                     f"Wait for it to finish or stop it. If no build is running under that PID, clear the lock "
                     f"and rerun:  Remove-Item \"{path}\"")
            raise StopRun(5)
        progress(f"STOP: stale lock file from PID {pid} (no longer running) at {path}. The previous build died "
                 f"without cleaning up; its finished pieces are kept. Clear the lock and rerun:  "
                 f"Remove-Item \"{path}\"")
        raise StopRun(5)
    path.write_text(json.dumps({"pid": os.getpid(), "started": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
                                "host": platform.node(), "mode": MODE}), encoding="utf-8")


def release_lock():
    try:
        P["lock"].unlink()
    except OSError:
        pass


def write_error_detail(exc, emit):
    """Python stack frames only; DuckDB messages (which can quote values) are never written anywhere."""
    frames = "".join(traceback.format_tb(exc.__traceback__))
    shown = type(exc).__name__ if isinstance(exc, duckdb.Error) else f"{type(exc).__name__}: {exc}"
    try:
        P["error"].write_text(f"build_dataset.py error {datetime.now():%Y-%m-%d %H:%M:%S}\n"
                              f"Stack frames (code locations only, no data):\n{frames}{shown}\n", encoding="utf-8")
    except OSError:
        pass
    emit(f"ERROR: {shown}")
    if isinstance(exc, duckdb.IOException):
        emit("       (an IOException usually means another program holds a database file open: close any "
             "DuckDB shell or notebook using build_work*.duckdb or loan_dataset*.duckdb, then rerun)")
    emit(f"       Stack frames in {P['error']} (DuckDB messages are never written there).")


# ---------------------------------------------------------------- small helpers

def qord(y, n):
    return y * 4 + n


def scope_sql(q):
    """SQL: the quarter ordinal q is one of the quarters in scope."""
    return "(" + " OR ".join(f"({q} BETWEEN {qord(*a)} AND {qord(*b)})" for a, b, _ in SCOPE) + ")"


def label_sql(q):
    whens = " ".join(f"WHEN {q} BETWEEN {qord(*a)} AND {qord(*b)} THEN {lit(lab)}" for a, b, lab in SCOPE)
    return f"CASE WHEN {q} IS NULL THEN {lit(UNKNOWN_QUARTER)} {whens} ELSE {lit(OUT_OF_SCOPE)} END"


def ym_sql(idx):
    return f"(CAST({idx} // 12 AS VARCHAR) || '-' || lpad(CAST({idx} % 12 + 1 AS VARCHAR), 2, '0'))"


def sha256_file(path):
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def code_map(col, mapping, na=()):
    """CASE for a coded field: mapping raw -> common value; na codes -> NULL.
    Returns (value_sql, unmapped_sql)."""
    whens = " ".join(f"WHEN {col} = {lit(k)} THEN {lit(v)}" for k, v in mapping.items())
    known = ", ".join(lit(k) for k in list(mapping) + list(na))
    return f"(CASE WHEN {col} IS NULL THEN NULL {whens} END)", f"({col} IS NOT NULL AND {col} NOT IN ({known}))"


# ---------------------------------------------------------------- working tables

PATH_COLS = [
    ("unit", "VARCHAR"), ("loan_id", "VARCHAR"), ("fpd_idx", "INTEGER"), ("fpd_state", "VARCHAR"),
    ("n_rows", "BIGINT"), ("n_per_bad", "BIGINT"), ("n_dup", "BIGINT"), ("n_sn_bad", "BIGINT"),
    ("n_unparseable_w", "BIGINT"), ("first_pay", "INTEGER"), ("last_pay", "INTEGER"), ("w_rows", "BIGINT"),
    ("rows_before_fpd", "BIGINT"), ("zb_pay", "INTEGER"), ("zb_code", "VARCHAR"), ("n_zb_rows", "BIGINT"),
    ("n_zb_codes", "BIGINT"), ("zb_rn", "BIGINT"), ("n_all", "BIGINT"), ("zb_date_idx", "INTEGER"),
    ("w_max_sn", "BIGINT"), ("w_sn_gt_pay", "BOOLEAN"), ("w_jump_any", "BOOLEAN"),
    ("fc_min", "INTEGER"), ("fc_max", "INTEGER"), ("fc_bad", "BOOLEAN"), ("hltv_any_y", "BOOLEAN"),
    ("la_maxdiff", "DOUBLE"), ("la_missing_w", "BOOLEAN"),
    ("first_per_f", "INTEGER"), ("first_per_r", "INTEGER"), ("first_per_t", "INTEGER"), ("first_per_d", "INTEGER"),
    ("bapblank_pre_other", "BIGINT"), ("bapblank_post_other", "BIGINT"), ("bapblank_pre_term", "BIGINT"),
    ("bapblank_post_term", "BIGINT"),
    ("fb_null_n", "BIGINT"), ("fb_null_first", "BIGINT"), ("fb_null_last", "BIGINT"),
    ("fcb_null_n", "BIGINT"), ("fcb_null_first", "BIGINT"), ("fcb_null_last", "BIGINT"),
    ("status48", "VARCHAR"), ("relief24", "VARCHAR"),
]
PATHRES_TYPES = {"proven_first": "INTEGER", "jump3_any": "BOOLEAN", "jump_any": "BOOLEAN", "relief_any": "BOOLEAN",
                 "ra_first": "INTEGER", "sn_gt_pay": "BOOLEAN", "run_fail_15": "BOOLEAN", "run_fail_16": "BOOLEAN",
                 "trailing_case": "BOOLEAN", "trailing_pass": "BOOLEAN", "freeze_applied": "BOOLEAN",
                 "freeze_gap": "BOOLEAN"}
assert list(PATHRES_TYPES) == lp.PATH_COLUMNS


def create_build_tables(con):
    for ag in inv.AGENCIES:
        con.execute(f"CREATE TABLE IF NOT EXISTS w.{ag}_path ({', '.join(f'{n} {t}' for n, t in PATH_COLS)})")
        cols = ", ".join(f"{c} {t}" for c, t in PATHRES_TYPES.items())
        for suffix in ("", "_check"):
            con.execute(f"CREATE TABLE IF NOT EXISTS w.{ag}_pathres{suffix} (loan_id VARCHAR, {cols}, src VARCHAR)")
    con.execute("CREATE TABLE IF NOT EXISTS w.checks (name VARCHAR, agency VARCHAR, value BIGINT, must_be_zero BOOLEAN)")


def check_build_meta(con, args):
    for key, value, what in (("build_version", SCRIPT_VERSION, "an older version of build_dataset.py"),
                             ("build_mode", MODE, "a different mode (full / smoke)"),
                             ("build_full_path", str(bool(args.full_path)), "a run with a different --full-path setting")):
        old = inv.meta_get(con, key)
        if old is None:
            inv.meta_set(con, key, value)
        elif old != value:
            progress(f"STOP: {P['work'].name} was built by {what}. Run with --clean{' --smoke' if MODE == 'smoke' else ''}, "
                     f"then rerun.")
            raise StopRun(2)


# ---------------------------------------------------------------- stage 1: one source file at a time

PATH_EXTRA = {
    "fre": dict(zb_date=month_idx("r.zero_balance_effective_date", "YYYYMM"),
                fb_null="NULL::BOOLEAN", fcb_null="NULL::BOOLEAN"),
    "fan": dict(zb_date="NULL::INTEGER",
                fb_null="r.borrower_credit_score_at_origination IS NULL",
                fcb_null="r.co_borrower_credit_score_at_origination IS NULL"),
}


def path_sql(ag, table, de):
    """Per-loan facts of one file's monthly rows: one sorted pass (window functions) and one GROUP BY, then the
    48-character status path and 24-character relief string (see loan_path.py for the tokens)."""
    X, E = PERF_EXPR[ag], PATH_EXTRA[ag]
    toks = ",\n    ".join(f"any_value(tok) FILTER (pay = {k}) AS t{k}" for k in range(1, 25))
    rels = ",\n    ".join(f"coalesce(bool_or(relief) FILTER (pay = {k}), false) AS r{k}" for k in range(1, 25))
    status = ",\n    ".join(
        f"CASE WHEN zb_pay IS NOT NULL AND {k} > zb_pay THEN 'TT' WHEN fpd_idx + {k - 1} > {de} THEN '__' "
        f"WHEN t{k} IS NULL THEN '..' WHEN zb_pay = {k} AND t{k} IN ('XX', '??') THEN 'TX' ELSE t{k} END"
        for k in range(1, 25))
    relief = ", ".join(f"CASE WHEN r{k} THEN 'F' ELSE '.' END" for k in range(1, 25))
    drop = ", ".join([f"t{k}" for k in range(1, 25)] + [f"r{k}" for k in range(1, 25)])
    return f"""
INSERT INTO w.{ag}_path BY NAME
WITH base AS (
  SELECT r.loan_identifier AS loan_id, u.fpd_idx, u.fpd_state,
         {X['per']} AS per, {X['st']} AS st, {X['zb']} AS zb, TRY_CAST({X['la']} AS DOUBLE) AS la,
         coalesce({X['F']}, false) AS ff, coalesce({X['D']}, false) AS fd,
         coalesce({X['R']}, false) AS fr, coalesce({X['T']}, false) AS ft,
         {X['bapnull']} AS bapnull, {X['fc']} AS fc, coalesce({X['fc_bad']}, false) AS fc_bad, {X['hltv']} AS hltv,
         {E['zb_date']} AS zb_date, {E['fb_null']} AS fb_null, {E['fcb_null']} AS fcb_null
  FROM {table} r JOIN _uloan u ON u.loan_id = r.loan_identifier
  WHERE r.source_file = $u
), cls AS (
  SELECT *, per - fpd_idx + 1 AS pay,
         CASE WHEN regexp_full_match(st, '[0-9]+') THEN TRY_CAST(st AS BIGINT) END AS sn0,
         row_number() OVER (PARTITION BY loan_id ORDER BY per NULLS LAST) AS rn,
         count(*) OVER (PARTITION BY loan_id) AS n_all
  FROM base
), seq AS (
  SELECT *, CASE WHEN sn0 BETWEEN 0 AND 99 THEN sn0 END AS sn,
         coalesce(pay BETWEEN 1 AND 24, false) AS win,
         CASE WHEN sn0 BETWEEN 0 AND 99 THEN lpad(CAST(sn0 AS VARCHAR), 2, '0')
              WHEN st = 'RA' THEN 'RA' WHEN st IS NULL OR st = 'XX' THEN 'XX' ELSE '??' END AS tok,
         (ff OR fd OR fr OR ft) AS relief,
         lag(per) OVER lw AS prev_per,
         lag(CASE WHEN sn0 BETWEEN 0 AND 99 AND pay >= 1 THEN {{'sn': sn0, 'pay': pay}} END IGNORE NULLS) OVER lw AS pn
  FROM cls
  WINDOW lw AS (PARTITION BY loan_id ORDER BY per NULLS LAST)
), agg AS (
  SELECT loan_id, any_value(fpd_idx) AS fpd_idx, any_value(fpd_state) AS fpd_state,
    count(*) AS n_rows, count(*) FILTER (per IS NULL) AS n_per_bad, count(*) FILTER (per = prev_per) AS n_dup,
    count(*) FILTER (sn0 IS NOT NULL AND sn0 NOT BETWEEN 0 AND 99) AS n_sn_bad,
    count(*) FILTER (win AND tok = '??') AS n_unparseable_w,
    min(pay) AS first_pay, max(pay) AS last_pay, count(*) FILTER (win) AS w_rows,
    count(*) FILTER (pay <= 0) AS rows_before_fpd,
    min(pay) FILTER (zb IS NOT NULL) AS zb_pay, arg_min(zb, per) FILTER (zb IS NOT NULL) AS zb_code,
    count(*) FILTER (zb IS NOT NULL) AS n_zb_rows, count(DISTINCT zb) AS n_zb_codes,
    max(rn) FILTER (zb IS NOT NULL) AS zb_rn, any_value(n_all) AS n_all,
    arg_min(zb_date, per) FILTER (zb IS NOT NULL) AS zb_date_idx,
    max(sn) FILTER (win) AS w_max_sn,
    coalesce(bool_or(win AND sn > pay), false) AS w_sn_gt_pay,
    coalesce(bool_or(win AND sn IS NOT NULL AND sn > coalesce(pn.sn, 0) + (pay - coalesce(pn.pay, 0))), false) AS w_jump_any,
    {toks},
    {rels},
    min(fc) AS fc_min, max(fc) AS fc_max, coalesce(bool_or(fc_bad), false) AS fc_bad,
    coalesce(bool_or(hltv = 'Y'), false) AS hltv_any_y,
    max(abs(la - pay)) FILTER (win) AS la_maxdiff, coalesce(bool_or(la IS NULL) FILTER (win), false) AS la_missing_w,
    min(per) FILTER (ff) AS first_per_f, min(per) FILTER (fr) AS first_per_r,
    min(per) FILTER (ft) AS first_per_t, min(per) FILTER (fd) AS first_per_d,
    count(*) FILTER (win AND bapnull AND per < {BAP_START} AND zb IS NULL) AS bapblank_pre_other,
    count(*) FILTER (win AND bapnull AND per >= {BAP_START} AND zb IS NULL) AS bapblank_post_other,
    count(*) FILTER (win AND bapnull AND per < {BAP_START} AND zb IS NOT NULL) AS bapblank_pre_term,
    count(*) FILTER (win AND bapnull AND per >= {BAP_START} AND zb IS NOT NULL) AS bapblank_post_term,
    count(*) FILTER (fb_null) AS fb_null_n, min(rn) FILTER (fb_null) AS fb_null_first, max(rn) FILTER (fb_null) AS fb_null_last,
    count(*) FILTER (fcb_null) AS fcb_null_n, min(rn) FILTER (fcb_null) AS fcb_null_first, max(rn) FILTER (fcb_null) AS fcb_null_last
  FROM seq GROUP BY loan_id
)
SELECT $u AS unit, * EXCLUDE ({drop}),
       concat({status}) AS status48,
       concat({relief}) AS relief24
FROM agg
"""


def run_path_piece(con, T, unit, de):
    started = time.time()
    rows = con.execute(f"SELECT count(*) FROM {T.table} WHERE source_file = ? AND loan_identifier IS NOT NULL",
                       [unit]).fetchone()[0]

    def write():
        con.execute(f"DELETE FROM w.{T.ag}_path WHERE unit = ?", [unit])
        con.execute(path_sql(T.ag, T.table, de), {"u": unit})

    inv.commit_piece(con, "path", T.table, unit, rows, started, write)
    return rows


def select_units(units, args):
    """Loaded files to process: all, or one per agency in smoke mode."""
    if not args.smoke:
        return units
    want = {"fre": args.smoke_units[0], "fan": args.smoke_units[1]} if args.smoke_units else dict(DEFAULT_SMOKE)
    chosen = [(t, u, r) for t, u, r in units if u == want["fre" if t.startswith("freddie") else "fan"]]
    for ag, tables in (("fre", ("freddie_orig", "freddie_perf")), ("fan", ("fannie_raw",))):
        have = {t for t, u, r in chosen if t in tables}
        if have != set(tables):
            progress(f"STOP: smoke unit {want[ag]} is not a loaded {AGENCY_NAME[ag]} file (load_log).")
            raise StopRun(1)
    return chosen


def plan_pieces(units, agencies):
    by = {}
    for table, unit, rows in units:
        by.setdefault(table, []).append((unit, rows or 0))
    plan = []
    if "fre" in agencies:
        plan += [("ll", "freddie_orig", u, r) for u, r in by.get("freddie_orig", [])]
        plan += [("path", "freddie_perf", u, r) for u, r in by.get("freddie_perf", [])]
    if "fan" in agencies:
        for u, r in by.get("fannie_raw", []):
            plan += [("ll", "fannie_raw", u, r), ("path", "fannie_raw", u, r)]
    return plan


def process_pieces(args, tabs, plan, des):
    """Run every unfinished piece, each on a fresh connection (the inventory found one long-lived connection
    slowed down about 4x after large spilling queries). Any error stops the build."""
    holder = [inv.open_db(args)]
    try:
        _process_pieces(holder, args, tabs, plan, des)
    finally:
        holder[0].close()


def _process_pieces(holder, args, tabs, plan, des):
    con = holder[0]
    done = set(con.execute("SELECT piece, tbl, unit FROM w.inv_done").fetchall())
    todo = [p for p in plan if (p[0], p[1], p[2]) not in done]
    progress(f"Stage 1 (file pieces): {len(plan) - len(todo)} of {len(plan)} already finished; {len(todo)} to run.")
    if not todo:
        return
    history = con.execute("SELECT piece, tbl, sum(rows), sum(seconds) FROM w.inv_done WHERE piece IN ('ll', 'path') "
                          "GROUP BY ALL").fetchall()
    rate = {(p, t): r / s for p, t, r, s in history if s and r}
    fre_ll = [(p, t, u) for p, t, u, r in plan if p == "ll" and t == "freddie_orig"]
    run_start = time.time()
    for n, (piece, table, unit, rows) in enumerate(todo, start=1):
        T = tabs[table]
        con.close()
        holder[0] = con = inv.open_db(args)
        if piece == "path":
            needs = fre_ll if T.ag == "fre" else [("ll", "fannie_raw", unit)]
            done = set(con.execute("SELECT piece, tbl, unit FROM w.inv_done").fetchall())
            missing = [k for k in needs if k not in done]
            if missing:
                progress(f"STOP: {table} {unit} path needs {len(missing)} unfinished loan-level piece(s).")
                raise StopRun(1)
        check_disk(args.min_free_gb, f"before {table} {unit} {piece}")
        t0 = time.time()
        if piece == "ll":
            inv.run_ll(con, T, unit)
        else:
            inv.build_uloan(con, T, unit)
            run_path_piece(con, T, unit, des[T.ag])
        secs = time.time() - t0
        if rows and secs:
            old = rate.get((piece, table))
            rate[(piece, table)] = rows / secs if old is None else 0.5 * old + 0.5 * rows / secs
        remaining, unknown = 0.0, False
        for p, t, u, r in todo[n:]:
            rt = rate.get((p, t))
            if rt:
                remaining += (r or 0) / rt
            else:
                unknown = True
        eta = (f"remaining ~{hms(remaining)}" + (" + pieces not yet timed" if unknown else "")
               + f" (finish ~{datetime.fromtimestamp(time.time() + remaining):%H:%M})")
        progress(f"[{n}/{len(todo)}] {table} {unit} {piece}: {mc(rows)} rows in {hms(secs)} | "
                 f"elapsed {hms(time.time() - run_start)} | {eta}")
        if _AFTER_PIECE_HOOK:
            _AFTER_PIECE_HOOK(n)


# ---------------------------------------------------------------- stage 2: values and value flags

# Rule 39: the loan-level Fannie fields whose values the model or the outcome uses.
FANNIE_USED_FIELDS = [
    "channel", "original_interest_rate", "original_upb", "original_loan_term", "original_loan_to_value_ratio_ltv",
    "original_combined_loan_to_value_ratio_cltv", "number_of_borrowers", "debt_to_income_dti",
    "borrower_credit_score_at_origination", "co_borrower_credit_score_at_origination",
    "first_time_home_buyer_indicator", "loan_purpose", "property_type", "number_of_units", "occupancy_status",
    "mortgage_insurance_percentage", "special_eligibility_program", "high_balance_loan_indicator",
]
NUMERIC_VARS = ["fico_b", "fico_cb", "ltv", "cltv", "dti", "mi", "upb", "rate", "term", "borrowers"]
CODED_VARS = ["fthb", "occ", "chan", "purp", "prop", "units", "sep", "hb"]
VAR_LABEL = {"fico_b": "credit score (borrower)", "fico_cb": "credit score (co-borrower)", "ltv": "LTV", "cltv": "combined LTV",
             "dti": "DTI", "mi": "mortgage insurance %", "upb": "original UPB", "rate": "interest rate", "term": "loan term",
             "borrowers": "number of borrowers", "fthb": "first-time homebuyer", "occ": "occupancy", "chan": "channel",
             "purp": "loan purpose", "prop": "property type", "units": "number of units", "sep": "special program",
             "hb": "high-balance flag"}


def vals_sql(ag):
    fmt = "YYYYMM" if ag == "fre" else "MMYYYY"
    fpd = month_idx("l.first_payment_date", fmt)
    if ag == "fre":
        src = f"""
  SELECT l.loan_id, l.unit, {unit_q('l.unit')} AS q_ord, l.unit AS orig_q,
         {fpd} AS fpd_idx, {fpd} - {unit_qend('l.unit')} AS months_to_fpd,
         (regexp_full_match(l.loan_id, '.[0-9]{{2}}Q[1-4].*')
          AND '20' || substr(l.loan_id, 2, 2) || substr(l.loan_id, 4, 2) <> l.unit) AS f_10,
         false AS f_9, false AS f_39, false AS rate_conf, (l.harp_indicator = 'Y') AS f_2,
         l.classic_fico AS fico_b_txt, NULL::VARCHAR AS fico_cb_txt, NULL::VARCHAR AS fico_classic_txt,
         l.original_loan_to_value_ltv AS ltv_txt, l.original_combined_loan_to_value_cltv AS cltv_txt,
         l.original_debt_to_income_dti_ratio AS dti_txt, l.mortgage_insurance_percentage_mi AS mi_txt, false AS mi_blank_all,
         l.original_upb AS upb_txt, l.original_interest_rate AS rate_txt, l.original_loan_term AS term_txt,
         l.number_of_borrowers AS borrowers_txt,
         l.first_time_homebuyer_indicator AS fthb_c, l.occupancy_status AS occ_c, l.channel AS chan_c,
         l.loan_purpose AS purp_c, l.property_type AS prop_c, l.number_of_units AS units_c,
         l.special_eligibility_program AS sep_c, l.super_conforming_flag AS hb_c,
         l.prepayment_penalty_indicator AS ppm_c, l.interest_only_i_o_indicator AS io_c
  FROM w.fre_lr l"""
    else:
        od = month_idx("l.origination_date", fmt)
        conf = " OR ".join(f"l.{c}__conf" for c in FANNIE_USED_FIELDS)
        src = f"""
  SELECT l.loan_id, l.unit,
         CASE WHEN {od} IS NOT NULL THEN ({od}) // 12 * 4 + ({od}) % 12 // 3 + 1 END AS q_ord,
         CASE WHEN {od} IS NOT NULL THEN CAST(({od}) // 12 AS VARCHAR) || 'Q' || CAST(({od}) % 12 // 3 + 1 AS VARCHAR) END AS orig_q,
         {fpd} AS fpd_idx, {fpd} - (({od}) // 3 * 3 + 2) AS months_to_fpd,
         false AS f_10, (l.origination_date__conf OR l.first_payment_date__conf) AS f_9,
         ({conf}) AS f_39, l.original_interest_rate__conf AS rate_conf, coalesce(p.hltv_any_y, false) AS f_2,
         l.borrower_credit_score_at_origination AS fico_b_txt, l.co_borrower_credit_score_at_origination AS fico_cb_txt,
         l.origination_classic_fico AS fico_classic_txt,
         l.original_loan_to_value_ratio_ltv AS ltv_txt, l.original_combined_loan_to_value_ratio_cltv AS cltv_txt,
         l.debt_to_income_dti AS dti_txt, l.mortgage_insurance_percentage AS mi_txt,
         (l.mortgage_insurance_percentage IS NULL AND NOT l.mortgage_insurance_percentage__conf) AS mi_blank_all,
         l.original_upb AS upb_txt, l.original_interest_rate AS rate_txt, l.original_loan_term AS term_txt,
         l.number_of_borrowers AS borrowers_txt,
         l.first_time_home_buyer_indicator AS fthb_c, l.occupancy_status AS occ_c, l.channel AS chan_c,
         l.loan_purpose AS purp_c, l.property_type AS prop_c, l.number_of_units AS units_c,
         l.special_eligibility_program AS sep_c, l.high_balance_loan_indicator AS hb_c,
         l.prepayment_penalty_indicator AS ppm_c, l.interest_only_loan_indicator AS io_c
  FROM w.fan_lr l LEFT JOIN (SELECT loan_id, bool_or(hltv_any_y) AS hltv_any_y FROM w.fan_path GROUP BY 1) p USING (loan_id)"""

    nums = ",\n         ".join(f"TRY_CAST({v}_txt AS DOUBLE) AS {v}_raw" for v in NUMERIC_VARS + ["fico_classic"])
    if ag == "fre":   # Freddie not-available codes -> missing
        conv = """
         CASE WHEN fico_b_raw = 9999 THEN NULL ELSE fico_b_raw END AS fico_b, fico_cb_raw AS fico_cb,
         CASE WHEN ltv_raw = 999 THEN NULL ELSE ltv_raw END AS ltv0,
         CASE WHEN cltv_raw = 999 THEN NULL ELSE cltv_raw END AS cltv0,
         CASE WHEN dti_raw = 999 THEN NULL ELSE dti_raw END AS dti0,
         CASE WHEN mi_raw = 999 THEN NULL ELSE mi_raw END AS mi_pct,
         false AS f_42r2_dti,
         CASE WHEN borrowers_raw = 99 THEN NULL WHEN borrowers_raw = 1 THEN '1' WHEN borrowers_raw >= 2 THEN '2 or more' END AS num_borrowers,
         (borrowers_txt IS NOT NULL AND (borrowers_raw IS NULL OR borrowers_raw < 1 OR (borrowers_raw > 10 AND borrowers_raw <> 99))) AS bad_borrowers,
         CASE WHEN sep_c IS NULL THEN 'none' WHEN sep_c IN ('F', 'H', 'R') THEN sep_c END AS special_program,
         (sep_c IS NOT NULL AND sep_c NOT IN ('F', 'H', 'R')) AS bad_sep"""
        fthb, bad_fthb = code_map("fthb_c", {"Y": "Y", "N": "N"}, na=("9",))
        occ, bad_occ = code_map("occ_c", {"P": "primary", "S": "second home", "I": "investor"}, na=("9",))
        chan, bad_chan = code_map("chan_c", {"R": "retail", "B": "broker", "C": "correspondent"}, na=("9",))
        purp, bad_purp = code_map("purp_c", {"P": "purchase", "C": "cash-out refinance", "N": "no-cash-out refinance"}, na=("9",))
        prop, bad_prop = code_map("prop_c", {c: c for c in ("SF", "PU", "CO", "CP", "MH")}, na=("99",))
        units, bad_units = code_map("units_c", {str(i): str(i) for i in range(1, 5)}, na=("99",))
    else:             # Fannie: U / 9 codes and blanks -> missing; MI blank on every row = 0; DTI 0 -> missing
        conv = """
         fico_b_raw AS fico_b, fico_cb_raw AS fico_cb, ltv_raw AS ltv0, cltv_raw AS cltv0,
         CASE WHEN dti_raw = 0 THEN NULL ELSE dti_raw END AS dti0,
         CASE WHEN mi_blank_all THEN 0 ELSE mi_raw END AS mi_pct,
         (dti_raw = 0) AS f_42r2_dti,
         CASE WHEN borrowers_raw = 1 THEN '1' WHEN borrowers_raw >= 2 THEN '2 or more' END AS num_borrowers,
         (borrowers_txt IS NOT NULL AND (borrowers_raw IS NULL OR borrowers_raw < 1)) AS bad_borrowers,
         CASE WHEN sep_c = '7' THEN 'none' WHEN sep_c IN ('F', 'H', 'R') THEN sep_c END AS special_program,
         (sep_c IS NOT NULL AND sep_c NOT IN ('F', 'H', 'R', '7', '9')) AS bad_sep"""
        fthb, bad_fthb = code_map("fthb_c", {"Y": "Y", "N": "N"})
        occ, bad_occ = code_map("occ_c", {"P": "primary", "S": "second home", "I": "investor"}, na=("U",))
        chan, bad_chan = code_map("chan_c", {"R": "retail", "B": "broker", "C": "correspondent"})
        purp, bad_purp = code_map("purp_c", {"P": "purchase", "C": "cash-out refinance", "R": "no-cash-out refinance"}, na=("U",))
        prop, bad_prop = code_map("prop_c", {c: c for c in ("SF", "PU", "CO", "CP", "MH")})
        units, bad_units = code_map("units_c", {str(i): str(i) for i in range(1, 5)})
    hb, bad_hb = code_map("hb_c", {"Y": "Y", "N": "N"})
    bad_nums = ",\n         ".join(f"({v}_txt IS NOT NULL AND {v}_raw IS NULL) AS bad_{v}" for v in NUMERIC_VARS if v != "borrowers")
    product = PRODUCT_SQL                 # the loan-term product groups (model_core.PRODUCT_SQL)
    return f"""
CREATE OR REPLACE TABLE w.{ag}_vals AS
WITH src AS ({src}
), num AS (
  SELECT *, {nums} FROM src
), conv AS (
  SELECT *, {conv},
         CASE WHEN rate_raw > 0 THEN rate_raw END AS rate,
         (rate_raw IS NOT NULL AND rate_raw <= 0) AS f_42r2_rate,
         (dti_raw = 0) AS dti_zero_raw,
         TRY_CAST(term_raw AS INTEGER) AS loan_term, upb_raw AS original_upb,
         {fthb} AS first_time_homebuyer, {bad_fthb} AS bad_fthb,
         {occ} AS occupancy, {bad_occ} AS bad_occ,
         {chan} AS channel, {bad_chan} AS bad_chan,
         {purp} AS loan_purpose, {bad_purp} AS bad_purp,
         {prop} AS property_type, {bad_prop} AS bad_prop,
         TRY_CAST({units} AS INTEGER) AS num_units, {bad_units} AS bad_units,
         {hb} AS high_balance, {bad_hb} AS bad_hb,
         {bad_nums},
         (ppm_c IS NULL) AS ppm_blank, (io_c IS NULL) AS io_blank,
         coalesce(ppm_c, 'N') AS ppm, coalesce(io_c, 'N') AS io
  FROM num
)
SELECT *,
       TRY_CAST(CASE WHEN fico_b IS NOT NULL AND fico_cb IS NOT NULL THEN least(fico_b, fico_cb)
                     ELSE coalesce(fico_b, fico_cb) END AS INTEGER) AS credit_score,
       CASE WHEN fico_b IS NOT NULL AND fico_cb IS NOT NULL THEN 'lowest of two'
            WHEN fico_b IS NOT NULL THEN 'borrower only' WHEN fico_cb IS NOT NULL THEN 'co-borrower only'
            ELSE 'missing' END AS score_source,
       (q_ord IS NOT NULL AND NOT {scope_sql('q_ord')}) AS f_1,
       {label_sql('q_ord')} AS label,
       coalesce(loan_term > 360, false) AS f_3,
       coalesce(months_to_fpd >= 4, false) AS f_11,
       coalesce(ltv0 > 97 OR cltv0 > 105, false) AS f_42r1,
       coalesce(cltv0 < ltv0, false) AS f_42r2_cltv_lt_ltv,
       CASE WHEN cltv0 < ltv0 THEN NULL ELSE ltv0 END AS ltv,
       CASE WHEN cltv0 < ltv0 THEN NULL ELSE cltv0 END AS cltv,
       dti0 AS dti,
       {product} AS rate_product,
       (NOT coalesce(f_2, false) AND NOT coalesce(loan_term > 360, false) AND NOT f_9
        AND NOT coalesce(ltv0 > 97 OR cltv0 > 105, false) AND rate IS NOT NULL AND NOT rate_conf
        AND {product} IS NOT NULL AND fpd_idx IS NOT NULL) AS in_median_pop
FROM conv
"""


def medians_sql(agencies):
    parts = " UNION ALL ".join(f"SELECT fpd_idx, rate_product, rate FROM w.{ag}_vals WHERE in_median_pop" for ag in agencies)
    return f"""
CREATE OR REPLACE TABLE w.rate_medians AS
SELECT fpd_idx, {ym_sql('fpd_idx')} AS first_payment_month, rate_product AS product, count(*) AS n_loans,
       quantile_cont(rate, 0.5) AS median_rate, count(*) >= 30 AS used
FROM ({parts}) GROUP BY fpd_idx, rate_product
"""


# ---------------------------------------------------------------- stage 2: the monthly path

FAST_SQL = ("fpd_state = 'ok' AND relief24 = '" + lp.NO_RELIEF + "' AND NOT regexp_matches(status48, 'RA|XX|\\?\\?|\\.\\.') "
            "AND coalesce(w_max_sn, 0) <= 2")


def pathsel_sql(ag, de):
    return f"""
CREATE OR REPLACE TABLE w.{ag}_pathsel AS
SELECT loan_id, status48, relief24, zb_pay, last_pay, {de} - fpd_idx + 1 AS de_pay, w_jump_any, w_sn_gt_pay,
       ({FAST_SQL}) AS fast
FROM w.{ag}_path WHERE fpd_state = 'ok'
"""


def fast_insert_sql(ag):
    values = {"proven_first": "NULL", "jump3_any": "false", "jump_any": "w_jump_any", "relief_any": "false",
              "ra_first": "NULL", "sn_gt_pay": "w_sn_gt_pay", "run_fail_15": "false", "run_fail_16": "false",
              "trailing_case": "false", "trailing_pass": "false", "freeze_applied": "false", "freeze_gap": "false"}
    cols = ", ".join(f"{values[c]}::{PATHRES_TYPES[c]} AS {c}" for c in lp.PATH_COLUMNS)
    return f"INSERT INTO w.{ag}_pathres SELECT loan_id, {cols}, 'fast' AS src FROM w.{ag}_pathsel WHERE fast"


def python_path(con, ag, fast, target, key):
    """Run loan_path.walk on the loans of w.{ag}_pathsel with fast = <fast>, in batches, resumable by loan_id."""
    last = inv.meta_get(con, key) or ""
    total = con.execute(f"SELECT count(*), count(*) FILTER (loan_id > ?) FROM w.{ag}_pathsel WHERE fast = ?",
                        [last, fast]).fetchone()
    done_n, todo_n = total[0] - total[1], total[1]
    progress(f"{AGENCY_NAME[ag]}: loan_path.py on {mc(total[0])} loans ({'fast-path check' if fast else 'non-trivial windows'}); "
             f"{mc(done_n)} already done")
    casts = ", ".join(f"unnest(${i + 2}::{PATHRES_TYPES[c]}[]) AS {c}" for i, c in enumerate(lp.PATH_COLUMNS))
    insert = f"INSERT INTO {target} SELECT unnest($1::VARCHAR[]) AS loan_id, {casts}, 'py' AS src"
    t0, n_done = time.time(), 0
    while True:
        rows = con.execute(f"SELECT loan_id, status48, relief24, zb_pay, last_pay, de_pay FROM w.{ag}_pathsel "
                           f"WHERE fast = ? AND loan_id > ? ORDER BY loan_id LIMIT {PY_BATCH}", [fast, last]).fetchall()
        if not rows:
            break
        ids, cols = [], {c: [] for c in lp.PATH_COLUMNS}
        for lid, s48, r24, zb, lastp, dep in rows:
            res = lp.walk(s48, r24, zb, lastp, dep)
            ids.append(lid)
            for c in lp.PATH_COLUMNS:
                cols[c].append(getattr(res, c))
        last = ids[-1]
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute(insert, [ids] + [cols[c] for c in lp.PATH_COLUMNS])
            inv.meta_set(con, key, last)
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise
        n_done += len(ids)
        rate = n_done / max(time.time() - t0, 1e-9)
        progress(f"  {AGENCY_NAME[ag]} path batch: {mc(n_done)} of {mc(todo_n)} loans done in this run; "
                 f"remaining ~{hms((todo_n - n_done) / rate)}")


# ---------------------------------------------------------------- stage 2: the decision (order of evaluation)

def decision_case():
    """The first reason, generated from loan_path.py's order constants (so SQL and Python agree by construction)."""
    hard = " ".join(f"WHEN {lp.flag_name(d)} THEN {lit(d)}" for d in lp.HARD_ORDER)
    bad = " ".join(f"WHEN {'bad_25_timed' if d == '25' else lp.flag_name(d)} THEN {lit(d)}" for d in lp.BAD_ORDER)
    soft = " ".join(f"WHEN {lp.flag_name(d)} THEN {lit(d)}" for d in lp.SOFT_ORDER)
    good = " ".join(f"WHEN {lp.flag_name(d)} THEN {lit(d)}" for d in lp.GOOD_ORDER)
    return f"CASE {hard} WHEN is_bad THEN (CASE {bad} END) {soft} {good} END"


def place_case(dec):
    def inlist(ds):
        return ", ".join(lit(d) for d in ds)
    return (f"CASE WHEN {dec} IN ({inlist(lp.HARD_ORDER + lp.SOFT_ORDER)}) THEN 'dropped' "
            f"WHEN {dec} IN ({inlist(lp.BAD_ORDER)}) THEN 'bad' WHEN {dec} IN ({inlist(lp.GOOD_ORDER)}) THEN 'good' "
            f"ELSE {lit(lp.NO_RULE)} END")


def decided_sql(ag, de, pathres="pathres", only_check_set=False):
    """One row per loan with values, path facts, every flag, the first reason and the place."""
    bad_all = ", ".join(f"CASE WHEN {'bad_25_timed' if d == '25' else lp.flag_name(d)} THEN {lit(d)} END" for d in lp.BAD_ORDER)
    buyback = lit(lp.BUYBACK_CODE[ag])
    bad_codes = ", ".join(lit(c) for c in lp.BAD_CODES)
    restrict = f"SEMI JOIN w.{ag}_{pathres} r2 USING (loan_id)" if only_check_set else ""
    # Every flag is a plain true/false (never NULL): mirrors loan_path.outcome_flags.
    flag_exprs = [
        ("f_9x", "f_9"), ("f_10x", "f_10"), ("f_1x", "f_1"), ("f_2x", "f_2"), ("f_3x", "f_3"), ("f_11x", "f_11"),
        ("f_39x", "f_39"), ("f_42r1x", "f_42r1"),
        ("f_12", "path_ok AND coalesce(ending, '') <> 'before1' AND (first_pay IS NULL OR first_pay >= 4 OR w_rows = 0)"),
        ("f_20", "path_ok AND sn_gt_pay"),
        ("f_14", f"path_ok AND ending IN ('before1', 'window') AND zb_code = {lit(lp.PAYOFF_CODE)}"),
        ("f_26", f"path_ok AND ending IN ('before1', 'window') AND zb_code IN ({bad_codes})"),
        ("f_29", f"path_ok AND ending IN ('before1', 'window') AND zb_code = {buyback}"),
        ("f_30", f"path_ok AND ending IN ('before1', 'window') AND zb_code = {lit(lp.RPL_CODE)}"),
        ("f_28_trailing", "path_ok AND trailing_case"),
        ("f_28", "path_ok AND ((last_pay >= 24 AND (zb_pay IS NULL OR zb_pay > 24)) OR trailing_case)"),
        ("f_18", "path_ok AND zb_pay IS NULL AND de_pay < 24"),
        ("f_15", "path_ok AND run_fail_15"), ("f_16", "path_ok AND run_fail_16"),
        ("f_23", "path_ok AND proven_first IS NOT NULL"),
        ("f_19", "path_ok AND jump3_any AND proven_first IS NULL"),
        ("f_24", "path_ok AND ra_first IS NOT NULL"),
        ("f_25", "path_ok AND fc_pay BETWEEN 1 AND 24"),
        ("f_freeze", "path_ok AND freeze_applied"), ("f_freeze_gap", "path_ok AND freeze_gap"),
        ("f_jump_any", "path_ok AND jump_any"), ("f_relief_any", "path_ok AND relief_any"),
    ]
    flags_sql = ",\n    ".join(f"coalesce({expr}, false) AS {name}" for name, expr in flag_exprs)
    return f"""
CREATE OR REPLACE TABLE w.{ag}_decided{'_check' if only_check_set else ''} AS
WITH base AS (
  SELECT i.loan_id, {lit(AGENCY_NAME[ag])} AS agency,
         v.* EXCLUDE (loan_id), p.* EXCLUDE (loan_id, unit, fpd_idx), r.* EXCLUDE (loan_id),
         m.median_rate, m.used AS median_used, m.n_loans AS median_n,
         (p.fpd_state = 'ok') AS path_ok,
         {de} - p.fpd_idx + 1 AS de_pay,
         p.fc_min - p.fpd_idx + 1 AS fc_pay,
         p.zb_date_idx - p.fpd_idx + 1 AS zb_eff_pay,
         CASE WHEN p.zb_pay IS NULL THEN NULL WHEN p.zb_pay <= 0 THEN 'before1' WHEN p.zb_pay <= 24 THEN 'window' ELSE 'after' END AS ending
  FROM w.{ag}_ids i {restrict}
  LEFT JOIN w.{ag}_vals v USING (loan_id)
  LEFT JOIN w.{ag}_path p USING (loan_id)
  LEFT JOIN w.{ag}_{pathres} r USING (loan_id)
  LEFT JOIN w.rate_medians m ON m.fpd_idx = v.fpd_idx AND m.product = v.rate_product
), flags AS (
  SELECT *,
    {flags_sql}
  FROM base
), timed AS (
  SELECT * EXCLUDE (f_9, f_10, f_1, f_2, f_3, f_11, f_39, f_42r1),
         f_9x AS f_9, f_10x AS f_10, f_1x AS f_1, f_2x AS f_2, f_3x AS f_3, f_11x AS f_11, f_39x AS f_39, f_42r1x AS f_42r1,
         coalesce(f_25 AND (NOT (f_29 OR f_30) OR fc_pay <= zb_pay), false) AS bad_25_timed
  FROM flags
), bad AS (
  SELECT *, (f_23 OR f_24 OR bad_25_timed OR f_26) AS is_bad FROM timed
), dec AS (
  SELECT *, {decision_case()} AS decision FROM bad
)
SELECT * EXCLUDE (f_9x, f_10x, f_1x, f_2x, f_3x, f_11x, f_39x, f_42r1x),
       {place_case('decision')} AS place,
       CASE WHEN is_bad AND {place_case('decision')} = 'bad' THEN concat_ws('+', {bad_all}) END AS bad_rules_all,
       CASE WHEN median_used THEN rate - median_rate END AS rate_spread,
       {ym_sql('fpd_idx')} AS first_payment_month,
       substr(orig_q, 1, 4) AS orig_year
FROM dec
"""


# ---------------------------------------------------------------- stage 2: checks and outputs

# (name, count SQL, policy): policy True = must be zero for both agencies; False = informational;
# a dict gives the policy per agency (None = not computed for that agency).
CHECK_ROWS = [
    ("loans with a loan-level row but no monthly rows", "SELECT count(*) FROM w.{ag}_ids WHERE has_ll AND NOT has_path", True),
    ("loans with monthly rows but no loan-level row", "SELECT count(*) FROM w.{ag}_ids WHERE has_path AND NOT has_ll", True),
    ("loans with monthly rows in more than one source file", "SELECT count(*) FROM (SELECT loan_id FROM w.{ag}_path GROUP BY 1 HAVING count(*) > 1)", True),
    ("loans with duplicate (loan, month) rows", "SELECT count(*) FROM w.{ag}_decided WHERE n_dup > 0", True),
    ("loans with an unparseable monthly period", "SELECT count(*) FROM w.{ag}_decided WHERE n_per_bad > 0", True),
    ("loans with a numeric status outside 00-99", "SELECT count(*) FROM w.{ag}_decided WHERE n_sn_bad > 0", True),
    ("loans with an unparseable status in the window (treated as unknown months)", "SELECT count(*) FROM w.{ag}_decided WHERE n_unparseable_w > 0", False),
    ("loans whose first payment date is missing or unparseable", "SELECT count(*) FROM w.{ag}_decided WHERE fpd_state IN ('missing', 'unparseable', 'no_orig')", True),
    ("loans whose first payment date is CONFLICTING (Fannie: rule 9)", "SELECT count(*) FROM w.{ag}_decided WHERE fpd_state = 'conflicting'", {"fre": True, "fan": False}),
    ("loans with more than one zero-balance code or termination row", "SELECT count(*) FROM w.{ag}_decided WHERE n_zb_codes > 1 OR n_zb_rows > 1", True),
    ("loans whose termination row is not their last row", "SELECT count(*) FROM w.{ag}_decided WHERE zb_rn IS NOT NULL AND zb_rn <> n_all", True),
    ("loans with conflicting foreclosure dates", "SELECT count(*) FROM w.{ag}_decided WHERE fc_min <> fc_max", True),
    ("loans with an unparseable foreclosure date", "SELECT count(*) FROM w.{ag}_decided WHERE fc_bad", True),
    ("loans whose origination quarter label is not a quarter", "SELECT count(*) FROM w.{ag}_decided WHERE q_ord IS NULL AND NOT f_9", True),
    ("loans whose zero-balance code in the window or before payment 1 has no rule", "SELECT count(*) FROM w.{ag}_decided WHERE ending IN ('before1', 'window') AND zb_code NOT IN ({codes})", True),
    ("Freddie loans with DTI 0 (not covered by the rules)", "SELECT count(*) FROM w.{ag}_decided WHERE dti_zero_raw", {"fre": True, "fan": None}),
    ("loans fitting no rule", "SELECT count(*) FROM w.{ag}_decided WHERE place = '" + lp.NO_RULE + "'", True),
    ("kept loans with no label", "SELECT count(*) FROM w.{ag}_decided WHERE place IN ('good', 'bad') AND label NOT IN ('train', 'validation', 'test')", True),
]


def run_checks(con, agencies):
    con.execute("DELETE FROM w.checks")
    for ag in agencies:
        codes = ", ".join(lit(c) for c in sorted(lp.KNOWN_CODES[ag]))
        for name, sql, policy in CHECK_ROWS:
            mbz = policy.get(ag) if isinstance(policy, dict) else policy
            if mbz is None:
                continue
            n = con.execute(sql.format(ag=ag, codes=codes)).fetchone()[0]
            con.execute("INSERT INTO w.checks VALUES (?, ?, ?, ?)", [name, ag, n, bool(mbz)])
        for v in NUMERIC_VARS + CODED_VARS:
            n = con.execute(f"SELECT count(*) FROM w.{ag}_decided WHERE bad_{v}").fetchone()[0]
            con.execute("INSERT INTO w.checks VALUES (?, ?, ?, true)", [f"loans with a value of {VAR_LABEL[v]} that has no rule", ag, n])


def outputs_sql(agencies):
    kept_vars = ("credit_score, ltv, cltv, dti, rate_spread, loan_purpose, mi_pct, num_borrowers, occupancy, first_time_homebuyer, "
                 "original_upb, channel, property_type, loan_term, num_units, special_program, high_balance")
    flags = ", ".join(lp.REASON_FLAGS + lp.INFO_FLAGS)

    def union(select):
        return " UNION ALL ".join(select.format(ag=ag) for ag in agencies)
    return [
        f"CREATE OR REPLACE TABLE o.kept_loans AS " + union(
            f"SELECT agency, loan_id, orig_q AS origination_quarter, label, place AS outcome, decision, bad_rules_all, "
            f"f_28_trailing AS trailing_run_case, {kept_vars} FROM w.{{ag}}_decided WHERE place IN ('good', 'bad')"),
        f"CREATE OR REPLACE TABLE o.kept_loans_audit AS " + union(
            f"SELECT agency, loan_id, rate AS original_interest_rate, median_rate AS rate_median, rate_product, first_payment_month "
            f"FROM w.{{ag}}_decided WHERE place IN ('good', 'bad')"),
        f"CREATE OR REPLACE TABLE o.loan_outcomes AS " + union(
            f"SELECT agency, loan_id, orig_q AS origination_quarter, "
            f"CASE WHEN label IN ('train', 'validation', 'test') THEN label END AS label, place, decision, {flags} FROM w.{{ag}}_decided"),
        "CREATE OR REPLACE TABLE o.rate_medians AS SELECT first_payment_month, product, n_loans, median_rate, used FROM w.rate_medians",
    ]


def write_meta(con, items):
    con.execute("CREATE OR REPLACE TABLE o.build_meta (key VARCHAR, value VARCHAR)")
    con.executemany("INSERT INTO o.build_meta VALUES (?, ?)", [(k, None if v is None else str(v)) for k, v in items])


def compare_paths(con, ag, de):
    """The SQL fast path must equal loan_path.py on every loan it handled, in every column."""
    con.execute(decided_sql(ag, de, pathres="pathres_check", only_check_set=True))
    cols = lp.PATH_COLUMNS + ["place", "decision", "bad_rules_all"] + lp.REASON_FLAGS + lp.INFO_FLAGS
    sel = ", ".join(f"count(*) FILTER (a.{c} IS DISTINCT FROM b.{c}) AS {c}" for c in cols)
    cur = con.execute(f"SELECT count(*), {sel} FROM w.{ag}_decided a JOIN w.{ag}_decided_check b USING (loan_id)")
    names = [d[0] for d in cur.description]
    row = cur.fetchone()
    n_fast = con.execute(f"SELECT count(*) FROM w.{ag}_pathres WHERE src = 'fast'").fetchone()[0]
    n_check = con.execute(f"SELECT count(*) FROM w.{ag}_pathres_check").fetchone()[0]
    return {"compared": row[0], "fast": n_fast, "checked": n_check, "mismatch": dict(zip(names[1:], row[1:]))}


# ---------------------------------------------------------------- stage 2 driver

def stage_done(con, name):
    return con.execute("SELECT count(*) FROM w.inv_done WHERE piece = 'stage' AND tbl = 'final' AND unit = ?",
                       [name]).fetchone()[0] > 0


def run_stage(con, name, fn):
    if stage_done(con, name):
        progress(f"Stage 2: {name} already finished")
        return
    started = time.time()
    progress(f"Stage 2: {name} ...")
    inv.commit_piece(con, "stage", "final", name, 0, started, fn)
    progress(f"Stage 2: {name} done in {hms(time.time() - started)}")


def final_stages(con, args, tabs, agencies, des):
    for ag in agencies:
        T = tabs["freddie_orig"] if ag == "fre" else tabs["fannie_raw"]
        run_stage(con, f"resolve_{ag}", lambda: inv.resolved_view(con, T))
        run_stage(con, f"ids_{ag}", lambda: con.execute(f"""
            CREATE OR REPLACE TABLE w.{ag}_ids AS
            SELECT loan_id, bool_or(src = 'll') AS has_ll, bool_or(src = 'path') AS has_path FROM (
                SELECT loan_id, 'll' AS src FROM w.{ag}_lr UNION ALL SELECT loan_id, 'path' FROM w.{ag}_path)
            GROUP BY loan_id"""))
    for ag in agencies:
        run_stage(con, f"vals_{ag}", lambda: con.execute(vals_sql(ag)))
    run_stage(con, "medians", lambda: con.execute(medians_sql(agencies)))
    for ag in agencies:
        run_stage(con, f"pathsel_{ag}", lambda: con.execute(pathsel_sql(ag, des[ag])))

        def fast_stage():
            con.execute(f"DELETE FROM w.{ag}_pathres")
            if not args.full_path:
                con.execute(fast_insert_sql(ag))
        run_stage(con, f"pathfast_{ag}", fast_stage)
        if not stage_done(con, f"pathpy_{ag}"):
            python_path(con, ag, False, f"w.{ag}_pathres", f"pathpy_{ag}_last")
            if args.full_path:
                python_path(con, ag, True, f"w.{ag}_pathres", f"pathpy_{ag}_last_fast")
            run_stage(con, f"pathpy_{ag}", lambda: None)
        if args.compare_paths and not args.full_path and not stage_done(con, f"pathcheck_{ag}"):
            con.execute(f"DELETE FROM w.{ag}_pathres_check")
            inv.meta_set(con, f"pathcheck_{ag}_last", "")
            python_path(con, ag, True, f"w.{ag}_pathres_check", f"pathcheck_{ag}_last")
            run_stage(con, f"pathcheck_{ag}", lambda: None)
        run_stage(con, f"decided_{ag}", lambda: con.execute(decided_sql(ag, des[ag])))
    run_stage(con, "checks", lambda: run_checks(con, agencies))


def write_outputs(con, agencies, meta_items):
    progress("Stage 3: writing the output database ...")
    con.execute("BEGIN TRANSACTION")
    try:
        for sql in outputs_sql(agencies):
            con.execute(sql)
        write_meta(con, meta_items)
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise


# ---------------------------------------------------------------- report

def fail(msg):
    FAILS.append(msg)
    out(f"  FAIL: {msg}")


def check_line(what, ok):
    out(f"  Check: {what}: {'OK' if ok else 'FAIL'}")
    if not ok:
        FAILS.append(what)


def rsection(title, fn, *args):
    try:
        fn(*args)
    except (KeyboardInterrupt, StopRun):
        raise
    except Exception as exc:
        out()
        out(f"SECTION FAILED: {title}")
        write_error_detail(exc, out)
        FAILS.append(f"report section failed: {title}")
    finally:
        inv.save_report()


def rep_header(args, units, selected, des, hashes, agencies):
    out(f"build_dataset.py v{SCRIPT_VERSION}  {datetime.now():%Y-%m-%d %H:%M}  mode {MODE}  database {load_raw.DB_PATH.name} (read-only)")
    out("Counts only: no loan IDs, no row-level values, no seller or servicer names. Counts from 1 to 10 show as '1-10'; "
        "totals are exact. Bad rates are blank when the bad count is 1-10.")
    out(f"Specification file SHA-256 {hashes['decision_log'] or 'file not found'}; "
        f"raw-load fingerprint {hashes['raw'][:16]}...; output {P['out'].name}; working database {P['work'].name}.")
    out(f"Memory limit {args.memory_limit}, threads {args.threads}, Python batches of {PY_BATCH:,} loans; "
        f"min free disk {args.min_free_gb:g} GB.")
    for ag in agencies:
        files = sorted({u for t, u, r in selected if t.startswith("freddie") == (ag == "fre")})
        out(f"{AGENCY_NAME[ag]}: data's last month {ym(des[ag])}"
            + (f"; files processed: {', '.join(files)}" if MODE == "smoke" else ""))
    per = {}
    for table, unit, rows in units:
        p = per.setdefault(table, [0, 0])
        p[0] += 1
        p[1] += rows or 0
    out("Loaded files (load_log): " + "; ".join(f"{t} {k} files, {mc(r)} rows" for t, (k, r) in sorted(per.items())))
    if MODE == "smoke":
        out("SMOKE MODE: one source file per agency; the rate medians come from these loans only, so spreads are not "
            "comparable to the full run; the agency totals are the loans in these files.")


def rep_1_decisions(con, agencies, expected):
    section("1. Final counts per decision (first rule that applies, in the order of evaluation), per agency and label")
    for ag in agencies:
        D = f"w.{ag}_decided"
        n = con.execute(f"SELECT count(*) FROM {D}").fetchone()[0]
        rows = con.execute(f"SELECT decision, place, label, count(*) FROM {D} GROUP BY ALL").fetchall()
        per = {}
        for dec, place, lab, k in rows:
            per.setdefault((dec, place), {})[lab] = k
        out()
        out(f"--- {AGENCY_NAME[ag]}: {mc(n)} loans")
        table = []
        for d in lp.ALL_DECISIONS:
            for (dec, place), labs in sorted(per.items(), key=lambda t: str(t[0][1])):
                if dec == d:
                    table.append([d, place, m(sum(labs.values()))] + [m(labs.get(lab, 0)) for lab in LABELS])
        for (dec, place), labs in per.items():
            if dec not in lp.ALL_DECISIONS:
                table.append([str(dec), place, m(sum(labs.values()))] + [m(labs.get(lab, 0)) for lab in LABELS])
        out_table(["decision", "place", "total"] + LABELS, table, left={0, 1})
        places = dict(con.execute(f"SELECT place, count(*) FROM {D} GROUP BY 1").fetchall())
        dropped, good, bad = places.get("dropped", 0), places.get("good", 0), places.get("bad", 0)
        no_rule = places.get(lp.NO_RULE, 0)
        out(f"  dropped {mc(dropped)} + good {mc(good)} + bad {mc(bad)}" + (f" + no rule {mc(no_rule)}" if no_rule else "")
            + f" = {dropped + good + bad + no_rule:,}")
        check_line(f"{AGENCY_NAME[ag]}: places add up to the loans in this run ({n:,})", dropped + good + bad + no_rule == n)
        if expected is not None:
            check_line(f"{AGENCY_NAME[ag]}: loans in this run = the agency's full loan count ({expected[ag]:,})", n == expected[ag])
        check_line(f"{AGENCY_NAME[ag]}: no loan fits no rule", no_rule == 0)
        out()
        out(f"  Kept loans by label ({AGENCY_NAME[ag]}):")
        kept = con.execute(f"SELECT label, count(*), count(*) FILTER (place = 'good'), count(*) FILTER (place = 'bad') "
                           f"FROM {D} WHERE place IN ('good', 'bad') GROUP BY 1").fetchall()
        kept = {r[0]: r[1:] for r in kept}
        trows = []
        for lab in ("train", "validation", "test"):
            k, g, b = kept.get(lab, (0, 0, 0))
            trows.append([lab, m(k), m(g), m(b), pct(b, k)])
        k, g, b = sum(v[0] for v in kept.values()), sum(v[1] for v in kept.values()), sum(v[2] for v in kept.values())
        trows.append(["all kept", m(k), m(g), m(b), pct(b, k)])
        out_table(["label", "kept", "good", "bad", "bad rate"], trows, left={0})
        check_line(f"{AGENCY_NAME[ag]}: every kept loan has a label", set(kept) <= {"train", "validation", "test"})


def rep_2_quarters(con, agencies):
    section("2. Kept loans per origination quarter and agency (good / bad / bad rate)")
    for ag in agencies:
        rows = con.execute(f"SELECT orig_q, label, count(*), count(*) FILTER (place = 'good'), count(*) FILTER (place = 'bad') "
                           f"FROM w.{ag}_decided WHERE place IN ('good', 'bad') GROUP BY ALL ORDER BY 1").fetchall()
        out()
        out(f"--- {AGENCY_NAME[ag]}")
        out_table(["quarter", "label", "kept", "good", "bad", "bad rate"],
                  [[q, lab, m(k), m(g), m(b), pct(b, k)] for q, lab, k, g, b in rows], left={0, 1})


def rep_3_crosstab(con, agencies):
    section("3. Cross-tab of every reason that applies to each loan")
    flags = lp.REASON_FLAGS
    short = [f[2:].upper() if f[2:].startswith("42") else f[2:] for f in flags]
    for ag in agencies:
        D = f"w.{ag}_decided"
        out()
        out(f"--- {AGENCY_NAME[ag]}: first reason (rows) x every reason that also applies (columns); the diagonal is the first reason's own count")
        sels = ", ".join(f"count(*) FILTER ({f})" for f in flags)
        rows = con.execute(f"SELECT decision, count(*), {sels} FROM {D} GROUP BY 1").fetchall()
        by = {r[0]: r[1:] for r in rows}
        table = [[d, m(by[d][0])] + [m(x) for x in by[d][1:]] for d in lp.ALL_DECISIONS if d in by]
        out_table(["first reason", "loans"] + short, table, left={0})
        nreasons = " + ".join(f"{f}::INTEGER" for f in flags)
        hist = con.execute(f"SELECT CASE WHEN n >= 4 THEN '4+' ELSE CAST(n AS VARCHAR) END, count(*) FROM "
                           f"(SELECT {nreasons} AS n FROM {D}) GROUP BY 1 ORDER BY 1").fetchall()
        out()
        out("  Number of reasons that apply per loan:")
        out_table(["reasons", "loans"], [[k, m(v)] for k, v in hist], left={0})
        combo = "concat_ws('+', " + ", ".join(f"CASE WHEN {f} THEN {lit(s)} END" for f, s in zip(flags, short)) + ")"
        top = con.execute(f"SELECT coalesce(nullif({combo}, ''), '(none)'), count(*) AS k FROM {D} GROUP BY 1 "
                          f"ORDER BY k DESC, 1 LIMIT 100").fetchall()
        out()
        out("  Top 100 combinations of reasons:")
        out_table(["rank", "reasons", "loans"], [[i, c, m(k)] for i, (c, k) in enumerate(top, 1)], left={1})


def rep_4_before1(con, agencies):
    section("4. Loans ending before payment 1, by ending code and final place")
    for ag in agencies:
        rows = con.execute(f"SELECT zb_code, place, decision, count(*) FROM w.{ag}_decided WHERE ending = 'before1' "
                           f"GROUP BY ALL ORDER BY 1, 2, 3").fetchall()
        out()
        out(f"--- {AGENCY_NAME[ag]}")
        out_table(["ending code", "place", "decision", "loans"], [[c, p, d, m(k)] for c, p, d, k in rows], left={0, 1, 2})
        after = con.execute(f"SELECT zb_code, count(*) FROM w.{ag}_decided WHERE ending = 'after' GROUP BY 1 ORDER BY 1").fetchall()
        out(f"  Termination codes after payment 24 (outside the window, no rule needed): "
            + ("; ".join(f"{c}: {mc(k)}" for c, k in after) or "none"))


def rep_5_score_blanks(con, agencies):
    section("5. Where Fannie credit-score blanks fall across a loan's rows")
    if "fan" not in agencies:
        out("  (Fannie Mae not in this run)")
        return
    for what, pre in (("borrower score", "fb"), ("co-borrower score", "fcb")):
        cat = f"""CASE WHEN {pre}_null_n = 0 THEN 'never blank' WHEN {pre}_null_n = n_rows THEN 'blank on every row'
                 WHEN {pre}_null_n = 1 AND {pre}_null_last = n_rows THEN 'final row only'
                 WHEN {pre}_null_first = 1 AND {pre}_null_last = {pre}_null_n THEN 'first row(s) only'
                 WHEN {pre}_null_last = n_rows AND {pre}_null_last - {pre}_null_first + 1 = {pre}_null_n THEN 'last rows only (more than one)'
                 ELSE 'scattered' END"""
        rows = con.execute(f"SELECT {cat}, count(*) FROM w.fan_decided GROUP BY 1 ORDER BY 1").fetchall()
        out()
        out(f"  Fannie {what}:")
        out_table(["where the blanks fall", "loans"], [[c, m(k)] for c, k in rows], left={0})


def rep_6_relief_first(con, agencies):
    section("6. First month each relief flag appears, per agency")
    for ag in agencies:
        r = con.execute(f"SELECT min(first_per_f), min(first_per_r), min(first_per_t), min(first_per_d), "
                        f"count(*) FILTER (first_per_f IS NOT NULL), count(*) FILTER (first_per_r IS NOT NULL), "
                        f"count(*) FILTER (first_per_t IS NOT NULL), count(*) FILTER (first_per_d IS NOT NULL) FROM w.{ag}_path").fetchone()
        rows = [["BAP F (forbearance)", ym(r[0]), m(r[4])], ["BAP R (repayment plan)", ym(r[1]), m(r[5])],
                ["BAP T (trial modification)", ym(r[2]), m(r[6])]]
        if ag == "fre":
            rows.append(["disaster flag Y", ym(r[3]), m(r[7])])
        out()
        out(f"--- {AGENCY_NAME[ag]} (any row, not only the window)")
        out_table(["flag", "first month", "loans with the flag"], rows, left={0, 1})


def rep_7_cltv_lt_ltv(con, agencies):
    section("7. Loans with combined LTV lower than LTV (both set to missing)")
    for ag in agencies:
        n, kept = con.execute(f"SELECT count(*) FILTER (f_42r2_cltv_lt_ltv), count(*) FILTER (f_42r2_cltv_lt_ltv AND place IN ('good', 'bad')) "
                              f"FROM w.{ag}_decided").fetchone()
        out(f"  {AGENCY_NAME[ag]}: {mc(n)} loans (all); {mc(kept)} among kept loans")


def rep_8_jump_only(con, agencies):
    section("8. Loans whose only 90+ was a jump month (rule 19)")
    for ag in agencies:
        r = con.execute(f"SELECT count(*) FILTER (f_19), count(*) FILTER (decision = '19'), count(*) FILTER (f_19 AND place = 'bad'), "
                        f"count(*) FILTER (f_19 AND place = 'dropped' AND decision <> '19') FROM w.{ag}_decided").fetchone()
        out(f"  {AGENCY_NAME[ag]}: condition holds for {mc(r[0])} loans; dropped under 19: {mc(r[1])}; "
            f"bad by 24/25/26 instead: {mc(r[2])}; dropped by an earlier rule: {mc(r[3])}")


def rep_9_loan_age(con, agencies):
    section("9. Freddie loan_age vs the payment number: largest difference in the window, by vintage and final place")
    if "fre" not in agencies:
        out("  (Freddie Mac not in this run)")
        return
    band = ("CASE WHEN NOT path_ok OR w_rows = 0 THEN 'no window rows' WHEN la_missing_w AND la_maxdiff IS NULL THEN 'loan_age missing' "
            "WHEN la_maxdiff = 0 THEN 'agrees' " + " ".join(
                f"WHEN la_maxdiff BETWEEN {lo} AND {hi if hi is not None else 10**9} THEN {lit(lab)}" for lab, lo, hi in BANDS_LA) + " END")
    rows = con.execute(f"SELECT orig_year, place, {band} AS b, count(*) FROM w.fre_decided GROUP BY ALL").fetchall()
    bands = ["agrees"] + [b for b, _, _ in BANDS_LA] + ["loan_age missing", "no window rows"]
    per = {}
    for y, p, b, k in rows:
        per.setdefault((y, p), {})[b] = k
    out_table(["vintage", "place", "loans"] + bands,
              [[y, p, m(sum(d.values()))] + [m(d.get(b, 0)) for b in bands] for (y, p), d in sorted(per.items(), key=lambda t: (str(t[0][0]), str(t[0][1])))],
              left={0, 1})


def rep_10_zb_timing(con, agencies):
    section("10. Freddie: termination class by row period vs by zero-balance effective date")
    if "fre" not in agencies:
        out("  (Freddie Mac not in this run)")
        return
    cls = "CASE WHEN {p} IS NULL THEN 'no date' WHEN {p} <= 0 THEN 'before payment 1' WHEN {p} <= 24 THEN 'payments 1-24' ELSE 'after payment 24' END"
    a, b = cls.format(p="zb_pay"), cls.format(p="zb_eff_pay")
    rows = con.execute(f"SELECT {a}, {b}, zb_code, count(*) FROM w.fre_decided WHERE zb_pay IS NOT NULL AND {a} <> {b} "
                       f"GROUP BY ALL ORDER BY 1, 2, 3").fetchall()
    total = sum(r[3] for r in rows)
    out(f"  Loans whose class would change: {mc(total)} (of {mc(con.execute('SELECT count(*) FROM w.fre_decided WHERE zb_pay IS NOT NULL').fetchone()[0])} terminated loans)")
    out_table(["by row period", "by effective date", "code", "loans"], [[x, y, c, m(k)] for x, y, c, k in rows], left={0, 1, 2})


def rep_11_xx_nonterm(con, agencies):
    section("11. Fannie loans with XX/blank status on non-termination window months")
    if "fan" not in agencies:
        out("  (Fannie Mae not in this run)")
        return
    r = con.execute("SELECT count(*) FILTER (status48 LIKE '%XX%'), "
                    "coalesce(sum((length(status48) - length(replace(status48, 'XX', ''))) / 2), 0), "
                    "count(*) FILTER (status48 LIKE '%TX%') FROM w.fan_decided WHERE path_ok").fetchone()
    out(f"  loans: {mc(r[0])}; such months: {mc(int(r[1]))}; loans whose termination row has an ignored XX/blank status: {mc(r[2])}")


def rep_12_bap_blank(con, agencies):
    section("12. Fannie window months with blank BAP, by date and by termination row vs other rows")
    if "fan" not in agencies:
        out("  (Fannie Mae not in this run)")
        return
    r = con.execute("SELECT sum(bapblank_pre_other), sum(bapblank_post_other), sum(bapblank_pre_term), sum(bapblank_post_term), "
                    "count(*) FILTER (bapblank_pre_other + bapblank_post_other > 0), count(*) FILTER (bapblank_pre_term + bapblank_post_term > 0) "
                    "FROM w.fan_path").fetchone()
    out_table(["rows", "before April 2020", "April 2020 or later"],
              [["other window rows", m(r[0]), m(r[1])], ["termination rows", m(r[2]), m(r[3])]], left={0})
    out(f"  loans with a blank-BAP ordinary window month: {mc(r[4])}; with a blank-BAP termination row in the window: {mc(r[5])}")


def rep_13_freeze(con, agencies):
    section("13. Loans the freeze rule affects")
    for ag in agencies:
        rows = con.execute(f"SELECT place, count(*) FILTER (f_freeze), count(*) FILTER (f_freeze_gap) FROM w.{ag}_decided "
                           f"WHERE f_freeze GROUP BY 1 ORDER BY 1").fetchall()
        tot = con.execute(f"SELECT count(*) FILTER (f_freeze), count(*) FILTER (f_freeze_gap) FROM w.{ag}_decided").fetchone()
        out()
        out(f"--- {AGENCY_NAME[ag]}: freeze rule applied {mc(tot[0])} loans; with a run of unknown months before the jump month {mc(tot[1])}")
        out_table(["final place", "freeze applied", "of which with a gap"], [[p, m(a), m(b)] for p, a, b in rows], left={0})


def rep_14_dti0(con, agencies):
    section("14. Loans with DTI 0 (Fannie: set to missing; Freddie: must be 0)")
    for ag in agencies:
        n = con.execute(f"SELECT count(*) FILTER (dti_zero_raw) FROM w.{ag}_decided").fetchone()[0]
        out(f"  {AGENCY_NAME[ag]}: {mc(n)}")


def rep_15_spread(con, agencies):
    section("15. Rate-spread groups: sizes, and loans with a missing spread because their group has fewer than 30 loans")
    rows = con.execute("SELECT product, count(*), count(*) FILTER (NOT used), sum(n_loans), sum(n_loans) FILTER (NOT used), "
                       "min(n_loans), max(n_loans) FROM w.rate_medians GROUP BY 1 ORDER BY 1").fetchall()
    out_table(["product", "groups", "groups under 30", "loans in population", "of which in small groups", "smallest group", "largest group"],
              [[p, m(g), m(s), m(n), m(ns), m(mn), m(mx)] for p, g, s, n, ns, mn, mx in rows], left={0})
    bands = [("1-9", 1, 9), ("10-29", 10, 29), ("30-99", 30, 99), ("100-999", 100, 999), ("1,000-9,999", 1000, 9999), ("10,000+", 10000, None)]
    hist = con.execute("SELECT " + ", ".join(f"count(*) FILTER (n_loans BETWEEN {lo} AND {hi if hi else 10**12})" for _, lo, hi in bands)
                       + " FROM w.rate_medians").fetchone()
    out("  Group sizes: " + "; ".join(f"{lab}: {mc(k)}" for (lab, _, _), k in zip(bands, hist)))
    for ag in agencies:
        r = con.execute(f"SELECT count(*) FILTER (rate IS NOT NULL AND rate_spread IS NULL AND median_used IS NOT TRUE), "
                        f"count(*) FILTER (rate IS NULL), count(*) FILTER (rate_spread IS NOT NULL) FROM w.{ag}_decided "
                        f"WHERE place IN ('good', 'bad')").fetchone()
        out(f"  {AGENCY_NAME[ag]} kept loans: spread missing because the group is too small {mc(r[0])}; "
            f"spread missing because the rate is missing {mc(r[1])}; spread present {mc(r[2])}")


def rep_16_ltv_mi(con, agencies):
    section("16. Loans with LTV over 80 and mortgage insurance 0, per agency (a check on 41)")
    for ag in agencies:
        r = con.execute(f"SELECT count(*) FILTER (ltv > 80 AND mi_pct = 0), count(*) FILTER (ltv > 80 AND mi_pct = 0 AND place IN ('good', 'bad')), "
                        f"count(*) FILTER (ltv > 80 AND mi_pct IS NOT NULL), count(*) FILTER (ltv > 80 AND mi_pct IS NOT NULL AND place IN ('good', 'bad')) "
                        f"FROM w.{ag}_decided").fetchone()
        out(f"  {AGENCY_NAME[ag]}: all loans {mc(r[0])} of {mc(r[2])} with LTV over 80; kept loans {mc(r[1])} of {mc(r[3])}")


def rep_17_classic_fico(con, agencies):
    section("17. Fannie origination_classic_fico vs the lower of the borrower and co-borrower scores, by vintage")
    if "fan" not in agencies:
        out("  (Fannie Mae not in this run)")
        return
    rows = con.execute("SELECT orig_year, count(*) FILTER (fico_classic_raw IS NOT NULL), "
                       "count(*) FILTER (fico_classic_raw IS NOT NULL AND fico_classic_raw = credit_score), "
                       "count(*) FILTER (fico_classic_raw IS NOT NULL AND credit_score IS NULL) FROM w.fan_decided GROUP BY 1 ORDER BY 1").fetchall()
    out_table(["vintage", "classic FICO filled", "equal to the lower score", "share", "lower score missing"],
              [[y, m(f), m(e), pct(e, f), m(ms)] for y, f, e, ms in rows], left={0})


def rep_18_trailing(con, agencies):
    section("18. Loans whose records stop before payment 24 with no termination code while the data goes on")
    for ag in agencies:
        r = con.execute(f"SELECT count(*) FILTER (f_28_trailing), count(*) FILTER (f_28_trailing AND trailing_pass), "
                        f"count(*) FILTER (f_28_trailing AND place = 'good'), count(*) FILTER (f_28_trailing AND place = 'bad'), "
                        f"count(*) FILTER (f_28_trailing AND place = 'dropped') FROM w.{ag}_decided").fetchone()
        out(f"  {AGENCY_NAME[ag]}: {mc(r[0])} loans; trailing run passed {mc(r[1])}; final place good {mc(r[2])}, bad {mc(r[3])}, dropped {mc(r[4])}")


def rep_conversions(con, agencies):
    section("Conversions applied and structural checks")
    for ag in agencies:
        D = f"w.{ag}_decided"
        out()
        out(f"--- {AGENCY_NAME[ag]}")
        if ag == "fre":
            items = [("credit score 9999 -> missing", "fico_b_raw = 9999"), ("LTV 999 -> missing", "ltv_raw = 999"),
                     ("combined LTV 999 -> missing", "cltv_raw = 999"), ("DTI 999 -> missing", "dti_raw = 999"),
                     ("MI % 999 -> missing", "mi_raw = 999"), ("first-time homebuyer 9 -> missing", "fthb_c = '9'"),
                     ("channel 9 -> missing", "chan_c = '9'"), ("occupancy 9 -> missing", "occ_c = '9'"),
                     ("purpose 9 -> missing", "purp_c = '9'"), ("units 99 -> missing", "units_c = '99'"),
                     ("borrowers 99 -> missing", "borrowers_raw = 99"), ("property type 99 -> missing", "prop_c = '99'"),
                     ("special program blank -> none", "sep_c IS NULL")]
        else:
            blanks = [("borrower score blank on every row", "borrower_credit_score_at_origination"),
                      ("co-borrower score blank on every row -> no co-borrower", "co_borrower_credit_score_at_origination"),
                      ("DTI blank on every row -> missing", "debt_to_income_dti"),
                      ("combined LTV blank on every row -> missing", "original_combined_loan_to_value_ratio_cltv"),
                      ("first-time homebuyer blank on every row -> missing", "first_time_home_buyer_indicator")]
            sels = ", ".join(f"count(*) FILTER (l.{c} IS NULL AND NOT l.{c}__conf)" for _, c in blanks)
            vals = con.execute(f"SELECT {sels} FROM w.fan_lr l").fetchone()
            out_table(["conversion (from the resolved loan-level values)", "loans"],
                      [[name, m(v)] for (name, _), v in zip(blanks, vals)], left={0})
            items = [("MI % blank on every row -> 0", "mi_blank_all"), ("purpose U -> missing", "purp_c = 'U'"),
                     ("occupancy U -> missing", "occ_c = 'U'"), ("special program 9 -> missing", "sep_c = '9'"),
                     ("special program 7 -> none", "sep_c = '7'"), ("prepayment-penalty flag blank -> N", "ppm_blank"),
                     ("interest-only flag blank -> N", "io_blank"), ("DTI 0 -> missing", "f_42r2_dti")]
        items += [("rate 0 or below -> missing", "f_42r2_rate"), ("combined LTV below LTV -> both missing", "f_42r2_cltv_lt_ltv")]
        sels = ", ".join(f"count(*) FILTER ({c})" for _, c in items)
        vals = con.execute(f"SELECT {sels} FROM {D}").fetchone()
        out_table(["conversion", "loans"], [[name, m(v)] for (name, _), v in zip(items, vals)], left={0})
        src = con.execute(f"SELECT score_source, count(*) FROM {D} GROUP BY 1 ORDER BY 1").fetchall()
        out("  Credit score source: " + "; ".join(f"{s}: {mc(k)}" for s, k in src))
        if ag == "fan":
            confs = ", ".join(f"count(*) FILTER (l.{c}__conf)" for c in FANNIE_USED_FIELDS)
            vals = con.execute(f"SELECT {confs} FROM w.fan_lr l").fetchone()
            out()
            out("  Rule 39: loans whose value differs across rows, per used field (all loans):")
            out_table(["field", "loans"], [[c, m(v)] for c, v in zip(FANNIE_USED_FIELDS, vals)], left={0})
        rows = con.execute("SELECT name, value, must_be_zero FROM w.checks WHERE agency = ? ORDER BY must_be_zero DESC, name", [ag]).fetchall()
        out()
        out("  Structural and value checks (must-be-zero items stop the build if above 0):")
        out_table(["check", "loans", "must be zero"], [[n, m(v), "yes" if z else "no"] for n, v, z in rows], left={0, 2})
        for n, v, z in rows:
            if z and v:
                FAILS.append(f"{AGENCY_NAME[ag]}: {n} = {mc(v)}")


def rep_profile(con, agencies):
    section("Kept-loans profile: value counts of the categorical variables and summary of the numeric ones (counts only)")
    cats = ["loan_purpose", "num_borrowers", "occupancy", "first_time_homebuyer", "channel", "property_type", "num_units",
            "special_program", "high_balance"]
    nums = [("credit_score", "score"), ("ltv", "ratio"), ("cltv", "ratio"), ("dti", "dti"), ("rate_spread", None), ("mi_pct", "mi_pct"),
            ("original_upb", "upb"), ("loan_term", "term")]
    for ag in agencies:
        D = f"w.{ag}_decided"
        out()
        out(f"--- {AGENCY_NAME[ag]} kept loans")
        for c in cats:
            rows = con.execute(f"SELECT coalesce(CAST({c} AS VARCHAR), '(missing)'), count(*), count(*) FILTER (place = 'bad') "
                               f"FROM {D} WHERE place IN ('good', 'bad') GROUP BY 1 ORDER BY 1").fetchall()
            out(f"  {c}: " + "; ".join(f"{v} {mc(k)} (bad rate {pct(b, k) or '-'})" for v, k, b in rows))
        table = []
        for c, band in nums:
            r = con.execute(f"SELECT count({c}), count(*) FILTER ({c} IS NULL), min({c}), max({c}) FROM {D} WHERE place IN ('good', 'bad')").fetchone()
            bands = ""
            if band:
                sels = ", ".join(f"count(*) FILTER ({cond.format(x=c)})" for _, cond in inv.BANDS[band])
                vals = con.execute(f"SELECT {sels} FROM {D} WHERE place IN ('good', 'bad')").fetchone()
                bands = "  ".join(f"{lab}: {mc(v)}" for (lab, _), v in zip(inv.BANDS[band], vals))
            table.append([c, m(r[0]), m(r[1]), inv.fmt_value(r[2]), inv.fmt_value(r[3]), bands])
        out_table(["variable", "present", "missing", "min", "max", "bands"], table, left={0, 5})


def rep_compare(results):
    section("Fast path vs full path: every column compared on every loan the SQL fast path handled")
    for ag, r in results.items():
        bad = {c: k for c, k in r["mismatch"].items() if k}
        out(f"  {AGENCY_NAME[ag]}: fast-path loans {mc(r['fast'])}; re-run through loan_path.py {mc(r['checked'])}; compared {mc(r['compared'])}; "
            f"columns with mismatches: {len(bad)}")
        if bad:
            out_table(["column", "mismatching loans"], [[c, m(k)] for c, k in bad.items()], left={0})
        check_line(f"{AGENCY_NAME[ag]}: fast path equals loan_path.py on every compared loan and column",
                   not bad and r["compared"] == r["fast"] == r["checked"])


def rep_outputs(con, agencies, expected_total):
    section("Output tables: uniqueness and row counts (requirements 2 and 3)")
    counts = {}
    for t in ("kept_loans", "kept_loans_audit", "loan_outcomes"):
        n, d = con.execute(f"SELECT count(*), count(DISTINCT (agency, loan_id)) FROM o.{t}").fetchone()
        counts[t] = n
        out(f"  {t}: {mc(n)} rows, {mc(d)} distinct (agency, loan_id)")
        check_line(f"{t}: (agency, loan_id) is unique", n == d)
    kept = con.execute("SELECT count(*) FROM o.loan_outcomes WHERE place IN ('good', 'bad')").fetchone()[0]
    check_line("every kept loan is in kept_loans (kept_loans rows = good + bad in loan_outcomes)", counts["kept_loans"] == kept)
    check_line("every kept loan is in kept_loans_audit (same row count as kept_loans)", counts["kept_loans_audit"] == counts["kept_loans"])
    check_line("kept_loans' label is never NULL", con.execute("SELECT count(*) FROM o.kept_loans WHERE label IS NULL").fetchone()[0] == 0)
    n_ids = sum(con.execute(f"SELECT count(*) FROM w.{ag}_ids").fetchone()[0] for ag in agencies)
    check_line(f"loan_outcomes holds every loan of this run exactly once ({n_ids:,} loans)", counts["loan_outcomes"] == n_ids)
    if expected_total is not None:
        check_line(f"loan_outcomes rows = both agencies' full loan count ({expected_total:,})", counts["loan_outcomes"] == expected_total)
    rows = con.execute("SELECT agency, place, count(*) FROM o.loan_outcomes GROUP BY ALL ORDER BY 1, 2").fetchall()
    out_table(["agency", "place", "loans"], [[a, p, m(k)] for a, p, k in rows], left={0, 1})


def rep_summary(started):
    section(f"SUMMARY: {len(FAILS)} problem(s)")
    for f in FAILS:
        out(f"  FAIL: {f}")
    if not FAILS:
        out("  SUCCESS: every loan landed in exactly one place, every total adds up, every check passed.")
    else:
        out("  FAILURE: the dataset must not be used until the problems above are resolved.")
    out(f"(build took {hms(time.time() - started)})")


# ---------------------------------------------------------------- main

def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--smoke", action="store_true", help="one source file per agency; writes *_smoke files")
    p.add_argument("--smoke-units", nargs=2, metavar=("FREDDIE_Q", "FANNIE_Q"), help=f"smoke files (default {DEFAULT_SMOKE['fre']} {DEFAULT_SMOKE['fan']})")
    p.add_argument("--agency", choices=["freddie", "fannie"], help="run one agency (default: both)")
    p.add_argument("--compare-paths", action="store_true", help="also run loan_path.py on the fast-path loans and compare (always on in smoke mode)")
    p.add_argument("--full-path", action="store_true", help="no SQL fast path: loan_path.py on every loan")
    p.add_argument("--report-only", action="store_true", help="rebuild the report from a finished working database")
    p.add_argument("--memory-limit", default=inv.DEFAULT_MEMORY_LIMIT, help=f"DuckDB memory limit (default {inv.DEFAULT_MEMORY_LIMIT})")
    p.add_argument("--threads", type=int, default=inv.DEFAULT_THREADS, help=f"DuckDB threads (default {inv.DEFAULT_THREADS})")
    p.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB, help=f"refuse to start or continue below this much free disk (default {DEFAULT_MIN_FREE_GB})")
    p.add_argument("--clean", action="store_true", help="delete this mode's working and output files and stop")
    args = p.parse_args(argv)
    if not re.fullmatch(r"\d+(\.\d+)?\s*[KMGT]i?B", args.memory_limit, re.IGNORECASE):
        p.error("--memory-limit must look like 5GB or 4096MB")
    if args.smoke:
        args.compare_paths = True
    return args


def clean():
    for path in (P["work"], P["work"].with_name(P["work"].name + ".wal"), P["out"], P["out"].with_name(P["out"].name + ".wal"), P["lock"]):
        if path.exists():
            path.unlink()
            print(f"deleted {path}")
    if P["spill"].exists():
        shutil.rmtree(P["spill"])
        print(f"deleted {P['spill']}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    configure("smoke" if args.smoke else "full")
    if args.clean:
        return clean()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    LOGS.mkdir(exist_ok=True)
    check_raw.report.clear()
    FAILS.clear()
    started = time.time()
    progress("=" * 78)
    progress(f"build_dataset.py v{SCRIPT_VERSION} started ({MODE} mode). Raw database {load_raw.DB_PATH} (read-only); "
             f"working database {P['work'].name}; output database {P['out'].name}.")
    con, code, locked = None, 0, False
    try:
        if not load_raw.DB_PATH.exists():
            progress(f"No database at {load_raw.DB_PATH}; run load_raw.py first.")
            return 1
        acquire_lock()
        locked = True
        check_disk(args.min_free_gb, "at start")
        progress(f"DuckDB memory limit {args.memory_limit}, threads {args.threads}; Python batches of {PY_BATCH:,} loans. "
                 f"Free disk {shutil.disk_usage(WORK_ROOT).free / 1e9:,.1f} GB; needs about: {NEED_GB_TEXT}; "
                 f"stops below {args.min_free_gb:g} GB.")
        con = inv.open_db(args)
        tabs = inv.build_tabs(read_layouts())
        inv.create_work_tables(con, tabs)
        create_build_tables(con)
        units = inv.load_units(con)
        inv.check_work_matches(con, units)
        check_build_meta(con, args)
        agencies = [a for a in inv.AGENCIES if args.agency is None or inv.AGENCIES[a].lower().startswith(args.agency)]
        selected = select_units(units, args)
        des = {ag: inv.data_end(con, ag) for ag in agencies}
        plan = plan_pieces(selected, agencies)
        con.close()
        con = None
        if not args.report_only:
            process_pieces(args, tabs, plan, des)
        con = inv.open_db(args)
        done = set(con.execute("SELECT piece, tbl, unit FROM w.inv_done").fetchall())
        missing = [p for p in plan if (p[0], p[1], p[2]) not in done]
        if missing:
            progress(f"STOP: {len(missing)} file piece(s) are not finished; rerun without --report-only.")
            raise StopRun(1)
        if not args.report_only:
            final_stages(con, args, tabs, agencies, des)
        compare = {}
        if args.compare_paths and not args.full_path:
            for ag in agencies:
                compare[ag] = compare_paths(con, ag, des[ag])
                if any(compare[ag]["mismatch"].values()) or compare[ag]["compared"] != compare[ag]["fast"]:
                    FAILS.append(f"{AGENCY_NAME[ag]}: the SQL fast path differs from loan_path.py")
        hashes = {"decision_log": sha256_file(DECISION_LOG), "raw": hashlib.sha256(repr(units).encode()).hexdigest()}
        con.execute(f"ATTACH {lit(P['out'])} AS o (READ_WRITE)")
        meta = [("script_version", SCRIPT_VERSION), ("built_at", f"{datetime.now():%Y-%m-%d %H:%M:%S}"), ("mode", MODE),
                ("decision_log_sha256", hashes["decision_log"]), ("raw_load_fingerprint", hashes["raw"]),
                ("memory_limit", args.memory_limit), ("threads", args.threads), ("full_path", args.full_path),
                ("files_processed", "; ".join(f"{t} {u}" for t, u, r in selected))]
        for ag in agencies:
            meta += [(f"data_end_{ag}", ym(des[ag])), (f"loans_{ag}", con.execute(f"SELECT count(*) FROM w.{ag}_ids").fetchone()[0]),
                     (f"kept_{ag}", con.execute(f"SELECT count(*) FROM w.{ag}_decided WHERE place IN ('good', 'bad')").fetchone()[0])]
        write_outputs(con, agencies, meta)
        progress("Writing the report ...")
        full = MODE == "full"
        rsection("header", rep_header, args, units, selected, des, hashes, agencies)
        rsection("1 decisions", rep_1_decisions, con, agencies, EXPECTED_LOANS if full else None)
        rsection("2 quarters", rep_2_quarters, con, agencies)
        rsection("3 cross-tab", rep_3_crosstab, con, agencies)
        rsection("4 before payment 1", rep_4_before1, con, agencies)
        rsection("5 score blanks", rep_5_score_blanks, con, agencies)
        rsection("6 relief first month", rep_6_relief_first, con, agencies)
        rsection("7 cltv below ltv", rep_7_cltv_lt_ltv, con, agencies)
        rsection("8 jump only", rep_8_jump_only, con, agencies)
        rsection("9 loan_age", rep_9_loan_age, con, agencies)
        rsection("10 termination timing", rep_10_zb_timing, con, agencies)
        rsection("11 XX non-termination", rep_11_xx_nonterm, con, agencies)
        rsection("12 blank BAP", rep_12_bap_blank, con, agencies)
        rsection("13 freeze rule", rep_13_freeze, con, agencies)
        rsection("14 DTI 0", rep_14_dti0, con, agencies)
        rsection("15 spread groups", rep_15_spread, con, agencies)
        rsection("16 LTV and MI", rep_16_ltv_mi, con, agencies)
        rsection("17 classic FICO", rep_17_classic_fico, con, agencies)
        rsection("18 trailing runs", rep_18_trailing, con, agencies)
        rsection("conversions and checks", rep_conversions, con, agencies)
        rsection("kept-loans profile", rep_profile, con, agencies)
        if compare:
            rsection("fast path comparison", rep_compare, compare)
        rsection("outputs", rep_outputs, con, agencies, sum(EXPECTED_LOANS[a] for a in agencies) if full else None)
        rep_summary(started)
        code = 1 if FAILS else 0
    except StopRun as stop:
        code = stop.code
    except KeyboardInterrupt:
        progress("Interrupted. The piece or stage in progress was rolled back; finished work is kept. Rerun the same command to resume.")
        code = 130
    except Exception as exc:
        write_error_detail(exc, progress)
        code = 1
    finally:
        if con is not None:
            con.close()
        if P["spill"].exists() and not any(P["spill"].iterdir()):
            P["spill"].rmdir()
        if check_raw.report:
            inv.save_report()
            progress(f"Report saved to {P['report']}")
        if locked:
            release_lock()
        word = "SUCCESS" if code == 0 else "INTERRUPTED (rerun to resume)" if code == 130 else "FAILURE"
        progress(f"{word}: exit code {code}, elapsed {hms(time.time() - started)}.")
    return code


if __name__ == "__main__":
    sys.exit(main())
