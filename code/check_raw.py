"""Check the raw DuckDB load built by load_raw.py.

Summaries only: counts, field counts, sizes. No row-level values, no loan IDs, and never the
rejected-row text (raw_rejects.csv_line / error_message are not read).

Reports
  (a) each file's detected delimiter and field count vs the layout (recorded in load_log at load
      time, since the unzipped copies are deleted after loading); files not loaded are flagged
  (b) rows per source file in every table, reconciled with load_log and the file's line count
  (c) Freddie: loan IDs in performance but not in origination, and vice versa (counts only)
  (d) rejected (unparseable) rows by file and error type
  (e) database size
  plus a column alignment check: entirely empty columns per table, and Fannie columns the layout
  marks as not populated in the SF Loan Performance dataset that nevertheless contain data.

The report also goes to logs/check_raw.txt.
Usage (from the repository root): python code/check_raw.py
"""

import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_raw import (  # noqa: E402
    DB_PATH, EXPECTED_DELIMITER, LOG_DIR, TABLES, connect, db_size, gb, list_units,
    quote_ident, report_error, show_delimiter,
)

REPORT_PATH = LOG_DIR / "check_raw.txt"
ERROR_DETAIL_PATH = LOG_DIR / "check_raw_error_detail.txt"
FREDDIE_LOAN_ID = quote_ident("loan_identifier")

report = []


def out(text=""):
    print(text, flush=True)
    report.append(text)


def out_table(headers, rows, left=None):
    """Fixed-width table; columns in `left` (default: first and last) left-aligned, the rest right-aligned."""
    def cell(v):
        return "-" if v is None else f"{v:,}" if isinstance(v, int) else str(v)
    cells = [list(map(str, headers))] + [[cell(v) for v in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    left = {0, len(headers) - 1} if left is None else set(left)
    for r in cells:
        out("  " + "  ".join(v.ljust(w) if i in left else v.rjust(w)
                             for i, (v, w) in enumerate(zip(r, widths))).rstrip())


def section(title):
    out()
    out("=" * 78)
    out(title)
    out("=" * 78)


def per_file(con, units):
    """(a) + (b): one line per expected file."""
    section("(a)+(b) PER FILE: field count vs layout, rows loaded, line reconciliation")
    out("fields = detected/expected. lines = lines in the unzipped file. unaccounted = lines - rows - rejected.")
    flags = []
    logged = {(r[0], r[1]): r[2:] for r in con.execute(
        "SELECT table_name, source_file, status, delimiter, detected_field_count, layout_field_count, "
        "sample_lines_other_field_count, line_count, rows_loaded, rejected_lines FROM load_log").fetchall()}
    for table in TABLES:
        in_table = dict(con.execute(f"SELECT source_file, count(*) FROM {table} GROUP BY 1").fetchall())
        mine = sorted((u for u in units if u.table == table), key=lambda u: u.source_file)
        rows = []
        for u in mine:
            n_rows = in_table.get(u.source_file, 0)
            entry = logged.get(u.key)
            if entry is None:
                rows.append([u.source_file, "-", "-", None, n_rows, None, None, "NOT LOADED"])
                flags.append(f"{u.label}: not loaded")
                continue
            status, delim, detected, expected, sample_other, lines, loaded, rejected = entry
            f = []
            if status != "complete":
                f.append(status.upper())
            if delim != EXPECTED_DELIMITER or detected != expected:
                f.append("FIELD COUNT MISMATCH")
            unaccounted = None
            if status == "complete":
                unaccounted = lines - n_rows - rejected
                if n_rows != loaded:
                    f.append("ROWS != LOAD_LOG")
                if unaccounted:
                    f.append("LINES UNACCOUNTED")
                if rejected:
                    f.append("HAS REJECTS")
            if sample_other:
                f.append(f"{sample_other:,} SAMPLED LINES WITH OTHER FIELD COUNT")
            rows.append([u.source_file, show_delimiter(delim), f"{detected}/{expected}", lines, n_rows,
                         rejected, unaccounted, ", ".join(f) or "ok"])
            if f:
                flags.append(f"{u.label}: {', '.join(f)}")
        out()
        out(f"--- {table}: {len(mine)} files expected, "
            f"{sum(1 for r in rows if r[-1] != 'NOT LOADED')} logged, {sum(in_table.values()):,} rows in table")
        out_table(["file", "delim", "fields", "lines", "rows", "rejected", "unaccounted", "flags"], rows)
        extra = sorted(set(in_table) - {u.source_file for u in mine})
        if extra:
            out(f"  FLAG: rows with source_file values not matching any zip: {', '.join(extra)}")
            flags.append(f"{table}: rows with unexpected source_file values")
    return flags


def freddie_ids(con, all_loaded):
    """(c) Freddie loan ID cross-check, counts only."""
    section("(c) FREDDIE LOAN IDs: origination vs performance (counts only)")
    if not all_loaded:
        out("NOTE: not every Freddie file is loaded yet, so these counts are partial.")
    lid = FREDDIE_LOAN_ID
    o_rows, o_null, o_multi = con.execute(f"""
        SELECT count(*), count(*) FILTER (WHERE {lid} IS NULL),
               (SELECT count(*) FROM (SELECT {lid} FROM freddie_orig WHERE {lid} IS NOT NULL
                                      GROUP BY 1 HAVING count(*) > 1))
        FROM freddie_orig""").fetchone()
    p_rows, p_null = con.execute(
        f"SELECT count(*), count(*) FILTER (WHERE {lid} IS NULL) FROM freddie_perf").fetchone()
    o_ids, p_ids, p_not_o, o_not_p = con.execute(f"""
        WITH o AS MATERIALIZED (SELECT DISTINCT {lid} AS id FROM freddie_orig WHERE {lid} IS NOT NULL),
             p AS MATERIALIZED (SELECT DISTINCT {lid} AS id FROM freddie_perf WHERE {lid} IS NOT NULL)
        SELECT (SELECT count(*) FROM o), (SELECT count(*) FROM p),
               (SELECT count(*) FROM p WHERE NOT EXISTS (SELECT 1 FROM o WHERE o.id = p.id)),
               (SELECT count(*) FROM o WHERE NOT EXISTS (SELECT 1 FROM p WHERE p.id = o.id))""").fetchone()
    checks = [
        ("origination rows", o_rows, False),
        ("origination distinct loan IDs", o_ids, False),
        ("origination rows with no loan ID", o_null, True),
        ("origination loan IDs on more than one row", o_multi, True),
        ("performance rows", p_rows, False),
        ("performance distinct loan IDs", p_ids, False),
        ("performance rows with no loan ID", p_null, True),
        ("loan IDs in performance but NOT in origination", p_not_o, True),
        ("loan IDs in origination but NOT in performance", o_not_p, True),
    ]
    out_table(["measure", "count", "flag"],
              [[name, n, "FLAG" if flag_if_nonzero and n else ""] for name, n, flag_if_nonzero in checks])
    return [f"freddie IDs: {name} = {n:,}" for name, n, flag_if_nonzero in checks if flag_if_nonzero and n]


def rejects(con):
    """(d) rejected rows, counts by file and error type (rejected text is never read)."""
    section("(d) REJECTED ROWS (failed to parse), by file and error type")
    rows = con.execute("""
        SELECT table_name, source_file, error_type, count(DISTINCT line_byte_position) AS lines, count(*) AS errors
        FROM raw_rejects GROUP BY ALL ORDER BY ALL""").fetchall()
    if not rows:
        out("No rejected rows in any file.")
        return []
    out_table(["table / file", "error type", "lines", "errors"],
              [[f"{t} {s}", e, n, k] for t, s, e, n, k in rows], left=(0, 1))
    out()
    out("Totals by table and error type:")
    totals = con.execute("""
        SELECT table_name, error_type, count(DISTINCT (source_file, line_byte_position)), count(*)
        FROM raw_rejects GROUP BY ALL ORDER BY ALL""").fetchall()
    out_table(["table", "error type", "lines", "errors"], totals, left=(0, 1))
    return [f"{t}: {n:,} rejected lines ({e})" for t, e, n, k in totals]


def column_alignment(con):
    """Extra: entirely empty columns; Fannie NA-for-SF columns that hold data."""
    section("COLUMN ALIGNMENT: non-empty value counts per column (column names only)")
    flags = []
    for table in TABLES:
        layout = con.execute("SELECT column_name, sf_loan_performance_populated FROM layout_columns "
                             "WHERE table_name = ? ORDER BY position", [table]).fetchall()
        names = [n for n, _ in layout]
        counts = con.execute(f"SELECT count(*), {', '.join(f'count({quote_ident(n)})' for n in names)} "
                             f"FROM {table}").fetchone()
        total, filled = counts[0], dict(zip(names, counts[1:]))
        out()
        if total == 0:
            out(f"--- {table}: no rows loaded")
            continue
        empty = [n for n in names if filled[n] == 0]
        out(f"--- {table}: {len(names)} layout columns, {len(empty)} entirely empty")
        if empty:
            out("  entirely empty: " + ", ".join(empty))
        if table == "fannie_raw":
            sf = dict(layout)
            na_with_data = [n for n in names if sf[n] is False and filled[n] > 0]
            sf_empty = [n for n in names if sf[n] and filled[n] == 0]
            out(f"  marked NA for SF Loan Performance: {sum(1 for n in names if sf[n] is False)} columns; "
                f"of these, {len(na_with_data)} contain data" + (" <-- FLAG" if na_with_data else ""))
            for n in na_with_data:
                out(f"    {n}: {filled[n]:,} non-empty rows")
            out(f"  marked populated for SF Loan Performance but entirely empty: {len(sf_empty)}"
                + (": " + ", ".join(sf_empty) if sf_empty else ""))
            if na_with_data:
                flags.append(f"fannie_raw: {len(na_with_data)} NA-for-SF columns contain data (possible misalignment)")
    return flags


def database_size(con):
    """(e)"""
    section("(e) DATABASE SIZE")
    cur = con.execute("PRAGMA database_size")
    info = dict(zip([d[0] for d in cur.description], cur.fetchone()))
    out(f"{DB_PATH.name}: {gb(db_size())} on disk")
    out(f"DuckDB: {info['used_blocks']:,} used blocks, {info['free_blocks']:,} free blocks, "
        f"block size {info['block_size']:,} bytes")
    for table in TABLES:
        out(f"  {table}: {con.execute(f'SELECT count(*) FROM {table}').fetchone()[0]:,} rows")
    return []


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    LOG_DIR.mkdir(exist_ok=True)
    if not DB_PATH.exists():
        print(f"No database at {DB_PATH}; run load_raw.py first.")
        return 1
    started = time.time()
    out(f"check_raw.py  {datetime.now():%Y-%m-%d %H:%M}  database {DB_PATH.name}")
    out("Summaries only: no row-level values, no loan IDs, no rejected-row text.")
    con = connect(memory_limit="6GB", read_only=True)
    code = 0
    try:
        units = list_units()
        logged = {tuple(r) for r in con.execute(
            "SELECT table_name, source_file FROM load_log WHERE status = 'complete'").fetchall()}
        freddie_done = all(u.key in logged for u in units if u.source == "freddie")
        flags = per_file(con, units)
        flags += freddie_ids(con, freddie_done)
        flags += rejects(con)
        flags += column_alignment(con)
        flags += database_size(con)
        section(f"SUMMARY: {len(flags)} flag(s)")
        for f in flags:
            out(f"  - {f}")
        if not flags:
            out("  No problems found.")
        out(f"(check took {time.time() - started:,.0f} s)")
    except Exception as exc:
        report_error(exc, ERROR_DETAIL_PATH, out)
        code = 1
    finally:
        con.close()
        REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
        print(f"Report saved to {REPORT_PATH}")
    return code


if __name__ == "__main__":
    sys.exit(main())
