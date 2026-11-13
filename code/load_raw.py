"""Load the raw Freddie Mac and Fannie Mae loan-level files into DuckDB, exactly as provided.

What it does
  * Reads column names and order from the layout documents:
      Freddie: freddie_mac_data/file_layout_july_2026.xlsx, cross-checked against
               freddie_mac_data/file_headers_july_2026.zip (the run stops if they disagree)
      Fannie:  fannie_mae_data/crt-file-layout-and-glossary.xlsx, sheet "Combined Glossary",
               positions 1-113 in order (position 114 is not in the files; see
               FANNIE_FIELDS_IN_FILES). The SF Loan Performance column is kept in
               layout_columns as sf_loan_performance_populated.
  * Loads Freddie origination -> freddie_orig, Freddie performance -> freddie_perf,
    Fannie combined files -> fannie_raw. Every column is text (VARCHAR); empty fields are NULL.
    No recoding, cleaning or filtering. Each row gets source_file = quarter label (e.g. 2014Q1):
    origination quarter for Freddie, acquisition quarter for Fannie.
  * One file at a time: extract one file from its zip into _unzip_tmp/, load it, delete it.
    The original zip files are only read, never modified.
  * Before loading a file, detects its delimiter and field count (from a sample of lines) and
    counts its lines; these go in load_log. If the delimiter or field count does not match the
    layout, nothing is loaded from that file and the run stops.
  * Unreadable rows go to DuckDB's rejected-rows tables and are copied to raw_rejects.
  * Resumable: each file is loaded in one transaction together with its load_log entry, so a file
    is either fully loaded and logged or not loaded at all. Re-running skips completed files.
  * Size check: the first Freddie quarter (orig + perf) and first Fannie quarter load first. Then,
    per source, (database growth / text bytes loaded) x (text bytes still to load) projects the
    final database size; the run stops if it exceeds --size-limit-gb. Re-checked after every file.
  * Stops before extracting a file if free disk would fall below the file size + 20 GB.

Output is summaries only (counts, sizes, timings). Row-level values are never printed.
Exit codes: 0 finished, 2 layout mismatch, 3 size limit, 4 not enough free disk.

Usage (from the repository root):
    python code/load_raw.py [--size-limit-gb 150] [--ignore-size-limit]
                            [--memory-limit 6GB] [--threads N]
"""

import argparse
import re
import shutil
import sys
import time
import traceback
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import openpyxl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FREDDIE_DIR = PROJECT_ROOT / "freddie_mac_data"
FANNIE_DIR = PROJECT_ROOT / "fannie_mae_data"
DB_PATH = PROJECT_ROOT / "freddie_fannie_raw.duckdb"
TMP_DIR = PROJECT_ROOT / "_unzip_tmp"
LOG_DIR = PROJECT_ROOT / "logs"
LOAD_LOG_PATH = LOG_DIR / "load_raw.log"
LOAD_ERROR_DETAIL_PATH = LOG_DIR / "load_raw_error_detail.txt"

FREDDIE_LAYOUT_XLSX = FREDDIE_DIR / "file_layout_july_2026.xlsx"
FREDDIE_HEADERS_ZIP = FREDDIE_DIR / "file_headers_july_2026.zip"
FANNIE_LAYOUT_XLSX = FANNIE_DIR / "crt-file-layout-and-glossary.xlsx"

# table -> (sheet in FREDDIE_LAYOUT_XLSX, header file inside FREDDIE_HEADERS_ZIP)
FREDDIE_LAYOUTS = {
    "freddie_orig": ("Origination Data File", "origination_data_file_header.txt"),
    "freddie_perf": ("Monthly Performance Data File", "performance_data_file_header.txt"),
}
FANNIE_SHEET = "Combined Glossary"
FANNIE_SF_HEADER = "Single-Family (SF) Loan Performance"
# The Fannie files hold 113 fields: glossary positions 1-113. Position 114,
# "Origination VantageScore 4.0", is not in them: the glossary marks it as populated only from a
# later activity period than these files cover.
# The field-count check stays strict, so a later 114-field release stops the load.
FANNIE_FIELDS_IN_FILES = 113
FANNIE_FIELDS_NOT_IN_FILES = ["origination_vantagescore_4_0"]
TABLES = ["freddie_orig", "freddie_perf", "fannie_raw"]

EXPECTED_DELIMITER = "|"
CANDIDATE_DELIMITERS = ["|", ",", "\t", ";"]  # ties go to the first
SNIFF_LINES = 20_000
COPY_CHUNK_BYTES = 16 * 1024 * 1024
DISK_HEADROOM_BYTES = 20 * 10**9
# DuckDB's rejected-rows tables are temporary (per connection); copied to raw_rejects per file.
REJECTS_TABLE = "csv_rejects_tmp"
REJECTS_SCAN = "csv_rejects_scan_tmp"

LOAD_LOG_COLUMNS = [
    "table_name", "source_file", "zip_file", "member_name", "status",
    "delimiter", "detected_field_count", "layout_field_count",
    "sample_lines", "sample_lines_other_field_count",
    "line_count", "rows_loaded", "rejected_lines", "uncompressed_bytes",
    "db_bytes_before", "db_bytes_after", "unzip_seconds", "load_seconds",
]


class LayoutError(Exception):
    pass


class StopRun(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class Column:
    position: int
    name: str                 # column name in DuckDB (mechanical snake_case of layout_name)
    layout_name: str          # attribute name as written in the layout document
    layout_type: str
    layout_max_length: str
    sf_populated: bool | None  # Fannie only: marked populated in the SF Loan Performance dataset


@dataclass(frozen=True)
class Layout:
    columns: list
    source_document: str


@dataclass(frozen=True)
class Unit:
    """One file to load."""
    table: str
    source_file: str       # quarter label, e.g. 2014Q1
    zip_path: Path         # original zip (read only)
    inner_zip: str | None  # Freddie: quarterly zip stored inside the yearly zip
    member: str            # the .txt / .csv file that gets loaded
    size: int              # uncompressed bytes

    @property
    def source(self):
        return self.table.split("_")[0]  # 'freddie' or 'fannie'

    @property
    def key(self):
        return (self.table, self.source_file)

    @property
    def label(self):
        return f"{self.table} {self.source_file}"

    @property
    def zip_label(self):
        return f"{self.zip_path.name}/{self.inner_zip}" if self.inner_zip else self.zip_path.name


# ---------------------------------------------------------------- small helpers

def log(msg=""):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(LOAD_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def gb(n):
    return f"{n / 1e9:,.2f} GB"


def hms(seconds):
    s = int(round(seconds))
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


def show_delimiter(d):
    return {"|": "'|'", ",": "','", "\t": "TAB", ";": "';'"}.get(d, "none found" if d is None else repr(d))


def snake_case(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def quote_ident(name):
    return '"' + name.replace('"', '""') + '"'


def db_size():
    wal = DB_PATH.with_name(DB_PATH.name + ".wal")
    return sum(p.stat().st_size for p in (DB_PATH, wal) if p.exists())


def report_error(exc, detail_path, emit):
    """Show an unexpected error without risking raw data on screen: DuckDB messages can quote raw
    rows, so for those only the error type is shown; the full traceback goes to detail_path."""
    with open(detail_path, "w", encoding="utf-8") as f:
        f.write("MAY CONTAIN RAW DATA ROWS - do not paste or share this file.\n\n")
        traceback.print_exception(exc, file=f)
    shown = type(exc).__name__ if isinstance(exc, duckdb.Error) else f"{type(exc).__name__}: {exc}"
    emit(f"ERROR: {shown}")
    emit(f"       Full details in {detail_path} (this file contains loan-level data; do not share it).")


def connect(memory_limit=None, threads=None, read_only=False):
    con = duckdb.connect(str(DB_PATH), read_only=read_only)
    if memory_limit:
        con.execute(f"SET memory_limit = '{memory_limit}'")
    if threads:
        con.execute(f"SET threads = {int(threads)}")
    con.execute("SET preserve_insertion_order = false")
    return con


# ---------------------------------------------------------------- layouts

def _clean(value):
    return re.sub(r"\s+", " ", str(value)).strip() if value is not None else ""


def _sheet_records(ws, wanted):
    """Rows under the header row (first cell 'Field Position'), as dicts of the wanted headers.
    Stops at the first row without a field position."""
    rows = ws.iter_rows(values_only=True)
    for row in rows:
        cells = [_clean(v) for v in row]
        if cells and cells[0] == "Field Position":
            missing = [h for h in wanted if h not in cells]
            if missing:
                raise LayoutError(f"sheet '{ws.title}' is missing header(s): {missing}")
            idx = {h: cells.index(h) for h in wanted}
            break
    else:
        raise LayoutError(f"sheet '{ws.title}' has no 'Field Position' header row")
    records = []
    for row in rows:
        cells = [_clean(v) for v in row]
        if not cells or not cells[0]:
            break
        records.append({h: cells[i] if i < len(cells) else "" for h, i in idx.items()})
    return records


def _build_columns(records, name_key, type_key, length_key, where, sf_key=None):
    columns = []
    for expected, rec in enumerate(records, start=1):
        position = int(float(rec["Field Position"]))
        if position != expected:
            raise LayoutError(f"{where}: field positions are not 1..n (found {position} where {expected} expected)")
        sf = None
        if sf_key:
            flag = rec[sf_key]
            if flag not in ("\u221a", "NA"):
                raise LayoutError(f"{where}: unexpected '{sf_key}' value at position {position}")
            sf = flag == "\u221a"
        columns.append(Column(position, snake_case(rec[name_key]), rec[name_key],
                              rec[type_key], rec[length_key], sf))
    names = [c.name for c in columns]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if not columns or dupes or "source_file" in names:
        raise LayoutError(f"{where}: empty layout or duplicate column names {dupes}")
    return columns


def read_layouts():
    """Column list per table, from the layout documents."""
    layouts = {}
    wb = openpyxl.load_workbook(FREDDIE_LAYOUT_XLSX, read_only=True, data_only=True)
    with zipfile.ZipFile(FREDDIE_HEADERS_ZIP) as headers:
        for table, (sheet, header_file) in FREDDIE_LAYOUTS.items():
            where = f"{FREDDIE_LAYOUT_XLSX.name} / {sheet}"
            records = _sheet_records(wb[sheet], ["Field Position", "Attribute Name", "Data Type & Format", "Max Length"])
            columns = _build_columns(records, "Attribute Name", "Data Type & Format", "Max Length", where)
            header_names = [snake_case(h) for h in headers.read(header_file).decode("utf-8-sig").strip().split("|")]
            if header_names != [c.name for c in columns]:
                raise LayoutError(f"{table}: {where} and {FREDDIE_HEADERS_ZIP.name}/{header_file} "
                                  f"disagree on column names or order")
            layouts[table] = Layout(columns, where)
    wb.close()

    wb = openpyxl.load_workbook(FANNIE_LAYOUT_XLSX, read_only=True, data_only=True)
    where = f"{FANNIE_LAYOUT_XLSX.name} / {FANNIE_SHEET}"
    records = _sheet_records(wb[FANNIE_SHEET], ["Field Position", "Field Name", FANNIE_SF_HEADER, "Type", "Max Length"])
    columns = _build_columns(records, "Field Name", "Type", "Max Length", where, FANNIE_SF_HEADER)
    if [c.name for c in columns[FANNIE_FIELDS_IN_FILES:]] != FANNIE_FIELDS_NOT_IN_FILES:
        raise LayoutError(f"{where}: positions after {FANNIE_FIELDS_IN_FILES} are not {FANNIE_FIELDS_NOT_IN_FILES}; "
                          f"the glossary changed, so review FANNIE_FIELDS_IN_FILES")
    layouts["fannie_raw"] = Layout(columns[:FANNIE_FIELDS_IN_FILES], f"{where}, positions 1-{FANNIE_FIELDS_IN_FILES}")
    wb.close()
    return layouts


# ---------------------------------------------------------------- files to load

def list_units():
    """Every file to load, found from the zip directories (no file contents are read)."""
    units = []
    for z in sorted(FREDDIE_DIR.glob("historical_data_[0-9][0-9][0-9][0-9].zip")):
        with zipfile.ZipFile(z) as outer:
            for info in sorted((i for i in outer.infolist() if not i.is_dir()), key=lambda i: i.filename):
                m = re.fullmatch(r"historical_data_(\d{4}Q[1-4])\.zip", info.filename)
                if not m:
                    raise LayoutError(f"unexpected file {info.filename} inside {z.name}")
                quarter = m.group(1)
                found = {}
                with outer.open(info) as f, zipfile.ZipFile(f) as inner:
                    for i in inner.infolist():
                        mm = re.fullmatch(r"(orig|perf)_(\d{4}Q[1-4])\.txt", i.filename, re.IGNORECASE)
                        if mm and mm.group(2).upper() == quarter:
                            found[mm.group(1).lower()] = i
                if set(found) != {"orig", "perf"}:
                    raise LayoutError(f"{z.name}/{info.filename} does not hold both orig_{quarter}.txt and perf_{quarter}.txt")
                for kind in ("orig", "perf"):
                    units.append(Unit(f"freddie_{kind}", quarter, z, info.filename,
                                      found[kind].filename, found[kind].file_size))
    for z in sorted(FANNIE_DIR.glob("[0-9][0-9][0-9][0-9]Q[1-4].zip")):
        with zipfile.ZipFile(z) as zf:
            members = [i for i in zf.infolist() if not i.is_dir()]
        if len(members) != 1:
            raise LayoutError(f"{z.name} holds {len(members)} files; expected 1")
        units.append(Unit("fannie_raw", z.stem, z, None, members[0].filename, members[0].file_size))
    keys = [u.key for u in units]
    if len(keys) != len(set(keys)):
        raise LayoutError("the same table/quarter appears in more than one zip")
    return units


def load_order(units):
    """First Freddie quarter (orig + perf) and first Fannie quarter first, for the size check."""
    first = {}
    for u in units:
        first[u.source] = min(first.get(u.source, u.source_file), u.source_file)
    pilot = [u for u in units if u.source_file == first[u.source]]
    return pilot + [u for u in units if u.source_file != first[u.source]]


def _copy_member(zf, member, dest):
    lines, last = 0, b""
    with zf.open(member) as src, open(dest, "wb") as out:
        while chunk := src.read(COPY_CHUNK_BYTES):
            lines += chunk.count(b"\n")
            last = chunk[-1:]
            out.write(chunk)
    return lines + (1 if last not in (b"", b"\n") else 0)


def extract(unit, dest):
    """Copy one file out of its zip (zipfile verifies the CRC); return its line count."""
    with zipfile.ZipFile(unit.zip_path) as outer:
        if unit.inner_zip:
            with outer.open(unit.inner_zip) as f, zipfile.ZipFile(f) as inner:
                return _copy_member(inner, unit.member, dest)
        return _copy_member(outer, unit.member, dest)


def sniff(path):
    """Delimiter and field count from the first SNIFF_LINES lines: the candidate delimiter whose
    per-line count is most often the same (nonzero) number wins.
    Returns (delimiter, field_count, sampled_lines, sampled_lines_with_another_field_count)."""
    counts = {d: Counter() for d in CANDIDATE_DELIMITERS}
    sampled = 0
    with open(path, "rb") as f:
        for raw in f:
            line = raw.rstrip(b"\r\n")
            for d in CANDIDATE_DELIMITERS:
                counts[d][line.count(d.encode())] += 1
            sampled += 1
            if sampled >= SNIFF_LINES:
                break
    best = None
    for d in CANDIDATE_DELIMITERS:
        if counts[d]:
            n, freq = counts[d].most_common(1)[0]
            if n > 0 and (best is None or freq > best[2]):
                best = (d, n + 1, freq)
    if best is None:
        return None, 1, sampled, 0
    return best[0], best[1], sampled, sampled - best[2]


# ---------------------------------------------------------------- database

def setup_tables(con, layouts):
    create_log_tables(con)
    con.execute("""
        CREATE OR REPLACE TABLE layout_columns (
            table_name VARCHAR, position INTEGER, column_name VARCHAR, layout_name VARCHAR,
            layout_type VARCHAR, layout_max_length VARCHAR,
            sf_loan_performance_populated BOOLEAN,  -- Fannie only
            source_document VARCHAR)""")
    for table, layout in layouts.items():
        con.executemany(
            "INSERT INTO layout_columns VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(table, c.position, c.name, c.layout_name, c.layout_type, c.layout_max_length,
              c.sf_populated, layout.source_document) for c in layout.columns])
        col_defs = ", ".join(f"{quote_ident(c.name)} VARCHAR" for c in layout.columns)
        con.execute(f"CREATE TABLE IF NOT EXISTS {table} ({col_defs}, source_file VARCHAR)")
        existing = [r[0] for r in con.execute(
            "SELECT column_name FROM duckdb_columns() WHERE schema_name = 'main' AND table_name = ? "
            "ORDER BY column_index", [table]).fetchall()]
        if existing != [c.name for c in layout.columns] + ["source_file"]:
            if con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]:
                raise LayoutError(f"existing table {table} has rows and does not match the layout documents")
            # Empty table built from an earlier layout: rebuild it and drop its stale load attempts.
            con.execute(f"DROP TABLE {table}")
            con.execute(f"CREATE TABLE {table} ({col_defs}, source_file VARCHAR)")
            con.execute("DELETE FROM load_log WHERE table_name = ?", [table])
            con.execute("DELETE FROM raw_rejects WHERE table_name = ?", [table])
            log(f"rebuilt empty table {table} with {len(existing) - 1} -> {len(layout.columns)} layout columns; "
                f"cleared its earlier load_log entries")


def create_log_tables(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS load_log (
            table_name VARCHAR, source_file VARCHAR, zip_file VARCHAR, member_name VARCHAR,
            status VARCHAR,  -- 'complete' or 'layout_mismatch'
            delimiter VARCHAR, detected_field_count INTEGER, layout_field_count INTEGER,
            sample_lines BIGINT, sample_lines_other_field_count BIGINT,
            line_count BIGINT, rows_loaded BIGINT, rejected_lines BIGINT, uncompressed_bytes BIGINT,
            db_bytes_before BIGINT, db_bytes_after BIGINT, unzip_seconds DOUBLE, load_seconds DOUBLE,
            logged_at TIMESTAMP)""")
    # csv_line and error_message hold raw row text: never select them for output.
    con.execute("""
        CREATE TABLE IF NOT EXISTS raw_rejects (
            table_name VARCHAR, source_file VARCHAR, line BIGINT, line_byte_position BIGINT,
            byte_position BIGINT, column_idx BIGINT, column_name VARCHAR, error_type VARCHAR,
            csv_line VARCHAR, error_message VARCHAR)""")


def completed_keys(con):
    return set(con.execute("SELECT table_name, source_file FROM load_log WHERE status = 'complete'").fetchall())


def write_load_log(con, row):
    con.execute("DELETE FROM load_log WHERE table_name = ? AND source_file = ?",
                [row["table_name"], row["source_file"]])
    con.execute(f"INSERT INTO load_log ({', '.join(LOAD_LOG_COLUMNS)}, logged_at) "
                f"VALUES ({', '.join(['?'] * len(LOAD_LOG_COLUMNS))}, current_timestamp)",
                [row.get(c) for c in LOAD_LOG_COLUMNS])


def projection(con, units):
    """Projected final database bytes, or None until every source has a loaded file."""
    loaded = {}
    for table, size, growth in con.execute(
            "SELECT table_name, uncompressed_bytes, db_bytes_after - db_bytes_before FROM load_log "
            "WHERE status = 'complete' AND db_bytes_after IS NOT NULL").fetchall():
        s = loaded.setdefault(table.split("_")[0], [0, 0])
        s[0] += size
        s[1] += growth
    complete = completed_keys(con)
    remaining = Counter()
    for u in units:
        if u.key not in complete:
            remaining[u.source] += u.size
    sources = {u.source for u in units}
    if any(loaded.get(s, [0, 0])[0] == 0 for s in sources):
        return None
    return db_size() + sum(loaded[s][1] / loaded[s][0] * remaining[s] for s in sources)


def check_size(con, units, args, next_label):
    projected = projection(con, units)
    if projected is not None and not args.ignore_size_limit and projected > args.size_limit_gb * 1e9:
        log(f"STOP: projected final database size {gb(projected)} exceeds the {args.size_limit_gb:g} GB limit.")
        log(f"      Stopped before {next_label}. Current database size {gb(db_size())}.")
        log("      To continue anyway, re-run with --ignore-size-limit (completed files are kept).")
        raise StopRun(3)
    return projected


def check_free_disk(unit):
    free = shutil.disk_usage(PROJECT_ROOT).free
    if free < unit.size + DISK_HEADROOM_BYTES:
        log(f"STOP: {unit.label} needs {gb(unit.size)} unzipped plus {gb(DISK_HEADROOM_BYTES)} headroom; "
            f"only {gb(free)} free. Completed files are kept; free space and re-run.")
        raise StopRun(4)


def load_one(con, unit, columns):
    """Extract, check, load and log one file; return its load_log row."""
    dest = TMP_DIR / Path(unit.member).name
    row = dict(table_name=unit.table, source_file=unit.source_file, zip_file=unit.zip_label,
               member_name=unit.member, layout_field_count=len(columns), uncompressed_bytes=unit.size)
    try:
        t0 = time.time()
        row["line_count"] = extract(unit, dest)
        row["unzip_seconds"] = time.time() - t0
        (row["delimiter"], row["detected_field_count"], row["sample_lines"],
         row["sample_lines_other_field_count"]) = sniff(dest)

        if row["delimiter"] != EXPECTED_DELIMITER or row["detected_field_count"] != len(columns):
            row["status"] = "layout_mismatch"
            write_load_log(con, row)
            log(f"STOP: {unit.label} ({unit.member}): detected delimiter {show_delimiter(row['delimiter'])} with "
                f"{row['detected_field_count']} fields; layout expects '|' with {len(columns)}. "
                f"Nothing loaded from this file.")
            raise StopRun(2)

        t1 = time.time()
        row["db_bytes_before"] = db_size()
        col_spec = "{" + ", ".join(f"'{c.name}': 'VARCHAR'" for c in columns) + "}"
        path_sql = str(dest).replace("'", "''")
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute(f"DELETE FROM {unit.table} WHERE source_file = ?", [unit.source_file])
            con.execute("DELETE FROM raw_rejects WHERE table_name = ? AND source_file = ?", list(unit.key))
            con.execute(f"DROP TABLE IF EXISTS temp.{REJECTS_TABLE}")
            con.execute(f"DROP TABLE IF EXISTS temp.{REJECTS_SCAN}")
            row["rows_loaded"] = con.execute(f"""
                INSERT INTO {unit.table} BY NAME
                SELECT *, ? AS source_file
                FROM read_csv('{path_sql}', delim = '|', header = false, quote = '', escape = '',
                              auto_detect = false, columns = {col_spec},
                              store_rejects = true, rejects_table = '{REJECTS_TABLE}',
                              rejects_scan = '{REJECTS_SCAN}')""", [unit.source_file]).fetchone()[0]
            con.execute(f"""
                INSERT INTO raw_rejects
                SELECT ?, ?, line, line_byte_position, byte_position, column_idx, column_name,
                       CAST(error_type AS VARCHAR), csv_line, error_message
                FROM {REJECTS_TABLE}""", list(unit.key))
            row["rejected_lines"] = con.execute(
                f"SELECT count(DISTINCT line_byte_position) FROM {REJECTS_TABLE}").fetchone()[0]
            row["status"] = "complete"
            row["load_seconds"] = time.time() - t1
            write_load_log(con, row)
            con.execute("COMMIT")
        except BaseException:
            try:
                con.execute("ROLLBACK")
            except duckdb.Error:
                pass
            raise
        con.execute("CHECKPOINT")
        row["db_bytes_after"] = db_size()
        con.execute("UPDATE load_log SET db_bytes_after = ? WHERE table_name = ? AND source_file = ?",
                    [row["db_bytes_after"], *unit.key])
        return row
    finally:
        dest.unlink(missing_ok=True)


# ---------------------------------------------------------------- main

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size-limit-gb", type=float, default=150.0,
                   help="stop if the projected final database size exceeds this (default 150)")
    p.add_argument("--ignore-size-limit", action="store_true", help="keep loading even if the projection exceeds the limit")
    p.add_argument("--memory-limit", default="6GB", help="DuckDB memory limit (default 6GB)")
    p.add_argument("--threads", type=int, default=None, help="DuckDB threads (default: all cores)")
    args = p.parse_args()
    if not re.fullmatch(r"\d+(\.\d+)?\s*[KMGT]i?B", args.memory_limit, re.IGNORECASE):
        p.error("--memory-limit must look like 6GB or 4096MB")
    return args


def main():
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    LOG_DIR.mkdir(exist_ok=True)
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)  # copies left by an interrupted run (this script's own temp folder)
    TMP_DIR.mkdir()
    started = time.time()
    log("=" * 78)
    log(f"load_raw.py started. Database: {DB_PATH}")
    con, units, exit_code = None, [], 0
    try:
        layouts = read_layouts()
        for table, layout in layouts.items():
            log(f"layout {table}: {len(layout.columns)} fields from {layout.source_document}")
        units = load_order(list_units())
        for table in TABLES:
            mine = [u for u in units if u.table == table]
            log(f"files for {table}: {len(mine)} ({gb(sum(u.size for u in mine))} unzipped)")

        con = connect(args.memory_limit, args.threads)
        setup_tables(con, layouts)
        con.execute("CHECKPOINT")
        complete = completed_keys(con)
        todo = [u for u in units if u.key not in complete]
        log(f"{len(units) - len(todo)} of {len(units)} files already loaded; {len(todo)} to load. "
            f"Size limit {'ignored' if args.ignore_size_limit else f'{args.size_limit_gb:g} GB'}.")
        for unit in todo:
            check_size(con, units, args, unit.label)
            check_free_disk(unit)
            n = units.index(unit) + 1
            log(f"[{n:3d}/{len(units)}] {unit.label}: extracting {unit.member} ({gb(unit.size)}) from {unit.zip_label}")
            row = load_one(con, unit, layouts[unit.table].columns)
            unaccounted = row["line_count"] - row["rows_loaded"] - row["rejected_lines"]
            projected = projection(con, units)
            log(f"          lines {row['line_count']:,} = loaded {row['rows_loaded']:,} + rejected "
                f"{row['rejected_lines']:,}{'' if unaccounted == 0 else f'  <-- {unaccounted:,} LINES UNACCOUNTED'}"
                f" | delimiter {show_delimiter(row['delimiter'])}, fields {row['detected_field_count']}/{len(layouts[unit.table].columns)}")
            log(f"          unzip {hms(row['unzip_seconds'])}, load {hms(row['load_seconds'])} | "
                f"db {gb(row['db_bytes_after'])} ({(row['db_bytes_after'] - row['db_bytes_before']) / 1e9:+,.2f} GB) | "
                f"free disk {gb(shutil.disk_usage(PROJECT_ROOT).free)} | projected final db "
                f"{'n/a until both sources have a file loaded' if projected is None else gb(projected)}")
        log(f"All {len(units)} files loaded.")
    except StopRun as stop:
        exit_code = stop.code
    except KeyboardInterrupt:
        log("Interrupted. The file in progress was rolled back; completed files are kept. Re-run to resume.")
        exit_code = 130
    except Exception as exc:
        report_error(exc, LOAD_ERROR_DETAIL_PATH, log)
        exit_code = 1
    finally:
        done = "?"
        if con is not None:
            try:
                done = len(completed_keys(con))
            except duckdb.Error:
                pass
            con.close()
        shutil.rmtree(TMP_DIR, ignore_errors=True)
        log(f"Finished with exit code {exit_code}: {done} of {len(units)} files loaded, "
            f"database {gb(db_size())}, elapsed {hms(time.time() - started)}.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
