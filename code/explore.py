"""Explore the training loans to set up Logistic Regression correctly. Summaries only.

Two questions, answered from kept loans labeled train (both agencies pooled; no validation or test loans):
  1. For each of the 17 model variables, does the bad rate change smoothly across its values, or curve or jump?
  2. Do any variables overlap so much that they would scramble Logistic Regression's odds ratios?
Descriptive only: nothing here chooses a model setting (missing values, bands, dropped variables are set in fit_models.py).

What it produces (from loan_dataset.duckdb, opened READ-ONLY)
  outputs/explore/bad_rate_<variable>.csv   one row per distinct value (numbers; rate spread rounded to 4 decimals)
                                            or per category (every documented value, 0 if absent), plus a
                                            missing row: value, loans, bad_loans, bad_rate_pct
  outputs/explore/bad_rate_<variable>.png   bad rate on the log-odds scale (95% Wilson intervals) over loan-count
                                            bars on a log scale; drawn from the masked CSV read back from disk
  outputs/explore/correlations.csv          every pair of the 8 numbers: Pearson, Spearman (average ranks), loans used
  outputs/explore/cramers_v.csv             every pair of the 9 categories: Cramer's V, loans used
  outputs/explore/correlation_ratio.csv     every number x category pair: correlation ratio (eta), loans used
  outputs/explore/vif_columns.csv           VIF of each column Logistic Regression would see (numbers + dummies)
  outputs/explore/vif_variables.csv         generalized VIF per variable (Fox & Monette) and GVIF^(1/(2 df))
  logs/explore_report.txt                   run info, row counts, both outputs, and a list of what looks like a
                                            jump or heavy overlap (descriptive)
  logs/explore_progress.log                 timestamps and the final SUCCESS / FAILURE line
  (--smoke: loan_dataset_smoke.duckdb -> outputs/explore_smoke/, logs/explore_report_smoke.txt)

Masking: every count from 1 to 10 (loans, bad or good) shows as "1-10"; a
rate is not shown when its bad or good count is 1-10; a row with 1-10 loans shows nothing but "1-10". A pair's
"loans used" shows exactly unless it is itself 1-10. The charts and the flag list read only the masked CSVs.
No loan IDs, no row-level output, no seller, servicer or location fields.

Overlap method: missing values are left out pair by pair (correlations, Cramer's V, eta); the VIF uses loans with
all 17 values present, and leaves out loans in a category value with 1-10 such loans (a near-empty dummy). The VIF
comes from one pass of covariances in DuckDB (covar_pop) and the inverse of the correlation matrix in numpy: no
row-level regression. Reference category = the most common value.

Usage (from the repository root):
    python code/explore.py [--smoke] [--threads 2] [--memory-limit 8GB]
Exit codes: 0 SUCCESS; 1 FAILURE (missing dataset, failed check or section); 130 interrupted (rerun).
"""

import argparse
import csv
import itertools
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset as bd  # noqa: E402  (reused: error detail, report sections, check lines)
import check_raw  # noqa: E402
import inventory as inv  # noqa: E402  (reused: progress log, report file, masking)
import load_raw  # noqa: E402
import model_core as core  # noqa: E402  (the variable list)
from check_raw import out, out_table, section  # noqa: E402
from inventory import StopRun, lit, m, mc, progress  # noqa: E402
from load_raw import LOG_DIR, PROJECT_ROOT, hms  # noqa: E402

SCRIPT_VERSION = "1"
MASK = "1-10"
MISSING = "(missing)"
SPREAD_DECIMALS = 4         # rate spread = rate (<= 3 decimals) - median (<= 4): rounded so float noise can't split a value
PRINT_ROWS = 150            # longer per-value tables are summarized in the report (the CSV has every row)
Z95 = 1.959963984540054
TOP_JUMPS = 10
CORR_HEAVY = 0.7            # |Pearson| or |Spearman|, and eta
V_HEAVY = 0.5
VIF_WATCH, VIF_SEVERE = 5.0, 10.0
PCT_TICKS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 80, 90, 95, 99]

# The 17 variables: column, label, kind, documented category values
# (defined in model_core.py).
VARIABLES = core.BASE_VARIABLES
NUMBERS = [v for v in VARIABLES if v[2] == "number"]
CATEGORIES = [v for v in VARIABLES if v[2] == "category"]
LABEL = {c: lab for c, lab, _, _ in VARIABLES}

WORK_ROOT = PROJECT_ROOT    # tests re-point these two before calling main()
LOGS = LOG_DIR
MODE = "full"
P = {}                      # paths of this mode: db, out, spill, report, progress, error
FAILS = bd.FAILS            # bd.check_line and bd.rsection record problems here
WRITTEN = []                # output files written by this run


def configure(mode):
    """Paths of this mode; load_raw, inventory and build_dataset helpers are re-pointed so they work on them."""
    global MODE
    MODE = mode
    sfx = "_smoke" if mode == "smoke" else ""
    P.clear()
    P.update(db=WORK_ROOT / f"loan_dataset{sfx}.duckdb", out=WORK_ROOT / "outputs" / f"explore{sfx}",
             spill=WORK_ROOT / f"explore_work{sfx}.tmp", report=LOGS / f"explore_report{sfx}.txt",
             progress=LOGS / "explore_progress.log", error=LOGS / "explore_error_detail.txt")
    load_raw.DB_PATH = P["db"]          # load_raw.connect opens this file (read-only, below)
    inv.REPORT_PATH = P["report"]
    inv.PROGRESS_PATH = P["progress"]
    bd.P["error"] = P["error"]          # bd.write_error_detail: stack frames only, never DuckDB messages


def open_dataset(args):
    """The dataset read-only (load_raw.connect); DuckDB spill goes to a capped folder of our own (as inv.open_db)."""
    con = load_raw.connect(args.memory_limit, args.threads, read_only=True)
    P["spill"].mkdir(exist_ok=True)
    con.execute(f"SET temp_directory = {lit(P['spill'])}")
    con.execute(f"SET max_temp_directory_size = '{inv.SPILL_CAP_GB}GB'")
    return con


def create_training_view(con):
    """Training loans only, both agencies; categories as text; rate spread rounded (SPREAD_DECIMALS)."""
    cols = []
    for c, _, kind, _ in VARIABLES:
        if kind == "category":
            cols.append(f"CAST({c} AS VARCHAR) AS {c}")
        elif c == "rate_spread":
            cols.append(f"round({c}, {SPREAD_DECIMALS}) AS {c}")
        else:
            cols.append(c)
    con.execute(f"CREATE OR REPLACE TEMP VIEW train AS SELECT agency, (outcome = 'bad')::INTEGER AS bad, "
                f"{', '.join(cols)} FROM kept_loans WHERE label = 'train'")


# ---------------------------------------------------------------- masking and formatting

def rate_shown(loans, bad):
    """A rate is shown only when neither its bad nor its good count is 1-10 (and there are loans)."""
    good = loans - bad
    return loans > 0 and not 1 <= bad <= 10 and not 1 <= good <= 10


def mask_row(value, loans, bad):
    """One per-value row as published: (value, loans, bad_loans, bad_rate_pct), all text.
    1-10 loans -> only '1-10'; a bad count of 1-10 -> '1-10'; rate blank unless rate_shown."""
    if 1 <= loans <= 10:
        return [value, MASK, "", ""]
    rate = f"{100 * bad / loans:.4f}" if rate_shown(loans, bad) else ""
    return [value, str(m(loans)), str(m(bad)), rate]


def num_text(v):
    """A number value as CSV text (no thousands separators; float() reads it back). A value that is exactly a float32
    (as the models see the numbers) prints as the shortest text that reads back to it: 99.97, not 99.97000122."""
    v = float(v)
    if v.is_integer():
        return str(int(v))
    if math.isfinite(v) and float(np.float32(v)) == v:
        return np.format_float_positional(np.float32(v), trim="-")
    return format(v, ".10g")


def commas(s):
    return f"{int(s):,}" if s.isdigit() else s


def pct_text(loans, bad):
    return f"{100 * bad / loans:.2f}%" if rate_shown(loans, bad) else "-"


def stat(x, d=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return "inf" if math.isinf(x) else f"{x:.{d}f}"


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    WRITTEN.append(path)


def read_masked(path):
    """A published CSV read back from disk: what the charts and the flag list are allowed to use."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- report: header and population

def rep_header(args, meta):
    out(f"explore.py v{SCRIPT_VERSION}  {datetime.now():%Y-%m-%d %H:%M}  mode {MODE}  dataset {P['db'].name} (read-only)")
    out("Summaries only: no loan IDs, no row-level values. Basic masking: every count from 1 to 10 (loans, bad or good) "
        "shows as '1-10'; a rate is not shown when its bad or good count is 1-10; a row with 1-10 loans shows only "
        "'1-10'. Totals are exact.")
    out(f"Dataset built by build_dataset.py v{meta.get('script_version', '?')} at {meta.get('built_at', '?')} "
        f"(mode {meta.get('mode', '?')}); specification file SHA-256 {meta.get('decision_log_sha256', '?')}.")
    out(f"Memory limit {args.memory_limit}, threads {args.threads}. Outputs: {P['out']}.")
    out("Descriptive only: no modeling decision is made here.")
    if MODE == "smoke":
        out("SMOKE MODE: the smoke dataset (one source file per agency); counts are not the full training population.")


def read_meta(con):
    try:
        return dict(con.execute("SELECT key, value FROM build_meta").fetchall())
    except duckdb.Error:
        return {}


def rep_population(con):
    """Section 1: kept loans by label; training loans by agency and pooled. Returns pooled (loans, bad)."""
    section("1. Population: kept loans labeled train, both agencies pooled")
    rows = con.execute("SELECT label, agency, count(*), count(*) FILTER (outcome = 'bad'), "
                       "count(*) FILTER (outcome IS NULL OR outcome NOT IN ('good', 'bad')) "
                       "FROM kept_loans GROUP BY ALL ORDER BY ALL").fetchall()
    labels = {r[0] for r in rows}
    train = [r for r in rows if r[0] == "train"]
    n = sum(r[2] for r in train)
    b = sum(r[3] for r in train)
    table = [[ag, m(k), m(bad), pct_text(k, bad)] for _, ag, k, bad, _ in train]
    table.append(["both agencies (used here)", m(n), m(b), pct_text(n, b)])
    out_table(["training loans", "loans", "bad loans", "bad rate"], table, left={0})
    other = {lab: sum(r[2] for r in rows if r[0] == lab) for lab in ("validation", "test")}
    out(f"  Not used here: validation {mc(other['validation'])} loans, test {mc(other['test'])} loans.")
    bd.check_line("every kept loan is labeled train, validation or test", labels <= {"train", "validation", "test"})
    bd.check_line("every training loan's outcome is good or bad", sum(r[4] for r in train) == 0)
    bd.check_line("there are training loans", n > 0)
    if n == 0:
        raise StopRun(1)
    return n, b


# ---------------------------------------------------------------- output 1: bad rate by value

def value_counts(con, col, kind, levels):
    """(value text, loans, bad) per distinct value or category, then the missing row; extra = undocumented values."""
    got = {v: (int(k), int(b)) for v, k, b in con.execute(f"SELECT {col}, count(*), sum(bad) FROM train GROUP BY 1").fetchall()}
    miss = got.pop(None, (0, 0))
    if kind == "number":
        rows, extra = [(num_text(v), *got[v]) for v in sorted(got)], []
    else:
        extra = sorted(v for v in got if v not in levels)
        rows = [(v, *got.get(v, (0, 0))) for v in levels + extra]
    return rows + [(MISSING, *miss)], extra


def csv_name(col):
    return f"bad_rate_{col}"


def rep_bad_rates(con):
    section("2. Bad rate by value (output 1): one row per distinct value or category, plus missing")
    out("  Columns: value, loans, bad loans, bad rate. Number tables longer than "
        f"{PRINT_ROWS} rows are summarized here; every row is in the CSV.")
    for col, label, kind, levels in VARIABLES:
        rows, extra = value_counts(con, col, kind, levels)
        masked = [mask_row(*r) for r in rows]
        path = P["out"] / f"{csv_name(col)}.csv"
        write_csv(path, ["value", "loans", "bad_loans", "bad_rate_pct"], masked)
        n_mask = sum(1 for r in masked if r[1] == MASK)
        n_norate = sum(1 for r in masked if r[1] not in (MASK, "0") and r[3] == "")
        out()
        out(f"--- {label} ({kind}, column {col}): {len(masked) - 1} {'values' if kind == 'number' else 'categories'} "
            f"+ missing row; rows with 1-10 loans: {n_mask}; other rows with the rate not shown: {n_norate}")
        out(f"    {path.relative_to(WORK_ROOT)}")
        if kind == "category":
            bd.check_line(f"{label}: every value is a documented category value", not extra)
        show = [[r[0], commas(r[1]), commas(r[2]), f"{float(r[3]):.2f}%" if r[3] else "-"] for r in masked]
        if len(masked) <= PRINT_ROWS:
            out_table(["value", "loans", "bad loans", "bad rate"], show, left={0})
        else:
            vals = [float(r[0]) for r in masked if r[0] != MISSING]
            out(f"  {len(vals):,} distinct values from {inv.fmt_value(min(vals))} to {inv.fmt_value(max(vals))} "
                "(table too long to print; see the CSV and the chart)")
            out_table(["value", "loans", "bad loans", "bad rate"], show[-1:], left={0})
    progress("Output 1 (bad rate by value) written")


# ---------------------------------------------------------------- charts (read the masked CSVs back from disk)

def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def logit(p):
    return math.log(p / (1 - p))


def wilson(bad, loans, z=Z95):
    p = bad / loans
    d = 1 + z * z / loans
    c = (p + z * z / (2 * loans)) / d
    h = z * math.sqrt(p * (1 - p) / loans + z * z / (4 * loans * loans)) / d
    return max(0.0, c - h), min(1.0, c + h)


def shown_rows(rows):
    """(value, loans, bad) of the rows whose rate is published, and (value, loans) of the rows whose loans are."""
    pts = [(r["value"], int(r["loans"]), int(r["bad_loans"])) for r in rows if r["bad_rate_pct"] != ""]
    bars = [(r["value"], int(r["loans"])) for r in rows if r["loans"].isdigit() and int(r["loans"]) > 0]
    return pts, bars


def draw_chart(csv_path, png_path, label, numeric):
    """Bad rate (log-odds scale, labeled in %, 95% Wilson intervals) over loan-count bars (log scale), from the
    masked CSV only. Returns what was drawn: {'points': [(value, rate)], 'bars': [(value, loans)]}."""
    plt = _pyplot()
    rows = read_masked(csv_path)
    body = [r for r in rows if r["value"] != MISSING]
    miss = [r for r in rows if r["value"] == MISSING]
    pts, bars = shown_rows(body)
    mpts, mbars = shown_rows(miss)
    pos = {r["value"]: (float(r["value"]) if numeric else i) for i, r in enumerate(body)}
    inner = [b / n for _, n, b in pts + mpts if 0 < b < n]
    lo_y = (min(map(logit, inner)) if inner else logit(0.001)) - 0.6
    hi_y = (max(map(logit, inner)) if inner else logit(0.1)) + 0.6

    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.3], width_ratios=[14, 1], hspace=0.06, wspace=0.04)
    ax_r = fig.add_subplot(gs[0, 0])
    ax_n = fig.add_subplot(gs[1, 0], sharex=ax_r)
    ax_mr = fig.add_subplot(gs[0, 1], sharey=ax_r)
    ax_mn = fig.add_subplot(gs[1, 1], sharey=ax_n)
    lw = 14 if not numeric else max(0.6, min(4.0, 500 / max(len(body), 1)))

    def put(axr, axn, pts, bars, xof, width):
        ys, xs, lo, hi = [], [], [], []
        floor, roof = [], []
        for v, n, b in pts:
            if b == 0:
                floor.append(xof(v))
            elif b == n:
                roof.append(xof(v))
            else:
                y = logit(b / n)
                wl, wh = wilson(b, n)
                xs.append(xof(v))
                ys.append(y)
                lo.append(y - (logit(wl) if wl > 0 else lo_y))
                hi.append((logit(wh) if wh < 1 else hi_y) - y)
        if xs:
            axr.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=3, color="#2b6cb0", ecolor="#90b4dd", elinewidth=0.8,
                         label="bad rate with 95% interval")
        if floor:
            axr.plot(floor, [lo_y + 0.12] * len(floor), "v", ms=5, color="#c05621", label="0% (drawn at the floor)")
        if roof:
            axr.plot(roof, [hi_y - 0.12] * len(roof), "^", ms=5, color="#c05621", label="100% (drawn at the top)")
        if bars:
            axn.vlines([xof(v) for v, _ in bars], 1, [k for _, k in bars], color="#a0aec0", lw=width)

    put(ax_r, ax_n, pts, bars, lambda v: pos[v], lw)
    put(ax_mr, ax_mn, mpts, mbars, lambda v: 0, 14)
    ticks = [p for p in PCT_TICKS if lo_y <= logit(p / 100) <= hi_y]
    ax_r.set_yticks([logit(p / 100) for p in ticks], [f"{p:g}%" for p in ticks])
    ax_r.set_ylim(lo_y, hi_y)
    ax_r.set_ylabel("bad rate (log-odds scale)")
    ax_r.grid(axis="y", color="#e2e8f0", lw=0.6)
    ax_n.set_yscale("log")
    ax_n.set_ylabel("loans (log)")
    if not bars and not mbars:
        ax_n.set_ylim(1, 10)
    plt.setp(ax_r.get_xticklabels(), visible=False)
    if not numeric:
        ax_n.set_xticks(range(len(body)), [r["value"] for r in body], rotation=20 if len(body) > 4 else 0)
    ax_n.set_xlabel(label)
    for ax in (ax_mr, ax_mn):
        ax.set_xlim(-1, 1)
        plt.setp(ax.get_yticklabels(), visible=False)
    ax_mr.set_xticks([])
    ax_mn.set_xticks([0], ["missing"])
    if not miss or miss[0]["loans"] in ("0", MASK):
        note = "none" if not miss or miss[0]["loans"] == "0" else "1-10 loans\n(not drawn)"
        ax_mr.text(0, (lo_y + hi_y) / 2, note, ha="center", va="center", color="#718096", fontsize=7)
    n_mask = sum(1 for r in rows if r["loans"] == MASK)
    n_norate = sum(1 for r in rows if r["loans"] not in (MASK, "0") and r["bad_rate_pct"] == "")
    ax_r.set_title(f"{label}: bad rate by value, training loans (log-odds scale)\n"
                   f"not drawn (masked): {n_mask} row(s) with 1-10 loans; rate not shown for {n_norate} other row(s)",
                   fontsize=10, loc="left")
    if ax_r.get_legend_handles_labels()[0]:
        ax_r.legend(loc="upper right", fontsize=8, frameon=False)
    fig.savefig(png_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    WRITTEN.append(png_path)
    return {"points": [(v, b / n) for v, n, b in pts + mpts], "bars": bars + mbars}


def draw_charts():
    for col, label, kind, _ in VARIABLES:
        draw_chart(P["out"] / f"{csv_name(col)}.csv", P["out"] / f"{csv_name(col)}.png", label, kind == "number")
    progress("Charts of output 1 drawn from the masked CSVs")


# ---------------------------------------------------------------- output 2: overlap

def masked_n(n):
    return str(m(n))


def pearson_rows(con):
    pairs = list(itertools.combinations([c for c, *_ in NUMBERS], 2))
    sel = []
    for a, b in pairs:
        both = f"{a} IS NOT NULL AND {b} IS NOT NULL"
        sel += [f"count(*) FILTER ({both})", f"corr({a}, {b}) FILTER ({both})"]
    vals = con.execute(f"SELECT {', '.join(sel)} FROM train").fetchone()
    return [(a, b, int(vals[2 * i]), vals[2 * i + 1]) for i, (a, b) in enumerate(pairs)]


def spearman(con, a, b):
    """Spearman = Pearson on average ranks (ties share their mean rank), on the pair's complete loans."""
    return con.execute(f"""
WITH d AS (SELECT {a} AS x, {b} AS y FROM train WHERE {a} IS NOT NULL AND {b} IS NOT NULL),
rx AS (SELECT x, sum(c) OVER (ORDER BY x) - (c - 1) / 2.0 AS r FROM (SELECT x, count(*) AS c FROM d GROUP BY x)),
ry AS (SELECT y, sum(c) OVER (ORDER BY y) - (c - 1) / 2.0 AS r FROM (SELECT y, count(*) AS c FROM d GROUP BY y))
SELECT corr(rx.r, ry.r) FROM d JOIN rx USING (x) JOIN ry USING (y)""").fetchone()[0]


def cramers_v(counts):
    """Cramer's V of a contingency table {(a, b): loans}; None when either side has fewer than 2 values."""
    ra = sorted({a for a, _ in counts})
    cb = sorted({b for _, b in counts})
    o = np.zeros((len(ra), len(cb)))
    for (a, b), k in counts.items():
        o[ra.index(a), cb.index(b)] = k
    o = o[o.sum(axis=1) > 0][:, o.sum(axis=0) > 0]
    if min(o.shape) < 2:
        return None
    n = o.sum()
    e = np.outer(o.sum(axis=1), o.sum(axis=0)) / n
    chi2 = ((o - e) ** 2 / e).sum()
    return math.sqrt(chi2 / (n * (min(o.shape) - 1)))


def cramers_rows(con):
    res = []
    for (a, *_), (b, *_) in itertools.combinations(CATEGORIES, 2):
        counts = {(x, y): int(k) for x, y, k in con.execute(
            f"SELECT {a}, {b}, count(*) FROM train WHERE {a} IS NOT NULL AND {b} IS NOT NULL GROUP BY ALL").fetchall()}
        res.append((a, b, sum(counts.values()), cramers_v(counts)))
    return res


def eta_rows(con):
    """Correlation ratio of each number across each category's values (both present)."""
    res = []
    for c, *_ in CATEGORIES:
        sel = ", ".join(f"count({x}), avg({x}), var_pop({x})" for x, *_ in NUMBERS)
        groups = con.execute(f"SELECT {sel} FROM train WHERE {c} IS NOT NULL GROUP BY {c}").fetchall()
        for i, (x, *_) in enumerate(NUMBERS):
            g = [(int(r[3 * i]), r[3 * i + 1], r[3 * i + 2]) for r in groups if r[3 * i]]
            n = sum(k for k, _, _ in g)
            eta = None
            if n:
                grand = sum(k * mu for k, mu, _ in g) / n
                between = sum(k * (mu - grand) ** 2 for k, mu, _ in g)
                within = sum(k * v for k, _, v in g)
                if between + within > 0:
                    eta = math.sqrt(between / (between + within))
            res.append((x, c, n, eta))
    return res


def logdet(a):
    if a.size == 0:
        return 0.0
    sign, ld = np.linalg.slogdet(a)
    return ld if sign > 0 else -math.inf


def vif(con, n_train):
    """VIF per column and GVIF per variable, from covariances of the columns Logistic Regression would see."""
    complete = " AND ".join(f"{c} IS NOT NULL" for c, *_ in VARIABLES)
    small = {c: [] for c, *_ in CATEGORIES}
    while True:     # a value with 1-10 complete loans gets no dummy: its loans are left out, then recount
        where = complete + "".join(f" AND {c} NOT IN ({', '.join(lit(v) for v in vs)})" for c, vs in small.items() if vs)
        counts = {c: {v: int(k) for v, k in con.execute(f"SELECT {c}, count(*) FROM train WHERE {where} GROUP BY 1").fetchall()}
                  for c in small}
        new = {c: [v for v, k in d.items() if 1 <= k <= 10] for c, d in counts.items()}
        if not any(new.values()):
            break
        for c, vs in new.items():
            small[c] += vs
    n_complete = con.execute(f"SELECT count(*) FROM train WHERE {complete}").fetchone()[0]
    n_used = sum(counts[CATEGORIES[0][0]].values())
    names, exprs, owner, refs = [], [], [], {}
    for c, label, kind, levels in VARIABLES:
        if kind == "number":
            names.append(label)
            exprs.append(f"CAST({c} AS DOUBLE)")
            owner.append(c)
            continue
        d = counts[c]
        order = [v for v in levels if v in d] + sorted(v for v in d if v not in levels)
        if not order:
            continue
        refs[c] = max(order, key=lambda v: d[v])    # the most common value; ties go to the documented category order
        for v in order:
            if v != refs[c]:
                names.append(f"{label} = {v}")
                exprs.append(f"CASE WHEN {c} = {lit(v)} THEN 1.0 ELSE 0.0 END")
                owner.append(c)
    res = {"n_train": n_train, "n_complete": n_complete, "n_used": n_used, "small": small, "refs": refs,
           "columns": [], "variables": [], "constant": [], "singular": False}
    if n_used < 2:
        return res
    k = len(exprs)
    inner = ", ".join(f"{e} AS c{i}" for i, e in enumerate(exprs))
    aggs = ", ".join(f"covar_pop(c{i}, c{j})" for i in range(k) for j in range(i, k))
    vals = iter(con.execute(f"SELECT {aggs} FROM (SELECT {inner} FROM train WHERE {where})").fetchone())
    cov = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            cov[i, j] = cov[j, i] = next(vals)
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    keep = [i for i in range(k) if sd[i] > 0]
    res["constant"] = [names[i] for i in range(k) if sd[i] == 0]
    r = cov[np.ix_(keep, keep)] / np.outer(sd[keep], sd[keep])
    ld = logdet(r)
    res["singular"] = ld < -30          # columns that overlap exactly: every VIF is infinite
    inv_r = None if res["singular"] else np.linalg.inv(r)
    for pos, i in enumerate(keep):
        res["columns"].append((names[i], owner[i], math.inf if inv_r is None else float(inv_r[pos, pos])))
    for c, label, kind, _ in VARIABLES:
        s = [pos for pos, i in enumerate(keep) if owner[i] == c]
        if not s:
            res["variables"].append((label, kind, 0, refs.get(c), None, None))
            continue
        o = [pos for pos in range(len(keep)) if pos not in s]
        g = math.inf if res["singular"] else math.exp(logdet(r[np.ix_(s, s)]) + logdet(r[np.ix_(o, o)]) - ld)
        res["variables"].append((label, kind, len(s), refs.get(c), g, g ** (1 / (2 * len(s)))))
    return res


def rep_overlap(con, n_train):
    section("3. Overlap (output 2): correlations, Cramer's V, correlation ratio, VIF")
    t0 = time.time()
    pr = pearson_rows(con)
    corr = [(a, b, n, p, spearman(con, a, b) if n > 1 else None) for a, b, n, p in pr]
    write_csv(P["out"] / "correlations.csv", ["variable_1", "variable_2", "loans_used", "pearson", "spearman"],
              [[LABEL[a], LABEL[b], masked_n(n), stat(p), stat(s)] for a, b, n, p, s in corr])
    out()
    out("--- Numbers: Pearson and Spearman (average ranks), each pair on loans with both values present")
    out_table(["variable 1", "variable 2", "loans used", "Pearson", "Spearman"],
              [[LABEL[a], LABEL[b], m(n), stat(p, 3) or "-", stat(s, 3) or "-"] for a, b, n, p, s in corr], left={0, 1})
    progress(f"Correlations done in {hms(time.time() - t0)}")

    t0 = time.time()
    cv = cramers_rows(con)
    write_csv(P["out"] / "cramers_v.csv", ["variable_1", "variable_2", "loans_used", "cramers_v"],
              [[LABEL[a], LABEL[b], masked_n(n), stat(v)] for a, b, n, v in cv])
    out()
    out("--- Categories: Cramer's V, each pair on loans with both values present")
    out_table(["variable 1", "variable 2", "loans used", "Cramer's V"],
              [[LABEL[a], LABEL[b], m(n), stat(v, 3) or "-"] for a, b, n, v in cv], left={0, 1})

    et = eta_rows(con)
    write_csv(P["out"] / "correlation_ratio.csv", ["number", "category", "loans_used", "eta"],
              [[LABEL[x], LABEL[c], masked_n(n), stat(e)] for x, c, n, e in et])
    out()
    out("--- Number x category: correlation ratio (eta; 0 = the number's mean is the same in every category, "
        "1 = the category fixes the number)")
    out_table(["number", "category", "loans used", "eta"],
              [[LABEL[x], LABEL[c], m(n), stat(e, 3) or "-"] for x, c, n, e in et], left={0, 1})
    progress(f"Cramer's V and correlation ratios done in {hms(time.time() - t0)}")

    t0 = time.time()
    v = vif(con, n_train)
    progress(f"VIF done in {hms(time.time() - t0)}")
    return corr, cv, et, v


def rep_vif(v, n_train):
    out()
    out("--- VIF: the 8 numbers plus category dummies (most common value = reference), loans with all 17 values present")
    out(f"  Training loans {mc(n_train)}; missing at least one value {mc(n_train - v['n_complete'])}; left out because "
        f"their category value has 1-10 such loans {mc(v['n_complete'] - v['n_used'])}; loans used {mc(v['n_used'])}.")
    left = [f"{LABEL[c]} = {x}" for c, vs in v["small"].items() for x in vs]
    if left:
        out("  Values left out (1-10 loans, no dummy): " + "; ".join(left))
    if v["constant"]:
        out("  Constant among the loans used (no variation, left out): " + "; ".join(v["constant"]))
    if v["singular"]:
        out("  The columns overlap exactly (the correlation matrix is singular): every VIF is infinite.")
    write_csv(P["out"] / "vif_columns.csv", ["column", "variable", "vif"],
              [[name, LABEL[c], stat(x)] for name, c, x in v["columns"]])
    write_csv(P["out"] / "vif_variables.csv",
              ["variable", "kind", "columns", "reference", "loans_used", "gvif", "gvif_adjusted"],
              [[lab, kind, df, ref or "", masked_n(v["n_used"]), stat(g), stat(a)] for lab, kind, df, ref, g, a in v["variables"]])
    out_table(["variable", "kind", "columns", "reference", "GVIF", "GVIF^(1/(2 df))"],
              [[lab, kind, df, ref or "-", stat(g, 2) or "-", stat(a, 2) or "-"] for lab, kind, df, ref, g, a in v["variables"]],
              left={0, 1, 3})
    out()
    out("  VIF of each column (for a number, its VIF equals its GVIF):")
    out_table(["column", "VIF"], [[name, stat(x, 2)] for name, _, x in v["columns"]], left={0})


# ---------------------------------------------------------------- what looks like a jump or heavy overlap

def logit_cc(n, b):
    return math.log((b + 0.5) / (n - b + 0.5))


def apart(a, b):
    """Two (value, loans, bad) rows whose 95% Wilson intervals do not overlap."""
    la, ha = wilson(a[2], a[1])
    lb, hb = wilson(b[2], b[1])
    return ha < lb or hb < la


def rate_of(row):
    return row[2] / row[1]


def shape_flags(csv_path, numeric, overall):
    """Descriptive flags for one variable, from its masked CSV read back from disk. overall = (label, loans, bad)
    of all training loans, or None when its rate is not shown."""
    pts, _ = shown_rows(read_masked(csv_path))
    body = [p for p in pts if p[0] != MISSING]
    miss = [p for p in pts if p[0] == MISSING]
    lines = []

    def rt(r):
        return f"{100 * rate_of(r):.2f}%"
    if numeric:
        body.sort(key=lambda p: float(p[0]))
        steps = [(abs(logit_cc(b[1], b[2]) - logit_cc(a[1], a[2])), a, b) for a, b in zip(body, body[1:]) if apart(a, b)]
        up = sum(1 for _, a, b in steps if rate_of(b) > rate_of(a))
        if steps:
            lines.append(f"{len(steps)} jump(s) between neighbouring shown values ({up} up, {len(steps) - up} down)"
                         + ("; the rate changes direction" if up and len(steps) - up else "") + f". Largest {min(TOP_JUMPS, len(steps))}:")
            for size, a, b in sorted(steps, key=lambda t: -t[0])[:TOP_JUMPS]:
                lines.append(f"  {a[0]} -> {b[0]}: {rt(a)} -> {rt(b)} ({'up' if rate_of(b) > rate_of(a) else 'down'}, "
                             f"log-odds change {size:.2f})")
    else:
        pairs = [(a, b) for a, b in itertools.combinations(body, 2) if apart(a, b)]
        if pairs:
            hi, lo = max(body, key=rate_of), min(body, key=rate_of)
            lines.append(f"{len(pairs)} of {len(body) * (len(body) - 1) // 2} pairs of categories have non-overlapping "
                         f"95% intervals; highest {hi[0]} {rt(hi)}, lowest {lo[0]} {rt(lo)}")
    if miss and overall and apart(miss[0], overall):
        lines.append(f"missing values: {rt(miss[0])} vs {rt(overall)} for all training loans (95% intervals do not overlap)")
    return lines


def rep_flags(n, b, corr, cv, et, v):
    section("4. What looks like a jump or heavy overlap (descriptive only; no modeling decision)")
    out("  Jump: neighbouring shown values whose 95% Wilson intervals do not overlap, largest by change in log-odds; "
        "categories: pairs whose intervals do not overlap. Read from the masked CSVs only.")
    out(f"  Heavy overlap: |Pearson| or |Spearman| >= {CORR_HEAVY}; Cramer's V >= {V_HEAVY}; eta >= {CORR_HEAVY}; "
        f"VIF >= {VIF_WATCH:g} (watch) or >= {VIF_SEVERE:g} (severe); GVIF^(1/(2 df)) >= {math.sqrt(VIF_WATCH):.2f} / "
        f"{math.sqrt(VIF_SEVERE):.2f}.")
    overall = ("all", n, b) if rate_shown(n, b) else None
    out()
    out("--- Bad rate by value")
    for col, label, kind, _ in VARIABLES:
        lines = shape_flags(P["out"] / f"{csv_name(col)}.csv", kind == "number", overall)
        out(f"  {label}: " + (lines[0] if lines else "nothing flagged"))
        for line in lines[1:]:
            out(f"    {line}")
    out()
    out("--- Overlap")
    flags = []
    for a, b2, _, p, s in corr:
        hits = [f"{nm} {x:.3f}" for nm, x in (("Pearson", p), ("Spearman", s)) if x is not None and abs(x) >= CORR_HEAVY]
        if hits:
            flags.append(f"{LABEL[a]} x {LABEL[b2]}: " + ", ".join(hits))
    flags += [f"{LABEL[a]} x {LABEL[b2]}: Cramer's V {x:.3f}" for a, b2, _, x in cv if x is not None and x >= V_HEAVY]
    flags += [f"{LABEL[x]} by {LABEL[c]}: eta {e:.3f}" for x, c, _, e in et if e is not None and e >= CORR_HEAVY]
    for name, _, x in v["columns"]:
        if x >= VIF_WATCH:
            flags.append(f"VIF {name}: {stat(x, 2)} ({'severe' if x >= VIF_SEVERE else 'watch'})")
    for lab, kind, df, _, g, a in v["variables"]:
        if kind == "category" and a is not None and a >= math.sqrt(VIF_WATCH):
            flags.append(f"GVIF {lab}: GVIF^(1/(2 df)) {stat(a, 2)} ({'severe' if a >= math.sqrt(VIF_SEVERE) else 'watch'})")
    for f in flags or ["nothing flagged"]:
        out(f"  {f}")


def rep_files():
    section("Files written")
    for p in WRITTEN:
        out(f"  {p.relative_to(WORK_ROOT)}")


# ---------------------------------------------------------------- main

def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--smoke", action="store_true", help="read loan_dataset_smoke.duckdb; write the *_smoke outputs")
    p.add_argument("--memory-limit", default=inv.DEFAULT_MEMORY_LIMIT, help=f"DuckDB memory limit (default {inv.DEFAULT_MEMORY_LIMIT})")
    p.add_argument("--threads", type=int, default=inv.DEFAULT_THREADS, help=f"DuckDB threads (default {inv.DEFAULT_THREADS})")
    args = p.parse_args(argv)
    if not re.fullmatch(r"\d+(\.\d+)?\s*[KMGT]i?B", args.memory_limit, re.IGNORECASE):
        p.error("--memory-limit must look like 5GB or 4096MB")
    return args


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
    progress(f"explore.py v{SCRIPT_VERSION} started ({MODE} mode). Dataset {P['db']} (read-only); outputs {P['out']}.")
    con, code = None, 0
    try:
        if not P["db"].exists():
            progress(f"STOP: no dataset at {P['db']}; run build_dataset.py{' --smoke' if args.smoke else ''} first.")
            raise StopRun(1)
        progress(f"DuckDB memory limit {args.memory_limit}, threads {args.threads}.")
        con = open_dataset(args)
        P["out"].mkdir(parents=True, exist_ok=True)
        create_training_view(con)
        rep_header(args, read_meta(con))
        n, b = rep_population(con)
        progress(f"Training loans (both agencies): {mc(n)}")
        bd.rsection("2 bad rate by value", rep_bad_rates, con)
        bd.rsection("charts", draw_charts)
        res = []
        bd.rsection("3 overlap", lambda: res.append(rep_overlap(con, n)))
        if res:
            corr, cv, et, v = res[0]
            bd.rsection("3 VIF", rep_vif, v, n)
            bd.rsection("4 flags", rep_flags, n, b, corr, cv, et, v)
        rep_files()
        section(f"SUMMARY: {len(FAILS)} problem(s)")
        for f in FAILS:
            out(f"  FAIL: {f}")
        out("  SUCCESS: every output written; every check passed." if not FAILS else
            "  FAILURE: see the problems above.")
        out(f"(explore took {hms(time.time() - started)})")
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
