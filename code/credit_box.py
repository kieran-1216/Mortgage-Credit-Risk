"""Q3, the credit box: where a lender should set approval cutoffs.
Summaries only.

Six methods: (1) no cutoff, (2) Logistic
Regression risk, (3) LightGBM risk, (4) each variable alone (one method: the best single-variable rule over the 17
model variables), (5) every pair of variables (136 pairs; X and Y; X or Y), (6) every trio (680 trios; the 9 and/or
forms using each variable once). The 17 variables include loan amount as a percent of its conforming
loan limit and the at-the-limit flag, not loan amount in dollars. Score = % of bad-loan dollars declined - % of
good-loan dollars declined (each loan's original amount in DOLLARS, read from the dataset's original UPB as a separate
column, never a model variable; the population's own totals). Cutoffs are set from TRAINING loans, every candidate is scored on VALIDATION loans, the best
per method is kept (ties: fewer declined validation dollars, then a fixed enumeration order), the winning method is the
highest validation score (ties: fewer declined dollars, fewer variables, the order above). The six chosen rules
go to outputs/q3/chosen_rules.json BEFORE any test outcome is read; then each is applied unchanged to the TEST loans
(all, then each agency) and the trade-off curve is written as data (no chart).

Definitions (all learned from training loans)
  * Riskier end of a number: its rank-based AUC (bad vs good, ties averaged) on training loans with a value; above 0.5
    high values are riskier, below 0.5 low values; exactly 0.5 (or undefined) stops the run.
  * Categories: groups ordered riskiest first by training bad rate (ties: documented category order, blank group last); blanks
    are their own group. Special program R and any value absent from training count as the reference category.
  * 1% steps (pairs and trios): the values at the 1%, 2%, ... 99% marks of the riskiest-first line-up of training
    loans with a value (mark p = the loan at position ceil(p x N / 100)); marks on the same value merge; the residual
    beyond the last mark is not a step. Single-variable cutoffs: every distinct training value; categories: each
    boundary in riskiest-first order. Model-risk cutoffs: every distinct adjusted (isotonic) predicted risk among
    training loans (the isotonic steps tie many loans, so there are far fewer cutoffs than loans).
  * Blanks of a number (one group): walking the steps from the risky end, step k is the first whose training bad rate
    is below the blank group's; blanks rank just riskier than step k's riskiest value (any cutoff that declines that
    value declines the blanks); no such step: blanks rank safest of all. A variable with no training blanks uses the
    overall training bad rate of loans with a value in place of the blank rate. Same placement in every method.
  * A cutoff that declines every loan with a value (the safest value, the boundary after the last group, the minimum
    training risk) is not a candidate. A pair or trio rule has a real cutoff on each of its variables.
  * Numbers are read as the models saw them (fit_models' float32 reader) and compared in float32; cutoff values are
    printed in their shortest float32 text.
  * Model risks: the final models exactly as in score_test.py (Logistic Regression with lr_edge_rule.json's result and
    the winner's isotonic steps saved there; the chosen LightGBM model and its isotonic steps in isotonic_calibration.json).

Reuses fit_models.py (paths, read-only connection, views, row reader, AUC, isotonic adjustment, column builders, masking),
check_beyond_range.py (saved-model loading, training ranges, edge holding, saved Logistic Regression scoring) and
score_test.py (--rescore-test parsing, the test-once lock, the per-agency test views). New here: the per-variable
ladders and blank placement, the cumulative-table search for pairs and trios, the prefix search for model risk, the
exact tie resolution in integer dollars, the curve snapping, the rule wording and the test application.

Files (full mode; --smoke writes the *_smoke variants and NO test result numbers: the test phase runs for its checks only)
  outputs/q3/chosen_rules.json      the 6 chosen rules, validation results and the winner, written before the test is read
  Masking: counts 1-10 show '1-10' and their dollars, rates and scores are hidden; a group whose bad rate is hidden has its
  position hidden too (number blank group: 'placement masked'; category: its other groups listed without rank numbers, its
  blank placement masked and its blank treatment 'masked' in every rule; a rule never names a hidden group). The code uses
  every placement internally.
  outputs/q3/variable_order.csv     directions, AUCs, category orders, blank groups and their placement
  outputs/q3/steps.csv              the 1% steps of every number variable (mark values, loans, bad rate)
  outputs/q3/validation_results.csv the 6 methods on validation loans
  outputs/q3/test_results.csv       each chosen rule on all test loans, then each agency (full mode only)
  outputs/q3/curve_lines.csv, curve_dots.csv   the trade-off curve data (full mode only)
  logs/credit_box_report.txt, logs/credit_box_progress.log, logs/credit_box_error_detail.txt (stack frames only)

Usage (from the repository root, after fit_models.py and check_beyond_range.py; --smoke needs check_beyond_range.py --smoke):
    python code/credit_box.py [--smoke] [--threads 8] [--memory-limit 2GB] [--rescore-test]
TEST ONCE: the script refuses to start if its report or its output folder exists; --rescore-test renames
both with a timestamp (nothing is deleted), once: a finished rescore (its report carries the RESCORED
statement and SUCCESS) is never replaced. Every full-mode report carries the RESCORED line.
Exit codes: 0 SUCCESS; 1 FAILURE or refused; 130 interrupted.
"""

import itertools
import json
import os
import sys
import time
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset as bd  # noqa: E402  (reused: check lines, report sections, error detail)
import check_beyond_range as cb  # noqa: E402  (reused: saved-model loading, training ranges, edge holding, LR scoring)
import check_raw  # noqa: E402
import explore as ex  # noqa: E402  (reused: dataset connection, masking helpers, CSV writing)
import fit_models as fm  # noqa: E402  (reused: views, readers, AUC, isotonic, column builders)
import inventory as inv  # noqa: E402  (reused: progress log, report file, 1-10 masking)
from model_core import adjust_scores, adjusted_risks, raw_scores  # noqa: E402,F401
import score_test as st  # noqa: E402  (reused: --rescore-test parsing, test-once lock, test views)
from check_raw import out, out_table, section  # noqa: E402
from inventory import StopRun, m, mc, progress  # noqa: E402
from load_raw import LOG_DIR, PROJECT_ROOT, hms  # noqa: E402

SCRIPT_VERSION = "2"                    # 2: 17 variables (percent of limit and at-the-limit flag); isotonic adjustment
MASK, MISSING = ex.MASK, ex.MISSING
LR, GBM = st.LR, st.GBM
AGENCIES = st.AGENCIES
MARKS = 99                      # the 1% ... 99% marks
CURVE_POINTS = 1000             # a point at every 0.1% of test loans
NEAR = 1e-12                    # float score band inside which candidates are compared exactly
CHUNK = fm.CHUNK
NV = len(fm.VARIABLES)
METHODS = ["no cutoff", "Logistic Regression risk", "LightGBM risk", "single variable", "pair of variables", "trio of variables"]
METHOD_VARS = [0, NV, NV, 1, 2, 3]       # "fewer variables" in the winner's tie rule
PAIR_FORMS = [("X and Y", lambda x, y: x & y), ("X or Y", lambda x, y: x | y)]
TRIO_FORMS = [("X and Y and Z", lambda x, y, z: x & y & z),
              ("X or Y or Z", lambda x, y, z: x | y | z),
              ("X and (Y or Z)", lambda x, y, z: x & (y | z)),
              ("Y and (X or Z)", lambda x, y, z: y & (x | z)),
              ("Z and (X or Y)", lambda x, y, z: z & (x | y)),
              ("(X and Y) or Z", lambda x, y, z: (x & y) | z),
              ("(X and Z) or Y", lambda x, y, z: (x & z) | y),
              ("(Y and Z) or X", lambda x, y, z: (y & z) | x),
              ("at least 2 of 3", lambda x, y, z: (x & y) | (x & z) | (y & z))]
TIE_ORDER = ("Within a method, after the score and fewer declined validation dollars: model risk, the higher risk value; "
             "single variable, the variables in fit_models' order then cutoffs from the risky end inward; pairs and "
             "trios, the combinations in that variable order, the rule forms in the listed order, the cutoff combinations "
             "in lexicographic step order from the risky end. Winner: score, fewer declined validation dollars, fewer "
             f"variables (no cutoff 0, single 1, pair 2, trio 3, each model {NV}), then the methods' listing order.")

WORK_ROOT = PROJECT_ROOT    # tests re-point these two before calling main()
LOGS = LOG_DIR
MODE = "full"
P = {}
FAILS = bd.FAILS
WRITTEN = []
LAST_RULES = {}             # tests only: {work root: the internal rules of the last run}


def configure(mode):
    """score_test's configuration (which sets check_beyond_range's and fit_models' paths), then this script's own files;
    score_test's test-once lock is re-pointed to them."""
    global MODE
    MODE = mode
    st.WORK_ROOT = WORK_ROOT
    st.LOGS = LOGS
    st.configure(mode)
    sfx = "_smoke" if mode == "smoke" else ""
    P.clear()
    P.update(models=cb.P["models"], out=WORK_ROOT / "outputs" / f"q3{sfx}", report=LOGS / f"credit_box_report{sfx}.txt",
             progress=LOGS / "credit_box_progress.log", error=LOGS / "credit_box_error_detail.txt")
    P["rules"] = P["out"] / "chosen_rules.json"
    inv.REPORT_PATH = P["report"]
    inv.PROGRESS_PATH = P["progress"]
    bd.P["error"] = P["error"]
    st.P.update(report=P["report"], out=P["out"])       # st.guard_test_once now guards this script's results


# ---------------------------------------------------------------- small helpers

def f32text(v):
    """Shortest text that reads back to the same float32 (46, -5.34, 0.0625)."""
    return np.format_float_positional(np.float32(v), unique=True, trim="-")


def masked(n):
    return 1 <= int(n) <= 10


def hidden(loans, bad):
    """A group whose bad rate is hidden (its loan, bad or good count is 1-10): its position is hidden too."""
    return loans > 0 and not ex.rate_shown(loans, bad)


def dollars(count, d):
    """A dollar sum shown only when its loan count is not 1-10."""
    return "-" if masked(count) else f"{int(d):,}"


def share(count, d, total):
    return "-" if masked(count) or total == 0 else f"{100 * d / total:.4f}%"


def score_text(bad_n, bad_d, good_n, good_d, B, G):
    """Score in percentage points; hidden when either declined count is 1-10."""
    if masked(bad_n) or masked(good_n) or B == 0 or G == 0:
        return "-"
    return f"{100 * (bad_d / B - good_d / G):.4f}"


def rate_cell(loans, bad):
    return f"{100 * bad / loans:.4f}%" if ex.rate_shown(loans, bad) else "-"


def values_of(lad, num, cat):
    if lad.kind == "number":
        return num[:, fm.NUM_COLS.index(lad.col)]
    return cat[:, fm.CAT_COLS.index(lad.col)]


# ---------------------------------------------------------------- ladders: one per variable, learned from training

class NumberLadder:
    """A number variable's riskiest-first order: direction (training AUC), distinct training values, the merged 1%
    marks and their steps, the blank group and its placement. Positions are 0-based: a cutoff at index c declines
    the loans whose position is <= c; the last position (D for values, M for steps) is declined by no candidate."""
    kind = "number"

    def __init__(self, index, x, y):
        self.index = index
        self.col, self.label = fm.VARIABLES[index][0], fm.VARIABLES[index][1]
        nan = np.isnan(x)
        vals, yb = x[~nan], y[~nan].astype(np.int64)
        self.blank_loans, self.blank_bad = int(nan.sum()), int(y[nan].sum())
        self.N, self.N_bad = int(len(vals)), int(yb.sum())
        self.auc = fm.auc(yb, vals.astype(np.float64)) if self.N else None   # float64: rankdata keeps float32 ranks otherwise and the rank sum loses precision
        self.high = None if self.auc is None or self.auc == 0.5 else bool(self.auc > 0.5)
        if self.high is None:
            return
        t = -vals if self.high else vals                             # ascending t = riskiest first (float32, exact)
        self.rt, inverse = np.unique(t, return_inverse=True)
        self.rv = -self.rt if self.high else self.rt                 # the values themselves, riskiest first
        self.D = len(self.rt)
        self.rc = np.bincount(inverse, minlength=self.D).astype(np.int64)
        self.rb = np.bincount(inverse, weights=yb, minlength=self.D).astype(np.int64)
        del inverse
        cum, cum_b = np.cumsum(self.rc), np.cumsum(self.rb)
        pos = (np.arange(1, MARKS + 1, dtype=np.int64) * self.N + 99) // 100       # ceil(p x N / 100), 1-based
        self.mark_idx = np.unique(np.searchsorted(cum, pos, side="left"))         # distinct-value index of each merged mark
        self.M = len(self.mark_idx)
        self.step_of_index = np.searchsorted(self.mark_idx, np.arange(self.D + 1), side="left")   # residual -> M
        ends, ends_b = cum[self.mark_idx], cum_b[self.mark_idx]
        self.step_loans = np.diff(np.concatenate([[0], ends]))
        self.step_bad = np.diff(np.concatenate([[0], ends_b]))
        self.last_mark_safest = bool(self.mark_idx[-1] == self.D - 1)
        self.phantom = self.blank_loans == 0
        self.blank_rate = Fraction(self.N_bad, self.N) if self.phantom else Fraction(self.blank_bad, self.blank_loans)
        self.k0 = next((s for s in range(self.M) if Fraction(int(self.step_bad[s]), int(self.step_loans[s])) < self.blank_rate), None)
        self.jk = None if self.k0 is None else (0 if self.k0 == 0 else int(self.mark_idx[self.k0 - 1]) + 1)   # index of u_k
        self.S = self.M + 1                                          # grid size for pairs and trios

    @property
    def placement_masked(self):
        return not self.phantom and hidden(self.blank_loans, self.blank_bad)

    def allowed_steps(self):
        """Grid mask (length S) of the step cutoffs that are candidates."""
        a = np.zeros(self.S, dtype=bool)
        a[:self.M] = True
        if self.last_mark_safest:
            a[self.M - 1] = False
        return a

    def allowed_values(self):
        a = np.ones(self.D + 1, dtype=bool)
        a[self.D - 1:] = False                                       # the safest value declines every loan with a value
        return a

    def positions(self, x):
        """(step position, value position), 0-based int32, for an array of values (NaN = blank)."""
        nan = np.isnan(x)
        t = np.where(nan, np.float32(0), -x if self.high else x).astype(np.float32)
        c0 = np.searchsorted(self.rt, t, side="left").astype(np.int32)
        vpos = c0.copy()
        vpos[nan] = self.D if self.jk is None else self.jk
        spos = self.step_of_index[c0].astype(np.int32)
        spos[nan] = self.M if self.k0 is None else self.k0
        return spos, vpos

    def curve_keys(self, x):
        """Sort keys for the trade-off curve: riskiest first by value; blanks as their own group just riskier than u_k."""
        nan = np.isnan(x)
        prim = np.where(nan, np.float32(np.inf) if self.jk is None else self.rt[self.jk], -x if self.high else x).astype(np.float32)
        sec = np.where(nan, 0, 1).astype(np.int8)
        return prim, sec

    def mark_value(self, s0):
        return self.rv[self.mark_idx[s0]]

    def term_text(self, value):
        return f"{self.label} {'>=' if self.high else '<='} {f32text(value)}"

    def term_notes(self, c, kind):
        declined = (self.jk is not None and self.jk <= c) if kind == "value" else (self.k0 is not None and self.k0 <= c)
        return [f"{self.label} blank: {'masked' if self.placement_masked else 'declined' if declined else 'approved'}"], declined, 0

    def placement_text(self):
        if self.placement_masked:
            return "placement masked"
        how = " (no training blanks: placed by the overall training bad rate)" if self.phantom else ""
        if self.k0 is None:
            return "safest of all (no step has a lower bad rate)" + how
        return f"just riskier than step {self.k0 + 1}'s riskiest value {f32text(self.rv[self.jk])}: declined by the cutoffs at " \
               f"{'>=' if self.high else '<='} {f32text(self.mark_value(self.k0))} or looser" + how


class CategoryLadder:
    """A category's groups ordered riskiest first by training bad rate (ties: documented category order, blank last); the blank
    group is a group of its own when training has blanks, otherwise it is placed by the overall rate just riskier than
    the first group below it. Position = group rank (0-based); a boundary at index b declines ranks <= b."""
    kind = "category"

    def __init__(self, index, codes_col, y, names, ref_code, max_code):
        self.index = index
        self.col, self.label = fm.VARIABLES[index][0], fm.VARIABLES[index][1]
        blank = codes_col == -1
        self.blank_loans, self.blank_bad = int(blank.sum()), int(y[blank].sum())
        present, inverse = np.unique(codes_col[~blank], return_inverse=True)
        counts = np.bincount(inverse, minlength=len(present))
        bads = np.bincount(inverse, weights=y[~blank].astype(np.int64), minlength=len(present)).astype(np.int64)
        groups = [(int(c), names[int(c)], int(n), int(b)) for c, n, b in zip(present, counts, bads)]
        self.N, self.N_bad = int(counts.sum()), int(bads.sum())
        self.phantom = self.blank_loans == 0
        if not self.phantom:
            groups.append((-1, MISSING, self.blank_loans, self.blank_bad))
        groups.sort(key=lambda g: (-Fraction(g[3], g[2]), g[0] == -1, g[0]))
        self.groups = groups
        self.G = len(groups)
        self.S = self.G + 1
        top = max(max(c for c, *_ in groups), max_code) + 3          # index = code + 2 (-2 undocumented, -1 blank)
        self.rank_of_code = np.full(top, next(r for r, g in enumerate(groups) if g[0] == ref_code), dtype=np.int32)
        for r, (c, *_) in enumerate(groups):
            if c >= 0:
                self.rank_of_code[c + 2] = r
        overall = Fraction(self.N_bad, self.N)
        if self.phantom:
            self.blank_rank0 = next((r for r, g in enumerate(groups) if Fraction(g[3], g[2]) < overall), self.G)
        else:
            self.blank_rank0 = next(r for r, g in enumerate(groups) if g[0] == -1)
        self.hidden = [hidden(g[2], g[3]) for g in groups]            # per group: rate hidden -> position hidden
        self.n_hidden_values = sum(h for h, g in zip(self.hidden, groups) if g[0] >= 0)

    @property
    def any_hidden(self):
        return any(self.hidden)

    @property
    def placement_masked(self):
        """Any hidden group masks the blank placement (real or phantom): a placement among the groups would name or
        locate the hidden group."""
        return self.any_hidden

    def allowed_steps(self):
        a = np.zeros(self.S, dtype=bool)
        a[:self.G - 1] = True                                        # the boundary after the last group is excluded
        return a

    allowed_values = allowed_steps

    def positions(self, codes):
        pos = self.rank_of_code[codes.astype(np.int64) + 2]
        pos[codes == -1] = self.blank_rank0
        return pos, pos

    def curve_keys(self, codes):
        pos, _ = self.positions(codes)
        sec = np.where((codes == -1) & self.phantom, 0, 1).astype(np.int8)
        return pos.astype(np.float32), sec

    def declined_names(self, b):
        """Names of the groups declined by boundary b; a group with a hidden position is never named."""
        return [g[1] if g[0] >= 0 else "blank" for r, g in enumerate(self.groups[:b + 1]) if not self.hidden[r]]

    def term_text(self, b):
        names = self.declined_names(b)
        if not names:
            return f"{self.label} is (masked groups only)"
        joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]
        return f"{self.label} is {joined}"

    def term_notes(self, b, kind=None):
        declined = self.blank_rank0 <= b
        notes = [f"{self.label} blank: {'masked' if self.placement_masked else 'declined' if declined else 'approved'}"]
        if self.n_hidden_values:
            notes.append(f"{self.label}: {self.n_hidden_values} group{'s' if self.n_hidden_values > 1 else ''} masked")
        return notes, declined, self.n_hidden_values

    def placement_text(self):
        if self.placement_masked:
            return "placement masked"
        if not self.phantom:
            return f"its own group at rank {self.blank_rank0 + 1} of {self.G}"
        if self.blank_rank0 >= self.G:
            return "safest of all (no group has a lower bad rate; no training blanks: placed by the overall training bad rate)"
        return (f"just riskier than rank {self.blank_rank0 + 1} ({self.groups[self.blank_rank0][1]}): declined with it "
                f"(no training blanks: placed by the overall training bad rate)")


def build_ladders(num, cat, y, codes, refs, present):
    ladders = []
    for i, (col, _, kind, _) in enumerate(fm.VARIABLES):
        if kind == "number":
            ladders.append(NumberLadder(i, num[:, fm.NUM_COLS.index(col)], y))
        else:
            names = {codes[col][v]: v for v in present[col]}
            ladders.append(CategoryLadder(i, cat[:, fm.CAT_COLS.index(col)], y, names, codes[col][refs[col]], max(codes[col].values())))
    return ladders


# ---------------------------------------------------------------- the models' adjusted risks

# ---------------------------------------------------------------- candidates and exact ties

class Cand:
    """One candidate's validation result: declined bad and good dollars (exact integers) and its enumeration order."""
    __slots__ = ("bad", "good", "order", "where")

    def __init__(self, bad, good, order, where=None):
        self.bad, self.good, self.order, self.where = int(bad), int(good), order, where

    def score(self, B, G):
        return self.bad / B - self.good / G


def better(a, b, B, G):
    """Exact comparison: higher score (integer cross-multiplication), then fewer declined dollars, then earlier order."""
    na, nb = a.bad * G - a.good * B, b.bad * G - b.good * B
    if na != nb:
        return na > nb
    da, db = a.bad + a.good, b.bad + b.good
    if da != db:
        return da < db
    return a.order < b.order


def best_in_grid(bad_grid, good_grid, allowed, B, G, prefix, reverse=False):
    """The best allowed cell: highest float score within NEAR, then exact. Cells are enumerated in C order (reverse:
    from the last), which is the documented order for every grid."""
    score = np.where(allowed, bad_grid / B - good_grid / G, -np.inf)
    smax = float(score.max())
    if not np.isfinite(smax):
        return None
    cand = np.flatnonzero(score >= smax - NEAR)
    if reverse:
        cand = cand[::-1]
    bg, gg = bad_grid.ravel(), good_grid.ravel()
    best = None
    for pos, k in enumerate(cand.tolist()):
        c = Cand(bg[k], gg[k], prefix + (pos,), k)
        if best is None or better(c, best, B, G):
            best = c
    return best


# ---------------------------------------------------------------- the searches (every candidate scored on validation)

def model_risk_table(r_train, r_val, bad_d, good_d):
    """Every distinct training risk value v (the minimum excluded) with its validation declined dollars under
    'decline if risk >= v'. Returns (values ascending, bad dollars, good dollars, loans declined)."""
    u = np.unique(r_train)[1:]
    order = np.argsort(-r_val, kind="stable")
    asc = r_val[order][::-1]
    cb_, cg_ = np.cumsum(bad_d[order]), np.cumsum(good_d[order])
    k = len(r_val) - np.searchsorted(asc, u, side="left")           # validation loans with risk >= v
    pick = np.maximum(k - 1, 0)
    return u, np.where(k > 0, cb_[pick], 0.0), np.where(k > 0, cg_[pick], 0.0), k


def single_table(pos, S, bad_d, good_d):
    """Declined dollars per cutoff index c (declines positions <= c), length S."""
    tb = np.bincount(pos, weights=bad_d, minlength=S)
    tg = np.bincount(pos, weights=good_d, minlength=S)
    return np.cumsum(tb), np.cumsum(tg)


def pair_grids(px, py, Sx, Sy, bad_d, good_d):
    """For each of the 2 pair forms: (bad grid, good grid) of declined dollars per cutoff pair, from one 2-D table."""
    idx = px.astype(np.int64) * Sy + py
    res = []
    for w in (bad_d, good_d):
        A = np.bincount(idx, weights=w, minlength=Sx * Sy).reshape(Sx, Sy).cumsum(0).cumsum(1)
        Cx, Cy = A[:, -1], A[-1, :]
        res.append([A, Cx[:, None] + Cy[None, :] - A])
    return list(zip(*res))


def trio_grids(px, py, pz, Sx, Sy, Sz, bad_d, good_d):
    """For each of the 9 trio forms: (bad grid, good grid) per cutoff triple, by inclusion-exclusion on one 3-D table."""
    idx = (px.astype(np.int64) * Sy + py) * Sz + pz
    res = []
    for w in (bad_d, good_d):
        A = np.bincount(idx, weights=w, minlength=Sx * Sy * Sz).reshape(Sx, Sy, Sz).cumsum(0).cumsum(1).cumsum(2)
        Dxy, Dxz, Dyz = A[:, :, -1][:, :, None], A[:, -1, :][:, None, :], A[-1, :, :][None, :, :]
        Cx, Cy, Cz = A[:, -1, -1][:, None, None], A[-1, :, -1][None, :, None], A[-1, -1, :][None, None, :]
        res.append([A,
                    Cx + Cy + Cz - Dxy - Dxz - Dyz + A,
                    Dxy + Dxz - A,
                    Dxy + Dyz - A,
                    Dxz + Dyz - A,
                    Dxy + Cz - A,
                    Dxz + Cy - A,
                    Dyz + Cx - A,
                    Dxy + Dxz + Dyz - 2 * A])
    return list(zip(*res))


def search_model(r_train, r_val, bad_d, good_d, B, G):
    u, tb, tg, k = model_risk_table(r_train, r_val, bad_d, good_d)
    best = best_in_grid(tb, tg, np.ones(len(u), dtype=bool), B, G, (), reverse=True)   # ties: the higher value first
    return best, float(u[best.where]), len(u)


def search_singles(ladders, vpos, bad_d, good_d, B, G):
    best, meta, n_cand = None, None, 0
    for i, lad in enumerate(ladders):
        allowed = lad.allowed_values()
        n_cand += int(allowed.sum())
        tb, tg = single_table(vpos[i], len(allowed), bad_d, good_d)
        c = best_in_grid(tb, tg, allowed, B, G, (i,))
        if c is not None and (best is None or better(c, best, B, G)):
            best, meta = c, i
    return best, meta, n_cand


def search_pairs(ladders, spos, allowed, bad_d, good_d, B, G):
    best, meta, n_cand = None, None, 0
    for pi, (i, j) in enumerate(itertools.combinations(range(NV), 2)):
        ax, ay = allowed[i], allowed[j]
        if not ax.any() or not ay.any():
            continue
        n_cand += int(ax.sum()) * int(ay.sum()) * len(PAIR_FORMS)
        grids = pair_grids(spos[i], spos[j], ladders[i].S, ladders[j].S, bad_d, good_d)
        al = ax[:, None] & ay[None, :]
        for r, (gb, gg) in enumerate(grids):
            c = best_in_grid(gb, gg, al, B, G, (pi, r))
            if c is not None and (best is None or better(c, best, B, G)):
                best, meta = c, (i, j, r)
    return best, meta, n_cand


def search_trios(ladders, spos, allowed, bad_d, good_d, B, G):
    best, meta, n_cand = None, None, 0
    trios = list(itertools.combinations(range(NV), 3))
    t0 = time.time()
    for ti, (i, j, k) in enumerate(trios):
        ax, ay, az = allowed[i], allowed[j], allowed[k]
        if not (ax.any() and ay.any() and az.any()):
            continue
        n_cand += int(ax.sum()) * int(ay.sum()) * int(az.sum()) * len(TRIO_FORMS)
        grids = trio_grids(spos[i], spos[j], spos[k], ladders[i].S, ladders[j].S, ladders[k].S, bad_d, good_d)
        al = ax[:, None, None] & ay[None, :, None] & az[None, None, :]
        for r, (gb, gg) in enumerate(grids):
            c = best_in_grid(gb, gg, al, B, G, (ti, r))
            if c is not None and (best is None or better(c, best, B, G)):
                best, meta = c, (i, j, k, r)
        if (ti + 1) % 56 == 0 or ti + 1 == len(trios):
            per = (time.time() - t0) / (ti + 1)
            progress(f"  trios: {ti + 1} of {len(trios)} searched; remaining ~{hms(per * (len(trios) - ti - 1))}")
    return best, meta, n_cand


# ---------------------------------------------------------------- rules: structure, wording, application

def make_rule(method, cand=None, **kw):
    """Internal rule: method index, type ('none', 'model', 'vars'), and the fields of that type."""
    rule = {"method": method, "type": "none", "candidates": 1, **kw}
    if cand is not None:
        rule["search_bad"], rule["search_good"] = cand.bad, cand.good
    return rule


def term_of(lad, kind, c):
    """One variable's part of a rule: kind 'value' (singles: every distinct training value) or 'step' (pairs, trios)."""
    notes, declined_blank, n_masked = lad.term_notes(c, kind)
    term = {"var": lad.index, "kind": kind, "cutoff": int(c), "blank_declined": bool(declined_blank), "note": "; ".join(notes), "groups_masked": n_masked}
    if lad.kind == "number":
        value = lad.rv[c] if kind == "value" else lad.mark_value(c)
        term.update(text=lad.term_text(value), value=value)
    else:
        term.update(text=lad.term_text(c), groups=lad.declined_names(c))
    return term


def rule_words(rule, ladders):
    if rule["type"] == "none":
        return "approve every loan (no cutoff)"
    if rule["type"] == "model":
        return f"decline if {rule['model']} adjusted predicted risk >= {100 * rule['cutoff']:.6f}% (exact value in chosen_rules.json)"
    texts = [t["text"] for t in rule["terms"]]
    form = rule["form_text"]
    if form == "at least 2 of 3":
        body = "at least 2 of: " + "; ".join(texts)
    else:
        body = form.replace("X", "{0}").replace("Y", "{1}").replace("Z", "{2}").format(*texts)
    return f"decline if {body} ({'; '.join(t['note'] for t in rule['terms'])})"


def declined_mask(rule, num, cat, risk_lr, risk_gbm, ladders):
    """The loans a rule declines, from the arrays of a loan set: the same ladders as the search, no tables."""
    n = num.shape[0]
    if rule["type"] == "none":
        return np.zeros(n, dtype=bool)
    if rule["type"] == "model":
        return (risk_lr if rule["model"] == LR else risk_gbm) >= rule["cutoff"]
    masks = []
    for t in rule["terms"]:
        lad = ladders[t["var"]]
        spos, vpos = lad.positions(values_of(lad, num, cat))
        masks.append((vpos if t["kind"] == "value" else spos) <= t["cutoff"])
    if len(masks) == 1:
        return masks[0]
    return rule["form_fn"](*masks)


def outcome_of(mask, y, upb):
    """Counts and dollars of (bad declined, good declined, bad approved, good approved)."""
    bad = y == 1
    cells = {}
    for name, sel in (("bad_declined", mask & bad), ("good_declined", mask & ~bad), ("bad_approved", ~mask & bad), ("good_approved", ~mask & ~bad)):
        cells[name] = (int(sel.sum()), int(upb[sel].sum()))
    return cells


# ---------------------------------------------------------------- the trade-off curve

def curve_line(prim, sec, y, upb, B, G):
    """Points at every 0.1% of the loans, ordered riskiest first by (prim, sec); ties never split: the nearest group
    boundary (equidistant: the smaller count). Rows: [mark %, loans, good loans, good $, bad loans, bad $]."""
    n = len(y)
    order = np.lexsort((sec, prim))
    p, s = prim[order], sec[order]
    change = np.flatnonzero((p[1:] != p[:-1]) | (s[1:] != s[:-1])) + 1
    bounds = np.concatenate([[0], change, [n]])
    bad = (y[order] == 1)
    cum_bad_n, cum_good_n = np.concatenate([[0], np.cumsum(bad)]), np.concatenate([[0], np.cumsum(~bad)])
    cum_bad_d = np.concatenate([[0], np.cumsum(np.where(bad, upb[order], 0))])
    cum_good_d = np.concatenate([[0], np.cumsum(np.where(bad, 0, upb[order]))])
    rows = []
    for q in range(1, CURVE_POINTS + 1):
        target = q * n / CURVE_POINTS
        i = np.searchsorted(bounds, target, side="left")
        lo, hi = bounds[max(i - 1, 0)], bounds[min(i, len(bounds) - 1)]
        k = int(lo if target - lo <= hi - target else hi)
        rows.append([q / 10, k, int(cum_good_n[k]), int(cum_good_d[k]), int(cum_bad_n[k]), int(cum_bad_d[k])])
    return rows


def curve_row_cells(loans, good_n, good_d, bad_n, bad_d, B, G):
    return [m(loans), m(good_n), dollars(good_n, good_d), share(good_n, good_d, G), m(bad_n), dollars(bad_n, bad_d), share(bad_n, bad_d, B)]


# ---------------------------------------------------------------- report sections

def rep_header(args, meta, stamp, saved):
    rule = saved["rule"]
    out(f"credit_box.py v{SCRIPT_VERSION}  {datetime.now():%Y-%m-%d %H:%M}  mode {MODE}  dataset {fm.P['db'].name} (read-only)  "
        f"saved models {P['models'].relative_to(WORK_ROOT)}" + ("  RESCORED (previous results renamed with the stamp " + stamp + ")" if stamp else ""))
    if MODE != "smoke":
        out(st.RESCORED_LINE)
    out("Summaries only: no loan IDs, no row-level values. Counts from 1 to 10 show as '1-10'; a dollar sum, rate, share or "
        "score whose loan count is 1-10 is not shown ('-'). A group whose bad rate is hidden (its loan, bad or good count is "
        "1-10) has its position hidden too: a number's blank group shows 'placement masked', a category with such a group "
        "lists its other groups in order without rank numbers with its blank placement masked, and a rule never names that group "
        "and shows the variable's blank treatment as 'masked'.")
    out(f"Dataset built by build_dataset.py v{meta.get('script_version', '?')} at {meta.get('built_at', '?')}; specification file SHA-256 "
        f"{meta.get('decision_log_sha256', '?')}. DuckDB memory limit {args.memory_limit}, threads {args.threads}.")
    out("Q3: six methods, cutoffs from TRAINING loans, every candidate scored on "
        "VALIDATION loans (score = % of bad-loan dollars declined - % of good-loan dollars declined, each loan's original "
        "amount, the population's own totals), the best per method kept, the winner by validation score, the chosen rules "
        "written to chosen_rules.json before any test outcome is read, then applied unchanged to the TEST loans.")
    out("Candidates: model risk, every distinct adjusted predicted risk among training loans ('decline if risk >= v'); single "
        "variable, every distinct training value (numbers) or each riskiest-first boundary (categories); pairs and trios, "
        "numbers at the merged 1%-99% marks of the training line-up (mark p = the loan at position ceil(p x N / 100)) and "
        "categories at their boundaries, in every combination and every and/or form. A cutoff that declines every loan with "
        "a value is not a candidate.")
    out("Directions and orders: a number's riskier end by its training AUC (above 0.5: high values riskier); categories "
        "riskiest first by training bad rate (ties: documented category order, blank last). Blanks: numbers, just riskier than the "
        "riskiest value of the first step whose training bad rate is below the blank group's (none: safest of all); a "
        "variable with no training blanks uses the overall training bad rate instead; categories with training blanks, their "
        "own group. Special program R and any value absent from training count as the reference category. Numbers are "
        "compared in float32 as the models saw them.")
    out("Ties: " + TIE_ORDER)
    out(f"Model risks as in score_test.py: Logistic Regression with lr_edge_rule.json result {rule['result']} "
        + ("(every number held at its training min / max when outside it)" if rule["result"] == "B" else "(inputs as stored)")
        + ", adjusted by that version's isotonic steps; LightGBM, the saved chosen model adjusted by its isotonic steps "
        "(isotonic_calibration.json).")
    out(f"Variables ({NV}): {', '.join(lab for _, lab, _, _ in fm.VARIABLES)}. {len(list(itertools.combinations(range(NV), 2)))} "
        f"pairs, {len(list(itertools.combinations(range(NV), 3)))} trios. Loan amount in DOLLARS is each loan's weight in the score, "
        "not a variable.")
    if MODE == "smoke":
        out(f"SMOKE MODE: a fixed 1-in-{fm.SMOKE_MOD} hash sample of every set and the models from outputs/models_smoke. The test "
            "phase runs for its checks only: no test result number is written anywhere, so a smoke run is rerunnable and "
            "overwrites its own _smoke outputs.")


def rep_variables(ladders, n_val_phantom):
    section("1. Variables: directions and training AUC (numbers), riskiest-first orders (categories), blank groups and their placement")
    rows, csv_rows = [], []
    for lad in ladders:
        if lad.kind != "number":
            continue
        rows.append([lad.label, "high values riskier" if lad.high else "low values riskier", f"{lad.auc:.6f}", f"{lad.D:,}", lad.M,
                     int(lad.allowed_steps().sum()), m(lad.blank_loans), m(lad.blank_bad) if lad.blank_loans else "0",
                     rate_cell(lad.blank_loans, lad.blank_bad) if lad.blank_loans else "-", lad.placement_text()])
        csv_rows.append([lad.label, "number", "direction", "", "high" if lad.high else "low", "", "", "",
                         f"training AUC {lad.auc:.6f}; {lad.D} distinct training values; {lad.M} steps; {int(lad.allowed_steps().sum())} usable in pairs and trios"])
        mr = ex.mask_row(MISSING, lad.blank_loans, lad.blank_bad)
        csv_rows.append([lad.label, "number", "blank group", "", MISSING, mr[1], mr[2], mr[3], lad.placement_text()])
    out_table(["number", "riskier end", "training AUC", "distinct values", "steps", "usable steps", "blank loans", "blank bad", "blank bad rate", "blank placement"],
              rows, left={0, 1, 9})
    out("  Steps: the merged 1% marks (steps.csv). Usable steps: step cutoffs that are candidates in pairs and trios.")
    for lad in ladders:
        if lad.kind != "category":
            continue
        n_hidden = sum(lad.hidden)
        out()
        out(f"--- {lad.label}: {lad.G} groups riskiest first" + (f" ({n_hidden} group{'s' if n_hidden > 1 else ''}'{'' if n_hidden > 1 else 's'} position masked: "
                                                                 "the other groups are listed in order without rank numbers)" if n_hidden else ""))
        table, masked_rows = [], []
        for r, (code, name, loans, bad) in enumerate(lad.groups):
            mr = ex.mask_row(name, loans, bad)
            if lad.hidden[r]:
                masked_rows.append([name, mr[1], mr[2] or "-"])
                csv_rows.append([lad.label, "category", "masked group", "", name, mr[1], mr[2], mr[3], "position masked"])
                continue
            table.append(["-" if n_hidden else r + 1, name, mr[1], mr[2] or "-", rate_cell(loans, bad)])
            csv_rows.append([lad.label, "category", "group", "" if n_hidden else r + 1, name, mr[1], mr[2], mr[3], ""])
        out_table(["rank", "group", "training loans", "bad loans", "bad rate"], table, left={1})
        if masked_rows:
            masked_rows.sort(key=lambda row: [g[1] for g in sorted(lad.groups)].index(row[0]))     # documented category (code) order, not risk order
            out_table(["position masked", "training loans", "bad loans"], masked_rows, left={0})
        note = lad.placement_text()
        if lad.phantom:
            mr = ex.mask_row(MISSING, lad.blank_loans, lad.blank_bad)
            out(f"  Blank group: {mr[1]} training loans; " + (note if note == "placement masked" else f"placement: {note}") + ".")
            csv_rows.append([lad.label, "category", "blank group", "", MISSING, mr[1], mr[2], mr[3], note])
    out()
    phantom = [lad for lad in ladders if lad.phantom]
    out("  A category with a hidden group shows 'placement masked' for its blank group (real or phantom) and 'masked' as its blank "
        "treatment in every rule: a placement among its groups would locate the hidden group.")
    out("  Variables with no training blanks (validation blanks placed by the overall training bad rate): "
        + ", ".join(f"{lad.label} {mc(n_val_phantom[lad.index])} validation loans" for lad in phantom) + ".")
    path = P["out"] / "variable_order.csv"
    ex.write_csv(path, ["variable", "kind", "item", "rank", "value_or_group", "training_loans", "bad_loans", "bad_rate_pct", "note"], csv_rows)
    WRITTEN.append(path)
    srows = []
    for lad in ladders:
        if lad.kind == "number":
            for s0 in range(lad.M):
                mr = ex.mask_row(f32text(lad.mark_value(s0)), int(lad.step_loans[s0]), int(lad.step_bad[s0]))
                srows.append([lad.label, s0 + 1, mr[0], mr[1], mr[2], mr[3], "yes" if lad.allowed_steps()[s0] else "no"])
    path = P["out"] / "steps.csv"
    ex.write_csv(path, ["variable", "step", "mark_value", "training_loans", "bad_loans", "bad_rate_pct", "candidate"], srows)
    WRITTEN.append(path)


def rep_validation(rules, ladders, tot, winner):
    section("2. Validation: each method's chosen rule, its score, declined loans and dollars; candidates per method; the winner")
    n, b, B, G = tot["loans"], tot["bad"], tot["bad_dollars"], tot["good_dollars"]
    out(f"  Validation loans {mc(n)}, bad {mc(b)} ({rate_cell(n, b)}); bad-loan dollars {B:,}; good-loan dollars {G:,}. "
        "Score in percentage points.")
    rows, csv_rows = [], []
    for rule in rules:
        v = rule["validation"]
        bn, bdol = v["bad_declined"]
        gn, gdol = v["good_declined"]
        words = rule_words(rule, ladders)
        rows.append([METHODS[rule["method"]], f"{rule['candidates']:,}", score_text(bn, bdol, gn, gdol, B, G),
                     m(bn + gn), dollars(bn + gn, bdol + gdol), m(bn), dollars(bn, bdol), share(bn, bdol, B), m(gn), dollars(gn, gdol), share(gn, gdol, G),
                     "winner" if rule["method"] == winner else ""])
        csv_rows.append([METHODS[rule["method"]], words, rule["candidates"], score_text(bn, bdol, gn, gdol, B, G), m(bn + gn), dollars(bn + gn, bdol + gdol),
                         m(bn), dollars(bn, bdol), share(bn, bdol, B), m(gn), dollars(gn, gdol), share(gn, gdol, G), "winner" if rule["method"] == winner else ""])
    out_table(["method", "candidates", "score", "declined loans", "declined $", "bad declined", "bad declined $", "bad $ %", "good declined", "good declined $", "good $ %", ""],
              rows, left={0, 11})
    out()
    for rule in rules:
        out(f"  {METHODS[rule['method']]}: {rule_words(rule, ladders)}")
    out()
    out(f"  Winning method (validation): {METHODS[winner]}.")
    path = P["out"] / "validation_results.csv"
    ex.write_csv(path, ["method", "rule", "candidates", "score_pct_points", "declined_loans", "declined_dollars", "bad_declined_loans", "bad_declined_dollars",
                        "bad_dollars_declined_pct", "good_declined_loans", "good_declined_dollars", "good_dollars_declined_pct", "winner"], csv_rows)
    WRITTEN.append(path)


def result_rows(name, rules, ladders, results, tot):
    B, G = tot["bad_dollars"], tot["good_dollars"]
    rows = []
    for rule in rules:
        c = results[rule["method"]]
        bn, bdol = c["bad_declined"]
        gn, gdol = c["good_declined"]
        an, adol = c["bad_approved"]
        hn, hdol = c["good_approved"]
        rows.append([name, METHODS[rule["method"]], score_text(bn, bdol, gn, gdol, B, G), m(bn), dollars(bn, bdol), m(gn), dollars(gn, gdol),
                     m(an), dollars(an, adol), m(hn), dollars(hn, hdol)])
    return rows


def rep_test_header():
    """Section 3's header, written before the first test outcome is read (the test-phase check lines follow it)."""
    if MODE == "smoke":
        section("3. Test (SMOKE MODE): the test phase runs on the sample for its checks only; no test result number is written")
        out("  The checks below cover the plumbing (rules applied, cells add up, agencies add up to all loans, the curve's shape).")
    else:
        section("3. Test: each chosen rule applied unchanged to the test loans, all loans then each agency (no winner is picked here)")
        out("  Score in percentage points with the population's own bad and good dollar totals. Counts and dollars of the four cells.")


def rep_test(rules, ladders, pops, curve, n_test_phantom):
    phantom = [lad for lad in ladders if lad.phantom]
    out("  Test blanks in variables with no training blanks (placed by the overall training bad rate): "
        + ", ".join(f"{lad.label} {mc(n_test_phantom[lad.index])}" for lad in phantom) + ".")
    rows = []
    for name, tot, results in pops:
        out()
        out(f"--- {name}: {mc(tot['loans'])} loans, bad {mc(tot['bad'])} ({rate_cell(tot['loans'], tot['bad'])}); bad-loan dollars {tot['bad_dollars']:,}, "
            f"good-loan dollars {tot['good_dollars']:,}")
        rr = result_rows(name, rules, ladders, results, tot)
        rows += rr
        out_table(["population", "method", "score", "bad declined", "bad declined $", "good declined", "good declined $", "bad approved", "bad approved $",
                   "good approved", "good approved $"], rr, left={0, 1})
    path = P["out"] / "test_results.csv"
    ex.write_csv(path, ["population", "method", "score_pct_points", "bad_declined_loans", "bad_declined_dollars", "good_declined_loans", "good_declined_dollars",
                        "bad_approved_loans", "bad_approved_dollars", "good_approved_loans", "good_approved_dollars"], rows)
    WRITTEN.append(path)
    lines, dots = curve
    out()
    out(f"  Trade-off curve (data only, no chart): {len(lines)} rows in curve_lines.csv ({CURVE_POINTS} points per line for "
        f"{LR} risk, {GBM} risk and the chosen single variable) and {len(dots)} dots in curve_dots.csv (the six chosen rules on all test loans; no cutoff at 0, 0).")
    path = P["out"] / "curve_lines.csv"
    ex.write_csv(path, ["line", "pct_of_loans_mark", "loans_declined", "good_loans_declined", "good_dollars_declined", "good_dollars_pct", "bad_loans_caught",
                        "bad_dollars_caught", "bad_dollars_pct"], lines)
    WRITTEN.append(path)
    path = P["out"] / "curve_dots.csv"
    ex.write_csv(path, ["method", "rule", "loans_declined", "good_loans_declined", "good_dollars_declined", "good_dollars_pct", "bad_loans_caught",
                        "bad_dollars_caught", "bad_dollars_pct"], dots)
    WRITTEN.append(path)


def rep_files():
    section("4. Files written")
    for p in WRITTEN:
        out(f"  {p.relative_to(WORK_ROOT)}")


# ---------------------------------------------------------------- main

def load_models():
    """Everything the scoring path needs, from the saved files (as score_test.py loads them)."""
    lr, cal_saved, gbm_settings, ranges = cb.load_saved()
    if not cb.P["rule"].exists():
        progress(f"STOP: {cb.P['rule']} is missing; run check_beyond_range.py first" + (" with --smoke" if MODE == "smoke" else "") + ".")
        raise StopRun(1)
    rule = json.loads(cb.P["rule"].read_text(encoding="utf-8"))
    if "calibration" not in rule:
        progress(f"STOP: {cb.P['rule']} has no isotonic calibration (made by an earlier check_beyond_range.py); rerun it.")
        raise StopRun(1)
    booster = lgb.Booster(model_file=str(P["models"] / "lightgbm_model.txt"))
    return {"lr": lr, "cal": cal_saved, "settings": gbm_settings, "ranges": ranges, "rule": rule, "booster": booster}


def dollars_of(amounts, y, where):
    """Whole-dollar amounts as int64 and the float64 bad / good dollar weights; stops on a blank or fractional amount.
    amounts: the loan amount in dollars (fit_models' 'dollars' column, read beside the variables; never a model input)."""
    upb = np.asarray(amounts, dtype=np.float64)
    ok = not np.isnan(upb).any() and bool(np.all(upb == np.floor(upb)))
    bd.check_line(f"{where}: every loan amount is a whole number of dollars and never blank", ok)
    if not ok:
        progress(f"STOP: {where} has a blank or fractional loan amount; the dollar score is undefined.")
        raise StopRun(1)
    upb = upb.astype(np.int64)
    bad = y == 1
    return upb, np.where(bad, upb, 0).astype(np.float64), np.where(bad, 0, upb).astype(np.float64)


def totals_of(y, upb):
    bad = y == 1
    return {"loans": int(len(y)), "bad": int(bad.sum()), "bad_dollars": int(upb[bad].sum()), "good_dollars": int(upb[~bad].sum())}


def rule_json(rule, ladders, tot):
    """The JSON form of a rule: structure, words, validation result (counts masked, dollars hidden with them)."""
    v = rule["validation"]
    B, G = tot["bad_dollars"], tot["good_dollars"]
    bn, bdol = v["bad_declined"]
    gn, gdol = v["good_declined"]
    j = {"method": METHODS[rule["method"]], "variables_used": METHOD_VARS[rule["method"]], "rule": rule_words(rule, ladders), "type": rule["type"],
         "candidates": rule["candidates"],
         "validation": {"score_pct_points": None if masked(bn) or masked(gn) else 100 * (bdol / B - gdol / G),
                        "declined_loans": m(bn + gn), "declined_dollars": None if masked(bn + gn) else bdol + gdol,
                        "bad_declined_loans": m(bn), "bad_declined_dollars": None if masked(bn) else bdol,
                        "good_declined_loans": m(gn), "good_declined_dollars": None if masked(gn) else gdol}}
    if rule["type"] == "model":
        j.update(model=rule["model"], cutoff_adjusted_risk=rule["cutoff"], how="decline if the model's adjusted predicted risk >= cutoff_adjusted_risk")
    elif rule["type"] == "vars":
        terms = []
        for t in rule["terms"]:
            lad = ladders[t["var"]]
            d = {"variable": lad.label, "column": lad.col, "kind": lad.kind, "cutoff_text": t["text"],
                 "blank": "masked" if lad.placement_masked else ("declined" if t["blank_declined"] else "approved"),
                 "cutoff_basis": "every distinct training value" if t["kind"] == "value" else "1% step of training loans"}
            if lad.kind == "number":
                d.update(direction="high values riskier" if lad.high else "low values riskier", cutoff=float(f32text(t["value"])),
                         compare="value >= cutoff (float32)" if lad.high else "value <= cutoff (float32)")
            else:
                d.update(declined_groups=t["groups"], groups_masked=t["groups_masked"], absent_values="count as the reference category")
            terms.append(d)
        j.update(form=rule["form_text"], terms=terms)
    return j


def main(argv=None):
    args = st.parse_args(argv)
    configure("smoke" if args.smoke else "full")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    LOGS.mkdir(parents=True, exist_ok=True)
    check_raw.report.clear()
    FAILS.clear()
    WRITTEN.clear()
    started = time.time()
    progress("=" * 78)
    progress(f"credit_box.py v{SCRIPT_VERSION} started ({MODE} mode). Dataset {fm.P['db']} (read-only); saved models {P['models']}; outputs {P['out']}.")
    con, code = None, 0
    try:
        stamp = st.guard_test_once(args.rescore_test) if MODE != "smoke" else None    # smoke writes no test number: rerunnable
        if not fm.P["db"].exists():
            progress(f"STOP: no dataset at {fm.P['db']}.")
            raise StopRun(1)
        saved = load_models()
        con = ex.open_dataset(args)
        fm.create_views(con)
        P["out"].mkdir(parents=True, exist_ok=True)
        rep_header(args, ex.read_meta(con), stamp, saved)
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

        # training loans: ladders (directions, orders, steps, blank placement) and the model-risk cutoffs
        t0 = time.time()
        num, cat, y, more = fm.fetch_rows(con, "train", codes, True, extra=("dollars",))
        progress(f"Training loans read: {mc(len(y))} in {hms(time.time() - t0)}")
        dollars_of(more.pop("dollars"), y, "training")
        t0 = time.time()
        ladders = build_ladders(num, cat, y, codes, refs, present)
        undefined = [lad for lad in ladders if lad.kind == "number" and lad.high is None]
        bd.check_line("every number variable has a training AUC different from 0.5 (its riskier end is defined)", not undefined)
        for lad in undefined:
            out(f"  STOP: {lad.label}: training AUC {'undefined (one outcome only)' if lad.auc is None else 'exactly 0.5'}; the riskier end cannot be set.")
        if undefined:
            raise StopRun(1)
        progress(f"Ladders built in {hms(time.time() - t0)}")
        t0 = time.time()
        risk_lr_tr, risk_g_tr = adjusted_risks(num, cat, saved)
        progress(f"Training loans scored with both models in {hms(time.time() - t0)}")
        del num, cat, y

        # validation loans: every candidate scored
        num_v, cat_v, y_v, more = fm.fetch_rows(con, "validation", codes, True, extra=("dollars",))
        upb_v, bad_d, good_d = dollars_of(more.pop("dollars"), y_v, "validation")
        tot_v = totals_of(y_v, upb_v)
        B, G = tot_v["bad_dollars"], tot_v["good_dollars"]
        bd.check_line("validation has bad-loan and good-loan dollars (the score's denominators)", B > 0 and G > 0)
        if FAILS:
            raise StopRun(1)
        n_val_phantom = {lad.index: int((np.isnan(values_of(lad, num_v, cat_v)) if lad.kind == "number" else values_of(lad, num_v, cat_v) == -1).sum())
                         for lad in ladders if lad.phantom}
        bd.rsection("1 variables", rep_variables, ladders, n_val_phantom)
        t0 = time.time()
        risk_lr_v, risk_g_v = adjusted_risks(num_v, cat_v, saved)
        progress(f"Validation loans read and scored in {hms(time.time() - t0)}")
        spos, vpos, allowed = [], [], []
        for lad in ladders:
            s, v = lad.positions(values_of(lad, num_v, cat_v))
            spos.append(s)
            vpos.append(v)
            allowed.append(lad.allowed_steps())

        rules = [make_rule(0)]
        t0 = time.time()
        for method, name, r_tr, r_v in ((1, LR, risk_lr_tr, risk_lr_v), (2, GBM, risk_g_tr, risk_g_v)):
            n_values = len(np.unique(r_tr))
            bd.check_line(f"{name}: its adjusted training risks take at least two values (a cutoff candidate exists)", n_values >= 2)
            if n_values < 2:
                out(f"  STOP: {name}'s isotonic adjustment gives every training loan the same risk; no model-risk cutoff exists.")
                raise StopRun(1)
            cand, v, n_cand = search_model(r_tr, r_v, bad_d, good_d, B, G)
            rules.append(make_rule(method, cand, type="model", model=name, cutoff=v, candidates=n_cand))
        progress(f"Model-risk cutoffs searched in {hms(time.time() - t0)}")
        del risk_lr_tr, risk_g_tr
        t0 = time.time()
        cand, i, n_cand = search_singles(ladders, vpos, bad_d, good_d, B, G)
        rules.append(make_rule(3, cand, type="vars", terms=[term_of(ladders[i], "value", cand.where)], form_text="X", form_fn=None, candidates=n_cand))
        progress(f"Single-variable cutoffs searched in {hms(time.time() - t0)}")
        t0 = time.time()
        cand, (i, j, r), n_cand = search_pairs(ladders, spos, allowed, bad_d, good_d, B, G)
        cx, cy = np.unravel_index(cand.where, (ladders[i].S, ladders[j].S))
        rules.append(make_rule(4, cand, type="vars", terms=[term_of(ladders[i], "step", int(cx)), term_of(ladders[j], "step", int(cy))],
                               form_text=PAIR_FORMS[r][0], form_fn=PAIR_FORMS[r][1], candidates=n_cand))
        progress(f"Pair rules searched in {hms(time.time() - t0)}")
        t0 = time.time()
        cand, (i, j, k, r), n_cand = search_trios(ladders, spos, allowed, bad_d, good_d, B, G)
        cx, cy, cz = np.unravel_index(cand.where, (ladders[i].S, ladders[j].S, ladders[k].S))
        rules.append(make_rule(5, cand, type="vars", terms=[term_of(ladders[i], "step", int(cx)), term_of(ladders[j], "step", int(cy)), term_of(ladders[k], "step", int(cz))],
                               form_text=TRIO_FORMS[r][0], form_fn=TRIO_FORMS[r][1], candidates=n_cand))
        progress(f"Trio rules searched in {hms(time.time() - t0)}")

        # self-check: each chosen rule applied with a plain mask gives the search's validation dollars; counts from the mask
        for rule in rules:
            mask = declined_mask(rule, num_v, cat_v, risk_lr_v, risk_g_v, ladders)
            rule["validation"] = outcome_of(mask, y_v, upb_v)
            if "search_bad" in rule:
                bd.check_line(f"{METHODS[rule['method']]}: the chosen rule's validation dollars from a plain mask equal the search's",
                              rule["validation"]["bad_declined"][1] == rule["search_bad"] and rule["validation"]["good_declined"][1] == rule["search_good"])
        winner = 0
        for rule in rules[1:]:
            a = Cand(rule["validation"]["bad_declined"][1], rule["validation"]["good_declined"][1], (METHOD_VARS[rule["method"]], rule["method"]))
            w = rules[winner]
            b = Cand(w["validation"]["bad_declined"][1], w["validation"]["good_declined"][1], (METHOD_VARS[w["method"]], w["method"]))
            if better(a, b, B, G):
                winner = rule["method"]
        bd.rsection("2 validation", rep_validation, rules, ladders, tot_v, winner)
        chosen = {"script": f"credit_box.py v{SCRIPT_VERSION}", "written_at": datetime.now().isoformat(timespec="seconds"), "mode": MODE,
                  "note": "written before any test outcome is read; the rules below are applied unchanged to the test loans",
                  "score": "% of bad-loan dollars declined - % of good-loan dollars declined (original amounts; the population's own totals)",
                  "tie_order": TIE_ORDER,
                  "validation_totals": {"loans": m(tot_v["loans"]), "bad_loans": m(tot_v["bad"]), "bad_dollars": B, "good_dollars": G},
                  "methods": [rule_json(rule, ladders, tot_v) for rule in rules], "winner": METHODS[winner]}
        P["rules"].write_text(json.dumps(chosen, indent=1), encoding="utf-8")
        WRITTEN.append(P["rules"])
        LAST_RULES[str(WORK_ROOT)] = rules                      # tests only: the internal rules (cutoff indices behind masked terms)
        progress(f"Chosen rules written to {P['rules']} (no test outcome read so far). Winner on validation: {METHODS[winner]}.")

        # ---- the test phase: the only reads of the test outcome ----
        rep_test_header()
        st.create_test_views(con)
        t0 = time.time()
        parts = {agency: fm.fetch_rows(con, view, codes, True, extra=("dollars",)) for agency, view in AGENCIES}
        num_t = np.concatenate([parts[a][0] for a, _ in AGENCIES])
        dollars_t = np.concatenate([parts[a][3]["dollars"] for a, _ in AGENCIES])
        cat_t = np.concatenate([parts[a][1] for a, _ in AGENCIES])
        y_t = np.concatenate([parts[a][2] for a, _ in AGENCIES]).astype(np.int8)
        agency_of = np.concatenate([np.full(len(parts[a][2]), k, dtype=np.int8) for k, (a, _) in enumerate(AGENCIES)])
        del parts
        progress(f"Test loans read: {mc(len(y_t))} in {hms(time.time() - t0)}")
        bd.check_line("there are test loans of both agencies with both outcomes",
                      all((agency_of == k).sum() > y_t[agency_of == k].sum() > 0 for k in range(len(AGENCIES))))
        upb_t, _, _ = dollars_of(dollars_t, y_t, "test")
        risk_lr_t, risk_g_t = adjusted_risks(num_t, cat_t, saved)
        n_test_phantom = {lad.index: int((np.isnan(values_of(lad, num_t, cat_t)) if lad.kind == "number" else values_of(lad, num_t, cat_t) == -1).sum())
                          for lad in ladders if lad.phantom}
        masks = {rule["method"]: declined_mask(rule, num_t, cat_t, risk_lr_t, risk_g_t, ladders) for rule in rules}
        pops = []
        for name, sel in [("all test loans", np.ones(len(y_t), dtype=bool))] + [(a, agency_of == k) for k, (a, _) in enumerate(AGENCIES)]:
            tot = totals_of(y_t[sel], upb_t[sel])
            pops.append((name, tot, {k: outcome_of(mask[sel], y_t[sel], upb_t[sel]) for k, mask in masks.items()}))
        ok_cells = all(sum(c[0] for c in res.values()) == tot["loans"] and sum(c[1] for c in res.values()) == tot["bad_dollars"] + tot["good_dollars"]
                       for _, tot, results in pops for res in results.values())
        ok_sum = all(sum(pops[a][2][k][cell][0] for a in (1, 2)) == pops[0][2][k][cell][0] and sum(pops[a][2][k][cell][1] for a in (1, 2)) == pops[0][2][k][cell][1]
                     for k in masks for cell in ("bad_declined", "good_declined", "bad_approved", "good_approved"))
        B_t, G_t = pops[0][1]["bad_dollars"], pops[0][1]["good_dollars"]
        single = ladders[rules[3]["terms"][0]["var"]]
        lines = []
        for name, (prim, sec) in ((f"{LR} risk", (-risk_lr_t, np.zeros(len(y_t), dtype=np.int8))), (f"{GBM} risk", (-risk_g_t, np.zeros(len(y_t), dtype=np.int8))),
                                  (f"single variable: {single.label}", single.curve_keys(values_of(single, num_t, cat_t)))):
            for row in curve_line(prim, sec, y_t, upb_t, B_t, G_t):
                lines.append([name, row[0]] + curve_row_cells(*row[1:], B_t, G_t))
        dots = []
        for rule in rules:
            c = pops[0][2][rule["method"]]
            bn, bdol = c["bad_declined"]
            gn, gdol = c["good_declined"]
            dots.append([METHODS[rule["method"]], rule_words(rule, ladders)] + curve_row_cells(bn + gn, gn, gdol, bn, bdol, B_t, G_t))
        ok_curve = len(lines) == 3 * CURVE_POINTS and len(dots) == len(METHODS)
        if MODE != "smoke":
            bd.rsection("3 test", rep_test, rules, ladders, pops, (lines, dots), n_test_phantom)
        bd.check_line("every chosen rule was applied to the test loans and its four cells add up to the population (all loans and each agency)", ok_cells)
        bd.check_line("the two agencies' cells add up to the all-loans cells", ok_sum)
        bd.check_line(f"the curve has {CURVE_POINTS} points for each of 3 lines and {len(METHODS)} dots", ok_curve)
        if MODE == "smoke":
            bd.check_line("smoke mode wrote no test result file", not any((P["out"] / f).exists() for f in ("test_results.csv", "curve_lines.csv", "curve_dots.csv")))
        rep_files()
        section(f"SUMMARY: {len(FAILS)} problem(s)")
        for f in FAILS:
            out(f"  FAIL: {f}")
        out("  SUCCESS: every output written; every check passed." if not FAILS else "  FAILURE: see the problems above.")
        out(f"(credit_box took {hms(time.time() - started)})")
        code = 1 if FAILS else 0
    except StopRun as stop:
        code = stop.code
    except KeyboardInterrupt:
        progress("Interrupted. Nothing to resume: rerun; use --rescore-test only if results were written.")
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
