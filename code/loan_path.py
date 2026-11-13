"""The monthly-path rules in plain Python, plus the order of evaluation.

This module is the single place where the path definitions, the relief and freeze rules and the path rules
(12, 14-16, 18-20, 23-26, 28-30) are written as code. The numbers are the rule IDs recorded in the decision column and the f_N flags.
build_dataset.py runs it on every loan whose window is not trivial, and derives its SQL fast path and
its SQL decision step from the same constants.

It never sees raw rows. Its inputs are per-loan facts the build derives from the monthly rows:

  status48   48 characters, two per payment 1-24 (payment k = characters 2k-2 and 2k-1):
               '00'-'99'  the reported delinquency status: a KNOWN month
               'RA'       Freddie REO status (rule 24): neither known nor unknown; the loan is bad
               'XX'       XX or blank status on a row that is not the termination row: an UNKNOWN month
               '??'       an unparseable status on such a row: an UNKNOWN month
               '..'       no row for that payment: an UNKNOWN month (a start-of-loan, inner or trailing run)
               'TX'       the termination row itself when its status is XX, blank or unparseable: that
                          status is ignored and the ending code decides (rule 16)
               'TT'       a payment after the termination row (the loan has ended)
               '__'       a payment after the data's last month (not observable)
  relief24   24 characters: 'F' if the row at that payment carries BAP F, R or T or the Freddie disaster
             flag Y, '.' otherwise. Relief counts only on known months.
  zb_pay     payment number of the termination row (None if the loan has no zero-balance code); it can be
             0 or less (ended before payment 1) or more than 24 (ended after the window)
  zb_code    the zero-balance code on that row
  fc_pay     Fannie foreclosure date as a payment number (None if none; Freddie always None)
  first_pay  payment number of the loan's first row (any payment, also before 1)
  last_pay   payment number of the loan's last row
  w_rows     number of rows with payments 1-24
  de_pay     payment number of the data's last month for this loan

Conventions:
  * the termination row's numeric status is a known month like any other, so a proven 90+, an RA month or a
    foreclosure date ON the 96/06/16 row counts as "before the buyback or sale";
  * the counted status before an episode is 0 whenever no known window month precedes it;
  * rule 12 looks at the loan's first row overall;
  * the basic test of a run uses the counted status of the last known month before it, also after an
    episode;
  * RA months are transparent: they neither form nor break a run, are never relief and are not a "last
    known status";
  * months after a termination row are not unknown months; for a start-of-loan run the reported status
    before the run is 0; a run that follows a relief episode gets no add-on whether or not the next known
    month is a jump; an unknown month interrupts an episode (the next relief month starts a new one).
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------- order of evaluation

HARD_ORDER = ["9", "10", "1", "2", "3", "11", "12", "20", "39", "42R1"]   # hard drops: the first that applies
BAD_ORDER = ["23", "24", "25", "26"]                                      # bad if any applies
SOFT_ORDER = ["15", "16", "18", "19", "29", "30"]                         # soft drops: the first that applies
GOOD_ORDER = ["14", "28"]                                                 # good
ALL_DECISIONS = HARD_ORDER + BAD_ORDER + SOFT_ORDER + GOOD_ORDER
NO_RULE = "NO RULE"


def flag_name(decision):
    """Column name of a decision's flag: f_1 ... f_42r1."""
    return "f_" + decision.lower()


REASON_FLAGS = [flag_name(d) for d in ALL_DECISIONS]
INFO_FLAGS = ["f_28_trailing", "f_42r2_cltv_lt_ltv", "f_freeze", "f_freeze_gap", "f_jump_any", "f_relief_any"]
# Columns the SQL fast path and this module must both fill with identical values.
PATH_COLUMNS = ["proven_first", "jump3_any", "jump_any", "relief_any", "ra_first", "sn_gt_pay",
                "run_fail_15", "run_fail_16", "trailing_case", "trailing_pass", "freeze_applied", "freeze_gap"]

PAYOFF_CODE = "01"                      # rule 14
BAD_CODES = ("02", "03", "09", "15")    # rule 26
BUYBACK_CODE = {"fre": "96", "fan": "06"}   # rule 29
RPL_CODE = "16"                         # rule 30
KNOWN_CODES = {"fre": {PAYOFF_CODE, RPL_CODE, BUYBACK_CODE["fre"], *BAD_CODES},
               "fan": {PAYOFF_CODE, RPL_CODE, BUYBACK_CODE["fan"], *BAD_CODES}}

END_TOKENS = ("TT", "__", "TX")
UNKNOWN_TOKENS = ("XX", "??", "..")
NO_RELIEF = "." * 24


@dataclass
class PathResult:
    proven_first: object = None     # payment of the first proven 90+ (counted 3+, not a jump), or None
    jump3_any: bool = False         # a month with counted status 3+ that is a jump month (rule 19)
    jump_any: bool = False          # any jump month
    relief_any: bool = False        # any known relief month
    ra_first: object = None         # payment of the first RA month, or None
    sn_gt_pay: bool = False         # a known month whose reported status exceeds its payment number (20)
    run_fail_15: bool = False       # a failed run holding at least one missing month (15)
    run_fail_16: bool = False       # a failed run made only of XX/blank/unparseable months (16)
    trailing_case: bool = False     # records stop before payment 24, no termination code, data goes on (28)
    trailing_pass: bool = False     # ... and that trailing run passed the basic test
    freeze_applied: bool = False    # the 32 freeze rule was applied at least once
    freeze_gap: bool = False        # ... with a run of unknown months between the episode and the jump month
    counted: list = field(default_factory=lambda: [None] * 24)   # counted status per payment (known months)
    jumps: list = field(default_factory=lambda: [False] * 24)    # jump month per payment

    def columns(self):
        return {c: getattr(self, c) for c in PATH_COLUMNS}


def tokens(status48):
    if len(status48) != 48:
        raise ValueError("status48 must hold 48 characters")
    return [status48[2 * i:2 * i + 2] for i in range(24)]


def walk(status48, relief24, zb_pay, last_pay, de_pay):
    """One pass over payments 1-24: runs (15, 16), episodes, backlog and counted status (relief episodes with the
    freeze rule), jump months and proven 90+ (19, 23), RA (24), the trailing-run case (28)."""
    toks = tokens(status48)
    if len(relief24) != 24:
        raise ValueError("relief24 must hold 24 characters")
    res = PathResult()
    backlog = 0
    in_ep = False            # inside a relief episode
    before = 0               # counted status before the current (or just-ended) episode
    last_relief_r = 0        # reported status at the episode's last relief month
    ep_pending = False       # an episode has ended; the next known month gets the freeze-rule check
    prev = None              # (pay, reported, counted, relief) of the last known month
    lk_r, lk_pay = 0, 0      # last known reported status and its payment; 0 at payment 0 before the first
    run = None               # [length, holds a missing month] of the open run of unknown months
    gap_since_known = False  # unknown months seen since the last known month
    last_run_failed = None

    def close_run(r_after):
        """Test the open run (if any). r_after = reported status of the known month that closes it, or None
        when the observation ends: then there is no add-on (15)."""
        nonlocal run, last_run_failed
        if run is None:
            return
        length, has_missing = run
        run = None
        if prev is None:                 # a run at the start of the loan
            c_before, r_before, rel_before = 0, 0, False
        else:
            _, r_before, c_before, rel_before = prev
        fail = c_before + length >= 3    # basic test, on counted status
        if not fail and r_after is not None and not rel_before:   # add-on, on reported status; none after relief
            fail = r_after > r_before + length + 1
        last_run_failed = fail
        if fail:
            if has_missing:
                res.run_fail_15 = True
            else:
                res.run_fail_16 = True

    for i, tok in enumerate(toks):
        k = i + 1
        if tok in END_TOKENS:
            break
        if tok == "RA":
            if res.ra_first is None:
                res.ra_first = k
            continue
        if tok in UNKNOWN_TOKENS:
            if run is None:
                run = [0, False]
            run[0] += 1
            if tok == "..":
                run[1] = True
            gap_since_known = True
            if in_ep:                    # an unknown month is never relief: the episode ends here
                in_ep = False
                backlog = max(last_relief_r - before, 0)
                ep_pending = True
            continue
        r = int(tok)
        close_run(r)
        if r > k:
            res.sn_gt_pay = True
        jump = r > lk_r + (k - lk_pay)
        rf = relief24[i] == "F"
        if in_ep and not rf:             # the episode ended at the previous known month
            in_ep = False
            backlog = max(last_relief_r - before, 0)
            ep_pending = True
        froze = False
        if ep_pending:                   # first known month after an episode
            ep_pending = False
            if jump:                     # freeze rule: this month is the episode's end for the backlog
                backlog = max(r - before, 0)
                c = before
                froze = True
                res.freeze_applied = True
                if gap_since_known:
                    res.freeze_gap = True
        if rf:
            res.relief_any = True
            if not in_ep:
                in_ep = True
                before = prev[2] if prev is not None else 0
            if not froze:
                c = min(r, before)       # payments skipped during relief do not count
            last_relief_r = r
        elif not froze:
            backlog = min(backlog, r)    # the backlog only shrinks (freeze rule)
            c = r - backlog
        res.counted[i] = c
        res.jumps[i] = jump
        if jump:
            res.jump_any = True
        if c >= 3:
            if jump:
                res.jump3_any = True
            elif res.proven_first is None:
                res.proven_first = k
        prev = (k, r, c, rf)
        lk_r, lk_pay = r, k
        gap_since_known = False

    last_run_failed = None
    close_run(None)
    res.trailing_case = zb_pay is None and last_pay is not None and last_pay < 24 and de_pay >= 24
    res.trailing_pass = bool(res.trailing_case and last_run_failed is False)
    return res


def is_fast(status48, relief24):
    """The SQL fast path handles a loan whose window has no unknown month, no RA month, no relief month and
    no reported status above 2: none of 15, 16, 19, 23, 24, 28 (trailing) or the freeze rule can apply."""
    if relief24 != NO_RELIEF:
        return False
    for tok in tokens(status48):
        if tok in UNKNOWN_TOKENS or tok == "RA":
            return False
        if tok[0].isdigit() and int(tok) > 2:
            return False
    return True


def fast_result(jump_any, sn_gt_pay):
    """What walk() returns for a fast-path loan; jump_any and sn_gt_pay come from the SQL pass."""
    return PathResult(jump_any=bool(jump_any), sn_gt_pay=bool(sn_gt_pay))


def ending_of(zb_pay):
    if zb_pay is None:
        return None
    return "before1" if zb_pay <= 0 else "window" if zb_pay <= 24 else "after"


def outcome_flags(agency, path, zb_pay, zb_code, fc_pay, first_pay, last_pay, w_rows, de_pay):
    """The flags of rules 12, 14-16, 18-20, 23-26, 28-30 for one loan (value flags 1, 2, 3, 9, 10, 11,
    39 and 42 R1 come from the loan-level values and are added by the caller)."""
    ending = ending_of(zb_pay)
    by_code = ending in ("before1", "window")
    f = {}
    f["f_12"] = ending != "before1" and (first_pay is None or first_pay >= 4 or not w_rows)
    f["f_20"] = bool(path.sn_gt_pay)
    f["f_14"] = by_code and zb_code == PAYOFF_CODE
    f["f_26"] = by_code and zb_code in BAD_CODES
    f["f_29"] = by_code and zb_code == BUYBACK_CODE[agency]
    f["f_30"] = by_code and zb_code == RPL_CODE
    reached_24 = last_pay is not None and last_pay >= 24 and (zb_pay is None or zb_pay > 24)
    f["f_28_trailing"] = bool(path.trailing_case)
    f["f_28"] = reached_24 or bool(path.trailing_case)
    f["f_18"] = zb_pay is None and de_pay < 24
    f["f_15"] = bool(path.run_fail_15)
    f["f_16"] = bool(path.run_fail_16)
    f["f_23"] = path.proven_first is not None
    f["f_19"] = bool(path.jump3_any) and not f["f_23"]
    f["f_24"] = path.ra_first is not None
    f["f_25"] = fc_pay is not None and 1 <= fc_pay <= 24
    # For 29/30 the proof must come on or before the buyback or sale. A proven 90+ or RA month is always
    # on or before the termination row (no rows exist after it); only the foreclosure date can be later.
    f["bad_25_timed"] = f["f_25"] and (not (f["f_29"] or f["f_30"]) or fc_pay <= zb_pay)
    f["f_freeze"] = bool(path.freeze_applied)
    f["f_freeze_gap"] = bool(path.freeze_gap)
    f["f_jump_any"] = bool(path.jump_any)
    f["f_relief_any"] = bool(path.relief_any)
    return f


@dataclass(frozen=True)
class Decision:
    place: str          # dropped / bad / good / NO RULE
    decision: object    # the first reason recorded, as text ('9', '42R1', '23', '14' ...), or None
    bad_rules_all: object  # every bad rule that applies ('23+26'), bad loans only


def is_bad(f):
    return bool(f.get("f_23") or f.get("f_24") or f.get("bad_25_timed") or f.get("f_26"))


def decide(f):
    """Order of evaluation: hard drops in order, then bad, then soft drops in order, then good; else no rule."""
    for d in HARD_ORDER:
        if f.get(flag_name(d)):
            return Decision("dropped", d, None)
    bad = [d for d in BAD_ORDER if (f.get("bad_25_timed") if d == "25" else f.get(flag_name(d)))]
    if bad:
        return Decision("bad", bad[0], "+".join(bad))
    for d in SOFT_ORDER:
        if f.get(flag_name(d)):
            return Decision("dropped", d, None)
    for d in GOOD_ORDER:
        if f.get(flag_name(d)):
            return Decision("good", d, None)
    return Decision(NO_RULE, None, None)
