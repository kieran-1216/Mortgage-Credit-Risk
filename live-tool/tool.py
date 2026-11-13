"""The live tool's logic, without Streamlit: load the saved models and the summary files, turn an application into the
models' 17 inputs, score it exactly as score_test.py scores a test loan, and build the rank, the per-loan SHAP shares
and the trade-off-curve readout. Research demo only; not for lending decisions.

Runs WITHOUT the dataset. Reads only (folders next to this file):
  models/  logistic_regression.json, lr_edge_rule.json, isotonic_calibration.json, lightgbm_model.txt,
           lightgbm_settings.json, beyond_range.csv, q1_auc.csv
  data/    app_summary.json, rank_lightgbm.csv, rank_logistic_regression.csv, curve_lightgbm.csv
           (written by build_app_summary.py)
and imports model_core.py from the repository's code/ folder (numpy only) for the scoring. The percent of limit, the
at-the-limit flag and the term group are computed by model_core's SQL text in an in-memory DuckDB (one row, no file),
so the arithmetic is the dataset's own.
"""

import csv
import json
import math
import sys
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))
import model_core as core  # noqa: E402

LR, GBM = "Logistic Regression", "LightGBM"
YEAR = 2023
QUARTER = "2023Q1"          # any 2023 quarter: the limit year is the year of the origination quarter
TOP = 10
OTHER = "All other variables"

# ---------------------------------------------------------------- plain labels

NUMBER_FIELDS = [            # (input key, label, model column or None)
    ("loan_amount", "Loan amount ($)", None),
    ("loan_term", "Loan term (months)", "loan_term"),
    ("interest_rate", "Interest rate (%)", None),
    ("credit_score", "Credit score", "credit_score"),
    ("dti", "Debt-to-income, DTI (%)", "dti"),
    ("cltv", "Combined loan-to-value, CLTV (%)", "cltv"),
    ("mi_pct", "Mortgage insurance coverage (%)", "mi_pct"),
]
CATEGORY_FIELDS = {          # model column -> (label, [(value, plain label)])
    "loan_purpose": ("Loan purpose", [("purchase", "Purchase"), ("cash-out refinance", "Cash-out refinance"),
                                      ("no-cash-out refinance", "No-cash-out refinance")]),
    "num_units": ("Number of units", [("1", "1"), ("2", "2"), ("3", "3"), ("4", "4")]),
    "property_type": ("Property type", [("SF", "Single-family"), ("PU", "Planned unit development (PUD)"), ("CO", "Condominium"),
                                        ("CP", "Co-op"), ("MH", "Manufactured home")]),
    "num_borrowers": ("Number of borrowers", [("1", "1"), ("2 or more", "2 or more")]),
    "first_time_homebuyer": ("First-time homebuyer", [("Y", "Yes"), ("N", "No")]),
    "occupancy": ("Occupancy", [("primary", "Primary residence"), ("second home", "Second home"), ("investor", "Investment property")]),
    "channel": ("Channel", [("retail", "Retail"), ("broker", "Broker"), ("correspondent", "Correspondent")]),
    "special_program": ("Special program", [("none", "None"), ("F", "F: Housing finance agency program"),
                                            ("H", "H: Affordable-lending program (Home Possible, HomeReady)"),
                                            ("R", "R: Refi Possible / RefiNow")]),
}
ROW_LABEL = {                # model variable label -> driver-table row label
    "credit score": "Credit score", "combined LTV": "Combined loan-to-value (CLTV)", "DTI": "Debt-to-income (DTI)",
    "rate spread": "Rate spread vs. 2023 median", "mortgage insurance %": "Mortgage insurance coverage",
    "loan amount % of limit": "Loan size vs. 2023 limit", "loan term": "Loan term", "loan purpose": "Loan purpose",
    "number of borrowers": "Number of borrowers", "occupancy": "Occupancy", "first-time homebuyer": "First-time homebuyer",
    "channel": "Channel", "property type": "Property type", "number of units": "Number of units",
    "special program": "Special program", "high-balance flag": "High-balance loan", "at-the-limit flag": "At the loan limit",
    core.INCOMPLETE: "Incomplete application",
}
YES_NO = {"Y": "Yes", "N": "No"}

NOTES = {
    "research": "Tool intended for research purposes only.",
    "scope": ("For conventional conforming, fully amortizing, fixed-rate, single-family (1-4 unit) mortgages purchased by "
              "Freddie Mac and Fannie Mae."),
    "spread": ("Rate spread = your rate minus the 2023 median rate for the same term group (10-, 15-, 20- or 30-year). "
               "The models use the spread, not the rate."),
    "r": "Scored the same as None: the models never saw R loans in training.",
    "blank": "The models rarely or never saw these blank in training, so treat this result with extra care.",
    "curve": ("The curve represents the share of delinquent loan balance caught and performing loan balance turned away at "
              "every approval cutoff, from most lenient to strictest. The optimal cutoff is the one that produced the largest "
              "gap on 2022 loans, the year before. Since the right balance depends on each lender's priorities, the adjustable "
              "slider allows testing stricter or more lenient cutoffs and indicates whether this applicant would be approved "
              "under each."),
    "drivers": ("Positive values raise calculated risk and negative values lower it; each value is that variable's share of "
                "the total effect."),
    "title": "Probability of Serious Delinquency (90+ Days Late Within First 24 Payments)",
}
UNITS = {"credit_score": "", "dti": "%", "cltv": "%", "mi_pct": "%", "loan_term": " months"}


# ---------------------------------------------------------------- small helpers

def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(text):
    """A number from the project's CSVs ("1,234", "0.0667%"); None when masked ("1-10", "-") or blank."""
    t = str(text).replace(",", "").replace("%", "").strip()
    try:
        return float(t)
    except ValueError:
        return None


def pct(x, decimals=2):
    return f"{100 * x:.{decimals}f}%"


def money(d):
    if d is None:
        return "hidden (1-10 loans)"
    if d >= 1e9:
        return f"${d / 1e9:,.2f} billion"
    if d >= 1e6:
        return f"${d / 1e6:,.1f} million"
    return f"${d:,.0f}"


def _f32_outside(v, lo, hi):
    """Outside the training range, compared in float32 as the edge rule compares (check_beyond_range)."""
    v32 = np.float32(v)
    return bool(v32 < np.float32(lo) or v32 > np.float32(hi))


def sql_calculated(amount, units, term):
    """Percent of limit, at-the-limit flag (as in fit_models' model view) and the term group, for one application in
    2023, computed by model_core's SQL text in an in-memory DuckDB."""
    con = duckdb.connect(":memory:")
    try:
        con.execute("CREATE TABLE kept_loans (agency VARCHAR, loan_id VARCHAR, origination_quarter VARCHAR, "
                    "original_upb DOUBLE, num_units VARCHAR, loan_term INTEGER)")
        con.execute("INSERT INTO kept_loans VALUES ('app', 'applicant', ?, ?, ?, ?)",
                    [QUARTER, None if amount is None else float(amount), units, None if term is None else int(term)])
        con.execute(core.model_view_sql())
        ratio, flag, product = con.execute(f"SELECT CAST({core.RATIO} AS DOUBLE), CAST({core.FLAG} AS VARCHAR), "
                                           f"{core.PRODUCT_SQL} FROM {core.SOURCE}").fetchone()
    finally:
        con.close()
    return ratio, flag, product


def load_curve(path):
    """The LightGBM trade-off curve the app shows (curve_lightgbm.csv, written by build_app_summary.py): a no-cutoff stop
    at 0% declined, then the file's safe stops. Returns (stops, index of the optimal stop)."""
    stops = [{"share": 0.0, "cutoff": math.inf, "bad_d": 0, "bad_pct": 0.0, "good_d": 0, "good_pct": 0.0, "optimal": False}]
    for r in _csv(path):
        stops.append({"share": float(r["share_declined"]), "cutoff": float(r["cutoff_adjusted_risk"]),
                      "bad_d": float(r["delinquent_dollars_caught"]), "bad_pct": float(r["delinquent_dollars_pct"]),
                      "good_d": float(r["performing_dollars_turned_away"]), "good_pct": float(r["performing_dollars_pct"]),
                      "optimal": r["optimal"] == "yes"})
    return stops, next(i for i, p in enumerate(stops) if p["optimal"])


def stop_label(p):
    """A stop's declined share as the slider and the table show it (2 decimals)."""
    return f"{100 * p['share']:.2f}%"


def readout_cells(p, risk_g=None):
    """The readout of one stop: declined share, cutoff, delinquent and performing balance (share and dollars) and, given
    the applicant's LightGBM risk, Approve / Decline (Decline when the risk is at or above the cutoff)."""
    return {
        "declined": stop_label(p),
        "cutoff": "No cutoff" if p["cutoff"] == math.inf else pct(p["cutoff"]),
        "bad": f"{p['bad_pct']:.1f}% · {money(p['bad_d'])}",
        "good": f"{p['good_pct']:.1f}% · {money(p['good_d'])}",
        "decision": None if risk_g is None else ("Decline" if risk_g >= p["cutoff"] else "Approve"),
    }


# ---------------------------------------------------------------- the tool

class Tool:
    def __init__(self, root=APP_DIR, data_dir=None):
        root = Path(root)
        mdir, q1 = root / "models", root / "models"
        data = Path(data_dir) if data_dir else root / "data"
        self.lr = _json(mdir / "logistic_regression.json")
        self.rule = _json(mdir / "lr_edge_rule.json")
        self.cal = _json(mdir / "isotonic_calibration.json")
        settings = _json(mdir / "lightgbm_settings.json")
        self.booster = lgb.Booster(model_file=str(mdir / "lightgbm_model.txt"))
        by_label = {r["variable"]: r for r in _csv(mdir / "beyond_range.csv")}
        self.range = {c: (float(by_label[core.LABEL[c]]["training_min"]), float(by_label[core.LABEL[c]]["training_max"]))
                      for c in core.NUM_COLS}
        summary = _json(data / "app_summary.json")
        codes = {c: settings["category_codes"][core.LABEL[c]] for c in core.CAT_COLS}
        refs = {c: settings["reference_categories"][core.LABEL[c]] for c in core.CAT_COLS}
        present = {c: summary["category_values"][core.LABEL[c]] for c in core.CAT_COLS}
        self.saved = {"lr": self.lr, "rule": self.rule, "cal": self.cal, "booster": self.booster,
                      "codes": codes, "refs": refs, "present": present}
        self.shares = summary["lr_column_shares"]
        self.medians = summary["median_rate_2023"]
        self.rare_blank = summary["rarely_blank_in_training"]
        self.mean_risk = summary["mean_adjusted_risk_2023"]
        self.ranks = {}
        for model, name in ((GBM, "rank_lightgbm.csv"), (LR, "rank_logistic_regression.csv")):
            rows = _csv(data / name)
            self.ranks[model] = tuple(np.array([float(r[k]) for r in rows]) for k in ("adjusted_risk", "share_below", "share_at_or_below"))
        auc = {r["model"]: r for r in _csv(q1 / "q1_auc.csv") if r["population"] == "all test loans"}
        self.q1 = {m: {"auc": float(auc[m]["auc"]), "actual": _num(auc[m]["actual_bad_rate"]) / 100,
                       "predicted": float(self.mean_risk[m])} for m in (LR, GBM)}
        self.curve, self.optimal = load_curve(data / "curve_lightgbm.csv")
        self._input_limits()

    # ------------------------------------------------------------ the curve

    def readout(self, i, risk_g=None):
        """The readout at stop i (0 = no cutoff)."""
        return readout_cells(self.curve[i], risk_g)

    def applicant_stop(self, risk_g):
        """The first stop (fewest declined) at which the applicant is declined (risk >= the stop's cutoff)."""
        return next((i for i, p in enumerate(self.curve) if i > 0 and risk_g >= p["cutoff"]), None)

    # ------------------------------------------------------------ inputs

    def _input_limits(self):
        """Box limits: whole-number boxes at the training ranges; the loan amount and the rate get static
        outer limits, and the exact range (by units / by term group) is checked when Submit is pressed."""
        lim = {}
        for key, _, col in NUMBER_FIELDS:
            if col:
                lo, hi = self.range[col]
                lim[key] = (int(math.ceil(lo)), int(math.floor(hi)))
        plo, phi = self.range[core.RATIO]
        lims = core.LIMITS[YEAR]
        lim["loan_amount"] = (int(math.ceil(plo / 100 * min(lims))), int(math.floor(phi / 100 * max(lims))))
        slo, shi = self.range["rate_spread"]
        meds = [m for m in self.medians.values() if m is not None]
        lim["interest_rate"] = ((math.floor((min(meds) + slo) * 1000) / 1000, math.ceil((max(meds) + shi) * 1000) / 1000)
                                if meds else (0.001, 20.0))
        self.limits = lim

    def rate_range(self, term):
        """The interest rates whose rate spread lies inside the training range, for the term's group: (group, low, high);
        low and high are None when the group has no 2023 median; None when the term is blank."""
        if term is None:
            return None
        product = sql_calculated(None, None, term)[2]
        median = self.medians.get(product) if product else None
        if median is None:
            return product, None, None
        slo, shi = self.range["rate_spread"]
        return product, math.ceil((median + slo) * 1000) / 1000, math.floor((median + shi) * 1000) / 1000

    def amount_range(self, units):
        """The loan amounts whose percent of the 2023 limit lies inside the training range, for the number of units."""
        if units is None:
            return None
        plo, phi = self.range[core.RATIO]
        lim = core.LIMITS[YEAR][int(units) - 1]
        return math.ceil(plo / 100 * lim), math.floor(phi / 100 * lim)

    def range_note(self, key, a):
        """The muted range text under a number box; the rate's depends on the loan term, the amount's on the units."""
        if key == "interest_rate":
            rr = self.rate_range(a.get("loan_term"))
            if rr is None:
                return "Select a loan term to see the range"
            if rr[1] is None:
                return f"No 2023 median rate for {rr[0]} loans: the rate spread will be blank"
            return f"Range for a {rr[0]} loan: {rr[1]:.3f}% to {rr[2]:.3f}%"
        if key == "loan_amount":
            ar = self.amount_range(a.get("num_units"))
            if ar is None:
                return "Select number of units to see the range"
            unit = "unit" if a["num_units"] == "1" else "units"
            return f"Range for {a['num_units']} {unit}: ${ar[0]:,} to ${ar[1]:,}"
        lo, hi = self.limits[key]
        return f"Range: {lo:,} to {hi:,}{UNITS[key]}"

    def calculated(self, a):
        """The fields calculated from the application (never asked): term group, 2023 median, rate spread, percent of
        limit, at-the-limit flag and high-balance flag (Y when the amount is above the 2023 basic limit for its
        units). Blank rules: amount or units blank -> percent, flag and high-balance blank; term or rate
        blank (or no 2023 median for the term group) -> spread blank."""
        amount, units, term, rate = a.get("loan_amount"), a.get("num_units"), a.get("loan_term"), a.get("interest_rate")
        ratio, flag, product = sql_calculated(amount, units, term)
        median = self.medians.get(product) if product else None
        spread = None if rate is None or median is None else float(rate) - float(median)
        if amount is None or units is None:
            high = None
        else:
            high = "Y" if float(amount) > core.LIMITS[YEAR][int(units) - 1] else "N"
        return {"product": product, "median": median, "rate_spread": spread, core.RATIO: ratio, core.FLAG: flag, "high_balance": high,
                "median_missing": product is not None and median is None}

    def problems(self, a, calc=None):
        """Why an application cannot be scored (empty: it can). Box limits, then the spread and percent-of-limit ranges."""
        calc = calc or self.calculated(a)
        out, box = [], set()
        for key, label, _ in NUMBER_FIELDS:
            v = a.get(key)
            if v is None:
                continue
            lo, hi = self.limits[key]
            if not lo <= v <= hi:
                out.append(f"{label}: enter a value from {lo:,} to {hi:,}.")
                box.add(key)
        if calc["rate_spread"] is not None and not box & {"interest_rate", "loan_term"}:
            slo, shi = self.range["rate_spread"]
            if _f32_outside(calc["rate_spread"], slo, shi):
                _, lo, hi = self.rate_range(a.get("loan_term"))
                out.append(f"Interest rate: for a {calc['product']} loan the models' range is {lo:.3f}% to {hi:.3f}%.")
        if calc[core.RATIO] is not None and "loan_amount" not in box:
            plo, phi = self.range[core.RATIO]
            if _f32_outside(calc[core.RATIO], plo, phi):
                lo, hi = self.amount_range(a["num_units"])
                out.append(f"Loan amount: for {a['num_units']} unit(s) the models' range is ${lo:,} to ${hi:,}.")
        return out

    def arrays(self, a, calc):
        """The models' inputs for one application: numbers (1 x 7 float32, NaN = blank) and category codes (1 x 10 int8:
        -1 blank), as fit_models.fetch_rows reads a loan."""
        vals = {"credit_score": a.get("credit_score"), "cltv": a.get("cltv"), "dti": a.get("dti"), "rate_spread": calc["rate_spread"],
                "mi_pct": a.get("mi_pct"), core.RATIO: calc[core.RATIO], "loan_term": a.get("loan_term")}
        num = np.full((1, len(core.NUM_COLS)), np.nan, dtype=np.float32)
        for j, c in enumerate(core.NUM_COLS):
            if vals[c] is not None:
                num[0, j] = float(vals[c])
        cat = np.full((1, len(core.CAT_COLS)), -1, dtype=np.int8)
        for j, c in enumerate(core.CAT_COLS):
            v = calc[c] if c in (core.FLAG, "high_balance") else a.get(c)
            if v is not None:
                cat[0, j] = self.saved["codes"][c].get(str(v), -2)
        return num, cat, vals

    # ------------------------------------------------------------ scoring

    def score_arrays(self, num, cat):
        """Raw scores and adjusted risks of both models, exactly as score_test.py (model_core.raw_scores / adjust_scores)."""
        z_lr, z_g = core.raw_scores(num, cat, self.saved)
        r_lr, r_g = core.adjust_scores(z_lr, z_g, self.saved)
        return {LR: float(z_lr[0]), GBM: float(z_g[0])}, {LR: float(r_lr[0]), GBM: float(r_g[0])}

    def shap(self, num, cat):
        """Per-variable SHAP values on the log-odds scale (as score_test.py's Q2): LightGBM pred_contrib; Logistic
        Regression coefficient x (value seen - training mean), categories summed, the incomplete column on its own."""
        X = core.lgb_matrix(num, cat, self.saved["codes"], self.saved["refs"], self.saved["present"])
        contrib = self.booster.predict(X, pred_contrib=True)[0]
        gbm = {core.LABEL[c]: float(contrib[j]) for j, c in enumerate(core.NUM_COLS + core.CAT_COLS)}
        Xl = core.lr_design(self.lr, core.lr_inputs(self.rule, num), cat, self.saved["codes"], self.saved["refs"], self.saved["present"])
        phi, base_lr = core.lr_shap(self.lr, Xl, self.shares)
        lr = {k: float(v[0]) for k, v in core.lr_by_variable(self.lr, phi).items()}
        return {GBM: (gbm, float(contrib[-1])), LR: (lr, base_lr)}

    def rank(self, model, risk):
        """Share of 2023 test loans with an adjusted risk strictly below this one (0-1). Exact at a table row;
        straight line between neighbouring rows."""
        v, below, at = self.ranks[model]
        if len(v) == 0:
            return None
        i = int(np.searchsorted(v, risk, side="left"))
        if i < len(v) and v[i] == risk:
            return float(below[i])
        if i == 0:
            return 0.0
        if i == len(v):
            return float(at[-1])
        w = (risk - v[i - 1]) / (v[i] - v[i - 1])
        return float(at[i - 1] + w * (below[i] - at[i - 1]))

    @staticmethod
    def share_rows(phi_by_label):
        """Top 10 variables by |SHAP| plus "All other variables": share of the total |SHAP| (sums to 1), direction
        (+1 raises risk, -1 lowers it; the rest by its net SHAP) and the signed share in percent (share x direction;
        the absolute signed shares sum to 100). Ties keep the model's variable order."""
        items = list(phi_by_label.items())
        total = sum(abs(p) for _, p in items)
        ranked = sorted(items, key=lambda kv: -abs(kv[1]))
        top, rest = ranked[:TOP], ranked[TOP:]
        share = (lambda x: abs(x) / total) if total > 0 else (lambda x: 0.0)
        rows = [{"variable": k, "share": share(p), "direction": int(np.sign(p)), "phi": p} for k, p in top]
        if rest:
            net = sum(p for _, p in rest)
            rows.append({"variable": OTHER, "share": sum(abs(p) for _, p in rest) / total if total > 0 else 0.0,
                         "direction": int(np.sign(net)), "phi": net})
        for r in rows:
            r["signed"] = 100 * r["share"] * (r["direction"] or 1)
        return rows

    def value_text(self, label, a, vals, calc, incomplete):
        """The value shown in a driver-table row: as entered; a blank field shows "blank"."""
        col = next((c for c in core.NUM_COLS + core.CAT_COLS if core.LABEL[c] == label), None)
        if label == core.INCOMPLETE:
            return "Yes" if incomplete else "No"
        if col in core.NUM_COLS:
            v = vals[col]
            if v is None:
                return "blank"
            if col == "credit_score":
                return f"{v:,.0f}"
            if col == "rate_spread":
                return f"{v:+.4f}".rstrip("0").rstrip(".") + " pts"
            if col == core.RATIO:
                return f"{v:.2f}%"
            if col == "loan_term":
                return f"{v:.0f} months"
            return f"{v:g}%"
        v = calc[col] if col in (core.FLAG, "high_balance") else a.get(col)
        if v is None:
            return "blank"
        if col in (core.FLAG, "high_balance", "first_time_homebuyer"):
            return YES_NO[v]
        text = dict(CATEGORY_FIELDS[col][1])[v]
        return text + " (scored as None)" if col == "special_program" and v == "R" else text

    def score(self, a):
        """Score one application. Returns {"problems": [...]} when it cannot be scored, else the full result."""
        calc = self.calculated(a)
        probs = self.problems(a, calc)
        if probs:
            return {"problems": probs}
        num, cat, vals = self.arrays(a, calc)
        z, risk = self.score_arrays(num, cat)
        shap = self.shap(num, cat)
        incomplete = bool(np.isnan(num).any() or (cat == -1).any())
        drivers = {}
        for model in (GBM, LR):
            rows = self.share_rows(shap[model][0])
            for r in rows:
                r["label"] = ROW_LABEL.get(r["variable"], r["variable"])
                r["value"] = None if r["variable"] == OTHER else self.value_text(r["variable"], a, vals, calc, incomplete)
            drivers[model] = rows
        blank = [core.LABEL[c] for j, c in enumerate(core.NUM_COLS) if np.isnan(num[0, j])] + \
                [core.LABEL[c] for j, c in enumerate(core.CAT_COLS) if cat[0, j] == -1]
        return {"problems": [], "calc": calc, "vals": vals, "z": z, "risk": risk, "shap": shap,
                "rank": {m: self.rank(m, risk[m]) for m in (GBM, LR)}, "drivers": drivers, "incomplete": incomplete,
                "blank": blank, "blank_note": [ROW_LABEL[b] for b in blank if self.rare_blank.get(b)],
                "special_program_r": a.get("special_program") == "R", "applicant_stop": self.applicant_stop(risk[GBM])}
