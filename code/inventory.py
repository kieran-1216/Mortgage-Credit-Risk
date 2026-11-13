"""Counts-only inventory of everything in the raw Freddie Mac and Fannie Mae data.

Facts only: no scope filters, no outcome labels, no matching of codes between the agencies.
Everything is reported per agency, using that agency's own raw codes.

What it reports (logs/inventory.txt and the screen)
  PART 1  Every column of freddie_orig, freddie_perf and fannie_raw, grouped by type:
            coded fields   every value with its count, marked documented or UNDOCUMENTED
            numeric fields count, NULL, non-numeric, min, max, special codes, outside the
                           documented range, bands
            dates          NULL, unparseable, earliest, latest, impossible dates
            free text      distinct count and Other share only (names are never shown)
            IDs            rows, distinct IDs, duplicates, cross-checks
          plus NULL and special codes of every loan-level field by origination year, Fannie
          repeating-field changes, and checks of rules stated in the documentation.
  PART 2  Every loan sorted into exactly one value on each dimension (a-q). Each dimension's
          values must add up to the agency's loan count, overall and per origination year.
          Also: the d x e x g x f table, payment-history patterns, and missing-value
          combinations.

How it runs
  * Opens freddie_fannie_raw.duckdb read-only (load_raw.connect(read_only=True)).
  * Works one loaded source file at a time (one load_log entry). For each file it writes
    per-loan facts and Part 1 partial counts to inventory_work.duckdb and marks the file done in
    the same transaction. A rerun skips finished files. The report is rebuilt every run from
    the working database.
  * DuckDB spills to inventory_work.duckdb.tmp/ (size-capped). The script refuses to start or
    to continue if free disk is below --min-free-gb.
  * Progress lines (timestamps, time per file, estimated time left) go to the screen and to
    logs/inventory_progress.log; the report itself goes to logs/inventory.txt.

Privacy: the report holds counts only. It shows no loan IDs, no row-level values and no seller
or servicer names. Counts from 1 to 10 show as "1-10". inventory_work.duckdb holds per-loan
derived data, including loan IDs. It stays on the local machine and must never be uploaded or
shared.

Usage (from the repository root):
    python code/inventory.py [--only part1|part2] [--agency freddie|fannie]
                             [--memory-limit 5GB] [--threads 6] [--min-free-gb 45]
    python code/inventory.py --clean     (delete the working files and stop)
Exit codes: 0 finished; 1 a section failed or a total did not add up; 2 the working files belong
to another load or script version (run --clean); 4 not enough free disk; 130 interrupted.
"""

import argparse
import hashlib
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_raw  # noqa: E402  (report buffer behind out())
import load_raw  # noqa: E402  (DB_PATH is read at connect time)
from check_raw import out, out_table, section  # noqa: E402
from load_raw import LOG_DIR, PROJECT_ROOT, hms, read_layouts, report_error  # noqa: E402
from load_raw import quote_ident as q  # noqa: E402

WORK_DB = PROJECT_ROOT / "inventory_work.duckdb"        # per-loan derived data: never share
SPILL_DIR = PROJECT_ROOT / "inventory_work.duckdb.tmp"  # DuckDB spill files (size-capped)
REPORT_PATH = LOG_DIR / "inventory.txt"
PROGRESS_PATH = LOG_DIR / "inventory_progress.log"
ERROR_DETAIL_PATH = LOG_DIR / "inventory_error_detail.txt"

SCHEMA_VERSION = "1"        # bump when the working tables change; old work then needs --clean
# 11 GB machine: 5 GB for DuckDB leaves room for Windows, Python and DuckDB's own overhead
# outside its limit. 6 of the 8 cores: about 0.8 GB per thread for the large per-loan GROUP BYs
# and the sorted pass, and the machine stays usable.
DEFAULT_MEMORY_LIMIT = "5GB"
DEFAULT_THREADS = 6
# Spill cap 30 GB. Minimum free disk 45 GB =
# 30 GB spill cap + up to ~6 GB working database + headroom so the drive never fills up.
SPILL_CAP_GB = 30
DEFAULT_MIN_FREE_GB = 45
LL_CHUNK = 20               # loan-level fields per GROUP BY when resolving per-loan values
YEAR_COLS = 8               # origination-year columns per printed table
TOP_PATTERNS = 300
TOP_COMBOS = 100
TOP_COMBOS_BY_YEAR = 20
BAP_START = 2020 * 12 + 3   # April 2020 as a month index: Fannie field 102 is populated from then
Q2018Q1 = 2018 * 4 + 1      # quarter ordinal (year * 4 + quarter) where Freddie ranges change
CATCH_ALL = "anything else"
SPLIT = "monthly records in more than one source file"
NO_RECORDS = "no monthly records"
NO_ORIG = "no origination row"

AGENCIES = {"fre": "Freddie Mac", "fan": "Fannie Mae"}
FAILS = []    # totals that did not add up
ERRORS = []   # sections or files that raised an error
_AFTER_PIECE_HOOK = None  # tests only: called after every finished piece


# ---------------------------------------------------------------- field documentation
# Codes, special values and ranges come only from the Freddie general user guide,
# the Freddie file layout and the Fannie "Combined Glossary"; anything else is UNDOCUMENTED.

@dataclass(frozen=True)
class Spec:
    kind: str               # code | status | geo | num | date | text | id | hist
    label: str              # short name used in tables and in missing-value combinations
    codes: tuple = ()       # documented codes (code, status)
    null_doc: bool = False  # the documentation describes blank/NULL as a value
    na: tuple = ()          # documented "not available"/unknown codes (missing-value combinations)
    special: tuple = ()     # numeric special codes; for geo, other documented special codes
    rng: object = None      # documented valid range: (lo, hi) or the name of a Freddie rule
    bands: str = ""         # band family (numeric)
    fmt: str = ""           # date format, or the documented format of a geo code
    listed: bool = True     # geo: list every value (True) or summarise (False)
    above: tuple = ()       # numeric: extra "count above X" lines stated in the documentation


def code(label, codes=(), null_doc=False, na=()):
    return Spec("code", label, tuple(codes), null_doc, tuple(na))


def status(label, extra, null_doc=False):
    return Spec("status", label, tuple(extra), null_doc)


def geo(label, fmt, listed, null_doc=False, na=(), special=()):
    return Spec("geo", label, null_doc=null_doc, na=tuple(na), special=tuple(special), fmt=fmt, listed=listed)


def num(label, bands, rng=None, special=(), above=()):
    return Spec("num", label, special=tuple(special), rng=rng, bands=bands, above=tuple(above))


def date(label, fmt):
    return Spec("date", label, fmt=fmt)


def text(label):
    return Spec("text", label)


def ident(label, null_doc=False):
    return Spec("id", label, null_doc=null_doc)


YN = ("Y", "N")

FREDDIE_ORIG = {
    "classic_fico": num("fico", "score", rng=(300, 850), special=(9999,)),
    "first_payment_date": date("fpd", "YYYYMM"),
    "first_time_homebuyer_indicator": code("fthb", ("Y", "N", "9"), na=("9",)),
    "maturity_date": date("maturity", "YYYYMM"),
    "metropolitan_statistical_area_msa_or_metropolitan_division": geo("msa", "[0-9]{5}", False, null_doc=True),
    "mortgage_insurance_percentage_mi": num("mi_pct", "mi_pct", rng=(0, 55), special=(999,)),
    "number_of_units": code("units", ("1", "2", "3", "4", "99"), na=("99",)),
    "occupancy_status": code("occupancy", ("P", "I", "S", "9"), na=("9",)),
    "original_combined_loan_to_value_cltv": num("cltv", "ratio", rng="fre_cltv", special=(999,)),
    "original_debt_to_income_dti_ratio": num("dti", "dti", rng="fre_dti", special=(999,)),
    "original_upb": num("upb", "upb"),
    "original_loan_to_value_ltv": num("ltv", "ratio", rng="fre_ltv", special=(999,)),
    "original_interest_rate": num("rate", "rate"),
    "channel": code("channel", ("R", "B", "C", "T", "9"), na=("9",)),
    "prepayment_penalty_indicator": code("ppm", YN),
    "amortization_type": code("amort", ("FRM", "ARM")),
    "property_state": geo("state", "[A-Z]{2}", True),
    "property_type": code("prop_type", ("CP", "CO", "PU", "SF", "MH", "99"), na=("99",)),
    "postal_code": geo("zip3", "[0-9]{3}", False, na=("000",)),
    "loan_identifier": ident("loan_id"),
    "loan_purpose": code("purpose", ("P", "C", "N", "R", "9"), na=("9",)),
    "original_loan_term": num("term", "term"),
    "number_of_borrowers": num("borrowers", "borrowers", rng="fre_borrowers", special=(99,)),
    "seller_name": text("seller"),
    "super_conforming_flag": code("super_conf", YN),
    "pre_harp_loan_sequence_number": ident("pre_harp", null_doc=True),
    "special_eligibility_program": code("sep", ("H", "F", "R"), null_doc=True),
    "harp_indicator": code("harp", YN),
    "property_valuation_method": code("valuation", ("1", "2", "3", "4", "7"), na=("7",)),
    "interest_only_i_o_indicator": code("io", YN),
    "vantagescore_4_0": num("vs4", "score", rng=(300, 850), special=(9999,)),
}

FREDDIE_PERF = {
    "loan_identifier": ident("loan_id"),
    "period": date("period", "YYYYMM"),
    "current_actual_upb": num("cur_upb", "upb"),
    "current_loan_delinquency_status": status("status", ("RA", "XX")),
    "loan_age": num("loan_age", "age"),
    "remaining_months_to_legal_maturity": num("rem_legal", "months"),
    "underwriting_defect_and_major_servicing_defect_settlement_date": date("defect_date", "YYYYMM"),
    "modification_flag": code("mod", ("Y", "P"), null_doc=True),
    "zero_balance_code": code("zb", ("01", "02", "03", "09", "15", "16", "96")),
    "zero_balance_effective_date": date("zb_date", "YYYYMM"),
    "current_interest_rate": num("cur_rate", "rate"),
    "current_non_interest_bearing_upb": num("nib_upb", "upb"),
    "due_date_of_last_paid_installment_ddlpi": date("ddlpi", "YYYYMM"),
    "mi_recoveries": num("mi_recov", "amount"),
    "net_sales_proceeds": num("net_sales", "amount"),
    "non_mi_recoveries": num("non_mi_recov", "amount"),
    "total_expenses": num("total_exp", "amount"),
    "legal_costs": num("legal", "amount"),
    "maintenance_and_preservation_costs": num("maint", "amount"),
    "taxes_and_insurance": num("taxes_ins", "amount"),
    "miscellaneous_expenses": num("misc_exp", "amount"),
    "actual_loss": num("actual_loss", "amount"),
    "cumulative_modification_costs": num("cum_mod_cost", "amount"),
    "interest_rate_step_indicator": code("step", YN, null_doc=True),
    "payment_deferral_flag": code("deferral", ("C", "P")),
    "estimated_loan_to_value_eltv": num("eltv", "ratio", rng=(1, 998), special=(999,)),
    "zero_balance_removal_upb": num("zb_removal_upb", "upb"),
    "delinquent_accrued_interest": num("dai", "amount"),
    "delinquency_due_to_disaster": code("disaster", ("Y",), null_doc=True),
    "borrower_assistance_plan": code("bap", ("F", "R", "T"), null_doc=True),
    "current_period_modification_costs": num("cur_mod_cost", "amount"),
    "current_interest_bearing_upb": num("ib_upb", "upb"),
    "mortgage_insurance_cancellation_indicator": code("mi_cancel", ("Y", "N", "7")),
    "servicer_name": text("servicer"),
    "bankruptcy_cramdown_costs": num("cramdown", "amount"),
}

FANNIE = {
    "reference_pool_id": code("ref_pool"),
    "loan_identifier": ident("loan_id"),
    "monthly_reporting_period": date("period", "MMYYYY"),
    "channel": code("channel", ("R", "C", "B")),
    "seller_name": text("seller"),
    "servicer_name": text("servicer"),
    "master_servicer": code("master_servicer"),
    "original_interest_rate": num("rate", "rate"),
    "current_interest_rate": num("cur_rate", "rate"),
    "original_upb": num("upb", "upb"),
    "upb_at_issuance": num("upb_issuance", "upb"),
    "current_actual_upb": num("cur_upb", "upb"),
    "original_loan_term": num("term", "term"),
    "origination_date": date("orig_date", "MMYYYY"),
    "first_payment_date": date("fpd", "MMYYYY"),
    "loan_age": num("loan_age", "age"),
    "remaining_months_to_legal_maturity": num("rem_legal", "months"),
    "remaining_months_to_maturity": num("rem_mat", "months"),
    "maturity_date": date("maturity", "MMYYYY"),
    "original_loan_to_value_ratio_ltv": num("ltv", "ratio", above=(97, 200)),
    "original_combined_loan_to_value_ratio_cltv": num("cltv", "ratio"),
    "number_of_borrowers": num("borrowers", "borrowers"),
    "debt_to_income_dti": num("dti", "dti"),
    "borrower_credit_score_at_origination": num("fico_b", "score"),
    "co_borrower_credit_score_at_origination": num("fico_cb", "score"),
    "first_time_home_buyer_indicator": code("fthb", YN, null_doc=True),
    "loan_purpose": code("purpose", ("C", "R", "P", "U")),
    "property_type": code("prop_type", ("CO", "CP", "PU", "MH", "SF")),
    "number_of_units": code("units", ("1", "2", "3", "4")),
    "occupancy_status": code("occupancy", ("P", "S", "I", "U"), na=("U",)),
    "property_state": geo("state", "[A-Z]{2}", True),
    "metropolitan_statistical_area_msa_or_metropolitan_statistical_division_area_msda":
        geo("msa", "[0-9]{5}", False, special=("00000",)),
    "zip_code_short": geo("zip3", "[0-9]{3}", False),
    "mortgage_insurance_percentage": num("mi_pct", "mi_pct"),
    "amortization_type": code("amort", ("ARM", "FRM")),
    "prepayment_penalty_indicator": code("ppm", YN),
    "interest_only_loan_indicator": code("io", YN),
    "interest_only_first_principal_and_interest_payment_date": date("io_date", "MMYYYY"),
    "months_to_amortization": num("months_to_amort", "months"),
    "current_loan_delinquency_status": status("status", ("XX",), null_doc=True),
    "loan_payment_history": Spec("hist", "pay_hist"),
    "modification_flag": code("mod", YN),
    "mortgage_insurance_cancellation_indicator": code("mi_cancel", ("Y", "N", "NA")),
    "zero_balance_code": code("zb", ("01", "02", "03", "06", "09", "15", "16", "96", "97", "98")),
    "zero_balance_effective_date": date("zb_date", "MMYYYY"),
    "upb_at_the_time_of_removal": num("removal_upb", "upb"),
    "repurchase_date": date("repurchase_date", "MMYYYY"),
    "scheduled_principal_current": num("sched_prin", "amount"),
    "total_principal_current": num("total_prin", "amount"),
    "unscheduled_principal_current": num("unsched_prin", "amount"),
    "last_paid_installment_date": date("lpi_date", "MMYYYY"),
    "foreclosure_date": date("fc_date", "MMYYYY"),
    "disposition_date": date("disp_date", "MMYYYY"),
    "foreclosure_costs": num("fc_costs", "amount"),
    "property_preservation_and_repair_costs": num("preservation", "amount"),
    "asset_recovery_costs": num("asset_recovery", "amount"),
    "miscellaneous_holding_expenses_and_credits": num("misc_holding", "amount"),
    "associated_taxes_for_holding_property": num("holding_taxes", "amount"),
    "net_sales_proceeds": num("net_sales", "amount"),
    "credit_enhancement_proceeds": num("ce_proceeds", "amount"),
    "repurchase_make_whole_proceeds": num("rmw_proceeds", "amount"),
    "other_foreclosure_proceeds": num("other_fc_proceeds", "amount"),
    "modification_related_non_interest_bearing_upb": num("mod_nib_upb", "upb"),
    "principal_forgiveness_amount": num("forgiveness", "amount"),
    "original_list_start_date": date("orig_list_date", "MMYYYY"),
    "original_list_price": num("orig_list_price", "upb"),
    "current_list_start_date": date("cur_list_date", "MMYYYY"),
    "current_list_price": num("cur_list_price", "upb"),
    "borrower_credit_score_at_issuance": num("fico_b_issuance", "score"),
    "co_borrower_credit_score_at_issuance": num("fico_cb_issuance", "score"),
    "borrower_credit_score_current": num("fico_b_current", "score"),
    "co_borrower_credit_score_current": num("fico_cb_current", "score"),
    "mortgage_insurance_type": code("mi_type", ("1", "2", "3"), null_doc=True),
    "servicing_activity_indicator": code("servicing_activity", YN),
    "current_period_modification_loss_amount": num("cur_mod_loss", "amount"),
    "cumulative_modification_loss_amount": num("cum_mod_loss", "amount"),
    "current_period_credit_event_net_gain_or_loss": num("cur_ce_gain_loss", "amount"),
    "cumulative_credit_event_net_gain_or_loss": num("cum_ce_gain_loss", "amount"),
    "special_eligibility_program": code("sep", ("F", "H", "R", "O", "7", "9"), na=("9",)),
    "foreclosure_principal_write_off_amount": num("fc_writeoff", "amount"),
    "relocation_mortgage_indicator": code("relocation", YN),
    "zero_balance_code_change_date": date("zb_change_date", "MMYYYY"),
    "loan_holdback_indicator": code("holdback", YN, null_doc=True),
    "loan_holdback_effective_date": date("holdback_date", "MMYYYY"),
    "delinquent_accrued_interest": num("dai", "amount"),
    "property_valuation_method": code("valuation", ("A", "C", "P", "R", "W", "O"), null_doc=True),
    "high_balance_loan_indicator": code("high_balance", YN),
    "arm_initial_fixed_rate_period_5_yr_indicator": code("arm_le5", YN),
    "arm_product_type": code("arm_product"),
    "initial_fixed_rate_period": num("arm_fixed_months", "months"),
    "interest_rate_adjustment_frequency": num("arm_adj_freq", "months"),
    "next_interest_rate_adjustment_date": date("arm_next_rate_date", "MMYYYY"),
    "next_payment_change_date": date("arm_next_pay_date", "MMYYYY"),
    "index": code("arm_index"),
    "arm_cap_structure": code("arm_caps"),
    "initial_interest_rate_cap_up_percent": num("arm_cap_initial", "rate"),
    "periodic_interest_rate_cap_up_percent": num("arm_cap_periodic", "rate"),
    "lifetime_interest_rate_cap_up_percent": num("arm_cap_lifetime", "rate"),
    "mortgage_margin": num("arm_margin", "rate"),
    "arm_balloon_indicator": code("arm_balloon", YN),
    "arm_plan_number": code("arm_plan"),
    "borrower_assistance_plan": code("bap", ("F", "R", "T", "O", "N", "7", "9")),
    "high_loan_to_value_hltv_refinance_option_indicator": code("hltv", YN),
    "deal_name": text("deal_name"),
    "repurchase_make_whole_proceeds_flag": code("rmw_flag", YN),
    "alternative_delinquency_resolution": code("adr", ("P", "C", "D", "7", "9")),
    "alternative_delinquency_resolution_count": num("adr_count", "count"),
    "total_deferral_amount": num("deferral_amt", "upb"),
    "payment_deferral_modification_event_indicator": code("pd_mod_event", ("Y", "N", "7")),
    "interest_bearing_upb": num("ib_upb", "upb"),
    "origination_classic_fico": num("fico_classic_orig", "score"),
    "issuance_classic_fico": num("fico_classic_issuance", "score"),
    "current_classic_fico": num("fico_classic_current", "score"),
}

# Fannie glossary positions that repeat the same origination value on every monthly row;
# every other position except the loan ID is a monthly field.
FANNIE_LOAN_LEVEL_POSITIONS = frozenset([1, 4, 5, 7, 8, 10, 11, 13, 14, 15, *range(19, 39), 69, 70, 73, 79,
                                         81, *range(86, 92), *range(94, 102), 104, 111, 112])

# Bands: (label, condition on {x}); each family covers every finite number exactly once
# (special codes are excluded before banding). The report checks that bands add up.
BANDS = {
    "score": [("<300", "{x} < 300"), ("300-579", "{x} >= 300 AND {x} < 580"), ("580-619", "{x} >= 580 AND {x} < 620"),
              ("620-659", "{x} >= 620 AND {x} < 660"), ("660-699", "{x} >= 660 AND {x} < 700"),
              ("700-739", "{x} >= 700 AND {x} < 740"), ("740-779", "{x} >= 740 AND {x} < 780"),
              ("780-850", "{x} >= 780 AND {x} <= 850"), (">850", "{x} > 850")],
    "ratio": [("<=60", "{x} <= 60"), ("60<x<=80", "{x} > 60 AND {x} <= 80"), ("80<x<=90", "{x} > 80 AND {x} <= 90"),
              ("90<x<=95", "{x} > 90 AND {x} <= 95"), ("95<x<=97", "{x} > 95 AND {x} <= 97"),
              ("97<x<=100", "{x} > 97 AND {x} <= 100"), ("100<x<=105", "{x} > 100 AND {x} <= 105"),
              ("105<x<=200", "{x} > 105 AND {x} <= 200"), (">200", "{x} > 200")],
    "dti": [("<=20", "{x} <= 20"), ("20<x<=30", "{x} > 20 AND {x} <= 30"), ("30<x<=36", "{x} > 30 AND {x} <= 36"),
            ("36<x<=43", "{x} > 36 AND {x} <= 43"), ("43<x<=45", "{x} > 43 AND {x} <= 45"),
            ("45<x<=50", "{x} > 45 AND {x} <= 50"), ("50<x<=65", "{x} > 50 AND {x} <= 65"), (">65", "{x} > 65")],
    "mi_pct": [("<0", "{x} < 0"), ("0", "{x} = 0"), ("0<x<12", "{x} > 0 AND {x} < 12"), ("12<=x<25", "{x} >= 12 AND {x} < 25"),
               ("25<=x<30", "{x} >= 25 AND {x} < 30"), ("30<=x<=35", "{x} >= 30 AND {x} <= 35"),
               ("35<x<=55", "{x} > 35 AND {x} <= 55"), (">55", "{x} > 55")],
    "rate": [("<=0", "{x} <= 0"), ("0<x<2", "{x} > 0 AND {x} < 2"), ("2<=x<3", "{x} >= 2 AND {x} < 3"),
             ("3<=x<4", "{x} >= 3 AND {x} < 4"), ("4<=x<5", "{x} >= 4 AND {x} < 5"), ("5<=x<6", "{x} >= 5 AND {x} < 6"),
             ("6<=x<7", "{x} >= 6 AND {x} < 7"), ("7<=x<8", "{x} >= 7 AND {x} < 8"), (">=8", "{x} >= 8")],
    "upb": [("<0", "{x} < 0"), ("0", "{x} = 0"), ("0<x<50k", "{x} > 0 AND {x} < 50000"),
            ("50k-<100k", "{x} >= 50000 AND {x} < 100000"), ("100k-<200k", "{x} >= 100000 AND {x} < 200000"),
            ("200k-<300k", "{x} >= 200000 AND {x} < 300000"), ("300k-<400k", "{x} >= 300000 AND {x} < 400000"),
            ("400k-<500k", "{x} >= 400000 AND {x} < 500000"), ("500k-<750k", "{x} >= 500000 AND {x} < 750000"),
            ("750k-<1M", "{x} >= 750000 AND {x} < 1000000"), (">=1M", "{x} >= 1000000")],
    "amount": [("<-100k", "{x} < -100000"), ("-100k-<-10k", "{x} >= -100000 AND {x} < -10000"),
               ("-10k-<0", "{x} >= -10000 AND {x} < 0"), ("0", "{x} = 0"), ("0<x<10k", "{x} > 0 AND {x} < 10000"),
               ("10k-<100k", "{x} >= 10000 AND {x} < 100000"), (">=100k", "{x} >= 100000")],
    "term": [("<120", "{x} < 120"), ("120-179", "{x} >= 120 AND {x} < 180"), ("180", "{x} = 180"),
             ("181-239", "{x} > 180 AND {x} < 240"), ("240", "{x} = 240"), ("241-359", "{x} > 240 AND {x} < 360"),
             ("360", "{x} = 360"), (">360", "{x} > 360")],
    "months": [("<0", "{x} < 0"), ("0", "{x} = 0"), ("1-120", "{x} > 0 AND {x} <= 120"),
               ("121-240", "{x} > 120 AND {x} <= 240"), ("241-360", "{x} > 240 AND {x} <= 360"), (">360", "{x} > 360")],
    "age": [("<0", "{x} < 0"), ("0", "{x} = 0"), ("1-6", "{x} > 0 AND {x} <= 6"), ("7-12", "{x} > 6 AND {x} <= 12"),
            ("13-24", "{x} > 12 AND {x} <= 24"), ("25-60", "{x} > 24 AND {x} <= 60"),
            ("61-120", "{x} > 60 AND {x} <= 120"), (">120", "{x} > 120")],
    "borrowers": [("<1", "{x} < 1"), ("1", "{x} >= 1 AND {x} < 2"), ("2", "{x} >= 2 AND {x} < 3"),
                  ("3", "{x} >= 3 AND {x} < 4"), ("4", "{x} >= 4 AND {x} < 5"), ("5-10", "{x} >= 5 AND {x} < 11"),
                  (">10", "{x} >= 11")],
    "count": [("<0", "{x} < 0"), ("0", "{x} >= 0 AND {x} < 1"), ("1", "{x} >= 1 AND {x} < 2"),
              ("2", "{x} >= 2 AND {x} < 3"), ("3+", "{x} >= 3")],
}

RANGE_TEXT = {
    "fre_cltv": "HARP: 1-998; 2018Q1 and prior: 6-200; 2018Q2 and later: 1-998",
    "fre_ltv": "HARP: 1-998; 2018Q1 and prior: 6-105; 2018Q2 and later: 1-998",
    "fre_dti": "0-65; HARP loans: only 999",
    "fre_borrowers": "2018Q1 and prior: 1-2; 2018Q2 and later: 1-10",
}


def range_sql(rule, v):
    """Documented valid range as SQL on value v; Freddie rules use the _q (quarter) and _harp columns."""
    if isinstance(rule, tuple):
        return f"{v} BETWEEN {rule[0]} AND {rule[1]}"
    new = f"_q IS NULL OR _q > {Q2018Q1}"
    if rule == "fre_cltv":
        return f"(CASE WHEN _harp THEN {v} BETWEEN 1 AND 998 WHEN {new} THEN {v} BETWEEN 1 AND 998 ELSE {v} BETWEEN 6 AND 200 END)"
    if rule == "fre_ltv":
        return f"(CASE WHEN _harp THEN {v} BETWEEN 1 AND 998 WHEN {new} THEN {v} BETWEEN 1 AND 998 ELSE {v} BETWEEN 6 AND 105 END)"
    if rule == "fre_dti":
        return f"(CASE WHEN _harp THEN false ELSE {v} BETWEEN 0 AND 65 END)"
    if rule == "fre_borrowers":
        return f"(CASE WHEN {new} THEN {v} BETWEEN 1 AND 10 ELSE {v} BETWEEN 1 AND 2 END)"
    raise ValueError(rule)


@dataclass
class Tab:
    ag: str          # 'fre' or 'fan'
    table: str       # raw table
    specs: dict
    ll: list         # loan-level fields (resolved to one value per loan)
    monthly: list    # monthly fields
    maxlen: dict     # layout max length per column (used to hide implausibly long code values)


def build_tabs(layouts):
    """Field lists per table, checked against the layout documents."""
    def maxlen(col):
        m = re.search(r"\d+", col.layout_max_length or "")
        return int(m.group()) if m else 12

    tabs = {}
    for table, specs in (("freddie_orig", FREDDIE_ORIG), ("freddie_perf", FREDDIE_PERF), ("fannie_raw", FANNIE)):
        cols = layouts[table].columns
        names = [c.name for c in cols]
        if set(names) != set(specs):
            raise ValueError(f"{table}: layout columns and inventory specs differ: "
                             f"{sorted(set(names) ^ set(specs))}; update the specs in inventory.py")
        lens = {c.name: maxlen(c) for c in cols}
        if table == "freddie_orig":
            tabs[table] = Tab("fre", table, specs, [n for n in names if n != "loan_identifier"], [], lens)
        elif table == "freddie_perf":
            tabs[table] = Tab("fre", table, specs, [], [n for n in names if n != "loan_identifier"], lens)
        else:
            ll = [c.name for c in cols if c.position in FANNIE_LOAN_LEVEL_POSITIONS]
            mo = [c.name for c in cols if c.position not in FANNIE_LOAN_LEVEL_POSITIONS and c.name != "loan_identifier"]
            tabs[table] = Tab("fan", table, specs, ll, mo, lens)
    return tabs


# ---------------------------------------------------------------- small helpers

def progress(msg=""):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(PROGRESS_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_report():
    REPORT_PATH.write_text("\n".join(check_raw.report) + "\n", encoding="utf-8")


def m(n):
    """Masked count for display: 1-10 shows as '1-10'."""
    if n is None:
        return None
    n = int(n)
    return "1-10" if 1 <= n <= 10 else n


def mc(n):
    """Masked count as text with thousands separators, for counts inside sentences."""
    v = m(n)
    return "-" if v is None else v if isinstance(v, str) else f"{v:,}"


def pct(n, total):
    return f"{100 * n / total:.2f}%" if n and n > 10 and total else ""


def count_share(n, total):
    share = pct(n, total)
    return f"{mc(n)} ({share})" if share else mc(n)


def lit(s):
    return "'" + str(s).replace("'", "''") + "'"


def ym(i):
    return "-" if i is None else f"{int(i) // 12:04d}-{int(i) % 12 + 1:02d}"


def fmt_value(v):
    if v is None:
        return "-"
    v = float(v)
    return f"{int(v):,}" if v.is_integer() else f"{v:,.4f}".rstrip("0").rstrip(".")


def month_idx(expr, fmt):
    """Month index (year * 12 + month - 1) of a YYYYMM or MMYYYY text date; NULL if unparseable."""
    if fmt == "YYYYMM":
        y, mo = f"substr({expr}, 1, 4)", f"substr({expr}, 5, 2)"
    else:
        mo, y = f"substr({expr}, 1, 2)", f"substr({expr}, 3, 4)"
    return (f"(CASE WHEN regexp_full_match({expr}, '[0-9]{{6}}') AND TRY_CAST({mo} AS INTEGER) BETWEEN 1 AND 12 "
            f"THEN TRY_CAST({y} AS INTEGER) * 12 + TRY_CAST({mo} AS INTEGER) - 1 END)")


UNIT_OK = "regexp_full_match({u}, '[0-9]{{4}}Q[1-4]')"


def unit_q(u):
    """Quarter ordinal (year * 4 + quarter) of a source_file label like 2014Q1."""
    return (f"(CASE WHEN {UNIT_OK.format(u=u)} THEN TRY_CAST(substr({u}, 1, 4) AS INTEGER) * 4 "
            f"+ TRY_CAST(substr({u}, 6, 1) AS INTEGER) END)")


def unit_qstart(u):
    return (f"(CASE WHEN {UNIT_OK.format(u=u)} THEN TRY_CAST(substr({u}, 1, 4) AS INTEGER) * 12 "
            f"+ 3 * TRY_CAST(substr({u}, 6, 1) AS INTEGER) - 3 END)")


def unit_qend(u):
    return f"({unit_qstart(u)} + 2)"


def state_sql(val, conf, fmt):
    """(state, month index) of a resolved per-loan date: ok / missing / unparseable / conflicting."""
    idx = month_idx(val, fmt)
    return (f"CASE WHEN {conf} THEN 'conflicting' WHEN {val} IS NULL THEN 'missing' "
            f"WHEN {idx} IS NULL THEN 'unparseable' ELSE 'ok' END"), idx


def case(whens, else_=CATCH_ALL):
    return "CASE " + " ".join(f"WHEN {c} THEN {v}" for c, v in whens) + f" ELSE {lit(else_)} END"


def buckets(expr, edges, prefix):
    """WHEN clauses sorting an integer expression into labelled buckets: edges = [(label, lo, hi)]."""
    whens = []
    for label, lo, hi in edges:
        if lo is None:
            cond = f"{expr} <= {hi}"
        elif hi is None:
            cond = f"{expr} >= {lo}"
        else:
            cond = f"{expr} BETWEEN {lo} AND {hi}"
        whens.append((f"{prefix} AND {cond}", lit(label)))
    return whens


def check_disk(min_free_gb, where):
    free = shutil.disk_usage(PROJECT_ROOT).free
    if free < min_free_gb * 1e9:
        progress(f"STOP ({where}): only {free / 1e9:,.1f} GB free on the project drive; need at least "
                 f"{min_free_gb:g} GB (DuckDB spill cap {SPILL_CAP_GB} GB + working database + headroom). "
                 f"Finished files are kept; free space and rerun.")
        raise StopRun(4)


class StopRun(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def attempt(title, fn, *args):
    """Run one report section; on error report it (DuckDB messages are not shown) and continue."""
    try:
        return fn(*args)
    except (KeyboardInterrupt, StopRun):
        raise
    except Exception as exc:
        out()
        out(f"SECTION FAILED: {title}")
        report_error(exc, ERROR_DETAIL_PATH, out)
        ERRORS.append(title)
        return None
    finally:
        save_report()


# ---------------------------------------------------------------- working database

PERF_COLUMNS = [
    ("loan_id", "VARCHAR"), ("unit", "VARCHAR"), ("fpd_idx", "INTEGER"), ("fpd_state", "VARCHAR"),
    ("n_rows", "BIGINT"), ("n_per_bad", "BIGINT"), ("n_dup", "BIGINT"), ("n_gap_months", "BIGINT"),
    ("first_per", "INTEGER"), ("last_per", "INTEGER"), ("zb_per", "INTEGER"), ("zb_code", "VARCHAR"),
    ("zb_multi", "BOOLEAN"), ("w_rows", "BIGINT"), ("w_dup", "BIGINT"), ("w_min_pay", "INTEGER"),
    ("w_max_pay", "INTEGER"), ("w_max_sn", "BIGINT"), ("w_ra", "BOOLEAN"), ("w_num", "BOOLEAN"),
    ("w_unp", "BOOLEAN"), ("xb_min", "INTEGER"), ("xb_max", "INTEGER"),
    ("f_forb", "BOOLEAN"), ("f_dis", "BOOLEAN"), ("f_rep", "BOOLEAN"), ("f_trial", "BOOLEAN"),
    ("f_def", "BOOLEAN"), ("f_mod", "BOOLEAN"), ("f_oth", "BOOLEAN"), ("f_3nf", "BOOLEAN"),
    ("bap_null", "BOOLEAN"), ("bap_null_3nf", "BOOLEAN"),
    ("g1", "BOOLEAN"), ("g2", "BOOLEAN"), ("g2_eq", "BOOLEAN"), ("g3", "BOOLEAN"), ("g3_mod", "BOOLEAN"),
    ("g3_na", "BOOLEAN"), ("n_stretch", "BIGINT"), ("w_delinq", "BOOLEAN"),
    ("la_missing", "BOOLEAN"), ("la_diff_unmod", "BOOLEAN"), ("la_diff_mod", "BOOLEAN"),
    *[(f"s{k}", "TINYINT") for k in range(1, 25)],
    ("fc_min", "INTEGER"), ("fc_max", "INTEGER"), ("fc_bad", "BOOLEAN"),
    ("hltv_first", "VARCHAR"), ("hltv_y", "BOOLEAN"), ("hltv_y_first", "INTEGER"), ("hltv_non_y_last", "INTEGER"),
    ("rows_before_fpd", "BIGINT"),
]


def create_work_tables(con, tabs):
    con.execute("CREATE TABLE IF NOT EXISTS w.inv_meta (key VARCHAR, value VARCHAR)")
    con.execute("CREATE TABLE IF NOT EXISTS w.inv_done (piece VARCHAR, tbl VARCHAR, unit VARCHAR, rows BIGINT, "
                "seconds DOUBLE, finished_at TIMESTAMP)")
    for T in (tabs["freddie_orig"], tabs["fannie_raw"]):
        defs = ["loan_id VARCHAR", "unit VARCHAR", "n_rows BIGINT"]
        defs += [f"{q(c)} VARCHAR, {q(c + '__conf')} BOOLEAN, {q(c + '__mix')} BOOLEAN" for c in T.ll]
        if T.ag == "fan":
            defs.append("maturity_first_row VARCHAR")
        con.execute(f"CREATE TABLE IF NOT EXISTS w.{T.ag}_ll ({', '.join(defs)})")
    for ag in AGENCIES:
        con.execute(f"CREATE TABLE IF NOT EXISTS w.{ag}_perf ({', '.join(f'{n} {t}' for n, t in PERF_COLUMNS)})")
    con.execute("CREATE TABLE IF NOT EXISTS w.p1_values (tbl VARCHAR, unit VARCHAR, col VARCHAR, value VARCHAR, n BIGINT)")
    con.execute("CREATE TABLE IF NOT EXISTS w.p1_stats (tbl VARCHAR, unit VARCHAR, col VARCHAR, stat VARCHAR, v DOUBLE)")
    con.execute("CREATE TABLE IF NOT EXISTS w.p1_text (tbl VARCHAR, unit VARCHAR, col VARCHAR, h UBIGINT, "
                "is_null BOOLEAN, is_other BOOLEAN, n BIGINT)")
    con.execute("CREATE TABLE IF NOT EXISTS w.p1_hist (unit VARCHAR, h UBIGINT, len BIGINT, n BIGINT)")
    con.execute("CREATE TABLE IF NOT EXISTS w.p1_tokens (unit VARCHAR, tok VARCHAR, n BIGINT)")
    con.execute("CREATE TABLE IF NOT EXISTS w.p1_misc (tbl VARCHAR, unit VARCHAR, key VARCHAR, n BIGINT)")


def meta_get(con, key):
    r = con.execute("SELECT value FROM w.inv_meta WHERE key = ?", [key]).fetchone()
    return r[0] if r else None


def meta_set(con, key, value):
    con.execute("DELETE FROM w.inv_meta WHERE key = ?", [key])
    con.execute("INSERT INTO w.inv_meta VALUES (?, ?)", [key, str(value)])


def load_units(con):
    """Loaded source files from load_log (status complete): [(table, source_file, rows)]."""
    return con.execute("SELECT table_name, source_file, rows_loaded FROM load_log WHERE status = 'complete' "
                       "ORDER BY table_name, source_file").fetchall()


def check_work_matches(con, units):
    """The working files must come from this load and this script version."""
    fingerprint = hashlib.sha256(repr(units).encode()).hexdigest()
    for key, value, what in (("schema_version", SCHEMA_VERSION, "an older version of inventory.py"),
                             ("raw_fingerprint", fingerprint, "a different load of the raw database")):
        old = meta_get(con, key)
        if old is None:
            meta_set(con, key, value)
        elif old != value:
            progress(f"STOP: {WORK_DB.name} was built by {what}. Run with --clean, then rerun.")
            raise StopRun(2)


def data_end(con, ag):
    """The agency's last monthly period (month index), computed once and kept in inv_meta."""
    key = f"data_end_{ag}"
    v = meta_get(con, key)
    if v is None:
        t0 = time.time()
        table, col, fmt = (("freddie_perf", "period", "YYYYMM") if ag == "fre"
                           else ("fannie_raw", "monthly_reporting_period", "MMYYYY"))
        v = con.execute(f"SELECT max({month_idx(q(col), fmt)}) FROM {table}").fetchone()[0]
        if v is None:
            raise ValueError(f"{table} has no parseable monthly period")
        meta_set(con, key, v)
        progress(f"{AGENCIES[ag]}: data's last month = {ym(int(v))} (one scan of {table}.{col}, "
                 f"{hms(time.time() - t0)})")
    return int(v)


# ---------------------------------------------------------------- one source file at a time

def commit_piece(con, piece, table, unit, rows, started, write):
    """Write a finished piece and its done marker in one transaction."""
    con.execute("BEGIN TRANSACTION")
    try:
        write()
        con.execute("DELETE FROM w.inv_done WHERE piece = ? AND tbl = ? AND unit = ?", [piece, table, unit])
        con.execute("INSERT INTO w.inv_done VALUES (?, ?, ?, ?, ?, current_timestamp)",
                    [piece, table, unit, rows, time.time() - started])
        con.execute("COMMIT")
    except BaseException:
        try:
            con.execute("ROLLBACK")
        except duckdb.Error:
            pass
        raise


def write_misc(con, table, unit, items):
    for key, n in items:
        con.execute("DELETE FROM w.p1_misc WHERE tbl = ? AND unit = ? AND key = ?", [table, unit, key])
        con.execute("INSERT INTO w.p1_misc VALUES (?, ?, ?, ?)", [table, unit, key, n])


def run_ll(con, T, unit):
    """Loan-level fields: one value per loan (agreed non-NULL value, NULL, or CONFLICTING)."""
    started = time.time()
    names = []
    chunks = [T.ll[i:i + LL_CHUNK] for i in range(0, len(T.ll), LL_CHUNK)]
    for k, chunk in enumerate(chunks):
        aggs = ["count(*) AS n_rows"] if k == 0 else []
        if k == 0 and T.ag == "fan":
            aggs.append(f"arg_min_null(maturity_date, {month_idx('monthly_reporting_period', 'MMYYYY')}) "
                        f"AS maturity_first_row")
        for c in chunk:
            aggs += [f"CASE WHEN min({q(c)}) = max({q(c)}) THEN min({q(c)}) END AS {q(c)}",
                     f"coalesce(min({q(c)}) <> max({q(c)}), false) AS {q(c + '__conf')}",
                     f"(min({q(c)}) IS NOT NULL AND bool_or({q(c)} IS NULL)) AS {q(c + '__mix')}"]
        con.execute(f"CREATE OR REPLACE TEMP TABLE _ll{k} AS SELECT loan_identifier AS loan_id, {', '.join(aggs)} "
                    f"FROM {T.table} WHERE source_file = ? AND loan_identifier IS NOT NULL "
                    f"GROUP BY loan_identifier", [unit])
        names.append(f"_ll{k}")
    rows, no_id = con.execute(f"SELECT count(*), count(*) FILTER (loan_identifier IS NULL) FROM {T.table} "
                              f"WHERE source_file = ?", [unit]).fetchone()
    joined = names[0] + "".join(f" JOIN {n} USING (loan_id)" for n in names[1:])

    def write():
        con.execute(f"DELETE FROM w.{T.ag}_ll WHERE unit = ?", [unit])
        con.execute(f"INSERT INTO w.{T.ag}_ll BY NAME SELECT ? AS unit, * FROM {joined}", [unit])
        write_misc(con, T.table, unit, [("ll_rows", rows), ("ll_rows_no_id", no_id)])

    commit_piece(con, "ll", T.table, unit, rows, started, write)
    for n in names:
        con.execute(f"DROP TABLE IF EXISTS {n}")
    return rows


def build_uloan(con, T, unit):
    """_uloan: each loan in this file with its resolved first payment date (state + month index)."""
    if T.ag == "fre":
        con.execute("CREATE OR REPLACE TEMP TABLE _uids AS SELECT DISTINCT loan_identifier AS loan_id "
                    "FROM freddie_perf WHERE source_file = ? AND loan_identifier IS NOT NULL", [unit])
        st, idx = state_sql("l.v", "l.k", "YYYYMM")
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _uloan AS
            SELECT i.loan_id, CASE WHEN l.loan_id IS NULL THEN 'no_orig' ELSE {st} END AS fpd_state,
                   CASE WHEN l.loan_id IS NOT NULL AND NOT l.k THEN {idx} END AS fpd_idx
            FROM _uids i LEFT JOIN (
                SELECT loan_id,
                       CASE WHEN bool_or(first_payment_date__conf) OR count(DISTINCT first_payment_date) > 1
                            THEN NULL ELSE max(first_payment_date) END AS v,
                       bool_or(first_payment_date__conf) OR count(DISTINCT first_payment_date) > 1 AS k
                FROM w.fre_ll SEMI JOIN _uids USING (loan_id) GROUP BY loan_id) l USING (loan_id)""")
    else:
        st, idx = state_sql("first_payment_date", "first_payment_date__conf", "MMYYYY")
        con.execute(f"CREATE OR REPLACE TEMP TABLE _uloan AS SELECT loan_id, {st} AS fpd_state, "
                    f"CASE WHEN NOT first_payment_date__conf THEN {idx} END AS fpd_idx "
                    f"FROM w.fan_ll WHERE unit = ?", [unit])


def numeric_stats(con, items, source, params=None, ctx=None):
    """items = [(key, raw_sql, spec, conf_sql or None)]; returns [(key, stat, value)].
    Pass 1 counts filled values per field (cheap); fields with nothing in them get their counts
    without further work. Pass 2 sorts every value of the other fields into one category (NULL,
    CONFLICTING, non-numeric, a special code or a band) and counts categories with histogram(),
    which is far faster than one filtered count per band. ctx: {name: sql} for range rules (_q, _harp)."""
    params = params or {}
    conf_of = [coalesce_false(conf) for _, _, _, conf in items]
    first = con.execute(f"SELECT count(*), {', '.join(f'count({raw}), count(*) FILTER ({k})' for (_, raw, _, _), k in zip(items, conf_of))} "
                        f"FROM {source}", params).fetchone()
    rows = first[0]
    out_stats, live = [], []
    for i, (key, raw, spec, conf) in enumerate(items):
        filled, n_conf = first[1 + 2 * i], first[2 + 2 * i]
        if filled:
            live.append(i)
            continue
        out_stats += [(key, "n", 0), (key, "null", rows - n_conf), (key, "conf", n_conf), (key, "nonnum", 0),
                      (key, "min", None), (key, "max", None), (key, "numeric_plain", 0)]
        out_stats += [(key, f"sp:{s}", 0) for s in spec.special] + [(key, f"gt:{g}", 0) for g in spec.above]
        out_stats += [(key, "oor", 0)] if spec.rng is not None else []
        out_stats += [(key, f"band:{lab}", 0) for lab, _ in BANDS[spec.bands]]
    if not live:
        return out_stats
    inner, mid, outer = [], [], []
    for i in live:
        key, raw, spec, _ = items[i]
        inner.append(f"TRY_CAST({raw} AS DOUBLE) AS d{i}, ({raw}) IS NOT NULL AS r{i}, {conf_of[i]} AS k{i}")
        mid.append(f"CASE WHEN isfinite(d{i}) THEN d{i} END AS v{i}, r{i}, k{i}")
        v = f"v{i}"
        sp = " OR ".join(f"{v} = {s}" for s in spec.special) or "false"
        cats = [f"WHEN k{i} THEN -3", f"WHEN NOT r{i} THEN -1", f"WHEN {v} IS NULL THEN -2"]
        cats += [f"WHEN {v} = {s} THEN {100 + j}" for j, s in enumerate(spec.special)]
        cats += [f"WHEN {cond.format(x=v)} THEN {k}" for k, (_, cond) in enumerate(BANDS[spec.bands])]
        outer += [f"histogram(CASE {' '.join(cats)} ELSE -9 END) AS h{i}", f"min({v}) AS mn{i}", f"max({v}) AS mx{i}"]
        if spec.rng is not None:
            outer.append(f"count(*) FILTER ({v} IS NOT NULL AND NOT ({sp}) AND NOT ({range_sql(spec.rng, v)})) AS oor{i}")
        outer += [f"count(*) FILTER ({v} > {g}) AS gt{i}_{n}" for n, g in enumerate(spec.above)]
    ctx = ctx or {}
    ctx_in = "".join(f", {sql} AS {name}" for name, sql in ctx.items())
    ctx_mid = "".join(f", {name}" for name in ctx)
    cur = con.execute(f"SELECT {', '.join(outer)} FROM (SELECT {', '.join(mid)}{ctx_mid} FROM "
                      f"(SELECT {', '.join(inner)}{ctx_in} FROM {source}))", params)
    got = dict(zip([d[0] for d in cur.description], cur.fetchone()))
    for i in live:
        key, _, spec, _ = items[i]
        h = got[f"h{i}"] or {}
        bands = [(lab, h.get(k, 0)) for k, (lab, _) in enumerate(BANDS[spec.bands])]
        out_stats += [(key, "n", rows - h.get(-1, 0) - h.get(-3, 0)), (key, "null", h.get(-1, 0)),
                      (key, "conf", h.get(-3, 0)), (key, "nonnum", h.get(-2, 0)),
                      (key, "min", got[f"mn{i}"]), (key, "max", got[f"mx{i}"]),
                      (key, "numeric_plain", sum(n for _, n in bands) + h.get(-9, 0))]
        out_stats += [(key, f"sp:{s}", h.get(100 + j, 0)) for j, s in enumerate(spec.special)]
        out_stats += [(key, f"gt:{g}", got[f"gt{i}_{n}"]) for n, g in enumerate(spec.above)]
        out_stats += [(key, "oor", got[f"oor{i}"])] if spec.rng is not None else []
        out_stats += [(key, f"band:{lab}", n) for lab, n in bands]
    return out_stats


def coalesce_false(sql):
    return f"coalesce({sql or 'false'}, false)"


def date_stats(con, items, source, de, checks=(), ctx=None, params=None):
    """items = [(key, label, raw_sql, fmt, conf_sql or None)]; checks = [(name, condition)] using
    x_<label> (month index), r_<label> (non-NULL) and ctx names. Returns [(key, stat, value)]."""
    inner, outer, keys = [], [], []
    for key, lab, raw, fmt, conf in items:
        inner.append(f"{month_idx(raw, fmt)} AS x_{lab}, ({raw}) IS NOT NULL AS r_{lab}, "
                     f"coalesce({conf or 'false'}, false) AS k_{lab}")
        for st, sql in (("null", f"count(*) FILTER (NOT r_{lab} AND NOT k_{lab})"),
                        ("unparse", f"count(*) FILTER (r_{lab} AND x_{lab} IS NULL)"),
                        ("conf", f"count(*) FILTER (k_{lab})"), ("min", f"min(x_{lab})"), ("max", f"max(x_{lab})"),
                        ("after_end", f"count(*) FILTER (x_{lab} > {de})")):
            outer.append(f"{sql} AS a{len(keys)}")
            keys.append((key, st))
    for name, cond in checks:
        outer.append(f"count(*) FILTER ({cond}) AS a{len(keys)}")
        keys.append(("_check", name))
    ctx_in = "".join(f", {sql} AS {name}" for name, sql in (ctx or {}).items())
    sql = f"SELECT {', '.join(outer)} FROM (SELECT {', '.join(inner)}{ctx_in} FROM {source})"
    vals = con.execute(sql, params or {}).fetchone()
    return [(k, st, v) for (k, st), v in zip(keys, vals)]


MONTHLY_DATE_CHECKS = {
    "fre": [
        ("rows dated before the loan's first payment date", "x_period < fpd"),
        ("rows of loans with no usable first payment date", "fpd IS NULL"),
        ("ZB code present, ZB effective date missing (rows)", "has_zb AND NOT r_zb_date"),
        ("ZB code present, ZB effective date differs from the row period (rows)", "has_zb AND x_zb_date <> x_period"),
        ("ZB effective date present, no ZB code (rows)", "r_zb_date AND NOT has_zb"),
        ("defect settlement date before first payment date (rows)", "x_defect_date < fpd"),
        ("DDLPI more than one month before first payment date (rows)", "x_ddlpi < fpd - 1"),
        ("DDLPI after the row period (rows)", "x_ddlpi > x_period"),
    ],
    "fan": [
        ("rows dated before the loan's first payment date", "x_period < fpd"),
        ("rows of loans with no usable first payment date", "fpd IS NULL"),
        ("ZB code present, ZB effective date missing (rows)", "has_zb AND NOT r_zb_date"),
        ("ZB code present, ZB effective date differs from the row period (rows)", "has_zb AND x_zb_date <> x_period"),
        ("ZB effective date present, no ZB code (rows)", "r_zb_date AND NOT has_zb"),
        ("foreclosure date before first payment date (rows)", "x_fc_date < fpd"),
        ("disposition date before foreclosure date, same row (rows)", "x_disp_date < x_fc_date"),
        ("last paid installment date more than one month before first payment date (rows)", "x_lpi_date < fpd - 1"),
        ("last paid installment date after the row period (rows)", "x_lpi_date > x_period"),
    ],
}


def run_p1(con, T, unit, de):
    """Part 1 counts for this file's monthly fields (rows): codes, numbers, dates, text, history."""
    started = time.time()
    tb = T.table
    rows, no_id = con.execute(f"SELECT count(*), count(*) FILTER (loan_identifier IS NULL) FROM {tb} "
                              f"WHERE source_file = ?", [unit]).fetchone()
    kinds = {c: T.specs[c].kind for c in T.monthly}
    nums = [c for c in T.monthly if kinds[c] == "num"]
    stats = numeric_stats(con, [(c, f"r.{q(c)}", T.specs[c], None) for c in nums],
                          f"{tb} r WHERE r.source_file = $u", {"u": unit}) if nums else []
    dates = [c for c in T.monthly if kinds[c] == "date"]
    stats += date_stats(
        con, [(c, T.specs[c].label, f"r.{q(c)}", T.specs[c].fmt, None) for c in dates],
        f"{tb} r LEFT JOIN _uloan u ON u.loan_id = r.loan_identifier WHERE r.source_file = $u", de,
        MONTHLY_DATE_CHECKS[T.ag], {"fpd": "u.fpd_idx", "has_zb": "r.zero_balance_code IS NOT NULL"}, {"u": unit})
    coded = [c for c in T.monthly if kinds[c] in ("code", "status")]
    texts = [c for c in T.monthly if kinds[c] == "text"]
    hist = [c for c in T.monthly if kinds[c] == "hist"]
    if hist:
        con.execute(f"CREATE OR REPLACE TEMP TABLE _hist AS SELECT {q(hist[0])} AS s, count(*) AS n FROM {tb} "
                    f"WHERE source_file = ? GROUP BY 1", [unit])

    def write():
        for t in ("p1_values", "p1_stats", "p1_text"):
            con.execute(f"DELETE FROM w.{t} WHERE tbl = ? AND unit = ?", [tb, unit])
        con.executemany("INSERT INTO w.p1_stats VALUES (?, ?, ?, ?, ?)",
                        [(tb, unit, k, st, None if v is None else float(v)) for k, st, v in stats])
        for c in coded:
            con.execute(f"INSERT INTO w.p1_values SELECT ?, ?, ?, {q(c)}, count(*) FROM {tb} WHERE source_file = ? "
                        f"GROUP BY {q(c)}", [tb, unit, c, unit])
        for c in texts:
            con.execute(f"INSERT INTO w.p1_text SELECT ?, ?, ?, CASE WHEN {q(c)} IS NOT NULL THEN hash({q(c)}) END, "
                        f"{q(c)} IS NULL, coalesce(upper(trim({q(c)})) = 'OTHER', false), count(*) FROM {tb} "
                        f"WHERE source_file = ? GROUP BY 4, 5, 6", [tb, unit, c, unit])
        items = [("rows", rows), ("rows_no_id", no_id)]
        if hist:
            con.execute("DELETE FROM w.p1_hist WHERE unit = ?", [unit])
            con.execute("DELETE FROM w.p1_tokens WHERE unit = ?", [unit])
            con.execute("INSERT INTO w.p1_hist SELECT ?, hash(s), length(s), n FROM _hist WHERE s IS NOT NULL", [unit])
            con.execute("""INSERT INTO w.p1_tokens SELECT ?, tok, sum(n) FROM (
                               SELECT n, unnest(list_transform(range(0, (length(s) + 1) // 2),
                                                               lambda k: substr(s, 2 * k + 1, 2))) AS tok
                               FROM _hist WHERE s IS NOT NULL) GROUP BY tok""", [unit])
            items.append(("hist_null", con.execute("SELECT coalesce(sum(n), 0) FROM _hist WHERE s IS NULL").fetchone()[0]))
        write_misc(con, tb, unit, items)

    commit_piece(con, "p1", tb, unit, rows, started, write)
    if hist:
        con.execute("DROP TABLE IF EXISTS _hist")
    return rows


PERF_EXPR = {
    "fre": dict(
        per=month_idx("r.period", "YYYYMM"), st="r.current_loan_delinquency_status", zb="r.zero_balance_code",
        la="r.loan_age", dd=month_idx("r.due_date_of_last_paid_installment_ddlpi", "YYYYMM"),
        F="r.borrower_assistance_plan = 'F'", D="r.delinquency_due_to_disaster = 'Y'",
        R="r.borrower_assistance_plan = 'R'", T="r.borrower_assistance_plan = 'T'",
        DEF="r.payment_deferral_flag IN ('C', 'P')", MOD="r.modification_flag IN ('Y', 'P')", O="false",
        modcur="r.modification_flag = 'Y' OR r.payment_deferral_flag = 'C'",
        modrow="r.modification_flag IN ('Y', 'P')", bapnull="NULL::BOOLEAN",
        fc="NULL::INTEGER", fc_bad="false", hltv="NULL::VARCHAR"),
    "fan": dict(
        per=month_idx("r.monthly_reporting_period", "MMYYYY"), st="r.current_loan_delinquency_status",
        zb="r.zero_balance_code", la="r.loan_age", dd=month_idx("r.last_paid_installment_date", "MMYYYY"),
        F="r.borrower_assistance_plan = 'F'", D="false",
        R="r.borrower_assistance_plan = 'R'", T="r.borrower_assistance_plan = 'T'",
        DEF="r.alternative_delinquency_resolution IN ('P', 'C', 'D')", MOD="r.modification_flag = 'Y'",
        O="r.borrower_assistance_plan = 'O'",
        modcur="r.modification_flag = 'Y' OR r.alternative_delinquency_resolution IN ('P', 'C', 'D')",
        modrow="r.modification_flag = 'Y'", bapnull="r.borrower_assistance_plan IS NULL",
        fc=month_idx("r.foreclosure_date", "MMYYYY"),
        fc_bad=f"r.foreclosure_date IS NOT NULL AND {month_idx('r.foreclosure_date', 'MMYYYY')} IS NULL",
        hltv="r.high_loan_to_value_hltv_refinance_option_indicator"),
}


def perf_sql(ag, table):
    """Per-loan facts from one file's monthly rows: one sorted pass (window) + one GROUP BY.
    pay = payment number = months from the first payment date + 1; window = payments 1-24.
    Status symbol rank (used for patterns): X/blank 0, ? 1, 0 2, 1 3, 2 4, 3+ 5, RA 6, F/disaster 7."""
    P = PERF_EXPR[ag]
    syms = ",\n  ".join(f"max(sym) FILTER (pay = {k}) AS s{k}" for k in range(1, 25))
    return f"""
INSERT INTO w.{ag}_perf BY NAME
WITH base AS (
  SELECT r.loan_identifier AS loan_id, u.fpd_idx, u.fpd_state,
         {P['per']} AS per, {P['st']} AS st, {P['zb']} AS zb,
         TRY_CAST({P['la']} AS DOUBLE) AS la, {P['dd']} AS dd,
         coalesce({P['F']}, false) AS ff, coalesce({P['D']}, false) AS fd, coalesce({P['R']}, false) AS fr,
         coalesce({P['T']}, false) AS ft, coalesce({P['DEF']}, false) AS fdef, coalesce({P['MOD']}, false) AS fmod,
         coalesce({P['O']}, false) AS fo, coalesce({P['modcur']}, false) AS modcur,
         coalesce({P['modrow']}, false) AS modrow, {P['bapnull']} AS bapnull,
         {P['fc']} AS fc, coalesce({P['fc_bad']}, false) AS fc_bad, {P['hltv']} AS hltv
  FROM {table} r JOIN _uloan u ON u.loan_id = r.loan_identifier
  WHERE r.source_file = $u
), cls AS (
  SELECT *, per - fpd_idx + 1 AS pay,
         CASE WHEN regexp_full_match(st, '[0-9]+') THEN TRY_CAST(st AS BIGINT) END AS sn
  FROM base
), seq AS (
  SELECT *, coalesce(pay BETWEEN 1 AND 24, false) AS win,
         CASE WHEN st IS NULL THEN 'B' WHEN st = 'XX' THEN 'X' WHEN st = 'RA' THEN 'R'
              WHEN sn IS NOT NULL THEN 'N' ELSE '?' END AS c,
         CASE WHEN ff OR fd THEN 7 WHEN st = 'RA' THEN 6 WHEN sn >= 3 THEN 5 WHEN sn = 2 THEN 4
              WHEN sn = 1 THEN 3 WHEN sn = 0 THEN 2 WHEN st IS NULL OR st = 'XX' THEN 0 ELSE 1 END AS sym,
         lag(per) OVER lw AS prev_per,
         lag(CASE WHEN sn IS NOT NULL THEN {{'sn': sn, 'per': per, 'pay': pay, 'dd': dd}} END IGNORE NULLS) OVER lw AS pn
  FROM cls
  WINDOW lw AS (PARTITION BY loan_id ORDER BY per NULLS LAST, sn NULLS LAST)
)
SELECT $u AS unit, loan_id, any_value(fpd_idx) AS fpd_idx, any_value(fpd_state) AS fpd_state,
  count(*) AS n_rows, count(*) FILTER (per IS NULL) AS n_per_bad,
  count(*) FILTER (per = prev_per) AS n_dup,
  coalesce(sum(per - prev_per - 1) FILTER (per - prev_per > 1), 0) AS n_gap_months,
  min(per) AS first_per, max(per) AS last_per,
  min(per) FILTER (zb IS NOT NULL) AS zb_per,
  arg_min(zb, per) FILTER (zb IS NOT NULL) AS zb_code,
  coalesce(min(zb) <> max(zb), false) AS zb_multi,
  count(*) FILTER (win) AS w_rows, count(*) FILTER (win AND per = prev_per) AS w_dup,
  min(pay) FILTER (win) AS w_min_pay, max(pay) FILTER (win) AS w_max_pay,
  max(sn) FILTER (win) AS w_max_sn,
  coalesce(bool_or(c = 'R') FILTER (win), false) AS w_ra,
  coalesce(bool_or(c = 'N') FILTER (win), false) AS w_num,
  coalesce(bool_or(c = '?') FILTER (win), false) AS w_unp,
  min(per) FILTER (win AND c IN ('X', 'B')) AS xb_min, max(per) FILTER (win AND c IN ('X', 'B')) AS xb_max,
  coalesce(bool_or(ff) FILTER (win), false) AS f_forb, coalesce(bool_or(fd) FILTER (win), false) AS f_dis,
  coalesce(bool_or(fr) FILTER (win), false) AS f_rep, coalesce(bool_or(ft) FILTER (win), false) AS f_trial,
  coalesce(bool_or(fdef) FILTER (win), false) AS f_def, coalesce(bool_or(fmod) FILTER (win), false) AS f_mod,
  coalesce(bool_or(fo) FILTER (win), false) AS f_oth,
  coalesce(bool_or(sn >= 3 AND NOT ff AND NOT fd) FILTER (win), false) AS f_3nf,
  bool_or(bapnull) FILTER (win) AS bap_null,
  bool_or(bapnull AND sn >= 3 AND NOT ff AND NOT fd) FILTER (win) AS bap_null_3nf,
  coalesce(bool_or(win AND sn - pn.sn > per - pn.per), false) AS g1,
  coalesce(bool_or(sn > pay) FILTER (win), false) AS g2,
  coalesce(bool_or(sn = pay) FILTER (win), false) AS g2_eq,
  coalesce(bool_or(win AND sn < pn.sn AND dd <= pn.dd AND NOT modcur), false) AS g3,
  coalesce(bool_or(win AND sn < pn.sn AND dd <= pn.dd AND modcur), false) AS g3_mod,
  coalesce(bool_or(win AND sn < pn.sn AND (dd IS NULL OR pn.dd IS NULL)), false) AS g3_na,
  count(*) FILTER (win AND sn >= 1 AND (pn IS NULL OR pn.pay < 1 OR pn.sn = 0)) AS n_stretch,
  coalesce(bool_or(sn >= 1 OR c = 'R') FILTER (win), false) AS w_delinq,
  coalesce(bool_or(la IS NULL) FILTER (win), false) AS la_missing,
  coalesce(bool_or(la <> pay AND NOT modrow) FILTER (win), false) AS la_diff_unmod,
  coalesce(bool_or(la <> pay AND modrow) FILTER (win), false) AS la_diff_mod,
  {syms},
  min(fc) AS fc_min, max(fc) AS fc_max, coalesce(bool_or(fc_bad), false) AS fc_bad,
  arg_min_null(hltv, per) AS hltv_first, coalesce(bool_or(hltv = 'Y'), false) AS hltv_y,
  min(per) FILTER (hltv = 'Y') AS hltv_y_first, max(per) FILTER (hltv IS DISTINCT FROM 'Y') AS hltv_non_y_last,
  count(*) FILTER (pay <= 0) AS rows_before_fpd
FROM seq GROUP BY loan_id
"""


def run_perf(con, T, unit):
    started = time.time()
    rows = con.execute(f"SELECT count(*) FROM {T.table} WHERE source_file = ? AND loan_identifier IS NOT NULL",
                       [unit]).fetchone()[0]

    def write():
        con.execute(f"DELETE FROM w.{T.ag}_perf WHERE unit = ?", [unit])
        con.execute(perf_sql(T.ag, T.table), {"u": unit})

    commit_piece(con, "perf", T.table, unit, rows, started, write)
    return rows


def open_db(args):
    """Raw database read-only (load_raw.connect) with the working database attached read-write."""
    con = load_raw.connect(args.memory_limit, args.threads, read_only=True)
    SPILL_DIR.mkdir(exist_ok=True)
    con.execute(f"SET temp_directory = {lit(SPILL_DIR)}")
    con.execute(f"SET max_temp_directory_size = '{SPILL_CAP_GB}GB'")
    con.execute(f"ATTACH {lit(WORK_DB)} AS w (READ_WRITE)")
    return con


def process_files(args, tabs, units, parts, agencies, des):
    """Run every unfinished piece: ll (loan-level values), p1 (Part 1 monthly counts), perf (Part 2 facts).
    Each piece gets a fresh connection: in testing, one long-lived connection slowed down about 4x
    after the first large spilling queries, and a fresh one does not."""
    holder = [open_db(args)]   # the connection currently open; pieces replace it
    try:
        _process_files(holder, args, tabs, units, parts, agencies, des)
    finally:
        holder[0].close()


def _process_files(holder, args, tabs, units, parts, agencies, des):
    con = holder[0]
    done = set(con.execute("SELECT piece, tbl, unit FROM w.inv_done").fetchall())
    wanted = {"ll"} | ({"p1"} if "part1" in parts else set()) | ({"perf"} if "part2" in parts else set())
    plan = []
    by_table = {}
    for table, unit, rows in units:
        by_table.setdefault(table, []).append((unit, rows or 0))
    if "fre" in agencies:
        plan += [("ll", "freddie_orig", u, r) for u, r in by_table.get("freddie_orig", [])]
        for u, r in by_table.get("freddie_perf", []):
            plan += [(p, "freddie_perf", u, r) for p in ("p1", "perf") if p in wanted]
    if "fan" in agencies:
        for u, r in by_table.get("fannie_raw", []):
            plan += [(p, "fannie_raw", u, r) for p in ("ll", "p1", "perf") if p in wanted]
    todo = [p for p in plan if (p[0], p[1], p[2]) not in done]
    progress(f"{len(plan) - len(todo)} of {len(plan)} file pieces already finished; {len(todo)} to run.")
    if not todo:
        return

    history = con.execute("SELECT piece, tbl, sum(rows), sum(seconds) FROM w.inv_done GROUP BY ALL").fetchall()
    rate = {(p, t): r / s for p, t, r, s in history if s and r}
    run_start = time.time()
    for n, (piece, table, unit, rows) in enumerate(todo, start=1):
        T = tabs[table]
        con.close()
        holder[0] = con = open_db(args)
        if piece != "ll":
            needs = ([("ll", "freddie_orig", u) for u, _ in by_table.get("freddie_orig", [])] if T.ag == "fre"
                     else [("ll", "fannie_raw", unit)])
            done = set(con.execute("SELECT piece, tbl, unit FROM w.inv_done").fetchall())
            missing = [k for k in needs if k not in done]
            if missing:
                progress(f"[{n}/{len(todo)}] {table} {unit} {piece}: SKIPPED, its loan-level values are not "
                         f"finished ({len(missing)} piece(s) missing)")
                ERRORS.append(f"{table} {unit} {piece}: skipped")
                continue
        check_disk(args.min_free_gb, f"before {table} {unit} {piece}")
        t0 = time.time()
        try:
            if piece == "ll":
                run_ll(con, T, unit)
            else:
                build_uloan(con, T, unit)
                if piece == "p1":
                    run_p1(con, T, unit, des[T.ag])
                else:
                    run_perf(con, T, unit)
        except (KeyboardInterrupt, StopRun):
            raise
        except Exception as exc:
            progress(f"[{n}/{len(todo)}] {table} {unit} {piece}: FAILED")
            report_error(exc, ERROR_DETAIL_PATH, progress)
            ERRORS.append(f"{table} {unit} {piece}: failed")
            continue
        secs = time.time() - t0
        if rows and secs:
            old = rate.get((piece, table))
            rate[(piece, table)] = rows / secs if old is None else 0.5 * old + 0.5 * rows / secs
        remaining = 0.0
        unknown = False
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


# ---------------------------------------------------------------- final phase: per-loan tables

def ll_specs(T):
    return [(c, T.specs[c]) for c in T.ll]


def resolved_view(con, T):
    """<ag>_lr (working database, read through the temp view _lr_<ag>): one row per loan; a loan with
    rows in more than one file combines to the agreed value / NULL / CONFLICTING. Stored once per run
    because about forty report queries read it."""
    ag = T.ag
    con.execute(f"CREATE OR REPLACE TEMP TABLE _dupll_{ag} AS SELECT loan_id FROM w.{ag}_ll GROUP BY loan_id "
                f"HAVING count(*) > 1")
    plain = ["loan_id", "unit", "1 AS n_units", "n_rows"]
    comb = ["loan_id", "min(unit) AS unit", "count(*) AS n_units", "sum(n_rows) AS n_rows"]
    for c in T.ll:
        qc, kc, xc = q(c), q(c + "__conf"), q(c + "__mix")
        conf = f"(bool_or({kc}) OR count(DISTINCT {qc}) > 1)"
        plain += [qc, kc, xc]
        comb += [f"CASE WHEN {conf} THEN NULL ELSE max({qc}) END AS {qc}", f"{conf} AS {kc}",
                 f"(bool_or({xc}) OR (count({qc}) >= 1 AND count(*) FILTER ({qc} IS NULL AND NOT {kc}) >= 1)) AS {xc}"]
    if ag == "fan":
        plain.append("maturity_first_row")
        comb.append("CASE WHEN count(DISTINCT maturity_first_row) = 1 THEN max(maturity_first_row) END AS maturity_first_row")
    con.execute(f"""CREATE OR REPLACE TABLE w.{ag}_lr AS
        SELECT {', '.join(plain)} FROM w.{ag}_ll ANTI JOIN _dupll_{ag} USING (loan_id)
        UNION ALL
        SELECT {', '.join(comb)} FROM w.{ag}_ll SEMI JOIN _dupll_{ag} USING (loan_id) GROUP BY loan_id""")
    con.execute(f"CREATE OR REPLACE TEMP VIEW _lr_{ag} AS SELECT * FROM w.{ag}_lr")


def missing_sql(spec, col, conf):
    """NULL (not CONFLICTING) or a documented not-available code: used for missing-value combinations."""
    na = "false"
    if spec.kind in ("code", "geo") and spec.na:
        na = f"{col} IN ({', '.join(lit(c) for c in spec.na)})"
    elif spec.kind == "num" and spec.special:
        na = f"TRY_CAST({col} AS DOUBLE) IN ({', '.join(str(s) for s in spec.special)})"
    return f"(NOT {conf} AND ({col} IS NULL OR {na}))"


def special_sql(spec, col):
    """Documented special codes for the by-year tables (not-available codes and other documented specials)."""
    if spec.kind in ("code", "geo"):
        codes = tuple(spec.na) + tuple(spec.special)
        return f"{col} IN ({', '.join(lit(c) for c in codes)})" if codes else None
    if spec.kind == "num" and spec.special:
        return f"TRY_CAST({col} AS DOUBLE) IN ({', '.join(str(s) for s in spec.special)})"
    return None


def prepare_agency(con, tabs, ag, de):
    """Universe of loans (<ag>_ids) and the loan-level dimensions (a, b, l, o, q, BAP coverage) in <ag>_l2.
    Both are rebuilt every run in the working database (on disk, compressed) rather than in memory."""
    T = tabs["freddie_orig"] if ag == "fre" else tabs["fannie_raw"]
    resolved_view(con, T)
    con.execute(f"""CREATE OR REPLACE TABLE w.{ag}_ids AS
        SELECT loan_id, bool_or(src = 'll') AS has_ll, bool_or(src = 'perf') AS has_perf FROM (
            SELECT loan_id, 'll' AS src FROM w.{ag}_ll UNION ALL SELECT loan_id, 'perf' FROM w.{ag}_perf)
        GROUP BY loan_id""")
    # missing-value combinations: fields missing for every loan are listed once instead
    qfields = [(c, s) for c, s in ll_specs(T) if s.kind != "id"]
    miss = [missing_sql(s, f"l.{q(c)}", f"l.{q(c + '__conf')}") for c, s in qfields]
    n_ll = con.execute(f"SELECT count(*) FROM _lr_{ag}").fetchone()[0]
    counts = con.execute(f"SELECT {', '.join(f'count(*) FILTER ({x})' for x in miss)} FROM _lr_{ag} l").fetchone() \
        if n_ll else [0] * len(miss)
    always = [s.label for (c, s), n in zip(qfields, counts) if n_ll and n == n_ll]
    used = [((c, s), x) for (c, s), x in zip(qfields, miss) if s.label not in always]
    nmiss = " + ".join(f"({x})::INTEGER" for _, x in used) or "0"
    combo = "concat_ws(',', " + ", ".join(f"CASE WHEN {x} THEN {lit(s.label)} END" for (c, s), x in used) + ")" \
        if used else "''"
    has = "l.loan_id IS NOT NULL"
    fpd_st, fpd = state_sql("l.first_payment_date", "l.first_payment_date__conf", "YYYYMM" if ag == "fre" else "MMYYYY")
    fpd_whens = [(f"{has} AND l.first_payment_date__conf", lit("first payment date CONFLICTING")),
                 (f"{has} AND l.first_payment_date IS NULL", lit("first payment date missing")),
                 (f"{has} AND {fpd} IS NULL", lit("first payment date unparseable"))]
    b_edges = [("<0", None, -1), ("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-6", 4, 6), ("7-12", 7, 12),
               ("13-24", 13, 24), (">24", 25, None)]
    q_edges = [("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4+", 4, None)]
    cols = {}
    if ag == "fre":
        uok = UNIT_OK.format(u="l.unit")
        one = f"{has} AND l.n_units = 1"
        cols["year"] = f"CASE WHEN {one} AND {uok} THEN substr(l.unit, 1, 4) ELSE 'unknown' END"
        cols["a"] = case([(f"NOT ({has})", lit(NO_ORIG)),
                          ("l.n_units > 1", lit("CONFLICTING (rows in more than one source file)")),
                          (uok, "l.unit")])
        cols["b"] = case([(f"NOT ({has})", lit(NO_ORIG)), ("l.n_units > 1", lit("origination quarter CONFLICTING"))]
                         + fpd_whens + buckets(f"({fpd} - {unit_qend('l.unit')})", b_edges, f"{one} AND {uok}"))
        cols["q_n"] = case([(f"NOT ({has})", lit(NO_ORIG))] + buckets(f"({nmiss})", q_edges, has))
        cols["d_ll_row"] = f"CASE WHEN {has} THEN 'yes' ELSE 'no' END"
        cols["d_unit_format"] = f"CASE WHEN {has} AND {uok} THEN 'ok' WHEN {has} THEN 'not a quarter label' ELSE 'n/a' END"
        cols["d_ll_fpd_state"] = f"CASE WHEN {has} THEN {fpd_st} ELSE 'n/a' END"
    else:
        od_st, od = state_sql("l.origination_date", "l.origination_date__conf", "MMYYYY")
        od_ok = f"{has} AND NOT l.origination_date__conf AND {od} IS NOT NULL"
        od_und = f"{has} AND NOT ({od_ok})"
        oq_end = f"(({od}) // 3 * 3 + 2)"
        cols["year"] = f"CASE WHEN {od_ok} THEN CAST(({od}) // 12 AS VARCHAR) ELSE 'unknown' END"
        cols["a"] = case([(f"{has} AND l.origination_date__conf", lit("origination date CONFLICTING")),
                          (f"{has} AND l.origination_date IS NULL", lit("origination date missing")),
                          (f"{has} AND {od} IS NULL", lit("origination date unparseable")),
                          (f"{od_ok} AND {od} <= {de}",
                           f"CAST(({od}) // 12 AS VARCHAR) || 'Q' || CAST(({od}) % 12 // 3 + 1 AS VARCHAR)")])
        cols["b"] = case([(od_und, lit("origination quarter undetermined"))] + fpd_whens
                         + buckets(f"({fpd} - {oq_end})", b_edges, f"{od_ok} AND {od} <= {de}"))
        acq = unit_q("l.unit")
        orig_q = f"(({od}) // 12 * 4 + ({od}) % 12 // 3 + 1)"
        cols["l"] = case([(f"{has} AND l.n_units > 1", lit("in more than one acquisition file")),
                          (od_und, lit("origination quarter undetermined"))]
                         + buckets(f"({acq} - {orig_q})",
                                   [("negative (acquired before the origination quarter)", None, -1), ("0", 0, 0),
                                    ("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5-8", 5, 8), (">8", 9, None)],
                                   f"{od_ok} AND {acq} IS NOT NULL"))
        nconf = " + ".join(f"l.{q(c + '__conf')}::INTEGER" for c in T.ll)
        cols["o"] = case([(f"{has} AND ({nconf}) = 0", lit("none")),
                          (f"{has} AND ({nconf}) > 0", lit("one or more fields CONFLICTING"))])
        cols["q_n"] = case(buckets(f"({nmiss})", q_edges, has))
        cols["bap"] = case([(f"{has} AND (l.first_payment_date__conf OR {fpd} IS NULL)", lit("first payment date unknown")),
                            (f"{has} AND {fpd} + 23 < {BAP_START}", lit("payments 1-24 entirely before April 2020")),
                            (f"{has} AND {fpd} >= {BAP_START}", lit("payments 1-24 entirely in April 2020 or later")),
                            (f"{has} AND {fpd} < {BAP_START} AND {fpd} + 23 >= {BAP_START}",
                             lit("payments 1-24 span April 2020"))])
        cols["d_ll_row"] = f"CASE WHEN {has} THEN 'yes' ELSE 'no' END"
        cols["d_unit_format"] = (f"CASE WHEN {has} AND {UNIT_OK.format(u='l.unit')} THEN 'ok' "
                                 f"WHEN {has} THEN 'not a quarter label' ELSE 'n/a' END")
        cols["d_ll_fpd_state"] = f"CASE WHEN {has} THEN {fpd_st} ELSE 'n/a' END"
        cols["d_orig_date_state"] = f"CASE WHEN {has} THEN {od_st} ELSE 'n/a' END"
    cols["q_combo"] = f"CASE WHEN {has} AND ({nmiss}) IS NOT NULL THEN {combo} END"
    cols["fpd_idx"] = f"CASE WHEN {has} AND NOT l.first_payment_date__conf THEN {fpd} END"
    sel = ", ".join(f"{expr} AS {name}" for name, expr in cols.items())
    con.execute(f"""CREATE OR REPLACE TABLE w.{ag}_l2 AS
        SELECT i.loan_id, i.has_ll, i.has_perf, {sel}
        FROM w.{ag}_ids i LEFT JOIN _lr_{ag} l USING (loan_id)""")
    return always


def perf_dims(ag, de):
    """Part 2 dimensions from per-loan monthly facts (alias p): name -> SQL."""
    ok = "(p.fpd_state = 'ok' AND p.fpd_idx IS NOT NULL AND p.first_per IS NOT NULL)"
    W = f"{ok} AND p.w_rows > 0"
    first_pay, last_pay = "(p.first_per - p.fpd_idx + 1)", "(p.last_per - p.fpd_idx + 1)"
    zb_pay = "(p.zb_per - p.fpd_idx + 1)"
    sp = [("p.fpd_state = 'no_orig'", lit(NO_ORIG)), ("p.fpd_state = 'missing'", lit("first payment date missing")),
          ("p.fpd_state = 'unparseable'", lit("first payment date unparseable")),
          ("p.fpd_state = 'conflicting'", lit("first payment date CONFLICTING")),
          ("p.n_rows > 0 AND p.first_per IS NULL", lit("monthly periods all unparseable"))]
    no_win = (f"{ok} AND p.w_rows = 0", lit("no rows in payments 1-24"))
    d = {}
    d["c"] = case(sp + buckets(first_pay, [("<=0", None, 0), ("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-6", 4, 6),
                                           ("7-12", 7, 12), ("13-24", 13, 24), (">24", 25, None)], ok))
    code_txt = "CASE WHEN length(p.zb_code) <= 3 THEN p.zb_code ELSE '(undocumented, longer than 3)' END"
    d["d"] = case(sp + [
        (f"{ok} AND p.zb_per IS NOT NULL AND {zb_pay} BETWEEN 1 AND 24", f"'terminated in payments 1-24: code ' || {code_txt}"),
        (f"{ok} AND p.zb_per IS NOT NULL AND {zb_pay} <= 0", lit("termination code before payment 1")),
        (f"{ok} AND {last_pay} >= 24", lit("reached payment 24, no code by then")),
        (f"{ok} AND p.last_per < {de} AND p.fpd_idx + 23 <= {de}", lit("records stop, no code; payment 24 within the data")),
        (f"{ok} AND p.last_per < {de} AND p.fpd_idx + 23 > {de}", lit("records stop, no code; payment 24 after the data end")),
        (f"{ok} AND p.last_per = {de} AND p.fpd_idx + 23 > {de}",
         lit("still reporting at the data end; payment 24 after the data end"))])
    d["e"] = case(sp + [no_win, (f"{W} AND p.w_ra", lit("REO (RA)")), (f"{W} AND p.w_max_sn >= 3", lit("3 or more")),
                        (f"{W} AND p.w_max_sn = 2", lit("2")), (f"{W} AND p.w_max_sn = 1", lit("1")),
                        (f"{W} AND p.w_max_sn = 0", lit("0")),
                        (f"{W} AND NOT p.w_num AND NOT p.w_ra AND NOT p.w_unp", lit("only XX/blank"))])
    kinds = "(p.g1::INTEGER + p.g2::INTEGER + p.g3::INTEGER)"
    d["g"] = case(sp + [(f"{ok} AND NOT p.w_num", lit("no numeric status in window")),
                        (f"{ok} AND p.w_num AND {kinds} = 0", lit("none")),
                        (f"{ok} AND p.w_num AND {kinds} >= 2", lit("more than one kind")),
                        (f"{ok} AND p.w_num AND p.g1", lit("only kind 1 (jump)")),
                        (f"{ok} AND p.w_num AND p.g2", lit("only kind 2 (status above payment number)")),
                        (f"{ok} AND p.w_num AND p.g3", lit("only kind 3 (drop with no payment)"))])
    d["la"] = case(sp + [no_win, (f"{W} AND p.la_diff_unmod", lit("differs on an unmodified row")),
                         (f"{W} AND p.la_diff_mod", lit("differs only on modified rows")),
                         (f"{W} AND p.la_missing", lit("loan_age missing on some rows, no differences")),
                         (f"{W} AND NOT p.la_missing", lit("agrees throughout the window"))])
    gap = "(p.w_max_pay - p.w_min_pay + 1 - (p.w_rows - p.w_dup))"
    d["k"] = case(sp + [no_win, (f"{W} AND p.w_dup = 0 AND {gap} = 0", lit("continuous")),
                        (f"{W} AND p.w_dup = 0 AND {gap} > 0", lit("missing months only")),
                        (f"{W} AND p.w_dup > 0 AND {gap} = 0", lit("duplicate period rows only")),
                        (f"{W} AND p.w_dup > 0 AND {gap} > 0", lit("missing months and duplicate period rows"))])
    d["m"] = case([("p.n_rows > 0 AND p.zb_per IS NULL", lit("no termination code")),
                   ("p.last_per = p.zb_per", lit("termination code on the last row")),
                   ("p.last_per > p.zb_per", lit("termination code, then more rows"))])
    d["n"] = case(sp + [(f"{ok} AND p.w_ra", lit("REO (RA) in window")),
                        (f"{ok} AND NOT p.w_num", lit("no numeric statuses in window")),
                        (f"{ok} AND p.n_stretch = 0", lit("never behind")),
                        (f"{ok} AND p.n_stretch = 1", lit("behind once (one stretch)")),
                        (f"{ok} AND p.n_stretch = 2", lit("behind, back to current, behind again (two stretches)")),
                        (f"{ok} AND p.n_stretch >= 3", lit("three or more stretches"))])
    if ag == "fan":
        fc_pay = "(p.fc_min - p.fpd_idx + 1)"
        d["h_fc"] = case(sp + [(f"{ok} AND p.fc_bad", lit("foreclosure date unparseable")),
                               (f"{ok} AND p.fc_min IS NULL", lit("no foreclosure date")),
                               (f"{ok} AND p.fc_min <> p.fc_max", lit("conflicting foreclosure dates")),
                               (f"{ok} AND {fc_pay} BETWEEN 1 AND 24", lit("yes, within payments 1-24")),
                               (f"{ok} AND {fc_pay} > 24", lit("after the window")),
                               (f"{ok} AND {fc_pay} <= 0", lit("before payment 1"))])
        d["h_hltv"] = case([("p.n_rows > 0 AND NOT p.hltv_y", lit("never")),
                            ("p.hltv_y AND p.hltv_non_y_last > p.hltv_y_first", lit("Y, then back to N/blank")),
                            ("p.hltv_y AND p.hltv_first = 'Y'", lit("Y from the first row")),
                            ("p.hltv_y", lit("Y appears later"))])
    flags = [("F", "p.f_forb"), ("D", "p.f_dis"), ("R", "p.f_rep"), ("T", "p.f_trial"), ("DEF", "p.f_def"),
             ("MOD", "p.f_mod"), ("O", "p.f_oth"),
             ("XB", "(p.xb_min IS NOT NULL AND (p.zb_per IS NULL OR p.xb_min <= p.zb_per))"),
             ("XBT", "(p.xb_max IS NOT NULL AND p.zb_per IS NOT NULL AND p.xb_max > p.zb_per)"),
             ("3NF", "p.f_3nf")]
    if ag == "fre":
        flags = [f for f in flags if f[0] != "O"]
    else:
        flags = [f for f in flags if f[0] != "D"]
    extra = {f"flag_{k}": f"CASE WHEN {W} THEN coalesce({v}, false) END" for k, v in flags}
    extra["f_combo"] = (f"CASE WHEN {W} THEN coalesce(nullif(concat_ws('+', "
                        + ", ".join(f"CASE WHEN {v} THEN '{k}' END" for k, v in flags) + "), ''), 'none') "
                        f"WHEN {ok} THEN 'no rows in payments 1-24' ELSE 'n/a' END")
    extra["g_any"] = (f"CASE WHEN {ok} AND p.w_num AND {kinds} > 0 THEN 'yes' WHEN {ok} AND p.w_num THEN 'no' "
                      f"ELSE 'n/a' END")
    extra["f3nf"] = f"CASE WHEN {W} AND p.f_3nf THEN 'yes' WHEN {W} THEN 'no' ELSE 'n/a' END"
    pieces = []
    for k in range(1, 25):
        pieces.append(f"CASE WHEN p.zb_per IS NOT NULL AND {k} > {zb_pay} THEN 'T' "
                      f"WHEN p.fpd_idx + {k - 1} > {de} THEN '_' WHEN p.s{k} IS NULL THEN '.' "
                      f"ELSE substr('X?0123RF', p.s{k} + 1, 1) END")
    extra["pattern"] = f"CASE WHEN {ok} THEN concat({', '.join(pieces)}) END"
    extra["w_delinq"] = f"CASE WHEN {ok} THEN p.w_delinq END"
    extra["zb_multi"] = "p.zb_multi"
    extra["zb_after24"] = f"CASE WHEN {ok} THEN p.zb_per IS NOT NULL AND {zb_pay} > 24 END"
    for c in ("g3_mod", "g3_na", "g2_eq"):
        extra[c] = f"CASE WHEN {ok} AND p.w_num THEN p.{c} END"
    extra["bap_null"] = f"CASE WHEN {W} THEN p.bap_null END"
    extra["bap_null_3nf"] = f"CASE WHEN {W} THEN p.bap_null_3nf END"
    extra["d_fpd_state"] = "p.fpd_state"
    extra["d_rows_in_window"] = "CASE WHEN p.w_rows > 0 THEN 'yes' WHEN p.w_rows = 0 THEN 'no' ELSE 'NULL' END"
    extra["d_numeric_status"] = "CASE WHEN p.w_num THEN 'yes' WHEN NOT p.w_num THEN 'no' ELSE 'NULL' END"
    extra["d_unparseable_status"] = "CASE WHEN p.w_unp THEN 'yes' WHEN NOT p.w_unp THEN 'no' ELSE 'NULL' END"
    extra["d_termination_code"] = "CASE WHEN p.zb_per IS NOT NULL THEN 'yes' ELSE 'no' END"
    extra["d_first_period"] = "CASE WHEN p.first_per IS NOT NULL THEN 'parsed' ELSE 'NULL' END"
    return d, extra


def build_p2(con, ag, de):
    """<ag>_p2 in the working database: one row per loan with every monthly-records dimension."""
    dims, extra = perf_dims(ag, de)
    perf_sel = ", ".join(f"{e} AS {n}" for n, e in {**dims, **extra}.items())

    def const(label):
        return ", ".join([f"{lit(label)} AS {n}" for n in dims] + [f"NULL AS {n}" for n in extra])

    con.execute(f"""CREATE OR REPLACE TABLE w.{ag}_p2 AS
        WITH sp AS (SELECT loan_id FROM w.{ag}_perf GROUP BY loan_id HAVING count(*) > 1)
        SELECT p.loan_id, coalesce(y.year, 'unknown') AS year, {perf_sel}
        FROM (SELECT * FROM w.{ag}_perf ANTI JOIN sp USING (loan_id)) p LEFT JOIN w.{ag}_l2 y USING (loan_id)
        UNION ALL BY NAME
        SELECT sp.loan_id, coalesce(y.year, 'unknown') AS year, {const(SPLIT)}
        FROM sp LEFT JOIN w.{ag}_l2 y USING (loan_id)
        UNION ALL BY NAME
        SELECT i.loan_id, coalesce(y.year, 'unknown') AS year, {const(NO_RECORDS)}
        FROM w.{ag}_ids i LEFT JOIN w.{ag}_l2 y USING (loan_id) WHERE NOT i.has_perf""")
    return list(dims), list(extra)


# ---------------------------------------------------------------- report helpers

def years_of(con, src):
    ys = [r[0] for r in con.execute(f"SELECT DISTINCT year FROM {src}").fetchall()]
    return sorted([y for y in ys if y != "unknown"]) + (["unknown"] if "unknown" in ys else [])


def by_year_table(left_headers, rows, years):
    """rows: [(left cells, total, {year: n})]; printed in chunks of YEAR_COLS year columns."""
    if not rows:
        out("  (none)")
        return
    for start in range(0, max(len(years), 1), YEAR_COLS):
        ys = years[start:start + YEAR_COLS]
        if start:
            out()
        out_table(left_headers + ["total"] + ys,
                  [list(left) + [m(total)] + [m(d.get(y, 0)) for y in ys] for left, total, d in rows],
                  left=set(range(len(left_headers))))


def sort_key(order):
    specials = [NO_ORIG, NO_RECORDS, SPLIT, "first payment date missing", "first payment date unparseable",
                "first payment date CONFLICTING", "monthly periods all unparseable"]

    def key(v):
        if v == CATCH_ALL:
            return (4, 0, "")
        if str(v).startswith("terminated in payments 1-24"):
            return (0, -1, str(v))
        if v in order:
            return (0, order.index(v), "")
        if v in specials:
            return (2, specials.index(v), "")
        return (1, 0, str(v))
    return key


def print_dim(con, src, col, title, n_total, year_tot, years, order=(), profile=()):
    """One dimension by origination year; checks that values add up overall and per year."""
    out()
    out(f"--- {title}")
    per = {}
    for v, y, n in con.execute(f"SELECT {col}, year, count(*) FROM {src} GROUP BY ALL").fetchall():
        per.setdefault("(NULL)" if v is None else v, {})[y] = n
    vals = sorted(per, key=sort_key(list(order)))
    rows = [([v], sum(per[v].values()), per[v]) for v in vals]
    total = sum(r[1] for r in rows)
    by_year_table(["value"], rows + [(["TOTAL"], total, {y: sum(per[v].get(y, 0) for v in per) for y in years})], years)
    bad_years = [y for y in years if sum(per[v].get(y, 0) for v in per) != year_tot.get(y, 0)]
    if total != n_total or bad_years or "(NULL)" in per:
        msg = (f"{title}: values add to {total:,}, loans = {n_total:,}"
               + (f"; years not matching: {', '.join(bad_years)}" if bad_years else "")
               + ("; some loans have no value" if "(NULL)" in per else ""))
        out(f"  FAIL: {msg}")
        FAILS.append(msg)
    else:
        out(f"  Check: values add up to the agency total ({mc(n_total)} loans) and to every year's total: OK")
    if CATCH_ALL in per:
        n = sum(per[CATCH_ALL].values())
        out(f"  {mc(n)} loan(s) in '{CATCH_ALL}'. What they have in common (counts among them):")
        for d in profile:
            rows = con.execute(f"SELECT {d}, count(*) FROM {src} WHERE {col} = '{CATCH_ALL}' GROUP BY 1 "
                               f"ORDER BY 1").fetchall()
            out(f"    {d[2:] if d.startswith('d_') else d}: " + "; ".join(f"{v} = {mc(c)}" for v, c in rows))


def doc_mark(spec, value):
    if value is None:
        return "documented (blank)" if spec.null_doc else "UNDOCUMENTED (blank not described)"
    if spec.kind == "status":
        if re.fullmatch(r"[0-9]{2}", value) or value in spec.codes:
            return "documented"
        if value.strip().isdigit() and int(value.strip()) <= 99:
            return f"UNDOCUMENTED (form of {int(value.strip()):02d})"
    elif spec.kind == "geo":
        return "documented format" if re.fullmatch(spec.fmt, value) else "UNDOCUMENTED (format)"
    elif value in spec.codes:
        return "documented"
    if not spec.codes and spec.kind == "code":
        return "UNDOCUMENTED (no codes in the documentation)"
    v2 = value.strip().upper()
    for c in spec.codes:
        if v2 == c or (v2.isdigit() and c.isdigit() and int(v2) == int(c)):
            return f"UNDOCUMENTED (form of {c})"
    return "UNDOCUMENTED"


def print_coded(name, spec, maxlen, rows, unit_word):
    """rows: [(value or None, is_conflicting, n)]."""
    total = sum(n for _, _, n in rows)
    out()
    out(f"  {name} ({spec.label}) - {unit_word}: {mc(total)}")
    table, hidden = [], {}
    for v, conf, n in sorted(rows, key=lambda r: (bool(r[1]), r[0] is None, str(r[0]))):
        if conf:
            table.append(["CONFLICTING (differs across rows)", m(n), pct(n, total), "-"])
        elif v is not None and (len(v) > min(maxlen, 12) or not v.isprintable()):
            h = hidden.setdefault(len(v), [0, 0])
            h[0] += 1
            h[1] += n
        else:
            table.append(["(blank)" if v is None else v, m(n), pct(n, total), doc_mark(spec, v)])
    for length, (k, n) in sorted(hidden.items()):
        table.append([f"(hidden: {mc(k)} value(s) of length {length}, longer than the layout allows)", m(n),
                      pct(n, total), "UNDOCUMENTED"])
    out_table(["value", "count", "share", "documentation"], table, left={0, 3})


def print_numeric(title, tabledesc, stats, specs, unit_word):
    """stats: {key: {stat: value}} merged; tabledesc: [(key, spec)]."""
    out()
    out(f"  Numeric fields ({unit_word}). min/max exclude non-numeric text; bands exclude NULL, "
        f"non-numeric and documented special codes.")
    rows = []
    for key, spec in tabledesc:
        s = stats.get(key, {})
        sp = "; ".join(f"{c}: {mc(s.get(f'sp:{c}', 0))}" for c in spec.special) or "-"
        rng = ("-" if spec.rng is None else RANGE_TEXT[spec.rng] if isinstance(spec.rng, str)
               else f"{spec.rng[0]}-{spec.rng[1]}")
        rows.append([f"{key} ({spec.label})", m(s.get("n", 0)), m(s.get("null", 0)), m(s.get("conf", 0)),
                     m(s.get("nonnum", 0)), fmt_value(s.get("min")), fmt_value(s.get("max")), sp,
                     "-" if spec.rng is None else m(s.get("oor", 0)), rng])
    out_table(["field", "non-NULL", "NULL", "CONFLICTING", "non-numeric", "min", "max", "special codes",
               "outside range", "documented range"], rows, left={0, 7, 9})
    out()
    out(f"  Bands ({unit_word}):")
    brows = []
    for key, spec in tabledesc:
        s = stats.get(key, {})
        bands = [(lab, s.get(f"band:{lab}", 0)) for lab, _ in BANDS[spec.bands]]
        band_sum = sum(n for _, n in bands)
        if band_sum != s.get("numeric_plain", 0):
            msg = f"{title} {key}: bands add to {band_sum:,}, numeric values (no special codes) = {s.get('numeric_plain', 0):,}"
            FAILS.append(msg)
            out(f"  FAIL: {msg}")
        extra = "; ".join(f">{g}: {mc(s.get(f'gt:{g}', 0))}" for g in spec.above)
        brows.append([f"{key} ({spec.label})", "  ".join(f"{lab}: {mc(n)}" for lab, n in bands)
                      + (f"   [{extra}]" if extra else "")])
    out_table(["field", "bands"], brows, left={0, 1})


def print_dates(stats, tabledesc, unit_word, de):
    out()
    out(f"  Date fields ({unit_word}). after data end = later than {ym(de)}.")
    rows = []
    for key, spec in tabledesc:
        s = stats.get(key, {})
        rows.append([f"{key} ({spec.label}, {spec.fmt})", m(s.get("null", 0)), m(s.get("unparse", 0)),
                     m(s.get("conf", 0)), ym(s.get("min")), ym(s.get("max")), m(s.get("after_end", 0))])
    out_table(["field", "NULL", "unparseable", "CONFLICTING", "earliest", "latest", "after data end"], rows, left={0})


def print_checks(title, items):
    out()
    out(f"  {title}")
    out_table(["check", "count"], [[name, m(n)] for name, n in items], left={0})


def merged_stats(con, table):
    stats = {}
    for col, stat, v in con.execute("""
            SELECT col, stat, CASE WHEN stat = 'min' THEN min(v) WHEN stat = 'max' THEN max(v) ELSE sum(v) END
            FROM w.p1_stats WHERE tbl = ? GROUP BY col, stat""", [table]).fetchall():
        stats.setdefault(col, {})[stat] = v
    return stats


def misc_sum(con, table, key):
    return con.execute("SELECT coalesce(sum(n), 0) FROM w.p1_misc WHERE tbl = ? AND key = ?", [table, key]).fetchone()[0]


# ---------------------------------------------------------------- PART 1

def part1_loan_level(con, T, de, title):
    """Loan-level fields from the per-loan resolved values (one count per loan)."""
    ag = T.ag
    section(f"PART 1 - {title}: loan-level fields (counts are LOANS; one value per loan, CONFLICTING if "
            f"the loan's rows disagree)")
    n_loans = con.execute(f"SELECT count(*) FROM _lr_{ag}").fetchone()[0]
    out(f"Loans with loan-level rows: {mc(n_loans)}")
    specs = dict(ll_specs(T))
    kinds = {c: s.kind for c, s in specs.items()}

    out()
    out("Coded fields (every value):")
    for c in [c for c in T.ll if kinds[c] == "code" or (kinds[c] == "geo" and specs[c].listed)]:
        conf = f"coalesce({q(c + '__conf')}, false)"
        rows = con.execute(f"SELECT CASE WHEN {conf} THEN NULL ELSE {q(c)} END, {conf}, count(*) "
                           f"FROM _lr_{ag} GROUP BY ALL").fetchall()
        print_coded(c, specs[c], T.maxlen[c], rows, "loans")

    geos = [c for c in T.ll if kinds[c] == "geo" and not specs[c].listed]
    if geos:
        out()
        out("Geographic codes, summarised (values not listed):")
        rows = []
        for c in geos:
            s = specs[c]
            spc = tuple(s.na) + tuple(s.special)
            spec_cond = f"{q(c)} IN ({', '.join(lit(x) for x in spc)})" if spc else "false"
            r = con.execute(f"""SELECT count(DISTINCT {q(c)}), count(*) FILTER ({q(c)} IS NULL AND NOT {q(c + '__conf')}),
                                       count(*) FILTER ({spec_cond}),
                                       count(*) FILTER ({q(c)} IS NOT NULL AND NOT regexp_full_match({q(c)}, '{s.fmt}')),
                                       count(*) FILTER ({q(c + '__conf')})
                                FROM _lr_{ag}""").fetchone()
            rows.append([f"{c} ({s.label})", m(r[0]), m(r[1]), f"{'/'.join(spc) or '-'}: {mc(r[2])}", m(r[3]),
                         m(r[4]), s.fmt, "documented" if s.null_doc else "UNDOCUMENTED"])
        out_table(["field", "distinct values", "NULL", "special code", "not in documented format", "CONFLICTING",
                   "documented format", "NULL is"], rows, left={0, 6, 7})

    nums = [c for c in T.ll if kinds[c] == "num"]
    if nums:
        ctx = ({"_q": unit_q("unit"), "_harp": "harp_indicator = 'Y'"} if ag == "fre" else None)
        st = numeric_stats(con, [(c, q(c), specs[c], q(c + "__conf")) for c in nums], f"_lr_{ag}", ctx=ctx)
        stats = {}
        for k, s, v in st:
            stats.setdefault(k, {})[s] = v
        print_numeric(title, [(c, specs[c]) for c in nums], stats, specs, "loans")

    dates = [c for c in T.ll if kinds[c] == "date"]
    fmt = "YYYYMM" if ag == "fre" else "MMYYYY"
    if ag == "fre":
        checks = [("first payment date before the start of the origination quarter (source_file)",
                   f"x_fpd < {unit_qstart('_unit')}"),
                  ("maturity date before first payment date", "x_maturity < x_fpd"),
                  ("original loan term differs from maturity - first payment + 1",
                   "TRY_CAST(_term AS DOUBLE) <> x_maturity - x_fpd + 1"),
                  ("first payment date after the data's last month", f"x_fpd > {de}")]
        ctx = {"_unit": "unit", "_term": "original_loan_term"}
    else:
        checks = [("first payment date before the origination month", "x_fpd < x_orig_date"),
                  ("first payment date in the origination month", "x_fpd = x_orig_date"),
                  ("maturity date before first payment date", "x_maturity < x_fpd"),
                  ("original loan term differs from maturity (earliest row) - first payment + 1",
                   f"TRY_CAST(_term AS DOUBLE) <> {month_idx('_mat_first', 'MMYYYY')} - x_fpd + 1"),
                  ("origination date after the end of its acquisition quarter (source_file)",
                   f"x_orig_date > {unit_qend('_unit')}"),
                  ("interest-only first P&I date before first payment date", "x_io_date < x_fpd"),
                  ("origination date after the data's last month", f"x_orig_date > {de}"),
                  ("first payment date after the data's last month", f"x_fpd > {de}")]
        ctx = {"_unit": "unit", "_term": "original_loan_term", "_mat_first": "maturity_first_row"}
    st = date_stats(con, [(c, specs[c].label, q(c), fmt, q(c + "__conf")) for c in dates], f"_lr_{ag}", de, checks, ctx)
    stats = {}
    for k, s, v in st:
        stats.setdefault(k, {})[s] = v
    print_dates(stats, [(c, specs[c]) for c in dates], "loans", de)
    print_checks("Date checks (loans):", list(stats.get("_check", {}).items()))

    for c in [c for c in T.ll if kinds[c] == "text"]:
        r = con.execute(f"""SELECT count(DISTINCT {q(c)}), count(*) FILTER (upper(trim({q(c)})) = 'OTHER'),
                                   count(*) FILTER ({q(c)} IS NULL AND NOT {q(c + '__conf')}),
                                   count(*) FILTER ({q(c + '__conf')}), count(*) FROM _lr_{ag}""").fetchone()
        out()
        out(f"  Free text {c} ({specs[c].label}), loans: distinct values {mc(r[0])}; labelled Other/OTHER "
            f"{count_share(r[1], r[4])}; NULL {mc(r[2])}; CONFLICTING {mc(r[3])}. Names are never shown.")

    # NULL and special codes by origination year
    years = years_of(con, f"w.{ag}_l2")
    out()
    out("NULL and documented special codes of every loan-level field, by origination year (loans):")
    sels, labels = [], []
    for c in T.ll:
        s = specs[c]
        sels.append(f"count(*) FILTER (l.{q(c)} IS NULL AND NOT l.{q(c + '__conf')})")
        labels.append((c, "NULL"))
        sp = special_sql(s, f"l.{q(c)}")
        if sp:
            sels.append(f"count(*) FILTER ({sp})")
            labels.append((c, "special " + "/".join(map(str, tuple(s.na) + tuple(s.special)))))
        sels.append(f"count(*) FILTER (l.{q(c + '__conf')})")
        labels.append((c, "CONFLICTING"))
    per = {}
    for r in con.execute(f"SELECT y.year, {', '.join(sels)} FROM _lr_{ag} l JOIN w.{ag}_l2 y USING (loan_id) "
                         f"GROUP BY y.year").fetchall():
        for (c, what), n in zip(labels, r[1:]):
            per.setdefault((c, what), {})[r[0]] = n
    rows = [([f"{c} ({specs[c].label})", what], sum(d.values()), d) for (c, what), d in per.items()
            if what != "CONFLICTING" or sum(d.values())]
    by_year_table(["field", "what"], rows, years)

    out()
    out("Fields that repeat on every row: loans whose value changes across rows (CONFLICTING) and loans "
        "with NULL on some rows and a value on others:")
    sels = [f"count(*) FILTER ({q(c + suffix)})" for c in T.ll for suffix in ("__conf", "__mix")]
    r = con.execute(f"SELECT {', '.join(sels)} FROM _lr_{ag}").fetchone()
    out_table(["field", "value changes", "NULL on some rows"],
              [[f"{c} ({specs[c].label})", m(r[2 * i]), m(r[2 * i + 1])] for i, c in enumerate(T.ll)])

    if ag == "fre":
        ltv_ok = "TRY_CAST(original_loan_to_value_ltv AS DOUBLE) BETWEEN 0 AND 998"
        cltv_ok = "TRY_CAST(original_combined_loan_to_value_cltv AS DOUBLE) BETWEEN 0 AND 998"
        r = con.execute(f"""SELECT
            count(*) FILTER (TRY_CAST(original_upb AS DOUBLE) % 1000 <> 0),
            count(*) FILTER ({ltv_ok} AND {cltv_ok} AND TRY_CAST(original_combined_loan_to_value_cltv AS DOUBLE)
                             < TRY_CAST(original_loan_to_value_ltv AS DOUBLE)),
            count(*) FILTER (pre_harp_loan_sequence_number IS NOT NULL AND harp_indicator IS DISTINCT FROM 'Y'),
            count(*) FILTER (harp_indicator = 'Y' AND pre_harp_loan_sequence_number IS NULL),
            count(*) FILTER (harp_indicator = 'Y' AND TRY_CAST(original_debt_to_income_dti_ratio AS DOUBLE) IS DISTINCT FROM 999)
            FROM _lr_fre""").fetchone()
        print_checks("Checks of documented rules (loans):", [
            ("Original UPB not a multiple of $1,000", r[0]),
            ("CLTV below LTV, both present and not 999", r[1]),
            ("pre-HARP ID filled but HARP indicator is not Y", r[2]),
            ("HARP indicator Y but pre-HARP ID empty", r[3]),
            ("HARP indicator Y and DTI is not 999", r[4])])


def part1_ids_freddie(con, T, title):
    section(f"PART 1 - {title}: loan IDs (counts only, IDs never shown)")
    pat = "'[FA][0-9]{2}Q[1-4][0-9]{7}'"
    qmis = "regexp_full_match(loan_id, '.[0-9]{2}Q[1-4].*') AND '20' || substr(loan_id, 2, 2) || substr(loan_id, 4, 2) <> unit"
    r = con.execute(f"""SELECT count(*), count(*) FILTER (n_rows > 1), count(*) FILTER (n_units > 1),
                               count(*) FILTER (NOT regexp_full_match(loan_id, {pat})), count(*) FILTER ({qmis})
                        FROM _lr_fre""").fetchone()
    rp = con.execute(f"""SELECT count(DISTINCT loan_id), count(*) FILTER ({qmis}) FROM w.fre_perf""").fetchone()
    multi_perf = con.execute("SELECT count(*) FROM (SELECT loan_id FROM w.fre_perf GROUP BY 1 HAVING count(*) > 1)").fetchone()[0]
    orphan = con.execute("SELECT count(*) FILTER (has_perf AND NOT has_ll), count(*) FILTER (has_ll AND NOT has_perf) "
                         "FROM w.fre_ids").fetchone()
    dup = con.execute("SELECT coalesce(sum(n_dup), 0), count(*) FILTER (n_dup > 0), count(*) FILTER (n_gap_months > 0), "
                      "coalesce(sum(n_gap_months), 0), coalesce(sum(rows_before_fpd), 0) FROM w.fre_perf").fetchone()
    pre = con.execute(f"""SELECT count(*) FILTER (pre_harp_loan_sequence_number IS NOT NULL),
                                 count(*) FILTER (pre_harp_loan_sequence_number IS NOT NULL
                                                  AND NOT regexp_full_match(pre_harp_loan_sequence_number, {pat})),
                                 count(*) FILTER (pre_harp_loan_sequence_number__conf)
                          FROM _lr_fre""").fetchone()
    found = con.execute("SELECT count(*) FROM _lr_fre a SEMI JOIN _lr_fre b ON a.pre_harp_loan_sequence_number = b.loan_id"
                        ).fetchone()[0]
    letters = con.execute("SELECT CASE WHEN regexp_full_match(substr(loan_id, 1, 1), '[A-Z]') THEN substr(loan_id, 1, 1) "
                          "ELSE '(other)' END, count(*) FROM _lr_fre GROUP BY 1 ORDER BY 1").fetchall()
    out_table(["measure", "count"], [
        ["freddie_orig rows", m(misc_sum(con, "freddie_orig", "ll_rows"))],
        ["freddie_orig rows with no loan ID", m(misc_sum(con, "freddie_orig", "ll_rows_no_id"))],
        ["origination loan IDs (distinct)", m(r[0])],
        ["origination loan IDs on more than one row", m(r[1])],
        ["origination loan IDs in more than one source file", m(r[2])],
        ["origination loan IDs not in the documented format PYYQnXXXXXXX", m(r[3])],
        ["origination loan IDs whose YYQn differs from their source_file quarter", m(r[4])],
        ["freddie_perf rows", m(misc_sum(con, "freddie_perf", "rows"))],
        ["freddie_perf rows with no loan ID", m(misc_sum(con, "freddie_perf", "rows_no_id"))],
        ["performance loan IDs (distinct)", m(rp[0])],
        ["performance loan IDs in more than one source file", m(multi_perf)],
        ["performance loan IDs whose YYQn differs from their source_file quarter (loan-file pairs)", m(rp[1])],
        ["performance loan IDs with no origination row", m(orphan[0])],
        ["origination loan IDs with no performance rows", m(orphan[1])],
        ["duplicate (loan, period) monthly rows", m(dup[0])],
        ["loans with duplicate (loan, period) rows", m(dup[1])],
        ["loans with months missing between their first and last row", m(dup[2])],
        ["months missing between first and last rows (sum over loans)", m(dup[3])],
        ["monthly rows dated before the loan's first payment date", m(dup[4])],
        ["pre-HARP loan sequence numbers filled", m(pre[0])],
        ["pre-HARP numbers not in the documented format", m(pre[1])],
        ["pre-HARP numbers CONFLICTING across rows", m(pre[2])],
        ["pre-HARP numbers found as a loan ID in freddie_orig", m(found)],
    ], left={0})
    out()
    out("First character of the origination loan ID (documented: F = FRM, A = ARM):")
    out_table(["letter", "loans", "documentation"],
              [[v, m(n), "documented" if v in ("F", "A") else "UNDOCUMENTED"] for v, n in letters], left={0, 2})


def part1_ids_fannie(con):
    section("PART 1 - Fannie Mae fannie_raw: loan IDs (counts only, IDs never shown)")
    r = con.execute("SELECT count(*), count(*) FILTER (n_units > 1) FROM _lr_fan").fetchone()
    multi_perf = con.execute("SELECT count(*) FROM (SELECT loan_id FROM w.fan_perf GROUP BY 1 HAVING count(*) > 1)").fetchone()[0]
    dup = con.execute("SELECT coalesce(sum(n_dup), 0), count(*) FILTER (n_dup > 0), count(*) FILTER (n_gap_months > 0), "
                      "coalesce(sum(n_gap_months), 0), coalesce(sum(rows_before_fpd), 0) FROM w.fan_perf").fetchone()
    out_table(["measure", "count"], [
        ["fannie_raw rows", m(misc_sum(con, "fannie_raw", "ll_rows"))],
        ["fannie_raw rows with no loan ID", m(misc_sum(con, "fannie_raw", "ll_rows_no_id"))],
        ["loan IDs (distinct)", m(r[0])],
        ["loan IDs in more than one source file (acquisition quarter)", m(r[1])],
        ["loan IDs with monthly facts in more than one source file", m(multi_perf)],
        ["duplicate (loan, period) monthly rows", m(dup[0])],
        ["loans with duplicate (loan, period) rows", m(dup[1])],
        ["loans with months missing between their first and last row", m(dup[2])],
        ["months missing between first and last rows (sum over loans)", m(dup[3])],
        ["monthly rows dated before the loan's first payment date", m(dup[4])],
    ], left={0})


def part1_monthly(con, T, de, title):
    """Monthly fields from the merged per-file counts (counts are LOAN-MONTHS / rows)."""
    tb = T.table
    section(f"PART 1 - {title}: monthly fields (counts are LOAN-MONTHS)")
    out(f"Rows: {mc(misc_sum(con, tb, 'rows'))}; rows with no loan ID: {mc(misc_sum(con, tb, 'rows_no_id'))}")
    specs = {c: T.specs[c] for c in T.monthly}
    out()
    out("Coded fields (every value):")
    vals = {}
    for col, v, n in con.execute("SELECT col, value, sum(n) FROM w.p1_values WHERE tbl = ? GROUP BY ALL", [tb]).fetchall():
        vals.setdefault(col, []).append((v, False, n))
    for c in [c for c in T.monthly if specs[c].kind in ("code", "status")]:
        print_coded(c, specs[c], T.maxlen[c], vals.get(c, []), "rows")
    hist = [c for c in T.monthly if specs[c].kind == "hist"]
    for c in hist:
        h = con.execute("SELECT count(DISTINCT h), coalesce(sum(n), 0) FROM w.p1_hist").fetchone()
        out()
        out(f"  {c} ({specs[c].label}) - 24 two-character statuses per row; full strings are never shown.")
        out(f"    rows NULL: {mc(misc_sum(con, tb, 'hist_null'))}; rows filled: {mc(h[1])}; distinct strings: {mc(h[0])}")
        lens = con.execute("SELECT len, sum(n) FROM w.p1_hist GROUP BY 1 ORDER BY 1").fetchall()
        out("    string length (rows): " + "; ".join(f"{ln}: {mc(n)}" for ln, n in lens))
        toks = con.execute("SELECT tok, sum(n) FROM w.p1_tokens GROUP BY 1 ORDER BY 1").fetchall()
        tspec = Spec("status", "token", ("XX",))
        out_table(["two-character code (all positions pooled)", "count", "documentation"],
                  [[t, m(n), doc_mark(tspec, t)] for t, n in toks], left={0, 2})
    stats = merged_stats(con, tb)
    nums = [c for c in T.monthly if specs[c].kind == "num"]
    if nums:
        print_numeric(title, [(c, specs[c]) for c in nums], stats, specs, "rows")
    dates = [c for c in T.monthly if specs[c].kind == "date"]
    print_dates(stats, [(c, specs[c]) for c in dates], "rows", de)
    print_checks("Date checks (rows):", list(stats.get("_check", {}).items()))
    for c in [c for c in T.monthly if specs[c].kind == "text"]:
        r = con.execute("""SELECT count(DISTINCT h), coalesce(sum(n) FILTER (is_other), 0),
                                  coalesce(sum(n) FILTER (is_null), 0), coalesce(sum(n), 0)
                           FROM w.p1_text WHERE tbl = ? AND col = ?""", [tb, c]).fetchone()
        out()
        out(f"  Free text {c} ({specs[c].label}), rows: distinct values {mc(r[0])}; labelled Other/OTHER "
            f"{count_share(r[1], r[3])}; NULL {mc(r[2])}. Names are never shown.")


# ---------------------------------------------------------------- PART 2

DIM_ORDER = {
    "b": ["<0", "0", "1", "2", "3", "4-6", "7-12", "13-24", ">24"],
    "c": ["<=0", "1", "2", "3", "4-6", "7-12", "13-24", ">24"],
    "d": ["termination code before payment 1", "reached payment 24, no code by then",
          "records stop, no code; payment 24 within the data", "records stop, no code; payment 24 after the data end",
          "still reporting at the data end; payment 24 after the data end"],
    "e": ["0", "1", "2", "3 or more", "REO (RA)", "only XX/blank", "no rows in payments 1-24"],
    "g": ["none", "only kind 1 (jump)", "only kind 2 (status above payment number)", "only kind 3 (drop with no payment)",
          "more than one kind", "no numeric status in window"],
    "la": ["agrees throughout the window", "differs only on modified rows", "differs on an unmodified row",
           "loan_age missing on some rows, no differences", "no rows in payments 1-24"],
    "k": ["continuous", "missing months only", "duplicate period rows only", "missing months and duplicate period rows",
          "no rows in payments 1-24"],
    "l": ["negative (acquired before the origination quarter)", "0", "1", "2", "3-4", "5-8", ">8"],
    "m": ["no termination code", "termination code on the last row", "termination code, then more rows"],
    "n": ["never behind", "behind once (one stretch)", "behind, back to current, behind again (two stretches)",
          "three or more stretches", "REO (RA) in window", "no numeric statuses in window"],
    "o": ["none", "one or more fields CONFLICTING"],
    "q_n": ["0", "1", "2", "3", "4+"],
    "h_fc": ["no foreclosure date", "yes, within payments 1-24", "after the window", "before payment 1",
             "conflicting foreclosure dates", "foreclosure date unparseable"],
    "h_hltv": ["never", "Y from the first row", "Y appears later", "Y, then back to N/blank"],
    "bap": ["payments 1-24 entirely before April 2020", "payments 1-24 span April 2020",
            "payments 1-24 entirely in April 2020 or later", "first payment date unknown"],
}
P2_PROFILE = ("d_fpd_state", "d_rows_in_window", "d_numeric_status", "d_unparseable_status", "d_termination_code",
              "d_first_period")


def part2(con, tabs, ag, de, always_missing):
    name = AGENCIES[ag]
    section(f"PART 2 - {name}: every loan's situation (counts are LOANS)")
    flags_info = build_p2(con, ag, de)
    n = con.execute(f"SELECT count(*) FROM w.{ag}_ids").fetchone()[0]
    n_p2, n_p2_ids = con.execute(f"SELECT count(*), count(DISTINCT loan_id) FROM w.{ag}_p2").fetchone()
    year_tot = dict(con.execute(f"SELECT year, count(*) FROM w.{ag}_l2 GROUP BY 1").fetchall())
    years = years_of(con, f"w.{ag}_l2")
    src = ("freddie_orig + freddie_perf (loan IDs in either table)" if ag == "fre" else "fannie_raw")
    out(f"Loans: {mc(n)} (distinct loan IDs in {src}). Data's last month: {ym(de)}.")
    out("Payment number = months from the loan's first payment date + 1. 'Window' = payments 1-24.")
    out("Loans by origination year:")
    by_year_table(["loans"], [(["all"], n, year_tot)], years)
    if n_p2 != n or n_p2_ids != n:
        msg = f"{name} Part 2: monthly-records table has {n_p2:,} rows / {n_p2_ids:,} loans, universe {n:,}"
        out(f"  FAIL: {msg}")
        FAILS.append(msg)
    l2_profile = ("d_ll_row", "d_unit_format", "d_ll_fpd_state") + (("d_orig_date_state",) if ag == "fan" else ())
    L, P = f"w.{ag}_l2", f"w.{ag}_p2"

    def dim(src_t, col, title, profile):
        attempt(title, print_dim, con, src_t, col, title, n, year_tot, years, DIM_ORDER.get(col, ()), profile)

    a_note = "Freddie: source_file" if ag == "fre" else "Fannie: quarter of origination_date"
    dim(L, "a", f"a. Origination quarter ({a_note})", l2_profile)
    dim(L, "b", "b. Months from the end of the origination quarter to the first payment date", l2_profile)
    dim(P, "c", "c. Payment number of the loan's first monthly record", P2_PROFILE)
    dim(P, "d", "d. How the first 24 payments end", P2_PROFILE)
    attempt("d extra counts", lambda: print_checks("d. Extra counts (loans):", con.execute(
        f"SELECT 'loans with more than one distinct termination code (any time)', count(*) FILTER (zb_multi) FROM {P} "
        f"UNION ALL SELECT 'loans whose first termination code is after payment 24', count(*) FILTER (zb_after24) "
        f"FROM {P}").fetchall()))
    dim(P, "e", "e. Highest delinquency status within payments 1-24", P2_PROFILE)
    attempt("f flags", part2_flags, con, ag, P, n)
    dim(P, "g", "g. Impossible patterns within payments 1-24", P2_PROFILE)
    attempt("g info", lambda: print_checks("g. Related counts, not counted as kind 3 or kind 2 (loans):", con.execute(
        f"SELECT 'status drop with no paid-through advance on a modification/deferral row', count(*) FILTER (g3_mod) FROM {P} "
        f"UNION ALL SELECT 'status drop that cannot be assessed (paid-through date missing)', count(*) FILTER (g3_na) FROM {P} "
        f"UNION ALL SELECT 'status equal to the payment number (possible under one delinquency convention)', "
        f"count(*) FILTER (g2_eq) FROM {P}").fetchall()))
    if ag == "fan":
        dim(P, "h_fc", "h. Fannie foreclosure date vs payments 1-24", P2_PROFILE)
        dim(P, "h_hltv", "h. Fannie high-LTV refinance option flag (field 103)", P2_PROFILE)
    dim(P, "la", ("i. Freddie" if ag == "fre" else "j. Fannie") + " loan_age vs the date-based payment number "
        "(payments 1-24)", P2_PROFILE)
    dim(P, "k", "k. Monthly record continuity within payments 1-24", P2_PROFILE)
    if ag == "fan":
        dim(L, "l", "l. Quarters from origination quarter to acquisition quarter (source_file)", l2_profile)
    dim(P, "m", "m. Rows after the loan's first termination row", P2_PROFILE)
    dim(P, "n", "n. Repeat delinquency within payments 1-24 (stretches of status 1+, XX/blank skipped)", P2_PROFILE)
    if ag == "fan":
        dim(L, "o", "o. Any origination field CONFLICTING across the loan's rows", l2_profile)
    attempt("p patterns", part2_patterns, con, ag, P, n)
    attempt("q missing values", part2_missing, con, tabs, ag, L, n, years, always_missing)
    attempt("combined table", part2_combined, con, ag, P, n)
    if ag == "fan":
        dim(L, "bap", "Fannie assistance-plan coverage: payments 1-24 vs April 2020 (field 102 populated from then)",
            l2_profile)
    return flags_info


def part2_flags(con, ag, P, n):
    out()
    out("--- f. Flags within payments 1-24 (a loan can have several). Counted over loans with rows in the window.")
    names = {"F": "forbearance month (BAP F)", "D": "Freddie disaster-flag month", "R": "repayment plan month (BAP R)",
             "T": "trial period month (BAP T)",
             "DEF": "deferral month (Freddie C/P; Fannie ADR P/C/D)", "MOD": "modification month (Freddie Y/P; Fannie Y)",
             "O": "Fannie other workout month (BAP O)", "XB": "XX or blank status while active",
             "XBT": "XX or blank status after the termination row", "3NF": "status 3+ not in forbearance/disaster"}
    keys = [k for k in names if (k != "D" or ag == "fre") and (k != "O" or ag == "fan")]
    r = con.execute(f"SELECT count(*) FILTER (f_combo NOT IN ('n/a', 'no rows in payments 1-24')), "
                    + ", ".join(f"count(*) FILTER (flag_{k})" for k in keys) + f" FROM {P}").fetchone()
    rows = [[f"{k}: {names[k]}", m(v)] for k, v in zip(keys, r[1:])]
    if ag == "fan":
        b = con.execute(f"SELECT count(*) FILTER (bap_null), count(*) FILTER (bap_null_3nf) FROM {P}").fetchone()
        rows += [["Fannie: any window month with BAP (field 102) blank", m(b[0])],
                 ["Fannie: any 3+ (not F) month with BAP blank", m(b[1])]]
    out(f"  loans with rows in the window: {mc(r[0])}")
    out_table(["flag", "loans"], rows, left={0})
    out()
    out("  Cross-tab of flag combinations with e (highest status); 'n/a' = loan has no usable window:")
    tab = con.execute(f"SELECT e, coalesce(f_combo, 'n/a'), count(*) FROM {P} GROUP BY ALL").fetchall()
    order = sort_key(DIM_ORDER["e"])
    tab.sort(key=lambda t: (order(t[0]), t[1]))
    out_table(["e (highest status)", "flags", "loans"], [[e, f, m(c)] for e, f, c in tab], left={0, 1})
    total = sum(c for _, _, c in tab)
    if total != n:
        FAILS.append(f"{AGENCIES[ag]} f cross-tab adds to {total:,}, loans {n:,}")
        out(f"  FAIL: cross-tab adds to {total:,}, loans {n:,}")
    else:
        out(f"  Check: cross-tab adds up to {mc(n)} loans: OK")


def part2_patterns(con, ag, P, n):
    out()
    out("--- p. Payment-history patterns, payments 1-24 (one symbol per payment):")
    out("  0 1 2 = status; 3 = 3 or more; R = REO (RA); F = forbearance or disaster month (any status);")
    out("  X = XX or blank; T = after the loan's first termination row; . = no row for that payment;")
    out("  ? = unparseable status; _ = payment falls after the data's last month (symbol added).")
    out("  Several rows for one payment: the most severe symbol wins (F > R > 3 > 2 > 1 > 0 > ? > X).")
    with_p = con.execute(f"SELECT count(*), count(DISTINCT pattern) FROM {P} WHERE pattern IS NOT NULL").fetchone()
    without = con.execute(f"SELECT c, count(*) FROM {P} WHERE pattern IS NULL GROUP BY 1 ORDER BY 1").fetchall()
    out(f"  loans with a pattern: {mc(with_p[0])}; distinct patterns: {mc(with_p[1])}")
    for c, k in without:
        out(f"  loans without a pattern - {c}: {mc(k)}")
    if with_p[0] + sum(k for _, k in without) != n:
        FAILS.append(f"{AGENCIES[ag]} patterns: loans with + without a pattern != {n:,}")
        out("  FAIL: loans with and without a pattern do not add up to the agency total")
    small = con.execute(f"SELECT count(*), coalesce(sum(k), 0) FROM (SELECT pattern, count(*) AS k FROM {P} "
                        f"WHERE pattern IS NOT NULL GROUP BY 1) WHERE k BETWEEN 1 AND 10").fetchone()
    out(f"  patterns held by 1-10 loans: {mc(small[0])} patterns, {mc(small[1])} loans in them")
    for title, cond in ((f"Top {TOP_PATTERNS} patterns, all loans with a pattern:", "TRUE"),
                        (f"Top {TOP_PATTERNS} patterns, loans with any status of 1 or more (or RA) in the window:",
                         "w_delinq")):
        rows = con.execute(f"SELECT pattern, count(*) AS k FROM {P} WHERE pattern IS NOT NULL AND {cond} "
                           f"GROUP BY 1 ORDER BY k DESC, pattern LIMIT {TOP_PATTERNS}").fetchall()
        out()
        out(f"  {title}")
        out_table(["rank", "pattern (payments 1..24)", "loans"], [[i, p, m(k)] for i, (p, k) in enumerate(rows, 1)],
                  left={1})


def part2_missing(con, tabs, ag, L, n, years, always_missing):
    T = tabs["freddie_orig"] if ag == "fre" else tabs["fannie_raw"]
    out()
    out("--- q. Missing-value combinations of loan-level fields (NULL or a documented not-available code; "
        "CONFLICTING is not missing)")
    legend = [(s.label, c, "/".join(s.na + tuple(map(str, s.special if s.kind == "num" else ())))
               or "-") for c, s in ll_specs(T) if s.kind != "id"]
    out("  Field labels: " + "; ".join(f"{lab} = {c}" + (f" (also {codes})" if codes != "-" else "")
                                       for lab, c, codes in legend))
    if always_missing:
        out(f"  Missing for every loan, left out of the combinations: {', '.join(always_missing)}")
    year_tot = dict(con.execute(f"SELECT year, count(*) FROM {L} GROUP BY 1").fetchall())
    attempt("q counts", print_dim, con, L, "q_n", "q. Number of missing loan-level fields per loan", n, year_tot, years,
            DIM_ORDER["q_n"], ("d_ll_row", "d_unit_format"))
    top = con.execute(f"SELECT coalesce(nullif(q_combo, ''), '(none missing)'), count(*) AS k FROM {L} "
                      f"WHERE q_combo IS NOT NULL GROUP BY 1 ORDER BY k DESC, 1 LIMIT {TOP_COMBOS}").fetchall()
    out()
    out(f"  Top {TOP_COMBOS} combinations of missing fields:")
    out_table(["rank", "missing fields", "loans"], [[i, c, m(k)] for i, (c, k) in enumerate(top, 1)], left={1})
    top20 = [c for c, _ in top[:TOP_COMBOS_BY_YEAR]]
    per = {}
    for c, y, k in con.execute(f"SELECT coalesce(nullif(q_combo, ''), '(none missing)') AS c, year, count(*) FROM {L} "
                               f"WHERE q_combo IS NOT NULL GROUP BY ALL").fetchall():
        if c in top20:
            per.setdefault(c, {})[y] = k
    out()
    out(f"  The {TOP_COMBOS_BY_YEAR} most common combinations by origination year:")
    by_year_table(["missing fields"], [([c], sum(per[c].values()), per[c]) for c in top20], years)


def part2_combined(con, ag, P, n):
    out()
    out("--- Combined: d x e x (any impossible pattern, g) x (any 3+ month outside forbearance/disaster, f)")
    rows = con.execute(f"SELECT d, e, coalesce(g_any, 'n/a'), coalesce(f3nf, 'n/a'), count(*) FROM {P} "
                       f"GROUP BY ALL").fetchall()
    kd, ke = sort_key(DIM_ORDER["d"]), sort_key(DIM_ORDER["e"])
    rows.sort(key=lambda r: (kd(r[0]), ke(r[1]), r[2], r[3]))
    out_table(["d (how payments 1-24 end)", "e (highest status)", "impossible", "3+ not F/D", "loans"],
              [[d, e, g, f, m(k)] for d, e, g, f, k in rows], left={0, 1, 2, 3})
    total = sum(r[4] for r in rows)
    if total != n:
        FAILS.append(f"{AGENCIES[ag]} combined table adds to {total:,}, loans {n:,}")
        out(f"  FAIL: combined table adds to {total:,}, loans {n:,}")
    else:
        out(f"  Check: combined table adds up to {mc(n)} loans: OK")


# ---------------------------------------------------------------- main

def header(units, parts, agencies, args, des):
    out(f"inventory.py  {datetime.now():%Y-%m-%d %H:%M}  database {load_raw.DB_PATH.name} (read-only)")
    out("Counts only: no loan IDs, no row-level values, no seller or servicer names. Counts from 1 to 10 show as "
        "'1-10'; totals are exact. Facts only: no filters, no labels, no matching of codes between agencies.")
    out(f"Parts: {', '.join(sorted(parts))}; agencies: {', '.join(AGENCIES[a] for a in agencies)}; "
        f"memory limit {args.memory_limit}, threads {args.threads}.")
    for ag in agencies:
        out(f"{AGENCIES[ag]}: data's last month {ym(des[ag])}")
    per = {}
    for table, unit, rows in units:
        p = per.setdefault(table, [0, 0])
        p[0] += 1
        p[1] += rows or 0
    out("Loaded files (load_log): " + "; ".join(f"{t} {k} files, {mc(r)} rows" for t, (k, r) in sorted(per.items())))


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--only", choices=["part1", "part2"], help="run one part (default: both)")
    p.add_argument("--agency", choices=["freddie", "fannie"], help="run one agency (default: both)")
    p.add_argument("--memory-limit", default=DEFAULT_MEMORY_LIMIT, help=f"DuckDB memory limit (default {DEFAULT_MEMORY_LIMIT})")
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS, help=f"DuckDB threads (default {DEFAULT_THREADS})")
    p.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB,
                   help=f"refuse to start or continue below this much free disk (default {DEFAULT_MIN_FREE_GB})")
    p.add_argument("--clean", action="store_true", help="delete the working files (inventory_work.duckdb*) and stop")
    args = p.parse_args(argv)
    if not re.fullmatch(r"\d+(\.\d+)?\s*[KMGT]i?B", args.memory_limit, re.IGNORECASE):
        p.error("--memory-limit must look like 5GB or 4096MB")
    return args


def clean():
    for path in (WORK_DB, WORK_DB.with_name(WORK_DB.name + ".wal")):
        if path.exists():
            path.unlink()
            print(f"deleted {path}")
    if SPILL_DIR.exists():
        shutil.rmtree(SPILL_DIR)
        print(f"deleted {SPILL_DIR}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.clean:
        return clean()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    LOG_DIR.mkdir(exist_ok=True)
    check_raw.report.clear()
    FAILS.clear()
    ERRORS.clear()
    parts = {"part1", "part2"} if args.only is None else {args.only}
    agencies = [a for a in AGENCIES if args.agency is None or AGENCIES[a].lower().startswith(args.agency)]
    started = time.time()
    progress("=" * 78)
    progress(f"inventory.py started. Database {load_raw.DB_PATH} (read-only); working database {WORK_DB.name}.")
    if not load_raw.DB_PATH.exists():
        progress(f"No database at {load_raw.DB_PATH}; run load_raw.py first.")
        return 1
    con, code = None, 0
    try:
        check_disk(args.min_free_gb, "at start")
        con = open_db(args)
        tabs = build_tabs(read_layouts())
        create_work_tables(con, tabs)
        units = load_units(con)
        check_work_matches(con, units)
        des = {ag: data_end(con, ag) for ag in agencies}
        con.close()
        con = None
        process_files(args, tabs, units, parts, agencies, des)

        progress("Building the report from the working database ...")
        con = open_db(args)
        header(units, parts, agencies, args, des)
        for ag in agencies:
            T = tabs["freddie_orig"] if ag == "fre" else tabs["fannie_raw"]
            always = attempt(f"{AGENCIES[ag]}: per-loan tables", prepare_agency, con, tabs, ag, des[ag])
            if always is None:
                continue
            if "part1" in parts:
                if ag == "fre":
                    attempt("Freddie IDs", part1_ids_freddie, con, T, "Freddie Mac freddie_orig + freddie_perf")
                    attempt("Freddie origination fields", part1_loan_level, con, T, des[ag], "Freddie Mac freddie_orig")
                    attempt("Freddie monthly fields", part1_monthly, con, tabs["freddie_perf"], des[ag],
                            "Freddie Mac freddie_perf")
                else:
                    attempt("Fannie IDs", part1_ids_fannie, con)
                    attempt("Fannie loan-level fields", part1_loan_level, con, T, des[ag], "Fannie Mae fannie_raw")
                    attempt("Fannie monthly fields", part1_monthly, con, T, des[ag], "Fannie Mae fannie_raw")
            if "part2" in parts:
                attempt(f"{AGENCIES[ag]} Part 2", part2, con, tabs, ag, des[ag], always)
            progress(f"{AGENCIES[ag]}: report sections done ({hms(time.time() - started)} so far)")
        section(f"SUMMARY: {len(FAILS)} total(s) not adding up, {len(ERRORS)} section/file error(s)")
        for f in FAILS:
            out(f"  FAIL: {f}")
        for e in ERRORS:
            out(f"  ERROR: {e}")
        if not FAILS and not ERRORS:
            out("  Every dimension adds up; no section failed.")
        out(f"(run took {hms(time.time() - started)})")
        code = 1 if FAILS or ERRORS else 0
    except StopRun as stop:
        code = stop.code
    except KeyboardInterrupt:
        progress("Interrupted. The file piece in progress was rolled back; finished pieces are kept. Rerun to resume.")
        code = 130
    except Exception as exc:
        report_error(exc, ERROR_DETAIL_PATH, progress)
        code = 1
    finally:
        if con is not None:
            con.close()
        if SPILL_DIR.exists() and not any(SPILL_DIR.iterdir()):
            SPILL_DIR.rmdir()
        if check_raw.report:
            save_report()
            progress(f"Report saved to {REPORT_PATH}")
        progress(f"Finished with exit code {code}, elapsed {hms(time.time() - started)}.")
    return code


if __name__ == "__main__":
    sys.exit(main())
