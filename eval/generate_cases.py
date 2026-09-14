"""
eval/generate_cases.py
Builds the golden-dataset evaluation cases.

Ground truth comes from CONSTRUCTION, not from re-running the implementation:
each case builds rows from explicit parameters ("this employee is short exactly
$40 this period") and the expected totals are the sum of those injected amounts.
That keeps the expectations independent of the code under test — a bug in
compliance.py cannot silently propagate into its own expected value.

Expectations are split by how they can be asserted:
  exact      — deterministic pure-Python results (SG/PAYG/casual/award maths)
  bounded    — a value that depends on an LLM choice (e.g. award LEVEL) but is
               provably inside a range whichever choice is made
  labelled   — an answer key for an LLM decision, scored as accuracy

Run:  python eval/generate_cases.py
"""

import csv
import json
from pathlib import Path

CASES_DIR = Path(__file__).parent / "cases"

SG_RATE = 0.12
PERIODS = ["2026-01-16", "2026-01-30", "2026-02-13"]

_REGISTRY = []


def case(case_id, description, tags):
    def wrap(fn):
        _REGISTRY.append({"id": case_id, "description": description,
                          "tags": tags, "build": fn})
        return fn
    return wrap


def row(emp_id, *, name=None, state="NSW", dept="Operations",
        emp_type="Full-time", pay_date=PERIODS[0], gross=2000.00,
        super_paid=None, expected_sg=None, payg_withheld=400.00,
        correct_payg=400.00, **extra):
    """One canonical, fully-compliant pay run unless overridden.

    `gross` may be a deliberately messy string (for cleaning cases); derived
    defaults are only computed when it is genuinely numeric.
    """
    if expected_sg is None:
        expected_sg = round(gross * SG_RATE, 2) if isinstance(gross, (int, float)) else 0.0
    super_paid = expected_sg if super_paid is None else super_paid
    base = {
        "employee_id": emp_id,
        "name": name or f"Employee {emp_id}",
        "state": state,
        "department": dept,
        "employment_type": emp_type,
        "pay_date": pay_date,
        "gross_wage": gross,
        "super_paid": super_paid,
        "expected_sg": expected_sg,
        "payg_withheld": payg_withheld,
        "correct_payg": correct_payg,
    }
    base.update(extra)
    return base


def over_periods(emp_id, n=3, **kw):
    return [row(emp_id, pay_date=PERIODS[i], **kw) for i in range(n)]


def zero_findings():
    return {"sg_total": 0.0, "payg_total": 0.0, "casual_total": 0.0,
            "sg_employees": 0, "payg_employees": 0, "casual_employees": 0}


# ── 1. Baseline ───────────────────────────────────────────────────────────────

@case("clean_baseline", "Fully compliant canonical data — every check must report zero.",
      ["deterministic", "baseline"])
def _clean_baseline():
    rows = (over_periods("EMP001") + over_periods("EMP002", gross=3000.00)
            + over_periods("EMP003", gross=2500.00, emp_type="Part-time"))
    return rows, {"exact": {**zero_findings(), "total_exposure": 0.0}}


# ── 2. Single-issue isolation ─────────────────────────────────────────────────

@case("sg_underpayment_only", "SG shortfall only; PAYG and casual checks must stay clean.",
      ["deterministic"])
def _sg_only():
    # EMP001: expected 240.00, paid 200.00 -> 40.00 short x3 = 120.00
    # EMP002: expected 360.00, paid 300.00 -> 60.00 short x3 = 180.00
    rows = (over_periods("EMP001", super_paid=200.00)
            + over_periods("EMP002", gross=3000.00, super_paid=300.00)
            + over_periods("EMP003"))
    return rows, {"exact": {"sg_total": 300.00, "sg_employees": 2, "sg_runs": 6,
                            "payg_total": 0.0, "casual_total": 0.0,
                            "total_exposure": 300.00}}


@case("payg_deviation_only", "PAYG deviation beyond the 15% threshold; SG stays compliant.",
      ["deterministic"])
def _payg_only():
    # EMP001: correct 400, withheld 300 -> 25.0% dev, 100.00 exposure x3 = 300.00
    # EMP002: correct 500, withheld 600 -> 20.0% dev, 100.00 exposure x3 = 300.00
    rows = (over_periods("EMP001", payg_withheld=300.00, correct_payg=400.00)
            + over_periods("EMP002", payg_withheld=600.00, correct_payg=500.00)
            + over_periods("EMP003"))
    return rows, {"exact": {"payg_total": 600.00, "payg_employees": 2, "payg_runs": 6,
                            "sg_total": 0.0, "casual_total": 0.0,
                            "total_exposure": 600.00}}


@case("payg_just_under_threshold",
      "PAYG deviation of exactly 15% must NOT flag (threshold is strictly greater-than).",
      ["deterministic", "boundary"])
def _payg_boundary():
    # correct 400, withheld 340 -> exactly 15.0% deviation -> not flagged
    rows = over_periods("EMP001", payg_withheld=340.00, correct_payg=400.00)
    return rows, {"exact": {"payg_total": 0.0, "payg_employees": 0,
                            "sg_total": 0.0, "total_exposure": 0.0}}


@case("casual_super_missing",
      "Casuals paid zero super. These dollars appear in BOTH the SG and casual checks — "
      "the roll-up must count them ONCE.",
      ["deterministic", "double_count"])
def _casual_missing():
    # EMP001 Casual: expected 180.00/period, paid 0 -> 540.00 across 3 periods.
    # Same dollars are flagged by check_sg_underpayment (no employment_type filter).
    # Correct total exposure is 540.00, NOT 1080.00.
    rows = (over_periods("EMP001", emp_type="Casual", gross=1500.00, super_paid=0.0)
            + over_periods("EMP002"))
    return rows, {"exact": {"casual_total": 540.00, "casual_employees": 1,
                            "sg_total": 540.00, "sg_employees": 1,
                            "payg_total": 0.0,
                            "total_exposure": 540.00}}


@case("combined_all_issues",
      "SG, PAYG and casual breaches together — checks overlap, roll-up must not double count.",
      ["deterministic", "double_count"])
def _combined():
    # SG-only breach:  EMP001 short 40.00 x3 = 120.00
    # PAYG breach:     EMP002 exposure 100.00 x3 = 300.00
    # Casual zero super: EMP003 expected 180.00 x3 = 540.00 (also counted by SG check)
    # SG total = 120.00 + 540.00 = 660.00 ; casual total = 540.00 (subset of SG)
    # Correct exposure = 660.00 (SG incl. casual) + 300.00 (PAYG) = 960.00
    rows = (over_periods("EMP001", super_paid=200.00)
            + over_periods("EMP002", payg_withheld=300.00, correct_payg=400.00)
            + over_periods("EMP003", emp_type="Casual", gross=1500.00, super_paid=0.0))
    return rows, {"exact": {"sg_total": 660.00, "payg_total": 300.00,
                            "casual_total": 540.00, "total_exposure": 960.00}}


# ── 3. Award engine (never exercised in this repo's history) ──────────────────

AWARD_COLS = {"job_title": "", "hourly_rate": 0.0, "hours_worked": 76.0}


@case("award_underpayment_severe",
      "Professional-award title paid $15/hr — below EVERY level of MA000065, so it must "
      "flag regardless of which level the classifier picks.",
      ["award", "llm_dependent"])
def _award_severe():
    # MA000065 levels: 31.55 / 34.42 / 38.72 / 42.51. At 76h the underpayment is
    # (min_rate - 15.00) * 76 -> between 1257.80 (L1) and 2090.76 (L4) per run.
    rows = over_periods("EMP001", gross=1140.00, job_title="Software Engineer",
                        hourly_rate=15.00, hours_worked=76.0,
                        expected_sg=136.80, super_paid=136.80,
                        payg_withheld=0.0, correct_payg=0.0)
    return rows, {
        "exact": {"award_employees": 1, "award_runs": 3},
        "bounded": {"award_total": [3773.40, 6272.28]},
        "labelled": {"classification": {"Software Engineer": "MA000065"}},
    }


@case("award_casual_loading",
      "Casual retail worker at $20/hr — the 25% casual loading must be applied to the "
      "minimum before comparison.",
      ["award", "llm_dependent"])
def _award_casual():
    # MA000004 base levels 26.55-32.43; +25% casual loading -> 33.19-40.54 minimum.
    # $20.00/hr is below every loaded level.
    rows = over_periods("EMP001", emp_type="Casual", gross=1520.00,
                        job_title="Retail Assistant", hourly_rate=20.00,
                        hours_worked=76.0, expected_sg=182.40, super_paid=182.40,
                        payg_withheld=0.0, correct_payg=0.0)
    return rows, {
        "exact": {"award_employees": 1, "award_runs": 3},
        "bounded": {"award_total": [3002.52, 4683.12]},
        "labelled": {"classification": {"Retail Assistant": "MA000004"}},
    }


@case("award_compliant_rate",
      "Paid well above every award minimum — award check must report zero underpayment.",
      ["award", "llm_dependent"])
def _award_ok():
    rows = over_periods("EMP001", gross=4560.00, job_title="Software Engineer",
                        hourly_rate=60.00, hours_worked=76.0,
                        expected_sg=547.20, super_paid=547.20,
                        payg_withheld=0.0, correct_payg=0.0)
    return rows, {"exact": {"award_total": 0.0, "award_employees": 0}}


@case("award_no_rate_columns",
      "No hourly_rate/hours_worked — engine falls back to gross/76. Documents the "
      "fallback path rather than silently trusting it.",
      ["award", "llm_dependent"])
def _award_fallback():
    # gross 1140 / 76 = $15.00/hr effective -> below all MA000065 levels.
    rows = over_periods("EMP001", gross=1140.00, job_title="Software Engineer",
                        expected_sg=136.80, super_paid=136.80,
                        payg_withheld=0.0, correct_payg=0.0)
    return rows, {"exact": {"award_employees": 1},
                  "bounded": {"award_total": [3773.40, 6272.28]}}


# ── 4. Schema mapping (LLM-dependent) ─────────────────────────────────────────

CRITICAL_COLS = ["employee_id", "gross_wage", "super_paid", "employment_type"]


@case("schema_xero", "Xero-style column names must map to the canonical schema.",
      ["schema", "llm_dependent"])
def _schema_xero():
    rows = [{"EmployeeID": "EMP001", "EmployeeName": "Alice Smith", "State": "NSW",
             "Department": "Finance", "EmploymentBasis": "Full-time",
             "PaymentDate": p, "GrossEarnings": 2000.00,
             "SuperannuationExpense": 200.00, "ExpectedSuper": 240.00,
             "PAYGWithheld": 400.00, "CorrectPAYG": 400.00} for p in PERIODS]
    return rows, {
        "exact": {"sg_total": 120.00, "sg_employees": 1},
        "labelled": {"mapping_required": {
            "EmployeeID": "employee_id", "GrossEarnings": "gross_wage",
            "SuperannuationExpense": "super_paid", "EmploymentBasis": "employment_type"}},
    }


@case("schema_myob", "MYOB-style column names with spaces.", ["schema", "llm_dependent"])
def _schema_myob():
    rows = [{"Card ID": "EMP001", "Employee Name": "Bob Jones", "State": "VIC",
             "Department": "Sales", "Employment Status": "Full-time",
             "Pay Date": p, "Gross Wages": 2000.00, "Super Expense": 200.00,
             "Expected Super": 240.00, "Tax Withheld": 400.00,
             "Correct Tax": 400.00} for p in PERIODS]
    return rows, {
        "exact": {"sg_total": 120.00, "sg_employees": 1},
        "labelled": {"mapping_required": {
            "Card ID": "employee_id", "Gross Wages": "gross_wage",
            "Super Expense": "super_paid", "Employment Status": "employment_type"}},
    }


@case("schema_split_name",
      "First_Name + Last_Name must merge into a single canonical name column.",
      ["schema", "llm_dependent"])
def _schema_split():
    rows = [{"Employee_ID": "EMP001", "First_Name": "Carol", "Last_Name": "White",
             "State": "QLD", "Department": "IT", "Employment_Type": "Full-time",
             "Pay_Date": p, "Gross_Wage": 2000.00, "Super_Paid": 200.00,
             "Expected_SG": 240.00, "PAYG_Withheld": 400.00,
             "Correct_PAYG": 400.00} for p in PERIODS]
    return rows, {
        "exact": {"sg_total": 120.00, "sg_employees": 1},
        "labelled": {"name_merged": True},
    }


@case("schema_pascal_case",
      "PascalCase export with hours/rate and no derived columns (shape of payroll_data_v2).",
      ["schema", "llm_dependent"])
def _schema_pascal():
    rows = [{"Employee_ID": "EMP001", "First_Name": "Dan", "Last_Name": "Brown",
             "Job_Title": "Data Analyst", "Pay_Date": p, "Base_Hours": 76.0,
             "Hourly_Rate": 45.00, "Total_Gross": 3420.00,
             "PAYG_Withholding": 800.00, "Super_Guarantee": 410.40}
            for p in PERIODS]
    return rows, {"labelled": {"mapping_required": {
        "Employee_ID": "employee_id", "Total_Gross": "gross_wage",
        "Super_Guarantee": "super_paid"}}}


@case("schema_unmappable_extras",
      "Extra columns with no canonical equivalent must be reported UNMAPPED, never guessed.",
      ["schema", "llm_dependent"])
def _schema_extras():
    rows = [dict(row("EMP001", pay_date=p), cost_centre_code="CC-99",
                 union_membership_flag="Y", locker_number=str(100 + i))
            for i, p in enumerate(PERIODS)]
    return rows, {"exact": {"sg_total": 0.0},
                  "labelled": {"expect_unmapped": ["cost_centre_code",
                                                   "union_membership_flag",
                                                   "locker_number"]}}


# ── 5. Data cleaning edge cases ───────────────────────────────────────────────

@case("messy_currency_formats",
      "Currency symbols, thousands separators and AUD codes must coerce to numbers.",
      ["cleaning", "deterministic"])
def _messy_currency():
    rows = [dict(row("EMP001", pay_date=p), gross_wage="$2,000.00",
                 super_paid="AUD 200.00", expected_sg="$240.00") for p in PERIODS]
    return rows, {"exact": {"sg_total": 120.00, "sg_employees": 1,
                            "quarantined": 0}}


@case("messy_european_decimals",
      "European decimal format (1.234,56) must parse as 1234.56.",
      ["cleaning", "deterministic"])
def _messy_euro():
    rows = [dict(row("EMP001", pay_date=p), gross_wage="2.000,00",
                 super_paid="200,00", expected_sg="240,00") for p in PERIODS]
    return rows, {"exact": {"sg_total": 120.00, "quarantined": 0}}


@case("messy_accounting_negatives",
      "Accounting negatives (123.45) and trailing-minus must parse as negative, and "
      "negative gross must be quarantined rather than analysed.",
      ["cleaning", "deterministic"])
def _messy_negatives():
    rows = [row("EMP001", pay_date=PERIODS[0], gross_wage="(500.00)"),
            row("EMP002", pay_date=PERIODS[0], gross_wage="500.00-"),
            row("EMP003", pay_date=PERIODS[0])]
    return rows, {"exact": {"quarantined": 2, "rows_analysable": 1,
                            "sg_total": 0.0}}


@case("messy_null_tokens",
      "Null tokens (N/A, TBC, --) must become NaN and quarantine the row, not be read as 0.",
      ["cleaning", "deterministic"])
def _messy_nulls():
    rows = [row("EMP001", pay_date=PERIODS[0], gross_wage="N/A"),
            row("EMP002", pay_date=PERIODS[0], gross_wage="TBC"),
            row("EMP003", pay_date=PERIODS[0], gross_wage="--"),
            row("EMP004", pay_date=PERIODS[0])]
    return rows, {"exact": {"quarantined": 3, "rows_analysable": 1}}


@case("excel_serial_dates",
      "5-digit Excel serial dates must be recovered as real dates.",
      ["cleaning", "deterministic"])
def _excel_serials():
    # 46038 -> 2026-01-16, 46052 -> 2026-01-30
    rows = [row("EMP001", pay_date="46038", super_paid=200.00),
            row("EMP001", pay_date="46052", super_paid=200.00)]
    return rows, {"exact": {"sg_total": 80.00, "quarantined": 0,
                            "dates_recovered": 2}}


@case("exact_duplicate_rows",
      "Byte-identical duplicate rows must be removed so exposure is not double counted.",
      ["cleaning", "deterministic"])
def _exact_dupes():
    r = row("EMP001", super_paid=200.00)
    rows = [r, dict(r), dict(r), row("EMP002")]
    return rows, {"exact": {"sg_total": 40.00, "sg_runs": 1,
                            "duplicates_removed": 2}}


@case("logical_duplicate_rows",
      "Same employee_id + pay_date twice must be FLAGGED but not silently dropped "
      "(could be a legitimate multi-line pay).",
      ["cleaning", "deterministic"])
def _logical_dupes():
    rows = [row("EMP001", super_paid=200.00, gross=2000.00),
            row("EMP001", super_paid=100.00, gross=1000.00, expected_sg=120.00)]
    return rows, {"exact": {"sg_total": 60.00, "sg_runs": 2,
                            "quarantined": 0},
                  "labelled": {"logical_duplicates_flagged": True}}


@case("quarantine_invalid_rows",
      "Missing employee_id, non-positive gross, and super>gross must all be quarantined "
      "and excluded from every figure.",
      ["cleaning", "deterministic"])
def _quarantine():
    rows = [row("", pay_date=PERIODS[0]),
            row("EMP002", pay_date=PERIODS[0], gross=0.0, expected_sg=0.0),
            row("EMP003", pay_date=PERIODS[0], gross=1000.00, super_paid=5000.00,
                expected_sg=120.00),
            row("EMP004", pay_date=PERIODS[0], super_paid=200.00)]
    return rows, {"exact": {"quarantined": 3, "rows_analysable": 1,
                            "sg_total": 40.00}}


@case("employment_type_variants",
      "FT / P/T / cas / perm variants must normalise — the casual-super check is an "
      "exact string match and silently misses breaches otherwise.",
      ["cleaning", "deterministic"])
def _emp_variants():
    # All three casual variants must be recognised as Casual and flagged for zero super.
    rows = [row("EMP001", emp_type="cas", gross=1500.00, super_paid=0.0),
            row("EMP002", emp_type="CASUAL", gross=1500.00, super_paid=0.0),
            row("EMP003", emp_type="Temp", gross=1500.00, super_paid=0.0),
            row("EMP004", emp_type="FT"),
            row("EMP005", emp_type="P/T")]
    return rows, {"exact": {"casual_employees": 3, "casual_total": 540.00,
                            "sg_total": 540.00, "total_exposure": 540.00}}


# ── 6. Classification ambiguity ───────────────────────────────────────────────

@case("ambiguous_job_titles",
      "Titles with no clear award must classify as UNKNOWN rather than being guessed "
      "into a rate that produces a fabricated underpayment.",
      ["award", "llm_dependent"])
def _ambiguous():
    titles = ["Chief Vibes Officer", "Growth Ninja", "Level 7 Widget Wrangler"]
    rows = [row(f"EMP00{i+1}", job_title=t, hourly_rate=15.00, hours_worked=76.0,
                gross=1140.00, expected_sg=136.80, super_paid=136.80,
                payg_withheld=0.0, correct_payg=0.0)
            for i, t in enumerate(titles)]
    return rows, {"labelled": {"expect_unknown_or_low_confidence": titles}}


# ── 7. Degenerate shapes ──────────────────────────────────────────────────────

@case("single_employee_single_period", "Smallest non-empty dataset.", ["deterministic", "boundary"])
def _single():
    return [row("EMP001", super_paid=200.00)], {
        "exact": {"sg_total": 40.00, "sg_employees": 1, "sg_runs": 1,
                  "total_exposure": 40.00}}


@case("all_rows_quarantined",
      "Every row unanalysable — pipeline must report zero findings without crashing, "
      "not emit a confident $0 clean bill of health.",
      ["deterministic", "boundary"])
def _all_bad():
    rows = [row("EMP001", gross=0.0, expected_sg=0.0),
            row("EMP002", gross_wage="N/A")]
    return rows, {"exact": {"quarantined": 2, "rows_analysable": 0,
                            "sg_total": 0.0, "total_exposure": 0.0},
                  "labelled": {"quality_score": "LOW"}}


# ── 8. Security ───────────────────────────────────────────────────────────────

@case("xss_in_text_fields",
      "Markup in CSV text fields must be escaped in the generated dashboard HTML, not "
      "rendered as live markup.",
      ["security"])
def _xss():
    rows = over_periods("EMP001", dept="<script>alert('xss')</script>",
                        name="<img src=x onerror=alert(1)>", super_paid=200.00)
    return rows, {"exact": {"sg_total": 120.00},
                  "labelled": {"dashboard_must_escape": [
                      "<script>alert('xss')</script>", "<img src=x onerror=alert(1)>"]}}


# ── 9. Reporting accuracy ─────────────────────────────────────────────────────

@case("period_outside_hardcoded_range",
      "Pay dates far outside Jan-Mar 2026 — the report's 'Period Analysed' line must "
      "reflect the real data, not the hardcoded prompt string.",
      ["reporting", "llm_dependent"])
def _period():
    rows = [row("EMP001", pay_date=d, super_paid=200.00)
            for d in ["2024-07-05", "2024-07-19", "2024-08-02"]]
    return rows, {"exact": {"sg_total": 120.00},
                  "labelled": {"report_period_must_mention": ["2024"],
                               "report_period_must_not_mention": ["January–March 2026"]}}


# ── Writer ────────────────────────────────────────────────────────────────────

def write_cases():
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    for spec in _REGISTRY:
        rows, expected = spec["build"]()
        case_dir = CASES_DIR / spec["id"]
        case_dir.mkdir(exist_ok=True)

        fieldnames = list(dict.fromkeys(k for r in rows for k in r))
        with open(case_dir / "input.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        meta = {
            "id": spec["id"],
            "description": spec["description"],
            "tags": spec["tags"],
            "row_count": len(rows),
            "expected": expected,
        }
        with open(case_dir / "expected.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        written.append((spec["id"], len(rows), spec["tags"]))

    return written


if __name__ == "__main__":
    written = write_cases()
    print(f"Wrote {len(written)} cases to {CASES_DIR}\n")
    for cid, n, tags in written:
        print(f"  {cid:<34} {n:>3} rows   [{', '.join(tags)}]")
