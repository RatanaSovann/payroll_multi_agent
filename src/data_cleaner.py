"""
data_cleaner.py
Deterministic, robust data cleaning for messy real-world payroll CSVs.
ZERO API tokens. Every transformation logged to the audit trail.

Runs inside the Data Validator, AFTER schema mapping/enrichment and BEFORE
any compliance math.

Compliance-grade philosophy:
  - Coerce what is SAFELY coercible.
  - Normalise categoricals so exact-match checks don't silently miss breaches.
  - QUARANTINE what is not safely interpretable — never guess a value that
    could hide or fabricate an underpayment.
  - Record EVERYTHING, with per-row evidence where it matters.
"""

import re
import unicodedata
import pandas as pd
import numpy as np

from src.audit import AUDIT

NUMERIC_COLS = ["gross_wage", "super_paid", "expected_sg",
                "payg_withheld", "correct_payg", "hours_worked", "hourly_rate"]
DATE_COLS = ["pay_date", "pay_period_start", "pay_period_end"]

EMPLOYMENT_NORMALISE = {
    "ft": "Full-time", "f/t": "Full-time", "fulltime": "Full-time",
    "full time": "Full-time", "full-time": "Full-time", "permanent": "Full-time",
    "perm": "Full-time", "permanent full-time": "Full-time", "ftm": "Full-time",
    "pt": "Part-time", "p/t": "Part-time", "parttime": "Part-time",
    "part time": "Part-time", "part-time": "Part-time", "permanent part-time": "Part-time",
    "casual": "Casual", "cas": "Casual", "c": "Casual", "temp": "Casual",
    "temporary": "Casual", "seasonal": "Casual",
    "contractor": "Contract", "contract": "Contract", "contract/agency": "Contract",
}

STATE_NORMALISE = {
    "new south wales": "NSW", "nsw": "NSW",
    "victoria": "VIC", "vic": "VIC",
    "queensland": "QLD", "qld": "QLD",
    "western australia": "WA", "wa": "WA",
    "south australia": "SA", "sa": "SA",
    "tasmania": "TAS", "tas": "TAS",
    "australian capital territory": "ACT", "act": "ACT",
    "northern territory": "NT", "nt": "NT",
}

NULL_TOKENS = {"", "n/a", "na", "null", "none", "-", "--", "---", "tbc", "tbd",
               "#n/a", "#value!", "#ref!", "nan", "nil", ".", "?", "xxx", "pending"}


# ── Numeric coercion ──────────────────────────────────────────────────────────

def _as_text(val):
    """Render a raw cell as the text a human would have typed."""
    if pd.isna(val):
        return val
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    return str(val).strip()


def _parse_number(val):
    """Parse one messy numeric cell. Returns (value_or_nan, status)."""
    if pd.isna(val):
        return np.nan, "already_null"
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val), "ok"

    text = str(val).strip()
    low = text.lower()
    if low in NULL_TOKENS:
        return np.nan, "null_token"

    # Accounting negatives: (123.45) or trailing minus 123.45-
    neg = (text.startswith("(") and text.endswith(")")) or text.endswith("-")
    text = text.strip("()").rstrip("-").strip()

    # Percentage — flag, don't silently treat as absolute
    is_pct = text.endswith("%")
    text = text.rstrip("%").strip()

    # Currency codes first, anchored to either end - "AUD 200.00", "200.00 AUD"
    # and "AUD200.00" all occur. Anchoring keeps this conservative: a code buried
    # mid-string ("200 AUD approx") stays unparseable rather than being guessed.
    text = re.sub(r"(?i)^(aud|usd|nzd)\s*|\s*(aud|usd|nzd)$", "", text.strip())

    # Strip currency symbols / whitespace
    text = re.sub(r"[$£€¥\s]", "", text)

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")  # 1.234,56 -> 1234.56
        else:
            text = text.replace(",", "")                    # 1,234.56 -> 1234.56
    elif "," in text:
        # A lone comma is ambiguous. Thousands groups are always exactly three
        # digits, so "1,234" is 1234 but "200,00" is a European decimal - reading
        # it as 20000 inflates a wage a hundredfold and silently flips the
        # super-exceeds-gross quarantine rule.
        head, _, tail = text.rpartition(",")
        if "," in head or re.fullmatch(r"\d{3}", tail):
            text = text.replace(",", "")                    # thousands separator
        else:
            text = text.replace(",", ".")                   # decimal comma

    try:
        num = float(text)
    except ValueError:
        return np.nan, "unparseable"

    if neg:
        num = -num
    if is_pct:
        return num, "percentage_flagged"
    return num, "ok"


def _clean_numeric_column(s: pd.Series):
    parsed = s.apply(_parse_number)
    values = parsed.apply(lambda t: t[0])
    statuses = parsed.apply(lambda t: t[1])
    counts = statuses.value_counts().to_dict()
    return values, counts


# ── Main entry ────────────────────────────────────────────────────────────────

def clean_dataframe(df: pd.DataFrame):
    work = df.copy()
    report = {"issues": [], "rows_before": len(work), "quarantined_rows": 0,
              "quarantine_detail": []}

    # 1. Unicode normalise + strip whitespace on all string columns
    #    (handles non-breaking spaces, smart quotes, zero-width chars from Excel)
    str_cols = work.select_dtypes(include="object").columns
    for c in str_cols:
        work[c] = work[c].apply(
            lambda x: unicodedata.normalize("NFKC", x).strip()
            if isinstance(x, str) else x
        )

    # 2. Numeric coercion
    numeric_summary = {}
    for col in NUMERIC_COLS:
        if col not in work.columns:
            continue
        # Gate on "already numeric", not "dtype is object": pandas 3 infers text
        # columns as StringDtype, so an `== object` test sends every messy cell
        # straight to to_numeric() and quietly nulls it — the cleaner below would
        # never run at all.
        if pd.api.types.is_numeric_dtype(work[col]):
            work[col] = pd.to_numeric(work[col], errors="coerce")
        else:
            values, counts = _clean_numeric_column(work[col])
            work[col] = values
            interesting = {k: v for k, v in counts.items() if k not in ("ok", "already_null")}
            if interesting:
                numeric_summary[col] = interesting

    if numeric_summary:
        AUDIT.log_transform("data_cleaner", "numeric_coercion", {
            "columns": numeric_summary,
            "method": "NFKC normalise; strip $/£/€/¥ and AUD/USD codes; thousands "
                      "separators; European decimals; accounting & trailing negatives; "
                      "null tokens → NaN; percentages flagged",
            "note": "'unparseable' or 'null_token' counts become NaN and may trigger "
                    "row quarantine below if in a required column",
        })
        report["issues"].append(f"Coerced messy numerics in {len(numeric_summary)} column(s)")

    # 3. Normalise employment_type (protects the casual-super exact-match check)
    if "employment_type" in work.columns:
        raw = work["employment_type"].astype(str).str.lower().str.strip()
        work["employment_type"] = raw.map(EMPLOYMENT_NORMALISE).fillna(
            work["employment_type"])
        n = int((raw != work["employment_type"].astype(str).str.lower()).sum())
        unmatched = sorted(set(
            raw[~raw.isin(EMPLOYMENT_NORMALISE)].dropna()
        ) - {"full-time", "part-time", "casual", "contract"})
        if n or unmatched:
            AUDIT.log_transform("data_cleaner", "normalise_employment_type", {
                "rows_normalised": n,
                "unmatched_values": unmatched[:20],
                "note": "Exact-match casual-super check depends on this. Unmatched "
                        "values left as-is and will NOT match 'Casual' — review if a "
                        "casual variant was missed.",
            })
            report["issues"].append(f"Normalised {n} employment_type value(s)")
            if unmatched:
                report["issues"].append(
                    f"⚠ {len(unmatched)} unrecognised employment_type value(s): {unmatched[:5]}")

    # 4. Normalise state
    if "state" in work.columns:
        work["state"] = work["state"].apply(
            lambda x: STATE_NORMALISE.get(str(x).lower().strip(),
                                          str(x).upper().strip())
            if pd.notna(x) else x)

    # 5. Dates — try dayfirst then fallback, capture Excel serials
    for col in DATE_COLS:
        if col not in work.columns:
            continue
        original_null = int(work[col].isna().sum())

        # Parse from TEXT, never from raw numerics. pandas reads an integer
        # column as epoch nanoseconds, so a CSV whose Excel serials arrived as
        # ints (46038) silently yields 1970-01-01 instead of NaT — which the
        # old "recover whatever failed to parse" guard could never see.
        if pd.api.types.is_datetime64_any_dtype(work[col]):
            as_text = work[col]
        else:
            as_text = work[col].apply(_as_text)

        # Excel serial dates (numbers like 46037) are not dates in any format —
        # identify them from the source value, not from a parse failure.
        serial_mask = as_text.apply(
            lambda x: bool(re.fullmatch(r"\d{5}", x)) if isinstance(x, str) else False)

        parsed = pd.to_datetime(as_text.where(~serial_mask), errors="coerce",
                                dayfirst=True)

        if serial_mask.any():
            serials = pd.to_numeric(as_text[serial_mask], errors="coerce")
            parsed.loc[serial_mask] = pd.to_datetime(serials, unit="D", origin="1899-12-30")
            AUDIT.log_transform("data_cleaner", "excel_serial_dates", {
                "column": col, "rows_recovered": int(serial_mask.sum()),
                "note": "5-digit Excel serial dates converted to real dates",
            })

        n_unparsed = int(parsed.isna().sum() - original_null)
        work[col] = parsed
        if n_unparsed > 0:
            AUDIT.log_transform("data_cleaner", "date_parse_failures", {
                "column": col, "unparseable_rows": n_unparsed,
                "note": "Set to NaT; rows retained (date isn't required for core checks)",
            })

    # 6. Duplicates — two levels: exact rows, and logical (same emp+pay_date)
    exact = int(work.duplicated().sum())
    if exact:
        work = work.drop_duplicates().reset_index(drop=True)
        AUDIT.log_transform("data_cleaner", "remove_exact_duplicates", {
            "rows_removed": exact,
            "note": "Identical rows removed to prevent double-counting exposure",
        })
        report["issues"].append(f"Removed {exact} exact duplicate row(s)")

    if {"employee_id", "pay_date"}.issubset(work.columns):
        logical = work.duplicated(subset=["employee_id", "pay_date"], keep=False)
        n_logical = int(logical.sum())
        if n_logical:
            # Don't auto-drop — same employee/date could be legitimate (2 pay types).
            # Flag it loudly instead.
            dupe_ids = (work.loc[logical, ["employee_id", "pay_date"]]
                        .astype(str).agg(" | ".join, axis=1).unique().tolist())
            AUDIT.log_transform("data_cleaner", "logical_duplicate_warning", {
                "rows_involved": n_logical,
                "duplicate_keys_sample": dupe_ids[:10],
                "note": "Same employee_id + pay_date appears more than once. NOT removed "
                        "automatically (could be legitimate multi-line pays), but flagged "
                        "for review — may indicate double-counting.",
            })
            report["issues"].append(
                f"⚠ {n_logical} rows share employee_id+pay_date (flagged, not removed)")

    # 7. Quarantine structurally invalid rows (with per-row evidence)
    quarantine_mask = pd.Series(False, index=work.index)
    reasons = {}

    if "employee_id" in work.columns:
        m = work["employee_id"].isna() | (work["employee_id"].astype(str).str.strip() == "")
        quarantine_mask |= m
        if m.sum():
            reasons["missing_employee_id"] = int(m.sum())

    if "gross_wage" in work.columns:
        m = work["gross_wage"].isna() | (work["gross_wage"] <= 0)
        quarantine_mask |= m
        if m.sum():
            reasons["missing_or_nonpositive_gross"] = int(m.sum())

    # Impossible values: super or PAYG wildly exceeding gross (data error)
    if {"super_paid", "gross_wage"}.issubset(work.columns):
        m = work["super_paid"] > work["gross_wage"]  # super > gross is impossible
        quarantine_mask |= m
        if m.sum():
            reasons["super_exceeds_gross"] = int(m.sum())

    quarantined = work[quarantine_mask].copy()
    clean = work[~quarantine_mask].reset_index(drop=True)

    if len(quarantined):
        evidence = quarantined.head(10).astype(str).to_dict(orient="records")
        AUDIT.log_transform("data_cleaner", "quarantine_invalid_rows", {
            "rows_quarantined": int(len(quarantined)),
            "reasons": reasons,
            "evidence_sample": evidence,
            "note": "EXCLUDED from all compliance figures and reported separately. "
                    "Never silently included or value-guessed — a fabricated value could "
                    "hide or invent an underpayment. Require manual review.",
        })
        report["issues"].append(f"Quarantined {len(quarantined)} unanalysable row(s): {reasons}")
        report["quarantined_rows"] = int(len(quarantined))
        report["quarantine_detail"] = reasons

    # 8. Quality score
    report["rows_after"] = len(clean)
    total = report["rows_before"] or 1
    clean_pct = round(len(clean) / total * 100, 1)
    report["clean_rate_pct"] = clean_pct
    report["quality_score"] = ("HIGH" if clean_pct >= 98 else
                               "MEDIUM" if clean_pct >= 90 else "LOW")

    AUDIT.log_transform("data_cleaner", "cleaning_summary", {
        "rows_in": report["rows_before"], "rows_analysable": report["rows_after"],
        "rows_quarantined": report["quarantined_rows"],
        "clean_rate_pct": clean_pct, "quality_score": report["quality_score"],
    })

    return clean, report
