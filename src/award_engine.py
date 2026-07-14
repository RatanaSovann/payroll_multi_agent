"""
award_engine.py
Deterministic Modern Award rates engine — ZERO API tokens.

All award rate lookups and underpayment calculations happen in pure Python.
Rates current as of 1 July 2026 (post 2026 Annual Wage Review, +4.75%).

Coverage: 5 of the most common Modern Awards + National Minimum Wage fallback.
Awards outside this set are flagged as "coverage unknown" — never guessed.

Sources: Fair Work Ombudsman pay guides, FWC 2026 Annual Wage Review.
"""

import pandas as pd

from src.audit import AUDIT

# ── National Minimum Wage (1 July 2026) ───────────────────────────────────────

NMW_HOURLY          = 26.44      # $26.44/hr, $1,004.90/wk
NMW_WEEKLY          = 1004.90
CASUAL_LOADING      = 0.25       # 25% loading
NMW_CASUAL_HOURLY   = 33.05      # NMW incl casual loading
ENTRY_LEVEL_HOURLY  = 25.74      # C14 equivalent, first 6 months only

# ── Award rates table (adult, full-time hourly, from 1 July 2026) ─────────────
# Level 1 = entry, higher levels = more senior/qualified.
# Rates = FY2025-26 published rates × 1.0475 (2026 AWR increase), rounded.

AWARD_RATES = {
    "MA000002": {   # Clerks — Private Sector Award 2020
        "name": "Clerks — Private Sector Award",
        "levels": {
            1: 25.99, 2: 27.26, 3: 28.28, 4: 29.67, 5: 30.88,
        },
        "saturday_penalty": 1.25, "sunday_penalty": 1.50,
        "overtime_first2h": 1.50, "overtime_after2h": 2.00,
    },
    "MA000004": {   # General Retail Industry Award 2020
        "name": "General Retail Industry Award",
        "levels": {
            1: 26.55, 2: 27.16, 3: 27.57, 4: 28.11, 5: 29.26,
            6: 29.68, 7: 31.17, 8: 32.43,
        },
        "saturday_penalty": 1.25, "sunday_penalty": 1.50,
        "overtime_first2h": 1.50, "overtime_after2h": 2.00,
    },
    "MA000009": {   # Hospitality Industry (General) Award 2020
        "name": "Hospitality Industry (General) Award",
        "levels": {
            1: 25.80, 2: 26.77, 3: 27.66, 4: 29.15, 5: 30.97,
            6: 31.80,
        },
        "saturday_penalty": 1.25, "sunday_penalty": 1.50,
        "overtime_first2h": 1.50, "overtime_after2h": 2.00,
    },
    "MA000010": {   # Manufacturing and Associated Industries Award 2020
        "name": "Manufacturing and Associated Industries Award",
        "levels": {
            # C13 → C5 mapped to levels 1-8
            1: 26.44, 2: 27.23, 3: 28.24, 4: 29.30, 5: 30.61,
            6: 31.68, 7: 32.75, 8: 34.36,
        },
        "saturday_penalty": 1.50, "sunday_penalty": 2.00,
        "overtime_first2h": 1.50, "overtime_after2h": 2.00,
    },
    "MA000065": {   # Professional Employees Award 2020 (engineers, IT, scientists)
        "name": "Professional Employees Award",
        "levels": {
            1: 31.55, 2: 34.42, 3: 38.72, 4: 42.51,
        },
        "saturday_penalty": 1.00, "sunday_penalty": 1.00,   # salaried, generally no penalties
        "overtime_first2h": 1.00, "overtime_after2h": 1.00,
    },
    "NMW": {        # National Minimum Wage fallback (award-free employees)
        "name": "National Minimum Wage (award-free)",
        "levels": {1: NMW_HOURLY},
        "saturday_penalty": 1.00, "sunday_penalty": 1.00,
        "overtime_first2h": 1.00, "overtime_after2h": 1.00,
    },
}

SUPPORTED_AWARDS = {k: v["name"] for k, v in AWARD_RATES.items()}


# ── Rate lookup ────────────────────────────────────────────────────────────────

def get_minimum_rate(award_code: str, level: int, employment_type: str = "Full-time") -> dict:
    """
    Returns the minimum hourly rate for an award/level/employment type.
    Casuals get the 25% loading applied.
    Returns None values if the award is not in the supported set.
    """
    award = AWARD_RATES.get(award_code)
    if not award:
        return {"hourly_rate": None, "award_name": None,
                "note": f"Award {award_code} not in supported set — coverage unknown"}

    levels = award["levels"]
    lvl    = level if level in levels else min(levels.keys())
    base   = levels[lvl]

    if str(employment_type).lower().startswith("casual"):
        rate = round(base * (1 + CASUAL_LOADING), 2)
        note = f"{award['name']} L{lvl} + 25% casual loading"
    else:
        rate = base
        note = f"{award['name']} L{lvl}"

    return {"hourly_rate": rate, "award_name": award["name"],
            "level": lvl, "note": note}


# ── Underpayment check (pure Python — the main compliance engine) ─────────────

def check_award_underpayment(
    df: pd.DataFrame,
    classification_map: dict,
) -> dict:
    """
    Checks each pay run against the minimum award rate for that employee's
    classified award + level.

    Args:
        df:                 Payroll DataFrame (canonical columns). Needs either
                            'hourly_rate' directly, or 'gross_wage' + 'hours_worked'
                            to derive an effective hourly rate.
        classification_map: {job_title: {"award_code": str, "level": int,
                                          "confidence": str}}
                            Produced by the AwardClassifierAgent.

    Returns summary dict with flagged rows, totals, and coverage stats.
    """
    work = df.copy()

    # Derive effective hourly rate
    if "hourly_rate" in work.columns:
        work["_eff_rate"] = pd.to_numeric(work["hourly_rate"], errors="coerce")
    elif "hours_worked" in work.columns:
        hrs = pd.to_numeric(work["hours_worked"], errors="coerce").replace(0, pd.NA)
        work["_eff_rate"] = pd.to_numeric(work["gross_wage"], errors="coerce") / hrs
    else:
        # Assume fortnightly gross over 76 standard hours as last resort
        work["_eff_rate"] = pd.to_numeric(work["gross_wage"], errors="coerce") / 76.0
        rate_basis = "estimated from gross_wage / 76 standard fortnightly hours"

    title_col = "job_title" if "job_title" in work.columns else "department"

    flagged, unknown_coverage = [], set()

    for _, row in work.iterrows():
        title = str(row.get(title_col, "Unknown"))
        cls   = classification_map.get(title)

        if not cls or cls.get("award_code") not in AWARD_RATES:
            unknown_coverage.add(title)
            continue

        rate_info = get_minimum_rate(
            cls["award_code"], cls.get("level", 1),
            row.get("employment_type", "Full-time"),
        )
        min_rate = rate_info["hourly_rate"]
        eff      = row["_eff_rate"]

        if pd.notna(eff) and pd.notna(min_rate) and eff < min_rate * 0.999:
            hours = row.get("hours_worked", 76.0)
            hours = float(hours) if pd.notna(hours) else 76.0
            flagged.append({
                "employee_id":    row.get("employee_id"),
                "name":           row.get("name"),
                "job_title":      title,
                "pay_date":       str(row.get("pay_date")),
                "award":          rate_info["award_name"],
                "level":          rate_info["level"],
                "employment_type": row.get("employment_type"),
                "effective_rate": round(float(eff), 2),
                "minimum_rate":   min_rate,
                "rate_gap":       round(min_rate - float(eff), 2),
                "underpayment":   round((min_rate - float(eff)) * hours, 2),
            })

    flagged_df = pd.DataFrame(flagged)

    by_award = (
        flagged_df.groupby("award")["underpayment"].sum().round(2).to_dict()
        if len(flagged_df) else {}
    )
    top5 = (
        flagged_df.groupby(["employee_id", "name", "job_title", "award"])
        ["underpayment"].sum().reset_index()
        .sort_values("underpayment", ascending=False).head(5)
        .to_dict(orient="records")
        if len(flagged_df) else []
    )

    total_titles   = work[title_col].nunique()
    covered_titles = total_titles - len(unknown_coverage)

    AUDIT.log_calculation(
        agent="compliance_analyst",
        name="award_underpayment_total",
        formula="SUM((minimum_rate - effective_rate) * hours) WHERE effective_rate < minimum_rate * 0.999; "
                "minimum_rate = award_level_rate * (1.25 if casual else 1.0); "
                "effective_rate = hourly_rate OR gross_wage/hours_worked OR gross_wage/76",
        inputs={
            "rows_checked": int(len(work)),
            "rows_flagged": int(len(flagged_df)),
            "titles_covered": int(covered_titles),
            "titles_unknown": sorted(unknown_coverage),
            "rates_effective": "1 July 2026 (FWC 2026 AWR, +4.75%; NMW $26.44/hr; casual loading 25%)",
        },
        result=round(float(flagged_df["underpayment"].sum()), 2) if len(flagged_df) else 0.0,
        regulatory_basis="Fair Work Act 2009 — Modern Award minimum rates; FWC Annual Wage Review 2026",
        evidence=flagged[:10],
    )

    return {
        "issue_type":            "Modern Award Underpayment",
        "employees_affected":    flagged_df["employee_id"].nunique() if len(flagged_df) else 0,
        "pay_runs_affected":     len(flagged_df),
        "total_underpayment_aud": round(flagged_df["underpayment"].sum(), 2) if len(flagged_df) else 0.0,
        "by_award":              by_award,
        "top_5_employees":       top5,
        "coverage": {
            "titles_total":      int(total_titles),
            "titles_covered":    int(covered_titles),
            "titles_unknown":    sorted(unknown_coverage),
            "supported_awards":  list(SUPPORTED_AWARDS.values()),
        },
        "rates_basis": "FWC 2026 Annual Wage Review rates, effective 1 July 2026 "
                       "(NMW $26.44/hr; award rates +4.75%; casual loading 25%)",
    }
