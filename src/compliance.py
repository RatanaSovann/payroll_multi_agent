"""
compliance.py
Reusable Australian payroll compliance check functions.
These are the tools the AI agent calls to analyse a payroll dataset.

Regulatory basis:
- Super Guarantee rate: 12% of OTE (FY2025-26)
- Max Contribution Base: $62,500/quarter
- PAYG: ATO tax bracket withholding rates
- Casual SG eligibility: all casuals post July 2022
"""

import pandas as pd
import numpy as np

from src.audit import AUDIT


# ── Tool 1: Schema Inspector ──────────────────────────────────────────────────

def inspect_data(df: pd.DataFrame, n_rows: int = 5) -> dict:
    """
    Returns schema, column types, sample rows, and basic stats.
    Called first so the agent understands what it's working with
    before running any compliance checks.
    """
    return {
        "row_count":        len(df),
        "column_count":     len(df.columns),
        "columns":          df.dtypes.astype(str).to_dict(),
        "sample_rows":      df.head(n_rows).to_dict(orient="records"),
        "null_counts":      df.isnull().sum().to_dict(),
        "employment_types": df["employment_type"].value_counts().to_dict()
                            if "employment_type" in df.columns else "column not found",
        "pay_periods":      df["period_number"].nunique()
                            if "period_number" in df.columns else "column not found",
        "states":           df["state"].value_counts().to_dict()
                            if "state" in df.columns else "column not found",
    }


# ── Tool 2: SG Underpayment ───────────────────────────────────────────────────

def check_sg_underpayment(df: pd.DataFrame, tolerance: float = 0.001) -> dict:
    """
    Flags pay runs where super paid is below the legislated 12% SG rate.
    Returns a summary dict + serialisable detail records.

    Args:
        df:        Payroll DataFrame
        tolerance: Rounding tolerance to exclude floating point noise (default 0.1%)
    """
    flagged = df[df["super_paid"] < df["expected_sg"] * (1 - tolerance)].copy()

    flagged["sg_shortfall"]      = (flagged["expected_sg"] - flagged["super_paid"]).round(2)
    flagged["effective_sg_rate"] = (flagged["super_paid"] / flagged["gross_wage"] * 100).round(2)

    # Top 5 worst offenders for the agent to reason about
    top5 = (
        flagged.groupby(["employee_id", "name", "department", "employment_type"])
        ["sg_shortfall"].sum()
        .reset_index()
        .sort_values("sg_shortfall", ascending=False)
        .head(5)
        .to_dict(orient="records")
    )

    by_department = (
        flagged.groupby("department")["sg_shortfall"]
        .sum().round(2)
        .sort_values(ascending=False)
        .to_dict()
    )

    by_state = (
        flagged.groupby("state")["sg_shortfall"]
        .sum().round(2)
        .sort_values(ascending=False)
        .to_dict()
    )

    AUDIT.log_rule(
        "compliance_analyst", "sg_rounding_tolerance", tolerance,
        "Excludes sub-0.1% differences caused by floating point rounding in payroll systems",
    )
    AUDIT.log_calculation(
        agent="compliance_analyst",
        name="sg_underpayment_total",
        formula="SUM(expected_sg - super_paid) WHERE super_paid < expected_sg * (1 - tolerance); expected_sg = gross_wage * 0.12",
        inputs={
            "rows_checked": int(len(df)),
            "rows_flagged": int(len(flagged)),
            "sg_rate": 0.12,
            "tolerance": tolerance,
        },
        result=round(float(flagged["sg_shortfall"].sum()), 2),
        regulatory_basis="Superannuation Guarantee (Administration) Act 1992 — SG rate 12% of OTE, FY2025-26",
        evidence=flagged[["employee_id", "pay_date", "gross_wage", "expected_sg",
                          "super_paid", "sg_shortfall"]].head(10).to_dict(orient="records"),
    )

    return {
        "issue_type":            "SG Underpayment",
        "employees_affected":    flagged["employee_id"].nunique(),
        "pay_runs_affected":     len(flagged),
        "total_shortfall_aud":   round(flagged["sg_shortfall"].sum(), 2),
        "avg_shortfall_per_run": round(flagged["sg_shortfall"].mean(), 2),
        "by_department":         by_department,
        "by_state":              by_state,
        "top_5_employees":       top5,
        "threshold_applied":     f"super_paid < expected_sg * {1 - tolerance}",
    }


# ── Tool 3: PAYG Inconsistency ────────────────────────────────────────────────

def check_payg_inconsistency(df: pd.DataFrame, threshold_pct: float = 15.0) -> dict:
    """
    Flags pay runs where PAYG withheld deviates beyond threshold from
    the expected ATO bracket amount.

    Args:
        df:            Payroll DataFrame
        threshold_pct: Deviation % to flag (default 15%, see README for rationale)
    """
    valid = df[df["correct_payg"] > 0].copy()

    valid["deviation_pct"] = (
        abs(valid["payg_withheld"] - valid["correct_payg"])
        / valid["correct_payg"] * 100
    ).round(1)

    valid["payg_status"] = np.where(
        valid["payg_withheld"] > valid["correct_payg"] * (1 + threshold_pct / 100),
        "Over-withheld",
        "Under-withheld"
    )

    flagged = valid[valid["deviation_pct"] > threshold_pct].copy()

    status_split = flagged["payg_status"].value_counts().to_dict()

    by_department = (
        flagged.groupby("department")["deviation_pct"]
        .mean().round(1)
        .sort_values(ascending=False)
        .to_dict()
    )

    total_exposure = abs(
        flagged["payg_withheld"] - flagged["correct_payg"]
    ).sum().round(2)

    AUDIT.log_rule(
        "compliance_analyst", "payg_deviation_threshold", threshold_pct,
        "Selected after sensitivity analysis at 5/10/15/20% — excludes rounding noise "
        "while capturing material variances. NOTE: correct_payg may be enriched via "
        "simplified flat brackets (no LITO/Medicare) — see enrichment audit entries.",
    )
    AUDIT.log_calculation(
        agent="compliance_analyst",
        name="payg_inconsistency_exposure",
        formula="SUM(ABS(payg_withheld - correct_payg)) WHERE ABS(payg_withheld - correct_payg) / correct_payg > threshold_pct/100",
        inputs={
            "rows_checked": int(len(valid)),
            "rows_flagged": int(len(flagged)),
            "threshold_pct": threshold_pct,
        },
        result=float(total_exposure),
        regulatory_basis="ITAA 1997 / ATO PAYG withholding schedules (simplified bracket approximation)",
        evidence=flagged[["employee_id", "pay_date", "gross_wage", "payg_withheld",
                          "correct_payg", "deviation_pct", "payg_status"]].head(10).to_dict(orient="records"),
    )

    return {
        "issue_type":           "PAYG Inconsistency",
        "employees_affected":   flagged["employee_id"].nunique(),
        "pay_runs_affected":    len(flagged),
        "total_exposure_aud":   total_exposure,
        "over_withheld_runs":   status_split.get("Over-withheld", 0),
        "under_withheld_runs":  status_split.get("Under-withheld", 0),
        "avg_deviation_pct":    round(flagged["deviation_pct"].mean(), 1),
        "max_deviation_pct":    round(flagged["deviation_pct"].max(), 1),
        "by_department":        by_department,
        "threshold_applied":    f"{threshold_pct}% deviation from ATO bracket amount",
    }


# ── Tool 4: Missing Super — Casuals ──────────────────────────────────────────

def check_missing_casual_super(df: pd.DataFrame) -> dict:
    """
    Identifies casual employees receiving zero super despite being SG-eligible.
    All casuals are eligible regardless of earnings (post July 2022).
    Summarised at employee level across all pay periods.
    """
    casuals = df[
        (df["employment_type"] == "Casual") &
        (df["gross_wage"] > 0)
    ].copy()

    casuals["is_breach"] = casuals["super_paid"] == 0

    summary = casuals.groupby(
        ["employee_id", "name", "state", "department"]
    ).agg(
        total_pay_runs    = ("pay_date",    "count"),
        breached_pay_runs = ("is_breach",   "sum"),
        total_super_owed  = ("expected_sg", "sum"),
        total_super_paid  = ("super_paid",  "sum"),
    ).reset_index()

    summary["total_shortfall"] = (
        summary["total_super_owed"] - summary["total_super_paid"]
    ).round(2)

    affected = summary[summary["breached_pay_runs"] > 0].copy()

    by_state = (
        affected.groupby("state")["total_shortfall"]
        .sum().round(2)
        .sort_values(ascending=False)
        .to_dict()
    )

    top5 = (
        affected[["employee_id", "name", "state", "department",
                  "breached_pay_runs", "total_shortfall"]]
        .sort_values("total_shortfall", ascending=False)
        .head(5)
        .to_dict(orient="records")
    )

    AUDIT.log_calculation(
        agent="compliance_analyst",
        name="missing_casual_super_shortfall",
        formula="SUM(expected_sg - super_paid) per casual employee WHERE super_paid = 0 AND gross_wage > 0, aggregated across all pay periods",
        inputs={
            "casual_rows_checked": int(len(casuals)),
            "employees_flagged": int(len(affected)),
        },
        result=round(float(affected["total_shortfall"].sum()), 2) if len(affected) else 0.0,
        regulatory_basis="SGAA 1992 — $450/month threshold removed 1 July 2022; all casuals SG-eligible",
        evidence=affected.head(10).to_dict(orient="records") if len(affected) else [],
    )

    return {
        "issue_type":          "Missing Super — Casuals",
        "employees_affected":  len(affected),
        "pay_runs_affected":   int(affected["breached_pay_runs"].sum()),
        "total_shortfall_aud": round(affected["total_shortfall"].sum(), 2),
        "by_state":            by_state,
        "top_5_employees":     top5,
        "note":                "All casuals are SG-eligible post July 2022 — no earnings threshold applies",
    }


# ── Tool 5: Executive Summary ─────────────────────────────────────────────────

def generate_executive_summary(sg: dict, payg: dict, casuals: dict) -> dict:
    """
    Rolls up all three compliance checks into a single findings summary.
    This is what gets presented to the client first.
    """
    total_exposure = round(
        sg["total_shortfall_aud"] +
        payg["total_exposure_aud"] +
        casuals["total_shortfall_aud"],
        2
    )

    return {
        "total_exposure_aud": total_exposure,
        "findings": [
            {
                "issue_type":          sg["issue_type"],
                "employees_affected":  sg["employees_affected"],
                "pay_runs_affected":   sg["pay_runs_affected"],
                "total_exposure_aud":  sg["total_shortfall_aud"],
            },
            {
                "issue_type":          payg["issue_type"],
                "employees_affected":  payg["employees_affected"],
                "pay_runs_affected":   payg["pay_runs_affected"],
                "total_exposure_aud":  payg["total_exposure_aud"],
            },
            {
                "issue_type":          casuals["issue_type"],
                "employees_affected":  casuals["employees_affected"],
                "pay_runs_affected":   casuals["pay_runs_affected"],
                "total_exposure_aud":  casuals["total_shortfall_aud"],
            },
        ],
    }
