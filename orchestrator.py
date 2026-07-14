"""
orchestrator.py
Coordinates the multi-agent payroll compliance pipeline.

Pipeline:
  CSV → DataValidatorAgent → ComplianceAnalystAgent → RiskAssessorAgent
      → ReportWriterAgent → DashboardGeneratorAgent
      → reports/findings_TIMESTAMP.md
      → reports/dashboard_TIMESTAMP.html

Usage:
  python orchestrator.py --file data/raw/payroll_data.csv
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import anthropic
from dotenv import load_dotenv

from src.agents import (
    DataValidatorAgent,
    ComplianceAnalystAgent,
    RiskAssessorAgent,
    ReportWriterAgent,
    DashboardGeneratorAgent,
    AwardClassifierAgent,
)
from src.agents.base_agent import reset_token_ledger, get_token_summary
from src.audit import AUDIT

load_dotenv()


def print_header():
    print("\n" + "═" * 62)
    print("  🤖 PAYROLL COMPLIANCE — MULTI-AGENT PIPELINE")
    print("═" * 62)


def print_step(n: int, total: int, name: str):
    print(f"\n{'─' * 62}")
    print(f"  AGENT {n}/{total}: {name}")
    print(f"{'─' * 62}")


def save_report(report: str, timestamp: str) -> str:
    Path("reports").mkdir(exist_ok=True)
    path = f"reports/findings_{timestamp}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    return path


def run_pipeline(csv_path: str) -> tuple[str, str]:

    print_header()
    print(f"\n📂 Input: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"   {len(df):,} rows · {df['employee_id'].nunique()} employees · "
          f"{len(df.columns)} columns\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    client    = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    reset_token_ledger()
    AUDIT.start_run(csv_path, df)

    # ── Agent 1: Data Validator ────────────────────────────────────────────────
    print_step(1, 6, "Data Validator")
    validator            = DataValidatorAgent(client)
    validation, mapped_df = validator.run(df)   # mapped_df has canonical column names

    status = "✅ Passed" if validation["validation_passed"] else "⚠️  Issues found"
    print(f"\n  {status}")
    print(f"   → Mapping confidence: {validation['mapping_confidence']}")

    if validation["unmapped_columns"]:
        print(f"   ⚠️  Unmapped columns: {validation['unmapped_columns']}")

    if validation["issues"]:
        for issue in validation["issues"]:
            print(f"   ⚠️  {issue}")
    else:
        print(f"   → {validation['stats']['row_count']:,} rows · "
              f"{validation['stats']['employee_count']} employees · "
              f"no critical issues")

    # Show mapping summary if any columns were renamed
    mapping = validation["column_mapping"]
    renamed = {k: v for k, v in mapping.items() if k != v and v != "UNMAPPED"}
    if renamed:
        print(f"   → Renamed {len(renamed)} column(s): "
              + ", ".join(f"{k} → {v}" for k, v in list(renamed.items())[:4])
              + ("..." if len(renamed) > 4 else ""))

    # ── Agent 1.5: Award Classifier (Haiku — unique titles only) ──────────────
    print_step(2, 6, "Award Classifier")
    classifier         = AwardClassifierAgent(client)
    classification_map = classifier.run(mapped_df)

    # ── Agent 2: Compliance Analyst ────────────────────────────────────────────
    print_step(3, 6, "Compliance Analyst")
    analyst    = ComplianceAnalystAgent(client)
    compliance = analyst.run(mapped_df, classification_map=classification_map)

    if "summary" in compliance:
        total = compliance["summary"].get("total_exposure_aud", 0)
        awards = compliance.get("awards", {})
        print(f"\n  ✅ Complete · Total exposure: ${total:,.2f}")
        if awards:
            cov = awards.get("coverage", {})
            print(f"   → Award check: ${awards.get('total_underpayment_aud',0):,.2f} underpayment · "
                  f"{cov.get('titles_covered',0)}/{cov.get('titles_total',0)} titles covered")
        for finding in compliance["summary"].get("findings", []):
            print(f"   → {finding['issue_type']}: "
                  f"${finding['total_exposure_aud']:,.2f} "
                  f"({finding['employees_affected']} employees)")
    else:
        print("  ✅ Complete")

    # ── Agent 3: Risk Assessor ─────────────────────────────────────────────────
    print_step(4, 6, "Risk Assessor")
    assessor = RiskAssessorAgent(client)
    risk     = assessor.run(compliance)

    if "risk_scores" in risk:
        print(f"\n  ✅ Complete · Overall risk: {risk.get('overall_risk_rating', 'N/A')}")
        for issue, detail in risk["risk_scores"].items():
            score = detail.get("score", "N/A")
            icon  = "🔴" if score == "HIGH" else "🟡" if score == "MEDIUM" else "🟢"
            print(f"   {icon} {issue.replace('_', ' ').title()}: {score}")
        if "estimated_sgc_penalty" in risk:
            print(f"   💰 Estimated SGC penalty: ${risk['estimated_sgc_penalty']:,.2f}")
    else:
        print("  ✅ Complete")

    # ── Agent 4: Report Writer ─────────────────────────────────────────────────
    print_step(5, 6, "Report Writer")
    writer = ReportWriterAgent(client)
    report = writer.run(
        validation=validation,
        compliance=compliance,
        risk=risk,
    )
    report_path = save_report(report, timestamp)
    print(f"\n  ✅ Report saved → {report_path}")

    # ── Agent 5: Dashboard Generator ──────────────────────────────────────────
    print_step(6, 6, "Dashboard Generator")
    dashboard_path = f"reports/dashboard_{timestamp}.html"
    generator      = DashboardGeneratorAgent()
    generator.run(
        compliance=compliance,
        risk=risk,
        df=mapped_df,                           # uses canonical column names
        output_path=dashboard_path,
    )
    print(f"\n  ✅ Dashboard saved → {dashboard_path}")

    # ── Audit trail ────────────────────────────────────────────────────────────
    audit_path = AUDIT.save()
    print(f"\n{'─' * 62}")
    print("  🧾 AUDIT TRAIL")
    print(f"{'─' * 62}")
    print(AUDIT.summary())
    print(f"  💾 Saved → {audit_path}")

    return report_path, dashboard_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-agent payroll compliance pipeline"
    )
    parser.add_argument(
        "--file",
        default="data/raw/payroll_data.csv",
        help="Path to payroll CSV (default: data/raw/payroll_data.csv)"
    )
    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"\n❌ File not found: {args.file}")
        print("   Run 01_generate_data.ipynb first to create the dataset.")
        sys.exit(1)

    report_path, dashboard_path = run_pipeline(args.file)

    tokens = get_token_summary()
    print(f"\n{'═' * 62}")
    print("  💰 TOKEN USAGE")
    print(f"{'═' * 62}")
    print(f"  API calls:     {tokens.get('calls', 0)}")
    print(f"  Input tokens:  {tokens.get('input', 0):,}")
    print(f"  Output tokens: {tokens.get('output', 0):,}")
    print(f"  Est. cost:     ${tokens.get('cost_usd', 0):.4f} USD")

    print(f"\n{'═' * 62}")
    print("  ✅ PIPELINE COMPLETE")
    print(f"{'═' * 62}")
    print(f"  📄 Report:    {report_path}")
    print(f"  📊 Dashboard: {dashboard_path}")
    print(f"{'═' * 62}\n")
    print("  Open the dashboard in your browser:")
    print(f"  open {dashboard_path}   (Mac)")
    print(f"  start {dashboard_path}  (Windows)\n")
