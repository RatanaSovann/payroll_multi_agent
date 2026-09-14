"""
eval/run_eval.py
Golden-dataset evaluation harness for the payroll compliance pipeline.

Two stages, because not every assertion needs to cost money:

  --stage deterministic   (default, FREE)
      Runs clean_dataframe() + the compliance checks directly. No API calls.
      Covers every 'exact' expectation on cases already in canonical schema.

  --stage full            (COSTS API CREDITS)
      Runs the real six-stage pipeline including the LLM agents. Required for
      'labelled' expectations (schema mapping, award classification, report
      prose) and for the security/reporting cases.

Assertion kinds (see docs/AGENTIC_EVALUATION.md):
  exact     — deterministic value, compared with a $0.01 tolerance
  bounded   — value depends on an LLM choice but is provably inside a range
  labelled  — an answer key for an LLM decision, scored as accuracy

Usage:
  python eval/run_eval.py
  python eval/run_eval.py --case sg_underpayment_only
  python eval/run_eval.py --tag cleaning
  python eval/run_eval.py --stage full --case schema_xero
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Same reason as orchestrator.py: cp1252 consoles mangle the separators and
# status glyphs in the summary table.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from src.audit import AUDIT
from src.data_cleaner import clean_dataframe
from src.compliance import (
    check_sg_underpayment,
    check_payg_inconsistency,
    check_missing_casual_super,
    generate_executive_summary,
)
from src.award_engine import check_award_underpayment

CASES_DIR = Path(__file__).parent / "cases"
MONEY_TOLERANCE = 0.01

# expectation key -> where to find it in the collected results
EXACT_PATHS = {
    "sg_total":         ("compliance", "sg", "total_shortfall_aud"),
    "sg_employees":     ("compliance", "sg", "employees_affected"),
    "sg_runs":          ("compliance", "sg", "pay_runs_affected"),
    "payg_total":       ("compliance", "payg", "total_exposure_aud"),
    "payg_employees":   ("compliance", "payg", "employees_affected"),
    "payg_runs":        ("compliance", "payg", "pay_runs_affected"),
    "casual_total":     ("compliance", "casuals", "total_shortfall_aud"),
    "casual_employees": ("compliance", "casuals", "employees_affected"),
    "award_total":      ("compliance", "awards", "total_underpayment_aud"),
    "award_employees":  ("compliance", "awards", "employees_affected"),
    "award_runs":       ("compliance", "awards", "pay_runs_affected"),
    "total_exposure":   ("compliance", "summary", "total_exposure_aud"),
    "quarantined":      ("quality", "quarantined_rows"),
    "rows_analysable":  ("quality", "rows_after"),
    "duplicates_removed": ("audit", "duplicates_removed"),
    "dates_recovered":  ("audit", "dates_recovered"),
}

MONEY_KEYS = {"sg_total", "payg_total", "casual_total", "award_total", "total_exposure"}

# Award results depend on the LLM classifier, so they are unassertable without it.
AWARD_KEYS = {"award_total", "award_employees", "award_runs"}

CANONICAL_REQUIRED = {"employee_id", "gross_wage", "super_paid", "expected_sg"}


def dig(results, path):
    cur = results
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def audit_facts():
    """Structured facts the cleaner recorded — used instead of parsing prose."""
    facts = {"duplicates_removed": 0, "dates_recovered": 0,
             "logical_duplicates_flagged": False}
    for e in AUDIT.events:
        action, detail = e.get("action"), e.get("detail", {})
        if action == "remove_exact_duplicates":
            facts["duplicates_removed"] += detail.get("rows_removed", 0)
        elif action == "excel_serial_dates":
            facts["dates_recovered"] += detail.get("rows_recovered", 0)
        elif action == "logical_duplicate_warning":
            facts["logical_duplicates_flagged"] = True
    return facts


# ── Stage 1: deterministic (free) ─────────────────────────────────────────────

def run_deterministic(csv_path, classification_map=None):
    df = pd.read_csv(csv_path)
    AUDIT.start_run(str(csv_path), df)

    clean, quality = clean_dataframe(df)

    if len(clean) == 0:
        compliance = {"sg": {}, "payg": {}, "casuals": {}, "awards": {},
                      "summary": {"total_exposure_aud": 0.0}}
        # Empty frame still has to produce zeroed findings, not a crash.
        for k in ("sg", "payg", "casuals"):
            compliance[k] = {"total_shortfall_aud": 0.0, "total_exposure_aud": 0.0,
                             "employees_affected": 0, "pay_runs_affected": 0}
        return {"compliance": compliance, "quality": quality, "audit": audit_facts()}

    sg = check_sg_underpayment(clean)
    payg = check_payg_inconsistency(clean)
    casuals = check_missing_casual_super(clean)
    awards = check_award_underpayment(clean, classification_map or {})
    summary = generate_executive_summary(sg, payg, casuals)

    return {
        "compliance": {"sg": sg, "payg": payg, "casuals": casuals,
                       "awards": awards, "summary": summary},
        "quality": quality,
        "audit": audit_facts(),
    }


# ── Stage 2: full pipeline (costs credits) ────────────────────────────────────

def run_full(csv_path):
    import anthropic
    from dotenv import load_dotenv
    from src.agents import (DataValidatorAgent, AwardClassifierAgent,
                            ComplianceAnalystAgent, RiskAssessorAgent,
                            ReportWriterAgent, DashboardGeneratorAgent)
    from src.agents.base_agent import reset_token_ledger, get_token_summary

    load_dotenv()
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY not set — required for --stage full")

    client = anthropic.Anthropic(api_key=key)
    df = pd.read_csv(csv_path)

    reset_token_ledger()
    AUDIT.start_run(str(csv_path), df)

    validation, mapped = DataValidatorAgent(client, verbose=False).run(df)
    classification = AwardClassifierAgent(client, verbose=False).run(mapped)
    compliance = ComplianceAnalystAgent(client, verbose=False).run(
        mapped, classification_map=classification)
    risk = RiskAssessorAgent(client, verbose=False).run(compliance)
    report = ReportWriterAgent(client, verbose=False).run(
        validation=validation, compliance=compliance, risk=risk)

    out_dir = Path(__file__).parent / "_artifacts"
    out_dir.mkdir(exist_ok=True)
    dash_path = out_dir / f"{Path(csv_path).parent.name}_dashboard.html"
    DashboardGeneratorAgent().run(compliance=compliance, risk=risk, df=mapped,
                                  output_path=str(dash_path))

    return {
        "compliance": compliance,
        "quality": validation.get("data_quality", {}),
        "audit": audit_facts(),
        "validation": validation,
        "classification": classification,
        "risk": risk,
        "report": report,
        "dashboard_html": dash_path.read_text(encoding="utf-8"),
        "tokens": get_token_summary(),
    }


# ── Comparison ────────────────────────────────────────────────────────────────

def check_exact(expected, results, stage):
    out = []
    for key, want in expected.items():
        if stage != "full" and key in AWARD_KEYS:
            out.append((key, "SKIP", "award check needs the classifier (--stage full)"))
            continue
        path = EXACT_PATHS.get(key)
        if path is None:
            out.append((key, "SKIP", f"no path mapping for '{key}'"))
            continue
        got = dig(results, path)
        if got is None:
            out.append((key, "FAIL", f"expected {want}, got <missing>"))
        elif key in MONEY_KEYS:
            ok = abs(float(got) - float(want)) <= MONEY_TOLERANCE
            out.append((key, "PASS" if ok else "FAIL",
                        f"expected ${want:,.2f}, got ${float(got):,.2f}"))
        else:
            ok = int(got) == int(want)
            out.append((key, "PASS" if ok else "FAIL", f"expected {want}, got {got}"))
    return out


def check_bounded(expected, results, stage):
    out = []
    for key, (lo, hi) in expected.items():
        if stage != "full" and key in AWARD_KEYS:
            out.append((key, "SKIP", "award check needs the classifier (--stage full)"))
            continue
        path = EXACT_PATHS.get(key)
        got = dig(results, path) if path else None
        if got is None:
            out.append((key, "FAIL", f"expected within [{lo}, {hi}], got <missing>"))
        else:
            ok = lo - MONEY_TOLERANCE <= float(got) <= hi + MONEY_TOLERANCE
            out.append((key, "PASS" if ok else "FAIL",
                        f"expected within [${lo:,.2f}, ${hi:,.2f}], got ${float(got):,.2f}"))
    return out


def check_labelled(expected, results, stage):
    """LLM-decision assertions. Only meaningful on the full pipeline."""
    out = []
    if stage != "full":
        return [(k, "SKIP", "needs --stage full") for k in expected]

    for key, want in expected.items():
        if key == "mapping_required":
            mapping = dig(results, ("validation", "column_mapping")) or {}
            wrong = {src: f"{mapping.get(src)} != {tgt}"
                     for src, tgt in want.items() if mapping.get(src) != tgt}
            out.append((key, "PASS" if not wrong else "FAIL",
                        "all critical columns mapped" if not wrong else str(wrong)))

        elif key == "expect_unmapped":
            unmapped = set(dig(results, ("validation", "unmapped_columns")) or [])
            missing = [c for c in want if c not in unmapped]
            out.append((key, "PASS" if not missing else "FAIL",
                        "reported UNMAPPED" if not missing
                        else f"silently mapped instead of flagged: {missing}"))

        elif key == "classification":
            cls = results.get("classification") or {}
            wrong = {t: f"{(cls.get(t) or {}).get('award_code')} != {code}"
                     for t, code in want.items()
                     if (cls.get(t) or {}).get("award_code") != code}
            out.append((key, "PASS" if not wrong else "FAIL",
                        "classified correctly" if not wrong else str(wrong)))

        elif key == "expect_unknown_or_low_confidence":
            cls = results.get("classification") or {}
            guessed = [t for t in want
                       if (cls.get(t) or {}).get("award_code") not in (None, "UNKNOWN")
                       and (cls.get(t) or {}).get("confidence") not in ("LOW",)]
            out.append((key, "PASS" if not guessed else "FAIL",
                        "ambiguous titles not guessed" if not guessed
                        else f"confidently guessed an award for: {guessed}"))

        elif key == "name_merged":
            mapped_cols = dig(results, ("validation", "stats", "mapped_columns")) or []
            out.append((key, "PASS" if "name" in mapped_cols else "FAIL",
                        "name column present" if "name" in mapped_cols
                        else "no merged name column"))

        elif key == "quality_score":
            got = dig(results, ("quality", "quality_score"))
            out.append((key, "PASS" if got == want else "FAIL",
                        f"expected {want}, got {got}"))

        elif key == "logical_duplicates_flagged":
            got = dig(results, ("audit", "logical_duplicates_flagged"))
            out.append((key, "PASS" if bool(got) == bool(want) else "FAIL",
                        f"expected {want}, got {got}"))

        elif key == "dashboard_must_escape":
            html = results.get("dashboard_html", "")
            leaked = [s for s in want if s in html]
            out.append((key, "PASS" if not leaked else "FAIL",
                        "markup escaped" if not leaked
                        else f"UNESCAPED markup in dashboard: {leaked}"))

        elif key == "report_period_must_mention":
            report = results.get("report", "")
            missing = [s for s in want if s not in report]
            out.append((key, "PASS" if not missing else "FAIL",
                        "period reflects data" if not missing
                        else f"report never mentions {missing}"))

        elif key == "report_period_must_not_mention":
            report = results.get("report", "")
            found = [s for s in want if s in report]
            out.append((key, "PASS" if not found else "FAIL",
                        "no hardcoded period" if not found
                        else f"hardcoded period leaked into report: {found}"))

        else:
            out.append((key, "SKIP", f"no checker for '{key}'"))
    return out


# ── Runner ────────────────────────────────────────────────────────────────────

def load_cases(case_filter=None, tag_filter=None):
    cases = []
    for d in sorted(CASES_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta = json.loads((d / "expected.json").read_text(encoding="utf-8"))
        if case_filter and meta["id"] != case_filter:
            continue
        if tag_filter and tag_filter not in meta["tags"]:
            continue
        meta["dir"] = d
        cases.append(meta)
    return cases


def main():
    ap = argparse.ArgumentParser(description="Payroll pipeline evaluation harness")
    ap.add_argument("--stage", choices=["deterministic", "full"], default="deterministic")
    ap.add_argument("--case", help="run a single case by id")
    ap.add_argument("--tag", help="run only cases carrying this tag")
    ap.add_argument("--json", help="write the full result report to this path")
    args = ap.parse_args()

    cases = load_cases(args.case, args.tag)
    if not cases:
        raise SystemExit("No matching cases. Run: python eval/generate_cases.py")

    print(f"\n{'=' * 78}")
    print(f"  PAYROLL PIPELINE EVAL — stage: {args.stage} — {len(cases)} case(s)")
    if args.stage == "full":
        print("  WARNING: --stage full makes real API calls and costs credits.")
    print(f"{'=' * 78}")

    report, totals = [], {"PASS": 0, "FAIL": 0, "SKIP": 0, "ERROR": 0}

    for meta in cases:
        csv_path = meta["dir"] / "input.csv"
        expected = meta["expected"]

        # Deterministic mode bypasses the LLM schema mapper, so a non-canonical
        # CSV simply cannot be run here — that's a skip, not a failure.
        header = set(pd.read_csv(csv_path, nrows=0).columns)
        if args.stage != "full" and not CANONICAL_REQUIRED.issubset(header):
            checks = [("<run>", "SKIP", "non-canonical schema needs the mapper "
                                        "(--stage full)")]
        else:
            try:
                results = (run_full(csv_path) if args.stage == "full"
                           else run_deterministic(csv_path))
                checks = []
                checks += check_exact(expected.get("exact", {}), results, args.stage)
                checks += check_bounded(expected.get("bounded", {}), results, args.stage)
                checks += check_labelled(expected.get("labelled", {}), results, args.stage)
                if args.stage == "full":
                    meta["tokens"] = results.get("tokens", {})
            except Exception as exc:
                checks = [("<run>", "ERROR", f"{type(exc).__name__}: {exc}")]

        for _, status, _ in checks:
            totals[status] = totals.get(status, 0) + 1

        worst = ("ERROR" if any(s == "ERROR" for _, s, _ in checks)
                 else "FAIL" if any(s == "FAIL" for _, s, _ in checks)
                 else "SKIP" if all(s == "SKIP" for _, s, _ in checks)
                 else "PASS")
        icon = {"PASS": "PASS", "FAIL": "FAIL", "ERROR": "ERR ", "SKIP": "SKIP"}[worst]
        print(f"\n[{icon}] {meta['id']}  ({', '.join(meta['tags'])})")
        for name, status, detail in checks:
            if status != "PASS":
                print(f"       {status:<5} {name}: {detail}")

        report.append({"id": meta["id"], "tags": meta["tags"], "verdict": worst,
                       "checks": [{"name": n, "status": s, "detail": d}
                                  for n, s, d in checks]})

    n_fail = sum(1 for r in report if r["verdict"] in ("FAIL", "ERROR"))
    n_skip = sum(1 for r in report if r["verdict"] == "SKIP")
    print(f"\n{'=' * 78}")
    print(f"  CASES: {len(report) - n_fail - n_skip} passed · {n_fail} failed · "
          f"{n_skip} skipped (of {len(report)})")
    print(f"  CHECKS: {totals['PASS']} pass · {totals['FAIL']} fail · "
          f"{totals['SKIP']} skipped · {totals['ERROR']} errored")
    print(f"{'=' * 78}\n")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"stage": args.stage, "totals": totals, "cases": report}, indent=2),
            encoding="utf-8")
        print(f"  Report written to {args.json}\n")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
