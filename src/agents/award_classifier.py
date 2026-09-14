"""
award_classifier.py — Agent 1.5 (runs between Validator and Compliance Analyst)
Maps unique job titles to Modern Awards + classification levels.

TOKEN EFFICIENCY DESIGN:
- Only UNIQUE job titles are sent (15 titles, not 900 rows)
- Uses claude-haiku (cheapest model) — classification is a simple task
- Single API call, small structured output
- Typical cost: ~500 input + ~300 output tokens ≈ $0.001

The actual rate lookups and underpayment math happen in award_engine.py
with ZERO tokens.
"""

import json
import pandas as pd
from .base_agent import BaseAgent
from src.audit import AUDIT
from src.award_engine import SUPPORTED_AWARDS

SYSTEM_PROMPT = """
You classify Australian job titles into Modern Awards and classification levels.

SUPPORTED AWARDS (only use these codes):
- MA000002: Clerks — Private Sector Award. Levels 1-5.
    Admin assistant, receptionist, data entry, bookkeeper, payroll officer,
    accounts payable/receivable, office manager, executive assistant, clerk.
- MA000004: General Retail Industry Award. Levels 1-8.
    Retail assistant, sales assistant, shop assistant, checkout operator,
    visual merchandiser, department manager, store manager, assistant manager.
- MA000009: Hospitality Industry (General) Award. Levels 1-6.
    Chef, cook, kitchen hand, waiter, food & beverage attendant, bartender,
    barista, housekeeper, porter, front office, duty manager, venue manager.
- MA000010: Manufacturing and Associated Industries Award. Levels 1-8.
    Production worker, machine operator, process worker, assembler, fitter,
    welder, boilermaker, storeperson, warehouse/forklift operator, dispatch.
- MA000065: Professional Employees Award. Levels 1-4.
    Engineer, software developer, programmer, IT/systems analyst, scientist,
    data analyst, data scientist, architect, surveyor, quantity surveyor.
- NMW: National Minimum Wage. Genuinely award-free roles — senior executives
    (CEO, CFO, GM, director) and salaried roles above award coverage.
- UNKNOWN: the honest answer when the role sits outside all five awards above.
    Common cases: nurses, aged care, teachers, childcare, construction and
    electrical trades, drivers/transport, security, cleaning, real estate,
    banking/finance specialists, legal, government. These have their own
    awards, for which this tool does not yet hold rate tables.

Match on the WORK PERFORMED, not the seniority word. "Warehouse Manager" is
storage/manufacturing (MA000010); "Payroll Manager" is clerical (MA000002).
A seniority word raises the LEVEL; it rarely changes the award.

Return UNKNOWN only when the role genuinely falls outside all five awards —
never merely because you are unsure of the level. Pick the level and mark
confidence LOW instead.

If a value is a DEPARTMENT rather than a job title ("Finance", "Operations",
"Head Office"), you cannot classify it: an award attaches to a person's duties,
not to a business unit. Return UNKNOWN, confidence LOW, and say so in the
rationale.

Classification levels: 1 = entry level, higher = more senior/qualified. Use the
median wage supplied with each title to place the level — a title paid well
above the award floor is usually mid-to-senior.

For each title give a one-sentence "rationale": which award clause or duty
makes this the right code and level. A reviewer checks that sentence against
the award text, so cite the coverage reason — not your own certainty.

Return ONLY valid JSON:
{
  "Job Title Here": {"award_code": "MA000065", "level": 2, "confidence": "HIGH",
                     "rationale": "why this award and level"},
  ...
}
"""


class AwardClassifierAgent(BaseAgent):
    """
    Agent 1.5: Classifies unique job titles → award + level.
    Uses Haiku for cost efficiency. One API call regardless of dataset size.
    """

    def __init__(self, client, verbose: bool = True):
        super().__init__(client, verbose)
        self.model = "claude-haiku-4-5-20251001"   # cheapest — classification is simple

    def run(self, df: pd.DataFrame) -> dict:
        title_col = "job_title" if "job_title" in df.columns else "department"
        unique_titles = sorted(df[title_col].dropna().astype(str).unique().tolist())

        # A pinned classification replaces the model call entirely. Titles the
        # pin doesn't cover fall through to unknown_coverage downstream, which
        # is reported rather than guessed.
        pinned = AUDIT.pinned_decision("award_classifier", "award_classification")
        if pinned is not None:
            self._log(f"      📌 Replaying pinned classification "
                      f"({len(pinned)} title(s), no API call)")
            self._log_award_summary(pinned)
            return pinned

        self._log(f"      🏷️  Classifying {len(unique_titles)} unique job titles "
                  f"(from {len(df)} rows — {len(df) - len(unique_titles)} rows saved)")

        # Median wage places the level; department disambiguates titles that
        # span awards (a "Manager" in Retail vs one in Warehouse).
        wage_context = {}
        if "gross_wage" in df.columns:
            med = df.groupby(title_col)["gross_wage"].median().round(0)
            wage_context = {t: f"median fortnightly gross ${med.get(t, 0):,.0f}"
                            for t in unique_titles}

        dept_context = {}
        if "department" in df.columns and title_col != "department":
            modes = df.groupby(title_col)["department"].agg(
                lambda x: x.mode().iat[0] if len(x.mode()) else "")
            dept_context = {t: str(modes.get(t, "")) for t in unique_titles}

        payload = [{
            "title":      t,
            "department": dept_context.get(t, ""),
            "context":    wage_context.get(t, ""),
        } for t in unique_titles]

        # Be explicit when there is no job_title column: the model is being shown
        # department names, and should say so rather than force a match.
        if title_col != "job_title":
            payload.insert(0, {"WARNING": f"This dataset has no job_title column. The "
                                          f"values below come from '{title_col}' and may "
                                          f"be departments rather than job titles."})

        user_msg = "Classify these job titles:\n" + json.dumps(payload, indent=1)

        raw = self.call(system=SYSTEM_PROMPT, user_message=user_msg, max_tokens=1024)

        try:
            clean = raw.strip().replace("```json", "").replace("```", "").strip()
            classification = json.loads(clean)
        except json.JSONDecodeError:
            self._log("      ⚠️  Classification parse failed — all titles marked UNKNOWN")
            classification = {t: {"award_code": "UNKNOWN", "level": 1, "confidence": "LOW"}
                              for t in unique_titles}

        AUDIT.log_llm_decision(
            agent="award_classifier",
            model=self.model,
            decision_type="award_classification",
            decision=classification,
        )

        self._log_award_summary(classification)
        return classification

    def _log_award_summary(self, classification: dict):
        by_award = {}
        for t, c in classification.items():
            by_award.setdefault(c.get("award_code", "UNKNOWN"), []).append(t)
        for code, titles in by_award.items():
            self._log(f"      → {SUPPORTED_AWARDS.get(code, code)}: {len(titles)} title(s)")
