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
- MA000002: Clerks — Private Sector Award (admin, office, reception, bookkeeping, data entry)
- MA000004: General Retail Industry Award (retail sales, shop assistants, store managers)
- MA000009: Hospitality Industry (General) Award (chefs, waiters, bar, hotel staff)
- MA000010: Manufacturing and Associated Industries Award (production, trades, machine operators, warehouse)
- MA000065: Professional Employees Award (engineers, IT professionals, scientists, data analysts)
- NMW: National Minimum Wage (use if genuinely award-free, e.g. most managers above award coverage)
- UNKNOWN: Use if the title doesn't clearly fit any of the above

Classification levels: 1 = entry level, higher = more senior/qualified.
Most awards have 4-8 levels. When unsure, use a middle level and mark confidence LOW.

Note: senior managers and executives are usually award-free (NMW code, but their
salaries far exceed it — they will not be flagged). Professionals like software
engineers fall under MA000065.

Return ONLY valid JSON:
{
  "Job Title Here": {"award_code": "MA000065", "level": 2, "confidence": "HIGH"},
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

        self._log(f"      🏷️  Classifying {len(unique_titles)} unique job titles "
                  f"(from {len(df)} rows — {len(df) - len(unique_titles)} rows saved)")

        # Include median wage per title — helps Haiku pick sensible levels
        wage_context = {}
        if "gross_wage" in df.columns:
            med = df.groupby(title_col)["gross_wage"].median().round(0)
            wage_context = {t: f"median fortnightly gross ${med.get(t, 0):,.0f}"
                            for t in unique_titles}

        user_msg = (
            "Classify these job titles:\n"
            + json.dumps(
                [{"title": t, "context": wage_context.get(t, "")} for t in unique_titles],
                indent=1,
            )
        )

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

        # Log summary
        by_award = {}
        for t, c in classification.items():
            code = c.get("award_code", "UNKNOWN")
            by_award.setdefault(code, []).append(t)
        for code, titles in by_award.items():
            name = SUPPORTED_AWARDS.get(code, code)
            self._log(f"      → {name}: {len(titles)} title(s)")

        return classification
