"""
risk_assessor.py — Agent 3
Scores each compliance finding by severity and priority.
Estimates ATO penalties. Identifies what needs to be fixed first.

This is the layer that turns raw findings into business risk language —
what a Senior Manager or Partner would care about.
"""

import json
from .base_agent import BaseAgent
from src.audit import AUDIT

SYSTEM_PROMPT = """
You are a senior Australian employment tax risk specialist at a Big 4 firm.
You have received structured compliance findings from the analyst team.

Your job is to:
1. Score each finding as HIGH / MEDIUM / LOW risk
2. Estimate ATO Super Guarantee Charge (SGC) penalties where applicable
   - SGC = unpaid super + 10% p.a. interest + $20 admin fee per employee per quarter
3. Identify the single highest-priority remediation action
4. Flag any Payday Super readiness concerns (effective 1 July 2026)

Scoring criteria:
- HIGH:   >$20,000 exposure OR >20 employees OR regulatory penalty risk
- MEDIUM: $5,000–$20,000 exposure OR 5–20 employees
- LOW:    <$5,000 exposure OR <5 employees

Return ONLY a valid JSON object with this structure:
{
  "risk_scores": {
    "sg_underpayment":       { "score": "HIGH|MEDIUM|LOW", "rationale": "..." },
    "payg_inconsistency":    { "score": "HIGH|MEDIUM|LOW", "rationale": "..." },
    "missing_casual_super":  { "score": "HIGH|MEDIUM|LOW", "rationale": "..." }
  },
  "estimated_sgc_penalty":   <number>,
  "highest_priority_action": "...",
  "payday_super_readiness":  "...",
  "overall_risk_rating":     "HIGH|MEDIUM|LOW",
  "narrative":               "2-3 sentence risk summary for the engagement partner"
}

Return only the JSON. No markdown, no explanation.
"""


class RiskAssessorAgent(BaseAgent):
    """
    Agent 3: Scores findings by severity and estimates penalties.
    Uses Haiku — scoring against an explicit rubric is structured, not creative.
    """

    def __init__(self, client, verbose: bool = True):
        super().__init__(client, verbose)
        self.model = "claude-haiku-4-5-20251001"

    def run(self, compliance_findings: dict) -> dict:
        self._log("      ⚖️  Scoring findings by severity...")

        raw = self.call(
            system=SYSTEM_PROMPT,
            user_message=f"Compliance findings:\n{json.dumps(compliance_findings, default=str, indent=2)}",
        )

        try:
            clean = raw.strip().replace("```json", "").replace("```", "").strip()
            result = json.loads(clean)
        except json.JSONDecodeError:
            result = {"raw_output": raw}

        AUDIT.log_llm_decision(
            agent="risk_assessor",
            model=self.model,
            decision_type="risk_scoring",
            decision={
                "risk_scores": result.get("risk_scores", {}),
                "overall_risk_rating": result.get("overall_risk_rating"),
                "estimated_sgc_penalty": result.get("estimated_sgc_penalty"),
                "rubric": "HIGH >$20k or >20 employees; MEDIUM $5-20k or 5-20 employees; LOW below",
            },
        )
        return result
