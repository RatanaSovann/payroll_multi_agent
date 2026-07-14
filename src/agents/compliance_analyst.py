"""
compliance_analyst.py — Agent 2
Runs all three compliance checks using tool use.
Tool results are captured directly in Python — no JSON parsing of Claude's output.
"""

import pandas as pd
from .base_agent import BaseAgent
from src.compliance import (
    inspect_data,
    check_sg_underpayment,
    check_payg_inconsistency,
    check_missing_casual_super,
    generate_executive_summary,
)
from src.award_engine import check_award_underpayment

SYSTEM_PROMPT = """
You are an Australian payroll compliance analyst.
Run the compliance tools in this exact order:
1. inspect_data
2. check_sg_underpayment
3. check_payg_inconsistency
4. check_missing_casual_super
5. check_award_rates (Modern Award minimum rate compliance)
6. generate_executive_summary (pass the results of steps 2, 3, 4 as arguments)

After all tools have run, confirm the analysis is complete with a one-line summary.
"""

TOOLS = [
    {
        "name": "inspect_data",
        "description": "Inspect schema, column types, and sample rows. Call this first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n_rows": {"type": "integer", "description": "Sample rows to return (default 5)"}
            },
        },
    },
    {
        "name": "check_sg_underpayment",
        "description": "Flag pay runs where super paid < 12% SG rate. Returns shortfall by dept/state.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tolerance": {"type": "number", "description": "Rounding tolerance (default 0.001)"}
            },
        },
    },
    {
        "name": "check_payg_inconsistency",
        "description": "Flag PAYG deviations beyond threshold. Returns over/under-withheld breakdown.",
        "input_schema": {
            "type": "object",
            "properties": {
                "threshold_pct": {"type": "number", "description": "Deviation % threshold (default 15.0)"}
            },
        },
    },
    {
        "name": "check_missing_casual_super",
        "description": "Identify casuals with zero super. All casuals are SG-eligible post July 2022.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "check_award_rates",
        "description": "Check pay rates against Modern Award minimums (FWC 2026 rates). Uses pre-classified job titles. Zero-parameter call.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "generate_executive_summary",
        "description": "Roll up all three findings into one summary. Call after all three checks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sg_result":      {"type": "object", "description": "Result from check_sg_underpayment"},
                "payg_result":    {"type": "object", "description": "Result from check_payg_inconsistency"},
                "casuals_result": {"type": "object", "description": "Result from check_missing_casual_super"},
            },
            "required": ["sg_result", "payg_result", "casuals_result"],
        },
    },
]


class ComplianceAnalystAgent(BaseAgent):
    """
    Agent 2: Runs all compliance checks via tool use loop.
    Captures tool results directly in Python — no JSON parsing of Claude's text output.
    Returns structured findings dict for downstream agents.
    """

    def run(self, df: pd.DataFrame, classification_map: dict = None) -> dict:
        import json
        classification_map = classification_map or {}

        # Capture tool results as they execute — keyed by tool name
        captured = {}

        def tool_executor(tool_name: str, tool_input: dict) -> dict:
            if tool_name == "inspect_data":
                result = inspect_data(df, n_rows=tool_input.get("n_rows", 5))

            elif tool_name == "check_sg_underpayment":
                result = check_sg_underpayment(df, tolerance=tool_input.get("tolerance", 0.001))

            elif tool_name == "check_payg_inconsistency":
                result = check_payg_inconsistency(df, threshold_pct=tool_input.get("threshold_pct", 15.0))

            elif tool_name == "check_missing_casual_super":
                result = check_missing_casual_super(df)

            elif tool_name == "check_award_rates":
                result = check_award_underpayment(df, classification_map)

            elif tool_name == "generate_executive_summary":
                result = generate_executive_summary(
                    sg=tool_input["sg_result"],
                    payg=tool_input["payg_result"],
                    casuals=tool_input["casuals_result"],
                )
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            # Capture every result directly in Python — don't rely on Claude's JSON
            captured[tool_name] = result
            return result

        # Run the agent loop — Claude decides the order and calls the tools
        self.run_loop(
            system=SYSTEM_PROMPT,
            user_message="Run all compliance checks on the payroll dataset now.",
            tools=TOOLS,
            tool_executor=tool_executor,
            max_tokens=4096,
        )

        # Build the compliance dict from captured Python results — not Claude's text
        return {
            "inspection": captured.get("inspect_data", {}),
            "sg":         captured.get("check_sg_underpayment", {}),
            "payg":       captured.get("check_payg_inconsistency", {}),
            "casuals":    captured.get("check_missing_casual_super", {}),
            "awards":     captured.get("check_award_rates", {}),
            "summary":    captured.get("generate_executive_summary", {}),
        }

