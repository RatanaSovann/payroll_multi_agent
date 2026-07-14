"""
data_validator.py — Agent 1
Validates the payroll dataset and maps any incoming schema to the
canonical column names the pipeline expects.

Two-step process:
  1. Schema mapping  — Claude maps incoming columns to canonical names
  2. Validation      — check completeness, nulls, plausibility

Returns (validation_dict, mapped_df) so the rest of the pipeline
always works with canonical column names regardless of the source system.
"""

import json
import pandas as pd
from .base_agent import BaseAgent
from src.audit import AUDIT

# ── Canonical schema ───────────────────────────────────────────────────────────
# These are the column names every downstream agent expects.
# If a client CSV uses different names, the mapper renames them to these.

CANONICAL_SCHEMA = {
    "employee_id":     "Unique identifier for each employee (e.g. EMP0001, E123, staff_id)",
    "name":            "Employee full name",
    "state":           "Australian state/territory code (NSW, VIC, QLD, WA, SA, TAS, ACT, NT)",
    "department":      "Department or business unit name",
    "employment_type": "Employment classification (Full-time, Part-time, Casual, Contract)",
    "pay_date":        "Date the payment was made (any date format)",
    "gross_wage":      "Total gross earnings/wage for the pay period before deductions",
    "super_paid":      "Superannuation/super contribution amount actually paid this period",
    "expected_sg":     "Expected super guarantee amount (typically 12% of gross OTE)",
    "payg_withheld":   "PAYG income tax withheld from the employee this period",
    "correct_payg":    "Correct PAYG withholding amount based on ATO tax bracket rates",
}

REQUIRED_COLUMNS = list(CANONICAL_SCHEMA.keys())

# ── System prompts ─────────────────────────────────────────────────────────────

MAPPER_PROMPT = """
You are a payroll data schema mapping specialist for Australian employment tax compliance.

Your job is to map incoming CSV column names to the canonical schema used by the compliance pipeline.

CANONICAL SCHEMA (target column names and what they mean):
{canonical}

Given the incoming columns and their sample values, return ONLY a valid JSON object
mapping each incoming column name to its canonical equivalent.

Rules:
- Map to the canonical name if there is a clear match (even if wording differs)
- Use "UNMAPPED" if a column has no clear canonical equivalent
- Every incoming column must appear as a key in your response
- Be liberal with matching — "GrossEarnings", "gross_pay", "TotalPay" all map to "gross_wage"
- Common payroll system patterns to recognise:
    Xero:    EmployeeID, FirstName+LastName, GrossEarnings, SuperannuationExpense
    MYOB:    Card ID, Gross Wages, Super Expense
    SAP:     PERNR, LGART, BETRG
    ADP:     Emp#, Reg Pay, EE Super
- If two columns together form one canonical column (e.g. FirstName + LastName → name),
  map BOTH to the same canonical name and note it in a "merge_notes" key

Return format:
{{
  "mapping": {{
    "IncomingColName": "canonical_col_name"
  }},
  "merge_notes": "any notes about columns that need merging (optional)",
  "confidence": "HIGH | MEDIUM | LOW"
}}

Return only the JSON. No markdown, no explanation.
"""

VALIDATOR_PROMPT = """
You are a data quality analyst specialising in Australian payroll datasets.
Your job is to validate a payroll dataset before compliance checks are run.

You will receive a validation report as a JSON object.
Produce a concise data quality summary covering:
1. Schema mapping result — were all required columns found and mapped?
2. Data completeness — any nulls or missing values to flag?
3. Data plausibility — do wages, super, and PAYG amounts look reasonable for AU?
4. Any risks or caveats the compliance team should know before proceeding

Be concise. Bullet points are fine. This is an internal handoff note, not a client document.
"""


class DataValidatorAgent(BaseAgent):
    """
    Agent 1: Maps incoming schema to canonical columns, then validates the dataset.
    Uses Haiku — schema mapping is structured classification, not deep reasoning.

    Returns:
        (validation_dict, mapped_df)
    """

    def __init__(self, client, verbose: bool = True):
        super().__init__(client, verbose)
        self.model = "claude-haiku-4-5-20251001"

    def run(self, df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:

        # ── Step 1: Schema mapping ─────────────────────────────────────────────
        self._log("      🗺️  Mapping schema to canonical columns...")

        mapping_result = self._map_schema(df)
        column_mapping = mapping_result.get("mapping", {})

        AUDIT.log_llm_decision(
            agent="data_validator",
            model=self.model,
            decision_type="schema_mapping",
            decision={
                "mapping": column_mapping,
                "confidence": mapping_result.get("confidence"),
                "merge_notes": mapping_result.get("merge_notes", ""),
            },
        )
        unmapped       = [k for k, v in column_mapping.items() if v == "UNMAPPED"]
        confidence     = mapping_result.get("confidence", "UNKNOWN")

        self._log(f"      ✓ Mapped {len(column_mapping) - len(unmapped)}/{len(column_mapping)} columns "
                  f"(confidence: {confidence})")
        if unmapped:
            self._log(f"      ⚠️  Unmapped: {unmapped}")

        # Apply the mapping — rename incoming columns to canonical names.
        # GUARD: multiple source columns can map to the same canonical target
        # (e.g. First_Name → name AND Last_Name → name). A blind rename would
        # create duplicate columns and break groupby downstream.
        rename_map, target_seen = {}, {}
        for src, tgt in column_mapping.items():
            if tgt == "UNMAPPED" or src not in df.columns:
                continue
            if tgt in target_seen:
                # Duplicate target — handle specially
                if tgt == "name":
                    target_seen[tgt].append(src)   # collect for merging below
                # For any other duplicate target: keep first source, skip rest
                continue
            target_seen[tgt] = [src]
            rename_map[src] = tgt

        mapped_df = df.rename(columns=rename_map)

        # Merge split name columns (First_Name + Last_Name → name)
        name_sources = target_seen.get("name", [])
        if len(name_sources) > 1:
            parts = [df[c].astype(str) for c in name_sources]
            merged = parts[0]
            for p in parts[1:]:
                merged = merged + " " + p
            mapped_df["name"] = merged
            self._log(f"      ✓ Merged {' + '.join(name_sources)} → name")
            AUDIT.log_transform("data_validator", "merge_name_columns", {
                "sources": name_sources,
                "method": "string concatenation with space separator",
            })

        # Safety net: drop any duplicate column names, keeping first occurrence
        if mapped_df.columns.duplicated().any():
            dupes = mapped_df.columns[mapped_df.columns.duplicated()].tolist()
            self._log(f"      ⚠️  Dropping duplicate columns: {dupes}")
            mapped_df = mapped_df.loc[:, ~mapped_df.columns.duplicated()]

        # ── Step 2: Column enrichment ──────────────────────────────────────────
        # After mapping, compute or fill any canonical columns still missing.
        # This handles real-world payroll exports that don't include derived fields.

        self._log("      🔧 Enriching missing canonical columns...")
        mapped_df = self._enrich_columns(mapped_df)

        # ── Step 3: Validation ─────────────────────────────────────────────────
        self._log("      📋 Validating data quality...")

        missing_cols = [c for c in REQUIRED_COLUMNS if c not in mapped_df.columns]
        null_counts  = mapped_df.isnull().sum()
        null_cols    = null_counts[null_counts > 0].to_dict()

        issues = []
        if missing_cols:
            issues.append(f"Missing required columns after mapping: {missing_cols}")
        if unmapped:
            issues.append(f"Unmapped source columns (excluded from analysis): {unmapped}")
        if null_cols:
            issues.append(f"Null values detected: {null_cols}")
        if "gross_wage" in mapped_df.columns:
            mapped_df["gross_wage"] = pd.to_numeric(mapped_df["gross_wage"], errors="coerce")
            if (mapped_df["gross_wage"] < 0).any():
                issues.append("Negative gross wages detected")
        if "super_paid" in mapped_df.columns:
            mapped_df["super_paid"] = pd.to_numeric(mapped_df["super_paid"], errors="coerce")
            if (mapped_df["super_paid"] < 0).any():
                issues.append("Negative super paid values detected")

        # Build stats from mapped df
        stats = {
            "source_columns":     list(df.columns),
            "mapped_columns":     list(mapped_df.columns),
            "column_mapping":     column_mapping,
            "unmapped_columns":   unmapped,
            "mapping_confidence": confidence,
            "merge_notes":        mapping_result.get("merge_notes", ""),
            "row_count":          len(mapped_df),
            "employee_count":     mapped_df["employee_id"].nunique()
                                  if "employee_id" in mapped_df.columns else "unknown",
            "pay_periods":        mapped_df["pay_date"].nunique()
                                  if "pay_date" in mapped_df.columns else "unknown",
            "employment_types":   mapped_df["employment_type"].value_counts().to_dict()
                                  if "employment_type" in mapped_df.columns else {},
            "states":             mapped_df["state"].value_counts().to_dict()
                                  if "state" in mapped_df.columns else {},
            "gross_wage_range":   {
                "min": round(float(mapped_df["gross_wage"].min()), 2),
                "max": round(float(mapped_df["gross_wage"].max()), 2),
                "avg": round(float(mapped_df["gross_wage"].mean()), 2),
            } if "gross_wage" in mapped_df.columns else {},
            "null_columns":       null_cols,
            "issues_found":       len(issues),
        }

        # Claude writes the narrative summary
        narrative = self.call(
            system=VALIDATOR_PROMPT,
            user_message=f"Validation report:\n{json.dumps(stats, indent=2)}\n\nIssues:\n{issues}",
        )

        validation = {
            "validation_passed":  len(missing_cols) == 0,
            "issues":             issues,
            "stats":              stats,
            "column_mapping":     column_mapping,
            "unmapped_columns":   unmapped,
            "mapping_confidence": confidence,
            "narrative":          narrative,
        }

        return validation, mapped_df

    # ── Column enricher ────────────────────────────────────────────────────────

    def _enrich_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        After schema mapping, compute or fill any canonical columns still missing.
        Handles four real-world scenarios:

        1. Split name columns  — First_Name + Last_Name → name
        2. Derived columns     — expected_sg, correct_payg (calculated from existing data)
        3. Missing categoricals — state, department, employment_type → filled with "Unknown"
        4. Hourly → gross wage — Base_Hours * Hourly_Rate if gross_wage missing
        """
        enriched = {}

        # 1. Merge split name columns
        if "name" not in df.columns:
            first = next((c for c in df.columns if c.lower() in
                          ["first_name","firstname","given_name","givenname"]), None)
            last  = next((c for c in df.columns if c.lower() in
                          ["last_name","lastname","surname","family_name"]), None)
            if first and last:
                df["name"] = df[first].astype(str) + " " + df[last].astype(str)
                enriched["name"] = f"merged from {first} + {last}"
            else:
                df["name"] = "Unknown"

        # 2. Derive gross_wage from hours × rate if missing
        if "gross_wage" not in df.columns:
            hours = next((c for c in df.columns if c.lower() in
                          ["base_hours","hours_worked","regular_hours"]), None)
            rate  = next((c for c in df.columns if c.lower() in
                          ["hourly_rate","rate","pay_rate","base_rate"]), None)
            total = next((c for c in df.columns if c.lower() in
                          ["total_gross","gross_pay","total_pay","gross_earnings"]), None)
            if total:
                df["gross_wage"] = pd.to_numeric(df[total], errors="coerce").fillna(0)
                enriched["gross_wage"] = f"from {total}"
            elif hours and rate:
                df["gross_wage"] = (
                    pd.to_numeric(df[hours], errors="coerce").fillna(0) *
                    pd.to_numeric(df[rate],  errors="coerce").fillna(0)
                )
                enriched["gross_wage"] = f"calculated from {hours} × {rate}"

        # 3. Compute expected_sg if missing (12% of gross_wage)
        if "expected_sg" not in df.columns and "gross_wage" in df.columns:
            df["expected_sg"] = (
                pd.to_numeric(df["gross_wage"], errors="coerce").fillna(0) * 0.12
            ).round(2)
            enriched["expected_sg"] = "calculated as gross_wage × 12%"

        # 4. Compute correct_payg from ATO brackets if missing
        if "correct_payg" not in df.columns and "gross_wage" in df.columns:
            def payg_bracket(gross_fortnightly):
                annual = gross_fortnightly * 26
                if annual <= 18200:   rate = 0.0
                elif annual <= 45000: rate = 0.19
                elif annual <= 120000: rate = 0.325
                elif annual <= 180000: rate = 0.37
                else:                  rate = 0.45
                return round(gross_fortnightly * rate, 2)

            df["correct_payg"] = df["gross_wage"].apply(
                lambda x: payg_bracket(float(x)) if pd.notna(x) else 0
            )
            enriched["correct_payg"] = "calculated from ATO fortnightly tax brackets"

        # 5. Fill missing categoricals with sensible defaults
        for col, default in [
            ("state",            "Unknown"),
            ("department",       "Unknown"),
            ("employment_type",  "Full-time"),  # safest default for SG eligibility
        ]:
            if col not in df.columns:
                # Try to infer department from Job_Title if available
                if col == "department":
                    job_col = next((c for c in df.columns if c.lower() in
                                    ["job_title","jobtitle","position","role","title"]), None)
                    if job_col:
                        df["department"] = df[job_col]
                        enriched["department"] = f"inferred from {job_col}"
                        continue
                df[col] = default
                enriched[col] = f"defaulted to '{default}' (not in source data)"

        if enriched:
            AUDIT.log_transform("data_validator", "column_enrichment", {
                "columns_enriched": enriched,
                "note": "Missing canonical columns computed or defaulted — "
                        "calculated figures (expected_sg, correct_payg) use simplified formulas, "
                        "see rule entries for limitations",
            })
            self._log(f"      ✓ Enriched {len(enriched)} column(s): "
                      + ", ".join(f"{k} ({v})" for k, v in list(enriched.items())[:3])
                      + ("..." if len(enriched) > 3 else ""))

        return df

    # ── Schema mapper ──────────────────────────────────────────────────────────

    def _map_schema(self, df: pd.DataFrame) -> dict:
        """
        Sends incoming column names + sample values to Claude.
        Claude returns a JSON mapping of incoming → canonical names.
        """
        # Build a sample of each column to help Claude understand content
        sample = {}
        for col in df.columns:
            vals = df[col].dropna().head(3).tolist()
            sample[col] = [str(v) for v in vals]

        canonical_desc = "\n".join(
            f'  "{k}": {v}' for k, v in CANONICAL_SCHEMA.items()
        )

        prompt = f"""
Incoming columns and sample values:
{json.dumps(sample, indent=2)}

Map these to the canonical schema.
"""
        system = MAPPER_PROMPT.format(canonical=canonical_desc)

        raw = self.call(system=system, user_message=prompt, max_tokens=1024)

        try:
            clean = raw.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            # Fallback: assume columns already match canonical schema
            self._log("      ⚠️  Schema mapping parse failed — assuming canonical schema")
            return {
                "mapping":    {col: col for col in df.columns},
                "confidence": "LOW",
            }
