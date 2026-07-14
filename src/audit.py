"""
audit.py
Central audit trail for the payroll compliance pipeline — ZERO API tokens.

Every calculation, rule application, LLM decision, and data transformation
is recorded with: sequence number, timestamp, formula, inputs, outputs,
regulatory basis, and row-level evidence.

Output: reports/audit_TIMESTAMP.json — a machine-readable record that answers
"how was this figure calculated?" for every number in the findings report.

Design: module-level singleton (AUDIT) so any function anywhere in the
pipeline can log without parameter threading. Same pattern as TOKEN_LEDGER.
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path


class AuditTrail:

    def __init__(self):
        self._reset()

    def _reset(self):
        self.run_id     = None
        self.started_at = None
        self.metadata   = {}
        self.events     = []
        self._seq       = 0

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start_run(self, input_file: str, df) -> str:
        """Begin a new audit run. Hashes the input file for integrity."""
        self._reset()
        now = datetime.now(timezone.utc)
        self.run_id     = now.strftime("%Y%m%d_%H%M%S")
        self.started_at = now.isoformat()

        file_hash = "unavailable"
        try:
            with open(input_file, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            pass

        self.metadata = {
            "run_id":          self.run_id,
            "started_at_utc":  self.started_at,
            "input_file":      str(input_file),
            "input_sha256":    file_hash,
            "input_rows":      int(len(df)),
            "input_columns":   list(df.columns),
            "pipeline_version": "v11-awards-audit",
        }
        self._log("PIPELINE", "orchestrator", "run_started", self.metadata)
        return self.run_id

    # ── Core logger ────────────────────────────────────────────────────────────

    def _log(self, category: str, agent: str, action: str, detail: dict):
        self._seq += 1
        self.events.append({
            "seq":       self._seq,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "category":  category,
            "agent":     agent,
            "action":    action,
            "detail":    detail,
        })

    # ── Public logging methods ─────────────────────────────────────────────────

    def log_calculation(
        self,
        agent: str,
        name: str,
        formula: str,
        inputs: dict,
        result,
        regulatory_basis: str = "",
        evidence: list = None,
    ):
        """
        Record a calculation: WHAT was computed, HOW (formula), FROM WHAT
        (inputs), the RESULT, WHY (regulatory basis), and row-level EVIDENCE.
        """
        self._log("CALCULATION", agent, name, {
            "formula":          formula,
            "inputs":           inputs,
            "result":           result,
            "regulatory_basis": regulatory_basis,
            "evidence_sample":  (evidence or [])[:10],   # cap evidence at 10 rows
            "evidence_note":    f"{len(evidence)} evidence rows total, first 10 shown"
                                if evidence and len(evidence) > 10 else None,
        })

    def log_rule(self, agent: str, rule: str, threshold, rationale: str):
        """Record a rule/threshold decision (e.g. 15% PAYG deviation cutoff)."""
        self._log("RULE_APPLIED", agent, rule, {
            "threshold": threshold,
            "rationale": rationale,
        })

    def log_llm_decision(
        self,
        agent: str,
        model: str,
        decision_type: str,
        decision,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        """
        Record a decision made BY the LLM (mapping, classification, scoring).
        These are the entries a reviewer scrutinises hardest — they are the
        non-deterministic part of the pipeline.
        """
        self._log("LLM_DECISION", agent, decision_type, {
            "model":         model,
            "decision":      decision,
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "note": "Non-deterministic step — decision recorded verbatim for review",
        })

    def log_transform(self, agent: str, action: str, detail: dict):
        """Record a data transformation (rename, merge, enrichment, default fill)."""
        self._log("DATA_TRANSFORM", agent, action, detail)

    def log_agent_boundary(self, agent: str, direction: str, summary: dict):
        """Record what entered/left an agent (handoff evidence)."""
        self._log("AGENT_HANDOFF", agent, direction, summary)

    # ── Output ─────────────────────────────────────────────────────────────────

    def save(self, output_dir: str = "reports") -> str:
        Path(output_dir).mkdir(exist_ok=True)
        path = f"{output_dir}/audit_{self.run_id}.json"
        payload = {
            "metadata": self.metadata,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_events": len(self.events),
            "events_by_category": self._category_counts(),
            "events": self.events,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        return path

    def _category_counts(self) -> dict:
        counts = {}
        for e in self.events:
            counts[e["category"]] = counts.get(e["category"], 0) + 1
        return counts

    def summary(self) -> str:
        counts = self._category_counts()
        lines = [f"  Audit events: {len(self.events)}"]
        for cat, n in sorted(counts.items()):
            lines.append(f"   → {cat}: {n}")
        return "\n".join(lines)


# Module-level singleton — import and log from anywhere
AUDIT = AuditTrail()
