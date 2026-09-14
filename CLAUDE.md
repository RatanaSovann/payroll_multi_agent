# CLAUDE.md

Australian payroll compliance pipeline. A payroll CSV in any vendor format goes in;
a findings report, an interactive dashboard, and a machine-readable audit trail come out.

## Commands

```bash
./.venv/Scripts/python.exe orchestrator.py --file data/raw/payroll_data.csv   # headless CLI
./.venv/Scripts/python.exe -m streamlit run streamlit_app.py                  # demo UI
```

Always use `./.venv/Scripts/python.exe`. The conda base env on this machine has
pandas 2; the venv has pandas 3, which is what the code is written against.

Secrets: `ANTHROPIC_API_KEY` from `.env` (CLI, via `load_dotenv`) or
`.streamlit/secrets.toml` (Streamlit). Both are gitignored.

## Pipeline

`orchestrator.run_pipeline()` is the single entry point. Six stages, in order:

| # | Agent | Model | Job |
|---|---|---|---|
| 1 | `DataValidatorAgent` | Haiku | Map arbitrary columns → canonical schema; `clean_dataframe`; enrich missing fields |
| 2 | `AwardClassifierAgent` | Haiku | Classify **unique** job titles only (flat token cost regardless of row count) |
| 3 | `ComplianceAnalystAgent` | Sonnet | The only agentic tool loop — Claude orchestrates the `src/compliance.py` + `src/award_engine.py` tools |
| 4 | `RiskAssessorAgent` | Haiku | HIGH/MED/LOW via explicit rubric; SGC penalty estimate |
| 5 | `ReportWriterAgent` | Sonnet | `reports/findings_<ts>.md` |
| 6 | `DashboardGeneratorAgent` | none | Pure Plotly → `reports/dashboard_<ts>.html` |

Then `AUDIT.save()` → `reports/audit_<run_id>.json`.

## Invariants — do not break these

1. **Canonical schema is the contract.** `CANONICAL_SCHEMA` in
   `src/agents/data_validator.py` (11 columns). Format chaos is quarantined in
   Agent 1; everything downstream is written once against those names. Adding a
   column means updating that dict *and* the enrichment step that fills it.
2. **Capture at execution.** Tool results are seized in Python the moment the
   tool runs (`compliance_analyst.py`), never parsed back out of model prose. A
   number that reaches the report must have come from a Python return value.
3. **LLM only where reasoning or language is needed.** Rates, PAYG math, charts,
   and orchestration are deterministic Python. Do not move arithmetic into a prompt.
4. **"Unknown" is a valid answer.** Unmappable columns, unclassifiable titles and
   uncovered awards are reported as unknown, never guessed.
5. **Everything material is audited.** `AUDIT` (module singleton in `src/audit.py`)
   is imported directly wherever it's needed — no parameter threading. Same pattern
   as `TOKEN_LEDGER` in `base_agent.py`. New calculations get a `log_calculation`
   with formula, inputs, result, regulatory basis and evidence rows.

## Layout

```
orchestrator.py            CLI entry + pipeline sequencing
streamlit_app.py           public demo UI (in-memory rate limit, 5k-row cap)
src/audit.py               AuditTrail singleton AUDIT — zero tokens
src/compliance.py          SG / PAYG / casual-super checks (the agent's tools)
src/award_engine.py        AWARD_RATES (FWC 2026) + underpayment math
src/data_cleaner.py        clean_dataframe — dedupe, coercion, flagging
src/agents/base_agent.py   BaseAgent.call / .run_loop, PRICING, TOKEN_LEDGER
src/agents/*.py            the six agents
```

## Domain constants

- SG rate 12%, 0.1% rounding tolerance (SGAA 1992).
- PAYG flag threshold: >15% deviation from bracket amount.
- Award rates: FWC 2026 AWR (+4.75%), casual loading 25%. Five awards + NMW covered
  out of 121+; penalty multipliers are stored but not yet applied.
- `correct_payg` fallback uses flat marginal brackets — no progressive calc, LITO or
  Medicare. It produces false "under-withheld" flags on real data; the audit trail
  records when the computed fallback was used.

## Current state

Branch `eval-harness-and-fixes`. In-flight, uncommitted: Langfuse tracing
(`@observe` decorators on `BaseAgent.call` / `run_loop`).

- `orchestrator.py` is **currently broken**: it imports
  `from langfuse.anthropic import Anthropic` but still calls `anthropic.Anthropic(...)`.
- `langfuse` is installed (4.15.2) but is not in `requirements.txt`.
- A golden-dataset eval harness was added in `5056757` and removed again in `571c793`.
