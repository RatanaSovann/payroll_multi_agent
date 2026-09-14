# Payroll Compliance AI 🇦🇺

**Six AI agents. Any payroll CSV in → findings report, interactive dashboard, and a full calculation audit trail out. ~90 seconds. ~$0.20 per run.**

🔗 **Live demo:** [https://payroll-compliance-ai.streamlit.app/](https://payroll-compliance-ai.streamlit.app/) *(2 runs per session — it runs on my personal API credits)*

![agents](https://img.shields.io/badge/agents-6-blue)
![cost](https://img.shields.io/badge/cost%2Frun-~$0.20-green)
![rates](https://img.shields.io/badge/FWC%20rates-1%20July%202026-orange)
![python](https://img.shields.io/badge/python-3.11+-yellow)

---

## Why this exists

Australian wage underpayment has cost employers hundreds of millions in back-pay (Woolworths $300M+, Qantas, and dozens more). From **1 July 2026, Payday Super** ties super contributions to every payday — turning each pay run into a potential compliance event instead of each quarter.

The traditional review — a consultant manually mapping columns, writing queries, checking ATO rules, drafting findings — takes days. This pipeline does it in minutes, and shows its working.

## What it checks

| Check | Rule applied | Basis |
|---|---|---|
| **SG underpayment** | `super_paid < gross × 12%` (0.1% rounding tolerance) | SGAA 1992, FY25–26 rate |
| **PAYG inconsistency** | >15% deviation from bracket amount | ATO withholding schedules |
| **Missing casual super** | Casuals with zero super | $450/mo threshold removed Jul 2022 |
| **Modern Award underpayment** | Effective hourly rate vs award minimum | FWC 2026 AWR (+4.75%); casual loading 25% |

## Architecture

```
payroll.csv (any format — Xero, MYOB, SAP, custom)
   │
   ▼
1  DATA VALIDATOR ····· Haiku ··· maps any columns → canonical schema;
   │                              merges split names; computes missing fields
   ▼
1.5 AWARD CLASSIFIER ·· Haiku ··· UNIQUE job titles only (15 strings, not 900
   │                              rows) → flat token cost at any dataset size
   ▼
2  COMPLIANCE ANALYST · Sonnet ·· the only agentic loop: Claude orchestrates
   │                              6 Python tools; results captured in Python
   ▼                              at execution — never parsed from prose
3  RISK ASSESSOR ······ Haiku ··· HIGH/MED/LOW via explicit rubric;
   │                              SGC penalty estimate (10% p.a. + admin fees)
   ├────────────────┐
   ▼                ▼
4  REPORT WRITER   5  DASHBOARD GENERATOR
   Sonnet             no model — pure Plotly
   │                  │
   ▼                  ▼
findings.md      dashboard.html      + audit.json
```

### Six design principles

1. **LLM only where reasoning or language is needed.** Award rates, PAYG math, charts, orchestration = deterministic Python. Zero tokens, zero hallucination surface.
2. **Canonical schema as contract.** Format chaos is quarantined in Agent 1; everything downstream is written once against 11 stable columns.
3. **Capture at execution.** Tool results are seized in Python the moment they run. Model prose is never the source of truth for a number.
4. **"Unknown" is a valid answer.** Unmappable columns, unclassifiable titles, uncovered awards are reported — never guessed. The compliance-grade stance.
5. **Model tiering.** Haiku for structured tasks, Sonnet only where orchestration reliability or prose quality pays for itself. ~$0.05/run vs $3–5 naive.
6. **Audit everything** ↓

### The audit trail (the differentiator)

Every run emits `audit_<run>.json`: every **formula + inputs + result + regulatory basis (Act & section)**, row-level evidence, every threshold with its rationale, and every **LLM decision recorded verbatim** and flagged as the non-deterministic step.

Chain of evidence: `report figure → CALCULATION event → formula → inputs → evidence rows`. Every figure can be recomputed by hand from the record. "The AI said so" is not an answer the ATO accepts — this is the answer it does.

The defensibility does not come from the model. It comes from the fact that the arithmetic is deterministic Python with a statutory citation attached, and from a qualified human who reviews and owns the result. The trail exists to make that review fast enough to be worth doing.

### Pinned decisions — the audit file is also an input

The pipeline has exactly **two** non-deterministic steps: the schema mapping (Agent 1) and the award classification (Agent 1.5). Everything downstream is deterministic. Pass an earlier run's audit file back in and those two decisions are replayed instead of re-made:

```bash
python orchestrator.py --file payroll.csv                            # fresh — the model decides
python orchestrator.py --file payroll.csv --pin reports/audit_X.json # replay — the file decides
```

This is what makes expert review actionable. If Haiku classifies `Store Manager` as award-free when they are covered by General Retail, a reviewer edits that one entry in the audit JSON:

```json
"Store Manager": {"award_code": "MA000004", "level": 5, "confidence": "HIGH",
                  "rationale": "reviewer override — General Retail covers store management"}
```

Re-run with `--pin` and every downstream figure — award underpayment, total exposure, risk rating, dashboard, report — recomputes from the corrected classification. The model is not consulted for that step, and the trail records the decision as replayed rather than freshly generated, so a later reader can always tell the two apart.

Each classification also carries a **`rationale`**: the award clause or duty that justifies the code and level. It is review scaffolding, not evidence — its value is that a human can check it against the award text, not that the model produced it.

Two honest limits: pinning fixes **figures**, not prose — the narrative and the final report are still generated fresh each run. And replay is refused outright if the pinned run's columns don't match the current input, because a mapping replayed onto a different schema produces figures that are silently wrong.

## Run locally

```bash
git clone https://github.com/YOUR-USERNAME/payroll-compliance-ai
cd payroll-compliance-ai && pip install -r requirements.txt
echo 'ANTHROPIC_API_KEY = "sk-ant-..."' > .streamlit/secrets.toml
streamlit run streamlit_app.py            # web UI
python orchestrator.py --file your.csv    # headless CLI
python orchestrator.py --file your.csv --pin reports/audit_X.json   # deterministic re-run
```

## Current limitations (deliberate transparency)

| Limitation | Detail | Path forward |
|---|---|---|
| **Simplified PAYG fallback** | When the CSV lacks `correct_payg`, it's computed from flat marginal brackets — no progressive calc, LITO, or Medicare → false "under-withheld" flags on real data. Audit trail flags when the computed value was used | Implement full progressive ATO scales + offsets (~25 lines, zero tokens) |
| **Award coverage: 5 of 121+** | Clerks, Retail, Hospitality, Manufacturing, Professional + NMW; base rates only — no penalty rates/allowances applied yet (multipliers already stored) | Data problem, not architecture: add rate tables; wire penalty multipliers when hours-by-day data exists |
| **Titles outside those 5 report UNKNOWN** | Nurses, teachers, construction, transport, aged care and others have their own awards with no rate table here, so they are reported uncovered rather than force-fitted. The audit splits uncovered titles into `titles_out_of_scope` (classified, no rates held) and `titles_unclassified` (never reached the classifier) | Add rate tables; the split tells you which of the two you are actually hitting |
| **Award check needs `job_title`** | Without one the classifier falls back to `department`, and business units ("Finance", "Operations") do not classify into awards. The mapper now maps `Position`/`Role`/`Occupation` to `job_title`, and the model is told explicitly when it is seeing departments | Supply a job title column; `title_column_used` in the audit records which was used |
| **Free-tier hosting** | App sleeps after inactivity (~60s cold start); in-memory rate limits reset on restart; 1GB RAM → 5,000-row cap | Containerise + persistent store (Redis) for production-grade limits |
| **No retry/backoff** | A mid-pipeline API failure aborts the run | Add exponential backoff + per-agent checkpointing |
| **Not tax advice** | A qualified human must review and own every finding | That's a feature — the audit trail exists to make that review fast |

## Potential

The architecture is domain-portable: swap the tools and prompts and the same 6-agent skeleton audits WHS compliance, financial services obligations, or modern-award classification reviews. The interesting shift it demonstrates is **AI-assistant → AI-analyst**: not a chatbot answering questions, but a pipeline completing a multi-step professional workflow with evidence attached.

## Repo map

```
streamlit_app.py          # rate-limited public demo UI
orchestrator.py           # CLI entry
src/audit.py              # audit trail + decision pinning (zero tokens)
src/compliance.py         # SG / PAYG / casual checks (agent tools)
src/award_engine.py       # FWC 2026 rates + underpayment math
src/agents/               # the six agents + base loop / token ledger
```

---

*Built by **Ratana Sovann** . Synthetic data only; rates current at 1 July 2026.*
