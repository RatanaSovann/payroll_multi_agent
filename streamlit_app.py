"""
streamlit_app.py
Payroll Compliance AI — public demo front-end.

Upload any Australian payroll CSV → 6-agent pipeline → dashboard, findings
report, and full audit trail.

Rate limiting (protects the demo API key):
  - 2 runs per browser session
  - Global daily cap across ALL users (hard budget protection)
"""

import os
import io
import json
import random
import tempfile
from datetime import date, timedelta, datetime

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import anthropic
from faker import Faker

from src.agents import (
    DataValidatorAgent,
    AwardClassifierAgent,
    ComplianceAnalystAgent,
    RiskAssessorAgent,
    ReportWriterAgent,
    DashboardGeneratorAgent,
)
from src.agents.base_agent import reset_token_ledger, get_token_summary
from src.audit import AUDIT

# ── Limits ────────────────────────────────────────────────────────────────────

MAX_RUNS_PER_SESSION = 2      # per browser session
MAX_RUNS_PER_DAY     = 25     # global, all users — the real wallet protection
MAX_UPLOAD_ROWS      = 5000   # keeps runs fast and cheap

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Payroll Compliance AI 🇦🇺", page_icon="🇦🇺",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
  #MainMenu, footer { visibility: hidden; }
  .hero { background: linear-gradient(135deg,#0d1117,#1a1f35);
          border:1px solid #30363d; border-radius:16px;
          padding:44px 36px; text-align:center; margin-bottom:28px; }
  .hero-title { font-size:2.3em; font-weight:700;
          background:linear-gradient(90deg,#90caf9,#64b5f6);
          -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
  .hero-sub { color:#8b949e; max-width:640px; margin:12px auto 20px; line-height:1.6; }
  .badge-row { display:flex; gap:8px; justify-content:center; flex-wrap:wrap; }
  .badge { background:#161b22; border:1px solid #30363d; border-radius:20px;
           padding:4px 12px; font-size:0.78em; color:#8b949e; }
  .quota { background:#161b22; border:1px solid #30363d; border-radius:8px;
           padding:8px 14px; font-size:0.82em; color:#8b949e; display:inline-block; }
  .privacy { background:#0d1f0f; border:1px solid #1b4332; border-radius:8px;
             padding:10px 14px; font-size:0.8em; color:#6ee7b7; margin-top:10px; }
</style>
""", unsafe_allow_html=True)


# ── Rate limiting ─────────────────────────────────────────────────────────────

@st.cache_resource
def _global_usage():
    """Shared across ALL sessions on this server instance."""
    return {"date": str(date.today()), "runs": 0}


def check_quota() -> tuple[bool, str]:
    g = _global_usage()
    if g["date"] != str(date.today()):        # daily reset
        g["date"], g["runs"] = str(date.today()), 0

    session_used = st.session_state.get("runs_used", 0)
    if session_used >= MAX_RUNS_PER_SESSION:
        return False, (f"You've used your {MAX_RUNS_PER_SESSION} runs for this session. "
                       "This is a personal demo running on my own API credits — thanks for trying it! "
                       "The full source is on GitHub if you'd like to run it with your own key.")
    if g["runs"] >= MAX_RUNS_PER_DAY:
        return False, ("The demo has hit its daily budget cap across all users. "
                       "It resets tomorrow — or grab the code from GitHub and run it locally.")
    return True, ""


def consume_quota():
    st.session_state["runs_used"] = st.session_state.get("runs_used", 0) + 1
    _global_usage()["runs"] += 1


# ── API key ───────────────────────────────────────────────────────────────────

def get_api_key() -> str:
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.getenv("ANTHROPIC_API_KEY", "")


# ── Sample data ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def generate_sample_data() -> pd.DataFrame:
    SEED = 42
    random.seed(SEED); np.random.seed(SEED)
    fake = Faker("en_AU"); Faker.seed(SEED)

    STATES = ["NSW","VIC","QLD","WA","SA","TAS","ACT","NT"]
    TITLES = {"Retail Assistant":"Casual","Store Manager":"Full-time",
              "Payroll Officer":"Full-time","Software Engineer":"Full-time",
              "Data Analyst":"Full-time","Chef":"Part-time","Waiter":"Casual",
              "Machine Operator":"Full-time"}
    WAGES  = {"Full-time":(3000,8500),"Part-time":(1500,4000),"Casual":(500,2800)}

    def payg(g):
        a = g*26
        r = 0.0 if a<=18200 else 0.19 if a<=45000 else 0.325 if a<=120000 else 0.37 if a<=180000 else 0.45
        return round(g*r, 2)

    emps = []
    for i in range(1, 101):
        title = random.choice(list(TITLES))
        et    = TITLES[title]
        emps.append({"employee_id": f"EMP{i:04d}", "name": fake.name(),
                     "state": random.choice(STATES), "job_title": title,
                     "department": title.split()[0], "employment_type": et,
                     "base": round(random.uniform(*WAGES[et]), 2)})
    ids        = [e["employee_id"] for e in emps]
    casuals    = [e["employee_id"] for e in emps if e["employment_type"]=="Casual"]
    sg_bad     = set(random.sample(ids, 14))
    casual_bad = set(random.sample(casuals, max(1, len(casuals)//3)))
    payg_bad   = set(random.sample(ids, 10))
    low_rate   = set(random.sample(ids, 8))     # award underpayment seed

    start = date(2026, 1, 1)
    rows  = []
    for e in emps:
        for p in range(6):
            gross = round(e["base"] * random.uniform(0.95, 1.05), 2)
            if e["employee_id"] in low_rate:
                gross = round(gross * 0.62, 2)  # push effective rate below award minimum
            exp_sg = round(gross * 0.12, 2)
            sup    = 0.0 if e["employee_id"] in casual_bad else \
                     round(gross * random.uniform(0.04, 0.09), 2) if e["employee_id"] in sg_bad else exp_sg
            cp     = payg(gross)
            pw     = round(cp * random.uniform(0.6, 1.4), 2) if e["employee_id"] in payg_bad else cp
            rows.append({
                "employee_id": e["employee_id"], "name": e["name"], "state": e["state"],
                "job_title": e["job_title"], "department": e["department"],
                "employment_type": e["employment_type"],
                "pay_date": (start + timedelta(weeks=2*p, days=15)).isoformat(),
                "hours_worked": 76 if e["employment_type"]=="Full-time" else random.randint(20, 60),
                "gross_wage": gross, "super_paid": sup, "expected_sg": exp_sg,
                "payg_withheld": pw, "correct_payg": cp,
            })
    return pd.DataFrame(rows)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(df: pd.DataFrame, api_key: str, source_name: str) -> dict:
    client = anthropic.Anthropic(api_key=api_key)
    reset_token_ledger()

    # Audit needs a file path for hashing — write upload to temp
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        AUDIT.start_run(f.name, df)
    AUDIT.metadata["input_file"] = source_name   # friendlier name than temp path

    steps = st.session_state["steps"]

    def mark(i, detail=""):
        steps[i]["status"], steps[i]["detail"] = "done", detail
        _draw_progress()

    def running(i):
        steps[i]["status"] = "running"; _draw_progress()

    running(0)
    validation, mapped_df = DataValidatorAgent(client, verbose=False).run(df)
    mark(0, f"{validation['stats']['row_count']:,} rows · confidence {validation['mapping_confidence']}")

    running(1)
    classification = AwardClassifierAgent(client, verbose=False).run(mapped_df)
    covered = sum(1 for c in classification.values() if c.get("award_code") not in ("UNKNOWN",))
    mark(1, f"{covered}/{len(classification)} job titles classified to awards")

    running(2)
    compliance = ComplianceAnalystAgent(client, verbose=False).run(
        mapped_df, classification_map=classification)
    total = compliance.get("summary", {}).get("total_exposure_aud", 0)
    mark(2, f"Total exposure: ${total:,.2f}")

    running(3)
    risk = RiskAssessorAgent(client, verbose=False).run(compliance)
    mark(3, f"Overall risk: {risk.get('overall_risk_rating','N/A')}")

    running(4)
    report = ReportWriterAgent(client, verbose=False).run(
        validation=validation, compliance=compliance, risk=risk)
    mark(4, "Findings report drafted")

    running(5)
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        dash_path = f.name
    DashboardGeneratorAgent().run(compliance=compliance, risk=risk,
                                  df=mapped_df, output_path=dash_path)
    with open(dash_path, encoding="utf-8") as f:
        dashboard_html = f.read()
    mark(5, "Interactive dashboard generated")

    audit_payload = {
        "metadata": AUDIT.metadata,
        "total_events": len(AUDIT.events),
        "events": AUDIT.events,
    }

    return {"validation": validation, "classification": classification,
            "compliance": compliance, "risk": risk, "report": report,
            "dashboard_html": dashboard_html, "audit": audit_payload,
            "tokens": get_token_summary()}


def _draw_progress():
    ph = st.session_state.get("progress_ph")
    if not ph:
        return
    icons = {"waiting": "⏳", "running": "🔄", "done": "✅"}
    lines = []
    for s in st.session_state["steps"]:
        line = f"{icons[s['status']]} **{s['label']}**"
        if s.get("detail"):
            line += f" — {s['detail']}"
        lines.append(line)
    ph.markdown("\n\n".join(lines))


# ── UI ────────────────────────────────────────────────────────────────────────

def main():
    st.markdown("""
    <div class="hero">
      <div class="hero-title">🇦🇺 Payroll Compliance AI</div>
      <div class="hero-sub">
        Six AI agents analyse any Australian payroll CSV for Super Guarantee,
        PAYG, casual super, and Modern Award compliance — then produce a findings
        report, interactive dashboard, and a full calculation audit trail.
      </div>
      <div class="badge-row">
        <div class="badge">🔍 SG Underpayment</div>
        <div class="badge">📊 PAYG Deviations</div>
        <div class="badge">👷 Casual Super</div>
        <div class="badge">⚖️ Modern Award Rates (FWC 2026)</div>
        <div class="badge">🧾 Full Audit Trail</div>
        <div class="badge">🗺️ Any CSV Format</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    api_key = get_api_key()
    if not api_key:
        st.error("Demo API key not configured. If you're running locally, add "
                 "ANTHROPIC_API_KEY to .streamlit/secrets.toml")
        st.stop()

    # Quota display
    used = st.session_state.get("runs_used", 0)
    st.markdown(f'<div class="quota">🎟️ Demo quota: <b>{MAX_RUNS_PER_SESSION - used}</b> of '
                f'{MAX_RUNS_PER_SESSION} runs remaining this session</div>',
                unsafe_allow_html=True)

    # ── Data input ────────────────────────────────────────────────────────────
    col_up, col_sample = st.columns([2, 1], gap="large")
    with col_up:
        uploaded = st.file_uploader("Upload a payroll CSV (any format — columns are auto-mapped)",
                                    type=["csv"])
        st.markdown('<div class="privacy">🔒 Only column names, sample values, and aggregated '
                    'summaries reach the AI — individual pay records stay in this session. '
                    'Please use synthetic or anonymised data for this public demo.</div>',
                    unsafe_allow_html=True)
    with col_sample:
        st.markdown("**No file? Use sample data**")
        st.caption("100 employees · 6 pay periods · SG, PAYG, casual-super and award breaches seeded")
        if st.button("🎲 Load Sample Dataset", use_container_width=True):
            st.session_state["sample_df"] = generate_sample_data()
        if "sample_df" in st.session_state:
            st.download_button("⬇️ Download sample CSV",
                st.session_state["sample_df"].to_csv(index=False).encode(),
                "sample_payroll.csv", "text/csv", use_container_width=True)

    df, source = None, ""
    if uploaded:
        df, source = pd.read_csv(uploaded), uploaded.name
    elif "sample_df" in st.session_state:
        df, source = st.session_state["sample_df"], "sample_payroll.csv"

    if df is None:
        st.stop()

    if len(df) > MAX_UPLOAD_ROWS:
        st.warning(f"File truncated to first {MAX_UPLOAD_ROWS:,} rows (demo limit).")
        df = df.head(MAX_UPLOAD_ROWS)

    st.success(f"✅ **{source}** · {len(df):,} rows · {len(df.columns)} columns")
    with st.expander("Preview"):
        st.dataframe(df.head(8), use_container_width=True)

    # ── Run ───────────────────────────────────────────────────────────────────
    ok, msg = check_quota()
    run = st.button("🚀 Run Compliance Analysis", type="primary", disabled=not ok)
    if not ok:
        st.warning(msg)

    if run and ok:
        consume_quota()
        st.session_state["steps"] = [
            {"label": "Agent 1 — Data Validator (schema map + enrichment)", "status": "waiting", "detail": ""},
            {"label": "Agent 1.5 — Award Classifier (unique job titles → awards)", "status": "waiting", "detail": ""},
            {"label": "Agent 2 — Compliance Analyst (5 checks via tool loop)", "status": "waiting", "detail": ""},
            {"label": "Agent 3 — Risk Assessor (rubric scoring + SGC penalty)", "status": "waiting", "detail": ""},
            {"label": "Agent 4 — Report Writer (client findings report)", "status": "waiting", "detail": ""},
            {"label": "Agent 5 — Dashboard Generator (zero-token Plotly)", "status": "waiting", "detail": ""},
        ]
        st.session_state["progress_ph"] = st.empty()
        _draw_progress()
        try:
            st.session_state["results"] = run_pipeline(df, api_key, source)
        except anthropic.AuthenticationError:
            st.error("API authentication failed — the demo key may be misconfigured.")
            st.stop()
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

    r = st.session_state.get("results")
    if not r:
        st.stop()

    # ── Results ───────────────────────────────────────────────────────────────
    st.divider()
    summary  = r["compliance"].get("summary", {})
    sg       = r["compliance"].get("sg", {})
    payg_c   = r["compliance"].get("payg", {})
    awards   = r["compliance"].get("awards", {})
    risk     = r["risk"]
    overall  = risk.get("overall_risk_rating", "N/A")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Total Exposure", f"${summary.get('total_exposure_aud',0):,.0f}")
    c2.metric("📉 SG Shortfall", f"${sg.get('total_shortfall_aud',0):,.0f}",
              f"{sg.get('employees_affected',0)} employees")
    c3.metric("📋 PAYG Exposure", f"${payg_c.get('total_exposure_aud',0):,.0f}")
    c4.metric("⚖️ Award Underpayment", f"${awards.get('total_underpayment_aud',0):,.0f}")
    c5.metric("🎯 Overall Risk", overall)

    if risk.get("narrative"):
        st.info(f"**Risk Assessment:** {risk['narrative']}")
    if risk.get("highest_priority_action"):
        st.warning(f"⚡ **Priority:** {risk['highest_priority_action']}")

    cov = awards.get("coverage", {})
    if cov.get("titles_unknown"):
        st.caption(f"⚠️ Award coverage: {cov.get('titles_covered',0)}/{cov.get('titles_total',0)} "
                   f"job titles assessable. Unknown (not guessed): {', '.join(cov['titles_unknown'])}")

    tokens = r["tokens"]
    st.caption(f"🔢 This run: {tokens.get('calls',0)} API calls · "
               f"{tokens.get('input',0):,} in / {tokens.get('output',0):,} out tokens · "
               f"~${tokens.get('cost_usd',0):.4f} USD")

    tab_dash, tab_report, tab_audit = st.tabs(
        ["📊 Dashboard", "📄 Findings Report", "🧾 Audit Trail"])

    with tab_dash:
        components.html(r["dashboard_html"], height=900, scrolling=True)

    with tab_report:
        st.markdown(r["report"])

    with tab_audit:
        audit = r["audit"]
        st.markdown(f"**Run ID:** `{audit['metadata'].get('run_id')}` · "
                    f"**Input SHA-256:** `{audit['metadata'].get('input_sha256','')[:16]}…` · "
                    f"**{audit['total_events']} events**")
        st.caption("Every figure in the report traces back to a CALCULATION event below — "
                   "formula, inputs, regulatory basis, and row-level evidence. "
                   "LLM decisions (non-deterministic steps) are recorded verbatim.")

        categories = sorted({e["category"] for e in audit["events"]})
        selected = st.multiselect("Filter by category", categories, default=categories)
        for e in audit["events"]:
            if e["category"] not in selected:
                continue
            icon = {"CALCULATION":"🧮","LLM_DECISION":"🤖","RULE_APPLIED":"📐",
                    "DATA_TRANSFORM":"🔧","PIPELINE":"🏁","AGENT_HANDOFF":"🤝"}.get(e["category"], "•")
            with st.expander(f"{icon} #{e['seq']} · {e['category']} · {e['agent']} · {e['action']}"):
                st.json(e["detail"])

    # Downloads
    st.divider()
    d1, d2, d3 = st.columns(3)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d1.download_button("⬇️ Findings Report (.md)", r["report"].encode("utf-8"),
                       f"findings_{ts}.md", "text/markdown", use_container_width=True)
    d2.download_button("⬇️ Dashboard (.html)", r["dashboard_html"].encode("utf-8"),
                       f"dashboard_{ts}.html", "text/html", use_container_width=True)
    d3.download_button("⬇️ Audit Trail (.json)",
                       json.dumps(r["audit"], indent=2, default=str).encode("utf-8"),
                       f"audit_{ts}.json", "application/json", use_container_width=True)

    st.markdown("""<div style="text-align:center;padding:24px 0 4px;font-size:0.78em;color:#484f58">
      Built by <b>Ratana</b> · 6-agent pipeline · Claude (Anthropic) + Python + Plotly ·
      Synthetic data demo — not tax advice · Source on GitHub
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
