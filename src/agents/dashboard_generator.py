"""
dashboard_generator.py — Agent 5
Generates a self-contained interactive HTML dashboard from compliance findings.
Uses Plotly for charts — no Power BI, no dependencies, opens in any browser.

Output: one .html file the client can open, share, or embed anywhere.
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path


# Colour palette — professional, compliance-appropriate
COLOURS = {
    "high":       "#E74C3C",   # red     — high risk
    "medium":     "#F39C12",   # amber   — medium risk
    "low":        "#27AE60",   # green   — low risk / compliant
    "primary":    "#2C3E50",   # dark    — titles, backgrounds
    "accent":     "#3498DB",   # blue    — bars, lines
    "light":      "#ECF0F1",   # light grey — backgrounds
    "over":       "#E74C3C",   # red     — over-withheld
    "under":      "#F39C12",   # amber   — under-withheld
}

RISK_COLOURS = {
    "HIGH":   COLOURS["high"],
    "MEDIUM": COLOURS["medium"],
    "LOW":    COLOURS["low"],
}


class DashboardGeneratorAgent:
    """
    Agent 5: Generates a self-contained HTML dashboard.
    Takes compliance findings + risk assessment as input.
    No API call needed — this agent is pure Python/Plotly logic.
    """

    def run(
        self,
        compliance:  dict,
        risk:        dict,
        df:          pd.DataFrame,
        output_path: str = None,
    ) -> str:
        """
        Builds and saves the HTML dashboard.
        Returns the path to the saved file.
        """
        timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path  = output_path or f"reports/dashboard_{timestamp}.html"
        Path(output_path).parent.mkdir(exist_ok=True)

        # Build all chart sections
        html = self._build_html(compliance, risk, df, timestamp)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return output_path

    # ── HTML assembly ──────────────────────────────────────────────────────────

    def _build_html(self, compliance, risk, df, timestamp):
        summary      = compliance.get("summary", {})
        sg           = compliance.get("sg", {})
        payg         = compliance.get("payg", {})
        casuals      = compliance.get("casuals", {})
        risk_scores  = risk.get("risk_scores", {})
        overall      = risk.get("overall_risk_rating", "N/A")
        total_exp    = summary.get("total_exposure_aud", 0)
        sgc_penalty  = risk.get("estimated_sgc_penalty", 0)

        # Build individual chart HTMLs
        exposure_chart   = self._exposure_by_issue(summary)
        sg_dept_chart    = self._sg_by_department(sg)
        sg_state_chart   = self._sg_by_state(sg)
        payg_split_chart = self._payg_split(payg)
        payg_dept_chart  = self._payg_by_department(payg)
        trend_chart      = self._sg_trend(df)

        # Risk badge colours
        def badge(score):
            c = RISK_COLOURS.get(score, "#999")
            return f'<span style="background:{c};color:white;padding:3px 10px;border-radius:4px;font-weight:bold;font-size:0.85em">{score}</span>'

        overall_colour = RISK_COLOURS.get(overall, "#999")

        sg_score      = risk_scores.get("sg_underpayment", {}).get("score", "N/A")
        payg_score    = risk_scores.get("payg_inconsistency", {}).get("score", "N/A")
        casual_score  = risk_scores.get("missing_casual_super", {}).get("score", "N/A")

        priority      = risk.get("highest_priority_action", "N/A")
        payday_note   = risk.get("payday_super_readiness", "N/A")
        narrative     = risk.get("narrative", "")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Payroll Compliance Dashboard — FY2025-26</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6f9; color: #2C3E50; }}

    .header {{
      background: {COLOURS["primary"]};
      color: white;
      padding: 28px 40px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .header h1 {{ font-size: 1.5em; font-weight: 600; }}
    .header .meta {{ font-size: 0.85em; opacity: 0.75; margin-top: 4px; }}
    .overall-badge {{
      background: {overall_colour};
      color: white;
      padding: 8px 20px;
      border-radius: 6px;
      font-weight: bold;
      font-size: 1em;
      text-align: center;
    }}
    .overall-badge .label {{ font-size: 0.75em; opacity: 0.85; }}

    .container {{ max-width: 1400px; margin: 0 auto; padding: 30px 40px; }}

    /* KPI Cards */
    .kpi-row {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 20px;
      margin-bottom: 30px;
    }}
    .kpi-card {{
      background: white;
      border-radius: 8px;
      padding: 22px 24px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.08);
      border-left: 4px solid {COLOURS["accent"]};
    }}
    .kpi-card.red   {{ border-left-color: {COLOURS["high"]}; }}
    .kpi-card.amber {{ border-left-color: {COLOURS["medium"]}; }}
    .kpi-card.green {{ border-left-color: {COLOURS["low"]}; }}
    .kpi-value {{ font-size: 1.9em; font-weight: 700; color: {COLOURS["primary"]}; }}
    .kpi-label {{ font-size: 0.82em; color: #7f8c8d; margin-top: 4px; }}

    /* Section layout */
    .section {{ margin-bottom: 30px; }}
    .section-title {{
      font-size: 1em;
      font-weight: 600;
      color: {COLOURS["primary"]};
      margin-bottom: 16px;
      padding-bottom: 8px;
      border-bottom: 2px solid {COLOURS["light"]};
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .chart-grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }}
    .chart-grid-3 {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 20px;
    }}
    .card {{
      background: white;
      border-radius: 8px;
      padding: 20px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }}
    .card-full {{ grid-column: 1 / -1; }}

    /* Risk table */
    .risk-table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
    .risk-table th {{
      background: {COLOURS["primary"]};
      color: white;
      padding: 10px 14px;
      text-align: left;
      font-weight: 500;
    }}
    .risk-table td {{ padding: 10px 14px; border-bottom: 1px solid #eee; }}
    .risk-table tr:last-child td {{ border-bottom: none; }}

    /* Priority box */
    .priority-box {{
      background: #FEF9E7;
      border-left: 4px solid {COLOURS["medium"]};
      border-radius: 6px;
      padding: 16px 20px;
      margin-bottom: 20px;
      font-size: 0.92em;
    }}
    .priority-box .label {{ font-weight: 600; margin-bottom: 4px; color: #B7770D; }}

    .payday-box {{
      background: #EBF5FB;
      border-left: 4px solid {COLOURS["accent"]};
      border-radius: 6px;
      padding: 16px 20px;
      font-size: 0.92em;
    }}
    .payday-box .label {{ font-weight: 600; margin-bottom: 4px; color: #1A5276; }}

    .narrative {{
      background: #FDFEFE;
      border: 1px solid #E8E8E8;
      border-radius: 6px;
      padding: 16px 20px;
      font-size: 0.92em;
      color: #555;
      line-height: 1.6;
      margin-bottom: 20px;
    }}

    .footer {{
      text-align: center;
      padding: 20px;
      font-size: 0.78em;
      color: #aaa;
      margin-top: 10px;
    }}
  </style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div>
    <div class="h1">Payroll Compliance Dashboard</div>
    <div class="meta">FY2025–26 · People Advisory Services — Employment Taxes · Generated {datetime.now().strftime("%d %b %Y %H:%M")}</div>
  </div>
  <div class="overall-badge">
    <div class="label">Overall Risk</div>
    {overall}
  </div>
</div>

<div class="container">

  <!-- KPI Cards -->
  <div class="section">
    <div class="kpi-row">
      <div class="kpi-card red">
        <div class="kpi-value">${total_exp:,.0f}</div>
        <div class="kpi-label">Total Compliance Exposure</div>
      </div>
      <div class="kpi-card red">
        <div class="kpi-value">${sg.get('total_shortfall_aud', 0):,.0f}</div>
        <div class="kpi-label">SG Shortfall &nbsp;{badge(sg_score)}</div>
      </div>
      <div class="kpi-card amber">
        <div class="kpi-value">${payg.get('total_exposure_aud', 0):,.0f}</div>
        <div class="kpi-label">PAYG Exposure &nbsp;{badge(payg_score)}</div>
      </div>
      <div class="kpi-card amber">
        <div class="kpi-value">${casuals.get('total_shortfall_aud', 0):,.0f}</div>
        <div class="kpi-label">Casual Super Shortfall &nbsp;{badge(casual_score)}</div>
      </div>
    </div>

    <!-- Second KPI row -->
    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-value">{sg.get('employees_affected', 0)}</div>
        <div class="kpi-label">Employees — SG Issues</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-value">{payg.get('employees_affected', 0)}</div>
        <div class="kpi-label">Employees — PAYG Issues</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-value">{casuals.get('employees_affected', 0)}</div>
        <div class="kpi-label">Casual Employees Affected</div>
      </div>
      <div class="kpi-card amber">
        <div class="kpi-value">${sgc_penalty:,.0f}</div>
        <div class="kpi-label">Est. ATO SGC Penalty</div>
      </div>
    </div>
  </div>

  <!-- Risk narrative -->
  <div class="section">
    <div class="section-title">Risk Assessment</div>
    <div class="narrative">{narrative}</div>

    <div class="priority-box">
      <div class="label">⚡ Highest Priority Action</div>
      {priority}
    </div>

    <div class="payday-box">
      <div class="label">📅 Payday Super Readiness (1 July 2026)</div>
      {payday_note}
    </div>
  </div>

  <!-- Charts row 1 -->
  <div class="section">
    <div class="section-title">Exposure Overview</div>
    <div class="chart-grid-2">
      <div class="card">{exposure_chart}</div>
      <div class="card">{trend_chart}</div>
    </div>
  </div>

  <!-- Charts row 2 — SG detail -->
  <div class="section">
    <div class="section-title">SG Underpayment Detail</div>
    <div class="chart-grid-2">
      <div class="card">{sg_dept_chart}</div>
      <div class="card">{sg_state_chart}</div>
    </div>
  </div>

  <!-- Charts row 3 — PAYG detail -->
  <div class="section">
    <div class="section-title">PAYG Inconsistency Detail</div>
    <div class="chart-grid-2">
      <div class="card">{payg_split_chart}</div>
      <div class="card">{payg_dept_chart}</div>
    </div>
  </div>

  <!-- Risk summary table -->
  <div class="section">
    <div class="section-title">Findings Summary</div>
    <div class="card">
      <table class="risk-table">
        <tr>
          <th>Issue Type</th>
          <th>Employees Affected</th>
          <th>Pay Runs Affected</th>
          <th>Total Exposure</th>
          <th>Risk Rating</th>
        </tr>
        {self._summary_rows(summary, risk_scores)}
      </table>
    </div>
  </div>

</div>

<div class="footer">
  Generated by Payroll Compliance Multi-Agent Pipeline · Synthetic data only · Not for client distribution
</div>

</body>
</html>"""

    # ── Chart builders ─────────────────────────────────────────────────────────

    def _plotly_to_html(self, fig, div_id: str, height: int = 340) -> str:
        fig.update_layout(
            height=height,
            margin=dict(l=10, r=30, t=44, b=10),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(family="Segoe UI, Arial", size=12),
        )
        return fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id=div_id,
            config={"displayModeBar": False},
        )

    def _exposure_by_issue(self, summary: dict) -> str:
        findings = summary.get("findings", [])
        if not findings:
            return "<p>No data</p>"
        labels  = [f["issue_type"] for f in findings]
        values  = [float(f["total_exposure_aud"]) for f in findings]
        colours = [COLOURS["high"], COLOURS["medium"], COLOURS["medium"]]

        fig = go.Figure(go.Bar(
            x=values, y=labels, orientation="h",
            marker_color=colours,
            text=[f"${v:,.0f}" for v in values],
            textposition="inside",
            textfont=dict(color="white", size=13),
            insidetextanchor="middle",
        ))
        fig.update_layout(
            title="Total Exposure by Issue Type ($)",
            xaxis=dict(title="Exposure ($)", tickformat="$,.0f"),
            yaxis=dict(autorange="reversed"),
        )
        return self._plotly_to_html(fig, "exposure_chart", height=280)

    def _sg_by_department(self, sg: dict) -> str:
        dept = sg.get("by_department", {})
        if not dept:
            return "<p>No data</p>"

        # Explicit float cast — prevents Plotly treating values as categorical
        labels = list(dept.keys())
        values = [float(v) for v in dept.values()]

        fig = go.Figure(go.Bar(
            x=values, y=labels, orientation="h",
            marker_color=COLOURS["high"],
            text=[f"${v:,.0f}" for v in values],
            textposition="inside",
            textfont=dict(color="white", size=12),
            insidetextanchor="middle",
        ))
        fig.update_layout(
            title="SG Shortfall by Department ($)",
            xaxis=dict(title="Shortfall ($)", tickformat="$,.0f"),
            yaxis=dict(autorange="reversed", title=""),
        )
        return self._plotly_to_html(fig, "sg_dept_chart", height=340)

    def _sg_by_state(self, sg: dict) -> str:
        state = sg.get("by_state", {})
        if not state:
            return "<p>No data</p>"

        labels = list(state.keys())
        values = [float(v) for v in state.values()]

        fig = go.Figure(go.Bar(
            x=labels, y=values,
            marker_color=COLOURS["accent"],
            text=[f"${v:,.0f}" for v in values],
            textposition="outside",
            textfont=dict(size=11),
        ))
        fig.update_layout(
            title="SG Shortfall by State ($)",
            xaxis=dict(title="State"),
            yaxis=dict(
                title="Shortfall ($)",
                tickformat="$,.0f",
                range=[0, max(values) * 1.25],  # headroom for outside labels
            ),
        )
        return self._plotly_to_html(fig, "sg_state_chart", height=340)

    def _payg_split(self, payg: dict) -> str:
        over  = int(payg.get("over_withheld_runs", 0))
        under = int(payg.get("under_withheld_runs", 0))
        if over + under == 0:
            return "<p>No data</p>"
        fig = go.Figure(go.Pie(
            labels=["Over-withheld", "Under-withheld"],
            values=[over, under],
            hole=0.5,
            marker_colors=[COLOURS["over"], COLOURS["under"]],
            textinfo="label+percent",
            textposition="inside",
        ))
        fig.update_layout(
            title="PAYG Over vs Under-Withheld Split",
            showlegend=True,
        )
        return self._plotly_to_html(fig, "payg_split_chart", height=340)

    def _payg_by_department(self, payg: dict) -> str:
        dept = payg.get("by_department", {})
        if not dept:
            return "<p>No data</p>"

        labels = list(dept.keys())
        values = [float(v) for v in dept.values()]

        fig = go.Figure(go.Bar(
            x=labels, y=values,
            marker_color=COLOURS["medium"],
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(size=11),
        ))
        fig.update_layout(
            title="Avg PAYG Deviation % by Department",
            xaxis=dict(title="Department"),
            yaxis=dict(
                title="Avg Deviation (%)",
                range=[0, max(values) * 1.3],  # headroom for outside labels
            ),
        )
        return self._plotly_to_html(fig, "payg_dept_chart", height=340)

    def _sg_trend(self, df: pd.DataFrame) -> str:
        """
        SG shortfall trend grouped by pay_date.
        Uses precomputed sg_shortfall from CSV if available.
        Falls back to calculating from expected_sg - super_paid.
        """
        if "pay_date" not in df.columns:
            return "<p>No trend data available</p>"

        trend = df.copy()
        trend["pay_date"] = pd.to_datetime(trend["pay_date"], errors="coerce")
        trend = trend.dropna(subset=["pay_date"])

        # Use precomputed sg_shortfall if it exists in the CSV (added by notebook)
        # Otherwise recalculate from expected_sg and super_paid
        if "sg_shortfall" in trend.columns:
            trend["_shortfall"] = pd.to_numeric(
                trend["sg_shortfall"], errors="coerce"
            ).fillna(0)
        elif "expected_sg" in trend.columns and "super_paid" in trend.columns:
            expected = pd.to_numeric(trend["expected_sg"], errors="coerce").fillna(0)
            paid     = pd.to_numeric(trend["super_paid"],  errors="coerce").fillna(0)
            trend["_shortfall"] = (expected - paid).clip(lower=0)
        else:
            return "<p>Insufficient columns for trend calculation</p>"

        by_date = (
            trend.groupby("pay_date")["_shortfall"]
            .sum()
            .reset_index()
            .sort_values("pay_date")
            .reset_index(drop=True)     # clean 0-N index after sort
        )

        # Convert to plain lists — strips pandas index before Plotly sees it
        x_vals = by_date["pay_date"].dt.strftime("%d %b %Y").tolist()
        y_vals = by_date["_shortfall"].tolist()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            name="SG Shortfall",
            line=dict(color=COLOURS["high"], width=2),
            marker=dict(size=8),
            text=[f"${v:,.0f}" for v in y_vals],
            hovertemplate="%{x}<br>Shortfall: %{text}<extra></extra>",
        ))
        fig.update_layout(
            title="SG Shortfall Trend by Pay Date ($)",
            xaxis=dict(title="Pay Date", tickangle=-30),
            yaxis=dict(
                title="Shortfall ($)",
                tickformat="$,.0f",
                range=[0, max(y_vals) * 1.2] if y_vals and max(y_vals) > 0 else [0, 1],
            ),
        )
        return self._plotly_to_html(fig, "trend_chart", height=340)

    def _summary_rows(self, summary: dict, risk_scores: dict) -> str:
        findings = summary.get("findings", [])
        score_map = {
            "SG Underpayment":         risk_scores.get("sg_underpayment", {}).get("score", "N/A"),
            "PAYG Inconsistency":      risk_scores.get("payg_inconsistency", {}).get("score", "N/A"),
            "Missing Super — Casuals": risk_scores.get("missing_casual_super", {}).get("score", "N/A"),
        }
        rows = ""
        for f in findings:
            score  = score_map.get(f["issue_type"], "N/A")
            colour = RISK_COLOURS.get(score, "#999")
            badge  = (f'<span style="background:{colour};color:white;padding:2px 8px;'
                      f'border-radius:4px;font-size:0.82em;font-weight:bold">{score}</span>')
            rows += f"""
        <tr>
          <td><strong>{f['issue_type']}</strong></td>
          <td>{f['employees_affected']}</td>
          <td>{f['pay_runs_affected']}</td>
          <td><strong>${float(f['total_exposure_aud']):,.2f}</strong></td>
          <td>{badge}</td>
        </tr>"""
        return rows
