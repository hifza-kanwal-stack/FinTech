"""
Utility functions for safe JSON parsing, schema validation, CSV parsing, and report generation.
"""
import json
import re
from typing import Dict, Any, Tuple
import pandas as pd


REQUIRED_JSON_KEYS = {
    "financial_summary": str,
    "financial_health_score": int,
    "spending_analysis": list,
    "risk_level": str,
    "top_priorities": list,
    "budget_recommendations": list,
    "savings_strategy": list,
    "next_month_action_plan": list,
}


def safe_parse_financial_json(
    raw_text: str, fallback_metrics: Dict[str, Any]
) -> Tuple[Dict[str, Any], bool, str]:
    """
    Safely parse and validate the LLM's JSON response against the Section 12 schema.
    Returns (parsed_dict, is_valid_json, status_message).
    """
    if not raw_text or not raw_text.strip():
        return build_fallback_response(fallback_metrics), False, "Empty response from LLM; loaded safe fallback."

    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned = cleaned[start_idx : end_idx + 1]

    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("Top-level JSON is not an object.")
        validated = normalize_schema(data, fallback_metrics)
        return validated, True, "JSON validated successfully against schema."
    except Exception as exc:
        fallback = build_fallback_response(fallback_metrics)
        return fallback, False, f"JSON parse warning ({exc}); recovered with safe fallback schema."


def normalize_schema(data: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure every key required by Section 12 exists and has the right type."""
    default = build_fallback_response(metrics)

    summary = str(data.get("financial_summary") or default["financial_summary"])
    try:
        score = int(float(data.get("financial_health_score", default["financial_health_score"])))
        score = max(0, min(100, score))
    except (ValueError, TypeError):
        score = default["financial_health_score"]

    risk = str(data.get("risk_level", default["risk_level"])).upper().strip()
    if risk not in {"LOW", "MEDIUM", "HIGH"}:
        risk = default["risk_level"]

    spending_raw = data.get("spending_analysis", [])
    spending_clean = []
    if isinstance(spending_raw, list):
        for item in spending_raw:
            if isinstance(item, dict):
                spending_clean.append(
                    {
                        "category": str(item.get("category", "General")),
                        "observation": str(item.get("observation", "Reviewed based on monthly budget.")),
                        "recommendation": str(item.get("recommendation", "Track this expense monthly.")),
                    }
                )
    if not spending_clean:
        spending_clean = default["spending_analysis"]

    def _clean_str_list(key: str) -> list:
        val = data.get(key, [])
        if isinstance(val, list) and val:
            return [str(x) for x in val]
        return default[key]

    return {
        "financial_summary": summary,
        "financial_health_score": score,
        "spending_analysis": spending_clean,
        "risk_level": risk,
        "top_priorities": _clean_str_list("top_priorities"),
        "budget_recommendations": _clean_str_list("budget_recommendations"),
        "savings_strategy": _clean_str_list("savings_strategy"),
        "next_month_action_plan": _clean_str_list("next_month_action_plan"),
    }


def build_fallback_response(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic fallback matching Section 12 schema if JSON parsing fails."""
    score = metrics.get("preliminary_score", 50)
    if score >= 70:
        risk = "LOW"
    elif score >= 45:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return {
        "financial_summary": (
            f"Educational overview: Your total monthly expenses are {metrics.get('total_expenses', 0):,.2f} "
            f"({metrics.get('expense_ratio', 0)}% of income), leaving a remaining balance of "
            f"{metrics.get('remaining_income', 0):,.2f}."
        ),
        "financial_health_score": score,
        "spending_analysis": [
            {
                "category": "Overall Spending",
                "observation": f"Expense ratio is {metrics.get('expense_ratio', 0)}% of monthly income.",
                "recommendation": "Aim to keep total essential and discretionary expenses below 80% of income.",
            }
        ],
        "risk_level": risk,
        "top_priorities": [
            "Maintain positive monthly cash flow",
            "Build or preserve an emergency savings buffer",
            "Review highest expense categories monthly",
        ],
        "budget_recommendations": [
            "Follow the 50/30/20 educational budgeting guideline where practical.",
            "Track variable expenses weekly to prevent end-of-month overruns.",
        ],
        "savings_strategy": [
            "Automate savings transfers at the start of each month.",
            "Direct any unexpected surplus toward your primary financial goal.",
        ],
        "next_month_action_plan": [
            "Week 1: Audit all recurring subscriptions and utility bills.",
            "Week 2: Set category spending caps for food and entertainment.",
            "Week 3: Transfer planned savings into a dedicated goal account.",
            "Week 4: Reconcile actual spending vs. target budget in FinWise AI.",
        ],
    }


def generate_sample_csv() -> str:
    """Return sample CSV content for the Bonus CSV upload feature."""
    df = pd.DataFrame(
        [
            {"category": "housing", "amount": 1200},
            {"category": "food", "amount": 450},
            {"category": "transportation", "amount": 220},
            {"category": "utilities", "amount": 160},
            {"category": "education", "amount": 100},
            {"category": "healthcare", "amount": 90},
            {"category": "entertainment", "amount": 150},
            {"category": "debt", "amount": 200},
            {"category": "insurance", "amount": 80},
            {"category": "other", "amount": 50},
        ]
    )
    return df.to_csv(index=False)


def generate_html_report(
    currency_symbol: str,
    goal: str,
    metrics: Dict[str, Any],
    ai_data: Dict[str, Any],
    narrative_text: str,
) -> str:
    """Generate a clean, printable HTML report (Bonus PDF/Report export feature)."""
    priorities_html = "".join(f"<li>{p}</li>" for p in ai_data.get("top_priorities", []))
    recs_html = "".join(f"<li>{r}</li>" for r in ai_data.get("budget_recommendations", []))
    action_html = "".join(f"<li>{a}</li>" for a in ai_data.get("next_month_action_plan", []))

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>FinWise AI - Monthly Financial Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #0f172a; margin: 40px; line-height: 1.6; }}
  .header {{ border-bottom: 3px solid #0f766e; padding-bottom: 12px; margin-bottom: 20px; }}
  .disclaimer {{ background: #fef3c7; border-left: 4px solid #d97706; padding: 12px; font-size: 13px; margin-bottom: 24px; }}
  .grid {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }}
  .card {{ flex: 1; min-width: 180px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; }}
  .card h4 {{ margin: 0 0 6px 0; color: #475569; font-size: 13px; }}
  .card p {{ margin: 0; font-size: 20px; font-weight: 700; color: #0f766e; }}
  h2 {{ color: #0f766e; font-size: 18px; margin-top: 24px; }}
</style>
</head>
<body>
  <div class="header">
    <h1>FinWise AI — Personal Financial Analysis Report</h1>
    <p>Primary Financial Goal: <strong>{goal}</strong> | Risk Level: <strong>{ai_data.get('risk_level')}</strong></p>
  </div>
  <div class="disclaimer">
    <strong>EDUCATIONAL USE ONLY:</strong> This report is generated by an educational AI prototype for informational purposes only. It is not financial advice. Consult a qualified financial professional.
  </div>
  <div class="grid">
    <div class="card"><h4>Monthly Income</h4><p>{currency_symbol}{metrics['monthly_income']:,.2f}</p></div>
    <div class="card"><h4>Total Expenses</h4><p>{currency_symbol}{metrics['total_expenses']:,.2f} ({metrics['expense_ratio']}%)</p></div>
    <div class="card"><h4>Remaining Balance</h4><p>{currency_symbol}{metrics['remaining_income']:,.2f}</p></div>
    <div class="card"><h4>Monthly Savings</h4><p>{currency_symbol}{metrics['savings']:,.2f} ({metrics['savings_ratio']}%)</p></div>
    <div class="card"><h4>Python / AI Health Score</h4><p>{metrics['preliminary_score']} / {ai_data.get('financial_health_score')} (out of 100)</p></div>
  </div>
  <h2>Executive Summary</h2>
  <p>{ai_data.get('financial_summary')}</p>
  <h2>Top Priorities</h2>
  <ul>{priorities_html}</ul>
  <h2>Budget Recommendations</h2>
  <ul>{recs_html}</ul>
  <h2>Next-Month Action Plan</h2>
  <ul>{action_html}</ul>
  <h2>AI Coaching Memo</h2>
  <div style="white-space: pre-wrap; background:#f8fafc; padding:16px; border-radius:8px;">{narrative_text}</div>
</body>
</html>"""
