"""
Deterministic Python Financial Calculations.
Same inputs always produce the exact same outputs (no AI used here).
"""
from typing import Dict, Any
from src.config import EXPENSE_CATEGORIES


def calculate_financial_metrics(
    monthly_income: float,
    expenses: Dict[str, float],
    savings: float,
) -> Dict[str, Any]:
    """
    Compute total expenses, remaining income, savings ratio, expense ratio,
    debt ratio, and a rule-based preliminary financial health score (0-100).
    Guards against divide-by-zero when monthly_income is 0.
    """
    income = max(0.0, float(monthly_income))
    clean_expenses = {k: max(0.0, float(v)) for k, v in expenses.items()}
    current_savings = max(0.0, float(savings))

    total_expenses = round(sum(clean_expenses.values()), 2)
    remaining_income = round(income - total_expenses, 2)

    # Guard against divide-by-zero
    if income > 0:
        savings_ratio = round((current_savings / income) * 100.0, 2)
        expense_ratio = round((total_expenses / income) * 100.0, 2)
        leftover_ratio = round((remaining_income / income) * 100.0, 2)
        debt_ratio = round((clean_expenses.get("debt", 0.0) / income) * 100.0, 2)
    else:
        savings_ratio = 0.0
        expense_ratio = 100.0 if total_expenses > 0 else 0.0
        leftover_ratio = -100.0 if total_expenses > 0 else 0.0
        debt_ratio = 100.0 if clean_expenses.get("debt", 0.0) > 0 else 0.0

    preliminary_score = compute_preliminary_score(
        income=income,
        total_expenses=total_expenses,
        savings_ratio=savings_ratio,
        expense_ratio=expense_ratio,
        leftover_ratio=leftover_ratio,
        debt_ratio=debt_ratio,
    )

    # Build readable expense breakdown string for prompts
    label_map = dict(EXPENSE_CATEGORIES)
    breakdown_lines = []
    for key, amount in clean_expenses.items():
        label = label_map.get(key, key.capitalize())
        pct = round((amount / income) * 100.0, 1) if income > 0 else 0.0
        breakdown_lines.append(f"- {label}: {amount:,.2f} ({pct}% of income)")

    return {
        "monthly_income": income,
        "expenses": clean_expenses,
        "total_expenses": total_expenses,
        "remaining_income": remaining_income,
        "savings": current_savings,
        "savings_ratio": savings_ratio,
        "expense_ratio": expense_ratio,
        "leftover_ratio": leftover_ratio,
        "debt_ratio": debt_ratio,
        "preliminary_score": preliminary_score,
        "score_band": get_score_band(preliminary_score),
        "expense_breakdown_str": "\n".join(breakdown_lines),
    }


def compute_preliminary_score(
    income: float,
    total_expenses: float,
    savings_ratio: float,
    expense_ratio: float,
    leftover_ratio: float,
    debt_ratio: float,
) -> int:
    """
    Weighted 0-100 heuristic based on:
    1. Expense Ratio (up to 35 pts)
    2. Savings Ratio (up to 30 pts)
    3. Leftover / Cashflow Buffer (up to 20 pts)
    4. Debt Burden (up to 15 pts)
    """
    if income <= 0:
        return 0 if total_expenses > 0 else 25

    # 1. Expense Ratio Component (0 - 35 points)
    if expense_ratio <= 50:
        expense_pts = 35.0
    elif expense_ratio <= 80:
        expense_pts = 35.0 - ((expense_ratio - 50.0) / 30.0) * 15.0  # 20 to 35
    elif expense_ratio <= 100:
        expense_pts = 20.0 - ((expense_ratio - 80.0) / 20.0) * 20.0  # 0 to 20
    else:
        expense_pts = max(-15.0, -((expense_ratio - 100.0) * 0.5))

    # 2. Savings Ratio Component (0 - 30 points, full marks at 20%+ savings)
    savings_pts = min(30.0, max(0.0, (savings_ratio / 20.0) * 30.0))

    # 3. Leftover / Cashflow Component (0 - 20 points, full marks at 20%+ leftover)
    if leftover_ratio < 0:
        leftover_pts = max(-10.0, leftover_ratio * 0.4)
    else:
        leftover_pts = min(20.0, (leftover_ratio / 20.0) * 20.0)

    # 4. Debt Ratio Component (0 - 15 points, penalized as debt exceeds 15-40% of income)
    if debt_ratio == 0:
        debt_pts = 15.0
    elif debt_ratio <= 15:
        debt_pts = 12.0
    elif debt_ratio <= 35:
        debt_pts = 12.0 - ((debt_ratio - 15.0) / 20.0) * 10.0
    else:
        debt_pts = 0.0

    raw_score = expense_pts + savings_pts + leftover_pts + debt_pts
    return int(round(max(0.0, min(100.0, raw_score))))


def get_score_band(score: int) -> str:
    """Return educational health score band per Section 12 of assignment."""
    if score >= 80:
        return "Strong (80–100)"
    if score >= 60:
        return "Generally Healthy (60–79)"
    if score >= 40:
        return "Needs Improvement (40–59)"
    return "High Attention (<40)"
