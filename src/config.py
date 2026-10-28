"""
Configuration settings, constants, disclaimers, and test scenarios for FinWise AI.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

APP_TITLE = "FinWise AI"
APP_SUBTITLE = "AI-Powered Personal Financial Analysis & Smart Budget Assistant"

EDUCATIONAL_DISCLAIMER = (
    "⚠️ **EDUCATIONAL USE ONLY:** FinWise AI is an educational prototype. "
    "It does not provide guaranteed investment advice, execute financial transactions, "
    "connect to real bank accounts, or guarantee any financial outcome. "
    "All outputs are for informational purposes only. Please consult a qualified "
    "financial professional before making real financial decisions."
)

CURRENCIES = {
    "USD ($)": "$",
    "EUR (€)": "€",
    "GBP (£)": "£",
    "PKR (₨)": "₨",
    "INR (₹)": "₹",
    "CAD (C$)": "C$",
    "AUD (A$)": "A$",
}

FINANCIAL_GOALS = [
    "Save money",
    "Emergency fund",
    "Pay off debt",
    "Vacation",
    "Start a business",
    "Improve budgeting",
]

# 10 Expense Categories (Aligned with Section 6 & Section 8)
EXPENSE_CATEGORIES = [
    ("housing", "Housing / Rent"),
    ("food", "Food & Dining"),
    ("transportation", "Transportation"),
    ("utilities", "Utilities & Bills"),
    ("education", "Education"),
    ("healthcare", "Healthcare"),
    ("entertainment", "Entertainment & Leisure"),
    ("debt", "Loan / Debt Repayment"),
    ("insurance", "Insurance & Protection"),
    ("other", "Other / Miscellaneous"),
]

MODEL_OPTIONS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-3.5-turbo",
]

# Section 16: 5 Official Testing Scenarios from Assignment PDF
TEST_SCENARIOS = {
    "Custom Input (Manual)": None,
    "Scenario 1: High Earner & Strong Saver (Income 8000, Exp ~2000)": {
        "monthly_income": 8000.0,
        "savings": 2500.0,
        "financial_goal": "Start a business",
        "goal_target": 30000.0,
        "expenses": {
            "housing": 900.0, "food": 400.0, "transportation": 200.0,
            "utilities": 150.0, "education": 100.0, "healthcare": 80.0,
            "entertainment": 100.0, "debt": 0.0, "insurance": 50.0, "other": 20.0,
        },
    },
    "Scenario 2: Deficit Spending (Income 2000, Exp ~2600)": {
        "monthly_income": 2000.0,
        "savings": 0.0,
        "financial_goal": "Improve budgeting",
        "goal_target": 5000.0,
        "expenses": {
            "housing": 1100.0, "food": 500.0, "transportation": 250.0,
            "utilities": 180.0, "education": 0.0, "healthcare": 70.0,
            "entertainment": 250.0, "debt": 150.0, "insurance": 50.0, "other": 50.0,
        },
    },
    "Scenario 3: Heavy Debt Burden (Income 5000, Debt 2500)": {
        "monthly_income": 5000.0,
        "savings": 200.0,
        "financial_goal": "Pay off debt",
        "goal_target": 15000.0,
        "expenses": {
            "housing": 1000.0, "food": 450.0, "transportation": 200.0,
            "utilities": 150.0, "education": 0.0, "healthcare": 100.0,
            "entertainment": 100.0, "debt": 2500.0, "insurance": 100.0, "other": 100.0,
        },
    },
    "Scenario 4: Balanced 30% Saver (Income 4000, Savings 1200)": {
        "monthly_income": 4000.0,
        "savings": 1200.0,
        "financial_goal": "Emergency fund",
        "goal_target": 12000.0,
        "expenses": {
            "housing": 1100.0, "food": 450.0, "transportation": 250.0,
            "utilities": 150.0, "education": 100.0, "healthcare": 100.0,
            "entertainment": 150.0, "debt": 100.0, "insurance": 100.0, "other": 100.0,
        },
    },
    "Scenario 5: Break-Even Tight Budget (Income 3000, Exp 3000)": {
        "monthly_income": 3000.0,
        "savings": 0.0,
        "financial_goal": "Save money",
        "goal_target": 6000.0,
        "expenses": {
            "housing": 1200.0, "food": 550.0, "transportation": 300.0,
            "utilities": 200.0, "education": 50.0, "healthcare": 100.0,
            "entertainment": 250.0, "debt": 150.0, "insurance": 100.0, "other": 100.0,
        },
    },
}


def get_env_api_key() -> str:
    """Retrieve OpenAI API key from environment if available."""
    return os.getenv("OPENAI_API_KEY", "").strip()
