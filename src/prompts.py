"""
LangChain PromptTemplates and ChatPromptTemplates for FinWise AI.
Includes required variables and strict JSON output instructions.
"""
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

SYSTEM_SAFETY_PROMPT = """You are FinWise AI, an educational personal financial analysis and smart budgeting assistant.
CRITICAL SAFETY & COMPLIANCE RULES:
1. You provide EDUCATIONAL and INFORMATIONAL insights only.
2. Never provide guaranteed investment advice, stock picks, or promise guaranteed financial returns.
3. Always remind users to consult a certified financial planner or professional for big decisions.
4. Base your analysis strictly on the user's actual numbers provided in the prompt.
"""

JSON_SCHEMA_INSTRUCTIONS = """You MUST respond with ONLY a valid, raw JSON object (no markdown code fences, no extra commentary outside JSON) matching this exact schema:
{{
  "financial_summary": "A concise 2-3 sentence educational overview of the user's current monthly financial posture.",
  "financial_health_score": 75,
  "spending_analysis": [
    {{
      "category": "Category Name",
      "observation": "Specific observation citing their numbers and percentage of income",
      "recommendation": "Actionable educational suggestion to optimize this category"
    }}
  ],
  "risk_level": "LOW | MEDIUM | HIGH",
  "top_priorities": [
    "Priority 1",
    "Priority 2",
    "Priority 3"
  ],
  "budget_recommendations": [
    "Recommendation 1",
    "Recommendation 2",
    "Recommendation 3"
  ],
  "savings_strategy": [
    "Savings strategy step 1",
    "Savings strategy step 2",
    "Savings strategy step 3"
  ],
  "next_month_action_plan": [
    "Week 1: ...",
    "Week 2: ...",
    "Week 3: ...",
    "Week 4: ..."
  ]
}}

Scoring guidance (0-100 integer for financial_health_score):
- 80-100: Strong financial habits, positive remaining balance, high savings ratio.
- 60-79: Generally healthy, room to optimize discretionary spending or boost savings.
- 40-59: Needs improvement, tight cash flow or elevated debt burden.
- Below 40: High attention required, negative remaining income or severe debt stress.
"""

# Section 11: Reusable single-string PromptTemplate with all 8 required variables
FINANCIAL_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=[
        "monthly_income",
        "total_expenses",
        "remaining_income",
        "savings",
        "savings_ratio",
        "expense_ratio",
        "financial_goal",
        "expense_breakdown",
    ],
    template=(
        "Analyze the following monthly financial profile (Educational Use Only):\n\n"
        "- Monthly Income: {monthly_income}\n"
        "- Total Monthly Expenses: {total_expenses}\n"
        "- Remaining Monthly Income: {remaining_income}\n"
        "- Current Monthly Savings: {savings}\n"
        "- Savings Ratio: {savings_ratio}%\n"
        "- Expense Ratio: {expense_ratio}%\n"
        "- Primary Financial Goal: {financial_goal}\n\n"
        "Detailed Expense Breakdown:\n{expense_breakdown}\n\n"
        + JSON_SCHEMA_INSTRUCTIONS
    ),
)

# Section 10 & 11: ChatPromptTemplate combining System instructions + Human template
STRUCTURED_CHAT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_SAFETY_PROMPT),
        ("human", FINANCIAL_PROMPT_TEMPLATE.template),
    ]
)

# Section 13: ChatPromptTemplate for streaming personalized narrative coaching
NARRATIVE_CHAT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_SAFETY_PROMPT
            + "\nWrite an encouraging, well-structured markdown coaching memo tailored to the user's numbers. "
            "Include: (1) Cash Flow Diagnosis, (2) 50/30/20 Rule Comparison, "
            "(3) Goal Acceleration Plan for their stated goal, and (4) An Educational Disclaimer.",
        ),
        (
            "human",
            "User Financial Snapshot:\n"
            "- Monthly Income: {monthly_income}\n"
            "- Total Expenses: {total_expenses} ({expense_ratio}% of income)\n"
            "- Remaining Income: {remaining_income}\n"
            "- Current Savings: {savings} ({savings_ratio}% of income)\n"
            "- Primary Goal: {financial_goal}\n"
            "- Expense Breakdown:\n{expense_breakdown}\n\n"
            "Please provide a personalized educational coaching narrative.",
        ),
    ]
)
