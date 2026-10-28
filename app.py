"""
FinWise AI - Streamlit Frontend & Console
Run with: streamlit run app.py
"""
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    APP_TITLE,
    APP_SUBTITLE,
    EDUCATIONAL_DISCLAIMER,
    CURRENCIES,
    FINANCIAL_GOALS,
    EXPENSE_CATEGORIES,
    MODEL_OPTIONS,
    TEST_SCENARIOS,
    get_env_api_key,
)
from src.financial_calculator import calculate_financial_metrics, get_score_band
from src.prompts import FINANCIAL_PROMPT_TEMPLATE
from src.chains import (
    get_chat_model,
    run_structured_analysis,
    stream_recommendations,
    build_message_demo_conversation,
    run_followup_chat,
)
from src.cache_manager import configure_cache, clear_active_cache
from src.utils import (
    safe_parse_financial_json,
    generate_sample_csv,
    generate_html_report,
)

# -----------------------------------------------------------------------------
# Page Config & Custom Minimalist FinTech CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_TITLE} | Smart Financial Assistant",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        /* Clean, Minimal, Professional FinTech Theme */
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2.5rem;
            max-width: 1280px;
        }
        .finwise-header {
            background: linear-gradient(135deg, #0F172A 0%, #0F766E 100%);
            color: #FFFFFF;
            padding: 1.6rem 2rem;
            border-radius: 14px;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 20px rgba(15, 23, 42, 0.08);
        }
        .finwise-header h1 {
            margin: 0;
            font-size: 1.9rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #FFFFFF;
        }
        .finwise-header p {
            margin: 0.35rem 0 0 0;
            color: #CCFBF1;
            font-size: 0.98rem;
        }
        .disclaimer-banner {
            background-color: #FFFBEB;
            border: 1px solid #FDE68A;
            border-left: 5px solid #D97706;
            color: #92400E;
            padding: 0.85rem 1.1rem;
            border-radius: 8px;
            font-size: 0.86rem;
            margin-bottom: 1.25rem;
        }
        .auth-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 2.2rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
            margin-top: 1.5rem;
        }
        .risk-badge {
            display: inline-block;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.85rem;
            letter-spacing: 0.04em;
        }
        .risk-low { background: #DCFCE7; color: #166534; border: 1px solid #86EFAC; }
        .risk-medium { background: #FEF9C3; color: #854D0E; border: 1px solid #FDE047; }
        .risk-high { background: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5; }
        div[data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            padding: 1rem 1.15rem;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
def init_session_state() -> None:
    if "api_key_authenticated" not in st.session_state:
        env_key = get_env_api_key()
        st.session_state.api_key_authenticated = bool(env_key and env_key.startswith("sk-"))
        st.session_state.openai_api_key = env_key if st.session_state.api_key_authenticated else ""

    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False
    if "metrics_result" not in st.session_state:
        st.session_state.metrics_result = None
    if "ai_json_result" not in st.session_state:
        st.session_state.ai_json_result = None
    if "json_status_msg" not in st.session_state:
        st.session_state.json_status_msg = ""
    if "narrative_stream_text" not in st.session_state:
        st.session_state.narrative_stream_text = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "csv_overrides" not in st.session_state:
        st.session_state.csv_overrides = {}


def reset_console_session() -> None:
    """Reset analysis state while keeping the user logged in."""
    st.session_state.analysis_complete = False
    st.session_state.metrics_result = None
    st.session_state.ai_json_result = None
    st.session_state.json_status_msg = ""
    st.session_state.narrative_stream_text = ""
    st.session_state.chat_history = []
    st.session_state.csv_overrides = {}
    clear_active_cache()


init_session_state()


# -----------------------------------------------------------------------------
# Reusable Header & Disclaimer
# -----------------------------------------------------------------------------
def render_header() -> None:
    st.markdown(
        f"""
        <div class="finwise-header">
            <h1>💳 {APP_TITLE}</h1>
            <p>{APP_SUBTITLE} • Powered by LangChain & OpenAI</p>
        </div>
        <div class="disclaimer-banner">
            {EDUCATIONAL_DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# STEP 1: API Key Authentication Gate (Before Console Access)
# -----------------------------------------------------------------------------
def render_api_key_gate() -> None:
    render_header()

    col_left, col_center, col_right = st.columns([1, 1.6, 1])
    with col_center:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.subheader("🔐 Authenticate to Access FinWise AI Console")
        st.caption(
            "Enter your OpenAI API key to unlock the personal financial analysis dashboard. "
            "Your key is stored only in your active browser session and is never logged."
        )

        with st.form("api_key_gate_form"):
            user_key = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-proj-...",
                help="Create an API key at https://platform.openai.com/api-keys",
            )
            verify_conn = st.checkbox(
                "Verify API key with a lightweight connection check", value=False
            )
            submitted = st.form_submit_button(
                "Unlock FinWise AI Console →", use_container_width=True, type="primary"
            )

            if submitted:
                cleaned_key = user_key.strip()
                if not cleaned_key.startswith("sk-") or len(cleaned_key) < 20:
                    st.error("Please enter a valid OpenAI API key starting with `sk-`.")
                else:
                    if verify_conn:
                        try:
                            with st.spinner("Verifying OpenAI API key..."):
                                test_llm = get_chat_model(cleaned_key, "gpt-4o-mini", 0.0)
                                test_llm.invoke("Respond with OK")
                        except Exception as exc:
                            st.error(f"API Key verification failed: {exc}")
                            st.stop()

                    st.session_state.openai_api_key = cleaned_key
                    st.session_state.api_key_authenticated = True
                    st.rerun()

        st.markdown("---")
        st.markdown(
            "**Don't have a key handy?** Create one at [platform.openai.com](https://platform.openai.com/api-keys) "
            "or place `OPENAI_API_KEY=sk-...` inside a local `.env` file."
        )
        st.markdown("</div>", unsafe_allow_html=True)


# Stop and show only the gate until authenticated
if not st.session_state.api_key_authenticated:
    render_api_key_gate()
    st.stop()


# -----------------------------------------------------------------------------
# STEP 2: Authenticated Sidebar (Settings, Caching, Scenarios, Reset)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"## 💳 {APP_TITLE}")
    st.caption("Smart Budget & Financial Health Console")

    st.success("🟢 OpenAI API Key Connected")
    if st.button("🔑 Change / Lock API Key", use_container_width=True):
        st.session_state.api_key_authenticated = False
        st.session_state.openai_api_key = ""
        reset_console_session()
        st.rerun()

    st.divider()
    st.subheader("⚙️ Model & LangChain Cache")
    selected_model = st.selectbox("OpenAI Model", MODEL_OPTIONS, index=0)
    temperature = st.slider(
        "Model Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1
    )

    cache_choice = st.radio(
        "LangChain Caching Mode (Sec. 14)",
        options=["InMemoryCache", "SQLiteCache", "Disabled"],
        index=0,
        help="InMemoryCache stores responses in RAM; SQLiteCache persists across restarts in .langchain_cache.db",
    )
    cache_status = configure_cache(cache_choice)
    st.caption(f"⚡ {cache_status}")

    st.divider()
    st.subheader("🧪 Assignment Test Scenarios")
    selected_scenario_name = st.selectbox(
        "Load Pre-configured Scenario (Sec. 16)",
        options=list(TEST_SCENARIOS.keys()),
        index=0,
        help="Quickly test the 5 required scenarios from Section 16 of the PDF.",
    )

    st.divider()
    if st.button("🔄 Reset Session & Clear Cache", use_container_width=True):
        reset_console_session()
        st.rerun()

    st.divider()
    st.info(EDUCATIONAL_DISCLAIMER)


# -----------------------------------------------------------------------------
# STEP 3: Main Console Layout & Financial Input Form
# -----------------------------------------------------------------------------
render_header()

scenario_preset = TEST_SCENARIOS.get(selected_scenario_name)

# Bonus Feature: Optional CSV Expense Importer
with st.expander("📂 Bonus Feature: Import Monthly Expenses from CSV (or Download Template)"):
    csv_col1, csv_col2 = st.columns([2, 1])
    with csv_col1:
        uploaded_csv = st.file_uploader(
            "Upload CSV with columns `category,amount`", type=["csv"]
        )
        if uploaded_csv is not None:
            try:
                df_csv = pd.read_csv(uploaded_csv)
                if {"category", "amount"}.issubset(set(df_csv.columns.str.lower())):
                    df_csv.columns = df_csv.columns.str.lower()
                    overrides = dict(zip(df_csv["category"].str.strip(), df_csv["amount"]))
                    st.session_state.csv_overrides = overrides
                    st.success("CSV expenses loaded into the form below!")
                else:
                    st.warning("CSV must contain `category` and `amount` columns.")
            except Exception as exc:
                st.error(f"Could not parse CSV: {exc}")
    with csv_col2:
        st.download_button(
            "⬇️ Download Sample Expense CSV",
            data=generate_sample_csv(),
            file_name="finwise_sample_expenses.csv",
            mime="text/csv",
            use_container_width=True,
        )

# Resolve defaults from Scenario Preset or CSV Overrides
default_income = scenario_preset["monthly_income"] if scenario_preset else 4500.0
default_savings = scenario_preset["savings"] if scenario_preset else 600.0
default_goal = scenario_preset["financial_goal"] if scenario_preset else "Save money"
default_goal_target = scenario_preset["goal_target"] if scenario_preset else 10000.0

st.subheader("1️⃣ Monthly Financial Profile Input")
with st.form("financial_input_form"):
    top_col1, top_col2, top_col3, top_col4 = st.columns([1.1, 1.1, 1.2, 0.8])
    with top_col1:
        monthly_income = st.number_input(
            "Monthly Income (After Tax)",
            min_value=0.0,
            value=float(default_income),
            step=100.0,
        )
    with top_col2:
        current_savings = st.number_input(
            "Current Monthly Savings",
            min_value=0.0,
            value=float(default_savings),
            step=50.0,
        )
    with top_col3:
        goal_idx = (
            FINANCIAL_GOALS.index(default_goal)
            if default_goal in FINANCIAL_GOALS
            else 0
        )
        financial_goal = st.selectbox(
            "Primary Financial Goal", options=FINANCIAL_GOALS, index=goal_idx
        )
    with top_col4:
        currency_label = st.selectbox("Currency", options=list(CURRENCIES.keys()), index=0)
        currency_symbol = CURRENCIES[currency_label]

    st.markdown("##### 📊 Monthly Expense Categories (10 Categories)")
    exp_cols = st.columns(5)
    expense_inputs = {}

    default_exp_map = (
        scenario_preset["expenses"]
        if scenario_preset
        else {
            "housing": 1200.0,
            "food": 500.0,
            "transportation": 250.0,
            "utilities": 180.0,
            "education": 100.0,
            "healthcare": 100.0,
            "entertainment": 200.0,
            "debt": 150.0,
            "insurance": 120.0,
            "other": 100.0,
        }
    )

    for idx, (cat_key, cat_label) in enumerate(EXPENSE_CATEGORIES):
        col = exp_cols[idx % 5]
        val = st.session_state.csv_overrides.get(
            cat_key, default_exp_map.get(cat_key, 0.0)
        )
        with col:
            expense_inputs[cat_key] = st.number_input(
                f"{cat_label} ({currency_symbol})",
                min_value=0.0,
                value=float(val),
                step=25.0,
                key=f"exp_{cat_key}_{selected_scenario_name}",
            )

    with st.expander("🎯 Optional Goal Target Tracker (Bonus Feature)"):
        goal_target_amount = st.number_input(
            f"Target Amount for '{financial_goal}' ({currency_symbol})",
            min_value=100.0,
            value=float(default_goal_target),
            step=500.0,
        )

    submit_analysis = st.form_submit_button(
        "🚀 Analyze Finances & Generate Smart Budget Plan",
        use_container_width=True,
        type="primary",
    )


# -----------------------------------------------------------------------------
# STEP 4: Deterministic Python Math + LangChain Execution
# -----------------------------------------------------------------------------
if submit_analysis:
    # 1. Deterministic Python Calculations
    metrics = calculate_financial_metrics(
        monthly_income=monthly_income,
        expenses=expense_inputs,
        savings=current_savings,
    )
    metrics["goal_target_amount"] = goal_target_amount
    st.session_state.metrics_result = metrics

    # 2. Prepare LangChain Prompt Inputs (8 Required Variables from Sec. 11)
    chain_inputs = {
        "monthly_income": f"{currency_symbol}{metrics['monthly_income']:,.2f}",
        "total_expenses": f"{currency_symbol}{metrics['total_expenses']:,.2f}",
        "remaining_income": f"{currency_symbol}{metrics['remaining_income']:,.2f}",
        "savings": f"{currency_symbol}{metrics['savings']:,.2f}",
        "savings_ratio": metrics["savings_ratio"],
        "expense_ratio": metrics["expense_ratio"],
        "financial_goal": financial_goal,
        "expense_breakdown": metrics["expense_breakdown_str"],
    }
    st.session_state.chain_inputs = chain_inputs
    st.session_state.currency_symbol = currency_symbol
    st.session_state.financial_goal = financial_goal

    # 3. Call LLMChain for Structured JSON Output
    with st.spinner("Running LangChain LLMChain & validating structured JSON output..."):
        try:
            llm_sync = get_chat_model(
                api_key=st.session_state.openai_api_key,
                model_name=selected_model,
                temperature=temperature,
                streaming=False,
            )
            raw_json_str = run_structured_analysis(llm_sync, chain_inputs)
            parsed_json, is_valid, status_msg = safe_parse_financial_json(
                raw_json_str, metrics
            )
            st.session_state.ai_json_result = parsed_json
            st.session_state.json_status_msg = status_msg
            st.session_state.narrative_stream_text = ""  # Will stream live below
            st.session_state.analysis_complete = True
        except Exception as exc:
            st.error(f"OpenAI / LangChain Error: {exc}")
            st.stop()


# -----------------------------------------------------------------------------
# STEP 5: Render Financial Overview & AI Dashboard
# -----------------------------------------------------------------------------
if st.session_state.analysis_complete and st.session_state.metrics_result:
    metrics = st.session_state.metrics_result
    ai_data = st.session_state.ai_json_result
    curr = st.session_state.currency_symbol
    goal = st.session_state.financial_goal
    chain_inputs = st.session_state.chain_inputs

    st.divider()
    st.subheader("2️⃣ Deterministic Financial Overview (Python Calculations)")

    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    m_col1.metric("Monthly Income", f"{curr}{metrics['monthly_income']:,.2f}")
    m_col2.metric(
        "Total Expenses",
        f"{curr}{metrics['total_expenses']:,.2f}",
        delta=f"{metrics['expense_ratio']}% of income",
        delta_color="inverse",
    )
    m_col3.metric(
        "Remaining Balance",
        f"{curr}{metrics['remaining_income']:,.2f}",
        delta="Positive Surplus" if metrics["remaining_income"] >= 0 else "Monthly Deficit",
        delta_color="normal" if metrics["remaining_income"] >= 0 else "inverse",
    )
    m_col4.metric(
        "Current Savings",
        f"{curr}{metrics['savings']:,.2f}",
        delta=f"{metrics['savings_ratio']}% savings rate",
    )
    m_col5.metric(
        "Python Heuristic Score",
        f"{metrics['preliminary_score']} / 100",
        delta=metrics["score_band"],
    )

    if metrics["remaining_income"] < 0:
        st.error(
            f"🚨 **Deficit Alert:** Your monthly expenses exceed your income by "
            f"**{curr}{abs(metrics['remaining_income']):,.2f}** ({metrics['expense_ratio']}% expense ratio). "
            "Immediate cost reduction is recommended."
        )
    elif metrics["debt_ratio"] >= 35:
        st.warning(
            f"⚠️ **Elevated Debt Burden:** Loan/debt payments consume **{metrics['debt_ratio']}%** of your monthly income."
        )

    # -------------------------------------------------------------------------
    # Health Score Comparison & Risk Level Banner
    # -------------------------------------------------------------------------
    st.subheader("3️⃣ FinWise AI Health Score & Risk Assessment")
    score_col1, score_col2, score_col3 = st.columns([1.2, 1.2, 1])

    py_score = metrics["preliminary_score"]
    ai_score = ai_data.get("financial_health_score", py_score)
    risk_level = ai_data.get("risk_level", "MEDIUM").upper()

    with score_col1:
        st.markdown(f"**Deterministic Python Score:** `{py_score}/100` ({get_score_band(py_score)})")
        st.progress(py_score / 100.0)
    with score_col2:
        st.markdown(f"**LangChain AI Health Score:** `{ai_score}/100` ({get_score_band(ai_score)})")
        st.progress(ai_score / 100.0)
    with score_col3:
        badge_class = (
            "risk-low"
            if risk_level == "LOW"
            else ("risk-medium" if risk_level == "MEDIUM" else "risk-high")
        )
        st.markdown(
            f"**AI Assessed Risk Pattern:**<br><span class='risk-badge {badge_class}'>{risk_level} RISK</span>",
            unsafe_allow_html=True,
        )

    st.info(f"💡 **Executive Summary:** {ai_data.get('financial_summary', '')}")

    # -------------------------------------------------------------------------
    # Interactive Tabs for Dashboard Sections & Bonus Features
    # -------------------------------------------------------------------------
    tab_insights, tab_charts, tab_stream, tab_goal, tab_chat, tab_langchain = st.tabs(
        [
            "📋 Structured AI Plan",
            "📊 Interactive Charts",
            "⚡ Live Streaming Coach",
            "🎯 Goal Tracker & Export",
            "💬 Follow-Up AI Chat",
            "🔍 LangChain Internals",
        ]
    )

    # TAB 1: Structured JSON Insights
    with tab_insights:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 🎯 Top Financial Priorities")
            for idx, item in enumerate(ai_data.get("top_priorities", []), 1):
                st.markdown(f"**{idx}.** {item}")

            st.markdown("#### 💡 Personalized Budget Recommendations")
            for item in ai_data.get("budget_recommendations", []):
                st.markdown(f"- {item}")

        with col_b:
            st.markdown("#### 🏦 Smart Savings Strategy")
            for item in ai_data.get("savings_strategy", []):
                st.markdown(f"- {item}")

            st.markdown("#### 🗓️ Next-Month Action Plan")
            for item in ai_data.get("next_month_action_plan", []):
                st.markdown(f"✅ {item}")

        st.divider()
        st.markdown("#### 🔍 Category-by-Category AI Spending Analysis")
        for entry in ai_data.get("spending_analysis", []):
            cat_title = entry.get("category", "Category")
            with st.expander(f"📌 {cat_title}", expanded=True):
                st.markdown(f"**Observation:** {entry.get('observation', '')}")
                st.markdown(f"**Recommendation:** {entry.get('recommendation', '')}")

    # TAB 2: Interactive Plotly Visualizations (Bonus Feature)
    with tab_charts:
        chart_col1, chart_col2 = st.columns(2)
        label_map = dict(EXPENSE_CATEGORIES)
        exp_df = pd.DataFrame(
            [
                {
                    "Category": label_map.get(k, k.title()),
                    "Amount": v,
                }
                for k, v in metrics["expenses"].items()
                if v > 0
            ]
        )

        with chart_col1:
            st.markdown("##### Expense Distribution (Donut Chart)")
            if not exp_df.empty:
                fig_pie = px.pie(
                    exp_df,
                    names="Category",
                    values="Amount",
                    hole=0.48,
                    color_discrete_sequence=px.colors.qualitative.Safe,
                )
                fig_pie.update_layout(margin=dict(t=20, b=20, l=10, r=10))
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No non-zero expenses entered to plot.")

        with chart_col2:
            st.markdown("##### Actual vs. 50/30/20 Educational Benchmark")
            inc = metrics["monthly_income"]
            needs_keys = {"housing", "food", "transportation", "utilities", "healthcare", "insurance", "debt"}
            wants_keys = {"entertainment", "education", "other"}
            actual_needs = sum(metrics["expenses"].get(k, 0.0) for k in needs_keys)
            actual_wants = sum(metrics["expenses"].get(k, 0.0) for k in wants_keys)
            actual_savings = metrics["savings"]

            bench_df = pd.DataFrame(
                {
                    "Bucket": ["Needs (50%)", "Wants (30%)", "Savings (20%)"],
                    "Actual": [actual_needs, actual_wants, actual_savings],
                    "Benchmark (50/30/20)": [inc * 0.50, inc * 0.30, inc * 0.20],
                }
            )
            fig_bar = go.Figure(
                data=[
                    go.Bar(name="Your Numbers", x=bench_df["Bucket"], y=bench_df["Actual"], marker_color="#0F766E"),
                    go.Bar(name="50/30/20 Target", x=bench_df["Bucket"], y=bench_df["Benchmark (50/30/20)"], marker_color="#94A3B8"),
                ]
            )
            fig_bar.update_layout(barmode="group", margin=dict(t=20, b=20, l=10, r=10))
            st.plotly_chart(fig_bar, use_container_width=True)

    # TAB 3: Section 13 Live Streaming Recommendations
    with tab_stream:
        st.markdown("#### ⚡ Live Streaming AI Financial Coaching Memo (`st.write_stream`)")
        st.caption("Demonstrates real-time token streaming using `llm.stream()` and `NARRATIVE_CHAT_TEMPLATE`.")

        if not st.session_state.narrative_stream_text:
            try:
                stream_llm = get_chat_model(
                    api_key=st.session_state.openai_api_key,
                    model_name=selected_model,
                    temperature=temperature,
                    streaming=True,
                )
                streamed_output = st.write_stream(
                    stream_recommendations(stream_llm, chain_inputs)
                )
                st.session_state.narrative_stream_text = str(streamed_output)
            except Exception as exc:
                st.error(f"Streaming error: {exc}")
        else:
            st.markdown(st.session_state.narrative_stream_text)
            if st.button("🔄 Re-Stream Coaching Memo"):
                st.session_state.narrative_stream_text = ""
                st.rerun()

    # TAB 4: Bonus Goal Tracker & Report Downloads
    with tab_goal:
        st.markdown(f"#### 🎯 Financial Goal Tracker: **{goal}**")
        target_amt = metrics.get("goal_target_amount", 10000.0)

        g_col1, g_col2, g_col3 = st.columns(3)
        g_col1.metric("Goal Target Amount", f"{curr}{target_amt:,.2f}")
        g_col2.metric("Current Monthly Savings", f"{curr}{metrics['savings']:,.2f}")
        if metrics["savings"] > 0:
            months_needed = target_amt / metrics["savings"]
            g_col3.metric("Estimated Time at Current Savings", f"{months_needed:.1f} months")
            progress_pct = min(1.0, metrics["savings"] / target_amt)
            st.progress(progress_pct, text=f"1st Month Contribution: {progress_pct*100:.1f}% of target")
        else:
            g_col3.metric("Estimated Time at Current Savings", "∞ (0 Savings)")
            st.warning("Increase monthly savings above 0 to begin progressing toward your goal target.")

        st.divider()
        st.markdown("#### 📥 Export Financial Analysis Report")
        dl_col1, dl_col2 = st.columns(2)
        html_report = generate_html_report(
            currency_symbol=curr,
            goal=goal,
            metrics=metrics,
            ai_data=ai_data,
            narrative_text=st.session_state.narrative_stream_text or ai_data.get("financial_summary", ""),
        )
        with dl_col1:
            st.download_button(
                "📄 Download Printable HTML/PDF Report",
                data=html_report,
                file_name="FinWise_AI_Financial_Report.html",
                mime="text/html",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                "🧾 Download Structured JSON Output",
                data=json.dumps(ai_data, indent=2),
                file_name="FinWise_AI_Structured_Output.json",
                mime="application/json",
                use_container_width=True,
            )

    # TAB 5: Bonus Multi-Turn Follow-up Chat
    with tab_chat:
        st.markdown("#### 💬 Ask Follow-Up Questions About Your Budget")
        st.caption("Uses `SystemMessage`, `HumanMessage`, and `AIMessage` conversation history.")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_q = st.chat_input("Ask e.g. 'How can I reduce my food & dining costs by 15%?'")
        if user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            with st.chat_message("assistant"):
                chat_llm = get_chat_model(
                    api_key=st.session_state.openai_api_key,
                    model_name=selected_model,
                    temperature=temperature,
                    streaming=True,
                )
                context_str = json.dumps({"metrics": metrics, "ai_analysis": ai_data})
                ans = st.write_stream(
                    run_followup_chat(
                        chat_llm,
                        context_str,
                        st.session_state.chat_history[:-1],
                        user_q,
                    )
                )
            st.session_state.chat_history.append({"role": "assistant", "content": str(ans)})

    # TAB 6: LangChain Architecture & Rubric Verification
    with tab_langchain:
        st.markdown("#### 🧩 LangChain PromptTemplate, Message Objects & Schema Verification")
        st.success(f"Schema Validation Status: {st.session_state.json_status_msg}")

        with st.expander("1. Rendered `PromptTemplate` (Section 10 & 11)"):
            st.code(FINANCIAL_PROMPT_TEMPLATE.format(**chain_inputs), language="markdown")

        with st.expander("2. `SystemMessage`, `HumanMessage`, and `AIMessage` Representation (Section 10)"):
            demo_msgs = build_message_demo_conversation(
                chain_inputs, ai_data.get("financial_summary", "")
            )
            for m in demo_msgs:
                st.markdown(f"**`{type(m).__name__}`**")
                st.code(m.content)

        with st.expander("3. Validated Structured JSON Output (Section 12)"):
            st.json(ai_data)

st.divider()
st.caption(EDUCATIONAL_DISCLAIMER)
