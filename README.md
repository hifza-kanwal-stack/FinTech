# 💳 FinWise AI — AI-Powered Personal Financial Analysis & Smart Budget Assistant

> **⚠️ EDUCATIONAL USE ONLY:** FinWise AI is an educational prototype built for the *Building LLM Applications with LangChain* course assignment. It does not provide guaranteed investment advice, execute financial transactions, or connect to real bank accounts. Always consult a qualified financial professional.

---

## 🚀 Key Features
- **API Key Security Gate:** Requests and validates the user's OpenAI API key (`sk-...`) before unlocking the financial console.
- **Deterministic Python Engine (`src/financial_calculator.py`):** Computes total expenses across 10 categories, remaining income, savings ratio, expense ratio, debt ratio, and a weighted `0–100` preliminary health score with divide-by-zero protection.
- **LangChain Integration (`src/prompts.py`, `src/chains.py`):**
  - `ChatOpenAI` model integration (`gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`).
  - Reusable `PromptTemplate` & `ChatPromptTemplate`.
  - Explicit `SystemMessage`, `HumanMessage`, and `AIMessage` handling.
  - Reusable `LLMChain` returning strict, validated JSON output.
- **Real-Time Streaming:** Streams personalized coaching recommendations live using `llm.stream()` and `st.write_stream()`.
- **Dual Caching (`src/cache_manager.py`):** Switch between `InMemoryCache` (RAM) and `SQLiteCache` (disk-backed `.db`) via `set_llm_cache()`.
- **All Bonus Features Included:** Interactive Plotly charts, CSV expense upload/template, Financial Goal Tracker, Printable HTML/PDF Report export, and Multi-turn AI follow-up chat.

---

## 🛠️ Local Setup & Run Instructions

1. **Extract or Clone the repository:**
   ```bash
   cd finwise_ai
   ```

2. **Create a virtual environment & install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **(Optional) Configure `.env`:**
   ```bash
   cp .env.example .env
   ```
   *Note: You can either set `OPENAI_API_KEY=sk-...` in `.env` OR enter your API key directly in the FinWise AI login screen when the app launches.*

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
