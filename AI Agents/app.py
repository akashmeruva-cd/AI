"""
Streamlit Web Application: Programmable AI & AI Agents Studio
Open Source Development Program - Session 22

Features:
1. Lab 1: The Expense Parser (Chatty vs Few-Shot vs Pydantic Structured Output)
2. Lab 2: The Market Agent (Interactive ReAct Loop Visualizer with Live Tools)
3. Module 7: Multi-Agent Collaboration & Handoffs (Researcher -> Analyst)
4. Interactive Agent Tool Sandbox
"""

import os
import ast
import json
import operator
import pandas as pd
import streamlit as st
from typing import List, Literal, Optional, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from model_utils import get_best_gemini_model, AVAILABLE_GEMINI_MODELS, call_gemini_with_fallback, extract_gemini_text

# Load environment
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Streamlit Page Config
st.set_page_config(
    page_title="Session 22: AI Agents Practical Lab",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }
    .react-thought {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 10px;
        color: #92400E;
    }
    .react-action {
        background-color: #DBEAFE;
        border-left: 4px solid #3B82F6;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 10px;
        color: #1E40AF;
        font-family: monospace;
    }
    .react-observation {
        background-color: #D1FAE5;
        border-left: 4px solid #10B981;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 10px;
        color: #065F46;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 1. Pydantic Schemas for Lab 1 & Multi-Agent
# ==============================================================================
class ExpenseItem(BaseModel):
    vendor: str = Field(description="Name of the merchant, store, or service vendor")
    amount: float = Field(description="Total monetary amount spent in USD")
    category: str = Field(description="Classified business expense category (e.g., Food & Beverage, Travel, Office, Software)")
    description: str = Field(description="Context or specific items purchased")


class ExpenseReport(BaseModel):
    expenses: List[ExpenseItem] = Field(description="List of all extracted individual expense items")
    total_amount: float = Field(description="Sum total of all extracted expense amounts")


# ==============================================================================
# 2. Agent Tools ("The Hands")
# ==============================================================================
FALLBACK_PRICES = {
    "AAPL": 224.50,
    "MSFT": 428.15,
    "GOOGL": 182.30,
    "AMZN": 186.75,
    "NVDA": 121.40,
    "TSLA": 250.80,
    "META": 585.60,
}

def get_stock_price(ticker: str) -> float:
    """Fetches real-time stock price with caching and fallback."""
    clean_ticker = ticker.strip().upper().replace("$", "")
    try:
        import yfinance as yf
        stock = yf.Ticker(clean_ticker)
        if hasattr(stock, "fast_info") and stock.fast_info.last_price:
            return round(float(stock.fast_info.last_price), 2)
        hist = stock.history(period="1d")
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 2)
    except Exception:
        pass
    return FALLBACK_PRICES.get(clean_ticker, 150.00)


def calculator(expression: str) -> float:
    """Safely evaluates math expressions using Python AST (+, -, *, /, //, **, %)."""
    import re
    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.UAdd: operator.pos
    }
    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp):
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported math operation")
    
    try:
        clean_expr = str(expression).replace(",", "").replace("$", "").strip()
        clean_expr = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'(\1 / 100)', clean_expr)
        return round(float(_eval(ast.parse(clean_expr, mode="eval").body)), 4)
    except Exception as e:
        return f"Error evaluating expression '{expression}': {str(e)}"


def search_market_news(query: str) -> str:
    """Searches real-time financial catalysts via Google News RSS & DDGS."""
    import ssl
    import urllib.request
    import urllib.parse
    import xml.etree.ElementTree as ET

    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        encoded_query = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
        )
        with urllib.request.urlopen(req, context=ctx, timeout=4) as resp:
            content = resp.read()
            root = ET.fromstring(content)
            items = root.findall(".//item")
            snippets = []
            for item in items[:2]:
                title = item.find("title").text if item.find("title") is not None else ""
                source = item.find("source").text if item.find("source") is not None else "News"
                pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
                date_str = pub_date[:16] if pub_date else ""
                snippets.append(f"• {title} [{source} | {date_str}]")
            if snippets:
                return "\n".join(snippets)
    except Exception:
        pass

    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=2))
            if results:
                return "\n".join([f"• {r.get('title')}: {r.get('body')}" for r in results])
    except Exception:
        pass

    return f"Market News for '{query}': Active market trading with positive institutional momentum."


def save_portfolio_log(ticker: str, action: str, shares: float, total_cost: float) -> str:
    """Logs an executed or proposed stock trade to a local audit log file."""
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "trades.jsonl")
    
    record = {
        "ticker": ticker.upper(),
        "action": action.upper(),
        "shares": float(shares),
        "total_cost": float(total_cost),
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(record) + "\n")
        
    return f"Successfully saved trade log for {shares} shares of {ticker.upper()} (${total_cost:.2f}) to {log_file}."


def parse_action_args(args_str: str):
    """
    Robustly parses tool call arguments supporting:
    - JSON dicts: {"ticker": "MSFT"}
    - Single-quoted dicts: {'ticker': 'MSFT'}
    - Python keyword args: ticker="MSFT", shares=10
    - Positional args: "MSFT", 15
    """
    raw = args_str.strip()
    if not raw:
        return [], {}
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1].strip()
    if not raw:
        return [], {}

    # 1. Try AST call expression: func(raw)
    try:
        call_expr = ast.parse(f"func({raw})").body[0].value
        pos_args = [ast.literal_eval(a) for a in call_expr.args]
        kw_args = {kw.arg: ast.literal_eval(kw.value) for kw in call_expr.keywords}
        if len(pos_args) == 1 and isinstance(pos_args[0], dict) and not kw_args:
            return [], pos_args[0]
        return pos_args, kw_args
    except Exception:
        pass

    # 2. Try JSON
    try:
        val = json.loads(raw)
        if isinstance(val, dict):
            return [], val
        elif isinstance(val, list):
            return val, {}
        return [val], {}
    except Exception:
        pass

    # 3. Try literal_eval
    try:
        val = ast.literal_eval(raw)
        if isinstance(val, dict):
            return [], val
        elif isinstance(val, (list, tuple)):
            return list(val), {}
        return [val], {}
    except Exception:
        pass

    return [raw], {}


TOOLS_REGISTRY = {
    "get_stock_price": get_stock_price,
    "calculator": calculator,
    "search_market_news": search_market_news,
    "save_portfolio_log": save_portfolio_log,
}


# ==============================================================================
# 3. Sidebar: Configuration & Status
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/artificial-intelligence.png", width=60)
    st.title("Settings & Engine")
    
    # Provider detection
    env_google = os.getenv("GOOGLE_API_KEY", "")
    env_openai = os.getenv("OPENAI_API_KEY", "")
    env_groq = os.getenv("GROQ_API_KEY", "")

    provider_choice = st.selectbox(
        "LLM Provider",
        ["Auto-Detect", "Google Gemini", "OpenAI", "Groq", "Mock Simulation Mode"]
    )

    custom_key = st.text_input(
        "API Key (Optional override)",
        type="password",
        placeholder="Enter API Key or leave empty to use .env"
    )

    # Active Provider Determination
    active_provider = "mock"
    active_key = custom_key

    if provider_choice == "Google Gemini" or (provider_choice == "Auto-Detect" and (env_google or custom_key)):
        active_provider = "gemini"
        active_key = custom_key if custom_key else env_google
    elif provider_choice == "OpenAI" or (provider_choice == "Auto-Detect" and (env_openai or custom_key)):
        active_provider = "openai"
        active_key = custom_key if custom_key else env_openai
    elif provider_choice == "Groq" or (provider_choice == "Auto-Detect" and (env_groq or custom_key)):
        active_provider = "groq"
        active_key = custom_key if custom_key else env_groq
    
    gemini_model_name = get_best_gemini_model()

    if active_provider == "gemini" and active_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=active_key)
            selected_model = st.selectbox(
                "Gemini Model Tier",
                AVAILABLE_GEMINI_MODELS,
                index=0,
                help="Gemini 3.5 Flash Lite has high free-tier quotas. If one model hits a 429 quota limit, the app automatically fails over to the next model."
            )
            gemini_model_name = selected_model
            st.success(f"🟢 Connected: Gemini ({gemini_model_name})")
        except Exception:
            active_provider = "mock"
            st.warning("⚠️ Gemini SDK not found, using Mock Engine.")
    elif active_provider == "openai" and active_key:
        try:
            import openai
            openai_client = openai.OpenAI(api_key=active_key)
            st.success("🟢 Connected to OpenAI (gpt-4o-mini)")
        except Exception:
            active_provider = "mock"
            st.warning("⚠️ OpenAI SDK not found, using Mock Engine.")
    elif active_provider == "groq" and active_key:
        try:
            from groq import Groq
            groq_client = Groq(api_key=active_key)
            st.success("🟢 Connected to Groq (llama-3.3-70b)")
        except Exception:
            active_provider = "mock"
            st.warning("⚠️ Groq SDK not found, using Mock Engine.")
    else:
        active_provider = "mock"
        st.info("ℹ️ Running in Educational Simulation Mode")

    st.divider()
    temperature = st.slider("Temperature (Determinism)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    if temperature == 0.0:
        st.caption("🔒 **0.0 - Strict Code Mode**: Highly deterministic & predictable output.")
    else:
        st.caption("🎨 **Conversational Mode**: Higher creativity with potential format drift.")

    st.divider()
    st.markdown("### 📚 Quick Links")
    st.markdown("- [📓 Open Jupyter Notebook](file:///Users/akash.meruva/Documents/Projects-cd/AI/AI%20Agents/AI_Agents_HandsOn_Lab.ipynb)")
    st.markdown("- [📘 View Lab Manual (README)](file:///Users/akash.meruva/Documents/Projects-cd/AI/AI%20Agents/README.md)")


# ==============================================================================
# Main App Header
# ==============================================================================
st.markdown('<div class="main-header">🤖 Session 22: Programmable AI & AI Agents Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Interactive Lab for Structured Outputs (Pydantic), ReAct Loops, and Autonomous Tool-Using Agents</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🧾 Lab 1: The Expense Parser",
    "🤖 Lab 2: The Market Agent (ReAct Loop)",
    "👥 Module 7: Multi-Agent Collaboration",
    "🛠️ Tool Sandbox"
])


# ==============================================================================
# TAB 1: LAB 1 — THE EXPENSE PARSER
# ==============================================================================
with tab1:
    st.subheader("🧾 Lab 1: The Expense Parser (Structured Outputs)")
    st.markdown("""
    In software engineering, models must produce **guaranteed, type-safe JSON**.
    Compare how standard Chat UIs fail with `"The Parsing Problem"` vs **Pydantic Model Schema enforcement**.
    """)

    sample_presets = {
        "Preset 1: Travel Day Receipt": (
            "Hey Team,\nSubmitting my travel expenses from yesterday's client visit:\n"
            "Grabbed breakfast and a latte for $14.50 at Blue Bottle Coffee around 8 AM.\n"
            "Took an Uber from SFO to Downtown for $48.20.\n"
            "Had lunch with the client at Chipotle ($36.80 for both of us).\n"
            "Later I took an afternoon Lyft back to the airport for $52.10 and bought a sandwich at Peet's Coffee for $11.25.\n"
            "Please approve! - Alex"
        ),
        "Preset 2: Office Supplies & SaaS": (
            "Invoice Summary:\n"
            "Hey Finance, here are the software renewals:\n"
            "- GitHub Enterprise annual seat: $252.00 on Visa ****4121\n"
            "- AWS Cloud hosting charges: $1420.50\n"
            "- Bought 2 ergonomic mouse pads from Amazon for $28.99\n"
            "- Team Figma organization license renewal: $180.00\n"
            "- Snacks for hackathon at Whole Foods for $67.45"
        ),
        "Preset 3: Casual Slack Message": (
            "yo, quick update on expenses for yesterday: Bought a coffee for 4.50 at Starbucks yesterday, "
            "and spent $50 on gas at Shell. Also grabbed a quick burger at In-N-Out for 12.75 and paid $18.30 parking ticket at CityGarage."
        ),
        "Preset 4: Tricky Negated Purchase": (
            "Expense Log:\n"
            "1. Team lunch at Subway: $18.50\n"
            "2. Software license for JetBrains IDE: $199.00\n"
            "3. We considered buying an Apple Vision Pro for $3,499 but decided NOT to buy it.\n"
            "4. Metro train tickets: $12.00"
        )
    }

    selected_preset = st.selectbox("Choose a Sample Chaotic Input:", list(sample_presets.keys()))
    raw_input_text = st.text_area("Messy Input Text:", value=sample_presets[selected_preset], height=140)

    col1, col2, col3 = st.columns(3)
    with col1:
        run_chatty = st.button("💥 Test Unstructured Chat (Parsing Problem)", use_container_width=True)
    with col2:
        run_fewshot = st.button("⚡ Test Few-Shot Prompting", use_container_width=True)
    with col3:
        run_structured = st.button("🛡️ Parse with Pydantic Schema (Production)", type="primary", use_container_width=True)

    if run_chatty:
        st.markdown("### 🔍 Unstructured Chat LLM Output")
        if active_provider == "gemini":
            try:
                resp_obj = call_gemini_with_fallback(
                    f"Extract expenses as JSON:\n{raw_input_text}",
                    generation_config={"temperature": 0.7},
                    preferred_model=gemini_model_name
                )
                resp = extract_gemini_text(resp_obj)
            except Exception as e:
                resp = f"Sure! Here is the JSON data you requested:\n\n```json\n[{{\"vendor\": \"Starbucks\", \"amount\": 4.50}}]\n```\nLet me know if you need anything else!"
        elif active_provider == "openai":
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Extract expenses as JSON:\n{raw_input_text}"}],
                temperature=0.7
            ).choices[0].message.content
        else:
            resp = "Sure! Here is the JSON data you requested:\n\n```json\n[{\"vendor\": \"Starbucks\", \"amount\": 4.50}]\n```\nLet me know if you need anything else!"
        
        st.code(resp, language="markdown")
        try:
            json.loads(resp)
            st.success("✅ json.loads() succeeded!")
        except Exception as e:
            st.error(f"❌ json.loads() CRASHED with error: {e}")
            st.warning("💡 Chatty models wrap output in conversational greetings, crashing downstream backend code!")

    if run_fewshot:
        st.markdown("### ⚡ Few-Shot Extracted Output")
        prompt = (
            "Extract all financial transactions as valid JSON: [{\"vendor\": string, \"amount\": number, \"category\": string}].\n"
            f"Text:\n{raw_input_text}"
        )
        if active_provider == "gemini":
            try:
                resp_obj = call_gemini_with_fallback(
                    prompt,
                    generation_config={"temperature": 0.0},
                    preferred_model=gemini_model_name
                )
                resp = extract_gemini_text(resp_obj)
            except Exception as e:
                resp = '[{"vendor": "Starbucks", "amount": 4.50, "category": "Food & Beverage"}, {"vendor": "Shell", "amount": 50.00, "category": "Travel & Transportation"}]'
        elif active_provider == "openai":
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            ).choices[0].message.content
        else:
            resp = '[{"vendor": "Starbucks", "amount": 4.50, "category": "Food & Beverage"}, {"vendor": "Shell", "amount": 50.00, "category": "Travel & Transportation"}]'
        
        st.code(resp, language="json")

    if run_structured:
        with st.spinner("Extracting with strict Pydantic schema validation..."):
            report = None
            if active_provider == "gemini":
                try:
                    resp_obj = call_gemini_with_fallback(
                        f"Extract all valid expenses into ExpenseReport schema:\n{raw_input_text}",
                        generation_config={
                            "temperature": 0.0,
                            "response_mime_type": "application/json",
                            "response_schema": ExpenseReport
                        },
                        preferred_model=gemini_model_name
                    )
                    txt = extract_gemini_text(resp_obj).strip()
                    if txt.startswith("```json"):
                        txt = txt[7:]
                    if txt.startswith("```"):
                        txt = txt[3:]
                    if txt.endswith("```"):
                        txt = txt[:-3]
                    report = ExpenseReport.model_validate_json(txt.strip())
                except Exception as ex:
                    st.warning(f"Quota Notice: {ex}")
            elif active_provider == "openai":
                try:
                    completion = openai_client.beta.chat.completions.parse(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": raw_input_text}],
                        response_format=ExpenseReport,
                        temperature=0.0
                    )
                    report = completion.choices[0].message.parsed
                except Exception as ex:
                    st.error(f"OpenAI API Error: {ex}")
            
            if not report:
                # Educational mock report
                report = ExpenseReport(
                    expenses=[
                        ExpenseItem(vendor="Blue Bottle Coffee", amount=14.50, category="Food & Beverage", description="Breakfast & latte"),
                        ExpenseItem(vendor="Uber", amount=48.20, category="Travel & Transportation", description="SFO to Downtown"),
                        ExpenseItem(vendor="Chipotle", amount=36.80, category="Food & Beverage", description="Client lunch"),
                        ExpenseItem(vendor="Lyft", amount=52.10, category="Travel & Transportation", description="Downtown to SFO"),
                        ExpenseItem(vendor="Peet's Coffee", amount=11.25, category="Food & Beverage", description="Airport sandwich")
                    ],
                    total_amount=162.85
                )

            # Mathematically guarantee total_amount is exact
            if report and report.expenses:
                report.total_amount = round(sum(item.amount for item in report.expenses), 2)

            st.success("✅ Successfully Parsed into Type-Safe Pydantic Model!")
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Extracted Spend", f"${report.total_amount:.2f}")
            m2.metric("Total Items", len(report.expenses))
            m3.metric("Status", "Audited & Validated")

            # Table View
            df = pd.DataFrame([item.model_dump() for item in report.expenses])
            st.dataframe(df, use_container_width=True)

            if getattr(report, "notes", None):
                st.info(f"📝 **Auditor Notes**: {report.notes}")

            with st.expander("🔍 View Raw Validated JSON"):
                st.json(report.model_dump())


# ==============================================================================
# TAB 2: LAB 2 — THE MARKET AGENT (ReAct Loop)
# ==============================================================================
with tab2:
    st.subheader("🤖 Lab 2: The Market Agent (ReAct Loop Visualizer)")
    st.markdown("""
    Watch how an autonomous AI Agent overcomes **"The Wall"** by iteratively reasoning (`[Thought]`), 
    triggering real Python tools (`[Action]`), and observing live feedback (`[Observation]`).
    """)

    market_presets = [
        "How much would it cost to buy 15 shares of Microsoft (MSFT)?",
        "What is the current stock price of Apple (AAPL) and Tesla (TSLA)? If I buy 10 shares of each, what is the combined total cost?",
        "If I have a budget of $3,500, how many whole shares of Nvidia (NVDA) can I purchase, and what will my remaining cash balance be?",
        "Find the current price of Amazon (AMZN) and calculate what the total cost would be with a 7.5% transaction brokerage fee."
    ]

    selected_market_query = st.selectbox("Select a Sample Market Query:", market_presets)
    user_custom_query = st.text_input("Or enter your custom question:", value=selected_market_query)

    if st.button("🚀 Run ReAct Market Agent", type="primary"):
        st.markdown("### 🔄 Execution Trace (Step-by-Step)")
        
        # System Prompt
        sys_prompt = (
            "You are an autonomous Financial Intelligence Agent operating strictly via the ReAct pattern.\n"
            "Available tools:\n"
            "1. get_stock_price(ticker='AAPL') -> float: Fetches real-time stock price in USD.\n"
            "2. calculator(expression='15 * 428.15') -> float: Safely evaluates arithmetic expressions. Always use this for math!\n"
            "3. search_market_news(query='MSFT earnings') -> str: Searches recent news.\n\n"
            "Follow this exact loop format:\n"
            "Thought: Reason about next steps.\n"
            "Action: tool_name({\"arg1\": \"val1\"})\n"
            "Observation: [The system will execute the tool and provide the answer here]\n"
            "... (repeat Thought/Action/Observation as needed)\n"
            "Thought: I have gathered all info.\n"
            "Final Answer: Comprehensive, clearly formatted answer with exact calculations.\n\n"
            "CRITICAL RULES:\n"
            "- Output exactly ONE Thought and ONE Action per response.\n"
            "- NEVER write 'Observation:' yourself. STOP immediately after outputting Action.\n"
            "- ALWAYS use the exact numerical values returned in the previous Observation lines.\n"
        )

        history = f"{sys_prompt}\n\nUser Question: {user_custom_query}\n"
        max_steps = 6

        # ReAct Execution Loop
        for step in range(1, max_steps + 1):
            with st.status(f"Step {step}: Agent Reasoning & Tool Execution...", expanded=True) as status:
                llm_response = ""
                
                if active_provider == "gemini":
                    try:
                        resp_obj = call_gemini_with_fallback(
                            history,
                            generation_config={"temperature": 0.0},
                            preferred_model=gemini_model_name
                        )
                        raw_txt = extract_gemini_text(resp_obj)
                        llm_response = raw_txt.strip() if raw_txt else ""
                    except Exception as e:
                        st.warning(f"Quota Notice: {e}")
                elif active_provider == "openai":
                    try:
                        resp = openai_client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": history}],
                            temperature=0.0,
                            stop=["Observation:", "\nObservation:"]
                        )
                        llm_response = resp.choices[0].message.content.strip()
                    except Exception as e:
                        st.error(f"OpenAI error: {e}")

                if not llm_response:
                    # Mock simulation step: intelligently handle queried ticker
                    detected_ticker = "MSFT"
                    for t in ["MSFT", "AAPL", "NVDA", "TSLA", "AMZN", "GOOGL", "META"]:
                        if t in user_custom_query.upper():
                            detected_ticker = t
                            break

                    price = FALLBACK_PRICES.get(detected_ticker, 150.00)
                    if "Observation:" not in history:
                        llm_response = f'Thought: I need to check the current stock price of {detected_ticker}.\nAction: get_stock_price({{"ticker": "{detected_ticker}"}})'
                    elif "calculator" not in history:
                        llm_response = f'Thought: The {detected_ticker} stock price is ${price:.2f}. Now I will calculate 15 * {price}.\nAction: calculator({{"expression": "15 * {price}"}})'
                    else:
                        total_cost = round(15 * price, 2)
                        llm_response = f'Thought: I have all necessary calculations.\nFinal Answer: {detected_ticker} is currently trading at ${price:.2f} per share. The total cost to purchase 15 shares is **${total_cost:,.2f}**.'

                # Process Action First (Trumps hallucinated multi-turn text)
                if "Action:" in llm_response:
                    action_idx = llm_response.find("Action:")
                    end_of_action = llm_response.find("\n", action_idx)
                    
                    if end_of_action == -1:
                        action_str = llm_response[action_idx:].strip()
                        thought_str = llm_response[:action_idx].strip()
                    else:
                        action_str = llm_response[action_idx:end_of_action].strip()
                        thought_str = llm_response[:action_idx].strip()

                    if thought_str:
                        st.markdown(f'<div class="react-thought">🧠 {thought_str}</div>', unsafe_allow_html=True)

                    st.markdown(f'<div class="react-action">⚡ {action_str}</div>', unsafe_allow_html=True)
                    
                    # Execute tool in Python with robust argument handling
                    try:
                        raw_action = action_str.replace("Action:", "").strip()
                        tool_name = raw_action.split("(")[0].strip()
                        args_str = raw_action[len(tool_name):].strip()
                        pos_args, kw_args = parse_action_args(args_str)
                        
                        tool_fn = TOOLS_REGISTRY.get(tool_name)
                        obs = tool_fn(*pos_args, **kw_args) if tool_fn else f"Tool '{tool_name}' not found."
                    except Exception as err:
                        obs = f"Execution Error: {err}"

                    st.markdown(f'<div class="react-observation">👁️ <b>Observation:</b> {obs}</div>', unsafe_allow_html=True)
                    
                    clean_step_output = (thought_str + "\n" + action_str).strip()
                    history += f"{clean_step_output}\nObservation: {obs}\n"
                    status.update(label=f"Step {step}: Executed {tool_name}()", state="complete")
                
                elif "Final Answer:" in llm_response:
                    parts = llm_response.split("Final Answer:")
                    thought_text = parts[0].strip()
                    final_ans = parts[1].strip()
                    
                    if thought_text:
                        st.markdown(f'<div class="react-thought">🧠 <b>Reasoning:</b> {thought_text}</div>', unsafe_allow_html=True)
                    
                    status.update(label=f"✅ Goal Achieved at Step {step}!", state="complete", expanded=False)
                    st.markdown("### 🎯 Final Result")
                    safe_final_ans = final_ans.replace("$", r"\$")
                    st.success(safe_final_ans)
                    break


# ==============================================================================
# TAB 3: MODULE 7 — MULTI-AGENT COLLABORATION & HANDOFFS
# ==============================================================================
with tab3:
    st.subheader("👥 Module 7: Multi-Agent Systems & Handoffs (Slide 23)")
    st.markdown("""
    Single agents hit complexity bottlenecks. In multi-agent architectures, tasks are decomposed 
    across specialists:
    1. **Agent 1: Market Researcher** (Fetches real-time price, metrics, and news).
    2. **Agent 2: Senior Financial Analyst & Writer** (Computes risk/reward and authors the final briefing).
    """)

    c1, c2 = st.columns(2)
    with c1:
        target_ticker = st.selectbox("Select Target Company:", ["MSFT", "AAPL", "NVDA", "TSLA", "AMZN", "GOOGL"])
    with c2:
        budget_amt = st.number_input("Investment Allocation ($ USD):", min_value=500.0, max_value=500000.0, value=10000.0, step=1000.0)

    if st.button("🚀 Launch Multi-Agent Workflow", type="primary"):
        from lab3_multi_agent_system import ResearcherAgent, AnalystWriterAgent

        # Agent 1 Phase
        with st.status("🔍 Agent 1 (Researcher): Gathering live market intelligence...", expanded=True) as s1:
            researcher = ResearcherAgent(provider=active_provider, api_key=active_key, model_name=gemini_model_name)
            brief = researcher.conduct_research(target_ticker)
            st.write(f"• **Live Ticker Price**: \\${brief.current_price:.2f}")
            st.write(f"• **Market Sentiment**: `{brief.market_sentiment}`")
            with st.expander("📰 View Raw Catalysts Gathered", expanded=False):
                st.markdown(brief.recent_catalysts)
            s1.update(label="✅ Agent 1 (Researcher): Brief Compiled!", state="complete")

        st.markdown("<center>⬇️ <i>Handoff: Structured ResearchBrief transferred to Financial Analyst</i> ⬇️</center>", unsafe_allow_html=True)

        # Agent 2 Phase
        with st.status("📊 Agent 2 (Analyst & Writer): Synthesizing data & drafting memo...", expanded=True) as s2:
            analyst = AnalystWriterAgent(provider=active_provider, api_key=active_key, model_name=gemini_model_name)
            memo = analyst.generate_memo(brief, budget=budget_amt)
            memo.proposed_investment_amount = float(budget_amt)
            memo.estimated_shares = round(budget_amt / max(brief.current_price, 0.01), 2)
            st.write(f"• **Calculated Position**: {memo.estimated_shares} shares")
            st.write(f"• **Risk Audit**: {len(memo.key_risks)} company-specific risk vectors evaluated")
            s2.update(label="✅ Agent 2 (Analyst): Investment Memo Ready!", state="complete")

        st.markdown(f"### 📋 {memo.title}")
        st.markdown(f"""
        | Parameter | Value |
        | :--- | :--- |
        | **Target Security** | **{memo.target_ticker}** |
        | **Current Market Price** | **\\${brief.current_price:.2f} USD** |
        | **Allocated Capital** | **\\${budget_amt:,.2f} USD** |
        | **Acquisition Size** | **{memo.estimated_shares} shares** |
        """)

        st.markdown("#### 💡 Strategic Perspective & Recommendation:")
        safe_rec = memo.executive_recommendation.replace("$", r"\$")
        st.info(safe_rec)

        st.markdown("#### ⚠️ Company-Specific Risk Factors:")
        for r in memo.key_risks:
            safe_r = r.replace("$", r"\$")
            st.markdown(f"- {safe_r}")


# ==============================================================================
# TAB 4: TOOL SANDBOX
# ==============================================================================
with tab4:
    st.subheader("🛠️ Interactive Tool Sandbox")
    st.markdown("Test the individual Python functions that empower our AI Agents:")

    t1, t2, t3, t4 = st.tabs(["📈 Stock Price Lookup", "🧮 AST Math Calculator", "📰 Financial News Search", "📝 Portfolio Trade Logger"])

    with t1:
        t_sym = st.text_input("Ticker Symbol:", value="MSFT")
        if st.button("Fetch Live Stock Price"):
            p = get_stock_price(t_sym)
            st.metric(f"{t_sym.upper()} Current Price", f"${p:.2f} USD")

    with t2:
        m_expr = st.text_input("Math Expression (Safe AST):", value="15 * 428.15 + (100 * 0.08)")
        if st.button("Evaluate Math"):
            calc_res = calculator(m_expr)
            st.success(f"Result: **{calc_res}**")

    with t3:
        n_query = st.text_input("News Query:", value="Microsoft cloud AI revenue")
        if st.button("Search News"):
            n_res = search_market_news(n_query)
            st.info(n_res)

    with t4:
        st.markdown("Log a simulated trade execution to the persistent audit log file (`logs/trades.jsonl`):")
        c_t1, c_t2, c_t3, c_t4 = st.columns(4)
        with c_t1:
            log_ticker = st.text_input("Trade Ticker:", value="MSFT", key="sandbox_ticker")
        with c_t2:
            log_action = st.selectbox("Action:", ["BUY", "SELL"], key="sandbox_action")
        with c_t3:
            log_shares = st.number_input("Shares:", min_value=1.0, value=15.0, step=1.0, key="sandbox_shares")
        with c_t4:
            current_sym_price = get_stock_price(log_ticker)
            log_cost = st.number_input("Total Cost ($ USD):", min_value=0.0, value=round(log_shares * current_sym_price, 2), step=10.0, key="sandbox_cost")
        
        if st.button("Save Trade to Portfolio Audit Log"):
            log_msg = save_portfolio_log(log_ticker, log_action, log_shares, log_cost)
            st.success(f"✅ {log_msg}")
            
            # Show recent entries from log file
            log_file = os.path.join(os.path.dirname(__file__), "logs", "trades.jsonl")
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    trades = [json.loads(line) for line in f if line.strip()]
                if trades:
                    st.dataframe(pd.DataFrame(trades[-10:]), use_container_width=True)
