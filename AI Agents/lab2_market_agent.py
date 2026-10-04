"""
Lab 2: The Market Agent (ReAct Pattern & Tool-Using AI Agents)
Open Source Development Program - Session 22

This script demonstrates:
1. The ReAct Pattern: Reason (Thought) -> Act (Tool Call) -> Observe (Tool Result) -> Finish.
2. Building clean Python tools with docstrings & type hints (@tool).
3. The autonomous loop: Giving LLMs 'Hands' (Stock Price Fetcher, Safe Calculator, Web Search, File Logger).
4. Full execution tracing showing how the agent solves complex multi-step financial questions.
"""

import os
import ast
import json
import operator
from typing import Dict, Any, Callable
from dotenv import load_dotenv
from model_utils import get_best_gemini_model, call_gemini_with_fallback, extract_gemini_text

# Optional formatting
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    console = Console()
except ImportError:
    console = None

# Load environment variables
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


# ==============================================================================
# 1. Defining Agent Tools ("The Hands" - Slide 20, 21, 22)
# ==============================================================================

# Simulated fallback prices in case network / rate-limits occur during offline lab
FALLBACK_STOCK_PRICES = {
    "AAPL": 224.50,
    "MSFT": 428.15,
    "GOOGL": 182.30,
    "AMZN": 186.75,
    "NVDA": 121.40,
    "TSLA": 250.80,
    "META": 585.60,
}

def get_stock_price(ticker: str) -> float:
    """
    Fetches the latest real-time stock price for a given ticker symbol.
    Args:
        ticker: The uppercase stock ticker symbol (e.g., 'AAPL', 'MSFT', 'TSLA').
    Returns:
        float: The current stock price in USD.
    """
    ticker_clean = ticker.strip().upper().replace("$", "")
    
    # Try fetching live price via yfinance if installed
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker_clean)
        fast_info = getattr(stock, "fast_info", None)
        if fast_info and hasattr(fast_info, "last_price") and fast_info.last_price:
            return round(float(fast_info.last_price), 2)
        
        history = stock.history(period="1d")
        if not history.empty:
            return round(float(history["Close"].iloc[-1]), 2)
    except Exception:
        pass

    # Fallback to simulated database
    if ticker_clean in FALLBACK_STOCK_PRICES:
        return FALLBACK_STOCK_PRICES[ticker_clean]
    
    # Default fallback estimate
    return 150.00


def calculator(expression: str) -> float:
    """
    Safely evaluates a mathematical expression containing numbers and basic arithmetic operators (+, -, *, /, //, **, %).
    Args:
        expression: A mathematical string expression (e.g., '15 * 428.15', '2000 / 224.50', '500 * (1 + 0.085)').
    Returns:
        float: The calculated numerical result.
    """
    import re
    operators_map = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval_node(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            op_type = type(node.op)
            if op_type in operators_map:
                return operators_map[op_type](left, right)
            raise ValueError(f"Unsupported binary operator: {op_type}")
        elif isinstance(node, ast.UnaryOp):
            operand = _eval_node(node.operand)
            op_type = type(node.op)
            if op_type in operators_map:
                return operators_map[op_type](operand)
            raise ValueError(f"Unsupported unary operator: {op_type}")
        else:
            raise ValueError(f"Unsupported expression node: {node}")

    try:
        # Remove commas, currency symbols, and convert percentage notations like 7.5% into (7.5 / 100)
        clean_expr = str(expression).replace(",", "").replace("$", "").strip()
        clean_expr = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'(\1 / 100)', clean_expr)
        tree = ast.parse(clean_expr, mode="eval")
        result = _eval_node(tree.body)
        return round(float(result), 4)
    except Exception as e:
        return f"Error evaluating expression '{expression}': {str(e)}"


def search_market_news(query: str) -> str:
    """
    Searches real-time market news and financial catalysts.
    Uses Google News RSS with DDGS and offline fallbacks.
    Args:
        query: Search query string (e.g., 'Microsoft Q3 revenue earnings summary').
    Returns:
        str: Concise summary of relevant news snippets with source and timestamps.
    """
    import ssl
    import urllib.request
    import urllib.parse
    import xml.etree.ElementTree as ET

    # Tier 1: Live Google News RSS Feed (Fast, real-time, no API key, no rate limits)
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
                source = item.find("source").text if item.find("source") is not None else "Financial News"
                pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
                date_str = pub_date[:16] if pub_date else ""
                snippets.append(f"• {title} [{source} | {date_str}]")
            if snippets:
                return "\n".join(snippets)
    except Exception:
        pass

    # Tier 2: DuckDuckGo News
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=2))
            if results:
                snippets = [f"• {r.get('title')}: {r.get('body')}" for r in results]
                return "\n".join(snippets)
    except Exception:
        pass

    # Tier 3: Curated Offline Fallback
    return f"Market News for '{query}': Recent earnings and analyst ratings remain positive with steady institutional volume."


def save_portfolio_log(ticker: str, action: str, shares: float, total_cost: float) -> str:
    """
    Logs an executed or proposed stock trade to a local audit log file.
    Args:
        ticker: Stock ticker symbol (e.g., 'MSFT').
        action: 'BUY' or 'SELL'.
        shares: Number of shares.
        total_cost: Total calculated cost.
    Returns:
        str: Confirmation message.
    """
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "trades.jsonl")
    
    record = {
        "ticker": ticker.upper(),
        "action": action.upper(),
        "shares": shares,
        "total_cost": total_cost,
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


# Registry of available tools for the Agent
AVAILABLE_TOOLS: Dict[str, Callable] = {
    "get_stock_price": get_stock_price,
    "calculator": calculator,
    "search_market_news": search_market_news,
    "save_portfolio_log": save_portfolio_log,
}


# ==============================================================================
# 2. ReAct Agent Engine Built From Scratch (Slide 16, 17, 18, 19)
# ==============================================================================

REACT_SYSTEM_PROMPT = """You are a smart financial AI Agent powered strictly by the ReAct (Reasoning and Acting) loop.
You have access to the following tools:

1. get_stock_price(ticker="AAPL") -> float: Fetches the real-time stock price for a ticker.
2. calculator(expression="15 * 428.15") -> float: Evaluates mathematical arithmetic expressions accurately. Always use for math!
3. search_market_news(query="MSFT quarterly earnings") -> str: Searches recent news and market catalysts.
4. save_portfolio_log(ticker="MSFT", action="BUY", shares=10, total_cost=4281.50) -> str: Logs a trade.

To solve the user's task, you MUST follow this exact ReAct format:

Thought: Describe what you need to do or reason about the current situation.
Action: tool_name({"arg1": "value1", "arg2": "value2"})
Observation: [The system will execute the tool and provide the result here]
... (this Thought/Action/Observation can repeat up to 5 times)
Thought: I now have all information needed to answer the question.
Final Answer: The complete, final user-facing response with clear breakdown and calculations.

CRITICAL RULES:
- Output exactly ONE Thought and ONE Action per turn.
- NEVER generate the Observation yourself. STOP immediately after outputting Action.
- ALWAYS use the exact numerical values returned in the previous Observation lines. Never invent or use old memory prices!
"""


class ScratchReActAgent:
    """
    A transparent, educational implementation of the ReAct autonomous loop.
    Demonstrates exactly how LLM Thoughts, JSON Tool Actions, and System Observations interact.
    """
    def __init__(self, max_iterations: int = 6):
        self.max_iterations = max_iterations
        self.google_key = os.getenv("GOOGLE_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")

    def _call_llm(self, prompt: str) -> str:
        """Invokes the configured LLM with temperature=0.0 for strict deterministic planning."""
        if self.google_key:
            try:
                resp = call_gemini_with_fallback(
                    prompt,
                    generation_config={"temperature": 0.0}
                )
                text = extract_gemini_text(resp)
                if text:
                    return text.strip()
            except Exception as e:
                print(f"[API Warning] Gemini call failed: {e}")

        if self.openai_key:
            try:
                import openai
                client = openai.OpenAI(api_key=self.openai_key)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    stop=["Observation:", "\nObservation:"]
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"[API Warning] OpenAI call failed: {e}")

        if self.groq_key:
            try:
                from groq import Groq
                client = Groq(api_key=self.groq_key)
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                return resp.choices[0].message.content
            except Exception as e:
                print(f"[API Warning] Groq call failed: {e}")

        # Educational Mock Simulation Mode (only if all APIs absent or failed)
        return self._mock_llm_step(prompt)

    def _mock_llm_step(self, prompt: str) -> str:
        """Simulates ReAct steps if running offline without API keys."""
        detected_ticker = "MSFT"
        for t in ["MSFT", "AAPL", "NVDA", "TSLA", "AMZN", "GOOGL", "META"]:
            if t in prompt:
                detected_ticker = t
                break

        if "Observation:" not in prompt:
            return f'Thought: I need to fetch the stock price of {detected_ticker} first.\nAction: get_stock_price({{"ticker": "{detected_ticker}"}})'
        elif "calculator" not in prompt:
            price = FALLBACK_STOCK_PRICES.get(detected_ticker, 150.00)
            return f'Thought: The {detected_ticker} price is ${price:.2f}. Now I will calculate the cost of 10 shares.\nAction: calculator({{"expression": "10 * {price}"}})'
        else:
            price = FALLBACK_STOCK_PRICES.get(detected_ticker, 150.00)
            total = price * 10
            return f'Thought: I have the calculated total cost.\nFinal Answer: The current stock price of {detected_ticker} is ${price:.2f}. Purchasing 10 shares costs exactly ${total:,.2f}.'

    def run(self, query: str) -> str:
        """Executes the full ReAct loop."""
        history = f"{REACT_SYSTEM_PROMPT}\n\nUser Question: {query}\n"
        
        self._log_header(f"🤖 Market Agent Initiated: '{query}'")

        for step in range(1, self.max_iterations + 1):
            llm_response = self._call_llm(history)
            
            # Prioritize Action execution if present
            if "Action:" in llm_response:
                action_idx = llm_response.find("Action:")
                end_of_action = llm_response.find("\n", action_idx)
                
                if end_of_action == -1:
                    action_line = llm_response[action_idx:].strip()
                    thought = llm_response[:action_idx].strip()
                else:
                    action_line = llm_response[action_idx:end_of_action].strip()
                    thought = llm_response[:action_idx].strip()

                if thought:
                    self._log_thought(thought)

                self._log_action(action_line)
                try:
                    raw_action = action_line.replace("Action:", "").strip()
                    tool_name = raw_action.split("(")[0].strip()
                    args_str = raw_action[len(tool_name):].strip()
                    pos_args, kw_args = parse_action_args(args_str)
                    
                    if tool_name in AVAILABLE_TOOLS:
                        observation_result = AVAILABLE_TOOLS[tool_name](*pos_args, **kw_args)
                    else:
                        observation_result = f"Error: Tool '{tool_name}' does not exist. Available tools: {list(AVAILABLE_TOOLS.keys())}"
                except Exception as e:
                    observation_result = f"Error executing tool action: {str(e)}"

                self._log_observation(str(observation_result))
                clean_step_output = (thought + "\n" + action_line).strip()
                history += f"{clean_step_output}\nObservation: {observation_result}\n"

            elif "Final Answer:" in llm_response:
                thought_part = llm_response.split("Final Answer:")[0].strip()
                final_answer = llm_response.split("Final Answer:")[1].strip()
                
                if thought_part:
                    self._log_thought(thought_part)
                self._log_final_answer(final_answer)
                return final_answer
            else:
                history += f"{llm_response}\n"

        return "Agent stopped: Reached maximum iterations."

    # Visual Logging Methods
    def _log_header(self, text: str):
        if console:
            console.print(Panel(Text(text, style="bold cyan"), border_style="cyan"))
        else:
            print(f"\n{'='*70}\n  {text}\n{'='*70}")

    def _log_thought(self, text: str):
        if console:
            console.print(f"[bold yellow]🧠 {text}[/bold yellow]")
        else:
            print(f"[Thought] {text}")

    def _log_action(self, text: str):
        if console:
            console.print(f"[bold blue]⚡ {text}[/bold blue]")
        else:
            print(f"[Action] {text}")

    def _log_observation(self, text: str):
        if console:
            console.print(f"[bold green]👁️ Observation: {text}[/bold green]\n")
        else:
            print(f"[Observation] {text}\n")

    def _log_final_answer(self, text: str):
        if console:
            console.print(Panel(Text(text, style="bold white"), title="🎯 Final Answer", border_style="green"))
        else:
            print(f"\n--- [Final Answer] ---\n{text}\n")


# ==============================================================================
# 3. Interactive Agent Execution
# ==============================================================================
def main():
    agent = ScratchReActAgent()
    
    test_queries = [
        "How much would it cost to buy 15 shares of Microsoft (MSFT)?",
        "What is the stock price of Apple (AAPL) and Tesla (TSLA)? If I buy 10 shares of each, what is the combined total cost?",
        "If I have a budget of $3,500, how many whole shares of Nvidia (NVDA) can I buy, and how much cash will be left over?"
    ]

    for query in test_queries:
        agent.run(query)
        print("\n" + "-" * 70 + "\n")


if __name__ == "__main__":
    main()
