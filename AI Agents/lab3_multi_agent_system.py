"""
Lab 3 (Bonus Extension): Multi-Agent Collaboration & Handoffs
Open Source Development Program - Session 22 (Slide 23)

This script demonstrates:
1. Why single mega-agents get confused on complex workflows.
2. Specialization: Role-based Agents with isolated prompt boundaries.
3. The Handoff Pattern:
   - Agent 1 (Market Researcher): Queries live data, technical metrics, and news.
   - Agent 2 (Senior Financial Analyst & Writer): Ingests the research brief, computes risk metrics, and writes an executive investment memo.
"""

import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Import tools from Lab 2
from lab2_market_agent import get_stock_price, calculator, search_market_news


# ==============================================================================
# 1. Pydantic Schemas for Structured Agent Handoffs
# ==============================================================================
class ResearchBrief(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")
    company_name: str = Field(description="Full company name")
    current_price: float = Field(description="Current market price in USD")
    recent_catalysts: str = Field(description="Summary of recent news, earnings, or market drivers")
    market_sentiment: str = Field(description="Bullish, Bearish, or Neutral")


class InvestmentMemo(BaseModel):
    title: str = Field(description="Headline summary of the investment analysis")
    target_ticker: str
    proposed_investment_amount: float
    estimated_shares: float
    key_risks: list[str] = Field(description="Top 2-3 potential risks to watch")
    executive_recommendation: str = Field(description="Actionable buy/hold/sell perspective with rationale")


# ==============================================================================
# 2. Agent 1: Market Researcher Specialist
# ==============================================================================
COMPANY_NAMES = {
    "MSFT": "Microsoft Corporation",
    "AAPL": "Apple Inc.",
    "NVDA": "NVIDIA Corporation",
    "TSLA": "Tesla, Inc.",
    "AMZN": "Amazon.com, Inc.",
    "GOOGL": "Alphabet Inc.",
    "META": "Meta Platforms, Inc.",
}

class ResearcherAgent:
    """Specialist agent focused on factual data retrieval, live quotes, and catalyst gathering."""
    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

    def conduct_research(self, ticker: str) -> ResearchBrief:
        t_clean = ticker.strip().upper()
        print(f"\n🔍 [Agent 1: Researcher] Fetching real-time market data for {t_clean}...")
        price = get_stock_price(t_clean)
        print(f"   ↳ Live Price: ${price:.2f}")

        print(f"🔍 [Agent 1: Researcher] Searching latest market catalysts...")
        news = search_market_news(f"{t_clean} stock financial earnings news")
        print(f"   ↳ News Snippets collected.")

        company = COMPANY_NAMES.get(t_clean, f"{t_clean} Inc.")
        brief = ResearchBrief(
            ticker=t_clean,
            company_name=company,
            current_price=price,
            recent_catalysts=news,
            market_sentiment="Bullish / Stable" if price > 100 else "Neutral"
        )
        print(f"✅ [Agent 1: Researcher] Research Brief compiled successfully!")
        return brief


from model_utils import call_gemini_with_fallback, extract_gemini_text

# ==============================================================================
# 3. Agent 2: Senior Financial Analyst & Writer Specialist
# ==============================================================================
class AnalystWriterAgent:
    """Specialist agent focused on synthesis, portfolio math, risk assessment, and executive memos."""
    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.provider = provider
        self.model_name = model_name
        self.google_key = (api_key if provider == "gemini" else None) or os.getenv("GOOGLE_API_KEY")
        self.openai_key = (api_key if provider == "openai" else None) or os.getenv("OPENAI_API_KEY")

    def generate_memo(self, brief: ResearchBrief, budget: float = 10000.0) -> InvestmentMemo:
        print(f"\n📊 [Agent 2: Analyst & Writer] Ingesting Research Brief from Researcher Agent...")
        
        # Exact arithmetic computation using calculator tool
        safe_price = max(brief.current_price, 0.01)
        shares_calc = calculator(f"{budget} / {safe_price}")
        shares_clean = round(float(shares_calc), 2) if isinstance(shares_calc, (int, float)) else round(budget / safe_price, 2)

        print(f"   ↳ Calculated Position Size: {shares_clean} shares with ${budget:,.2f} budget.")
        print(f"📝 [Agent 2: Analyst & Writer] Synthesizing news catalysts & drafting Investment Memo...")

        # Prompt for LLM synthesis
        prompt = (
            f"You are a Senior Financial Analyst and Portfolio Manager.\n"
            f"Analyze this structured Research Brief produced by the Market Researcher Agent:\n"
            f"- Target Security: {brief.ticker} ({brief.company_name})\n"
            f"- Current Market Price: ${brief.current_price:.2f} USD\n"
            f"- Allocated Capital: ${budget:,.2f} USD\n"
            f"- Computed Share Acquisition: {shares_clean} shares\n"
            f"- Market Sentiment: {brief.market_sentiment}\n"
            f"- Live Market Catalysts / Headlines:\n{brief.recent_catalysts}\n\n"
            f"Generate a company-tailored, highly specific InvestmentMemo:\n"
            f"1. title: Executive title for {brief.ticker}.\n"
            f"2. target_ticker: '{brief.ticker}'\n"
            f"3. proposed_investment_amount: {budget}\n"
            f"4. estimated_shares: {shares_clean}\n"
            f"5. key_risks: Top 3 specific, differentiated risk factors tailored specifically to {brief.ticker} and the news catalysts above (DO NOT use generic boilerplate!).\n"
            f"6. executive_recommendation: A detailed strategic thesis explaining the investment rationale, entry strategy, and catalyst interpretation."
        )

        if self.provider != "mock" and self.google_key:
            try:
                resp = call_gemini_with_fallback(
                    prompt,
                    generation_config={
                        "temperature": 0.2,
                        "response_mime_type": "application/json",
                        "response_schema": InvestmentMemo,
                        "max_output_tokens": 2048,
                    },
                    preferred_model=self.model_name or "gemini-3.5-flash-lite",
                    timeout=25
                )
                txt = extract_gemini_text(resp).strip()
                if txt.startswith("```json"):
                    txt = txt[7:]
                if txt.startswith("```"):
                    txt = txt[3:]
                if txt.endswith("```"):
                    txt = txt[:-3]
                data = json.loads(txt.strip())
                return InvestmentMemo(
                    title=data.get("title", f"Strategic Analysis: {brief.ticker} Asset Allocation"),
                    target_ticker=brief.ticker,
                    proposed_investment_amount=float(budget),
                    estimated_shares=float(shares_clean),
                    key_risks=data.get("key_risks") if (isinstance(data.get("key_risks"), list) and data.get("key_risks")) else [
                        f"{brief.company_name} valuation sensitivity to interest rate fluctuations.",
                        f"Competitive and margin pressures in core operational sectors.",
                        "Supply chain and macro regulatory compliance risks."
                    ],
                    executive_recommendation=data.get("executive_recommendation") or (
                        f"Allocate ${budget:,.2f} to acquire approximately {shares_clean} shares of {brief.ticker} "
                        f"at current price of ${brief.current_price:.2f}."
                    )
                )
            except Exception as e:
                print(f"[Analyst Notice] Gemini structured generation: {e}")

        if self.provider != "mock" and self.openai_key:
            try:
                import openai
                client = openai.OpenAI(api_key=self.openai_key)
                completion = client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format=InvestmentMemo,
                    temperature=0.2
                )
                memo_res = completion.choices[0].message.parsed
                memo_res.proposed_investment_amount = float(budget)
                memo_res.estimated_shares = float(shares_clean)
                return memo_res
            except Exception as e:
                print(f"[Analyst Notice] OpenAI generation: {e}")

        # Ticker-specific educational fallback if running offline
        ticker_risks = {
            "TSLA": [
                "Intense EV price competition in North America and China pressuring automotive gross margins.",
                "Regulatory scrutiny and autonomous vehicle timeline uncertainties for Full Self-Driving (FSD).",
                "Execution risk associated with scaling next-generation vehicle platforms and robotics."
            ],
            "NVDA": [
                "High customer concentration among hyper-scaler cloud titans for datacenter AI accelerators.",
                "Geopolitical trade restrictions and semiconductor export licensing headwinds.",
                "Potential cyclical digestion phase following multi-year exponential AI capex growth."
            ],
            "AAPL": [
                "Elongated consumer hardware upgrade cycles and mature global smartphone penetration.",
                "Regulatory antitrust pressure on App Store commission structures in the US and EU.",
                "Supply chain concentration and foreign exchange volatility in international markets."
            ],
            "MSFT": [
                "High datacenter and custom AI silicon infrastructure capital expenditures.",
                "Monetization pacing and corporate enterprise adoption velocity of Copilot tools.",
                "Macro enterprise IT budget scrutiny and cloud migration pacing."
            ]
        }
        
        default_risks = [
            f"Valuation multiple contraction if quarterly earnings miss analyst consensus.",
            f"Industry-specific competitive pressures and product pipeline execution delays.",
            f"Macroeconomic interest rate and foreign exchange currency volatility."
        ]

        memo = InvestmentMemo(
            title=f"Strategic Investment Briefing: {brief.ticker} Asset Allocation",
            target_ticker=brief.ticker,
            proposed_investment_amount=budget,
            estimated_shares=shares_clean,
            key_risks=ticker_risks.get(brief.ticker, default_risks),
            executive_recommendation=(
                f"Allocate ${budget:,.2f} to acquire approximately {shares_clean} shares of {brief.ticker} "
                f"at the entry price of ${brief.current_price:.2f}. "
                f"Strategic Rationale: Catalysts and live news indicate {brief.market_sentiment.lower()} institutional sentiment."
            )
        )
        return memo


# ==============================================================================
# 4. Multi-Agent Orchestration & Pipeline
# ==============================================================================
def run_multi_agent_pipeline(ticker: str = "MSFT", budget: float = 5000.0):
    print("=" * 75)
    print(f"🚀 MULTI-AGENT PIPELINE: Research -> Analysis Handoff")
    print(f"🎯 Objective: Evaluate ${budget:,.2f} allocation in {ticker.upper()}")
    print("=" * 75)

    # Instantiate specialized agents
    researcher = ResearcherAgent()
    analyst = AnalystWriterAgent()

    # Step 1: Research Phase
    brief = researcher.conduct_research(ticker)

    # Step 2: Handoff to Analyst Phase
    memo = analyst.generate_memo(brief, budget=budget)

    # Step 3: Print Executive Presentation
    print("\n" + "=" * 75)
    print(f"📋 FINAL DELIVERABLE: EXECUTIVE INVESTMENT MEMO")
    print("=" * 75)
    print(f"📌 Title:          {memo.title}")
    print(f"🏢 Target Asset:   {memo.target_ticker} (Current Price: ${brief.current_price:.2f})")
    print(f"💰 Capital:        ${memo.proposed_investment_amount:,.2f} ({memo.estimated_shares} shares)")
    print(f"⚠️  Key Risks:")
    for risk in memo.key_risks:
        print(f"   - {risk}")
    print(f"💡 Strategy:       {memo.executive_recommendation}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_multi_agent_pipeline("MSFT", 5000.0)
