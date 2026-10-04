"""
Lab 1: The Expense Parser (Structured Output & Programmable AI)
Open Source Development Program - Session 22

This script demonstrates:
1. The difference between Chat AI vs Developer APIs (Roles, Determinism, Temperature=0.0).
2. The Parsing Problem (Chatty LLM markdown failing json.loads).
3. Zero-shot vs Few-shot extraction.
4. Strict Structured Output using Pydantic Models & JSON Schema.
"""

import os
import json
from typing import List, Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from model_utils import get_best_gemini_model, call_gemini_with_fallback

# Rich formatting for terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    console = Console()
except ImportError:
    console = None


# Load environment variables
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


# ==============================================================================
# 1. Pydantic Blueprint for Strict Structured Output (Slide 9 & 11)
# ==============================================================================
class ExpenseItem(BaseModel):
    vendor: str = Field(description="Name of the merchant, store, or service vendor")
    amount: float = Field(description="Total monetary amount spent")
    category: str = Field(description="Classified business expense category (e.g. Food & Beverage, Travel, Office, Software)")
    description: str = Field(description="Brief description or context of the purchase")


class ExpenseReport(BaseModel):
    expenses: List[ExpenseItem] = Field(description="List of all extracted individual expense items")
    total_amount: float = Field(description="Sum total of all extracted expense amounts")


# ==============================================================================
# 2. LLM Client Factory (Gemini, OpenAI, Groq, or Mock Fallback)
# ==============================================================================
def get_llm_client():
    """Detects available API key and returns configured provider function."""
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    if google_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=google_key)
            return "gemini", genai
        except ImportError:
            pass

    if openai_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            return "openai", client
        except ImportError:
            pass

    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            return "groq", client
        except ImportError:
            pass

    return "mock", None


# ==============================================================================
# 3. Method 1: The Chatty AI & The Parsing Problem (Slide 7)
# ==============================================================================
def parse_unstructured_chatty(text: str) -> str:
    """
    Demonstrates what happens when a standard chat prompt is used.
    Often returns commentary like 'Sure! Here is your JSON: ...' which crashes json.loads().
    """
    provider, client = get_llm_client()
    
    system_prompt = "You are an assistant. Extract expenses from the user's text."
    user_prompt = f"Extract all expenses from this text:\n{text}"

    if provider == "gemini":
        try:
            response = call_gemini_with_fallback(user_prompt, system_instruction=system_prompt)
            return response.text
        except Exception:
            return f"Sure! Here is the data you requested:\n\n{{\"vendor\": \"Starbucks\", \"amount\": 4.50}}\n\nLet me know if you need anything else!"
    elif provider == "openai":
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    elif provider == "groq":
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    else:
        # Mock simulation of chatty response
        return f"Sure! Here is the data you requested:\n\n{{\"vendor\": \"Starbucks\", \"amount\": 4.50}}\n\nLet me know if you need anything else!"


# ==============================================================================
# 4. Method 2: Few-Shot Prompting with Strict System Instructions (Slide 6 & 13)
# ==============================================================================
def parse_with_few_shot_prompt(text: str) -> str:
    """
    Uses Few-Shot examples and temperature=0.0 to improve format adherence.
    """
    provider, client = get_llm_client()

    system_prompt = (
        "You are a strict data extraction engine. "
        "Extract all financial transactions from the user text. "
        "Output ONLY valid JSON matching this schema: [{\"vendor\": string, \"amount\": number, \"category\": string}]. "
        "Do NOT include markdown formatting, backticks, or any conversational text."
    )

    few_shot_examples = [
        {"role": "user", "content": "I grabbed lunch at Subway for 9.50 and refueled at Chevron for $45.00."},
        {"role": "assistant", "content": '[{"vendor": "Subway", "amount": 9.50, "category": "Food & Beverage"}, {"vendor": "Chevron", "amount": 45.00, "category": "Travel & Transportation"}]'},
        {"role": "user", "content": "Bought a coffee for 4.50 at Starbucks yesterday, and spent $50 on gas at Shell."},
        {"role": "assistant", "content": '[{"vendor": "Starbucks", "amount": 4.50, "category": "Food & Beverage"}, {"vendor": "Shell", "amount": 50.00, "category": "Travel & Transportation"}]'},
    ]

    messages = [{"role": "system", "content": system_prompt}] + few_shot_examples + [{"role": "user", "content": text}]

    if provider == "gemini":
        try:
            response = call_gemini_with_fallback(
                f"Extract expenses:\n{text}",
                generation_config={"temperature": 0.0},
                system_instruction=system_prompt
            )
            return response.text
        except Exception:
            return '[{"vendor": "Starbucks", "amount": 4.50, "category": "Food & Beverage"}, {"vendor": "Shell", "amount": 50.00, "category": "Travel & Transportation"}]'
    elif provider in ("openai", "groq"):
        model_name = "gpt-4o-mini" if provider == "openai" else "llama-3.3-70b-versatile"
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0
        )
        return response.choices[0].message.content
    else:
        return '[{"vendor": "Starbucks", "amount": 4.50, "category": "Food & Beverage"}, {"vendor": "Shell", "amount": 50.00, "category": "Travel & Transportation"}]'


# ==============================================================================
# 5. Method 3: Production Grade Structured Output (Pydantic / JSON Schema) (Slide 8-10)
# ==============================================================================
def parse_with_structured_output(text: str) -> ExpenseReport:
    """
    Uses schema-enforced structured generation (Pydantic / response_schema).
    Guarantees 100% type-safe JSON output that parses directly into Python objects.
    """
    provider, client = get_llm_client()

    if provider == "gemini":
        prompt = (
            f"Extract all financial transactions from this messy text into the strict ExpenseReport schema.\n"
            f"Calculate the total amount accurately and classify each category appropriately.\n\n"
            f"Text to parse:\n{text}"
        )
        try:
            response = call_gemini_with_fallback(
                prompt,
                generation_config={
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                    "response_schema": ExpenseReport,
                }
            )
            return ExpenseReport.model_validate_json(response.text)
        except Exception as e:
            print("Gemini fallback on error:", e)

    elif provider == "openai":
        # OpenAI native structured outputs via beta.chat.completions.parse
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a corporate expense auditor. Extract all verified transactions and calculate total sum."
                },
                {"role": "user", "content": text}
            ],
            response_format=ExpenseReport,
            temperature=0.0,
        )
        return completion.choices[0].message.parsed

    elif provider == "groq":
        # Groq json_object format
        schema_json = json.dumps(ExpenseReport.model_json_schema())
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a strict data extraction system. Output JSON conforming to this schema:\n{schema_json}"
                },
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return ExpenseReport.model_validate_json(response.choices[0].message.content)

    else:
        # Mock mode for testing without API keys
        return ExpenseReport(
            expenses=[
                ExpenseItem(vendor="Blue Bottle Coffee", amount=14.50, category="Food & Beverage", description="Breakfast & latte"),
                ExpenseItem(vendor="Uber", amount=48.20, category="Travel & Transportation", description="SFO to Downtown"),
                ExpenseItem(vendor="Chipotle", amount=36.80, category="Food & Beverage", description="Lunch with client"),
                ExpenseItem(vendor="Lyft", amount=52.10, category="Travel & Transportation", description="Downtown to SFO airport"),
                ExpenseItem(vendor="Peet's Coffee", amount=11.25, category="Food & Beverage", description="Airport sandwich")
            ],
            total_amount=162.85
        )


# ==============================================================================
# 6. Interactive Demo & Display
# ==============================================================================
def display_results(report: ExpenseReport, raw_text: str):
    """Prints a styled summary table of the parsed expenses."""
    if console:
        console.print(Panel(raw_text.strip(), title="📥 Raw Input String", border_style="cyan"))
        
        table = Table(title="📊 Extracted Expense Items (Validated Pydantic Model)", border_style="green")
        table.add_column("Vendor", style="bold yellow")
        table.add_column("Amount", justify="right", style="green")
        table.add_column("Category", style="magenta")
        table.add_column("Description", style="white")

        for item in report.expenses:
            table.add_row(item.vendor, f"${item.amount:.2f}", item.category, item.description or "")

        console.print(table)
        console.print(f"[bold green]💰 Total Calculated Amount:[/bold green] [bold white]${report.total_amount:.2f}[/bold white]")
        if getattr(report, "extracted_notes", None):
            console.print(f"[italic cyan]📝 Notes:[/italic cyan] {report.extracted_notes}")
        console.print("\n" + "=" * 70 + "\n")
    else:
        print("\n--- Extracted Expenses ---")
        for item in report.expenses:
            print(f"- {item.vendor}: ${item.amount:.2f} [{item.category}] ({item.description})")
        print(f"Total: ${report.total_amount:.2f}")


def run_lab_demonstration():
    """Runs all 3 stages of Lab 1 to demonstrate progression."""
    provider, _ = get_llm_client()
    print(f"\n=======================================================")
    print(f"  🧪 Lab 1: The Expense Parser (Provider: {provider.upper()})")
    print(f"=======================================================\n")

    sample_input = (
        "Bought a coffee for 4.50 at Starbucks yesterday, and spent $50 on gas at Shell. "
        "Also grabbed a quick burger at In-N-Out for 12.75 and paid $18.30 parking ticket at CityGarage."
    )

    print("--- [STAGE 1] Testing Unstructured / Chatty AI Response ---")
    chatty_output = parse_unstructured_chatty(sample_input)
    print("Raw LLM Output:\n", chatty_output)
    
    # Try parsing chatty output
    try:
        json.loads(chatty_output)
        print("✅ json.loads() succeeded!")
    except Exception as e:
        print(f"❌ json.loads() FAILED as expected with error: {e}")
        print("💡 Why? Chatty models wrap output in conversational greetings, crashing software parsers!")

    print("\n--- [STAGE 2] Testing Few-Shot Prompting (Temp=0.0) ---")
    few_shot_output = parse_with_few_shot_prompt(sample_input)
    print("Few-shot Output:\n", few_shot_output)

    print("\n--- [STAGE 3] Testing Production Structured Output (Pydantic Schema) ---")
    report = parse_with_structured_output(sample_input)
    display_results(report, sample_input)


if __name__ == "__main__":
    run_lab_demonstration()
