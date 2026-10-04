"""
Model Selection & Provider Utilities
Automatically discovers the best active model for Google Gemini, OpenAI, and Groq,
with automatic fallback on 429 quota exhaustion.
"""

import os
from typing import Optional, List, Any

# Supported Gemini models in priority order of reliability, speed & available free-tier quota
AVAILABLE_GEMINI_MODELS: List[str] = [
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
]

_CACHED_GEMINI_MODEL: Optional[str] = None

def get_best_gemini_model() -> str:
    """Discovers and returns the best active Gemini model with available quota."""
    global _CACHED_GEMINI_MODEL
    if _CACHED_GEMINI_MODEL:
        return _CACHED_GEMINI_MODEL

    env_override = os.getenv("AI_MODEL_NAME")
    if env_override and "gemini" in env_override.lower():
        _CACHED_GEMINI_MODEL = env_override
        return _CACHED_GEMINI_MODEL

    # Default to ultrafast, stable Gemini 3.5 Flash Lite
    _CACHED_GEMINI_MODEL = "gemini-3.5-flash-lite"
    return _CACHED_GEMINI_MODEL


def call_gemini_with_fallback(
    prompt: str,
    generation_config: Optional[dict] = None,
    system_instruction: Optional[str] = None,
    preferred_model: Optional[str] = None,
    timeout: int = 30
) -> Any:
    """
    Executes a Gemini generate_content call. If a 429 ResourceExhausted / quota error,
    timeout, an empty response, or a malformed JSON completion occurs on the model,
    it automatically fails over to the next available model in the priority queue.
    """
    import json
    import google.generativeai as genai

    # Ensure API key is configured if present in environment
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            genai.configure(api_key=api_key)
        except Exception:
            pass

    start_model = preferred_model or get_best_gemini_model()
    model_queue = [start_model] + [m for m in AVAILABLE_GEMINI_MODELS if m != start_model]

    # Enforce safe default max_output_tokens to prevent runaway whitespace token loops
    cfg = dict(generation_config) if generation_config else {}
    cfg.setdefault("max_output_tokens", 2048)

    last_error = None
    for model_name in model_queue:
        try:
            kwargs = {"generation_config": cfg}
            if system_instruction:
                kwargs["system_instruction"] = system_instruction

            model = genai.GenerativeModel(model_name, **kwargs)
            response = model.generate_content(prompt, request_options={"timeout": timeout})
            
            # Verify that response actually contains text content
            txt = extract_gemini_text(response)
            if not txt or not txt.strip():
                # Model returned empty candidate (e.g. FinishReason 19), failover to next model
                continue
                
            # If JSON structured output was requested, verify that txt is valid JSON
            if cfg and (
                cfg.get("response_mime_type") == "application/json"
                or "response_schema" in cfg
            ):
                try:
                    clean_json = txt.strip()
                    if clean_json.startswith("```json"):
                        clean_json = clean_json[7:]
                    if clean_json.startswith("```"):
                        clean_json = clean_json[3:]
                    if clean_json.endswith("```"):
                        clean_json = clean_json[:-3]
                    json.loads(clean_json.strip())
                except Exception:
                    # Model returned truncated or malformed JSON, try next model in queue
                    continue

            return response
        except Exception as e:
            err_str = str(e).lower()
            last_error = e
            # If 429 / quota limit exceeded, 404 not found, timeout, or server error, try next model
            if any(k in err_str for k in ("429", "resourceexhausted", "404", "quota", "not found", "timeout", "deadline", "timed out", "503", "500")):
                continue
            else:
                raise e

    if last_error:
        raise last_error


def extract_gemini_text(response: Any) -> str:
    """Safely extracts text content from a Gemini response object regardless of finish_reason or stop_sequence."""
    if not response:
        return ""
    try:
        if hasattr(response, "text"):
            return response.text
    except Exception:
        pass

    try:
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                return "".join(p.text for p in candidate.content.parts if hasattr(p, "text"))
    except Exception:
        pass

    return str(response)
