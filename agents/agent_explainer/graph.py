"""
LangGraph graph definition for the Explainer Agent (Web Page Q&A).

Architecture:
  User Input (url, question)
       │
       ▼
  clarify ──► Asks for URL if missing
       │
       ▼ (If URL present)
  scrape_url ──► crawl4ai fetches content (skipped if already cached)
       │
       ▼
  answer_question ──► LLM answers the user's question from scraped content
       │
       ▼
     END
"""

import json
import logging
import time
import asyncio
from pathlib import Path
import os

from groq import RateLimitError
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langsmith import traceable

# crawl4ai-based search & scraping
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from crawl4ai_utils import crawl4ai_scrape

from state import (
    ExplainerState,
    ClarifyDecision,
    ExplanationResult,
)
from configuration import Configuration
from prompts import (
    CLARIFY_DECISION_PROMPT,
    ANSWER_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────

def _get_llm(model_name: str, temperature: float = 0.0) -> ChatGroq:
    """Create an LLM instance."""
    return ChatGroq(
        model=model_name,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=temperature,
        max_retries=0,
    )


def _invoke_with_backoff(
    chain,
    messages: list,
    max_attempts: int = 6,
    pydantic_schema=None,
):
    import random
    import threading

    _DAILY_KEYWORDS = (
        'tokens per day', 'tpd',
        'requests per day', 'rpd',
        'daily limit', 'daily token',
    )
    _FALLBACK_MODEL = "llama-3.1-8b-instant"

    if not hasattr(_invoke_with_backoff, "_semaphore"):
        _invoke_with_backoff._semaphore = threading.Semaphore(2)

    delay = 4.0
    for attempt in range(max_attempts):
        try:
            with _invoke_with_backoff._semaphore:
                return chain.invoke(messages)

        except RateLimitError as e:
            error_msg = str(e).lower()

            if any(kw in error_msg for kw in _DAILY_KEYWORDS):
                limit_type = "RPD" if ("requests per day" in error_msg or "rpd" in error_msg) else "TPD"
                logger.warning(
                    f"⛔ {limit_type} limit hit! Attempting fallback to '{_FALLBACK_MODEL}'."
                )
                fallback_llm = _get_llm(_FALLBACK_MODEL)
                fallback_chain = (
                    fallback_llm.with_structured_output(pydantic_schema)
                    if pydantic_schema else fallback_llm
                )
                try:
                    return fallback_chain.invoke(messages)
                except Exception as fallback_e:
                    logger.error(f"Fallback model also failed: {fallback_e}")
                    raise e

            if attempt == max_attempts - 1:
                logger.error(f"Rate limit persisted after {max_attempts} attempts. Giving up.")
                raise

            jitter = random.uniform(0, delay * 0.5)
            wait = delay + jitter
            logger.warning(
                f"⏳ Rate limit hit (attempt {attempt + 1}/{max_attempts}). "
                f"Waiting {wait:.1f}s before retry..."
            )
            time.sleep(wait)
            delay = min(delay * 2, 60.0)


# ─────────────────────────────────────────────────────
# Node: Clarify
# ─────────────────────────────────────────────────────

def clarify(state: ExplainerState) -> dict:
    config = Configuration.from_env()

    if state.get("clarify_question"):
        logger.info("💬 Clarification already done, skipping.")
        return {}

    # ── Fast path: if URL is already known, skip the LLM entirely ─────
    if state.get("url"):
        logger.info(f"✅ URL already present: {state['url']} — skipping clarify LLM.")
        updates = {}
        # Use the user's message as the question if no explicit question was set
        if not state.get("question") and state.get("user_message_raw"):
            updates["question"] = state["user_message_raw"]
        return updates

    prompt = CLARIFY_DECISION_PROMPT.format(
        url=state.get("url", ""),
        question=state.get("question", ""),
        user_message=state.get("user_message_raw", ""),
    )

    try:
        llm = _get_llm(config.clarify_model, temperature=0.0)
        structured_llm = llm.with_structured_output(ClarifyDecision)
        decision: ClarifyDecision = _invoke_with_backoff(
            structured_llm,
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Extraé la información del mensaje y decidí si necesito pedir algo más."},
            ],
            pydantic_schema=ClarifyDecision,
        )

        updates = {}
        if decision.extracted_url and not state.get("url"):
            updates["url"] = decision.extracted_url
        if decision.extracted_question and not state.get("question"):
            updates["question"] = decision.extracted_question

        # ── Hard safety net: URL is the ONLY mandatory field ──────────
        effective_url = updates.get("url") or state.get("url", "")

        if effective_url:
            # URL is present → ALWAYS proceed, ignore the LLM's need_clarification
            logger.info(f"✅ URL present: {effective_url} (extracted_question={decision.extracted_question})")
            return updates

        # URL is missing → ask for it
        logger.info(f"🚨 URL is missing — asking user for it.")
        updates["clarify_question"] = (
            decision.question
            if decision.question
            else "Para poder ayudarte necesito una URL. ¿Podés indicarme el link de la página web que querés analizar?"
        )
        updates.setdefault("missing_fields", ["url"])
        return updates

    except Exception as e:
        logger.warning(f"Clarify decision failed: {e}. Proceeding without clarification.")
        if not state.get("url"):
            return {
                "clarify_question": "Para poder ayudarte necesito una URL. ¿Podés indicarme el link de la página web que querés analizar?",
                "missing_fields": ["url"],
            }
        return {}


# ─────────────────────────────────────────────────────
# Node: Scrape URL
# ─────────────────────────────────────────────────────

async def scrape_url(state: ExplainerState) -> dict:
    """Uses crawl4ai to scrape the target URL content.
    Skips scraping if scraped_content is already present (cached by executor)."""
    # If content was already provided (cached), skip scraping
    if state.get("scraped_content"):
        logger.info("📦 Using cached scraped content — skipping scrape.")
        return {}

    url = state.get("url", "")
    if not url:
        return {"scraped_content": "No URL provided."}
    
    logger.info(f"🌐 Scraping {url} with crawl4ai...")
    try:
        scraped_content = await crawl4ai_scrape(url)
        # Prevent oversized inputs
        if len(scraped_content) > 40000:
            scraped_content = scraped_content[:40000] + "\n...[TRUNCATED]..."
        return {"scraped_content": scraped_content}
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return {"scraped_content": f"Error al recuperar el contenido de la URL: {e}"}


# ─────────────────────────────────────────────────────
# Node: Answer Question
# ─────────────────────────────────────────────────────

def answer_question(state: ExplainerState) -> dict:
    """Uses LLM to answer the user's question based on the scraped content."""
    config = Configuration.from_env()
    llm = _get_llm(config.planner_model, temperature=0.3)

    scraped_content = state.get("scraped_content", "")
    url = state.get("url", "")

    # Use the question if provided, otherwise use the raw user message,
    # otherwise default to a general summary request.
    question = (
        state.get("question")
        or state.get("user_message_raw")
        or "Hacé un resumen general de la página."
    )

    prompt = ANSWER_SYSTEM_PROMPT.format(
        url=url,
        question=question,
        scraped_content=scraped_content,
    )

    try:
        response = _invoke_with_backoff(
            llm,
            [
                {"role": "system", "content": prompt},
            ]
        )
        
        text_output = response.content.strip() if hasattr(response, "content") else str(response)
        
        found = True
        if text_output.startswith("NOT_FOUND"):
            found = False
            text_output = text_output.replace("NOT_FOUND", "").strip()
            
        logger.info(f"🧠 Answer generated. Found info? {found}")
        
        return {"final_explanation": {"found": found, "explanation": text_output}}
    except Exception as e:
        logger.error(f"Answer LLM failed: {e}")
        return {"final_explanation": {"found": False, "explanation": "Lo siento, ocurrió un error analizando el contenido."}}


# ─────────────────────────────────────────────────────
# Build the Graph
# ─────────────────────────────────────────────────────

def should_clarify(state: ExplainerState) -> str:
    """Route through clarify on first pass; skip to scrape_url if already clarified."""
    if state.get("clarify_question"):
        return "scrape_url"
    return "clarify"

def _after_clarify(state: ExplainerState) -> str:
    """If a question was generated, go to END to wait for user answer;
    otherwise proceed to scrape."""
    if state.get("clarify_question"):
        return END
    return "scrape_url"

def build_graph():
    """Construct and compile the LangGraph StateGraph."""
    graph = StateGraph(ExplainerState)

    graph.add_node("clarify", clarify)
    graph.add_node("scrape_url", scrape_url)
    graph.add_node("answer_question", answer_question)

    graph.add_conditional_edges(START, should_clarify, ["clarify", "scrape_url"])
    graph.add_conditional_edges("clarify", _after_clarify, [END, "scrape_url"])
    
    graph.add_edge("scrape_url", "answer_question")
    graph.add_edge("answer_question", END)

    return graph.compile()
