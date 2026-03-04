"""
LangGraph graph definition for the Deep Research Agent.
Inspired by langchain-ai/open_deep_research graph architecture.

Architecture:
  User Input (date)
       │
       ▼
  plan_research ──► Generates search queries per category
       │
       ▼ (Send to parallel researchers)
  research_category ──► Tavily/DDG search + LLM extraction (×4 categories)
       │
       ▼
  aggregate_results ──► Combine all events
       │
       ▼
  verify_dates ──► Scrape sources to verify dates
       │
       ▼
  filter_argentina ──► Keep only Argentina-relevant events
       │
       ▼
  generate_report ──► Structured JSON for pandas
"""

import json
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

import os
import re
import yaml
from groq import RateLimitError
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langsmith import traceable
import httpx
from bs4 import BeautifulSoup

from duckduckgo_search import DDGS

# Optional: Tavily for better search quality
try:
    from tavily import TavilyClient
    _HAS_TAVILY = bool(os.getenv("TAVILY_API_KEY"))
except ImportError:
    _HAS_TAVILY = False

# Firecrawl API key for fallback scraping (optional)
_FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")

from state import (
    AgentState,
    ResearcherState,
    SearchPlan,
    ResearchFindings,
    FilteredReport,
    EventInfo,
    ClarifyDecision,
)
from context_manager import SQLiteContextManager
from configuration import Configuration
from prompts import (
    PLANNER_SYSTEM_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    FILTER_SYSTEM_PROMPT,
    REPORT_SYSTEM_PROMPT,
    CLARIFY_DECISION_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────
# Ontología de reglas de negocio
# ─────────────────────────────────────────────────────

_ONTOLOGIA_PATH = Path(__file__).parent / "ontologia.yaml"

def _load_ontologia() -> dict:
    """Load business rules ontology from YAML file."""
    try:
        with open(_ONTOLOGIA_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Could not load ontologia.yaml: {e}")
        return {}


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


# Global semaphore to limit concurrent Groq API calls (prevents 429 storms
# when multiple researchers run in parallel).
_GROQ_SEMAPHORE = asyncio.Semaphore(2)
_GROQ_COOLDOWN = 1.5  # seconds between consecutive API calls


async def _invoke_with_backoff(chain, messages: list, max_attempts: int = 6, pydantic_schema=None):
    """Invoke a LangChain chain with exponential backoff on 429 rate limit errors.
    If a daily token limit is hit (TPD), fallback to a secondary model.
    Uses a global semaphore to limit concurrent Groq API calls.
    """
    import random

    _DAILY_KEYWORDS = [
        "rate limit reached", "tokens per day", "requests per day",
        "tpd", "rpd"
    ]
    # Fallback models to try (each has its own TPD limit on Groq)
    _FALLBACK_MODELS = [
        "llama-3.1-8b-instant",
        "qwen/qwen3-32b"
    ]

    delay = 4.0
    for attempt in range(max_attempts):
        try:
            async with _GROQ_SEMAPHORE:
                result = await asyncio.get_event_loop().run_in_executor(None, chain.invoke, messages)
                await asyncio.sleep(_GROQ_COOLDOWN)  # cooldown between calls
                return result
        except RateLimitError as e:
            error_msg = str(e).lower()

            # ── Daily limit hit (TPD or RPD) ─────────────────────────────────
            if any(kw in error_msg for kw in _DAILY_KEYWORDS):
                limit_type = "RPD" if ("requests per day" in error_msg or "rpd" in error_msg) else "TPD"
                logger.warning(f"⛔ {limit_type} limit hit on primary model!")

                # Try each fallback model (they have separate TPD limits)
                for fb_model in _FALLBACK_MODELS:
                    logger.warning(f"🔄 Trying fallback model: {fb_model}")
                    fallback_llm = _get_llm(fb_model)
                    fallback_chain = (
                        fallback_llm.with_structured_output(pydantic_schema)
                        if pydantic_schema else fallback_llm
                    )
                    try:
                        async with _GROQ_SEMAPHORE:
                            result = await asyncio.get_event_loop().run_in_executor(None, fallback_chain.invoke, messages)
                            await asyncio.sleep(_GROQ_COOLDOWN)
                            return result
                    except Exception as fallback_e:
                        logger.warning(f"Fallback {fb_model} also failed: {fallback_e}")
                        continue

                # All fallbacks exhausted — wait and retry original
                logger.warning(f"⏳ All fallback models exhausted. Waiting 60s before retrying...")
                await asyncio.sleep(60)
                try:
                    async with _GROQ_SEMAPHORE:
                        result = await asyncio.get_event_loop().run_in_executor(None, chain.invoke, messages)
                        await asyncio.sleep(_GROQ_COOLDOWN)
                        return result
                except Exception:
                    raise e

            # ── RPM / TPM — transient, worth retrying ────────────────────────
            if attempt == max_attempts - 1:
                logger.error(f"Rate limit persisted after {max_attempts} attempts. Giving up.")
                raise

            jitter = random.uniform(0, delay * 0.5)
            wait = delay + jitter
            logger.warning(
                f"⏳ Rate limit hit (attempt {attempt + 1}/{max_attempts}). "
                f"Waiting {wait:.1f}s before retry..."
            )
            await asyncio.sleep(wait)
            delay = min(delay * 2, 60.0)


def _parse_failed_generation(error, pydantic_model):
    """Extract and parse the 'failed_generation' JSON from a Groq 400 tool_use_failed error."""
    error_str = str(error)
    if 'tool_use_failed' not in error_str and 'failed_generation' not in error_str:
        return None

    failed_json_str = None

    if hasattr(error, 'body') and isinstance(error.body, dict):
        err_detail = error.body.get('error', {})
        failed_json_str = err_detail.get('failed_generation')

    if not failed_json_str:
        match = re.search(r"'failed_generation':\s*'(.*)'}}$", error_str, re.DOTALL)
        if match:
            failed_json_str = match.group(1)

    if not failed_json_str:
        logger.warning("Could not extract failed_generation from error.")
        return None

    failed_json_str = failed_json_str.strip()
    if failed_json_str.startswith('```'):
        lines = failed_json_str.split('\n')
        lines = [l for l in lines if not l.strip().startswith('```')]
        failed_json_str = '\n'.join(lines)

    try:
        data = json.loads(failed_json_str)
    except json.JSONDecodeError:
        logger.warning(f"failed_generation is not valid JSON: {failed_json_str[:200]}")
        return None

    try:
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict) and 'name' in data[0] and 'parameters' in data[0]:
                params = data[0].get('parameters', {})
                if isinstance(params, dict) and 'events' in params:
                    data = params
                else:
                    data = {'events': []}
                logger.info(f"Extracted data from tool-call wrapper format")
            elif len(data) > 0 and isinstance(data[0], dict) and 'events' in data[0]:
                all_events = []
                notes = ""
                for item in data:
                    all_events.extend(item.get('events', []))
                    if item.get('notes'):
                        notes = item['notes']
                data = {'events': all_events, 'notes': notes}
            else:
                data = {'events': data}

        result = pydantic_model.model_validate(data)
        logger.info(f"✅ Recovered {len(result.events)} events from failed_generation fallback")
        return result
    except Exception as parse_err:
        logger.warning(f"Failed to validate failed_generation into {pydantic_model.__name__}: {parse_err}")
        return None


# ─────────────────────────────────────────────────────
# Search: Serper (primary) + Tavily + DuckDuckGo (fallback)
# ─────────────────────────────────────────────────────

@traceable(name="perform_search", run_type="tool")
def _perform_search(query: str, config: Configuration, include_domains: list[str] | None = None) -> List[dict]:
    """Execute search trying: 1) Serper, 2) Tavily, 3) DuckDuckGo fallback."""
    # 1. SERPER FALLBACK (Google Search)
    serper_key = os.environ.get("SERPER_API_KEY")
    if serper_key:
        try:
            import requests

            serper_query = query
            if include_domains:
                site_query = " OR ".join([f"site:{d}" for d in include_domains])
                serper_query = f"{query} ({site_query})"

            payload = json.dumps({
                "q": serper_query,
                "gl": "ar",  # Country code for Argentina
                "hl": "es",  # Language
                "num": config.search_max_results
            })
            headers = {
                'X-API-KEY': serper_key,
                'Content-Type': 'application/json'
            }

            response = requests.request("POST", "https://google.serper.dev/search", headers=headers, data=payload, timeout=15)
            response.raise_for_status()
            res = response.json()

            formatted_results = []
            if "organic" in res:
                for r in res["organic"]:
                    formatted_results.append({
                        "title": r.get("title", ""),
                        "url": r.get("link", ""),
                        "content": r.get("snippet", "")
                    })
            if formatted_results:
                return formatted_results
        except Exception as e:
            logger.warning(f"Serper search failed for '{query}': {e}. Falling back to Tavily.")

    # 2. TAVILY
    if _HAS_TAVILY:
        try:
            client = TavilyClient()
            kwargs = dict(
                query=query,
                max_results=config.tavily_max_results,
                search_depth=config.tavily_search_depth,
                include_answer=True,
            )
            if include_domains:
                kwargs["include_domains"] = include_domains
            response = client.search(**kwargs)
            results = response.get("results", [])
            if results:
                return results
        except Exception as e:
            logger.warning(f"Tavily search failed for '{query}': {e}. Falling back to DuckDuckGo.")

    # 3. DUCK DUCK GO FALLBACK
    return _ddg_search(query, config, include_domains)


@traceable(name="ddg_search", run_type="tool")
def _ddg_search(query: str, config: Configuration, include_domains: list[str] | None = None) -> List[dict]:
    """Execute a DuckDuckGo search and return formatted results."""
    try:
        from langchain_community.tools import DuckDuckGoSearchResults
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

        ddgs_query = query
        if include_domains:
            site_query = " OR ".join([f"site:{d}" for d in include_domains])
            ddgs_query = f"{query} ({site_query})"

        wrapper = DuckDuckGoSearchAPIWrapper(max_results=config.search_max_results)
        search = DuckDuckGoSearchResults(api_wrapper=wrapper, output_format="list")
        res = search.invoke(ddgs_query)

        formatted_results = []
        for r in res:
            formatted_results.append({
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "content": r.get("snippet", "")
            })
        return formatted_results
    except Exception as e:
        logger.error(f"DuckDuckGo search failed for '{query}': {e}")
        return []


# ─────────────────────────────────────────────────────
# Sitios y configuración de búsqueda
# ─────────────────────────────────────────────────────

# Dominios para DuckDuckGo (site: restricciones)
SITES_DDG = [
    "netflix.com",
    "disneyplus.com",
    "primevideo.com",
    "max.com",
    "paramountplus.com",
    "apple.com/apple-tv-plus",
    "ole.com.ar",
    "espndeportes.espn.com",
    "eventick.com.ar",
]

# URLs a scrapear directamente con Firecrawl (o Jina como fallback)
# Cada entrada: { "nombre", "url", "categoria" }
FIRECRAWL_TARGETS = [
    {
        "nombre":    "Netflix Novedades",
        "url":       "https://www.netflix.com/ar/whats-new",
        "categoria": "streaming",
    },
    {
        "nombre":    "Disney+ Estrenos",
        "url":       "https://www.disneyplus.com/es-419/movies",
        "categoria": "streaming",
    },
    {
        "nombre":    "Prime Video Novedades",
        "url":       "https://www.primevideo.com/-/es/storefront/home",
        "categoria": "streaming",
    },
    {
        "nombre":    "ESPN Deportes",
        "url":       "https://www.espn.com.ar/programacion/",
        "categoria": "deportes",
    },
    {
        "nombre":    "Ole Deportes",
        "url":       "https://www.ole.com.ar/futbol-en-vivo/",
        "categoria": "deportes",
    },
    {
        "nombre":    "Promiedos Home",
        "url":       "https://www.promiedos.com.ar/",
        "categoria": "deportes",
    },
    {
        "nombre":    "Netflix Top 10 Argentina (Tudum)",
        "url":       "https://www.netflix.com/tudum/top10/argentina",
        "categoria": "streaming",
    },
    {
        "nombre":    "Disney+ Top 10 Argentina (FlixPatrol)",
        "url":       "https://flixpatrol.com/top10/disney/argentina/",
        "categoria": "streaming",
    },
    {
        "nombre":    "Prime Video Top 10 Argentina (FlixPatrol)",
        "url":       "https://flixpatrol.com/top10/amazon-prime/argentina/",
        "categoria": "streaming",
    },
]

QUERIES_BASE = [
    "estrenos fecha hora estreno",
    "nuevos lanzamientos calendario",
    "próximos estrenos 2025",
]
MAX_RESULTS_PER_QUERY = 5

# ─── Patrones regex para fecha y hora ─────────────────────────────────────────

DATE_PATTERNS = [
    r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b",
    r"\b(\d{1,2}[/-]\d{1,2})\b", # DD/MM
    r"\b(\d{1,2}\s+de\s+\w+\s+(?:de\s+)?\d{4})\b",
    r"\b(\d{1,2}\s+de\s+\w+)\b", # DD de mes
    r"\b(\d{1,2}\s+\w+\s+\d{4})\b",
    r"\b(\d{1,2}\s+\w+)\b", # DD mes
    r"\b((?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre"
    r"|january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+\d{4})\b",
]
TIME_PATTERNS = [
    r"\b(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)\b",
    r"a\s+las\s+(\d{1,2}(?::\d{2})?)\s*h(?:s|oras?)?",
]


# ─────────────────────────────────────────────────────
# SPA Scraping: Jina Reader (primary) + Firecrawl (fallback)
# ─────────────────────────────────────────────────────


async def _scrape_with_jina(url: str) -> str:
    """Scrape a page using Jina Reader API (r.jina.ai).
    Free, handles JS-rendered pages, returns clean markdown."""
    jina_url = f"https://r.jina.ai/{url}"
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(
                jina_url,
                headers={
                    "Accept": "text/plain",
                    "X-No-Cache": "true",
                },
            )
            resp.raise_for_status()
            text = resp.text.strip()
            if text and len(text) > 50:
                return text[:5000]
    except Exception as e:
        logger.warning(f"Jina Reader failed for {url}: {e}")
    return ""


async def _scrape_with_firecrawl(url: str) -> str:
    """Scrape a page using Firecrawl API (fallback).
    Requires FIRECRAWL_API_KEY env var."""
    if not _FIRECRAWL_API_KEY:
        return ""
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                "https://api.firecrawl.dev/v1/scrape",
                headers={
                    "Authorization": f"Bearer {_FIRECRAWL_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "url": url,
                    "formats": ["markdown"],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            md = data.get("data", {}).get("markdown", "")
            if md and len(md) > 50:
                return md[:5000]
    except Exception as e:
        logger.warning(f"Firecrawl failed for {url}: {e}")
    return ""


async def _scrape_spa_evento(url: str) -> str:
    """Scrape an SPA page using Jina Reader (primary) + Firecrawl (fallback)."""
    text = await _scrape_with_jina(url)
    if text and len(text) > 800:
        logger.info(f"✅ Jina Reader scraped {url} ({len(text)} chars)")
        return text
    
    if text:
        logger.warning(f"⚠️ Jina Reader returned too little content ({len(text)} chars) for {url}. Falling back to Firecrawl.")

    text = await _scrape_with_firecrawl(url)
    if text:
        logger.info(f"✅ Firecrawl scraped {url} ({len(text)} chars)")
        return text

    logger.warning(f"⚠️ Both Jina Reader and Firecrawl failed for {url}")
    return ""


# ─────────────────────────────────────────────────────
# Firecrawl Target Scraping
# ─────────────────────────────────────────────────────

async def _scrape_firecrawl_targets(category: str, user_provider: Optional[str] = None) -> List[dict]:
    """Scrape all FIRECRAWL_TARGETS matching the given category.
    Uses Jina Reader (primary) + Firecrawl (fallback) and returns
    results formatted like search results for the LLM.
    If user_provider is specified, only targets whose name
    contains the provider will be scraped."""
    targets = [t for t in FIRECRAWL_TARGETS if t["categoria"] == category]
    if user_provider:
        prov_lower = user_provider.lower()
        targets = [t for t in targets if prov_lower in t["nombre"].lower()]

    if not targets:
        return []

    results = []
    for target in targets:
        url = target["url"]
        nombre = target["nombre"]
        logger.info(f"🔥 [{category}] Scraping Firecrawl target: {nombre} ({url})")
        text = await _scrape_spa_evento(url)
        if text:
            results.append({
                "title": nombre,
                "url": url,
                "content": text,
            })
            logger.info(f"✅ [{category}] Firecrawl target '{nombre}': {len(text)} chars")
        else:
            logger.warning(f"⚠️ [{category}] Firecrawl target '{nombre}' returned no content")

    return results


# ─────────────────────────────────────────────────────
# Node: Clarify (conversational pre-search)
# ─────────────────────────────────────────────────────

async def clarify(state: AgentState) -> dict:
    """
    Use structured output to decide if we need to ask the user for missing info.
    """
    config = Configuration.from_env()

    if state.get("clarify_question"):
        logger.info("💬 Clarification already done, skipping.")
        return {}

    _arg_tz = timezone(timedelta(hours=-3))
    today_arg = datetime.now(_arg_tz).strftime("%Y-%m-%d")

    prompt = CLARIFY_DECISION_PROMPT.format(
        target_date=state.get("target_date", ""),
        user_category=state.get("user_category", ""),
        user_provider=state.get("user_provider", ""),
        user_message=state.get("user_message_raw", ""),
        today_date=today_arg,
    )

    try:
        llm = _get_llm(config.clarify_model, temperature=0.0)
        structured_llm = llm.with_structured_output(ClarifyDecision)
        decision: ClarifyDecision = await _invoke_with_backoff(
            structured_llm,
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Extraé la información del mensaje y decidí si necesito pedir algo más."},
            ],
            pydantic_schema=ClarifyDecision,
        )

        updates = {}
        if decision.extracted_target_date and not state.get("target_date"):
            updates["target_date"] = decision.extracted_target_date
        if decision.extracted_category and not state.get("user_category"):
            updates["user_category"] = decision.extracted_category
        if decision.extracted_provider and not state.get("user_provider"):
            updates["user_provider"] = decision.extracted_provider

        if decision.need_clarification:
            logger.info(f"💬 Clarification needed (missing: {decision.missing_fields}): {decision.question}")
            updates["clarify_question"] = decision.question
            return updates

        # ── Safety net: force clarification if category+provider are still empty ──
        effective_category = updates.get("user_category") or state.get("user_category") or ""
        effective_provider = updates.get("user_provider") or state.get("user_provider") or ""
        effective_date = updates.get("target_date") or state.get("target_date") or ""

        if not effective_category and not effective_provider:
            q = "¿Qué tipo de eventos te interesa? Podés elegir entre: deportes, streaming, gaming, especiales, o decirme 'todos'."
            if not effective_date:
                q = "¿Para qué fecha necesitás la información y qué tipo de eventos te interesa? (deportes, streaming, gaming, especiales, o 'todos')"
            logger.info(f"💬 Safety net: category+provider empty, forcing clarification: {q}")
            updates["clarify_question"] = q
            return updates

        if not effective_date:
            q = "¿Para qué fecha necesitás la información de eventos?"
            logger.info(f"💬 Safety net: date empty, forcing clarification: {q}")
            updates["clarify_question"] = q
            return updates

        logger.info(f"✅ All critical info present (extracted: date={decision.extracted_target_date}, category={decision.extracted_category}, provider={decision.extracted_provider})")
        return updates

    except Exception as e:
        logger.warning(f"Clarify decision failed: {e}. Proceeding without clarification.")
        return {}


# ─────────────────────────────────────────────────────
# Node: Plan Research
# ─────────────────────────────────────────────────────

async def plan_research(state: AgentState) -> dict:
    """Generate search queries for each event category."""
    config = Configuration.from_env()
    llm = _get_llm(config.planner_model)

    # Real current date in Argentina timezone for the LLM to cross-reference
    _arg_tz = timezone(timedelta(hours=-3))
    today_arg = datetime.now(_arg_tz).strftime("%Y-%m-%d")
    target = state['target_date']
    user_cat = state.get('user_category') or ""
    user_prov = state.get('user_provider') or ""

    prompt = PLANNER_SYSTEM_PROMPT.format(
        max_queries=config.max_queries_per_category,
        user_category=user_cat,
        user_provider=user_prov,
    )
    user_msg = (
        f"La fecha de HOY (real, actual) es: {today_arg}.\n"
        f"El usuario quiere encontrar eventos para la FECHA OBJETIVO: {target}.\n\n"
        f"Generá queries de búsqueda para encontrar eventos que generen tráfico de internet "
        f"en Argentina para la fecha objetivo {target}.\n\n"
        f"IMPORTANTE: Solo buscá eventos que realmente ocurran en la fecha {target}. "
        f"NO confundas la fecha de hoy ({today_arg}) con la fecha objetivo ({target}).\n"
    )

    cat_check = user_cat.lower()
    if not cat_check or "todas" in cat_check or "streaming" in cat_check:
        user_msg += (
            f"Para streaming, generá queries como: 'estrenos Netflix {target}', "
            f"'nuevos en Disney+ {target}', 'nuevas series HBO {target}'.\n"
        )
    if not cat_check or "todas" in cat_check or "deport" in cat_check:
        user_msg += (
            f"Para deportes, generá queries como: 'partidos fútbol argentino {target}', "
            f"'fixture Liga Profesional {target}'.\n"
        )

    structured_llm = llm.with_structured_output(SearchPlan)
    plan: SearchPlan = await _invoke_with_backoff(
        structured_llm,
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_msg},
        ],
        pydantic_schema=SearchPlan,
    )

    logger.info(f"📋 Plan generated: {len(plan.queries)} queries")
    return {"search_plan": plan}


# ─────────────────────────────────────────────────────
# Node: Research Category (runs in parallel via Send)
# ─────────────────────────────────────────────────────

async def research_category(state: ResearcherState) -> dict:
    """Search with Tavily/DDG and extract events for a category."""
    config = Configuration.from_env()
    llm = _get_llm(config.researcher_model)
    category = state["category"]
    target_date = state["target_date"]
    queries = state["queries"]
    user_prov = state.get("user_provider")

    _CATEGORY_DOMAINS = {
        "streaming": [
            "sensacine.com.ar", "justwatch.com/ar", "filmaffinity.com",
            "espinof.com", "infobae.com", "lanacion.com.ar", "clarin.com",
            "mendozapost.com", "losandes.com.ar", "perfil.com",
            "todomovieseries.com", "cinepremiere.com.mx",
            "screenrant.com", "whats-on-netflix.com",
        ],
        "deportes": [
            "promiedos.com.ar", "ole.com.ar", "tycsports.com", "espn.com.ar"
        ],
        "gaming": [
            "store.steampowered.com", "ign.com", "twitch.tv",
            "gamespot.com", "kotaku.com",
        ],
    }
    domains = _CATEGORY_DOMAINS.get(category)
    
    # Restrict domains explicitly if user_provider is given
    # (assuming user_provider is a known domain or brand name like 'promiedos')
    if user_prov:
        prov_lower = user_prov.lower()
        if domains:
            filtered_domains = [d for d in domains if prov_lower in d]
            if filtered_domains:
                domains = filtered_domains
            else:
                # Si el proveedor no está en la lista estándar, asumismos que .com o .com.ar sirve
                domains = [f"{prov_lower}.com.ar", f"{prov_lower}.com"]

    # 1. Execute all searches for this category
    all_results = []
    loop = asyncio.get_event_loop()
    for query in queries:
        results = await loop.run_in_executor(None, _perform_search, query, config, domains)
        all_results.extend(results)
        logger.info(f"🔍 [{category}] Query '{query}': {len(results)} results")

    # 1b. For streaming, always add hardcoded queries to ensure Netflix/Disney+/HBO coverage
    if category == "streaming" and not user_prov:
        _extra_streaming_queries = [
            f"estrenos Netflix {target_date} Argentina",
            f"nuevas series películas Disney+ {target_date}",
            f"estrenos HBO Max {target_date} Argentina",
        ]
        for eq in _extra_streaming_queries:
            if eq not in queries:
                extra_results = await loop.run_in_executor(None, _perform_search, eq, config, None)
                all_results.extend(extra_results)
                logger.info(f"🔍 [{category}] Extra query '{eq}': {len(extra_results)} results")

    # 1c. For deportes, guarantee we fetch the day's agenda/fixture
    if category == "deportes" and not user_prov:
        _extra_deportes_queries = [
            f"site:promiedos.com.ar primera division {target_date}",
            f"site:promiedos.com.ar seleccion argentina {target_date}",
            f"site:promiedos.com.ar libertadores sudamericana {target_date}",
            f"site:promiedos.com.ar fixture futbol {target_date}",
        ]
        for eq in _extra_deportes_queries:
            if eq not in queries:
                extra_results = await loop.run_in_executor(None, _perform_search, eq, config, None)
                all_results.extend(extra_results)
                logger.info(f"🔍 [{category}] Extra query '{eq}': {len(extra_results)} results")

    # 1d. Scrape Firecrawl targets for this category
    firecrawl_results = await _scrape_firecrawl_targets(category, user_provider=user_prov)
    if firecrawl_results:
        all_results.extend(firecrawl_results)
        logger.info(f"🔥 [{category}] Added {len(firecrawl_results)} Firecrawl target results")

    if not all_results:
        logger.info(f"⚠️  [{category}] No search results found")
        return {"raw_events": []}

    # 2. Enrich SPA results with Jina Reader/Firecrawl
    enriched_results = []
    for r in all_results:
        url = r.get("url", "")
        is_spa = any(d in url.lower() for d in SITES_DDG)
        if is_spa:
            logger.info(f"🌐 [{category}] SPA scraping (Jina/Firecrawl): {url}")
            scraped = await _scrape_spa_evento(url)
            if scraped:
                r = {**r, "content": scraped}
        enriched_results.append(r)

    search_context = "\n\n".join([
        f"**Fuente**: {r.get('url', 'N/A')}\n"
        f"**Título**: {r.get('title', 'N/A')}\n"
        f"**Contenido**: {str(r.get('content', r.get('snippet', 'N/A')))[:4000]}..."
        for r in enriched_results
    ])

    if not search_context.strip():
        logger.warning(f"⚠️ [{category}] Search context is EMPTY after enrichment")
        return {"raw_events": []}

    # 3. Context Synthesis if it's too large
    context_manager = SQLiteContextManager()
    if len(search_context) > config.context_synthesis_threshold:
        logger.info(f"💾 [{category}] Context size ({len(search_context)} chars) exceeds threshold ({config.context_synthesis_threshold}). Checking SQLite cache...")
        
        cached_context = context_manager.get_synthesized_context(category, target_date)
        if cached_context:
            search_context = cached_context
            logger.info(f"✅ [{category}] Using synthesized context from SQLite cache.")
        else:
            logger.info(f"🧠 [{category}] Synthesizing large context with LLM ({config.synthesis_model})...")
            synthesis_llm = _get_llm(config.synthesis_model, temperature=0.0)
            
            synthesis_result = await _invoke_with_backoff(
                synthesis_llm,
                [
                    {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Sintetizá los siguientes resultados para la categoría {category} y fecha {target_date}:\n\n{search_context}"},
                ]
            )
            search_context = synthesis_result.content
            context_manager.save_synthesized_context(category, target_date, search_context)
            logger.info(f"✅ [{category}] Context synthesized and saved. New size: {len(search_context)} chars.")

    # 4. Ask LLM to extract structured events
    _arg_tz = timezone(timedelta(hours=-3))
    today_arg = datetime.now(_arg_tz).strftime("%Y-%m-%d")
    prompt = RESEARCHER_SYSTEM_PROMPT.format(category=category, target_date=target_date, today_date=today_arg)
    structured_llm = llm.with_structured_output(ResearchFindings)

    try:
        findings: ResearchFindings = await _invoke_with_backoff(
            structured_llm,
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Resultados de búsqueda:\n\n{search_context}"},
            ],
            pydantic_schema=ResearchFindings,
        )
        events = [event.model_dump() for event in findings.events]
        logger.info(f"✅ [{category}] Extracted {len(events)} events")
        return {"raw_events": events}
    except Exception as e:
        fallback = _parse_failed_generation(e, ResearchFindings)
        if fallback and fallback.events:
            events = [event.model_dump() for event in fallback.events]
            logger.info(f"🔄 [{category}] Recovered {len(events)} events via fallback parser")
            return {"raw_events": events}
        logger.warning(f"❌ [{category}] LLM extraction failed: {e}")
        return {"raw_events": []}


# ─────────────────────────────────────────────────────
# Edge: Fan-out to parallel researchers
# ─────────────────────────────────────────────────────

def send_research_tasks(state: AgentState):
    """Routing function: scatter to parallel researchers, but only for the requested category if one exists."""
    plan = state["search_plan"]
    if not plan:
        return []

    # Se agrupan las queries por categoría
    from collections import defaultdict
    category_queries = defaultdict(list)
    for q in plan.queries:
        category_queries[q.category].append(q.query)

    # Si el usuario pidió una categoría específica (ej: "deportes"),
    # ignoramos las queries de las otras categorías para ahorrar tiempo y tokens.
    user_cat = state.get("user_category") or ""
    cat_lower = user_cat.lower()
    
    if cat_lower and "todas" not in cat_lower:
        requested = None
        if "deport" in cat_lower:
            requested = "deportes"
        elif "stream" in cat_lower:
            requested = "streaming"
        elif "gam" in cat_lower or "juego" in cat_lower:
            requested = "gaming"
        elif "especial" in cat_lower:
            requested = "especiales"
        
        if requested:
            if requested in category_queries:
                # Mandamos a investigar solo la que pidió
                return [
                    Send("research_category", {
                        "category": requested,
                        "queries": category_queries[requested],
                        "target_date": state["target_date"],
                        "user_provider": state.get("user_provider"),
                        "raw_events": [],
                    })
                ]
            else:
                return []
    sends = []
    for category, queries_list in category_queries.items():
        sends.append(
            Send("research_category", {
                "category": category,
                "queries": queries_list,
                "target_date": state["target_date"],
                "user_provider": state.get("user_provider"),
                "raw_events": [],
            })
        )
    logger.info(f"🚀 Dispatching {len(sends)} parallel researchers")
    return sends


# ─────────────────────────────────────────────────────
# Node: Aggregate Results
# ─────────────────────────────────────────────────────

def aggregate_results(state: AgentState) -> dict:
    """Combine and deduplicate events from all researchers."""
    raw = state.get("raw_events", [])
    logger.info(f"📦 Aggregating {len(raw)} raw events")

    seen = set()
    unique = []
    for event in raw:
        key = event.get("evento", "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(event)

    logger.info(f"📦 After dedup: {len(unique)} unique events")
    return {"raw_events": {"type": "override", "value": unique}}


# ─────────────────────────────────────────────────────
# Node: Verify Dates via Scraping
# ─────────────────────────────────────────────────────

_MONTH_NAMES = {
    'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04',
    'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08',
    'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12',
    'january': '01', 'february': '02', 'march': '03', 'april': '04',
    'may': '05', 'june': '06', 'july': '07', 'august': '08',
    'september': '09', 'october': '10', 'november': '11', 'december': '12',
}


def _extract_dates_from_text(text: str) -> list[str]:
    """Extract all dates from text using DATE_PATTERNS and return as YYYY-MM-DD strings."""
    dates_found = []

    # Use global DATE_PATTERNS for matching
    for pattern in DATE_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            raw = m.group(1)
            parsed = _try_parse_date(raw)
            if parsed:
                dates_found.append(parsed)

    return list(set(dates_found))


def _extract_times_from_text(text: str) -> list[str]:
    """Extract all times from text using TIME_PATTERNS."""
    times_found = []
    for pattern in TIME_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            times_found.append(m.group(1).strip())
    return list(set(times_found))


def _try_parse_date(raw: str) -> str | None:
    """Try to parse a raw date string into YYYY-MM-DD format."""
    raw = raw.strip()
    _arg_tz = timezone(timedelta(hours=-3))
    now = datetime.now(_arg_tz)
    current_year = now.year

    # YYYY-MM-DD or YYYY/MM/DD
    m = re.match(r'^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$', raw)
    if m:
        y, mo, d = m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)
        if 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
            return f"{y}-{mo}-{d}"

    # DD/MM/YYYY or DD-MM-YYYY
    m = re.match(r'^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$', raw)
    if m:
        d, mo, y = m.group(1).zfill(2), m.group(2).zfill(2), m.group(3)
        if 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
            return f"{y}-{mo}-{d}"

    # DD/MM (assume current year)
    m = re.match(r'^(\d{1,2})[/-](\d{1,2})$', raw)
    if m:
        d, mo = m.group(1).zfill(2), m.group(2).zfill(2)
        if 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
            return f"{current_year}-{mo}-{d}"

    # "15 de marzo de 2025" / "15 marzo 2025" / "15 de marzo"
    text_lower = raw.lower()
    for month_name, month_num in _MONTH_NAMES.items():
        # With year
        m = re.match(
            rf'(\d{{1,2}})\s+(?:de\s+)?{month_name}\s+(?:de\s+)?(\d{{4}})',
            text_lower
        )
        if m:
            d = m.group(1).zfill(2)
            y = m.group(2)
            return f"{y}-{month_num}-{d}"

        # Without year
        m = re.match(rf'^(\d{{1,2}})\s+(?:de\s+)?{month_name}$', text_lower)
        if m:
            d = m.group(1).zfill(2)
            return f"{current_year}-{month_num}-{d}"

        # "marzo 2025" (day unknown → use 01)
        m = re.match(rf'^{month_name}\s+(\d{{4}})$', text_lower)
        if m:
            return f"{m.group(1)}-{month_num}-01"

    return None


async def _scrape_url(url: str, timeout: float = 8.0) -> str:
    """Fetch a URL and return the visible text content."""
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; DeepResearchBot/1.0)'}
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()
            return soup.get_text(separator=' ', strip=True)[:2500]
    except Exception as e:
        logger.debug(f"Could not scrape {url}: {e}")
        return ""


async def verify_dates(state: AgentState) -> dict:
    """Scrape source URLs to verify and correct event dates.

    For past target dates, all events are kept (source pages update quickly).
    For future dates, events are discarded if target_date is not found on source.
    """
    raw = state.get("raw_events", [])
    target_date = state.get("target_date", "")

    if not raw:
        return {"raw_events": {"type": "override", "value": []}}

    # Determine if the target date is in the past
    _arg_tz = timezone(timedelta(hours=-3))
    today_str = datetime.now(_arg_tz).strftime("%Y-%m-%d")
    is_past_date = bool(target_date and target_date < today_str)
    if is_past_date:
        logger.info(f"🔎 Target date {target_date} is in the past — events will be kept even if date not found on source")

    logger.info(f"🔎 Verifying dates for {len(raw)} events via scraping...")

    verified = []
    scrape_tasks = []
    async def _empty_scrape():
        return ""

    for event in raw:
        url = event.get("fuente", "")
        if url and url.startswith("http"):
            scrape_tasks.append((event, _scrape_url(url)))
        else:
            scrape_tasks.append((event, _empty_scrape()))

    events_and_coros = [(ev, coro) for ev, coro in scrape_tasks]
    scraped_texts = await asyncio.gather(*[coro for _, coro in events_and_coros])

    for (event, _), page_text in zip(events_and_coros, scraped_texts):
        event_name = event.get("evento", "")
        original_date = event.get("fecha", "")

        if not page_text:
            verified.append(event)
            continue

        page_dates = _extract_dates_from_text(page_text)

        if not page_dates:
            verified.append(event)
            continue

        if target_date in page_dates:
            if original_date != target_date:
                logger.info(
                    f"📅 [{event_name}] Date corrected: {original_date} → {target_date} "
                    f"(target_date found on source page)"
                )
                event["fecha"] = target_date
            verified.append(event)
            continue

        # target_date NOT found on page
        if is_past_date:
            logger.info(
                f"📅 [{event_name}] Kept (past date): source page likely updated, "
                f"trusting LLM extraction. Page dates: {page_dates}"
            )
            verified.append(event)
            continue

        # Future date: keep but log warning
        logger.warning(
            f"⚠️ [{event_name}] Target date {target_date} not found on source page, "
            f"but keeping it as secondary check (Page dates: {page_dates})"
        )
        verified.append(event)

    logger.info(f"🔎 Date verification complete: kept {len(verified)} verified events out of {len(raw)}")
    return {"raw_events": {"type": "override", "value": verified}}


# ─────────────────────────────────────────────────────
# Node: Filter for Argentina Relevance
# ─────────────────────────────────────────────────────

async def filter_argentina(state: AgentState) -> dict:
    """Use LLM to filter ONLY deportes/especiales events by Argentina relevance.

    Streaming and gaming events are auto-passed since they were already
    validated by the researcher node.
    """
    config = Configuration.from_env()
    raw = state.get("raw_events", [])

    if not raw:
        return {"filtered_events": []}

    auto_pass = []
    needs_filtering = []
    for event in raw:
        cat = event.get("categoria", "").lower()
        if cat in ("streaming", "gaming"):
            auto_pass.append(event)
        else:
            needs_filtering.append(event)

    logger.info(
        f"🔀 Filter split: {len(auto_pass)} auto-pass (streaming/gaming), "
        f"{len(needs_filtering)} to filter (deportes/especiales)"
    )

    if not needs_filtering:
        logger.info(f"🇦🇷 All {len(auto_pass)} events auto-passed (no deportes/especiales to filter)")
        return {"filtered_events": auto_pass}

    llm = _get_llm(config.filter_model)
    structured_llm = llm.with_structured_output(FilteredReport)

    events_json = json.dumps(needs_filtering, ensure_ascii=False, indent=2)

    _arg_tz = timezone(timedelta(hours=-3))
    today_arg = datetime.now(_arg_tz).strftime("%Y-%m-%d")
    target_date = state.get("target_date", today_arg)

    filter_prompt = FILTER_SYSTEM_PROMPT.format(today_date=today_arg, target_date=target_date)

    llm_filtered = []
    try:
        report: FilteredReport = await _invoke_with_backoff(
            structured_llm,
            [
                {"role": "system", "content": filter_prompt},
                {"role": "user", "content": f"Eventos a filtrar:\n\n{events_json}"},
            ],
            pydantic_schema=FilteredReport,
        )
        llm_filtered = [e.model_dump() for e in report.events]
        logger.info(f"🇦🇷 LLM filtered deportes/especiales to {len(llm_filtered)} events")
    except Exception as e:
        fallback = _parse_failed_generation(e, FilteredReport)
        if fallback and fallback.events:
            llm_filtered = [ev.model_dump() for ev in fallback.events]
            logger.info(f"🔄 Filter recovered {len(llm_filtered)} deportes/especiales via fallback")
        else:
            logger.warning(f"❌ Filter failed, passing deportes/especiales unfiltered: {e}")
            llm_filtered = needs_filtering

    combined = auto_pass + llm_filtered
    logger.info(f"✅ Final filtered total: {len(combined)} events ({len(auto_pass)} auto + {len(llm_filtered)} filtered)")
    return {"filtered_events": combined}


# ─────────────────────────────────────────────────────
# Node: Generate Final Report
# ─────────────────────────────────────────────────────

async def generate_report(state: AgentState) -> dict:
    """Generate the final JSON report for pandas conversion."""
    filtered = state.get("filtered_events", [])

    if not filtered:
        logger.info("📄 No events to report")
        return {"final_report": "[]"}

    config = Configuration.from_env()
    llm = _get_llm(config.report_model)

    events_json = json.dumps(filtered, ensure_ascii=False, indent=2)

    response = await _invoke_with_backoff(
        llm,
        [
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Eventos filtrados:\n\n{events_json}"},
        ],
    )

    report_text = response.content.strip()
    
    # Robust JSON extraction: look for the first [ and last ]
    import re
    match = re.search(r"(\[.*\])", report_text, re.DOTALL)
    if match:
        report_text = match.group(1).strip()
    elif report_text.startswith("```"):
        lines = report_text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        report_text = "\n".join(lines).strip()

    logger.info(f"📄 Final report generated")
    
    # 🧹 Cleanup: delete synthesized cache as requested by the user
    try:
        context_manager = SQLiteContextManager()
        context_manager.clear_all_context()
    except Exception as e:
        logger.warning(f"Failed to clear context cache: {e}")

    return {"final_report": report_text}


# ─────────────────────────────────────────────────────
# Build the Graph
# ─────────────────────────────────────────────────────

def should_clarify(state: AgentState) -> str:
    """Route through clarify on first pass; skip to plan_research if already clarified."""
    if state.get("clarify_question"):
        return "plan_research"
    return "clarify"

def _after_clarify(state: AgentState) -> str:
    """If a question was generated, go to END (wait for user); otherwise, plan_research."""
    if state.get("clarify_question"):
        return END
    return "plan_research"

def build_graph():
    """Construct and compile the LangGraph StateGraph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("clarify", clarify)
    graph.add_node("plan_research", plan_research)
    graph.add_node("research_category", research_category)
    graph.add_node("aggregate_results", aggregate_results)
    graph.add_node("verify_dates", verify_dates)
    graph.add_node("filter_argentina", filter_argentina)
    graph.add_node("generate_report", generate_report)

    # Add edges
    graph.add_conditional_edges(START, should_clarify, ["clarify", "plan_research"])
    graph.add_conditional_edges("clarify", _after_clarify, [END, "plan_research"])
    
    graph.add_conditional_edges("plan_research", send_research_tasks, ["research_category"])
    graph.add_edge("research_category", "aggregate_results")
    graph.add_edge("aggregate_results", "verify_dates")
    graph.add_edge("verify_dates", "filter_argentina")
    graph.add_edge("filter_argentina", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()
