"""
LangGraph graph definition for the Travel Agent.
Same architecture as the Events agent — plan → parallel search → aggregate → rank → report.

Architecture:
  User Input (origin, destination, dates)
       │
       ▼
  plan_search ──► Generates search queries per type
       │
       ▼ (Send to parallel searchers)
  search_flights ──► Tavily search + LLM extraction (×4 types)
       │
       ▼
  aggregate_routes ──► Combine all routes
       │
       ▼
  rank_and_optimize ──► Rank by price/duration/convenience
       │
       ▼
  generate_itinerary ──► Structured JSON for pandas
"""

import json
import logging
import time
from pathlib import Path
from typing import List

import os
import yaml
import asyncio
from bs4 import BeautifulSoup
import httpx
# Firecrawl API key for fallback scraping (optional)
_FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")

from duckduckgo_search import DDGS

from groq import RateLimitError
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langsmith import traceable

from state import (
    TravelState,
    SegmentSearcherState,
    RoutePlan,
    RouteCandidate,
    RouteFindings,
    RankedRoutes,
    RouteOption,
    ClarifyDecision,
)
from configuration import Configuration
from prompts import (
    PLANNER_SYSTEM_PROMPT,
    CLARIFY_DECISION_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    RANKER_SYSTEM_PROMPT,
    ITINERARY_SYSTEM_PROMPT,
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


def _invoke_with_backoff(
    chain,
    messages: list,
    max_attempts: int = 6,
    pydantic_schema=None,
):
    """Invoke a LangChain chain with exponential backoff + jitter on rate limit errors.

    Handles all Groq rate limit surfaces:
      - TPM / RPM  -> 429 transient -> exponential backoff with full jitter
      - TPD        -> 429 with "tokens per day" -> try fallback model immediately
      - RPD        -> 429 with "requests per day" -> try fallback model immediately

    Args:
        chain:           The LangChain chain (LLM or LLM.with_structured_output(...)).
        messages:        Messages list to pass to chain.invoke().
        max_attempts:    Max retry attempts for RPM/TPM errors.
        pydantic_schema: If chain uses with_structured_output, pass the Pydantic class here
                         so that the fallback model can be properly reconstructed.
    """
    import random
    import threading

    _DAILY_KEYWORDS = (
        'tokens per day', 'tpd',
        'requests per day', 'rpd',
        'daily limit', 'daily token',
    )
    _FALLBACK_MODEL = "llama-3.3-70b-versatile"

    # Global semaphore to limit concurrency across LangGraph threads (fixes 429 errors)
    if not hasattr(_invoke_with_backoff, "_semaphore"):
        _invoke_with_backoff._semaphore = threading.Semaphore(2) # Max 2 concurrent LLM calls

    delay = 4.0
    for attempt in range(max_attempts):
        try:
            with _invoke_with_backoff._semaphore:
                return chain.invoke(messages)

        except RateLimitError as e:
            error_msg = str(e).lower()

            # ── Daily limit hit (TPD or RPD) ─────────────────────────────────
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
            time.sleep(wait)
            delay = min(delay * 2, 60.0)

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

@traceable(name="ddg_search_flights", run_type="tool")
def _ddg_search(query: str, config: Configuration) -> List[dict]:
    """Execute a single DuckDuckGo search using Langchain wrapper and return formatted results."""
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=config.search_max_results)
        search = DuckDuckGoSearchResults(api_wrapper=wrapper, output_format="list")
        res = search.invoke(query)
        
        # Format the Langchain output (snippet, title, link) to our standard expected format
        formatted_results = []
        for r in res:
            formatted_results.append({
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "content": r.get("snippet", "")
            })
        return formatted_results
    except Exception as e:
        logger.error(f"DuckDuckGo (Langchain) search failed for '{query}': {e}")
        return []

def _scrape_with_jina_sync(url: str) -> str:
    """Scrape a page using Jina Reader API (r.jina.ai). Synchronous version.
    Free, handles JS-rendered pages, returns clean markdown."""
    import httpx as _httpx
    jina_url = f"https://r.jina.ai/{url}"
    try:
        resp = _httpx.get(
            jina_url,
            timeout=15.0,
            follow_redirects=True,
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


def _scrape_with_firecrawl_sync(url: str) -> str:
    """Scrape a page using Firecrawl API (fallback). Synchronous version.
    Requires FIRECRAWL_API_KEY env var."""
    import httpx as _httpx
    if not _FIRECRAWL_API_KEY:
        return ""
    try:
        resp = _httpx.post(
            "https://api.firecrawl.dev/v1/scrape",
            timeout=20.0,
            headers={
                "Authorization": f"Bearer {_FIRECRAWL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"url": url, "formats": ["markdown"]},
        )
        resp.raise_for_status()
        data = resp.json()
        md = data.get("data", {}).get("markdown", "")
        if md and len(md) > 50:
            return md[:5000]
    except Exception as e:
        logger.warning(f"Firecrawl failed for {url}: {e}")
    return ""


def _scrape_spa(url: str) -> str:
    """Scrape an SPA page using Jina Reader (primary) + Firecrawl (fallback)."""
    text = _scrape_with_jina_sync(url)
    if text:
        logger.info(f"✅ Jina Reader scraped {url} ({len(text)} chars)")
        return text

    text = _scrape_with_firecrawl_sync(url)
    if text:
        logger.info(f"✅ Firecrawl scraped {url} ({len(text)} chars)")
        return text

    logger.warning(f"⚠️ Both Jina and Firecrawl failed for {url}")
    return ""


# ─────────────────────────────────────────────────────
# Node: Clarify (conversational pre-search)
# ─────────────────────────────────────────────────────

def clarify(state: TravelState) -> dict:
    """
    Use structured output to decide if we need to ask the user for missing info.
    The LLM extracts origin/destination/dates from the user message AND decides
    if clarification is needed. Extracted fields are always populated into state.
    """
    config = Configuration.from_env()

    # If we already asked a clarifying question (re-invocation with answer), skip
    if state.get("clarify_question"):
        logger.info("💬 Clarification already done, skipping.")
        return {}

    import time as _time_mod
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    _arg_tz = _tz(offset=_td(hours=-3))
    today_arg = _dt.now(_arg_tz).strftime("%Y-%m-%d")

    prompt = CLARIFY_DECISION_PROMPT.format(
        origin=state.get("origin", ""),
        destination=state.get("destination", ""),
        travel_dates=state.get("travel_dates", ""),
        user_message=state.get("user_message_raw", ""),
        today_date=today_arg,
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

        # Always populate state with whatever the LLM extracted
        updates = {}
        if decision.extracted_origin and not state.get("origin"):
            updates["origin"] = decision.extracted_origin
        if decision.extracted_destination and not state.get("destination"):
            updates["destination"] = decision.extracted_destination
        if decision.extracted_travel_dates and not state.get("travel_dates"):
            updates["travel_dates"] = decision.extracted_travel_dates

        # Default origin if still empty
        if not state.get("origin") and not updates.get("origin"):
            updates["origin"] = "Buenos Aires"

        if decision.need_clarification:
            logger.info(f"💬 Clarification needed (missing: {decision.missing_fields}): {decision.question}")
            updates["clarify_question"] = decision.question
            return updates

        logger.info(f"✅ All critical info present (extracted: origin={decision.extracted_origin}, dest={decision.extracted_destination}, dates={decision.extracted_travel_dates})")
        return updates

    except Exception as e:
        logger.warning(f"Clarify decision failed: {e}. Proceeding without clarification.")
        updates = {}
        if not state.get("origin"):
            updates["origin"] = "Buenos Aires"
        return updates


# ─────────────────────────────────────────────────────
# Node: Plan Search
# ─────────────────────────────────────────────────────

def plan_search(state: TravelState) -> dict:
    """Generate potential multi-segment travel routes."""
    config = Configuration.from_env()
    llm = _get_llm(config.planner_model)

    prompt = PLANNER_SYSTEM_PROMPT
    
    budget_info = ""
    if state.get("budget_max_usd"):
        budget_info = f"\nPresupuesto máximo: USD {state['budget_max_usd']}"
    
    user_msg = (
        f"Buscá viajes con estas condiciones:\n"
        f"- Origen: {state['origin']}\n"
        f"- Destino: {state['destination']}\n"
        f"- Fechas: {state['travel_dates']}\n"
        f"- Flexibilidad: ±{state.get('flexibility_days', 3)} días"
        f"{budget_info}\n\n"
        f"Generá rutas de viaje divididas por segmentos."
    )

    structured_llm = llm.with_structured_output(RoutePlan)
    plan: RoutePlan = _invoke_with_backoff(
        structured_llm,
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_msg},
        ],
        pydantic_schema=RoutePlan,
    )

    logger.info(f"📋 Route plan generated: {len(plan.routes)} candidate routes")
    # Dump each RouteCandidate to dicts: list of {segments: [...], description: ...}
    routes_dumped = [
        [seg.model_dump() for seg in candidate.segments]
        for candidate in plan.routes
    ]
    return {"route_plan": routes_dumped}


# ─────────────────────────────────────────────────────
# Node: Search Segment (runs in parallel via Send)
# ─────────────────────────────────────────────────────

def search_segment(state: SegmentSearcherState) -> dict:
    """Search with DDG and extract options for a specific segment."""
    config = Configuration.from_env()
    llm = _get_llm(config.researcher_model)
    segment = state["segment"]
    travel_dates = state["travel_dates"]
    
    origin = segment.get("from_city")
    destination = segment.get("to_city")
    mode_hint = segment.get("mode_hint")

    # 1. Execute DDG search. 
    # If the user provided dates spanning multiple days/months, it's likely a round trip.
    round_trip_hint = " ida y vuelta" if " al " in travel_dates.lower() or " a " in travel_dates.lower() or " y " in travel_dates.lower() or "-" in travel_dates else ""
    query = f"viaje más barato {mode_hint} {origin} a {destination} {travel_dates}{round_trip_hint}"

    results = _ddg_search(query, config)
    logger.info(f"🔍 [Segmento {origin}->{destination}] Query '{query}': {len(results)} results")

    if not results:
        logger.info(f"⚠️  [Segmento {origin}->{destination}] No search results found")
        return {"segment_results": []}

    # SPA domains that require javascript rendering
    spa_domains = ["jetsmart.com", "flybondi.com", "plataforma10", "recorrido.cl", "andesmar", "centraldepasajes", "despegar.com", "kayak.com"]

    # 2. Format search results for the LLM
    search_context_parts = []
    for r in results:
        url = r.get('url', '')
        content = r.get('content', '')
        
        # Override content via Jina/Firecrawl if it's an SPA domain
        is_spa = any((d in url.lower() for d in spa_domains))
        if is_spa:
            logger.info(f"🌐 Scraping SPA {url} with Jina/Firecrawl...")
            scraped_content = _scrape_spa(url)
            if scraped_content:
                content = scraped_content
                
        search_context_parts.append(
            f"**Fuente**: {url}\n"
            f"**Título**: {r.get('title', 'N/A')}\n"
            f"**Contenido**: {content}"
        )
        
    search_context = "\n\n".join(search_context_parts)

    # 3. Ask LLM to extract structured options
    prompt = RESEARCHER_SYSTEM_PROMPT.format(
        origin=origin,
        destination=destination,
        mode_hint=mode_hint,
        travel_dates=travel_dates,
    )
    structured_llm = llm.with_structured_output(RouteFindings)

    try:
        findings: RouteFindings = _invoke_with_backoff(
            structured_llm,
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Resultados de búsqueda:\n\n{search_context}"},
            ],
            pydantic_schema=RouteFindings,
        )
        routes = [route.model_dump() for route in findings.routes]
        logger.info(f"✅ [Segmento {origin}->{destination}] Extracted {len(routes)} options")
        return {"segment_results": routes}
    except Exception as e:
        logger.warning(f"❌ [Segmento {origin}->{destination}] LLM extraction failed: {e}")
        return {"segment_results": []}


# ─────────────────────────────────────────────────────
# Edge: Fan-out to parallel segment searchers
# ─────────────────────────────────────────────────────

def route_to_segments(state: TravelState) -> list[Send]:
    """Create a Send for each unique segment to run searchers in parallel."""
    routes = state.get("route_plan", [])
    if not routes:
        return []

    # Get unique segments to avoid duplicate searches
    unique_segments = {}
    for route in routes:
        for seg in route:
            key = f"{seg.get('from_city')}-{seg.get('to_city')}-{seg.get('mode_hint')}"
            unique_segments[key] = seg

    sends = []
    for key, seg in unique_segments.items():
        sends.append(
            Send(
                "search_segment",
                {
                    "segment": seg,
                    "travel_dates": state["travel_dates"],
                    "segment_results": [],
                },
            )
        )

    logger.info(f"🚀 Dispatching {len(sends)} parallel segment searchers")
    return sends


# ─────────────────────────────────────────────────────
# Node: Aggregate Routes (now combining segments)
# ─────────────────────────────────────────────────────

def aggregate_routes(state: TravelState) -> dict:
    """Combine segment results and build end-to-end routes."""
    segment_results = state.get("segment_results", [])
    logger.info(f"📦 Aggregating {len(segment_results)} raw options from segments")

    # For this simplified version without a full cartesian product optimizer,
    # we just pass these up to the ranker directly, as the LLM Ranker can build the final combo
    # or the segment_searcher found end-to-end prices if queried well.
    # A true combiner would build cartesian paths here.
    
    unique = []
    seen = set()
    for route in segment_results:
        key = route.get("ruta", "").strip().lower() + "-" + str(route.get("precio_usd"))
        if key and key not in seen:
            seen.add(key)
            unique.append(route)

    logger.info(f"📦 After dedup: {len(unique)} unique option segments")
    # Store in segment_results overriding so we don't duplicate
    return {"segment_results": {"type": "override", "value": unique}}

# ─────────────────────────────────────────────────────
# Node: Rank and Optimize
# ─────────────────────────────────────────────────────

def rank_and_optimize(state: TravelState) -> dict:
    """Use LLM to rank routes or combined segments by price/duration/convenience."""
    config = Configuration.from_env()
    raw = state.get("segment_results", [])

    if not raw:
        return {"final_ranking": []}

    llm = _get_llm(config.ranker_model)
    structured_llm = llm.with_structured_output(RankedRoutes)

    routes_json = json.dumps(raw, ensure_ascii=False, indent=2)

    try:
        ranked: RankedRoutes = _invoke_with_backoff(
            structured_llm,
            [
                {"role": "system", "content": RANKER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Rutas a rankear:\n\n{routes_json}"},
            ],
            pydantic_schema=RankedRoutes,
        )
        ranked_list = [r.model_dump() for r in ranked.routes]
        logger.info(f"🏆 Ranked {len(ranked_list)} routes. Recommendation: {ranked.recommendation[:80]}...")
        return {"final_ranking": ranked_list}
    except Exception as e:
        logger.warning(f"❌ Ranking failed: {e}")
        return {"final_ranking": raw}


# ─────────────────────────────────────────────────────
# Node: Generate Itinerary
# ─────────────────────────────────────────────────────

def generate_itinerary(state: TravelState) -> dict:
    """Generate the final JSON itinerary for pandas conversion."""
    ranked = state.get("final_ranking", [])

    if not ranked:
        logger.info("📄 No routes to report")
        return {"final_itinerary": "[]"}

    config = Configuration.from_env()
    llm = _get_llm(config.report_model)

    # Enforce top-3 at code level — the LLM prompt also says max 3, but this prevents
    # passing a huge context that could cause hallucination or excess token usage.
    top_routes = ranked[:3]

    routes_json = json.dumps(top_routes, ensure_ascii=False, indent=2)

    response = _invoke_with_backoff(
        llm,
        [
            {"role": "system", "content": ITINERARY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Rutas ranqueadas:\n\n{routes_json}"},
        ],
    )

    report_text = response.content.strip()

    # Clean markdown fences if present
    if report_text.startswith("```"):
        lines = report_text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        report_text = "\n".join(lines)

    logger.info(f"📄 Final itinerary generated")
    return {"final_itinerary": report_text}


# ─────────────────────────────────────────────────────
# Build the Graph
# ─────────────────────────────────────────────────────

def should_clarify(state: TravelState) -> str:
    """Route through clarify on first pass; skip to plan_search if already clarified."""
    if state.get("clarify_question"):
        # Already clarified in a previous invocation, skip to search
        return "plan_search"
    # First pass — always go through LLM-based clarify decision
    return "clarify"

def _after_clarify(state: TravelState) -> str:
    """After the clarify node: if a question was generated, go to END (wait for user);
    otherwise proceed to plan_search."""
    if state.get("clarify_question"):
        return END
    return "plan_search"

def build_graph():
    """Construct and compile the LangGraph StateGraph."""
    graph = StateGraph(TravelState)

    # Add nodes
    graph.add_node("clarify", clarify)
    graph.add_node("plan_search", plan_search)
    graph.add_node("search_segment", search_segment)
    graph.add_node("aggregate_routes", aggregate_routes)
    graph.add_node("rank_and_optimize", rank_and_optimize)
    graph.add_node("generate_itinerary", generate_itinerary)

    # Add edges
    graph.add_conditional_edges(START, should_clarify, ["clarify", "plan_search"])
    graph.add_conditional_edges(
        "clarify",
        _after_clarify,
        [END, "plan_search"],
    )
    
    graph.add_conditional_edges("plan_search", route_to_segments, ["search_segment"])
    graph.add_edge("search_segment", "aggregate_routes")
    graph.add_edge("aggregate_routes", "rank_and_optimize")
    graph.add_edge("rank_and_optimize", "generate_itinerary")
    graph.add_edge("generate_itinerary", END)

    return graph.compile()
