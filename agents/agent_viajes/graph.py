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
try:
    from playwright.async_api import async_playwright
except ImportError:
    async_playwright = None

from duckduckgo_search import DDGS

from groq import RateLimitError
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langsmith import traceable
from tavily import TavilyClient

from state import (
    TravelState,
    SegmentSearcherState,
    RoutePlan,
    RouteCandidate,
    RouteFindings,
    RankedRoutes,
    RouteOption,
)
from configuration import Configuration
from prompts import (
    PLANNER_SYSTEM_PROMPT,
    CLARIFY_SYSTEM_PROMPT,
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


def _invoke_with_backoff(chain, messages: list, max_attempts: int = 5):
    """Invoke a LangChain chain with exponential backoff on 429 rate limit errors."""
    delay = 5.0
    for attempt in range(max_attempts):
        try:
            return chain.invoke(messages)
        except RateLimitError as e:
            error_msg = str(e).lower()
            if 'tokens per day' in error_msg or 'tpd' in error_msg:
                logger.warning("Daily Token Limit hit in Viajes! Falling back to secondary model 'llama-3.3-70b-versatile'.")
                fallback_llm = _get_llm("llama-3.3-70b-versatile")
                if hasattr(chain, 'bound') and hasattr(chain, 'kwargs') and 'response_format' in chain.kwargs:
                    raise Exception(f"RateLimitError on structured LLM, requires manual schema rescue: {e}")
                
                try:
                    return fallback_llm.invoke(messages)
                except Exception as fallback_e:
                    logger.error(f"Fallback model also failed: {fallback_e}")
                    raise e

            if attempt == max_attempts - 1:
                raise
            logger.warning(
                f"Rate limit hit (attempt {attempt + 1}/{max_attempts}). "
                f"Waiting {delay:.1f}s before retry..."
            )
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
        except Exception:
            raise


@traceable(name="tavily_search_flights", run_type="tool")
def _tavily_search(query: str, config: Configuration) -> List[dict]:
    """Execute a single Tavily search and return results. Falls back to DuckDuckGo on error."""
    client = TavilyClient()
    try:
        response = client.search(
            query=query,
            max_results=config.tavily_max_results,
            search_depth=config.tavily_search_depth,
            include_answer=True,
        )
        return response.get("results", [])
    except Exception as e:
        logger.warning(f"Tavily search failed for '{query}': {e}. Falling back to DuckDuckGo.")
        try:
            ddgs = DDGS()
            results = ddgs.text(query, max_results=config.tavily_max_results)
            formatted_results = []
            for r in results:
                formatted_results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "content": r.get("body", "")
                })
            return formatted_results
        except Exception as fallback_e:
            logger.error(f"DuckDuckGo fallback also failed for '{query}': {fallback_e}")
            return []

async def _async_scrape_spa(url: str, timeout: int = 15000) -> str:
    """Async Playwright scraping logic."""
    if not async_playwright:
        return ""

    # Use pinned Chromium binary that is confirmed to be installed
    _CHROMIUM_EXEC = "/home/dracero/.cache/ms-playwright/chromium-1179/chrome-linux/chrome"

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            executable_path=_CHROMIUM_EXEC if os.path.exists(_CHROMIUM_EXEC) else None,
        )
        try:
            page = await browser.new_page()
            # Basic anti-bot measures
            await page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept-Language": "es-AR,es;q=0.9,en-US;q=0.8,en;q=0.7",
            })
            await page.goto(url, wait_until="networkidle", timeout=timeout)
            
            # Additional wait for SPA frameworks to render async fetches
            await page.wait_for_timeout(3000)
            
            html = await page.content()
            
            # Clean HTML to extract just readable text
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator=' ', strip=True)
            
            # Trim to prevent massive context inputs
            return text[:5000]
        except Exception as e:
            logger.warning(f"Playwright err on {url}: {e}")
            return ""
        finally:
            await browser.close()

def _scrape_spa_playwright(url: str) -> str:
    """Synchronous wrapper for Playwright SPA scraping."""
    try:
        # Create a new event loop just for this synchronous thread scope
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_async_scrape_spa(url))
    except Exception as e:
        logger.warning(f"Sync Playwright wrapper failed for {url}: {e}")
        return ""
    finally:
        try:
            loop.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────
# Node: Clarify (conversational pre-search)
# ─────────────────────────────────────────────────────

def clarify(state: TravelState) -> dict:
    """
    Generate a clarifying question for the user before launching the deep search.
    Returns the question in `clarify_question`.
    """
    config = Configuration.from_env()

    # Detect if we already have a clarifying answer from the user
    # (i.e. there's already a clarify_question in state from a previous run)
    if state.get("clarify_question"):
        logger.info("💬 Clarification already done, skipping.")
        return {}

    travel_dates = state.get("travel_dates", "no especificadas")

    # If dates are very vague, ask for specifics. Otherwise ask about budget/comfort.
    prompt = CLARIFY_SYSTEM_PROMPT.format(
        user_message=state.get("user_message_raw", ""),
        origin=state.get("origin", "el origen"),
        destination=state.get("destination", "el destino"),
        travel_dates=travel_dates,
    )

    try:
        llm_plain = _get_llm(config.clarify_model, temperature=0.5)  # Fast model for simple Q
        response = _invoke_with_backoff(
            llm_plain,
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Hacé la pregunta."},
            ],
        )
        question = response.content.strip()
        logger.info(f"💬 Clarifying question: {question}")
        return {"clarify_question": question}
    except Exception as e:
        logger.warning(f"Clarify failed: {e}")
        return {"clarify_question": ""}


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
    """Search with Tavily and extract options for a specific segment."""
    config = Configuration.from_env()
    llm = _get_llm(config.researcher_model)
    segment = state["segment"]
    travel_dates = state["travel_dates"]
    
    origin = segment.get("from_city")
    destination = segment.get("to_city")
    mode_hint = segment.get("mode_hint")

    # 1. Execute Tavily search
    query = f"viaje más barato {mode_hint} {origin} a {destination} {travel_dates}"

    results = _tavily_search(query, config)
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
        
        # Override content via Playwright if it's an SPA domain
        is_spa = any((d in url.lower() for d in spa_domains))
        if is_spa and async_playwright:
            logger.info(f"🌐 Scraping SPA {url} with Playwright...")
            scraped_content = _scrape_spa_playwright(url)
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

    routes_json = json.dumps(ranked, ensure_ascii=False, indent=2)

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

    # Add edges — clarify node temporarily disconnected (causes loop without persistent state)
    graph.add_edge(START, "plan_search")   # Direct to plan_search
    graph.add_conditional_edges("plan_search", route_to_segments, ["search_segment"])
    graph.add_edge("search_segment", "aggregate_routes")
    graph.add_edge("aggregate_routes", "rank_and_optimize")
    graph.add_edge("rank_and_optimize", "generate_itinerary")
    graph.add_edge("generate_itinerary", END)

    return graph.compile()
