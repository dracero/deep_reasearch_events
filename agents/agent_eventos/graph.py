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
  research_category ──► Tavily search + LLM extraction (×4 categories)
       │
       ▼
  aggregate_results ──► Combine all events
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
from typing import List

import os
import re
import yaml
from groq import RateLimitError
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langsmith import traceable
from tavily import TavilyClient
import httpx
from bs4 import BeautifulSoup
try:
    from playwright.async_api import async_playwright
except ImportError:
    async_playwright = None

from state import (
    AgentState,
    ResearcherState,
    SearchPlan,
    ResearchFindings,
    FilteredReport,
    EventInfo,
)
from configuration import Configuration
from prompts import (
    PLANNER_SYSTEM_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    FILTER_SYSTEM_PROMPT,
    REPORT_SYSTEM_PROMPT,
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
        max_retries=0,  # We handle retries manually with backoff
    )


async def _invoke_with_backoff(chain, messages: list, max_attempts: int = 5):
    """Invoke a LangChain chain with exponential backoff on 429 rate limit errors.
    Uses asyncio.sleep to avoid blocking the event loop during waits.
    """
    delay = 5.0
    for attempt in range(max_attempts):
        try:
            # Run the synchronous chain.invoke in a thread so the event loop stays free
            return await asyncio.get_event_loop().run_in_executor(None, chain.invoke, messages)
        except RateLimitError:
            if attempt == max_attempts - 1:
                raise
            logger.warning(
                f"Rate limit hit (attempt {attempt + 1}/{max_attempts}). "
                f"Waiting {delay:.1f}s before retry..."
            )
            await asyncio.sleep(delay)  # Non-blocking wait
            delay = min(delay * 2, 60.0)  # Cap at 60s
        except Exception:
            raise


def _parse_failed_generation(error, pydantic_model):
    """Extract and parse the 'failed_generation' JSON from a Groq 400 tool_use_failed error.

    Groq sometimes returns valid JSON but fails to invoke the function/tool properly,
    resulting in a 400 Bad Request with the actual data inside 'failed_generation'.
    This helper rescues that data and constructs the Pydantic model manually.

    Returns the parsed Pydantic model instance, or None if parsing fails.
    """
    error_str = str(error)
    if 'tool_use_failed' not in error_str and 'failed_generation' not in error_str:
        return None

    # Try to extract the failed_generation JSON from the error body
    failed_json_str = None

    # Method 1: If the error has a .body dict (groq / openai SDK errors)
    if hasattr(error, 'body') and isinstance(error.body, dict):
        err_detail = error.body.get('error', {})
        failed_json_str = err_detail.get('failed_generation')

    # Method 2: Regex extraction from the string representation
    if not failed_json_str:
        match = re.search(r"'failed_generation':\s*'(.*)'\}\}$", error_str, re.DOTALL)
        if match:
            failed_json_str = match.group(1)

    if not failed_json_str:
        logger.warning("Could not extract failed_generation from error.")
        return None

    # Clean markdown fences if present
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

    # The data might be in several formats Groq produces:
    #   1. A flat list of event dicts  →  wrap as {"events": data}
    #   2. A list containing one object with "events" key  →  unwrap
    #   3. A dict with "events" key directly  →  use as-is
    #   4. Tool-call wrapper: [{"name": "...", "parameters": {"events": [...]}}]  →  extract parameters
    try:
        if isinstance(data, list):
            # Format 4: Tool-call wrapper [{"name": "...", "parameters": {...}}]
            if len(data) > 0 and isinstance(data[0], dict) and 'name' in data[0] and 'parameters' in data[0]:
                # Extract from tool-call envelope
                params = data[0].get('parameters', {})
                if isinstance(params, dict) and 'events' in params:
                    data = params
                else:
                    data = {'events': []}
                logger.info(f"Extracted data from tool-call wrapper format")
            # Format 2: [{\"events\": [...], \"notes\": \"...\"}]
            elif len(data) > 0 and isinstance(data[0], dict) and 'events' in data[0]:
                # Merge all events from wrapper objects
                all_events = []
                notes = ""
                for item in data:
                    all_events.extend(item.get('events', []))
                    if item.get('notes'):
                        notes = item['notes']
                data = {'events': all_events, 'notes': notes}
            else:
                # Format 1: flat list of event dicts
                data = {'events': data}

        result = pydantic_model.model_validate(data)
        logger.info(f"✅ Recovered {len(result.events)} events from failed_generation fallback")
        return result
    except Exception as parse_err:
        logger.warning(f"Failed to validate failed_generation into {pydantic_model.__name__}: {parse_err}")
        return None


@traceable(name="tavily_search", run_type="tool")
def _tavily_search(query: str, config: Configuration, include_domains: list[str] | None = None) -> List[dict]:
    """Execute a single Tavily search and return results."""
    client = TavilyClient()
    try:
        kwargs = dict(
            query=query,
            max_results=config.tavily_max_results,
            search_depth=config.tavily_search_depth,
            include_answer=True,
        )
        if include_domains:
            kwargs["include_domains"] = include_domains
        response = client.search(**kwargs)
        return response.get("results", [])
    except Exception as e:
        logger.warning(f"Tavily search failed for '{query}': {e}")
        return []


# Dominios SPA que requieren JS rendering para contenido actualizado
_EVENTOS_SPA_DOMAINS = [
    "netflix.com", "disneyplus.com", "hbomax.com", "max.com",
    "primevideo.com", "twitch.tv", "store.steampowered.com",
    "paramountplus.com", "star-plus.com", "apple.com/tv",
]

async def _async_scrape_spa_evento(url: str, timeout: int = 15000) -> str:
    """Playwright async scraper for SPA event sites (Netflix, Disney+, Twitch, etc.)"""
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
            await page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept-Language": "es-AR,es;q=0.9,en-US;q=0.8",
            })
            await page.goto(url, wait_until="networkidle", timeout=timeout)
            await page.wait_for_timeout(3000)  # Wait for SPAs to finish async data fetch
            html = await page.content()

            soup = BeautifulSoup(html, "html.parser")
            # Remove noise tags
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe", "svg"]):
                tag.decompose()

            # For streaming platforms, look for content-rich sections
            content_selectors = [
                "[class*='title']", "[class*='release']", "[class*='date']",
                "[class*='card']", "[class*='movie']", "[class*='series']",
                "[class*='show']", "[class*='game']", "[class*='stream']",
            ]
            extracted_texts = []
            for sel in content_selectors:
                try:
                    elements = soup.select(sel)
                    for el in elements[:5]:  # Max 5 per selector
                        text = el.get_text(separator=' ', strip=True)
                        if len(text) > 10:
                            extracted_texts.append(text)
                except Exception:
                    pass

            if extracted_texts:
                combined = ' | '.join(extracted_texts)
            else:
                combined = soup.get_text(separator=' ', strip=True)

            return combined[:4000]
        except Exception as e:
            logger.warning(f"Playwright evento scrape error on {url}: {e}")
            return ""
        finally:
            await browser.close()


async def _scrape_spa_evento(url: str) -> str:
    """Wrapper that invokes Playwright SPA scraper."""
    try:
        return await _async_scrape_spa_evento(url)
    except Exception as e:
        logger.warning(f"SPA scrape failed for {url}: {e}")
        return ""


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

    prompt = PLANNER_SYSTEM_PROMPT.format(max_queries=config.max_queries_per_category)
    user_msg = (
        f"La fecha de HOY (real, actual) es: {today_arg}.\n"
        f"El usuario quiere encontrar eventos para la FECHA OBJETIVO: {target}.\n\n"
        f"Generá queries de búsqueda para encontrar eventos que generen tráfico de internet "
        f"en Argentina para la fecha objetivo {target}.\n\n"
        f"IMPORTANTE: Solo buscá eventos que realmente ocurran en la fecha {target}. "
        f"NO confundas la fecha de hoy ({today_arg}) con la fecha objetivo ({target}).\n"
        f"Para streaming, generá queries como: 'estrenos Netflix {target}', "
        f"'nuevos en Disney+ {target}', 'nuevas series HBO {target}'.\n"
        f"Para deportes, generá queries como: 'partidos fútbol argentino {target}', "
        f"'fixture Liga Profesional {target}'."
    )

    structured_llm = llm.with_structured_output(SearchPlan)
    plan: SearchPlan = await _invoke_with_backoff(
        structured_llm,
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_msg},
        ],
    )

    logger.info(f"📋 Plan generated: {len(plan.queries)} queries")
    return {"search_plan": plan}


# ─────────────────────────────────────────────────────
# Node: Research Category (runs in parallel via Send)
# ─────────────────────────────────────────────────────

async def research_category(state: ResearcherState) -> dict:
    """Search with Tavily and extract events for a category."""
    config = Configuration.from_env()
    llm = _get_llm(config.researcher_model)
    category = state["category"]
    target_date = state["target_date"]
    queries = state["queries"]

    # Define domains per category for Tavily's include_domains filter.
    # NOTE: For streaming, we use entertainment NEWS sites (not the platforms
    # themselves) because netflix.com/disneyplus.com are apps, not article sites.
    _CATEGORY_DOMAINS = {
        "streaming": [
            "sensacine.com.ar", "justwatch.com", "filmaffinity.com",
            "espinof.com", "infobae.com", "lanacion.com.ar",
            "todomovieseries.com", "cinepremiere.com.mx",
            "screenrant.com", "whats-on-netflix.com",
        ],
        "deportes": [
            "espn.com.ar", "tycsports.com", "ole.com.ar",
            "fifa.com", "nba.com", "afa.com.ar", "conmebol.com",
            "promiedos.com.ar", "infobae.com", "clarin.com",
        ],
        "gaming": [
            "store.steampowered.com", "ign.com", "twitch.tv",
            "gamespot.com", "kotaku.com",
        ],
    }
    domains = _CATEGORY_DOMAINS.get(category)

    # 1. Execute all Tavily searches for this category (run in executor to not block)
    all_results = []
    loop = asyncio.get_event_loop()
    for query in queries:
        results = await loop.run_in_executor(None, _tavily_search, query, config, domains)
        all_results.extend(results)
        logger.info(f"🔍 [{category}] Query '{query}': {len(results)} results")

    # 1b. For streaming, always add hardcoded queries to ensure Netflix/Disney+/HBO coverage
    if category == "streaming":
        _extra_streaming_queries = [
            f"estrenos Netflix {target_date} Argentina",
            f"nuevas series películas Disney+ {target_date}",
            f"estrenos HBO Max {target_date} Argentina",
        ]
        for eq in _extra_streaming_queries:
            if eq not in queries:  # avoid duplicates
                extra_results = await loop.run_in_executor(None, _tavily_search, eq, config, None)
                all_results.extend(extra_results)
                logger.info(f"🔍 [{category}] Extra query '{eq}': {len(extra_results)} results")

    # 1c. For deportes, guarantee we fetch the day's agenda/fixture
    if category == "deportes":
        _extra_deportes_queries = [
            f"agenda deportiva hoy {target_date} futbol argentino",
            f"fixture partidos primera division argentina {target_date}",
            f"programacion horarios futbol copa {target_date}",
        ]
        for eq in _extra_deportes_queries:
            if eq not in queries:
                # Intentionally passing None for domains so Tavily can hit any sports site
                extra_results = await loop.run_in_executor(None, _tavily_search, eq, config, None)
                all_results.extend(extra_results)
                logger.info(f"🔍 [{category}] Extra query '{eq}': {len(extra_results)} results")

    if not all_results:
        logger.info(f"⚠️  [{category}] No search results found")
        return {"raw_events": []}

    # 2. Enrich SPA results with Playwright for accurate dates/providers
    enriched_results = []
    for r in all_results:
        url = r.get("url", "")
        is_spa = async_playwright and any(d in url.lower() for d in _EVENTOS_SPA_DOMAINS)
        if is_spa:
            logger.info(f"🌐 [{category}] Playwright scraping: {url}")
            scraped = await _scrape_spa_evento(url)
            if scraped:
                r = {**r, "content": scraped}  # Replace static snippet with live content
        enriched_results.append(r)

    # 3. Format search results for the LLM (crucial to truncate string length to save rate limits)
    search_context = "\n\n".join([
        f"**Fuente**: {r.get('url', 'N/A')}\n"
        f"**Título**: {r.get('title', 'N/A')}\n"
        f"**Contenido**: {str(r.get('content', 'N/A'))[:450]}..."  # TRUNCATE to avoid 429 Token Rate Limit exhaustion
        for r in enriched_results
    ])

    # 3. Ask LLM to extract structured events
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
        )
        events = [event.model_dump() for event in findings.events]
        logger.info(f"✅ [{category}] Extracted {len(events)} events")
        return {"raw_events": events}
    except Exception as e:
        # Fallback: try to rescue data from Groq's failed_generation
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

def route_to_researchers(state: AgentState) -> list[Send]:
    """Create a Send for each category to run researchers in parallel."""
    plan = state["search_plan"]
    if not plan:
        return []

    # Group queries by category
    categories: dict[str, list[str]] = {}
    for sq in plan.queries:
        categories.setdefault(sq.category, []).append(sq.query)

    sends = []
    for category, queries in categories.items():
        sends.append(
            Send(
                "research_category",
                {
                    "target_date": state["target_date"],
                    "category": category,
                    "queries": queries,
                    "raw_events": [],
                },
            )
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

    # Simple deduplication by event name (lowercase)
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

# Date patterns to search for in scraped pages (DD/MM/YYYY, DD de Mes de YYYY, YYYY-MM-DD, etc.)
_MONTH_NAMES = {
    'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04',
    'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08',
    'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12',
    'january': '01', 'february': '02', 'march': '03', 'april': '04',
    'may': '05', 'june': '06', 'july': '07', 'august': '08',
    'september': '09', 'october': '10', 'november': '11', 'december': '12',
}


def _extract_dates_from_text(text: str) -> list[str]:
    """Extract all dates from text and return as YYYY-MM-DD strings."""
    dates_found = []

    # Pattern 1: YYYY-MM-DD (ISO format)
    for m in re.finditer(r'(\d{4})-(\d{2})-(\d{2})', text):
        dates_found.append(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")

    # Pattern 2: DD/MM/YYYY or DD-MM-YYYY
    for m in re.finditer(r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})', text):
        day, month, year = m.group(1).zfill(2), m.group(2).zfill(2), m.group(3)
        if 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
            dates_found.append(f"{year}-{month}-{day}")

    # Pattern 3: "14 de febrero de 2026" / "February 14, 2026"
    text_lower = text.lower()
    for month_name, month_num in _MONTH_NAMES.items():
        # Spanish: "14 de febrero de 2026" or "14 de febrero, 2026"
        for m in re.finditer(
            rf'(\d{{1,2}})\s+de\s+{month_name}\s+(?:de\s+|,?\s*)(\d{{4}})',
            text_lower
        ):
            day = m.group(1).zfill(2)
            year = m.group(2)
            dates_found.append(f"{year}-{month_num}-{day}")

        # English: "February 14, 2026"
        for m in re.finditer(
            rf'{month_name}\s+(\d{{1,2}}),?\s+(\d{{4}})',
            text_lower
        ):
            day = m.group(1).zfill(2)
            year = m.group(2)
            dates_found.append(f"{year}-{month_num}-{day}")

    return list(set(dates_found))


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
            # Remove script and style elements
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()
            return soup.get_text(separator=' ', strip=True)[:2500]  # Cap at 2.5k chars to save tokens (was 5k)
    except Exception as e:
        logger.debug(f"Could not scrape {url}: {e}")
        return ""


async def verify_dates(state: AgentState) -> dict:
    """Scrape source URLs to verify and correct event dates.
    
    The LLM sometimes assigns incorrect dates to events (e.g. Bridgerton on Feb 14
    when the source says Feb 20). This node fetches the source pages and extracts
    real dates to correct any mismatches.
    """
    raw = state.get("raw_events", [])
    target_date = state.get("target_date", "")

    if not raw:
        return {"raw_events": {"type": "override", "value": []}}

    logger.info(f"🔎 Verifying dates for {len(raw)} events via scraping...")

    verified = []
    # Scrape all URLs concurrently (with a limit)
    scrape_tasks = []
    async def _empty_scrape():
        return ""

    for event in raw:
        url = event.get("fuente", "")
        if url and url.startswith("http"):
            scrape_tasks.append((event, _scrape_url(url)))
        else:
            scrape_tasks.append((event, _empty_scrape()))

    # Gather all scraping results
    events_and_coros = [(ev, coro) for ev, coro in scrape_tasks]
    scraped_texts = await asyncio.gather(*[coro for _, coro in events_and_coros])

    for (event, _), page_text in zip(events_and_coros, scraped_texts):
        event_name = event.get("evento", "")
        original_date = event.get("fecha", "")

        if not page_text:
            # Couldn't scrape — keep original
            verified.append(event)
            continue

        # Extract all dates from the scraped page
        page_dates = _extract_dates_from_text(page_text)

        if not page_dates:
            # No dates found on page — keep original
            verified.append(event)
            continue

        # If target_date is explicitly mentioned on the page, assume the LLM found it 
        # but messed up the extraction date. We keep it and correct to target_date.
        if target_date in page_dates:
            if original_date != target_date:
                logger.info(
                    f"📅 [{event_name}] Date corrected: {original_date} → {target_date} "
                    f"(target_date found on source page)"
                )
                event["fecha"] = target_date
            verified.append(event)
            continue

        # If we got here, the target_date is NOT anywhere on the page, but OTHER dates are.
        # This usually means the LLM grabbed a monthly release list and pulled the wrong event.
        logger.warning(
            f"🗑️ [{event_name}] Discarded: Target date {target_date} not found on source page. "
            f"Page dates found: {page_dates}"
        )
        # Event is purposefully NOT appended to `verified` list

    logger.info(f"🔎 Date verification complete: kept {len(verified)} verified events out of {len(raw)}")
    return {"raw_events": {"type": "override", "value": verified}}


# ─────────────────────────────────────────────────────
# Node: Filter for Argentina Relevance
# ─────────────────────────────────────────────────────

async def filter_argentina(state: AgentState) -> dict:
    """Use LLM to filter ONLY deportes/especiales events by Argentina relevance.
    
    Streaming and gaming events are auto-passed since they were already
    validated by the researcher node and the LLM consistently drops them incorrectly.
    """
    config = Configuration.from_env()
    raw = state.get("raw_events", [])

    if not raw:
        return {"filtered_events": []}

    # Split events: auto-pass streaming/gaming, only filter deportes/especiales with LLM
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

    # If nothing needs LLM filtering, just return all
    if not needs_filtering:
        logger.info(f"🇦🇷 All {len(auto_pass)} events auto-passed (no deportes/especiales to filter)")
        return {"filtered_events": auto_pass}

    # Filter only deportes/especiales through the LLM
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
        )
        llm_filtered = [e.model_dump() for e in report.events]
        logger.info(f"🇦🇷 LLM filtered deportes/especiales to {len(llm_filtered)} events")
    except Exception as e:
        # Fallback: try to rescue data from Groq's failed_generation
        fallback = _parse_failed_generation(e, FilteredReport)
        if fallback and fallback.events:
            llm_filtered = [ev.model_dump() for ev in fallback.events]
            logger.info(f"🔄 Filter recovered {len(llm_filtered)} deportes/especiales via fallback")
        else:
            logger.warning(f"❌ Filter failed, passing deportes/especiales unfiltered: {e}")
            llm_filtered = needs_filtering

    # Combine: auto-passed streaming/gaming + LLM-filtered deportes/especiales
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

    # Clean markdown fences if present
    if report_text.startswith("```"):
        lines = report_text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        report_text = "\n".join(lines)

    logger.info(f"📄 Final report generated")
    return {"final_report": report_text}


# ─────────────────────────────────────────────────────
# Build the Graph
# ─────────────────────────────────────────────────────

def build_graph():
    """Construct and compile the LangGraph StateGraph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("plan_research", plan_research)
    graph.add_node("research_category", research_category)
    graph.add_node("aggregate_results", aggregate_results)
    graph.add_node("verify_dates", verify_dates)
    graph.add_node("filter_argentina", filter_argentina)
    graph.add_node("generate_report", generate_report)

    # Add edges
    graph.add_edge(START, "plan_research")
    graph.add_conditional_edges("plan_research", route_to_researchers, ["research_category"])
    graph.add_edge("research_category", "aggregate_results")
    graph.add_edge("aggregate_results", "verify_dates")
    graph.add_edge("verify_dates", "filter_argentina")
    graph.add_edge("filter_argentina", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()
