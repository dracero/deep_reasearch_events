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
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from tavily import TavilyClient

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
# Helpers
# ─────────────────────────────────────────────────────

def _get_llm(model_name: str, temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """Create an LLM instance."""
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)


def _tavily_search(query: str, config: Configuration) -> List[dict]:
    """Execute a single Tavily search and return results."""
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
        logger.warning(f"Tavily search failed for '{query}': {e}")
        return []


# ─────────────────────────────────────────────────────
# Node: Plan Research
# ─────────────────────────────────────────────────────

def plan_research(state: AgentState) -> dict:
    """Generate search queries for each event category."""
    config = Configuration.from_env()
    llm = _get_llm(config.planner_model)

    prompt = PLANNER_SYSTEM_PROMPT.format(max_queries=config.max_queries_per_category)
    user_msg = (
        f"Generá queries de búsqueda para encontrar eventos que generen tráfico de internet "
        f"en Argentina para la fecha: {state['target_date']}.\n\n"
        f"Hoy es {state['target_date']}. Buscá eventos de esa fecha y los días cercanos."
    )

    structured_llm = llm.with_structured_output(SearchPlan)
    plan: SearchPlan = structured_llm.invoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_msg},
    ])

    logger.info(f"📋 Plan generated: {len(plan.queries)} queries")
    return {"search_plan": plan}


# ─────────────────────────────────────────────────────
# Node: Research Category (runs in parallel via Send)
# ─────────────────────────────────────────────────────

def research_category(state: ResearcherState) -> dict:
    """Search with Tavily and extract events for a category."""
    config = Configuration.from_env()
    llm = _get_llm(config.researcher_model)
    category = state["category"]
    target_date = state["target_date"]
    queries = state["queries"]

    # 1. Execute all Tavily searches for this category
    all_results = []
    for query in queries:
        results = _tavily_search(query, config)
        all_results.extend(results)
        logger.info(f"🔍 [{category}] Query '{query}': {len(results)} results")

    if not all_results:
        logger.info(f"⚠️  [{category}] No search results found")
        return {"raw_events": []}

    # 2. Format search results for the LLM
    search_context = "\n\n".join([
        f"**Fuente**: {r.get('url', 'N/A')}\n"
        f"**Título**: {r.get('title', 'N/A')}\n"
        f"**Contenido**: {r.get('content', 'N/A')}"
        for r in all_results
    ])

    # 3. Ask LLM to extract structured events
    prompt = RESEARCHER_SYSTEM_PROMPT.format(category=category, target_date=target_date)
    structured_llm = llm.with_structured_output(ResearchFindings)

    try:
        findings: ResearchFindings = structured_llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Resultados de búsqueda:\n\n{search_context}"},
        ])
        events = [event.model_dump() for event in findings.events]
        logger.info(f"✅ [{category}] Extracted {len(events)} events")
        return {"raw_events": events}
    except Exception as e:
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
# Node: Filter for Argentina Relevance
# ─────────────────────────────────────────────────────

def filter_argentina(state: AgentState) -> dict:
    """Use LLM to filter events by Argentina internet traffic impact."""
    config = Configuration.from_env()
    raw = state.get("raw_events", [])

    if not raw:
        return {"filtered_events": []}

    llm = _get_llm(config.filter_model)
    structured_llm = llm.with_structured_output(FilteredReport)

    events_json = json.dumps(raw, ensure_ascii=False, indent=2)

    try:
        report: FilteredReport = structured_llm.invoke([
            {"role": "system", "content": FILTER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Eventos a filtrar:\n\n{events_json}"},
        ])
        filtered = [e.model_dump() for e in report.events]
        logger.info(f"🇦🇷 Filtered to {len(filtered)} Argentina-relevant events")
        return {"filtered_events": filtered}
    except Exception as e:
        logger.warning(f"❌ Filter failed: {e}")
        return {"filtered_events": raw}


# ─────────────────────────────────────────────────────
# Node: Generate Final Report
# ─────────────────────────────────────────────────────

def generate_report(state: AgentState) -> dict:
    """Generate the final JSON report for pandas conversion."""
    filtered = state.get("filtered_events", [])

    if not filtered:
        logger.info("📄 No events to report")
        return {"final_report": "[]"}

    config = Configuration.from_env()
    llm = _get_llm(config.report_model)

    events_json = json.dumps(filtered, ensure_ascii=False, indent=2)

    response = llm.invoke([
        {"role": "system", "content": REPORT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Eventos filtrados:\n\n{events_json}"},
    ])

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
    graph.add_node("filter_argentina", filter_argentina)
    graph.add_node("generate_report", generate_report)

    # Add edges
    graph.add_edge(START, "plan_research")
    graph.add_conditional_edges("plan_research", route_to_researchers, ["research_category"])
    graph.add_edge("research_category", "aggregate_results")
    graph.add_edge("aggregate_results", "filter_argentina")
    graph.add_edge("filter_argentina", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()
