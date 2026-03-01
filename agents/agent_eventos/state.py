"""
State definitions for the Deep Research Agent.
Inspired by langchain-ai/open_deep_research state.py.
"""

import operator
from typing import Annotated, Optional, List

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ─────────────────────────────────────
# Structured Outputs (LLM responses)
# ─────────────────────────────────────

class SearchQuery(BaseModel):
    """A single search query for Tavily."""
    query: str = Field(description="Search query string optimised for Tavily.")
    category: str = Field(description="Category: deportes | streaming | gaming | especiales")


class SearchPlan(BaseModel):
    """The planner output: a list of search queries grouped by category."""
    queries: List[SearchQuery] = Field(
        description="List of search queries to execute, covering deportes, streaming, gaming, and especiales."
    )


class EventInfo(BaseModel):
    """Structured info about a single event."""
    evento: str = Field(description="Nombre del evento")
    categoria: str = Field(description="Categoría: Deportes / Streaming / Gaming / Especiales")
    proveedor: str = Field(default="No especificado", description="Empresa/proveedor a cargo del evento (ej: Netflix, Disney+, Steam, AFA, ESPN, etc)")
    fecha: str = Field(description="Fecha del evento (YYYY-MM-DD)")
    hora_argentina: str = Field(default="A confirmar", description="Hora en Argentina (HH:MM ART), o 'A confirmar' si no se conoce")
    descripcion: str = Field(default="", description="Breve descripción del evento")
    impacto_estimado: str = Field(default="Medio", description="Impacto estimado en tráfico: Alto / Medio / Bajo")
    fuente: str = Field(default="", description="URL o fuente de la información")


class ResearchFindings(BaseModel):
    """Output from a single researcher: list of events found."""
    events: List[EventInfo] = Field(default_factory=list, description="Events found by this researcher.")
    notes: str = Field(default="", description="Free-form notes about the research.")


class FilteredReport(BaseModel):
    """Final filtered and validated events."""
    events: List[EventInfo] = Field(default_factory=list, description="Events that affect Argentina internet traffic.")


# ─────────────────────────────────────
# Graph State Definitions
# ─────────────────────────────────────

def _list_reducer(current: list, new: list | dict) -> list:
    """Append-style reducer for lists, with override support."""
    if isinstance(new, dict) and new.get("type") == "override":
        return new.get("value", [])
    return current + new


class AgentState(TypedDict):
    """Main graph state."""
    target_date: str                                         # fecha objetivo (YYYY-MM-DD)
    search_plan: Optional[SearchPlan]                        # plan de búsqueda generado
    raw_events: Annotated[List[dict], _list_reducer]         # eventos crudos de los researchers
    filtered_events: Optional[List[dict]]                    # eventos filtrados por relevancia
    final_report: Optional[str]                              # reporte JSON final


class ResearcherState(TypedDict):
    """State for an individual researcher subgraph invocation."""
    target_date: str
    category: str
    queries: List[str]
    raw_events: Annotated[List[dict], _list_reducer]
