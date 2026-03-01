"""
State definitions for the Travel Agent.
Same pattern as the Events agent — Pydantic structured outputs + TypedDict graph states.
"""

import operator
from typing import Annotated, Optional, List

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ─────────────────────────────────────
# Structured Outputs (LLM responses)
# ─────────────────────────────────────

class FlightQuery(BaseModel):
    """Una query de búsqueda para Tavily orientada a vuelos."""
    query: str = Field(description="Search query optimizada para encontrar vuelos/precios.")
    search_type: str = Field(description="Tipo de búsqueda: directo | escala | low_cost | alternativo")


class Segment(BaseModel):
    from_city: str = Field(description="Ciudad de origen del segmento")
    to_city: str = Field(description="Ciudad de destino del segmento")
    mode_hint: str = Field(description="Modo sugerido (ej: avion, bus, tren)")

class RouteCandidate(BaseModel):
    """Una ruta candidata compuesta por segmentos."""
    segments: List[Segment] = Field(description="Lista de segmentos en orden para esta ruta")
    description: str = Field(default="", description="Descripción breve de la estrategia (ej: 'Bus Mendoza+SCL+Vuelo low cost')")

class RoutePlan(BaseModel):
    """Plan de búsqueda generado por el planificador, dividido en rutas candidatas."""
    routes: List[RouteCandidate] = Field(
        description="Lista de rutas candidatas, donde cada ruta tiene sus segmentos separados."
    )
    strategy_notes: str = Field(
        default="",
        description="Notas del planificador sobre la estrategia de rutas alternativas."
    )


class RouteOption(BaseModel):
    """Una opción de ruta encontrada."""
    ruta: str = Field(description="Descripción de la ruta, ej: 'BUE → MIA → NYC'")
    transporte: str = Field(description="Aerolínea, empresa de micro/bus, tren o combinación involucrada")
    precio_usd: str = Field(description="Precio estimado en USD")
    duracion_total: str = Field(description="Duración total del viaje incluyendo escalas")
    escalas: str = Field(description="Cantidad y ciudades de escala, o 'Directo'")
    fecha_ida: str = Field(description="Fecha de salida (YYYY-MM-DD) o rango")
    fecha_vuelta: str = Field(description="Fecha de regreso (YYYY-MM-DD) o rango, o 'Solo ida'")
    tipo: str = Field(description="directo | escala | terrestre | alternativo")
    notas: str = Field(default="", description="Notas adicionales (equipaje, restricciones, etc.)")
    fuente: str = Field(description="URL de la fuente de información")


class RouteFindings(BaseModel):
    """Resultados de un researcher de vuelos."""
    routes: List[RouteOption] = Field(default_factory=list, description="Rutas encontradas.")
    notes: str = Field(default="", description="Notas sobre la búsqueda.")


class RankedRoutes(BaseModel):
    """Rutas ranqueadas y optimizadas."""
    routes: List[RouteOption] = Field(
        default_factory=list,
        description="Rutas ordenadas de más barata a más cara."
    )
    recommendation: str = Field(
        default="",
        description="Recomendación del agente sobre la mejor opción."
    )


# ─────────────────────────────────────
# Graph State Definitions
# ─────────────────────────────────────

def _list_reducer(current: list, new: list | dict) -> list:
    """Append-style reducer for lists, with override support."""
    if isinstance(new, dict) and new.get("type") == "override":
        return new.get("value", [])
    return current + new


class TravelState(TypedDict):
    """Main graph state for travel search."""
    origin: str
    destination: str
    travel_dates: str
    flexibility_days: int
    budget_max_usd: Optional[float]
    user_message_raw: str                                     # mensaje original del usuario
    clarify_question: str                                     # pregunta de clarificación generada
    route_plan: Optional[List[dict]]                          # lista de rutas candidatas (lista de segmentos)
    segment_results: Annotated[List[dict], _list_reducer]     # resultados paralelos por segmento
    normalized_prices: Optional[List[dict]]
    final_ranking: Optional[List[dict]]
    final_itinerary: Optional[str]


class SegmentSearcherState(TypedDict):
    """State for an individual segment searcher invocation."""
    segment: dict
    travel_dates: str
    segment_results: Annotated[List[dict], _list_reducer]
