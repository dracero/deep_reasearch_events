"""
State definitions for the Content Explainer Agent (Web Page Q&A).
"""

from typing import Annotated, Optional, List

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ─────────────────────────────────────
# Structured Outputs (LLM responses)
# ─────────────────────────────────────

class ClarifyDecision(BaseModel):
    """Decisión estructurada sobre si falta la URL para analizar contenido web."""
    need_clarification: bool = Field(
         description="True si falta la URL que NO se puede extraer del mensaje"
    )
    question: str = Field(
         default="",
         description="Pregunta al usuario si need_clarification=True. Vacío si no hace falta preguntar."
    )
    missing_fields: List[str] = Field(
         default_factory=list,
         description="Lista de campos que realmente faltan después de analizar el mensaje: 'url'."
    )
    extracted_url: str = Field(
         default="",
         description="URL extraída del mensaje del usuario. Vacío si no se menciona."
    )
    extracted_question: str = Field(
         default="",
         description="Pregunta o solicitud extraída del mensaje del usuario (ej: 'buscá ejemplos de código', 'explicame qué es X'). Vacío si no se menciona."
    )

class ExplanationResult(BaseModel):
    """Resultado del agente explicador."""
    found: bool = Field(description="True si se encontró información relevante en el contenido de la URL, False en caso contrario.")
    explanation: str = Field(description="Respuesta basada en el contenido. Si found=False, indica que no se encontró.")


# ─────────────────────────────────────
# Graph State Definitions
# ─────────────────────────────────────

class ExplainerState(TypedDict):
    """Main graph state for web page Q&A."""
    url: str
    question: str
    user_message_raw: str
    clarify_question: str
    scraped_content: str
    final_explanation: Optional[ExplanationResult]
