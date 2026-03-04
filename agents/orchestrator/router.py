import json
import logging
import httpx
from pydantic import BaseModel
import os
from datetime import datetime

from beeai_framework.adapters.groq.backend.chat import GroqChatModel
from beeai_framework.backend.message import SystemMessage, UserMessage, AssistantMessage

from dotenv import load_dotenv
from pathlib import Path
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env")

logger = logging.getLogger(__name__)

# Configuración de URLs A2A
AGENT_URLS = {
    "eventos": "http://localhost:8001",
    "viajes": "http://localhost:8002",
}

class RoutingDecision(BaseModel):
    intent: str
    target_date: str = ""
    origin: str = ""
    destination: str = ""
    travel_dates: str = ""
    text_response: str = ""

def build_router_model() -> GroqChatModel:
    return GroqChatModel(
        model_id="llama-3.3-70b-versatile",  # Reverting because llama-4 is not yet accessible
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0,
    )

ROUTER_PROMPT = """\
Sos el BeeAI Orchestrator de la plataforma Deep Research.
Tu objetivo es analizar el mensaje del usuario y clasificar la intención principal (intent) para rutearlo al agente A2A correcto.

AGENTES Y REGLAS DE DECISIÓN:
1. "eventos" -> El usuario quiere INFORMARSE sobre qué eventos, torneos o streamings van a ocurrir. (Ej: "¿Qué eventos hay en junio?", "¿Qué partidos juega Argentina?", "¿Cuándo es el mundial?").
2. "viajes" -> El usuario pide EXPRESAMENTE cómo LLEGAR/VIAJAR a un destino físico. Si el mensaje central es sobre "ruta", "vuelo", "pasaje", "combinación", "precio de viaje", "ida y vuelta", O "cómo ir de X a Y", el intent **SIEMPRE ES VIAJES**. Si el usuario menciona un evento (ej. "el partido", "el recital", "el mundial") pero SOLO lo hace para dar contexto al viaje y conseguir una "ruta o vuelo" hacia allí, el intent ES SOLAMENTE VIAJES.
3. "ambos" -> El usuario hace explícitamente DOS pedidos distintos: quiere información/conocer qué eventos hay Y TAMBIÉN quiere opciones de vuelo para ir a verlos. (Ej: "Decime qué partidos hay en Miami y buscame vuelos para ir a verlos").
4. "conversacional" -> El usuario simplemente está saludando, despidiéndose, dando las gracias, o haciendo charla general ("hola", "cómo estás", "gracias", "quién sos"). 

Respondé SOLO con este JSON estricto (no uses markdown):
{
  "intent": "(eventos|viajes|ambos|conversacional)",
  "target_date": "YYYY-MM-DD (si aplica para eventos)",
  "origin": "Ciudad de origen (si aplica para viajes)",
  "destination": "Destino o Evento (si aplica para viajes)",
  "travel_dates": "Fechas estimadas (si aplica para viajes)",
  "text_response": "Si el intent es conversacional, escribí acá una respuesta amable y cortés para el usuario. Sino dejalo vacío."
}
"""

# ─── Regex patterns for zero-cost pre-routing ────────────────────────────────
import re

# Pattern: "de [city] a [city]" or "desde [city] hasta [city]" combined with
# a travel verb or noun anywhere in the message. This fires ONLY when an explicit
# origin+destination pair is present alongside a travel-intent phrase.
_TRAVEL_ORIGIN_DEST_RE = re.compile(
    r"\b(de|desde)\b.{1,50}\b(a|hasta|hacia|para)\b.{1,50}",
    re.IGNORECASE | re.DOTALL,
)
_TRAVEL_VERB_RE = re.compile(
    r"\b(viaj|volar|vuelo|pasaje|boleto|ticket|ruta|cómo ir|como ir|llegar a|"
    r"mejor opci[oó]n para ir|la forma más barata|quiero ir a|"
    r"trasladarme|trasladar|combinaci[oó]n de vuelos|opciones para viajar)\b",
    re.IGNORECASE,
)


def _quick_travel_detect(user_message: str) -> dict | None:
    """Return a viajes routing decision immediately when the message contains
    an explicit origin+destination pair AND a travel verb/noun.

    This pre-filter runs BEFORE calling the LLM to save tokens and avoid
    misclassification when the intent is unambiguous.
    Returns None if the message doesn't match, meaning the LLM should decide.
    """
    lower = user_message.lower()
    has_route = bool(_TRAVEL_ORIGIN_DEST_RE.search(lower))
    has_travel_word = bool(_TRAVEL_VERB_RE.search(lower))

    if has_route and has_travel_word:
        return {
            "intent": "viajes",
            "origin": "",       # Orchestrator server will still parse these from LLM context
            "destination": "",
            "travel_dates": "",
        }
    return None


async def determine_intent(user_message: str, history: list[dict] = None) -> dict:
    # ── Zero-cost pre-filter: detect unambiguous travel requests ────────────
    # Patterns like "viajar de X a Y" or "vuelo de X a Y" are always viajes.
    # This avoids calling the LLM (and wasting tokens) on clear-cut cases.
    quick_decision = _quick_travel_detect(user_message)
    if quick_decision:
        logger.info(f"[Router] Quick travel detection fired → {quick_decision}")
        return quick_decision

    llm = build_router_model()

    # Inject real date so the LLM resolves "hoy", "mañana", etc.
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    system_prompt = (
        ROUTER_PROMPT
        + f"\n\n[CONTEXTO DEL SISTEMA]\n"
        f"HOY es: {current_date}. Usá esta fecha para resolver referencias "
        f"temporales relativas (hoy, mañana, la semana que viene, etc.).\n"
        f"IMPORTANTE: Analizá el último mensaje usando el contexto histórico reciente si existe."
    )

    messages = [SystemMessage(system_prompt)]

    if history:
        for msg in history[:-1]:
            if msg.get("role") == "user":
                messages.append(UserMessage(msg.get("content", "")))
            elif msg.get("role") == "assistant":
                messages.append(AssistantMessage(msg.get("content", "")))

    messages.append(UserMessage(user_message))

    try:
        # BeeAI GroqChatModel.run() — sin response_format para evitar
        # errores de tool-call validation del adapter de Groq
        result = await llm.run(messages)

        # Extraer el texto de la respuesta
        raw_text = result.result.text if hasattr(result, "result") else ""
        if not raw_text:
            # fallback: intentar otros formatos de salida
            if hasattr(result, "output") and result.output:
                raw_text = result.output[-1].text
            elif hasattr(result, "messages") and result.messages:
                raw_text = result.messages[-1].text

        logger.info(f"Router raw LLM response: {raw_text}")

        # Limpiar posibles fences de markdown (```json ... ```)
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        decision = json.loads(cleaned)

        # Validar que el intent sea uno de los esperados
        valid_intents = {"eventos", "viajes", "ambos", "conversacional"}
        if decision.get("intent") not in valid_intents:
            logger.warning(f"Invalid intent '{decision.get('intent')}', defaulting to 'eventos'")
            decision["intent"] = "eventos"

        return decision

    except json.JSONDecodeError as e:
        logger.error(f"Routing JSON parse error: {e} — raw: {raw_text}", exc_info=True)
        # Try to infer intent from the raw text or the original user message
        return _infer_intent_from_text(raw_text or user_message)

    except Exception as e:
        logger.error(f"Routing error: {e}", exc_info=True)
        # Try to infer intent from the user message directly
        return _infer_intent_from_text(user_message)


def _infer_intent_from_text(text: str) -> dict:
    """Fallback intent classifier using simple keyword matching.

    Avoids the bug where any error would incorrectly default to 'eventos'.
    """
    lower = text.lower()

    _VIAJES_KEYWORDS = (
        "viaj", "vuelo", "pasaje", "boleto", "ticket", "bus", "avion", "avión",
        "ruta", "trayecto", "ida", "llegada", "salida", "cómo ir", "como ir",
        "cuánto cuesta ir", "precio de ir", "transporte", "aerolinea", "aerolínea",
        "jetsmart", "flybondi", "latam", "aerolíneas", "aerolineas",
    )
    _EVENTOS_KEYWORDS = (
        "evento", "partido", "recital", "mundial", "torneo", "concierto",
        "estreno", "show", "competencia", "liga", "copa",
    )

    has_travel = any(kw in lower for kw in _VIAJES_KEYWORDS)
    has_event = any(kw in lower for kw in _EVENTOS_KEYWORDS)

    if has_travel and has_event:
        # Both present → route to viajes since "events" is given as context for the trip
        return {"intent": "viajes", "origin": "", "destination": "", "travel_dates": ""}
    elif has_travel:
        return {"intent": "viajes", "origin": "", "destination": "", "travel_dates": ""}
    elif has_event:
        return {"intent": "eventos", "target_date": ""}

    # True default if no keywords match at all
    return {"intent": "eventos", "target_date": ""}
