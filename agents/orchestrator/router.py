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
    "explainer": "http://localhost:8002",
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

AGENT Y REGLAS DE DECISIÓN:
1. "eventos" -> El usuario quiere INFORMARSE sobre qué eventos, torneos o streamings van a ocurrir. (Ej: "¿Qué eventos hay en junio?", "¿Qué partidos juega Argentina?", "¿Cuándo es el mundial?").
2. "explainer" -> El usuario pide EXPLICAR un contenido a partir de un link o URL. Si el mensaje central es "explicame", "resumi", "leé este link", "de qué trata el link", el intent **SIEMPRE ES EXPLAINER**.
3. "ambos" -> El usuario hace explícitamente DOS pedidos distintos: quiere información/conocer qué eventos hay Y TAMBIÉN quiere que le expliques un link.
4. "conversacional" -> El usuario simplemente está saludando, despidiéndose, dando las gracias, o haciendo charla general. 

REGLA CRÍTICA SOBRE URLs:
- SOLO extraé una URL si el usuario la escribió TEXTUALMENTE en su mensaje (debe contener "http://", "https://", o "www.").
- Si el usuario NO escribió ningún link, dejá "url" como cadena VACÍA ("").
- NUNCA inventes, adivines ni construyas URLs a partir del tema mencionado por el usuario.

Respondé SOLO con este JSON estricto (no uses markdown):
{
  "intent": "(eventos|explainer|ambos|conversacional)",
  "target_date": "YYYY-MM-DD (si aplica para eventos)",
  "url": "URL TEXTUAL del mensaje del usuario. VACÍO si no escribió ningún link.",
  "topic": "Tema a explicar (si aplica para explainer)",
  "text_response": "Si el intent es conversacional, escribí acá una respuesta amable y cortés para el usuario."
}
"""

# ─── Regex patterns for zero-cost pre-routing ────────────────────────────────
import re

_EXPLAINER_RE = re.compile(
    r"\b(explic|resumi|lee|leyendo|link|url|p[aá]gina|sitio|web)\b",
    re.IGNORECASE,
)


def _quick_explainer_detect(user_message: str) -> dict | None:
    """Return an explainer routing decision immediately when the message contains
    explainer keywords like 'explicar' and 'link' or 'url'.
    """
    lower = user_message.lower()
    has_explainer_word = bool(_EXPLAINER_RE.search(lower))
    has_url_indicator = "http" in lower or "www" in lower or "link" in lower or "url" in lower

    if has_explainer_word and has_url_indicator:
        return {
            "intent": "explainer",
            "url": "",
            "topic": "",
        }
    return None


async def determine_intent(user_message: str, history: list[dict] = None) -> dict:
    # ── Zero-cost pre-filter: detect unambiguous explainer requests ────────────
    quick_decision = _quick_explainer_detect(user_message)
    if quick_decision:
        logger.info(f"[Router] Quick explainer detection fired → {quick_decision}")
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
        valid_intents = {"eventos", "explainer", "ambos", "conversacional"}
        if decision.get("intent") not in valid_intents:
            logger.warning(f"Invalid intent '{decision.get('intent')}', defaulting to 'explainer'")
            decision["intent"] = "explainer"

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

    _EXPLAINER_KEYWORDS = (
        "explic", "resum", "url", "link", "web", "página", "sitio", "pasa",
    )
    _EVENTOS_KEYWORDS = (
        "evento", "partido", "recital", "mundial", "torneo", "concierto",
        "estreno", "show", "competencia", "liga", "copa",
    )

    has_explainer = any(kw in lower for kw in _EXPLAINER_KEYWORDS)
    has_event = any(kw in lower for kw in _EVENTOS_KEYWORDS)

    if has_explainer and has_event:
        return {"intent": "explainer", "url": "", "topic": ""}
    elif has_explainer:
        return {"intent": "explainer", "url": "", "topic": ""}
    elif has_event:
        return {"intent": "eventos", "target_date": ""}

    # True default if no keywords match at all. Default to explainer for testing the explainer.
    return {"intent": "explainer", "url": "", "topic": ""}
