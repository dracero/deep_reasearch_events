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
        model_id="llama-3.3-70b-versatile",  # Smart enough to understand nuanced intent
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

async def determine_intent(user_message: str, history: list[dict] = None) -> dict:
    llm = build_router_model()

    # Inyectamos la fecha/hora real para que el LLM resuelva "hoy", "mañana", etc.
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
        # Añadir al prompt los mensajes pasados excepto el mensaje actual que ya vino en 'history' pero lo repetimos en user_message
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
        # Intentar inferir intent del texto crudo
        lower = raw_text.lower() if raw_text else ""
        if "evento" in lower:
            return {"intent": "eventos", "target_date": datetime.now().strftime("%Y-%m-%d")}
        return {"intent": "eventos", "target_date": datetime.now().strftime("%Y-%m-%d")}

    except Exception as e:
        logger.error(f"Routing error: {e}", exc_info=True)
        # Default fallback — eventos (no viajes) ya que es el caso más común
        return {"intent": "eventos", "target_date": datetime.now().strftime("%Y-%m-%d")}
