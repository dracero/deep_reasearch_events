import asyncio
import json
import logging
import uuid
from typing import AsyncGenerator, List, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
from a2a.client import ClientFactory
from a2a.client.client import ClientConfig
from a2a.types import (
    Message,
    TextPart,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)

from router import determine_intent, AGENT_URLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BeeAI Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []
    active_agent: str = ""   # "viajes" | "eventos" | "" — bypass the router when set
    travel_context: dict = {}  # Carry over origin/destination from the previous turn


async def stream_a2a_agent(agent_name: str, url: str, params: dict) -> AsyncGenerator[str, None]:
    """
    Connects to the A2A Agent on `url` using ClientFactory and streams UI components back to the frontend.
    Sends SSE heartbeat comments every 15s to prevent connection timeouts during Groq rate-limit retries.
    """
    yield f"data: {json.dumps({'type': 'ui', 'component': 'AgentBadge', 'props': {'agent_name': agent_name}})}\n\n"

    message = Message(
        messageId=str(uuid.uuid4()),
        role=Role.user,
        parts=[Part(root=TextPart(text=json.dumps(params)))]
    )

    try:
        # Use a long timeout since Groq rate-limit retries can take 60s+
        _httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
        client = await ClientFactory.connect(
            agent=url,
            client_config=ClientConfig(streaming=True, httpx_client=_httpx_client),
        )

        # Wrap the event iterator to interleave SSE heartbeats every 15s
        # so browsers / proxies don't close the idle connection during retries
        async def _iter_with_heartbeat():
            queue: asyncio.Queue = asyncio.Queue()

            async def _producer():
                try:
                    async for ev in client.send_message(message):
                        await queue.put(ev)
                finally:
                    await queue.put(None)  # sentinel

            producer = asyncio.create_task(_producer())
            try:
                while True:
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=15.0)
                        if item is None:
                            break
                        yield item
                    except asyncio.TimeoutError:
                        # No event for 15s → send a keep-alive comment
                        yield "__heartbeat__"
            finally:
                producer.cancel()

        async for event in _iter_with_heartbeat():
            if event == "__heartbeat__":
                yield ": keep-alive\n\n"  # SSE comment — not parsed as data by the client
                continue

            if isinstance(event, tuple):
                task, update = event
                if isinstance(update, TaskStatusUpdateEvent):
                    status_str = update.status.state if hasattr(update.status, 'state') else str(update.status)
                    yield f"data: {json.dumps({'type': 'ui', 'component': 'LoadingState', 'props': {'message': f'({agent_name}) {status_str}'}})}\n\n"
                elif isinstance(update, TaskArtifactUpdateEvent):
                    for part in update.artifact.parts:
                        if hasattr(part, 'root') and isinstance(part.root, TextPart):
                            yield f"data: {part.root.text}\n\n"
                        elif isinstance(part, TextPart):
                            yield f"data: {part.text}\n\n"
            elif isinstance(event, Message):
                for part in event.parts:
                    if hasattr(part, 'root') and isinstance(part.root, TextPart):
                        yield f"data: {part.root.text}\n\n"
                    elif isinstance(part, TextPart):
                        yield f"data: {part.text}\n\n"

    except Exception as e:
        logger.error(f"Error calling {url}: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'ui', 'component': 'ErrorState', 'props': {'error': str(e)}})}\n\n"


async def unified_chat_stream(request: ChatRequest) -> AsyncGenerator[str, None]:
    """
    1. Orquesta con BeeAI (Router) la intención.
    2. Delega vía A2A SDK al agente apropiado.
    3. Retorna SSE events al Frontend.
    """
    message = request.message
    active_agent = request.active_agent
    travel_context = request.travel_context

    yield f"data: {json.dumps({'type': 'ui', 'component': 'LoadingState', 'props': {'message': 'Orquestando agentes...'}})}\n\n"

    # --- Shortcut: If the frontend already knows which agent is in context,
    # skip the LLM router entirely. This prevents follow-up answers (e.g., "la segunda de junio")
    # from being misclassified as a new unrelated intent.
    if active_agent == "viajes":
        intent = "viajes"
        # Accumulate params: keep what we already know, fill in the missing field
        prev_origin = travel_context.get("origin", "")
        prev_destination = travel_context.get("destination", "")
        prev_dates = travel_context.get("travel_dates", "")

        decision = {
            "intent": "viajes",
            "origin": prev_origin,
            "destination": prev_destination,
            "travel_dates": prev_dates,
        }
        # The user's reply fills whichever critical field is still empty
        if not decision["destination"]:
            decision["destination"] = message
        elif not decision["travel_dates"]:
            decision["travel_dates"] = message
        else:
            # Both present — treat the reply as raw text for the agent to interpret
            decision["travel_dates"] = message
        logger.info(f"[⚡️ Bypass] Active agent is '{active_agent}' — skipping LLM router.")
    elif active_agent == "eventos":
        intent = "eventos"
        decision = {"intent": "eventos", "target_date": message}
        logger.info(f"[⚡️ Bypass] Active agent is '{active_agent}' — skipping LLM router.")
    else:
        decision = await determine_intent(message, request.history)
        intent = decision.get("intent", "eventos")

    logger.info(f"Orchestrator Decision: {intent} (params: {decision})")

    if intent == "conversacional":
        # Atajo: respondemos directamente sin llamar a los agentes
        respuesta = decision.get("text_response", "¡Hola! Estoy acá para ayudarte con búsquedas de eventos o viajes. ¿Qué necesitas?")
        yield f"data: {json.dumps({'type': 'ui', 'component': 'AgentBadge', 'props': {'agent_name': 'Orchestrator'}})}\n\n"
        yield f"data: {json.dumps({'type': 'ui', 'component': 'UserChat', 'props': {'message': respuesta}})}\n\n"
        return

    if intent in ["eventos", "ambos"]:
        # Params para el Agente Eventos — sin defaults duros para que Clarify funcione
        params = {
            "target_date": decision.get("target_date", ""),
            "user_message_raw": message,
        }
        async for sse in stream_a2a_agent("Agent Eventos", AGENT_URLS["eventos"], params):
            yield sse

    if intent in ["viajes", "ambos"]:
        # Params para Agente Viajes sin defaults duros para permitir que Clarify funcione
        params = {
            "origin": decision.get("origin", ""),
            "destination": decision.get("destination", ""),
            "travel_dates": decision.get("travel_dates", ""),
            "user_message_raw": message, # Necesario para que el nodo Clarify sepa qué dijo el usuario
        }
        async for sse in stream_a2a_agent("Agent Viajes", AGENT_URLS["viajes"], params):
            yield sse
        # Emit accumulated context so the frontend can carry it to the next turn
        ctx = {"origin": params["origin"], "destination": params["destination"], "travel_dates": params["travel_dates"]}
        yield f"data: {json.dumps({'type': 'ui', 'component': 'ContextUpdate', 'props': {'context': ctx}})}\n\n"


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    return StreamingResponse(unified_chat_stream(request), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
