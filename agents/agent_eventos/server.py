from dotenv import load_dotenv
from pathlib import Path
import logging

_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env")

import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from agent_card import AGENT_CARD
from agent_executor import EventosAgentExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

# Build the A2A application object
a2a_app_builder = A2AStarletteApplication(
    agent_card=AGENT_CARD,
    http_handler=DefaultRequestHandler(
        agent_executor=EventosAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
)

# Build the Starlette app with CORS middleware
app = a2a_app_builder.build(
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]
)

# Add a health check route
async def health_check(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "agent": "eventos", "a2a": True})

app.routes.append(Route("/health", health_check, methods=["GET"]))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)
