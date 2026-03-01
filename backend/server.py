from dotenv import load_dotenv
load_dotenv()

import json
import logging
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from graph import build_graph
from langgraph.types import Send

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Deep Research Argentina API")

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the LangGraph once on startup
graph = build_graph()

class ResearchRequest(BaseModel):
    date: str

async def generate_a2ui_events(date: str) -> AsyncGenerator[str, None]:
    """
    Runs the graph and yields A2UI-compatible JSON lines as Server-Sent Events (SSE).
    """
    initial_state = {
        "target_date": date,
        "search_plan": None,
        "raw_events": [],
        "filtered_events": [],
        "final_report": ""
    }
    
    # 1. Emit Initial Loading State A2UI component
    yield f"data: {json.dumps({'type': 'ui', 'component': 'LoadingState', 'props': {'message': 'Generando plan de búsqueda...'}})}\n\n"

    try:
        # Stream the graph execution
        async for event in graph.astream(initial_state, stream_mode="updates"):
            
            # --- plan_research Node ---
            if "plan_research" in event:
                plan = event["plan_research"].get("search_plan")
                if plan:
                    queries_list = [q.query for q in plan.queries]
                    yield f"data: {json.dumps({'type': 'ui', 'component': 'SearchPlan', 'props': {'queries': queries_list}})}\n\n"
                    
            # --- research_category Node (Parallel) ---
            if "research_category" in event:
                research_output = event["research_category"]
                
                # Check if it's the raw dictionary or a Send object
                if isinstance(research_output, dict):
                    # We only know category inside the state actually, but we can just say "Finding events..."
                    yield f"data: {json.dumps({'type': 'ui', 'component': 'LoadingState', 'props': {'message': 'Investigando categoría...'}})}\n\n"

            # --- aggregate_results Node ---
            if "aggregate_results" in event:
                agg = event["aggregate_results"].get("raw_events")
                count = len(agg) if isinstance(agg, list) else 0 # Actually it's an override dict now
                if isinstance(agg, dict) and agg.get("type") == "override":
                    count = len(agg.get("value", []))
                
                yield f"data: {json.dumps({'type': 'ui', 'component': 'LoadingState', 'props': {'message': f'Se encontraron {count} eventos crudos. Filtrando por relevancia en Argentina...'}})}\n\n"

            # --- filter_argentina Node ---
            if "filter_argentina" in event:
                filtered = event["filter_argentina"].get("filtered_events", [])
                
                # This is the meat of the A2UI! Emit the interactive table component
                yield f"data: {json.dumps({'type': 'ui', 'component': 'EventTable', 'props': {'events': filtered}})}\n\n"
                
            # --- generate_report Node ---
            if "generate_report" in event:
                yield f"data: {json.dumps({'type': 'ui', 'component': 'LoadingState', 'props': {'message': 'Investigación completada.'}})}\n\n"
                # Stop streaming
                break
                
    except Exception as e:
        logger.error(f"Error executing graph: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'ui', 'component': 'ErrorState', 'props': {'error': str(e)}})}\n\n"

@app.post("/api/research")
async def research_endpoint(request: ResearchRequest):
    """
    Endpoint that triggers the LangGraph deep research process and returns
    A2UI component events via SSE.
    """
    return StreamingResponse(
        generate_a2ui_events(request.date),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
