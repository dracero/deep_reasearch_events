# Deep Research Agent - Eventos Argentina

## What is this project?

This is a LangGraph multi-agent system that searches for events generating internet traffic in Argentina (such as sports, streaming, gaming, and awards ceremonies). It is inspired by [open_deep_research](https://github.com/langchain-ai/open_deep_research).

## Architecture

```mermaid
graph TD
    A["User Input: Date"] --> B["plan_research"]
    B -->|"Send() ×4"| C1["research_category: Deportes"]
    B -->|"Send() ×4"| C2["research_category: Streaming"]
    B -->|"Send() ×4"| C3["research_category: Gaming"]
    B -->|"Send() ×4"| C4["research_category: Especiales"]
    C1 --> D["aggregate_results"]
    C2 --> D
    C3 --> D
    C4 --> D
    D --> E["filter_argentina"]
    E --> F["generate_report"]
    F --> G["Pandas DataFrame"]
```

## Files Overview

| File | Purpose |
|------|---------|
| `.env` | Environment variables (Tavily + Gemini) |
| `state.py` | Graph states + Pydantic structured outputs |
| `configuration.py` | Agent configuration (models, search params) |
| `prompts.py` | System prompts for Planner, Researcher, Filter, Report |
| `graph.py` | LangGraph StateGraph with `Send()` for parallelism |
| `main.py` | CLI entry point, date input, pandas output |

## How to Run

1. Edit the API keys in your `.env` (use `.env.example` as a template). You'll need credentials for `TAVILY_API_KEY` and `GEMINI_API_KEY`.
2. Ensure you have `uv` installed, then run:

```bash
uv run python main.py
```

## A2UI Frontend Architecture (Server-Driven UI)

This project uses the Agent-to-UI (A2UI) protocol, where the LangGraph backend controls exactly what React components the frontend renders at any given step of the agent's reasoning process.

```mermaid
sequenceDiagram
    participant User
    participant App as React Frontend (App.tsx)
    participant Server as FastAPI Server (server.py)
    participant Graph as LangGraph Engine
    
    User->>App: Clicks "Investigar" (Search)
    App->>Server: POST /api/research
    Note over App,Server: Opens Server-Sent Events (SSE) Stream
    Server->>Graph: graph.astream(initial_state)
    
    Note over Server,Graph: Node: plan_research
    Graph-->>Server: yields "plan_research" state
    Server-->>App: SSE: { component: 'SearchPlan', props: {...} }
    App->>App: Dynamically renders <SearchPlan />
    
    Note over Server,Graph: Node: research_category
    Graph-->>Server: yields "research_category" state
    Server-->>App: SSE: { component: 'LoadingState', props: {...} }
    App->>App: Dynamically renders <LoadingState />
    
    Note over Server,Graph: Node: filter_argentina
    Graph-->>Server: yields "filter_argentina" state
    Server-->>App: SSE: { component: 'EventTable', props: {events} }
    App->>App: Dynamically renders <EventTable />
    
    Graph-->>Server: Finished
    Server-->>App: Stream closed
```
