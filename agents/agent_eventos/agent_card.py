from a2a.types import AgentCard, AgentSkill, AgentCapabilities

AGENT_CARD = AgentCard(
    name="Agent Eventos Argentina",
    description="Deep Research Agent para encontrar eventos que generen tráfico de internet en Argentina",
    url="http://localhost:8001",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=True, pushNotifications=False),
    skills=[
        AgentSkill(
            id="search_events",
            name="Buscar Eventos",
            description="Encuentra eventos deportivos, streaming, gaming y especiales que impacten el tráfico en Argentina para una fecha dada",
            tags=["eventos", "tráfico", "argentina", "deportes", "streaming", "gaming"],
            input_schema={
                "type": "object",
                "properties": {
                    "target_date": {"type": "string"},
                },
                "required": ["target_date"],
            },
            output_schema={
                "type": "array",
                "items": {"type": "object"},
            },
            examples=[
                "¿Qué eventos hay mañana que generen tráfico?",
                "Investigá los eventos del 14 de julio de 2026",
            ]
        )
    ],
    defaultInputModes=["text", "json"],
    defaultOutputModes=["text", "json"],
)
