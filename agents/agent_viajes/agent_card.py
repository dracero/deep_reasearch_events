from a2a.types import AgentCard, AgentSkill, AgentCapabilities

AGENT_CARD = AgentCard(
    name="Agent Viajes Baratos",
    description="Deep Research Agent para encontrar rutas aéreas baratas y combinaciones óptimas",
    url="http://localhost:8002",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=True, pushNotifications=False),
    skills=[
        AgentSkill(
            id="search_cheap_routes",
            name="Buscar Rutas Baratas",
            description="Encuentra las combinaciones de vuelos más baratas dados un origen, destino y fechas",
            tags=["viajes", "vuelos", "ofertas", "rutas", "turismo"],
            input_schema={
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "travel_dates": {"type": "string"},
                    "flexibility_days": {"type": "integer"},
                    "budget_max_usd": {"type": "number"},
                },
                "required": ["origin", "destination", "travel_dates"],
            },
            output_schema={
                "type": "array",
                "items": {"type": "object"},
            },
            examples=[
                "Buscame la ruta más barata de Buenos Aires a USA para el mundial en 2026",
            ]
        )
    ],
    defaultInputModes=["text", "json"],
    defaultOutputModes=["text", "json"],
)
