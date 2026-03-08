from a2a.types import AgentCard, AgentSkill, AgentCapabilities

AGENT_CARD = AgentCard(
    name="Agent Explicador",
    description="Deep Research Agent para explicar contenido web a partir de una URL",
    url="http://localhost:8002",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=True, pushNotifications=False),
    skills=[
        AgentSkill(
            id="explain_web_content",
            name="Explicar Contenido",
            description="Lee el contenido de una página web y lo explica",
            tags=["explicador", "resumen", "link", "url"],
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "topic": {"type": "string"},
                },
                "required": ["url", "topic"],
            },
            output_schema={
                "type": "string"
            },
            examples=[
                "Explicame la trama de matrix leyendo el artículo de wikipedia url:...",
            ]
        )
    ],
    defaultInputModes=["text", "json"],
    defaultOutputModes=["text", "json"],
)
