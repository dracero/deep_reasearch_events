import asyncio
from typing import List
from agents.agent_viajes.prompts import RESEARCHER_SYSTEM_PROMPT
from agents.agent_viajes.state import RouteFindings
from agents.agent_viajes.configuration import Configuration
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

def _get_llm(model_name: str, temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(model=model_name, temperature=temperature, max_retries=1, api_key=os.environ.get("GROQ_API_KEY"))

async def test_extraction():
    config = Configuration.from_env()
    llm = _get_llm(config.researcher_model)
    structured_llm = llm.with_structured_output(RouteFindings)
    
    prompt = RESEARCHER_SYSTEM_PROMPT.format(
        origin="Buenos Aires",
        destination="Junin",
        mode_hint="bus",
        travel_dates="2024-10-10",
    )
    
    mock_search_context = """
    **Fuente**: https://www.plataforma10.com.ar/pasajes-a-mendoza
    **Título**: Pasajes a Mendoza
    **Contenido**: Viajes populares: Buenos Aires a Mendoza $50000. Salida 14:00, llegada 08:00.
    
    **Fuente**: https://www.kayak.com.ar/
    **Título**: Vuelos baratos
    **Contenido**: Ofertas desde Retiro a Junín por $20000. Sale a las 10:00 y llega a las 14:00.
    """
    
    res = structured_llm.invoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Resultados de búsqueda:\n\n{mock_search_context}"}
    ])
    
    print(res.model_dump())

if __name__ == "__main__":
    asyncio.run(test_extraction())
