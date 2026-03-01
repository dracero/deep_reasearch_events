import asyncio
from agents.orchestrator.router import determine_intent

async def main():
    msg = "me podes dar la ruta mas barata para ir de buenos airea a kasas city el 15 de Junio para ver el debut de argentina en el mundial"
    res = await determine_intent(msg)
    print("RESULT:", res)

asyncio.run(main())
