import asyncio
from agents.orchestrator.router import determine_intent

async def main():
    msg = "hola quien sos"
    res = await determine_intent(msg)
    print("RESULT:", res)

asyncio.run(main())
