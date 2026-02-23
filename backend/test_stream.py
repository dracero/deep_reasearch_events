import httpx
import sys

async def main():
    date = "2026-02-23" if len(sys.argv) < 2 else sys.argv[1]
    url = "http://localhost:8000/api/research"
    
    print(f"Testando streaming A2UI para la fecha: {date}")
    
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, json={"date": date}) as response:
            async for line in response.aiter_lines():
                if line:
                    print(line)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
