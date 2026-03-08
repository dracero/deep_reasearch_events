import asyncio
import os
import sys

# Add agent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agents', 'agent_explainer'))

from state import ExplainerState
from graph import build_graph

async def test_clarify():
    graph = build_graph()
    
    # Simulate user sending a message without URL
    initial_state = {
        "url": "",
        "question": "",
        "user_message_raw": "explicame de qué trata",
        "clarify_question": "",
        "scraped_content": "",
        "final_explanation": None,
    }
    
    print("Testing Clarify node direct execution...")
    try:
        async for event in graph.astream(initial_state, stream_mode="updates"):
            print("EVENT:", event)
            if "clarify" in event and event["clarify"] is not None:
                print("Clarify Output:", event["clarify"])
                break
    except Exception as e:
        print("ERROR:", e)

if __name__ == "__main__":
    asyncio.run(test_clarify())
