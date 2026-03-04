import json
import logging
import uuid

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    TaskStatusUpdateEvent,
    TaskStatus,
    TaskState,
    TaskArtifactUpdateEvent,
    Artifact,
    Part,
    TextPart,
)
from graph import build_graph

logger = logging.getLogger(__name__)

# Build the LangGraph once
graph = build_graph()


class EventosAgentExecutor(AgentExecutor):
    """
    Executes events tasks via the A2A protocol using LangGraph.
    """

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel a running task — A2A required abstract method."""
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.canceled),
                final=True,
            )
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        task_id = context.task_id
        context_id = context.context_id

        # Use the SDK helper that extracts all text parts from the incoming message
        user_message = context.get_user_input()
        logger.info(f"Received A2A task with message: {user_message.strip()}")

        # Emit standard 'working' update (non-final)
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.working),
                final=False,
            )
        )

        # Parse target_date and new params from the incoming message (may be plain text or JSON)
        categoria_evento = None
        proveedor_esperado = None
        try:
            if "{" in user_message and "}" in user_message:
                start_idx = user_message.find("{")
                end_idx = user_message.rfind("}") + 1
                json_str = user_message[start_idx:end_idx]
                params = json.loads(json_str)
                target_date = params.get("target_date", "")
                categoria_evento = params.get("categoria_evento", "") or None
                proveedor_esperado = params.get("proveedor_esperado", "") or None
            else:
                target_date = ""
        except Exception:
            target_date = ""

        if not target_date:
            logger.info("No target_date provided, clarify node will handle it")

        initial_state = {
            "target_date": target_date,
            "user_category": categoria_evento,
            "user_provider": proveedor_esperado,
            "user_message_raw": user_message,   # For the clarify node
            "clarify_question": "",             # Empty = clarify node will generate one
            "search_plan": None,
            "raw_events": [],
            "filtered_events": None,
            "final_report": None,
        }

        try:
            # Execute LangGraph and stream intermediate steps as Artifacts
            async for event in graph.astream(initial_state, stream_mode="updates"):
                # --- clarify Node: stream question to the user as a chat message ---
                if "clarify" in event and event["clarify"] is not None:
                    question = event["clarify"].get("clarify_question", "")
                    if question:
                        # Stream the question to the user as a UI component
                        await event_queue.enqueue_event(
                            TaskArtifactUpdateEvent(
                                task_id=task_id,
                                context_id=context_id,
                                artifact=Artifact(
                                    artifact_id=str(uuid.uuid4()),
                                    parts=[
                                        Part(root=TextPart(text=json.dumps({
                                            "type": "ui",
                                            "component": "AgentQuestion",
                                            "props": {"question": question},
                                        })))
                                    ],
                                ),
                            )
                        )
                        # Signal input_required so the frontend knows to wait for user answer
                        await event_queue.enqueue_event(
                            TaskStatusUpdateEvent(
                                task_id=task_id,
                                context_id=context_id,
                                status=TaskStatus(state=TaskState.input_required),
                                final=True,
                            )
                        )
                        return  # Stop execution until the user responds

                # --- filter_argentina Node ---
                if "filter_argentina" in event and event["filter_argentina"] is not None:
                    filtered = event["filter_argentina"].get("filtered_events", [])
                    await event_queue.enqueue_event(
                        TaskArtifactUpdateEvent(
                            task_id=task_id,
                            context_id=context_id,
                            artifact=Artifact(
                                artifact_id=str(uuid.uuid4()),
                                parts=[
                                    Part(root=TextPart(text=json.dumps({
                                        "type": "ui",
                                        "component": "EventTable",
                                        "props": {"events": filtered},
                                    })))
                                ],
                            ),
                        )
                    )

                # --- generate_report Node ---
                if "generate_report" in event:
                    break

            # Finish task successfully (final=True closes the stream)
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(state=TaskState.completed),
                    final=True,
                )
            )

        except Exception as e:
            logger.error(f"Error executing graph: {e}", exc_info=True)
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(state=TaskState.failed),
                    final=True,
                )
            )
