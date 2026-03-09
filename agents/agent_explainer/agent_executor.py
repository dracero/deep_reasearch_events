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


class ExplainerAgentExecutor(AgentExecutor):
    """
    Executes web page Q&A tasks via the A2A protocol using LangGraph.
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

        # Parse explainer params from the incoming message (may be plain text or JSON)
        try:
            if "{" in user_message and "}" in user_message:
                start_idx = user_message.find("{")
                end_idx = user_message.rfind("}") + 1
                json_str = user_message[start_idx:end_idx]
                params = json.loads(json_str)
                url = params.get("url", "")
                question = params.get("question", "")
                user_message_raw = params.get("user_message_raw", user_message)
            else:
                url = ""
                question = ""
                user_message_raw = user_message
        except Exception:
            url = ""
            question = ""
            user_message_raw = user_message

        # Check if we have cached scraped content for this URL
        cached_content = ""

        initial_state = {
            "url": url,
            "question": question,
            "user_message_raw": user_message_raw,
            "clarify_question": "",
            "scraped_content": cached_content,  # Pre-fill if cached
            "final_explanation": None,
        }

        try:
            # Execute LangGraph and stream intermediate steps as Artifacts
            async for event in graph.astream(initial_state, stream_mode="updates"):
                # --- clarify Node ---
                if "clarify" in event and event["clarify"] is not None:
                    clarify_data = event["clarify"]
                    # Track URL/question extracted by the clarify node
                    extracted_url = clarify_data.get("url", "")
                    extracted_question = clarify_data.get("question", "")
                    # Update our local tracking so the final ContextUpdate is correct
                    if extracted_url:
                        url = extracted_url
                    if extracted_url or extracted_question:
                        await event_queue.enqueue_event(
                            TaskArtifactUpdateEvent(
                                task_id=task_id,
                                context_id=context_id,
                                artifact=Artifact(
                                    artifact_id=str(uuid.uuid4()),
                                    parts=[
                                        Part(root=TextPart(text=json.dumps({
                                            "type": "ui",
                                            "component": "ContextUpdate",
                                            "props": {
                                                "context": {"url": extracted_url, "question": extracted_question},
                                                "agent": "explainer",
                                            },
                                        })))
                                    ],
                                ),
                            )
                        )

                    question_text = clarify_data.get("clarify_question", "")
                    if question_text:
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
                                            "props": {"question": question_text},
                                        })))
                                    ],
                                ),
                            )
                        )
                        await event_queue.enqueue_event(
                            TaskStatusUpdateEvent(
                                task_id=task_id,
                                context_id=context_id,
                                status=TaskStatus(state=TaskState.input_required),
                                final=True,
                            )
                        )
                        return

                # --- scrape_url Node ---
                if "scrape_url" in event and event["scrape_url"] is not None:
                    new_content = event["scrape_url"].get("scraped_content", "")
                    if new_content and url:
                        logger.info(f"📦 Scraped content ready for {url} ({len(new_content)} chars)")

                # --- answer_question Node ---
                if "answer_question" in event:
                    result = event["answer_question"].get("final_explanation")
                    if isinstance(result, dict):
                        explanation = result.get("explanation", "Hubo un error al generar la respuesta.")
                    else:
                        explanation = getattr(result, "explanation", "Hubo un error al generar la respuesta.")
                    
                    await event_queue.enqueue_event(
                        TaskArtifactUpdateEvent(
                            task_id=task_id,
                            context_id=context_id,
                            artifact=Artifact(
                                artifact_id=str(uuid.uuid4()),
                                parts=[
                                    Part(root=TextPart(text=explanation))
                                ],
                            ),
                        )
                    )

                    # Emit context update with the URL so the frontend carries it forward
                    effective_url = url
                    await event_queue.enqueue_event(
                        TaskArtifactUpdateEvent(
                            task_id=task_id,
                            context_id=context_id,
                            artifact=Artifact(
                                artifact_id=str(uuid.uuid4()),
                                parts=[
                                    Part(root=TextPart(text=json.dumps({
                                        "type": "ui",
                                        "component": "ContextUpdate",
                                        "props": {
                                            "context": {"url": effective_url, "question": ""},
                                            "agent": "explainer",
                                        },
                                    })))
                                ],
                            ),
                        )
                    )

            # After answering, signal input_required so the user can ask more questions
            # about the same page. The frontend will keep the conversation open.
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(state=TaskState.input_required),
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
