"""
api/routers/chat.py
────────────────────
Chat endpoints.

POST /api/chat/message           — send a message, get full response (blocking)
POST /api/chat/stream            — send a message, stream the response via SSE
GET  /api/chat/export/{id}       — export full chat as markdown text

v4: Both endpoints now accept an optional `model` field in the request body.
  - Omitted / null → server default LLM
  - "auto"         → query-based auto-routing via model_selector.classify_query()
  - specific id    → use that exact model (e.g. "gemma-4-31b-it")

The response / SSE events include a `model_used` field so the frontend
can display which model actually answered.
"""

import json
import asyncio
import threading
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse, PlainTextResponse
from datetime import datetime

from api.schemas.chat import ChatMessageIn, ChatMessageOut
from api.core.errors import not_found, bad_request

from chat_store import (
    get_session, update_session_after_message,
)
from agent import ask, get_executor_for_session, get_session_messages_for_display, get_active_llm_name
from workspace_manager import load_workspace

router = APIRouter()


def _resolve_model_params(model_field: str | None) -> tuple[str | None, bool]:
    """
    Parse the `model` request field into (model_override, auto_route).

    Returns:
        model_override  — explicit model ID, or None
        auto_route      — True when model == "auto"
    """
    if not model_field or model_field.strip() == "":
        return None, False
    if model_field.strip().lower() == "auto":
        return None, True
    return model_field.strip(), False


# ── Blocking chat ─────────────────────────────────────────────────────────────

@router.post("/message", response_model=ChatMessageOut)
def send_message(body: ChatMessageIn):
    """
    Send a message and wait for the complete agent response.
    Use this for simple integrations; prefer /stream for a responsive UI.
    """
    session = get_session(body.session_id)
    if not session:
        raise not_found(f"Session '{body.session_id}' not found.")

    if not body.message.strip():
        raise bad_request("Message cannot be empty.")

    is_first = session.get("message_count", 0) == 0
    model_override, auto_route = _resolve_model_params(body.model)

    cfg = load_workspace(body.session_id)
    executor = get_executor_for_session(body.session_id, cfg)

    response, model_used = ask(
        executor,
        body.message,
        session_id=body.session_id,
        model_override=model_override,
        auto_route=auto_route,
        has_codebase=cfg.indexed,
    )
    update_session_after_message(body.session_id, body.message, is_first_message=is_first)

    return ChatMessageOut(
        session_id=body.session_id,
        role="assistant",
        content=response,
        is_first_message=is_first,
        model_used=model_used,
    )


# ── Streaming chat (SSE) ──────────────────────────────────────────────────────

@router.post("/stream")
async def stream_message(body: ChatMessageIn):
    """
    Send a message and stream the agent response via Server-Sent Events.

    SSE event shapes:
        {"type": "thinking",  "content": "", "model_used": "..."}  — agent is working
        {"type": "done",      "content": "<full>", "model_used": "..."}  — final response
        {"type": "error",     "content": "<msg>", "model_used": ""}  — agent error
    """
    session = get_session(body.session_id)
    if not session:
        raise not_found(f"Session '{body.session_id}' not found.")

    if not body.message.strip():
        raise bad_request("Message cannot be empty.")

    is_first = session.get("message_count", 0) == 0
    model_override, auto_route = _resolve_model_params(body.model)

    async def event_stream() -> AsyncGenerator[str, None]:
        def _sse(payload: dict) -> str:
            return f"data: {json.dumps(payload)}\n\n"

        # Determine thinking label based on model selection
        if auto_route:
            thinking_hint = "🧠 Selecting best model…"
        elif model_override:
            thinking_hint = f"⚙️ Using {model_override}…"
        else:
            thinking_hint = "Thinking…"

        yield _sse({
            "type": "thinking",
            "content": thinking_hint,
            "session_id": body.session_id,
            "model_used": "",
        })
        await asyncio.sleep(0)

        result_container: dict = {}
        done_event = threading.Event()

        def _run():
            try:
                cfg = load_workspace(body.session_id)
                executor = get_executor_for_session(body.session_id, cfg)
                response, model_used = ask(
                    executor,
                    body.message,
                    session_id=body.session_id,
                    model_override=model_override,
                    auto_route=auto_route,
                    has_codebase=cfg.indexed,
                )
                result_container["response"] = response
                result_container["model_used"] = model_used
            except Exception as e:
                result_container["error"] = str(e)
                result_container["model_used"] = ""
            finally:
                done_event.set()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        while not done_event.wait(timeout=0.5):
            yield ": keep-alive\n\n"
            await asyncio.sleep(0.5)

        if "error" in result_container:
            yield _sse({
                "type": "error",
                "content": result_container["error"],
                "session_id": body.session_id,
                "model_used": result_container.get("model_used", ""),
            })
        else:
            response_text = result_container.get("response", "")
            model_used = result_container.get("model_used", get_active_llm_name())
            update_session_after_message(body.session_id, body.message, is_first_message=is_first)
            yield _sse({
                "type": "done",
                "content": response_text,
                "session_id": body.session_id,
                "model_used": model_used,
            })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── Export chat ────────────────────────────────────────────────────────────────

@router.get("/export/{session_id}", response_class=PlainTextResponse)
def export_chat(session_id: str):
    """Export the full chat history as a Markdown text file."""
    session = get_session(session_id)
    if not session:
        raise not_found(f"Session '{session_id}' not found.")

    messages = get_session_messages_for_display(session_id)
    if not messages:
        return PlainTextResponse("No messages in this session.", media_type="text/plain")

    lines = [
        f"# Poshak-AI Export",
        f"_{datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        f"**Session:** {session.get('title', 'Untitled')}",
        "",
    ]
    for msg in messages:
        role = msg.get("role", "?").capitalize()
        content = msg.get("content", "")
        lines.append(f"**{role}:**\n\n{content}\n")
        lines.append("---")

    return PlainTextResponse("\n".join(lines), media_type="text/plain; charset=utf-8")