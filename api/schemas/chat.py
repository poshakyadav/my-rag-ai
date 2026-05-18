"""
api/schemas/chat.py
────────────────────
Request / Response models for the chat router.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class ChatMessageIn(BaseModel):
    message: str = Field(..., min_length=1, description="The user's message text")
    session_id: str = Field(..., description="Active session ID")
    model: Optional[str] = Field(
        default=None,
        description=(
            "Model ID to use for this message. "
            "Pass 'auto' for intelligent auto-routing, "
            "or a specific model ID from GET /api/system/models. "
            "Omit or pass null to use the server default."
        ),
    )


class ChatMessageOut(BaseModel):
    session_id: str
    role: str = "assistant"
    content: str
    is_first_message: bool = False
    model_used: str = ""


class StreamChunkOut(BaseModel):
    """Shape of each SSE data payload during streaming."""
    type: str           # "token" | "done" | "error" | "thinking"
    content: str        # Token text, final full response, or error detail
    session_id: str
    model_used: str = ""