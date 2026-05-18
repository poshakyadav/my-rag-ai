"""
api/schemas/session.py
───────────────────────
Request / Response models for the sessions router.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


# ── Response models ───────────────────────────────────────────────────────────

class SessionOut(BaseModel):
    session_id: str
    workspace: str
    title: str
    preview: str
    message_count: int
    created_at: str
    updated_at: str


class SessionGroup(BaseModel):
    label: str          # "Today" | "Yesterday" | "This Week" | …
    sessions: list[SessionOut]


class SessionListOut(BaseModel):
    groups: list[SessionGroup]
    total: int


class MessageOut(BaseModel):
    role: str           # "user" | "assistant"
    content: str


class SessionDetailOut(SessionOut):
    messages: list[MessageOut]


# ── Request models ────────────────────────────────────────────────────────────

class CreateSessionIn(BaseModel):
    workspace: Optional[str] = Field(default="", description="Workspace name / label")


class RenameSessionIn(BaseModel):
    title: str = Field(..., min_length=1, max_length=80)