"""
api/schemas/workspace.py
─────────────────────────
Request / Response models for the workspace router.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


# ── Response models ───────────────────────────────────────────────────────────

class WorkspaceOut(BaseModel):
    session_id: str
    mode: str                       # "none" | "existing" | "new"
    paths: dict[str, str]           # {label: absolute_path}
    label: str
    indexed: bool
    collection_name: str
    badge: str                      # Short display string for the top bar


class ValidationResultOut(BaseModel):
    ok: bool
    valid_paths: dict[str, str]
    errors: list[str]
    warnings: list[str]
    message: str


class IndexProgressOut(BaseModel):
    """Emitted repeatedly over SSE during indexing."""
    stage: str          # "scanning" | "splitting" | "embedding" | "indexing" | "done" | "error"
    message: str
    chunks_done: Optional[int] = None
    chunks_total: Optional[int] = None


class IndexResultOut(BaseModel):
    success: bool
    message: str
    chunks: Optional[int] = None
    files: Optional[int] = None


# ── Request models ────────────────────────────────────────────────────────────

class ValidatePathsIn(BaseModel):
    raw_paths: str = Field(..., description="Raw path string (one or more paths, optionally labeled)")
    mode: str = Field(default="existing", description="'existing' | 'new' | 'none'")


class ApplyWorkspaceIn(BaseModel):
    raw_paths: str = Field(default="")
    mode: str = Field(default="existing")