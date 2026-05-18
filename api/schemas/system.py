"""
api/schemas/system.py
──────────────────────
Request / Response models for the system/status router.
"""

from __future__ import annotations
from pydantic import BaseModel


class SystemStatusOut(BaseModel):
    llm_model: str
    db_chunks: int
    session_count: int
    timestamp: str
    version: str = "1.0.0"


class ExamplePromptsOut(BaseModel):
    prompts: list[str]


class ModelInfo(BaseModel):
    """Single model entry returned from GET /api/system/models."""
    id: str
    label: str
    description: str
    tier: str       # "fast" | "balanced" | "heavy" | "auto"
    provider: str   # "google" | "groq" | "auto"
    available: bool = True


class ModelsOut(BaseModel):
    models: list[ModelInfo]
    default: str    # the model ID currently used as server default