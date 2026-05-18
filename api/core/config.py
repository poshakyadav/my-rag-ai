"""
api/core/config.py
──────────────────
Centralised settings for the FastAPI layer.
All project-level settings (API keys, model names, DB paths) are still
read from the existing config.py at the project root.  This file only
adds HTTP-layer concerns (CORS, host, port, JWT secret, etc.).
"""

from functools import lru_cache
from typing import List
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Server ────────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Comma-separated string read from .env  (CORS_ORIGINS_STR=http://a,http://b)
    # Defaults include the Angular dev server and common local ports.
    CORS_ORIGINS_STR: str = (
        "http://localhost:4200,"
        "http://127.0.0.1:4200,"
        "http://localhost:3000,"
        "http://127.0.0.1:3000"
    )

    # Parsed list — populated automatically by the validator below.
    # Do NOT set this in .env; set CORS_ORIGINS_STR instead.
    CORS_ORIGINS: List[str] = []

    @model_validator(mode="after")
    def _parse_cors_origins(self) -> "Settings":
        """Split CORS_ORIGINS_STR into a real list after the model is built."""
        self.CORS_ORIGINS = [
            o.strip()
            for o in self.CORS_ORIGINS_STR.split(",")
            if o.strip()
        ]
        return self

    # ── Streaming / SSE ───────────────────────────────────────────────────────
    SSE_TIMEOUT: int = 120

    # ── Pagination ────────────────────────────────────────────────────────────
    MAX_SESSIONS_PER_PAGE: int = 200


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()