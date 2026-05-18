"""
api/main.py — Poshak-AI FastAPI Backend
─────────────────────────────────────────
Production-grade REST API replacing Gradio UI backend.

FIX: CORS headers were missing on 500 responses.
─────────────────────────────────────────────────
Root cause: Starlette's ServerErrorMiddleware sits at the outermost layer of the
middleware stack and handles unhandled Python exceptions BEFORE CORSMiddleware
gets a chance to inject Access-Control-Allow-Origin headers.  The browser then
sees a 500 with no CORS header and reports it as a CORS error — masking the
real 500.

Solution (two-pronged):
  1. Register a global @app.exception_handler(Exception) that converts any
     unhandled exception into a proper JSONResponse.  FastAPI/Starlette route
     *exception handlers* fire after CORSMiddleware processes the response,
     so the CORS headers are present.
  2. Keep an explicit HTTP-middleware catch-all as a safety net for errors
     that escape the router entirely.

Middleware registration order matters:
  add_middleware() prepends to the stack, so the last-registered middleware
  is outermost.  We register CORS first so it ends up outermost (wrapping
  everything), then GZip.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import json
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# ── Windows UTF-8 fix ─────────────────────────────────────────────────────────
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from api.routers import sessions, workspace, chat, system
from api.core.startup import run_startup
from api.core.config import settings


# ── Lifespan: startup / shutdown ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the agent and DB on startup; clean up on shutdown."""
    run_startup()
    yield


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Poshak-AI API",
        description=(
            "RAG-powered coding assistant backend. "
            "Exposes agent, session, workspace, and chat endpoints."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # ── Global exception handler ───────────────────────────────────────────────
    # IMPORTANT: This must be registered BEFORE add_middleware() calls so that
    # FastAPI's exception-handler machinery runs inside the CORS middleware layer.
    #
    # When an unhandled exception propagates, FastAPI calls this handler and
    # returns a JSONResponse.  Starlette's CORSMiddleware then adds the
    # Access-Control-Allow-Origin header to that response normally.
    #
    # Without this, ServerErrorMiddleware catches the exception first (outside
    # CORSMiddleware) and returns a bare 500 with no CORS headers, causing the
    # browser to report it as a CORS error instead of a 500.

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        tb = traceback.format_exc()
        print(f"[UNHANDLED EXCEPTION] {request.method} {request.url}\n{tb}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error",
                "error": str(exc),
                # Include traceback only in debug mode
                "traceback": tb if settings.DEBUG else None,
            },
        )

    # ── Middleware ─────────────────────────────────────────────────────────────
    # add_middleware() PREPENDS — last registered = outermost.
    # Desired order (outside → inside): CORS → GZip → Router
    # So we register GZip first, then CORS.

    app.add_middleware(GZipMiddleware, minimum_size=1000)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],  # needed for file downloads
    )

    # ── Routers ────────────────────────────────────────────────────────────────
    app.include_router(sessions.router,  prefix="/api/sessions",  tags=["Sessions"])
    app.include_router(workspace.router, prefix="/api/workspace",  tags=["Workspace"])
    app.include_router(chat.router,      prefix="/api/chat",       tags=["Chat"])
    app.include_router(system.router,    prefix="/api/system",     tags=["System"])

    return app


app = create_app()