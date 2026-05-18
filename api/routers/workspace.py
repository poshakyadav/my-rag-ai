"""
api/routers/workspace.py
─────────────────────────
Workspace management endpoints.

GET    /api/workspace/{session_id}          — get current workspace for session
POST   /api/workspace/{session_id}/validate — validate raw path input
POST   /api/workspace/{session_id}/apply    — validate + index paths (SSE stream)
DELETE /api/workspace/{session_id}          — clear workspace → general chat mode
"""

import json
import asyncio
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from api.schemas.workspace import (
    WorkspaceOut, ValidationResultOut,
    ValidatePathsIn, ApplyWorkspaceIn,
    IndexProgressOut, IndexResultOut,
)
from api.core.errors import not_found, bad_request

from chat_store import get_session
from workspace_manager import (
    WorkspaceConfig,
    validate_paths,
    scaffold_new_project,
    build_session_index,
    load_workspace,
    save_workspace,
    clear_workspace,
    format_workspace_badge,
)

router = APIRouter()


# ── Helper ────────────────────────────────────────────────────────────────────

def _workspace_out(cfg: WorkspaceConfig) -> WorkspaceOut:
    return WorkspaceOut(
        session_id=cfg.session_id,
        mode=cfg.mode,
        paths=cfg.paths,
        label=cfg.label,
        indexed=cfg.indexed,
        collection_name=cfg.collection_name,
        badge=format_workspace_badge(cfg),
    )


def _require_session(session_id: str):
    session = get_session(session_id)
    if not session:
        raise not_found(f"Session '{session_id}' not found.")
    return session


# ── Get workspace ─────────────────────────────────────────────────────────────

@router.get("/{session_id}", response_model=WorkspaceOut)
def get_workspace(session_id: str):
    """Return the current workspace configuration for a session."""
    _require_session(session_id)
    cfg = load_workspace(session_id)
    return _workspace_out(cfg)


# ── Validate paths ────────────────────────────────────────────────────────────

@router.post("/{session_id}/validate", response_model=ValidationResultOut)
def validate_workspace_paths(session_id: str, body: ValidatePathsIn):
    """
    Validate raw path input without persisting anything.
    Returns structured errors / warnings so the Angular UI can display them.
    """
    _require_session(session_id)

    if body.mode == "none":
        return ValidationResultOut(
            ok=True,
            valid_paths={},
            errors=[],
            warnings=[],
            message="✅ General chat mode — no project needed.",
        )

    if not body.raw_paths.strip():
        return ValidationResultOut(
            ok=False,
            valid_paths={},
            errors=["No path provided. Please enter at least one directory path."],
            warnings=[],
            message="",
        )

    result = validate_paths(body.raw_paths, mode=body.mode)
    return ValidationResultOut(
        ok=result.ok,
        valid_paths=result.valid_paths,
        errors=result.errors,
        warnings=result.warnings,
        message=result.message,
    )


# ── Apply + Index (SSE streaming) ─────────────────────────────────────────────

@router.post("/{session_id}/apply")
async def apply_workspace(session_id: str, body: ApplyWorkspaceIn, request: Request):
    """
    Validate paths, persist workspace, scaffold (if new), then index the project.
    Streams progress events via Server-Sent Events so the frontend can show a
    live progress bar.

    SSE event format:
        data: {"stage": "...", "message": "...", "chunks_done": N, "chunks_total": M}

    Final event:
        data: {"stage": "done", ...}   or   {"stage": "error", ...}
    """
    _require_session(session_id)

    async def event_stream() -> AsyncGenerator[str, None]:
        def _sse(payload: dict) -> str:
            return f"data: {json.dumps(payload)}\n\n"

        # ── Mode: none ─────────────────────────────────────────────────────
        if body.mode == "none":
            cfg = WorkspaceConfig(session_id=session_id, mode="none")
            save_workspace(cfg)
            yield _sse({"stage": "done", "message": "Switched to general chat mode.", "badge": format_workspace_badge(cfg)})
            return

        # ── Validate ───────────────────────────────────────────────────────
        if not body.raw_paths.strip():
            yield _sse({"stage": "error", "message": "No path provided."})
            return

        yield _sse({"stage": "validating", "message": "Validating paths…"})
        await asyncio.sleep(0)   # yield control so the SSE flushes

        result = validate_paths(body.raw_paths, mode=body.mode)
        if not result.ok:
            errors_str = "; ".join(result.errors)
            yield _sse({"stage": "error", "message": f"Validation failed: {errors_str}"})
            return

        # ── Scaffold new project ───────────────────────────────────────────
        if body.mode == "new":
            path_str = list(result.valid_paths.values())[0]
            ok, msg = scaffold_new_project(path_str)
            if not ok:
                yield _sse({"stage": "error", "message": msg})
                return
            yield _sse({"stage": "scaffolded", "message": msg})
            await asyncio.sleep(0)

        # ── Build workspace config + persist ───────────────────────────────
        label = ", ".join(result.valid_paths.keys())
        cfg = WorkspaceConfig(
            session_id=session_id,
            mode=body.mode,
            paths=result.valid_paths,
            label=label,
            indexed=False,
        )
        save_workspace(cfg)
        yield _sse({"stage": "scanning", "message": f"Workspace saved. Scanning {len(result.valid_paths)} project(s)…"})
        await asyncio.sleep(0)

        # ── Index (blocking, run in thread pool) ──────────────────────────
        progress_events: list[str] = []

        def _progress(msg: str):
            progress_events.append(msg)

        # Build index synchronously in the same thread (it's CPU/IO bound,
        # but for simplicity we keep it synchronous and stream accumulated
        # progress after it completes).  For a true async stream use
        # loop.run_in_executor with a Queue.
        loop = asyncio.get_event_loop()

        # Queue-based live streaming
        import queue, threading

        q: queue.Queue = queue.Queue()
        done_event = threading.Event()
        result_container: dict = {}

        def _run_index():
            def cb(msg: str):
                q.put(("progress", msg))
            success, message = build_session_index(cfg, progress_cb=cb)
            result_container["success"] = success
            result_container["message"] = message
            q.put(("done", None))

        thread = threading.Thread(target=_run_index, daemon=True)
        thread.start()

        while True:
            try:
                kind, msg = q.get(timeout=0.3)
            except queue.Empty:
                await asyncio.sleep(0)
                continue

            if kind == "progress":
                yield _sse({"stage": "indexing", "message": msg})
                await asyncio.sleep(0)
            elif kind == "done":
                break

        # ── Emit final result ──────────────────────────────────────────────
        if result_container.get("success"):
            fresh_cfg = load_workspace(session_id)
            yield _sse({
                "stage": "done",
                "message": result_container["message"],
                "badge": format_workspace_badge(fresh_cfg),
            })
        else:
            yield _sse({"stage": "error", "message": result_container.get("message", "Indexing failed.")})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


# ── Clear workspace ────────────────────────────────────────────────────────────

@router.delete("/{session_id}", response_model=WorkspaceOut)
def clear_workspace_endpoint(session_id: str):
    """Reset the session's workspace to general-chat mode."""
    _require_session(session_id)
    clear_workspace(session_id)
    cfg = load_workspace(session_id)
    return _workspace_out(cfg)