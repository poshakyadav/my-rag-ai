"""
api/routers/sessions.py
────────────────────────
CRUD endpoints for chat sessions.

GET    /api/sessions              — list all sessions (grouped by date)
POST   /api/sessions              — create a new session
GET    /api/sessions/{id}         — get session metadata + message history
DELETE /api/sessions/{id}         — delete a session
PATCH  /api/sessions/{id}/rename  — rename a session title
GET    /api/sessions/{id}/messages — get only the message list

FIX: get_session_detail() now wraps get_session_messages_for_display() in a
try/except and raises HTTPException(500) instead of letting raw Python
exceptions escape the router.  HTTPException-based 500s are handled by
FastAPI's exception machinery which runs *inside* CORSMiddleware — so the
browser receives the CORS headers and can read the actual error.
"""

from fastapi import APIRouter, HTTPException, status

from api.schemas.session import (
    SessionOut, SessionListOut, SessionGroup,
    SessionDetailOut, MessageOut,
    CreateSessionIn, RenameSessionIn,
)
from api.core.errors import not_found, bad_request, internal

from chat_store import (
    create_session, get_session, list_sessions, delete_session,
    rename_session, group_sessions_by_date,
)
from agent import get_session_messages_for_display
from workspace_manager import WorkspaceConfig, save_workspace

router = APIRouter()


# ── List sessions ─────────────────────────────────────────────────────────────

@router.get("", response_model=SessionListOut)
def list_all_sessions():
    """Return all sessions grouped by date (Today / Yesterday / This Week…)."""
    try:
        all_sessions = list_sessions(workspace="", limit=200)
        grouped = group_sessions_by_date(all_sessions)

        groups: list[SessionGroup] = []
        for label, sessions in grouped.items():
            groups.append(SessionGroup(
                label=label,
                sessions=[SessionOut(**s) for s in sessions],
            ))

        return SessionListOut(groups=groups, total=len(all_sessions))
    except Exception as exc:
        raise internal(f"Failed to list sessions: {exc}")


# ── Create session ────────────────────────────────────────────────────────────

@router.post("", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
def create_new_session(body: CreateSessionIn = CreateSessionIn()):
    """Create a new chat session. Optionally attach a workspace label."""
    try:
        session_id = create_session(workspace=body.workspace or "")

        # Persist a "none" workspace so workspace queries don't 404 later.
        cfg = WorkspaceConfig(session_id=session_id, mode="none")
        save_workspace(cfg)

        session = get_session(session_id)
        if not session:
            raise internal("Session was created but could not be retrieved.")
        return SessionOut(**session)
    except HTTPException:
        raise
    except Exception as exc:
        raise internal(f"Failed to create session: {exc}")


# ── Get session detail ────────────────────────────────────────────────────────

@router.get("/{session_id}", response_model=SessionDetailOut)
def get_session_detail(session_id: str):
    """
    Return session metadata plus full message history.

    FIX: previously, any exception from get_session_messages_for_display()
    (e.g. SqliteSaver.get() AttributeError with old LangGraph API) would
    propagate as an unhandled Python exception.  Starlette's
    ServerErrorMiddleware would catch it BEFORE CORSMiddleware, returning a
    bare 500 with no CORS headers.  The browser would then report a CORS
    error, hiding the real 500.

    Now we either catch it here and raise HTTPException(500), which FastAPI
    handles inside the CORS layer, OR agent.py's get_session_messages_for_display
    itself swallows the exception and returns [] — both paths are safe.
    """
    session = get_session(session_id)
    if not session:
        raise not_found(f"Session '{session_id}' not found.")

    try:
        raw_messages = get_session_messages_for_display(session_id)
        messages = [MessageOut(**m) for m in raw_messages]
    except Exception as exc:
        # Don't fail the whole endpoint if messages can't be loaded.
        # Return the session with an empty message list and log the error.
        print(f"  [!] Could not load messages for session {session_id}: {exc}")
        messages = []

    return SessionDetailOut(**session, messages=messages)


# ── Get messages only ─────────────────────────────────────────────────────────

@router.get("/{session_id}/messages", response_model=list[MessageOut])
def get_session_messages(session_id: str):
    """Return only the message history for a session."""
    session = get_session(session_id)
    if not session:
        raise not_found(f"Session '{session_id}' not found.")

    try:
        raw = get_session_messages_for_display(session_id)
        return [MessageOut(**m) for m in raw]
    except Exception as exc:
        print(f"  [!] Could not load messages for session {session_id}: {exc}")
        return []


# ── Delete session ────────────────────────────────────────────────────────────

@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session_endpoint(session_id: str):
    """Permanently delete a session and its message history."""
    session = get_session(session_id)
    if not session:
        raise not_found(f"Session '{session_id}' not found.")
    try:
        delete_session(session_id)
    except Exception as exc:
        raise internal(f"Failed to delete session: {exc}")


# ── Rename session ────────────────────────────────────────────────────────────

@router.patch("/{session_id}/rename", response_model=SessionOut)
def rename_session_endpoint(session_id: str, body: RenameSessionIn):
    """Rename a session title."""
    session = get_session(session_id)
    if not session:
        raise not_found(f"Session '{session_id}' not found.")

    new_title = body.title.strip()
    if not new_title:
        raise bad_request("Title cannot be empty.")

    try:
        rename_session(session_id, new_title)
        updated = get_session(session_id)
        return SessionOut(**updated)
    except HTTPException:
        raise
    except Exception as exc:
        raise internal(f"Failed to rename session: {exc}")