# chat_store.py
"""
chat_store.py — Poshak-AI
──────────────────────────
Manages persistent chat sessions in SQLite.
Stores session metadata (title, timestamps, workspace).
LangGraph's SqliteSaver handles the actual message storage separately.

Database: ./conversations.db  (auto-created on first run)
Tables:
  - sessions  : one row per conversation (metadata only)

Fix: list_sessions() now returns ALL sessions (no message_count filter).
     New chats appear in the sidebar immediately after creation.
     get_most_recent_session() still filters for message_count > 0
     so startup doesn't resume into a blank empty session.
"""

import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from config import CONVERSATIONS_DB, ACTIVE_PROJECT


# ── Schema ────────────────────────────────────────────────────────────────────

_CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id    TEXT PRIMARY KEY,
    workspace     TEXT NOT NULL,
    title         TEXT NOT NULL DEFAULT 'New Chat',
    preview       TEXT NOT NULL DEFAULT '',
    message_count INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_sessions_workspace_updated
ON sessions (workspace, updated_at DESC);
"""


# ── Connection ────────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    """Opens a connection to the conversations DB. Creates file if missing."""
    db_path = Path(CONVERSATIONS_DB)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Creates tables and indexes if they don't exist. Safe to call repeatedly."""
    with _get_conn() as conn:
        conn.execute(_CREATE_SESSIONS_TABLE)
        conn.execute(_CREATE_INDEX)
        conn.commit()


# ── CRUD ──────────────────────────────────────────────────────────────────────

def create_session(workspace: str = ACTIVE_PROJECT) -> str:
    """
    Creates a new chat session and returns its session_id (UUID).
    This UUID is also used as LangGraph's thread_id.
    """
    session_id = str(uuid.uuid4())
    now = _now()
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO sessions
                (session_id, workspace, title, preview, message_count, created_at, updated_at)
            VALUES (?, ?, 'New Chat', '', 0, ?, ?)
            """,
            (session_id, workspace, now, now),
        )
        conn.commit()
    return session_id


def get_session(session_id: str) -> Optional[dict]:
    """Returns a single session by ID, or None if not found."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    return dict(row) if row else None


def list_sessions(workspace: str = ACTIVE_PROJECT, limit: int = 200) -> list[dict]:
    """
    Returns ALL sessions for a workspace, newest first.

    FIX: Removed the  message_count > 0  filter from the original version.
    That filter caused newly created sessions to be invisible in the sidebar
    until the user sent their first message — a confusing UX gap.

    Startup logic (get_most_recent_session) still uses message_count > 0
    so the app doesn't resume into an empty shell on boot.
    """
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT * FROM sessions
            WHERE workspace = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (workspace, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def update_session_after_message(
    session_id: str,
    first_user_message: str,
    is_first_message: bool,
) -> None:
    """
    Called after every user message to keep metadata fresh.
    - Sets title from first user message (truncated to 60 chars)
    - Increments message_count
    - Updates updated_at timestamp
    """
    now = _now()
    title_update = ""
    params: list = []

    if is_first_message:
        title = first_user_message.strip()[:60]
        if len(first_user_message.strip()) > 60:
            title += "…"
        preview = first_user_message.strip()[:120]
        title_update = ", title = ?, preview = ?"
        params = [title, preview]

    with _get_conn() as conn:
        conn.execute(
            f"""
            UPDATE sessions
            SET message_count = message_count + 1,
                updated_at = ?
                {title_update}
            WHERE session_id = ?
            """,
            [now] + params + [session_id],
        )
        conn.commit()


def delete_session(session_id: str) -> None:
    """Deletes session metadata. LangGraph checkpoint is deleted separately."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()


def get_most_recent_session(workspace: str = ACTIVE_PROJECT) -> Optional[dict]:
    """
    Returns the most recently updated session that has messages, or None.
    Used on app startup to resume the last conversation.

    Intentionally keeps the  message_count > 0  filter so the app doesn't
    resume into a blank new-chat shell on startup.
    """
    with _get_conn() as conn:
        row = conn.execute(
            """
            SELECT * FROM sessions
            WHERE workspace = ? AND message_count > 0
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (workspace,),
        ).fetchone()
    return dict(row) if row else None


def rename_session(session_id: str, new_title: str) -> None:
    """Allows manual rename of a session title."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET title = ? WHERE session_id = ?",
            (new_title.strip()[:80], session_id),
        )
        conn.commit()


# ── Date grouping helper (used by UI) ─────────────────────────────────────────

def group_sessions_by_date(sessions: list[dict]) -> dict[str, list[dict]]:
    """
    Groups sessions into buckets for sidebar display:
    Today / Yesterday / This Week / This Month / Older
    """
    now       = datetime.utcnow()
    today     = now.date()
    yesterday = today - timedelta(days=1)
    week_ago  = today - timedelta(days=7)
    month_ago = today - timedelta(days=30)

    groups: dict[str, list] = {
        "Today":      [],
        "Yesterday":  [],
        "This Week":  [],
        "This Month": [],
        "Older":      [],
    }

    for s in sessions:
        try:
            raw       = s.get("updated_at", "")
            raw_clean = raw.split(".")[0]   # strip microseconds if present
            dt        = datetime.strptime(raw_clean, "%Y-%m-%dT%H:%M:%S").date()
        except Exception:
            dt = today

        if dt == today:
            groups["Today"].append(s)
        elif dt == yesterday:
            groups["Yesterday"].append(s)
        elif dt >= week_ago:
            groups["This Week"].append(s)
        elif dt >= month_ago:
            groups["This Month"].append(s)
        else:
            groups["Older"].append(s)

    # Remove empty groups
    return {k: v for k, v in groups.items() if v}


# ── Internal ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")