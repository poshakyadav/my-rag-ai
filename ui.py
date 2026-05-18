# ui.py

"""
ui.py — Poshak-AI
──────────────────
Gradio 6.x web interface with:
  • Dynamic Workspace panel — floating popup modal triggered by sidebar button
  • Per-session workspace — each chat has its own project context
  • Path validation with helpful error messages
  • Live indexing progress in the UI
  • All original sidebar features (session management, rename, delete, URL routing)
  • Proper processing/thinking indicator during AI responses
  • Fixed sidebar chat history display

Workspace states per session:
  "none"     — general chat, no project
  "existing" — one or more existing directories indexed and searched
  "new"      — user is creating a new project (scaffolding mode)
"""

import sys

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import gradio as gr

from config import LLM_MODEL, DB_PATH, validate_config
from agent import (
    ask, warmup, get_active_llm_name,
    get_session_messages_for_display,
    get_executor_for_session,
)
from workspace_manager import (
    WorkspaceConfig,
    ValidationResult,
    validate_paths,
    scaffold_new_project,
    build_session_index,
    load_workspace,
    save_workspace,
    clear_workspace,
    format_workspace_badge,
    GUIDANCE_MESSAGE,
    init_workspace_db,
)
from chat_store import (
    init_db, create_session, list_sessions, delete_session,
    rename_session, update_session_after_message, group_sessions_by_date,
    get_most_recent_session, get_session,
)

# ── Startup ───────────────────────────────────────────────────────────────────

validate_config()
init_db()
init_workspace_db()

print()
print("-" * 56)
print("  Poshak-AI — Loading agent...")
print("-" * 56)
warmup()
print("-" * 56)
print()

_gradio_version = tuple(int(x) for x in gr.__version__.split(".")[:2])
if _gradio_version < (6, 0):
    print(f"  [!] Gradio {gr.__version__} detected. Requires >= 6.0")
    raise SystemExit(1)
print(f"  [+] Gradio {gr.__version__} ready")


# ── Helpers ───────────────────────────────────────────────────────────────────

def db_exists() -> bool:
    return Path(DB_PATH).exists()


def get_chunk_count(collection_name: str = "") -> int:
    try:
        import chromadb
        client = chromadb.PersistentClient(path=DB_PATH)
        col_name = collection_name or "codebase"
        col = client.get_collection(col_name)
        return col.count()
    except Exception:
        return 0


def _sessions_as_json(active_session_id: str = "") -> str:
    """Build sidebar JSON — workspace-agnostic (no workspace filter)."""
    # List ALL sessions regardless of workspace so sidebar always shows history
    all_sessions = list_sessions(workspace="", limit=200)
    grouped = group_sessions_by_date(all_sessions)
    groups = []
    for group_name, group_sessions in grouped.items():
        items = []
        for s in group_sessions:
            items.append({
                "id":     s["session_id"],
                "title":  s["title"],
                "count":  s["message_count"],
                "active": s["session_id"] == active_session_id,
            })
        groups.append({"label": group_name, "sessions": items})
    return json.dumps({"groups": groups, "active_id": active_session_id})


def format_status(session_id: str = "") -> str:
    llm_name = get_active_llm_name() or LLM_MODEL
    sessions = list_sessions(workspace="", limit=200)

    chunks = 0
    if session_id:
        cfg = load_workspace(session_id)
        if cfg.indexed:
            chunks = get_chunk_count(cfg.collection_name)

    return "\n".join([
        f"LLM  : {llm_name}",
        f"DB   : {chunks:,} chunks indexed",
        f"Chats: {len(sessions)} saved",
        f"Time : {datetime.now().strftime('%H:%M:%S')}",
    ])


# ── Workspace panel actions ───────────────────────────────────────────────────

def validate_and_preview(raw_paths: str, mode: str) -> tuple[str, str]:
    if not raw_paths.strip():
        if mode == "none":
            return "✅ General chat mode — no project needed.", "none"
        return (
            "⚠️ Please enter at least one project path before validating.\n\n"
            + _path_guide(mode),
            "empty"
        )

    result = validate_paths(raw_paths, mode=mode)

    if result.ok:
        lines = ["### ✅ Paths look good!\n"]
        for lbl, p in result.valid_paths.items():
            lines.append(f"- **{lbl}**: `{p}`")
        if result.warnings:
            lines.append("\n**Warnings:**")
            for w in result.warnings:
                lines.append(f"- ⚠️ {w}")
        lines.append(
            "\nClick **Apply & Index** to index this project and start chatting."
        )
        return "\n".join(lines), "valid"
    else:
        lines = ["### ❌ Validation failed\n"]
        for e in result.errors:
            lines.append(f"- {e}")
        if result.warnings:
            lines.append("\n**Warnings:**")
            for w in result.warnings:
                lines.append(f"- ⚠️ {w}")
        lines.append("\n---\n" + _path_guide(mode))
        return "\n".join(lines), "invalid"


def _path_guide(mode: str) -> str:
    if mode == "new":
        return (
            "**How to provide a path for a new project:**\n"
            "- Provide the full path to the folder to create:\n"
            "  - Windows: `D:\\Projects\\MyNewApp`\n"
            "  - Linux/Mac: `/home/user/projects/MyNewApp`\n"
            "- The parent directory must exist. The leaf folder will be created."
        )
    return (
        "**How to provide project paths:**\n\n"
        "Single project:\n"
        "```\nD:\\Projects\\MyApp\n```\n\n"
        "Multiple projects (one per line):\n"
        "```\nD:\\Projects\\Backend\nD:\\Projects\\Frontend\n```\n\n"
        "Labeled projects:\n"
        "```\nbackend: D:\\Projects\\Backend\nfrontend: D:\\Projects\\Frontend\n```\n\n"
        "Linux/Mac:\n"
        "```\n/home/user/myapp\n```"
    )


def apply_workspace(
    raw_paths: str,
    mode: str,
    session_id: str,
    history: list,
) -> tuple[list, str, str, str, str]:
    if mode == "none":
        cfg = WorkspaceConfig(session_id=session_id, mode="none")
        save_workspace(cfg)
        msg = (
            "✅ Switched to **general chat mode**. "
            "No project is loaded — I'll answer from general knowledge. "
            "You can paste code snippets directly into the chat."
        )
        history = history + [{"role": "assistant", "content": msg}]
        badge = format_workspace_badge(cfg)
        return history, session_id, _sessions_as_json(session_id), badge, format_status(session_id)

    if not raw_paths.strip():
        err_msg = (
            "⚠️ No path entered. Please type a project path in the text area above.\n\n"
            + _path_guide(mode)
        )
        history = history + [{"role": "assistant", "content": err_msg}]
        return history, session_id, _sessions_as_json(session_id), format_workspace_badge(
            load_workspace(session_id)
        ), format_status(session_id)

    result = validate_paths(raw_paths, mode=mode)

    if not result.ok:
        lines = ["❌ **Path validation failed.** Please fix the errors below:\n"]
        for e in result.errors:
            lines.append(f"- {e}")
        lines.append("\n" + _path_guide(mode))
        err_msg = "\n".join(lines)
        history = history + [{"role": "assistant", "content": err_msg}]
        return history, session_id, _sessions_as_json(session_id), format_workspace_badge(
            load_workspace(session_id)
        ), format_status(session_id)

    if mode == "new":
        path_str = list(result.valid_paths.values())[0]
        ok, scaffold_msg = scaffold_new_project(path_str)
        if not ok:
            history = history + [{"role": "assistant", "content": scaffold_msg}]
            return history, session_id, _sessions_as_json(session_id), format_workspace_badge(
                load_workspace(session_id)
            ), format_status(session_id)
        history = history + [{"role": "assistant", "content": scaffold_msg}]

    label = ", ".join(result.valid_paths.keys())
    cfg = WorkspaceConfig(
        session_id=session_id,
        mode=mode,
        paths=result.valid_paths,
        label=label,
        indexed=False,
    )
    save_workspace(cfg)

    progress_msgs: list[str] = []

    def _progress(msg: str):
        progress_msgs.append(msg)

    success, index_msg = build_session_index(cfg, progress_cb=_progress)

    if not success:
        lines = [
            "❌ **Indexing failed.**\n",
            index_msg,
            "\nPlease check the paths and try again.",
        ]
        history = history + [{"role": "assistant", "content": "\n".join(lines)}]
        return history, session_id, _sessions_as_json(session_id), format_workspace_badge(cfg), format_status(session_id)

    cfg = load_workspace(session_id)

    confirm_lines = [
        f"🎉 **Workspace ready!**\n",
        index_msg,
        "",
    ]
    if result.warnings:
        confirm_lines.append("**Warnings:**")
        for w in result.warnings:
            confirm_lines.append(f"- ⚠️ {w}")
        confirm_lines.append("")

    confirm_lines += [
        "**You can now ask me:**",
        "- *Explain the overall project structure*",
        "- *How is authentication implemented?*",
        "- *Write a new REST endpoint following existing style*",
        "- *Find potential bugs or missing error handling*",
    ]

    history = history + [{"role": "assistant", "content": "\n".join(confirm_lines)}]
    badge = format_workspace_badge(cfg)
    return history, session_id, _sessions_as_json(session_id), badge, format_status(session_id)


def clear_workspace_action(session_id: str, history: list):
    clear_workspace(session_id)
    msg = (
        "🗑️ Workspace cleared. I'm now in **general chat mode**.\n\n"
        "To load a new project, use the Workspace button in the sidebar."
    )
    history = history + [{"role": "assistant", "content": msg}]
    cfg = load_workspace(session_id)
    badge = format_workspace_badge(cfg)
    return history, badge, format_status(session_id)


# ── Chat / session actions ────────────────────────────────────────────────────

def respond(message: str, history: list, session_id: str):
    if not message.strip():
        yield "", history, session_id, _sessions_as_json(session_id)
        return

    is_first = (len(history) == 0)
    # Add user message immediately
    history = history + [{"role": "user", "content": message}]
    # Show thinking indicator as assistant placeholder
    history_with_thinking = history + [{"role": "assistant", "content": "⏳ _Thinking..._"}]
    yield "", history_with_thinking, session_id, _sessions_as_json(session_id)

    cfg = load_workspace(session_id)
    executor = get_executor_for_session(session_id, cfg)
    response = ask(executor, message, session_id=session_id)

    # Replace thinking placeholder with real response
    history = history + [{"role": "assistant", "content": response}]
    update_session_after_message(session_id, message, is_first_message=is_first)
    yield "", history, session_id, _sessions_as_json(session_id)


def new_chat_action(current_session_id: str, current_history: list):
    """
    Smart new-chat: if the current session is blank (no user messages sent),
    reuse it instead of spawning a duplicate empty shell.
    Always opens the (new or reused) session in the main content area immediately.
    """
    # Check if current session is already empty (no messages ever sent)
    if current_session_id:
        session_meta = get_session(current_session_id)
        is_empty = session_meta and session_meta.get("message_count", 0) == 0
        if is_empty:
            # Reuse the existing blank session — just refresh the view
            cfg = load_workspace(current_session_id)
            sidebar_json = _sessions_as_json(current_session_id)
            welcome = [{"role": "assistant", "content": GUIDANCE_MESSAGE}]
            badge = format_workspace_badge(cfg)
            return welcome, current_session_id, sidebar_json, current_session_id, badge

    # Create a fresh session
    session_id = create_session()
    cfg = WorkspaceConfig(session_id=session_id, mode="none")
    save_workspace(cfg)
    sidebar_json = _sessions_as_json(session_id)
    welcome = [{"role": "assistant", "content": GUIDANCE_MESSAGE}]
    badge = format_workspace_badge(cfg)
    return welcome, session_id, sidebar_json, session_id, badge


def load_session_action(session_id: str):
    if not session_id or session_id in ("__none__", ""):
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
    messages = get_session_messages_for_display(session_id)
    sidebar_json = _sessions_as_json(session_id)
    cfg = load_workspace(session_id)
    badge = format_workspace_badge(cfg)
    return messages, session_id, sidebar_json, session_id, badge


def delete_session_action(session_id_to_delete: str, current_session_id: str):
    if session_id_to_delete:
        delete_session(session_id_to_delete)
    if session_id_to_delete == current_session_id:
        new_id = create_session()
        cfg = WorkspaceConfig(session_id=new_id, mode="none")
        save_workspace(cfg)
        messages = [{"role": "assistant", "content": GUIDANCE_MESSAGE}]
    else:
        new_id = current_session_id
        messages = get_session_messages_for_display(new_id)
    sidebar_json = _sessions_as_json(new_id)
    cfg = load_workspace(new_id)
    badge = format_workspace_badge(cfg)
    return messages, new_id, sidebar_json, new_id, badge


def rename_session_action(session_id: str, new_title: str) -> str:
    if not new_title.strip():
        return _sessions_as_json(session_id)
    rename_session(session_id, new_title.strip())
    return _sessions_as_json(session_id)


def on_page_load(request: gr.Request, saved_session_id: str):
    url_session_id = None
    try:
        if request and hasattr(request, "query_params"):
            url_session_id = request.query_params.get("s", None)
    except Exception:
        pass

    for candidate in [url_session_id, saved_session_id]:
        if candidate:
            try:
                session = get_session(candidate)
                if session:
                    messages = get_session_messages_for_display(candidate)
                    if not messages:
                        messages = [{"role": "assistant", "content": GUIDANCE_MESSAGE}]
                    sidebar_json = _sessions_as_json(candidate)
                    cfg = load_workspace(candidate)
                    badge = format_workspace_badge(cfg)
                    return messages, candidate, sidebar_json, candidate, badge
            except Exception:
                pass

    recent = get_most_recent_session(workspace="")
    if recent:
        sid = recent["session_id"]
        messages = get_session_messages_for_display(sid)
        if not messages:
            messages = [{"role": "assistant", "content": GUIDANCE_MESSAGE}]
        sidebar_json = _sessions_as_json(sid)
        cfg = load_workspace(sid)
        badge = format_workspace_badge(cfg)
        return messages, sid, sidebar_json, sid, badge

    sid = create_session()
    cfg = WorkspaceConfig(session_id=sid, mode="none")
    save_workspace(cfg)
    sidebar_json = _sessions_as_json(sid)
    badge = format_workspace_badge(cfg)
    return [{"role": "assistant", "content": GUIDANCE_MESSAGE}], sid, sidebar_json, sid, badge


def export_chat(history: list):
    if not history:
        return "No history to export.", gr.update(visible=False)
    lines = [f"# Poshak-AI Export\n_{datetime.now().strftime('%Y-%m-%d %H:%M')}_\n"]
    for msg in history:
        role    = msg.get("role", "?").capitalize()
        content = msg.get("content", "")
        lines.append(f"**{role}:** {content}\n")
    return "\n".join(lines), gr.update(visible=True)


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap');

:root {
    --bg:        #0d0d0d;
    --bg-sidebar:#111111;
    --bg-card:   #1a1a1a;
    --bg-hover:  #1e1e1e;
    --bg-active: #1a2e1a;
    --border:    #252525;
    --accent:    #1D9E75;
    --accent-dim:#0f6e52;
    --text:      #e2e2e2;
    --text-muted:#666;
    --text-dim:  #444;
    --danger:    #e05252;
    --warn:      #e09a2e;
    --success:   #2ebc85;
    --radius:    9px;
    --font:      'DM Sans', sans-serif;
    --mono:      'JetBrains Mono', monospace;
}
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
}
footer, .footer { display: none !important; }
.gradio-container { padding: 0 !important; }
.contain { max-width: 100% !important; }
aside, [class*="sidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Hidden elements (CSS-only) ──────────────────────────────────────────────*/
.sidebar-data-wrap, .sidebar-data-wrap * {
    position: fixed !important; top: -9999px !important; left: -9999px !important;
    width: 1px !important; height: 1px !important;
    overflow: hidden !important; opacity: 0 !important;
    pointer-events: none !important; z-index: -9999 !important;
}
#hidden-actions {
    position: fixed !important; top: -9999px !important; left: -9999px !important;
    width: 1px !important; height: 1px !important;
    overflow: hidden !important; opacity: 0 !important;
    pointer-events: none !important; z-index: -9999 !important;
}
#hidden-actions button { pointer-events: auto !important; }

/* ── Sidebar ─────────────────────────────────────────────────────────────────*/
#sidebar-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 12px 10px; border-bottom: 1px solid var(--border);
}
#sidebar-logo {
    font-weight: 700; font-size: 14px; color: var(--accent);
    display: flex; align-items: center; gap: 7px;
}
#sidebar-header-actions {
    display: flex; align-items: center; gap: 6px;
}
#new-chat-sidebar-btn, #workspace-open-btn {
    background: var(--accent); border: none; border-radius: 7px;
    width: 28px; height: 28px; display: flex; align-items: center;
    justify-content: center; color: #fff; font-size: 15px;
    cursor: pointer; transition: background 0.18s; flex-shrink: 0; line-height: 1;
}
#new-chat-sidebar-btn:hover, #workspace-open-btn:hover { background: var(--accent-dim); }
#workspace-open-btn {
    background: #1e3a2e; font-size: 13px;
    border: 1px solid var(--accent-dim);
}
#workspace-open-btn:hover { background: var(--accent); border-color: var(--accent); }
#sidebar-search-wrap { padding: 9px 10px; border-bottom: 1px solid var(--border); }
#sidebar-search-input {
    width: 100%; background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 7px; padding: 6px 10px; color: var(--text);
    font-size: 12px; font-family: var(--font); outline: none; transition: border-color 0.18s;
}
#sidebar-search-input:focus { border-color: var(--accent); }
#sidebar-search-input::placeholder { color: var(--text-dim); }
#sessions-list {
    overflow-y: auto; padding: 4px 0 12px;
    scrollbar-width: thin; scrollbar-color: var(--border) transparent;
    height: calc(100vh - 200px);
}
#sessions-list::-webkit-scrollbar { width: 3px; }
#sessions-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Session rows ────────────────────────────────────────────────────────────*/
.session-group-label {
    font-size: 10px; font-weight: 600; color: var(--text-dim);
    letter-spacing: 0.9px; text-transform: uppercase;
    padding: 12px 13px 3px; user-select: none;
}
.session-row {
    display: flex; align-items: center; margin: 1px 5px;
    border-radius: var(--radius); cursor: pointer; transition: background 0.12s;
}
.session-row:hover { background: var(--bg-hover); }
.session-row.active { background: var(--bg-active); border-left: 2px solid var(--accent); }
.session-row.active .session-title { color: var(--accent); }
.session-main { flex: 1; padding: 8px 5px 8px 9px; min-width: 0; user-select: none; }
.session-title {
    font-size: 12.5px; font-weight: 500; color: var(--text);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    line-height: 1.3; max-width: 182px;
}
.session-meta { font-size: 10px; color: var(--text-dim); margin-top: 1px; }
.session-dots {
    background: none; border: none; color: var(--text-muted);
    padding: 4px 7px; border-radius: 6px; cursor: pointer;
    font-size: 15px; line-height: 1; opacity: 0;
    transition: opacity 0.12s, background 0.12s; flex-shrink: 0;
}
.session-row:hover .session-dots { opacity: 1; }
.session-dots:hover { background: var(--bg-card); color: var(--text); }

/* ── Context menu ────────────────────────────────────────────────────────────*/
.ctx-popup {
    position: fixed; background: #1c1c1c; border: 1px solid #333;
    border-radius: 9px; box-shadow: 0 10px 36px rgba(0,0,0,0.55);
    z-index: 99999; min-width: 148px; overflow: hidden;
    animation: popIn 0.11s ease-out;
}
@keyframes popIn { from{opacity:0;transform:scale(0.91) translateY(-3px)} to{opacity:1;transform:scale(1) translateY(0)} }
.ctx-popup button {
    display: flex; align-items: center; gap: 9px;
    width: 100%; text-align: left; background: none; border: none;
    padding: 9px 14px; font-size: 12.5px; font-family: var(--font);
    cursor: pointer; color: var(--text); transition: background 0.1s;
}
.ctx-popup button:hover { background: var(--bg-hover); }
.ctx-popup button.danger { color: var(--danger); }
.ctx-popup button.danger:hover { background: #221010; }
.ctx-divider { height: 1px; background: var(--border); margin: 2px 0; }

/* ── Rename popup ────────────────────────────────────────────────────────────*/
#rename-popup {
    display: none; position: fixed; background: #1c1c1c;
    border: 1px solid #333; border-radius: 10px;
    box-shadow: 0 10px 36px rgba(0,0,0,0.6); z-index: 99999;
    padding: 13px; min-width: 230px;
}
#rename-input {
    width: 100%; background: var(--bg-card); border: 1px solid var(--accent);
    border-radius: 6px; padding: 7px 10px; color: var(--text);
    font-size: 12.5px; font-family: var(--font); outline: none; margin-bottom: 9px;
}
#rename-actions { display: flex; gap: 6px; }
#rename-actions button {
    flex: 1; padding: 7px; border-radius: 7px; border: none;
    font-size: 12px; font-family: var(--font); font-weight: 500; cursor: pointer;
}
#rename-save { background: var(--accent); color: #fff; }
#rename-cancel { background: var(--bg-hover); color: var(--text); border: 1px solid var(--border); }

/* ── Workspace Modal Overlay ─────────────────────────────────────────────────*/
#workspace-modal-overlay {
    display: none;
    position: fixed; inset: 0;
    background: rgba(0,0,0,0.72);
    z-index: 88888;
    backdrop-filter: blur(3px);
    animation: fadeOverlay 0.18s ease;
}
@keyframes fadeOverlay { from{opacity:0} to{opacity:1} }
#workspace-modal-overlay.open { display: block; }

#workspace-modal {
    position: fixed;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%) scale(0.96);
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 14px;
    box-shadow: 0 24px 80px rgba(0,0,0,0.8), 0 0 0 1px rgba(29,158,117,0.08);
    z-index: 99999;
    width: 480px;
    max-width: calc(100vw - 32px);
    max-height: 90vh;
    overflow-y: auto;
    animation: modalSlide 0.2s cubic-bezier(0.34,1.56,0.64,1) forwards;
    padding: 0;
}
@keyframes modalSlide {
    from { opacity:0; transform: translate(-50%,-48%) scale(0.93); }
    to   { opacity:1; transform: translate(-50%,-50%) scale(1); }
}
#workspace-modal-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 20px 14px;
    border-bottom: 1px solid #1e1e1e;
}
#workspace-modal-title {
    font-size: 15px; font-weight: 700; color: var(--text);
    display: flex; align-items: center; gap: 8px;
}
#workspace-modal-close {
    background: none; border: none;
    color: var(--text-muted); font-size: 20px; line-height: 1;
    cursor: pointer; padding: 2px 6px; border-radius: 6px;
    transition: background 0.14s, color 0.14s;
}
#workspace-modal-close:hover { background: #2a1010; color: var(--danger); }

#workspace-modal-body { padding: 16px 20px 20px; }

/* Mode selector */
#ws-modal-mode-row {
    display: flex; gap: 6px; margin-bottom: 14px;
}
.ws-mode-btn {
    flex: 1; padding: 7px 4px; font-size: 11px;
    background: var(--bg-card); border: 1px solid var(--border);
    color: var(--text-muted); border-radius: 7px; cursor: pointer;
    transition: all 0.15s; font-family: var(--font); text-align: center;
    font-weight: 500;
}
.ws-mode-btn.active {
    background: var(--accent); border-color: var(--accent);
    color: #fff; font-weight: 600;
}
.ws-mode-btn:hover:not(.active) { border-color: var(--accent); color: var(--text); }

/* Current workspace badge */
#ws-current-badge {
    display: flex; align-items: center; gap: 8px;
    background: #0f1f18; border: 1px solid #1a3028;
    border-radius: 8px; padding: 8px 12px; margin-bottom: 12px;
    font-size: 11px; color: var(--text-muted);
}
#ws-current-badge .badge-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--accent); flex-shrink: 0;
    box-shadow: 0 0 6px var(--accent);
}

/* Path textarea */
#ws-path-label {
    font-size: 11px; font-weight: 600; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 0.7px; margin-bottom: 6px;
}
#ws-path-input {
    width: 100%; background: #0d0d0d; border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 12px; color: var(--text);
    font-size: 12px; font-family: var(--mono); outline: none;
    resize: vertical; min-height: 80px; transition: border-color 0.18s;
    margin-bottom: 10px; line-height: 1.55;
}
#ws-path-input:focus { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(29,158,117,0.1); }
#ws-path-input::placeholder { color: var(--text-dim); font-family: var(--font); font-size: 11px; }
#ws-path-input:disabled { opacity: 0.35; cursor: not-allowed; }

/* Status strip */
#ws-status-strip {
    min-height: 18px; font-size: 11px; border-radius: 6px;
    padding: 4px 8px; margin-bottom: 12px; transition: all 0.2s;
    display: none;
}
#ws-status-strip.visible { display: block; }
#ws-status-strip.valid   { color: var(--success); background: #0a1f12; border: 1px solid #1a3f28; }
#ws-status-strip.invalid { color: var(--danger);  background: #1f0a0a; border: 1px solid #3f1a1a; }
#ws-status-strip.indexing{ color: var(--warn);    background: #1f180a; border: 1px solid #3f2f1a; }

/* Action buttons */
#ws-modal-actions { display: flex; flex-direction: column; gap: 7px; }
.ws-btn {
    width: 100%; padding: 9px; border-radius: 8px; border: none;
    font-size: 12.5px; font-family: var(--font); font-weight: 600;
    cursor: pointer; transition: background 0.15s, transform 0.1s;
}
.ws-btn:active { transform: scale(0.98); }
.ws-btn-primary { background: var(--accent); color: #fff; }
.ws-btn-primary:hover { background: var(--accent-dim); }
.ws-btn-secondary {
    background: var(--bg-card); color: var(--text-muted);
    border: 1px solid var(--border);
}
.ws-btn-secondary:hover { border-color: var(--danger); color: var(--danger); }

/* ── Processing / Thinking indicator ─────────────────────────────────────────*/
#processing-bar {
    display: none;
    position: fixed; bottom: 0; left: 0; right: 0;
    height: 3px; z-index: 77777;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    background-size: 200% 100%;
    animation: shimmerBar 1.5s linear infinite;
}
#processing-bar.active { display: block; }
@keyframes shimmerBar {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
#processing-toast {
    display: none;
    position: fixed; bottom: 22px; left: 50%; transform: translateX(-50%);
    background: #1a1a1a; border: 1px solid var(--border);
    border-radius: 24px; padding: 8px 18px;
    font-size: 12px; color: var(--text-muted);
    z-index: 77778; gap: 8px; align-items: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5);
    animation: fadeOverlay 0.2s ease;
}
#processing-toast.active { display: flex; }
.processing-spinner {
    width: 12px; height: 12px; border-radius: 50%;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    animation: spin 0.7s linear infinite; flex-shrink: 0;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Top bar ─────────────────────────────────────────────────────────────────*/
#topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 11px 20px; border-bottom: 1px solid var(--border);
    background: var(--bg); flex-shrink: 0;
}
#topbar-title { font-weight: 700; font-size: 14px; color: var(--text); }
#topbar-ws {
    font-family: var(--mono); font-size: 10.5px; color: var(--text-dim);
    background: var(--bg-card); border: 1px solid var(--border);
    padding: 3px 9px; border-radius: 5px; max-width: 340px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* ── Chat ────────────────────────────────────────────────────────────────────*/
.gradio-chatbot, [data-testid="chatbot"] { background: var(--bg) !important; border: none !important; }
.message.user > div {
    background: #172817 !important; border: 1px solid #243824 !important;
    border-radius: 12px 12px 4px 12px !important; color: var(--text) !important;
}
.message.bot > div {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: 12px 12px 12px 4px !important; color: var(--text) !important;
}

/* ── Input ───────────────────────────────────────────────────────────────────*/
.gr-textbox textarea {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important; color: var(--text) !important;
    font-family: var(--font) !important; font-size: 13.5px !important;
}
.gr-textbox textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(29,158,117,0.12) !important; outline: none !important;
}
.send-btn {
    background: var(--accent) !important; color: #fff !important;
    border-radius: var(--radius) !important; font-weight: 600 !important;
    font-family: var(--font) !important; border: none !important;
    transition: background 0.18s !important;
}
.send-btn:hover { background: var(--accent-dim) !important; }
.ex-btn {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: 7px !important;
    font-size: 11px !important; font-family: var(--font) !important;
    text-align: left !important; white-space: normal !important;
    line-height: 1.35 !important; padding: 7px 9px !important;
    margin-bottom: 3px !important; transition: border-color 0.14s, background 0.14s !important;
    width: 100% !important;
}
.ex-btn:hover { border-color: var(--accent) !important; background: var(--bg-hover) !important; }
.gr-accordion {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important; margin-bottom: 7px !important;
}
.mono-box textarea {
    font-family: var(--mono) !important; font-size: 11px !important;
    color: var(--text-muted) !important; background: var(--bg) !important;
    border-color: var(--border) !important;
}
"""

# ── JavaScript ────────────────────────────────────────────────────────────────

JS_HEAD = r"""
<script>
'use strict';

let _ctxPopup = null, _renameId = null, _lastJson = '', _navigating = false, _watchStarted = false;
let _processingActive = false, _popupsInjected = false;

// ── Modal & Popup HTML — injected into document.body at init ─────────────────
// WHY: Gradio's sidebar <aside> has CSS transform which creates a stacking
// context. Any position:fixed child of it is positioned relative to that
// context, not the viewport, causing the modal to appear offset/cropped.
// Injecting directly into <body> escapes that stacking context entirely.
const MODAL_HTML = `
<div id="workspace-modal-overlay">
  <div id="workspace-modal" role="dialog" aria-modal="true" aria-label="Workspace settings">
    <div id="workspace-modal-header">
      <div id="workspace-modal-title">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#1D9E75" stroke-width="2">
          <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
        </svg>
        Workspace
      </div>
      <button id="workspace-modal-close" onclick="closeWorkspaceModal()" title="Close">\u2715</button>
    </div>
    <div id="workspace-modal-body">
      <div id="ws-current-badge">
        <div class="badge-dot"></div>
        <span id="ws-current-badge-text">No project loaded</span>
      </div>
      <div id="ws-modal-mode-row">
        <button class="ws-mode-btn active" data-mode="existing" onclick="setWsMode('existing')">\uD83D\uDCC2 Existing</button>
        <button class="ws-mode-btn" data-mode="new" onclick="setWsMode('new')">\uD83C\uDD95 New</button>
        <button class="ws-mode-btn" data-mode="none" onclick="setWsMode('none')">\uD83D\uDCAC None</button>
      </div>
      <div id="ws-path-label">Project Path(s)</div>
      <textarea id="ws-path-input"
        placeholder="Project path(s)&#10;e.g. D:\\\\Projects\\\\Backend&#10;D:\\\\Projects\\\\Frontend&#10;&#10;Labeled: backend: D:\\\\Projects\\\\Api"
        oninput="writeHidden('ws-paths-state', this.value); clearWsStatus();"></textarea>
      <div id="ws-status-strip"></div>
      <div id="ws-modal-actions">
        <button class="ws-btn ws-btn-primary" onclick="triggerValidate()">\uD83D\uDD0D Validate Paths</button>
        <button class="ws-btn ws-btn-primary" onclick="triggerApply()">\u26A1 Apply &amp; Index Project</button>
        <button class="ws-btn ws-btn-secondary" onclick="triggerClear()">\uD83D\uDDD1\uFE0F Clear Workspace</button>
      </div>
      <div style="margin-top:14px;padding:10px 12px;background:#0a0a0a;border-radius:8px;border:1px solid #1a1a1a;">
        <div style="font-size:10px;color:var(--text-dim);line-height:1.7;">
          <strong style="color:var(--text-muted);">Tips:</strong><br>
          \u2022 One path per line, or use <code style="color:var(--accent)">label: /path</code> format<br>
          \u2022 Works with Java, TypeScript, Python, React &amp; more<br>
          \u2022 Re-index after changing project files
        </div>
      </div>
    </div>
  </div>
</div>

<div id="rename-popup">
  <input id="rename-input" type="text" placeholder="New title..." />
  <div id="rename-actions">
    <button id="rename-save" onclick="commitRename()">Save</button>
    <button id="rename-cancel" onclick="hideRename()">Cancel</button>
  </div>
</div>
`;

function injectBodyPopups() {
    if (_popupsInjected) return;
    _popupsInjected = true;
    const wrapper = document.createElement('div');
    wrapper.id = 'poshak-body-popups';
    // Use innerHTML via template to avoid XSS but we control this fully
    wrapper.innerHTML = MODAL_HTML;
    document.body.appendChild(wrapper);
    // Wire overlay click-outside
    const overlay = document.getElementById('workspace-modal-overlay');
    if (overlay) {
        overlay.addEventListener('click', function(e) {
            if (e.target === overlay) closeWorkspaceModal();
        });
    }
}

// ── URL routing ───────────────────────────────────────────────────────────────
function pushSessionUrl(sessionId) {
    if (!sessionId) return;
    const target = '/?s=' + encodeURIComponent(sessionId);
    if (window.location.pathname + window.location.search !== target)
        history.pushState({ sessionId }, '', target);
}
window.addEventListener('popstate', function(e) {
    const sid = new URLSearchParams(window.location.search).get('s');
    if (sid) { _navigating = true; triggerLoad(sid); setTimeout(function() { _navigating = false; }, 1200); }
});

// ── Utils ─────────────────────────────────────────────────────────────────────
function closeCtx() { if (_ctxPopup) { _ctxPopup.remove(); _ctxPopup = null; } }
function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

// ── Processing indicator ──────────────────────────────────────────────────────
function showProcessing(label) {
    _processingActive = true;
    let bar = document.getElementById('processing-bar');
    let toast = document.getElementById('processing-toast');
    if (!bar) {
        bar = document.createElement('div'); bar.id = 'processing-bar';
        document.body.appendChild(bar);
    }
    if (!toast) {
        toast = document.createElement('div'); toast.id = 'processing-toast';
        toast.innerHTML = '<div class="processing-spinner"></div><span id="processing-label">Processing...</span>';
        document.body.appendChild(toast);
    }
    bar.classList.add('active');
    toast.classList.add('active');
    const lbl = document.getElementById('processing-label');
    if (lbl) lbl.textContent = label || 'Thinking...';
    const sendBtn = document.querySelector('.send-btn');
    if (sendBtn) { sendBtn.disabled = true; sendBtn.style.opacity = '0.5'; }
}
function hideProcessing() {
    _processingActive = false;
    const bar = document.getElementById('processing-bar');
    const toast = document.getElementById('processing-toast');
    if (bar) bar.classList.remove('active');
    if (toast) toast.classList.remove('active');
    const sendBtn = document.querySelector('.send-btn');
    if (sendBtn) { sendBtn.disabled = false; sendBtn.style.opacity = ''; }
}

// ── Sidebar renderer ──────────────────────────────────────────────────────────
function renderSidebar(jsonStr) {
    if (!jsonStr || jsonStr === _lastJson) return;
    _lastJson = jsonStr;
    let data; try { data = JSON.parse(jsonStr); } catch(e) { return; }
    const container = document.getElementById('sessions-list');
    if (!container) return;
    if (data.active_id && !_navigating) pushSessionUrl(data.active_id);
    const search = (document.getElementById('sidebar-search-input')?.value || '').toLowerCase().trim();
    container.innerHTML = '';
    const groups = data.groups || [];
    if (!groups.length) {
        container.innerHTML = '<div style="color:var(--text-dim);font-size:11px;padding:16px 14px;">No chats yet. Click + to start.</div>';
        return;
    }
    for (const group of groups) {
        const rows = group.sessions.filter(s => !search || s.title.toLowerCase().includes(search));
        if (!rows.length) continue;
        const grpEl = document.createElement('div');
        const lbl = document.createElement('div');
        lbl.className = 'session-group-label'; lbl.textContent = group.label; grpEl.appendChild(lbl);
        for (const s of rows) {
            const row = document.createElement('div');
            row.className = 'session-row' + (s.active ? ' active' : ''); row.dataset.id = s.id;
            const main = document.createElement('div'); main.className = 'session-main';
            const metaText = s.count > 0 ? s.count + ' msg' + (s.count !== 1 ? 's' : '') : 'New chat';
            main.innerHTML = '<div class="session-title">' + esc(s.title) + '</div><div class="session-meta">' + metaText + '</div>';
            const dots = document.createElement('button');
            dots.className = 'session-dots'; dots.title = 'Options'; dots.textContent = '\u22EE';
            main.addEventListener('click', function() { closeCtx(); triggerLoad(s.id); });
            dots.addEventListener('click', function(e) { e.stopPropagation(); closeCtx(); showCtx(e, s.id, s.title); });
            row.appendChild(main); row.appendChild(dots); grpEl.appendChild(row);
        }
        container.appendChild(grpEl);
    }
}
function forceRender(jsonStr) { _lastJson = ''; renderSidebar(jsonStr); }

// ── Context menu ──────────────────────────────────────────────────────────────
function showCtx(e, id, title) {
    const popup = document.createElement('div'); popup.className = 'ctx-popup';
    const renBtn = document.createElement('button');
    renBtn.innerHTML = '&#x270F;&#xFE0F;&nbsp; Rename';
    renBtn.onclick = function() { closeCtx(); showRename(id, title, e.clientX, e.clientY); };
    const divider = document.createElement('div'); divider.className = 'ctx-divider';
    const delBtn = document.createElement('button'); delBtn.className = 'danger';
    delBtn.innerHTML = '&#x1F5D1;&#xFE0F;&nbsp; Delete';
    delBtn.onclick = function() { closeCtx(); triggerDelete(id); };
    popup.appendChild(renBtn); popup.appendChild(divider); popup.appendChild(delBtn);
    document.body.appendChild(popup); _ctxPopup = popup;
    const W = window.innerWidth, H = window.innerHeight;
    let left = e.clientX + 6, top = e.clientY;
    if (left + 160 > W) left = e.clientX - 166;
    if (top + 90 > H) top = H - 95;
    popup.style.left = left + 'px'; popup.style.top = top + 'px';
}

// ── Rename popup ──────────────────────────────────────────────────────────────
function showRename(id, title, x, y) {
    _renameId = id;
    const popup = document.getElementById('rename-popup');
    const input = document.getElementById('rename-input');
    if (!popup || !input) return;
    input.value = title;
    const W = window.innerWidth, H = window.innerHeight;
    popup.style.left = Math.min(x, W - 250) + 'px'; popup.style.top = Math.min(y, H - 130) + 'px';
    popup.style.display = 'block'; input.focus(); input.select();
}
function hideRename() {
    const p = document.getElementById('rename-popup');
    if (p) p.style.display = 'none';
    _renameId = null;
}
function commitRename() {
    if (!_renameId) return;
    const title = (document.getElementById('rename-input')?.value || '').trim();
    if (!title) return;
    writeHidden('hid-rename-id', _renameId);
    writeHidden('hid-rename-title', title);
    setTimeout(function() { clickHidden('hid-rename-btn'); hideRename(); }, 60);
}

// ── Workspace Modal ───────────────────────────────────────────────────────────
function openWorkspaceModal() {
    injectBodyPopups(); // ensure modal is in body (idempotent)
    const overlay = document.getElementById('workspace-modal-overlay');
    if (!overlay) return;
    overlay.classList.add('open');
    document.body.style.overflow = 'hidden';
    // Sync current badge text into modal
    const topbarWs = document.getElementById('topbar-ws');
    const badge = document.getElementById('ws-current-badge-text');
    if (topbarWs && badge) badge.textContent = topbarWs.textContent || 'No project loaded';
}
function closeWorkspaceModal() {
    const overlay = document.getElementById('workspace-modal-overlay');
    if (overlay) overlay.classList.remove('open');
    document.body.style.overflow = '';
}

function setWsMode(mode) {
    document.querySelectorAll('.ws-mode-btn').forEach(function(b) {
        b.classList.toggle('active', b.dataset.mode === mode);
    });
    const pathArea = document.getElementById('ws-path-input');
    if (pathArea) {
        pathArea.disabled = (mode === 'none');
        if (mode === 'none') {
            pathArea.placeholder = 'No project \u2014 general chat mode';
            pathArea.value = '';
        } else if (mode === 'new') {
            pathArea.placeholder = 'Path for new project\ne.g. D:\\Projects\\MyNewApp';
        } else {
            pathArea.placeholder = 'Project path(s)\ne.g. D:\\Projects\\Backend\nD:\\Projects\\Frontend\n\nLabeled: backend: D:\\Projects\\Api';
        }
    }
    writeHidden('ws-mode-state', mode);
    clearWsStatus();
}

function clearWsStatus() {
    const strip = document.getElementById('ws-status-strip');
    if (strip) { strip.className = ''; strip.textContent = ''; }
}

function setWsStatus(msg, type) {
    const strip = document.getElementById('ws-status-strip');
    if (!strip) return;
    strip.textContent = msg;
    strip.className = 'visible ' + (type || '');
}

// ── Gradio DOM bridge ─────────────────────────────────────────────────────────
function findTextarea(elemId) {
    const wrap = document.getElementById(elemId); if (!wrap) return null;
    return wrap.querySelector('textarea') || wrap.querySelector('input[type="text"]');
}
function writeHidden(elemId, value) {
    const ta = findTextarea(elemId);
    if (!ta) { setTimeout(function() { writeHidden(elemId, value); }, 200); return; }
    const proto = ta.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, 'value');
    if (setter && setter.set) setter.set.call(ta, value); else ta.value = value;
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    ta.dispatchEvent(new Event('change', { bubbles: true }));
}
function clickHidden(elemId) {
    const el = document.getElementById(elemId);
    if (!el) { setTimeout(function() { clickHidden(elemId); }, 200); return; }
    const btn = el.tagName === 'BUTTON' ? el : el.querySelector('button');
    if (btn) btn.click(); else el.click();
}
function triggerNewChat()  { clickHidden('hid-new-btn'); }
function triggerLoad(id)   { writeHidden('hid-load-id', id); setTimeout(function() { clickHidden('hid-load-btn'); }, 80); }
function triggerDelete(id) { writeHidden('hid-delete-id', id); setTimeout(function() { clickHidden('hid-delete-btn'); }, 80); }

function triggerValidate() {
    const paths = document.getElementById('ws-path-input')?.value || '';
    writeHidden('ws-paths-state', paths);
    setWsStatus('\uD83D\uDD0D Validating paths...', 'indexing');
    setTimeout(function() { clickHidden('hid-ws-validate-btn'); }, 80);
}
function triggerApply() {
    const paths = document.getElementById('ws-path-input')?.value || '';
    writeHidden('ws-paths-state', paths);
    setWsStatus('\u26A1 Indexing project\u2026 This may take a minute.', 'indexing');
    showProcessing('Indexing project...');
    closeWorkspaceModal();
    setTimeout(function() { clickHidden('hid-ws-apply-btn'); }, 120);
}
function triggerClear() {
    closeWorkspaceModal();
    setTimeout(function() { clickHidden('hid-ws-clear-btn'); }, 150);
}

// ── sidebar_data watcher ──────────────────────────────────────────────────────
function watchSidebarData() {
    if (_watchStarted) return;
    const STORE_ID = 'sidebar-data-store';
    function attempt(delay) {
        const wrap = document.getElementById(STORE_ID);
        if (!wrap) { setTimeout(function() { attempt(Math.min(delay * 1.5, 1000)); }, delay); return; }
        const ta = wrap.querySelector('textarea') || wrap.querySelector('input');
        if (!ta) { setTimeout(function() { attempt(Math.min(delay * 1.5, 800)); }, delay); return; }
        _watchStarted = true;
        if (ta.value) renderSidebar(ta.value);
        ta.addEventListener('input',  function() { renderSidebar(ta.value); });
        ta.addEventListener('change', function() { renderSidebar(ta.value); });
        new MutationObserver(function() {
            const el = wrap.querySelector('textarea') || wrap.querySelector('input');
            const val = el ? el.value : '';
            if (val) renderSidebar(val);
        }).observe(wrap, { attributes: true, childList: true, subtree: true, characterData: true });
        let lastPoll = ta.value;
        setInterval(function() {
            const el = wrap.querySelector('textarea') || wrap.querySelector('input');
            const cur = el ? el.value : '';
            if (cur && cur !== lastPoll) { lastPoll = cur; renderSidebar(cur); }
        }, 350);
    }
    attempt(200);
}

// ── Watch chatbot for processing state ───────────────────────────────────────
function watchChatbot() {
    setInterval(function() {
        const sendBtn = document.querySelector('.send-btn');
        if (!sendBtn) return;
        const isBusy = sendBtn.disabled || sendBtn.getAttribute('aria-busy') === 'true';
        if (isBusy && !_processingActive) {
            showProcessing('Thinking...');
        } else if (!isBusy && _processingActive) {
            hideProcessing();
        }
    }, 300);
}

function initSearch() {
    const inp = document.getElementById('sidebar-search-input');
    if (!inp) { setTimeout(initSearch, 400); return; }
    inp.addEventListener('input', function() {
        const wrap = document.getElementById('sidebar-data-store');
        const el = wrap ? wrap.querySelector('textarea') || wrap.querySelector('input') : null;
        if (el && el.value) forceRender(el.value);
    });
}

// ── Global event delegation ───────────────────────────────────────────────────
document.addEventListener('click', function(e) {
    if (_ctxPopup && !_ctxPopup.contains(e.target)) closeCtx();
    const rp = document.getElementById('rename-popup');
    if (rp && rp.style.display !== 'none' && !rp.contains(e.target)) hideRename();
});
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') { closeCtx(); hideRename(); closeWorkspaceModal(); }
    if (e.key === 'Enter') {
        const rp = document.getElementById('rename-popup');
        if (rp && rp.style.display !== 'none') { e.preventDefault(); commitRename(); }
    }
});

// ── Boot ──────────────────────────────────────────────────────────────────────
function boot() {
    injectBodyPopups();
    watchSidebarData();
    initSearch();
    watchChatbot();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
} else {
    boot();
}
</script>
"""

# ── Sidebar HTML ──────────────────────────────────────────────────────────────

SIDEBAR_HTML = """
<div style="display:flex;flex-direction:column;height:100%;overflow:hidden;">

  <div id="sidebar-header">
    <div id="sidebar-logo">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3">
        <polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>
      </svg>
      Poshak-AI
    </div>
    <div id="sidebar-header-actions">
      <button id="workspace-open-btn" onclick="openWorkspaceModal()" title="Manage Workspace">
        &#x1F4C1;
      </button>
      <button id="new-chat-sidebar-btn" onclick="triggerNewChat()" title="New chat">+</button>
    </div>
  </div>

  <div id="sidebar-search-wrap">
    <input id="sidebar-search-input" type="text" placeholder="Search chats..." />
  </div>

  <div id="sessions-list">
    <div style="color:var(--text-dim);font-size:11px;padding:16px 14px;">Loading chats...</div>
  </div>

</div>
"""


# ── Build the UI ──────────────────────────────────────────────────────────────

with gr.Blocks(title="Poshak-AI", head=JS_HEAD) as demo:

    browser_session   = gr.BrowserState(default_value="", storage_key="poshak_active_session")
    active_session_id = gr.State(value="")
    workspace_badge   = gr.State(value="💬 General chat")

    # ── sidebar_data bus (hidden by CSS) ──────────────────────────────────────
    sidebar_data = gr.Textbox(
        value="", elem_id="sidebar-data-store",
        elem_classes=["sidebar-data-wrap"], label="", show_label=False, container=False,
    )

    with gr.Sidebar(position="left", width=270, open=True):
        gr.HTML(SIDEBAR_HTML)

    with gr.Column():

        # ── Topbar ────────────────────────────────────────────────────────────
        topbar_html = gr.HTML("""
        <div id="topbar">
          <div id="topbar-title">&#x1F9E0; Poshak-AI</div>
          <div id="topbar-ws">&#x1F4AC; General chat</div>
        </div>
        """)

        with gr.Row():

            with gr.Column(scale=5):

                chatbot = gr.Chatbot(
                    label="", value=[], height=520,
                    render_markdown=True, sanitize_html=True,
                    allow_tags=["thinking"], resizable=False, layout="bubble",
                    buttons=["copy", "copy_all"],
                    placeholder="*Ask anything — or add a project via the Workspace button (📁) in the sidebar...*",
                    autoscroll=True,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask anything... (Shift+Enter for new line)",
                        label="", lines=2, max_lines=6, scale=9, autofocus=True,
                    )
                    send_btn = gr.Button(
                        "Send", variant="primary", scale=1,
                        elem_classes=["send-btn"], min_width=80,
                    )

                export_btn = gr.Button("Export chat", size="sm", variant="secondary")
                export_out = gr.Textbox(label="Exported", visible=False, lines=8, max_lines=20)

            with gr.Column(scale=0, min_width=215):

                gr.Markdown("### Examples")
                gr.Markdown("*Click to fill*")

                EXAMPLES = [
                    "Explain the overall project structure",
                    "How is the database layer implemented?",
                    "What dependencies does this project use?",
                    "Find potential bugs or missing error handling",
                    "Write a new REST endpoint following existing style",
                    "Explain the authentication setup",
                    "Show the project directory structure",
                    "Search web for latest Spring Boot best practices",
                    "Run the tests and tell me what's failing",
                    "What design patterns are used here?",
                ]
                for ex in EXAMPLES:
                    gr.Button(ex, size="sm", elem_classes=["ex-btn"]).click(
                        fn=lambda e=ex: e, outputs=msg_input, api_visibility="private",
                    )

                with gr.Accordion("System", open=False):
                    status_box = gr.Textbox(
                        value=format_status(), label="", lines=5,
                        interactive=False, elem_classes=["mono-box"],
                    )
                    gr.Button("Refresh", size="sm").click(
                        fn=lambda sid: format_status(sid),
                        inputs=[active_session_id],
                        outputs=status_box, api_visibility="private",
                    )

                with gr.Accordion("Tips", open=False):
                    gr.Markdown("""
**Load a project:**
> Click 📁 in the sidebar → paste path → Validate → Apply & Index

**Codebase queries:**
> "How does auth work?"  
> "Explain UserService"

**Code gen:**
> "Write endpoint for X"  
> "Add validation to Y"

**General (no project):**
> Paste code and ask questions  
> "Explain this stack trace: ..."
                    """)

    # ── Hidden action bridge ──────────────────────────────────────────────────
    with gr.Row(elem_id="hidden-actions"):
        hid_new_btn      = gr.Button("__new__",    elem_id="hid-new-btn")
        hid_load_id      = gr.Textbox(value="", elem_id="hid-load-id",    label="", show_label=False, container=False)
        hid_load_btn     = gr.Button("__load__",   elem_id="hid-load-btn")
        hid_delete_id    = gr.Textbox(value="", elem_id="hid-delete-id",  label="", show_label=False, container=False)
        hid_delete_btn   = gr.Button("__delete__", elem_id="hid-delete-btn")
        hid_rename_id    = gr.Textbox(value="", elem_id="hid-rename-id",  label="", show_label=False, container=False)
        hid_rename_title = gr.Textbox(value="", elem_id="hid-rename-title", label="", show_label=False, container=False)
        hid_rename_btn   = gr.Button("__rename__", elem_id="hid-rename-btn")
        ws_paths_state   = gr.Textbox(value="", elem_id="ws-paths-state", label="", show_label=False, container=False)
        ws_mode_state    = gr.Textbox(value="existing", elem_id="ws-mode-state", label="", show_label=False, container=False)
        hid_ws_validate  = gr.Button("__ws_validate__", elem_id="hid-ws-validate-btn")
        hid_ws_apply     = gr.Button("__ws_apply__",    elem_id="hid-ws-apply-btn")
        hid_ws_clear     = gr.Button("__ws_clear__",    elem_id="hid-ws-clear-btn")

    # ── Helper: update topbar when workspace badge changes ────────────────────
    def _topbar_html(badge: str) -> str:
        return f"""
        <div id="topbar">
          <div id="topbar-title">&#x1F9E0; Poshak-AI</div>
          <div id="topbar-ws">{badge}</div>
        </div>
        """

    # ── Page load ─────────────────────────────────────────────────────────────
    demo.load(
        fn=on_page_load,
        inputs=[browser_session],
        outputs=[chatbot, active_session_id, sidebar_data, browser_session, workspace_badge],
        api_visibility="private",
        show_progress="hidden",
    )
    workspace_badge.change(
        fn=_topbar_html, inputs=workspace_badge, outputs=topbar_html,
        api_visibility="private",
    )

    # ── Chat ──────────────────────────────────────────────────────────────────
    def _send(msg, hist, sid):
        for r in respond(msg, hist, sid):
            yield r

    for trigger in [send_btn.click, msg_input.submit]:
        trigger(
            fn=_send,
            inputs=[msg_input, chatbot, active_session_id],
            outputs=[msg_input, chatbot, active_session_id, sidebar_data],
            queue=True, api_visibility="private",
        )

    # ── Session sidebar actions ───────────────────────────────────────────────
    hid_new_btn.click(
        fn=new_chat_action,
        inputs=[active_session_id, chatbot],
        outputs=[chatbot, active_session_id, sidebar_data, browser_session, workspace_badge],
        api_visibility="private",
    )
    hid_load_btn.click(
        fn=load_session_action,
        inputs=[hid_load_id],
        outputs=[chatbot, active_session_id, sidebar_data, browser_session, workspace_badge],
        api_visibility="private",
    )
    hid_delete_btn.click(
        fn=delete_session_action,
        inputs=[hid_delete_id, active_session_id],
        outputs=[chatbot, active_session_id, sidebar_data, browser_session, workspace_badge],
        api_visibility="private",
    )
    hid_rename_btn.click(
        fn=rename_session_action,
        inputs=[hid_rename_id, hid_rename_title],
        outputs=[sidebar_data],
        api_visibility="private",
    )

    # ── Workspace panel actions ───────────────────────────────────────────────
    def _validate_to_chat(paths: str, mode: str, history: list) -> list:
        msg, _status = validate_and_preview(paths, mode)
        return history + [{"role": "assistant", "content": msg}]

    hid_ws_validate.click(
        fn=_validate_to_chat,
        inputs=[ws_paths_state, ws_mode_state, chatbot],
        outputs=[chatbot],
        api_visibility="private",
    )

    hid_ws_apply.click(
        fn=apply_workspace,
        inputs=[ws_paths_state, ws_mode_state, active_session_id, chatbot],
        outputs=[chatbot, active_session_id, sidebar_data, workspace_badge, status_box],
        api_visibility="private",
    )
    hid_ws_apply.click(
        fn=lambda badge: _topbar_html(badge),
        inputs=[workspace_badge],
        outputs=[topbar_html],
        api_visibility="private",
    )

    hid_ws_clear.click(
        fn=clear_workspace_action,
        inputs=[active_session_id, chatbot],
        outputs=[chatbot, workspace_badge, status_box],
        api_visibility="private",
    )

    # ── Export ────────────────────────────────────────────────────────────────
    export_btn.click(
        fn=export_chat, inputs=chatbot,
        outputs=[export_out, export_out], api_visibility="private",
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poshak-AI Web UI")
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print(f"  Starting Poshak-AI UI -> http://localhost:{args.port}")
    print()

    demo.queue(max_size=5).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
        show_error=True,
        quiet=False,
        theme=gr.themes.Base(
            primary_hue="emerald",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("DM Sans"),
        ),
        css=CSS,
        footer_links=[],
    )