"""
ui.py — Poshak-AI
──────────────────
Gradio 6.x web interface with:
  • Native gr.Sidebar  — proper left-panel layout
  • gr.BrowserState    — active_session_id survives page refresh
  • demo.load()        — restores last session + sidebar on every page load
  • Custom JS sidebar  — Grok/ChatGPT-style rows, three-dot menu, search
  • SQLite persistence — all chats stored in conversations.db
  • URL routing        — each session gets its own URL  /?s=<session_id>

Fixes in this version:
  1. list_sessions() now returns ALL sessions (no message_count filter) so
     newly created chats appear immediately in the sidebar.
  2. _sessions_as_json() is called with the correct active_session_id on
     every action so the active highlight is always correct.
  3. on_page_load injects sidebar_data with the initial session JSON so the
     sidebar renders immediately on page load (no blank "Loading..." state).
  4. JS watcher uses a robust multi-strategy approach: events + aggressive
     polling + MutationObserver on the wrapper element (not just the textarea)
     to catch Gradio's Svelte-based virtual-DOM value updates reliably.
  5. sidebar_data initial value is pre-populated so JS has data on first paint.
  6. URL routing pushes /?s=<session_id> on every session switch and reads
     it back on popstate / page load.
  7. New-chat button creates the session, updates sidebar, pushes URL.
"""

import sys

# ── Windows UTF-8 fix — must be before any print() ───────────────────────────
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import gradio as gr

from config import (
    ACTIVE_PROJECT,
    LLM_MODEL,
    COLLECTION_NAME,
    DB_PATH,
    validate_config,
)
from agent import ask, get_executor, get_session_messages_for_display, get_active_llm_name
from chat_store import (
    init_db,
    create_session,
    list_sessions,
    delete_session,
    rename_session,
    update_session_after_message,
    group_sessions_by_date,
    get_most_recent_session,
    get_session,
)

# ── Startup ───────────────────────────────────────────────────────────────────

validate_config()
init_db()

print()
print("-" * 56)
print("  Poshak-AI - Loading agent...")
print("-" * 56)
executor = get_executor(verbose=False)
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


def get_chunk_count() -> int:
    try:
        import chromadb
        client = chromadb.PersistentClient(path=DB_PATH)
        col = client.get_collection(COLLECTION_NAME)
        return col.count()
    except Exception:
        return 0


def _sessions_as_json(active_session_id: str = "") -> str:
    """
    Build JSON string consumed by the sidebar JS renderer.

    CRITICAL FIX: list_sessions() is called WITHOUT any message_count filter
    so newly created sessions appear immediately in the sidebar.
    The chat_store.list_sessions() already returns all sessions ordered by
    updated_at DESC — we just group them here.
    """
    # Fetch ALL sessions for this workspace (no message_count filter here)
    all_sessions = list_sessions()  # returns all, ordered newest first

    # group_sessions_by_date handles the bucketing (Today / Yesterday / etc.)
    grouped = group_sessions_by_date(all_sessions)

    groups = []
    for group_name, group_sessions in grouped.items():
        items = []
        for s in group_sessions:
            items.append({
                "id":      s["session_id"],
                "title":   s["title"],
                "count":   s["message_count"],
                "active":  s["session_id"] == active_session_id,
            })
        groups.append({"label": group_name, "sessions": items})

    return json.dumps({"groups": groups, "active_id": active_session_id})


def format_status() -> str:
    llm_name = get_active_llm_name() or LLM_MODEL
    db_ok    = db_exists()
    chunks   = get_chunk_count() if db_ok else 0
    db_icon  = "OK" if db_ok and chunks > 0 else "!!"
    sessions = list_sessions()
    return "\n".join([
        f"LLM  : {llm_name}",
        f"DB   : {chunks:,} chunks  [{db_icon}]",
        f"Chats: {len(sessions)} saved",
        f"WS   : {ACTIVE_PROJECT}",
        f"Time : {datetime.now().strftime('%H:%M:%S')}",
    ])


# ── Chat / session actions ────────────────────────────────────────────────────

def respond(message: str, history: list, session_id: str):
    if not message.strip():
        yield "", history, session_id, _sessions_as_json(session_id)
        return

    is_first = (len(history) == 0)
    history  = history + [{"role": "user", "content": message}]
    yield "", history, session_id, _sessions_as_json(session_id)

    response = ask(executor, message, session_id=session_id)
    history  = history + [{"role": "assistant", "content": response}]
    update_session_after_message(session_id, message, is_first_message=is_first)

    yield "", history, session_id, _sessions_as_json(session_id)


def new_chat_action():
    """
    Creates a new session and returns updated state.
    The new session appears in the sidebar immediately because
    _sessions_as_json() fetches ALL sessions with no count filter.
    """
    session_id = create_session()
    sidebar_json = _sessions_as_json(session_id)
    return [], session_id, sidebar_json, session_id


def load_session_action(session_id: str):
    if not session_id or session_id in ("__none__", ""):
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    messages = get_session_messages_for_display(session_id)
    sidebar_json = _sessions_as_json(session_id)
    return messages, session_id, sidebar_json, session_id


def delete_session_action(session_id_to_delete: str, current_session_id: str):
    if session_id_to_delete:
        delete_session(session_id_to_delete)
    if session_id_to_delete == current_session_id:
        new_id   = create_session()
        messages = []
    else:
        new_id   = current_session_id
        messages = get_session_messages_for_display(new_id)
    sidebar_json = _sessions_as_json(new_id)
    return messages, new_id, sidebar_json, new_id


def rename_session_action(session_id: str, new_title: str) -> str:
    if not new_title.strip():
        return _sessions_as_json(session_id)
    rename_session(session_id, new_title.strip())
    return _sessions_as_json(session_id)


def on_page_load(request: gr.Request, saved_session_id: str):
    """
    Fires on every page load.

    CRITICAL: gr.Request must be the FIRST parameter. Gradio auto-injects
    the request object only when it precedes all gr.State/gr.BrowserState
    inputs in the function signature.

    FIX: We now ALWAYS return a populated sidebar_data JSON so the JS
    renderer has data on first paint and never shows a stale "Loading..."

    Priority order:
      1. ?s= URL query param  (direct link, browser back/forward)
      2. gr.BrowserState localStorage (last active session)
      3. Most recent session with messages
      4. Brand new session
    """
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
                    messages     = get_session_messages_for_display(candidate)
                    sidebar_json = _sessions_as_json(candidate)
                    return messages, candidate, sidebar_json, candidate
            except Exception:
                pass

    recent = get_most_recent_session()
    if recent:
        sid          = recent["session_id"]
        messages     = get_session_messages_for_display(sid)
        sidebar_json = _sessions_as_json(sid)
        return messages, sid, sidebar_json, sid

    sid          = create_session()
    sidebar_json = _sessions_as_json(sid)
    return [], sid, sidebar_json, sid


def reindex_project() -> str:
    try:
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "index_project.py"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        output = result.stdout + result.stderr
        if result.returncode == 0:
            import agent as agent_module
            agent_module._db = None
            agent_module._get_db()
            return f"Re-indexed successfully!\n\n{output[-800:]}"
        return f"Re-index failed:\n\n{output[-800:]}"
    except subprocess.TimeoutExpired:
        return "Timed out (5 min)."
    except Exception as e:
        return f"Error: {e}"


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

/* ── sidebar-data-wrap ───────────────────────────────────────────────────────
   The hidden data bus. MUST be visible=True in Gradio so the <textarea> is
   rendered into the DOM. We hide it purely with CSS.
   ─────────────────────────────────────────────────────────────────────────── */
.sidebar-data-wrap,
.sidebar-data-wrap * {
    position: fixed !important;
    top: -9999px !important;
    left: -9999px !important;
    width: 1px !important;
    height: 1px !important;
    overflow: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    z-index: -9999 !important;
}

/* Hidden action bridge */
#hidden-actions {
    position: fixed !important;
    top: -9999px !important;
    left: -9999px !important;
    width: 1px !important;
    height: 1px !important;
    overflow: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    z-index: -9999 !important;
}
#hidden-actions button { pointer-events: auto !important; }

/* ── Sidebar layout ─────────────────────────────────────────────────────────*/
#sidebar-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 12px 10px;
    border-bottom: 1px solid var(--border);
}
#sidebar-logo {
    font-weight: 700;
    font-size: 14px;
    color: var(--accent);
    display: flex;
    align-items: center;
    gap: 7px;
}
#new-chat-sidebar-btn {
    background: var(--accent);
    border: none;
    border-radius: 7px;
    width: 28px; height: 28px;
    display: flex; align-items: center; justify-content: center;
    color: #fff; font-size: 20px; font-weight: 300;
    cursor: pointer;
    transition: background 0.18s;
    flex-shrink: 0;
    line-height: 1;
}
#new-chat-sidebar-btn:hover { background: var(--accent-dim); }

#sidebar-search-wrap {
    padding: 9px 10px;
    border-bottom: 1px solid var(--border);
}
#sidebar-search-input {
    width: 100%;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 6px 10px;
    color: var(--text);
    font-size: 12px;
    font-family: var(--font);
    outline: none;
    transition: border-color 0.18s;
}
#sidebar-search-input:focus { border-color: var(--accent); }
#sidebar-search-input::placeholder { color: var(--text-dim); }

#sessions-list {
    overflow-y: auto;
    padding: 4px 0 12px;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
    height: calc(100vh - 120px);
}
#sessions-list::-webkit-scrollbar { width: 3px; }
#sessions-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

.session-group-label {
    font-size: 10px; font-weight: 600;
    color: var(--text-dim);
    letter-spacing: 0.9px; text-transform: uppercase;
    padding: 12px 13px 3px;
    user-select: none;
}
.session-row {
    display: flex; align-items: center;
    margin: 1px 5px;
    border-radius: var(--radius);
    cursor: pointer;
    transition: background 0.12s;
}
.session-row:hover { background: var(--bg-hover); }
.session-row.active {
    background: var(--bg-active);
    border-left: 2px solid var(--accent);
}
.session-row.active .session-title { color: var(--accent); }

.session-main { flex: 1; padding: 8px 5px 8px 9px; min-width: 0; user-select: none; }
.session-title {
    font-size: 12.5px; font-weight: 500; color: var(--text);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    line-height: 1.3; max-width: 182px;
}
.session-meta { font-size: 10px; color: var(--text-dim); margin-top: 1px; }

.session-dots {
    background: none; border: none;
    color: var(--text-muted);
    padding: 4px 7px; border-radius: 6px;
    cursor: pointer; font-size: 15px; line-height: 1;
    opacity: 0;
    transition: opacity 0.12s, background 0.12s;
    flex-shrink: 0;
}
.session-row:hover .session-dots { opacity: 1; }
.session-dots:hover { background: var(--bg-card); color: var(--text); }

/* ── Context menu ───────────────────────────────────────────────────────────*/
.ctx-popup {
    position: fixed;
    background: #1c1c1c;
    border: 1px solid #333;
    border-radius: 9px;
    box-shadow: 0 10px 36px rgba(0,0,0,0.55);
    z-index: 99999;
    min-width: 148px;
    overflow: hidden;
    animation: popIn 0.11s ease-out;
}
@keyframes popIn {
    from { opacity: 0; transform: scale(0.91) translateY(-3px); }
    to   { opacity: 1; transform: scale(1)   translateY(0); }
}
.ctx-popup button {
    display: flex; align-items: center; gap: 9px;
    width: 100%; text-align: left;
    background: none; border: none;
    padding: 9px 14px;
    font-size: 12.5px; font-family: var(--font);
    cursor: pointer; color: var(--text);
    transition: background 0.1s;
}
.ctx-popup button:hover { background: var(--bg-hover); }
.ctx-popup button.danger { color: var(--danger); }
.ctx-popup button.danger:hover { background: #221010; }
.ctx-divider { height: 1px; background: var(--border); margin: 2px 0; }

/* ── Rename popup ───────────────────────────────────────────────────────────*/
#rename-popup {
    display: none; position: fixed;
    background: #1c1c1c; border: 1px solid #333;
    border-radius: 10px; box-shadow: 0 10px 36px rgba(0,0,0,0.6);
    z-index: 99999; padding: 13px; min-width: 230px;
}
#rename-input {
    width: 100%; background: var(--bg-card);
    border: 1px solid var(--accent); border-radius: 6px;
    padding: 7px 10px; color: var(--text);
    font-size: 12.5px; font-family: var(--font);
    outline: none; margin-bottom: 9px;
}
#rename-actions { display: flex; gap: 6px; }
#rename-actions button {
    flex: 1; padding: 7px; border-radius: 7px; border: none;
    font-size: 12px; font-family: var(--font); font-weight: 500; cursor: pointer;
}
#rename-save { background: var(--accent); color: #fff; }
#rename-save:hover { background: var(--accent-dim); }
#rename-cancel { background: var(--bg-hover); color: var(--text); border: 1px solid var(--border); }

/* ── Top bar ────────────────────────────────────────────────────────────────*/
#topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 11px 20px;
    border-bottom: 1px solid var(--border);
    background: var(--bg);
    flex-shrink: 0;
}
#topbar-title { font-weight: 700; font-size: 14px; color: var(--text); }
#topbar-ws {
    font-family: var(--mono); font-size: 10.5px; color: var(--text-dim);
    background: var(--bg-card); border: 1px solid var(--border);
    padding: 3px 9px; border-radius: 5px;
}

/* ── Chat ───────────────────────────────────────────────────────────────────*/
.gradio-chatbot, [data-testid="chatbot"] {
    background: var(--bg) !important;
    border: none !important;
}
.message.user > div {
    background: #172817 !important;
    border: 1px solid #243824 !important;
    border-radius: 12px 12px 4px 12px !important;
    color: var(--text) !important;
}
.message.bot > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px 12px 12px 4px !important;
    color: var(--text) !important;
}

/* ── Input ──────────────────────────────────────────────────────────────────*/
.gr-textbox textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
    font-size: 13.5px !important;
}
.gr-textbox textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(29,158,117,0.12) !important;
    outline: none !important;
}
.send-btn {
    background: var(--accent) !important;
    color: #fff !important;
    border-radius: var(--radius) !important;
    font-weight: 600 !important;
    font-family: var(--font) !important;
    border: none !important;
    transition: background 0.18s !important;
}
.send-btn:hover { background: var(--accent-dim) !important; }

.ex-btn {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 7px !important;
    font-size: 11px !important;
    font-family: var(--font) !important;
    text-align: left !important;
    white-space: normal !important;
    line-height: 1.35 !important;
    padding: 7px 9px !important;
    margin-bottom: 3px !important;
    transition: border-color 0.14s, background 0.14s !important;
    width: 100% !important;
}
.ex-btn:hover { border-color: var(--accent) !important; background: var(--bg-hover) !important; }

.gr-accordion {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    margin-bottom: 7px !important;
}
.mono-box textarea {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    color: var(--text-muted) !important;
    background: var(--bg) !important;
    border-color: var(--border) !important;
}
"""

# ── JavaScript ────────────────────────────────────────────────────────────────
# Key fixes vs original:
#  1. watchSidebarData() retries indefinitely until it finds the textarea,
#     using exponential backoff so it doesn't hog the CPU.
#  2. Multiple event strategies: 'input', 'change', MutationObserver on the
#     WRAPPER div (catches Svelte prop updates that bypass textarea events),
#     and a 350ms polling interval as final fallback.
#  3. renderSidebar() is idempotent (skips if JSON unchanged) but we force
#     a re-render when search text changes.
#  4. pushSessionUrl() is called on every session switch, including on load.
#  5. popstate handler restores the session from the URL.

JS_HEAD = r"""
<script>
'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
let _ctxPopup      = null;
let _renameId      = null;
let _lastJson      = '';
let _navigating    = false;
let _watchStarted  = false;

// ── URL routing ───────────────────────────────────────────────────────────────

function pushSessionUrl(sessionId) {
    if (!sessionId) return;
    const encoded = encodeURIComponent(sessionId);
    const target  = '/?s=' + encoded;
    if (window.location.pathname + window.location.search !== target) {
        history.pushState({ sessionId }, '', target);
    }
}

window.addEventListener('popstate', function(e) {
    const sid = new URLSearchParams(window.location.search).get('s');
    if (sid) {
        _navigating = true;
        triggerLoad(sid);
        setTimeout(() => { _navigating = false; }, 1200);
    }
});

// ── Utilities ─────────────────────────────────────────────────────────────────

function closeCtx() {
    if (_ctxPopup) { _ctxPopup.remove(); _ctxPopup = null; }
}

function esc(s) {
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

// ── Sidebar renderer ──────────────────────────────────────────────────────────

function renderSidebar(jsonStr) {
    if (!jsonStr || jsonStr === _lastJson) return;
    _lastJson = jsonStr;

    let data;
    try { data = JSON.parse(jsonStr); } catch(e) { return; }

    const container = document.getElementById('sessions-list');
    if (!container) return;

    // Push URL for active session (skip if triggered by popstate)
    if (data.active_id && !_navigating) {
        pushSessionUrl(data.active_id);
    }

    const search = (
        document.getElementById('sidebar-search-input')?.value || ''
    ).toLowerCase().trim();

    container.innerHTML = '';

    const groups = data.groups || [];
    if (groups.length === 0) {
        container.innerHTML =
            '<div style="color:var(--text-dim);font-size:11px;padding:16px 14px;">' +
            'No chats yet. Click + to start.</div>';
        return;
    }

    for (const group of groups) {
        const rows = group.sessions.filter(s =>
            !search || s.title.toLowerCase().includes(search)
        );
        if (!rows.length) continue;

        const grpEl = document.createElement('div');

        const lbl = document.createElement('div');
        lbl.className   = 'session-group-label';
        lbl.textContent = group.label;
        grpEl.appendChild(lbl);

        for (const s of rows) {
            const row = document.createElement('div');
            row.className  = 'session-row' + (s.active ? ' active' : '');
            row.dataset.id = s.id;

            const main = document.createElement('div');
            main.className = 'session-main';
            const metaText = s.count > 0
                ? s.count + ' msg' + (s.count !== 1 ? 's' : '')
                : 'New chat';
            main.innerHTML =
                '<div class="session-title">' + esc(s.title) + '</div>' +
                '<div class="session-meta">'  + metaText    + '</div>';

            const dots = document.createElement('button');
            dots.className   = 'session-dots';
            dots.title       = 'Options';
            dots.textContent = '\u22EE';

            main.addEventListener('click', function() {
                closeCtx();
                triggerLoad(s.id);
            });
            dots.addEventListener('click', function(e) {
                e.stopPropagation();
                closeCtx();
                showCtx(e, s.id, s.title);
            });

            row.appendChild(main);
            row.appendChild(dots);
            grpEl.appendChild(row);
        }
        container.appendChild(grpEl);
    }
}

// Force re-render (clears cache so same JSON re-renders after search change)
function forceRender(jsonStr) {
    _lastJson = '';
    renderSidebar(jsonStr);
}

// ── Context menu ──────────────────────────────────────────────────────────────

function showCtx(e, id, title) {
    const popup = document.createElement('div');
    popup.className = 'ctx-popup';

    const renBtn = document.createElement('button');
    renBtn.innerHTML = '&#x270F;&#xFE0F;&nbsp; Rename';
    renBtn.onclick = function() { closeCtx(); showRename(id, title, e.clientX, e.clientY); };

    const divider = document.createElement('div');
    divider.className = 'ctx-divider';

    const delBtn = document.createElement('button');
    delBtn.className = 'danger';
    delBtn.innerHTML = '&#x1F5D1;&#xFE0F;&nbsp; Delete';
    delBtn.onclick = function() { closeCtx(); triggerDelete(id); };

    popup.appendChild(renBtn);
    popup.appendChild(divider);
    popup.appendChild(delBtn);
    document.body.appendChild(popup);
    _ctxPopup = popup;

    const W = window.innerWidth, H = window.innerHeight;
    let left = e.clientX + 6, top = e.clientY;
    if (left + 160 > W) left = e.clientX - 166;
    if (top  + 90  > H) top  = H - 95;
    popup.style.left = left + 'px';
    popup.style.top  = top  + 'px';
}

// ── Rename popup ──────────────────────────────────────────────────────────────

function showRename(id, title, x, y) {
    _renameId = id;
    const popup = document.getElementById('rename-popup');
    const input = document.getElementById('rename-input');
    if (!popup || !input) return;
    input.value = title;
    const W = window.innerWidth, H = window.innerHeight;
    popup.style.left    = Math.min(x, W - 250) + 'px';
    popup.style.top     = Math.min(y, H - 130) + 'px';
    popup.style.display = 'block';
    input.focus();
    input.select();
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
    writeHidden('hid-rename-id',    _renameId);
    writeHidden('hid-rename-title', title);
    setTimeout(function() { clickHidden('hid-rename-btn'); hideRename(); }, 60);
}

// ── Gradio DOM bridge ─────────────────────────────────────────────────────────
// Gradio 6 renders textboxes as <label><textarea></textarea></label> inside
// the wrapper div. We locate the textarea regardless of nesting depth.

function findTextarea(elemId) {
    const wrap = document.getElementById(elemId);
    if (!wrap) return null;
    return wrap.querySelector('textarea') || wrap.querySelector('input[type="text"]');
}

function writeHidden(elemId, value) {
    const ta = findTextarea(elemId);
    if (!ta) {
        // Element not ready yet — retry
        setTimeout(function() { writeHidden(elemId, value); }, 200);
        return;
    }
    const proto  = ta.tagName === 'TEXTAREA'
        ? HTMLTextAreaElement.prototype
        : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, 'value');
    if (setter && setter.set) {
        setter.set.call(ta, value);
    } else {
        ta.value = value;
    }
    ta.dispatchEvent(new Event('input',  { bubbles: true }));
    ta.dispatchEvent(new Event('change', { bubbles: true }));
}

function clickHidden(elemId) {
    // elemId might be the button itself or a wrapper containing a button
    const el = document.getElementById(elemId);
    if (!el) {
        setTimeout(function() { clickHidden(elemId); }, 200);
        return;
    }
    const btn = el.tagName === 'BUTTON' ? el : el.querySelector('button');
    if (btn) btn.click(); else el.click();
}

function triggerNewChat()  { clickHidden('hid-new-btn'); }
function triggerLoad(id)   { writeHidden('hid-load-id', id);    setTimeout(function() { clickHidden('hid-load-btn');    }, 80); }
function triggerDelete(id) { writeHidden('hid-delete-id', id);  setTimeout(function() { clickHidden('hid-delete-btn');  }, 80); }

// ── sidebar_data watcher ──────────────────────────────────────────────────────
//
// Strategy (layered, most-to-least reliable):
//
//  A) 'input' / 'change' events on the <textarea> — fires when Gradio sets
//     value via React/Svelte dispatcher. Sometimes works, sometimes doesn't.
//
//  B) MutationObserver on the WRAPPER div — catches attribute/childList
//     mutations that happen when Svelte re-renders the element.
//
//  C) setInterval polling at 350 ms — the guaranteed fallback. Gradio sets
//     the textarea .value property directly (bypassing all DOM events), so
//     polling is the only 100% reliable mechanism.
//
// We start watching as soon as the wrapper div exists in the DOM (which
// happens before the textarea is hydrated), using exponential backoff.

function watchSidebarData() {
    if (_watchStarted) return;

    const STORE_ID = 'sidebar-data-store';

    function attempt(delay) {
        const wrap = document.getElementById(STORE_ID);
        if (!wrap) {
            // Wrapper not in DOM yet — keep retrying
            setTimeout(function() { attempt(Math.min(delay * 1.5, 1000)); }, delay);
            return;
        }

        const ta = wrap.querySelector('textarea') || wrap.querySelector('input');
        if (!ta) {
            // Wrapper exists but textarea not hydrated yet
            setTimeout(function() { attempt(Math.min(delay * 1.5, 800)); }, delay);
            return;
        }

        // ── Got the textarea ──────────────────────────────────────────────────
        _watchStarted = true;
        console.log('[Poshak] sidebar-data textarea found — watcher active');

        // Render immediately if already populated
        if (ta.value) renderSidebar(ta.value);

        // Strategy A: events on textarea
        ta.addEventListener('input',  function() { renderSidebar(ta.value); });
        ta.addEventListener('change', function() { renderSidebar(ta.value); });

        // Strategy B: MutationObserver on wrapper
        new MutationObserver(function() {
            const val = (wrap.querySelector('textarea') || wrap.querySelector('input') || {}).value;
            if (val) renderSidebar(val);
        }).observe(wrap, { attributes: true, childList: true, subtree: true, characterData: true });

        // Strategy C: polling (primary in practice — Gradio bypasses DOM events)
        let lastPoll = ta.value;
        setInterval(function() {
            const el  = wrap.querySelector('textarea') || wrap.querySelector('input');
            const cur = el ? el.value : '';
            if (cur && cur !== lastPoll) {
                lastPoll = cur;
                renderSidebar(cur);
            }
        }, 350);
    }

    attempt(200);
}

// ── Search ────────────────────────────────────────────────────────────────────

function initSearch() {
    const inp = document.getElementById('sidebar-search-input');
    if (!inp) { setTimeout(initSearch, 400); return; }
    inp.addEventListener('input', function() {
        const wrap = document.getElementById('sidebar-data-store');
        const el   = wrap
            ? wrap.querySelector('textarea') || wrap.querySelector('input')
            : null;
        if (el && el.value) forceRender(el.value);
    });
}

// ── Startup diagnostics ───────────────────────────────────────────────────────

function verifyBridge() {
    const ids = [
        'sidebar-data-store',
        'hid-new-btn', 'hid-load-id', 'hid-load-btn',
        'hid-delete-id', 'hid-delete-btn',
        'hid-rename-id', 'hid-rename-title', 'hid-rename-btn',
    ];
    const missing = ids.filter(function(id) { return !document.getElementById(id); });
    if (missing.length) {
        console.error('[Poshak] MISSING bridge elements:', missing);
    } else {
        console.log('[Poshak] All bridge elements found');
    }
}

// ── Global event handlers ─────────────────────────────────────────────────────

document.addEventListener('click', function(e) {
    if (_ctxPopup && !_ctxPopup.contains(e.target)) closeCtx();
    const rp = document.getElementById('rename-popup');
    if (rp && rp.style.display !== 'none' && !rp.contains(e.target)) hideRename();
});

document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') { closeCtx(); hideRename(); }
    if (e.key === 'Enter') {
        const rp = document.getElementById('rename-popup');
        if (rp && rp.style.display !== 'none') { e.preventDefault(); commitRename(); }
    }
});

// ── Boot ──────────────────────────────────────────────────────────────────────
// Start watching immediately — don't wait for 'load' which fires after all
// resources (fonts, images) are ready. DOMContentLoaded is earlier and
// sufficient since we only need the DOM.

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
        watchSidebarData();
        initSearch();
        setTimeout(verifyBridge, 1500);
    });
} else {
    // Already past DOMContentLoaded (script injected late)
    watchSidebarData();
    initSearch();
    setTimeout(verifyBridge, 1500);
}
</script>
"""

# ── Sidebar HTML ──────────────────────────────────────────────────────────────

SIDEBAR_HTML = f"""
<div style="display:flex;flex-direction:column;height:100%;overflow:hidden;">

  <div id="sidebar-header">
    <div id="sidebar-logo">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2.3">
        <polyline points="16 18 22 12 16 6"/>
        <polyline points="8 6 2 12 8 18"/>
      </svg>
      Poshak-AI
    </div>
    <button id="new-chat-sidebar-btn"
            onclick="triggerNewChat()"
            title="New chat">+</button>
  </div>

  <div id="sidebar-search-wrap">
    <input id="sidebar-search-input" type="text" placeholder="Search chats..." />
  </div>

  <div id="sessions-list">
    <div style="color:var(--text-dim);font-size:11px;padding:16px 14px;">
      Loading sessions...
    </div>
  </div>

</div>

<div id="rename-popup">
  <input id="rename-input" type="text" placeholder="New title..." />
  <div id="rename-actions">
    <button id="rename-save"   onclick="commitRename()">Save</button>
    <button id="rename-cancel" onclick="hideRename()">Cancel</button>
  </div>
</div>
"""


# ── Build the UI ──────────────────────────────────────────────────────────────

with gr.Blocks(title="Poshak-AI", head=JS_HEAD) as demo:

    browser_session    = gr.BrowserState(default_value="", storage_key="poshak_active_session")
    active_session_id  = gr.State(value="")

    # ── sidebar_data ──────────────────────────────────────────────────────────
    # MUST be visible=True — Gradio 6 does not render <textarea> for hidden
    # components, which would break the JS bridge. We hide it via CSS only.
    sidebar_data = gr.Textbox(
        value="",
        elem_id="sidebar-data-store",
        elem_classes=["sidebar-data-wrap"],
        label="",
        show_label=False,
        container=False,
    )

    with gr.Sidebar(position="left", width=265, open=True):
        gr.HTML(SIDEBAR_HTML)

    with gr.Column():

        gr.HTML(f"""
        <div id="topbar">
          <div id="topbar-title">&#x1F9E0; Poshak-AI</div>
          <div id="topbar-ws">&#x1F4C1; {ACTIVE_PROJECT}</div>
        </div>
        """)

        with gr.Row():

            with gr.Column(scale=5):

                chatbot = gr.Chatbot(
                    label="",
                    value=[],
                    height=540,
                    render_markdown=True,
                    sanitize_html=True,
                    allow_tags=["thinking"],
                    resizable=False,
                    layout="bubble",
                    buttons=["copy", "copy_all"],
                    placeholder="*Ask anything about your codebase...*",
                    autoscroll=True,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask anything about your code... (Shift+Enter for new line)",
                        label="",
                        lines=2,
                        max_lines=6,
                        scale=9,
                        autofocus=True,
                    )
                    send_btn = gr.Button(
                        "Send",
                        variant="primary",
                        scale=1,
                        elem_classes=["send-btn"],
                        min_width=80,
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
                        fn=lambda e=ex: e,
                        outputs=msg_input,
                        api_visibility="private",
                    )

                with gr.Accordion("System", open=False):
                    status_box = gr.Textbox(
                        value=format_status(),
                        label="",
                        lines=5,
                        interactive=False,
                        elem_classes=["mono-box"],
                    )
                    gr.Button("Refresh", size="sm").click(
                        fn=format_status,
                        outputs=status_box,
                        api_visibility="private",
                    )

                with gr.Accordion("Maintenance", open=False):
                    reindex_btn = gr.Button("Re-index project", size="sm", variant="secondary")
                    reindex_out = gr.Textbox(label="", lines=4, visible=False, interactive=False)
                    reindex_btn.click(
                        fn=lambda: gr.update(visible=True, value="Re-indexing... please wait"),
                        outputs=reindex_out,
                    ).then(fn=reindex_project, outputs=reindex_out)

                with gr.Accordion("Tips", open=False):
                    gr.Markdown("""
**Codebase:**
> "How does auth work?"
> "Explain UserService"

**Code gen:**
> "Write endpoint for X"
> "Add validation to Y"

**Files:**
> "Read AuthController.java"

**Git:**
> "Show last 5 commits"
                    """)

    # ── Hidden action bridge ──────────────────────────────────────────────────
    # visible=True, hidden by CSS (#hidden-actions rule above).
    # Each element needs elem_id so JS can locate it.
    with gr.Row(elem_id="hidden-actions"):
        hid_new_btn      = gr.Button("__new__",    elem_id="hid-new-btn")
        hid_load_id      = gr.Textbox(value="", elem_id="hid-load-id",
                                      label="", show_label=False, container=False)
        hid_load_btn     = gr.Button("__load__",   elem_id="hid-load-btn")
        hid_delete_id    = gr.Textbox(value="", elem_id="hid-delete-id",
                                      label="", show_label=False, container=False)
        hid_delete_btn   = gr.Button("__delete__", elem_id="hid-delete-btn")
        hid_rename_id    = gr.Textbox(value="", elem_id="hid-rename-id",
                                      label="", show_label=False, container=False)
        hid_rename_title = gr.Textbox(value="", elem_id="hid-rename-title",
                                      label="", show_label=False, container=False)
        hid_rename_btn   = gr.Button("__rename__", elem_id="hid-rename-btn")

    # ── Page load ─────────────────────────────────────────────────────────────
    # gr.Request MUST be the first parameter so Gradio auto-injects it.
    demo.load(
        fn=on_page_load,
        inputs=[browser_session],
        outputs=[chatbot, active_session_id, sidebar_data, browser_session],
        api_visibility="private",
        show_progress="hidden",
    )

    # ── Chat ──────────────────────────────────────────────────────────────────
    def _send(msg, hist, sid):
        for r in respond(msg, hist, sid):
            yield r

    send_btn.click(
        fn=_send,
        inputs=[msg_input, chatbot, active_session_id],
        outputs=[msg_input, chatbot, active_session_id, sidebar_data],
        queue=True,
        api_visibility="private",
    )
    msg_input.submit(
        fn=_send,
        inputs=[msg_input, chatbot, active_session_id],
        outputs=[msg_input, chatbot, active_session_id, sidebar_data],
        queue=True,
        api_visibility="private",
    )

    # ── Sidebar actions ───────────────────────────────────────────────────────
    hid_new_btn.click(
        fn=new_chat_action,
        outputs=[chatbot, active_session_id, sidebar_data, browser_session],
        api_visibility="private",
    )

    hid_load_btn.click(
        fn=load_session_action,
        inputs=[hid_load_id],
        outputs=[chatbot, active_session_id, sidebar_data, browser_session],
        api_visibility="private",
    )

    hid_delete_btn.click(
        fn=delete_session_action,
        inputs=[hid_delete_id, active_session_id],
        outputs=[chatbot, active_session_id, sidebar_data, browser_session],
        api_visibility="private",
    )

    hid_rename_btn.click(
        fn=rename_session_action,
        inputs=[hid_rename_id, hid_rename_title],
        outputs=[sidebar_data],
        api_visibility="private",
    )

    # ── Export ────────────────────────────────────────────────────────────────
    export_btn.click(
        fn=export_chat,
        inputs=chatbot,
        outputs=[export_out, export_out],
        api_visibility="private",
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