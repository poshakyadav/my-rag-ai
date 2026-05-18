# workspace_manager.py

"""
workspace_manager.py — Poshak-AI
──────────────────────────────────
Manages dynamic, per-session workspace configuration.

Instead of hard-coding paths in config.py, workspaces are set at runtime
via the UI. Each chat session can have its own workspace (or none at all).

Workspace modes:
  - "none"     : General chat, no project context
  - "existing" : One or more existing directories on disk
  - "new"      : A new project to be scaffolded in a chosen directory

Workspace state is stored in SQLite alongside chat sessions so it
survives restarts and page reloads.

Key design decisions:
  1. Validation is strict and early — bad paths are caught before any
     indexing or agent invocation happens.
  2. Each session's workspace is independent — switching sessions
     switches the workspace context automatically.
  3. The module is stateless at the module level (no global mutable
     workspace). The caller always passes session_id.
  4. ChromaDB collections are namespaced per session so multi-project
     sessions don't bleed into each other.
"""

from __future__ import annotations

import os
import re
import json
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from config import CONVERSATIONS_DB

# ── Schema ────────────────────────────────────────────────────────────────────

_CREATE_WORKSPACES_TABLE = """
CREATE TABLE IF NOT EXISTS workspaces (
    session_id   TEXT PRIMARY KEY,
    mode         TEXT NOT NULL DEFAULT 'none',
    paths_json   TEXT NOT NULL DEFAULT '{}',
    label        TEXT NOT NULL DEFAULT '',
    indexed      INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
"""

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class WorkspaceConfig:
    """
    Represents the workspace attached to a single chat session.

    mode:
        "none"     — no project; pure chat
        "existing" — one or more paths that exist on disk
        "new"      — single path where a new project will be created

    paths:
        dict of {label: absolute_path_string}
        e.g. {"backend": "D:/Projects/Api", "frontend": "D:/Projects/Web"}

    label:
        human-readable display name (derived from folder names)

    indexed:
        True once all paths have been embedded into ChromaDB for this session
    """
    session_id: str
    mode: str = "none"                          # "none" | "existing" | "new"
    paths: dict[str, str] = field(default_factory=dict)
    label: str = ""
    indexed: bool = False

    @property
    def collection_name(self) -> str:
        """Unique ChromaDB collection name for this session."""
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", self.session_id)
        return f"ws_{safe[:40]}"

    @property
    def has_paths(self) -> bool:
        return bool(self.paths)

    @property
    def primary_path(self) -> Optional[str]:
        """First path — used when a single cwd is needed."""
        if self.paths:
            return list(self.paths.values())[0]
        return None

    def summary(self) -> str:
        if self.mode == "none":
            return "No project — general chat mode"
        if not self.paths:
            return "Workspace configured but no paths set"
        lines = [f"Mode: {self.mode}  |  {len(self.paths)} path(s)"]
        for lbl, p in self.paths.items():
            lines.append(f"  • {lbl}: {p}")
        return "\n".join(lines)


# ── Validation ────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    ok: bool
    valid_paths: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def message(self) -> str:
        parts = []
        if self.errors:
            parts.append("**Errors:**\n" + "\n".join(f"  ❌ {e}" for e in self.errors))
        if self.warnings:
            parts.append("**Warnings:**\n" + "\n".join(f"  ⚠️ {w}" for w in self.warnings))
        if self.ok and self.valid_paths:
            parts.append("**Valid paths:**\n" + "\n".join(
                f"  ✅ `{lbl}` → `{p}`" for lbl, p in self.valid_paths.items()
            ))
        return "\n\n".join(parts)


def validate_paths(raw_input: str, mode: str = "existing") -> ValidationResult:
    """
    Parse and validate one or more project paths from a raw string.

    Accepted formats (all produce the same dict result):
      Single path:
        D:\\Projects\\MyApp
        /home/user/myapp

      Multiple paths (newline or semicolon separated):
        D:\\Projects\\Backend
        D:\\Projects\\Frontend

      Labeled paths (label: path):
        backend: D:\\Projects\\Backend
        frontend: D:\\Projects\\Frontend

    For mode="new":
      - Parent directory must exist.
      - The leaf folder need NOT exist (it will be created).
      - Exactly one path is accepted.

    For mode="existing":
      - Each path must be an existing directory.
      - At least one valid path is required.
    """
    result = ValidationResult(ok=False)

    if not raw_input or not raw_input.strip():
        result.errors.append("No path provided. Please enter at least one directory path.")
        return result

    # ── Parse lines into candidate {label: path} pairs ────────────────────────
    candidates: dict[str, str] = {}
    for i, line in enumerate(re.split(r"[\n;]+", raw_input.strip()), start=1):
        line = line.strip()
        if not line:
            continue

        # "label: path" or "label = path"
        m = re.match(r"^([a-zA-Z0-9_\- ]+?)\s*[:=]\s*(.+)$", line)
        if m:
            label = m.group(1).strip().lower().replace(" ", "_")
            path_str = m.group(2).strip()
        else:
            # No label — auto-generate from folder name
            path_str = line
            label = f"project_{i}" if i > 1 else "main"

        # Strip surrounding quotes that users sometimes paste
        path_str = path_str.strip("'\"")
        candidates[label] = path_str

    if not candidates:
        result.errors.append("Could not parse any paths from your input.")
        return result

    if mode == "new" and len(candidates) > 1:
        result.errors.append(
            "Only one path is supported when creating a new project. "
            "Please provide a single directory path."
        )
        return result

    # ── Validate each candidate ───────────────────────────────────────────────
    for label, raw_path in candidates.items():
        try:
            p = Path(raw_path)

            # ── NEW PROJECT ───────────────────────────────────────────────────
            if mode == "new":
                if p.exists() and not p.is_dir():
                    result.errors.append(
                        f"`{raw_path}` exists but is a file, not a directory."
                    )
                    continue
                # Parent must exist
                if not p.parent.exists():
                    result.errors.append(
                        f"Parent directory does not exist: `{p.parent}`\n"
                        f"  Please create `{p.parent}` first, or choose a different path."
                    )
                    continue
                if p.exists():
                    result.warnings.append(
                        f"`{raw_path}` already exists — it will be used as-is (not overwritten)."
                    )
                result.valid_paths[label] = str(p.resolve())

            # ── EXISTING PROJECT ──────────────────────────────────────────────
            else:
                if not p.exists():
                    # Give a helpful diagnostic
                    suggestions = _suggest_fix(p)
                    msg = f"`{raw_path}` does not exist."
                    if suggestions:
                        msg += f"\n  Did you mean one of these?\n" + "\n".join(
                            f"    • `{s}`" for s in suggestions
                        )
                    result.errors.append(msg)
                    continue
                if not p.is_dir():
                    result.errors.append(
                        f"`{raw_path}` is a file, not a directory. "
                        "Please provide the project root folder."
                    )
                    continue

                # Warn if directory looks empty (no source files)
                src_count = sum(1 for _ in p.rglob("*") if _.is_file())
                if src_count == 0:
                    result.warnings.append(
                        f"`{raw_path}` appears to be empty — "
                        "no files were found inside it."
                    )

                result.valid_paths[label] = str(p.resolve())

        except (OSError, ValueError) as exc:
            result.errors.append(f"Invalid path `{raw_path}`: {exc}")

    # ── Final verdict ─────────────────────────────────────────────────────────
    if result.valid_paths:
        result.ok = True
        # Auto-label when only one path and label is generic
        if len(result.valid_paths) == 1 and list(result.valid_paths.keys())[0] in ("main", "project_1"):
            first_path = list(result.valid_paths.values())[0]
            result.valid_paths = {Path(first_path).name: first_path}

    return result


def _suggest_fix(p: Path) -> list[str]:
    """
    Try to find similar paths on disk to help the user fix a typo.
    Checks: parent dir, drive root, common project dirs.
    """
    suggestions = []
    try:
        parent = p.parent
        if parent.exists():
            siblings = [
                str(child) for child in parent.iterdir()
                if child.is_dir() and child.name.lower().startswith(p.name[:3].lower())
            ]
            suggestions.extend(siblings[:3])
    except OSError:
        pass
    return suggestions


# ── SQLite persistence ────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(CONVERSATIONS_DB, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_workspace_db() -> None:
    """Create workspace table if missing. Safe to call repeatedly."""
    with _conn() as conn:
        conn.execute(_CREATE_WORKSPACES_TABLE)
        conn.commit()


def save_workspace(cfg: WorkspaceConfig) -> None:
    from datetime import datetime
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO workspaces
                (session_id, mode, paths_json, label, indexed, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                mode=excluded.mode,
                paths_json=excluded.paths_json,
                label=excluded.label,
                indexed=excluded.indexed,
                updated_at=excluded.updated_at
            """,
            (
                cfg.session_id,
                cfg.mode,
                json.dumps(cfg.paths),
                cfg.label,
                int(cfg.indexed),
                now, now,
            ),
        )
        conn.commit()


def load_workspace(session_id: str) -> WorkspaceConfig:
    """Load a session's workspace, or return a blank "none" workspace."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM workspaces WHERE session_id = ?", (session_id,)
        ).fetchone()
    if row is None:
        return WorkspaceConfig(session_id=session_id)
    return WorkspaceConfig(
        session_id=session_id,
        mode=row["mode"],
        paths=json.loads(row["paths_json"] or "{}"),
        label=row["label"],
        indexed=bool(row["indexed"]),
    )


def mark_indexed(session_id: str) -> None:
    from datetime import datetime
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    with _conn() as conn:
        conn.execute(
            "UPDATE workspaces SET indexed=1, updated_at=? WHERE session_id=?",
            (now, session_id),
        )
        conn.commit()


def clear_workspace(session_id: str) -> None:
    """Reset a session's workspace to 'none'."""
    cfg = WorkspaceConfig(session_id=session_id)
    save_workspace(cfg)


# ── ChromaDB per-session indexing ─────────────────────────────────────────────

def build_session_index(cfg: WorkspaceConfig, progress_cb=None) -> tuple[bool, str]:
    """
    Index all paths in `cfg` into a session-specific ChromaDB collection.

    progress_cb(msg: str) — optional callback for live UI progress updates.

    Returns (success: bool, message: str).
    """
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    import chromadb

    from config import (
        ALL_EXTENSIONS, EXCLUDE_DIRS, EMBEDDING_MODEL, EMBEDDING_DEVICE,
        DB_PATH, CHUNK_SIZE, CHUNK_OVERLAP,
    )

    def _prog(msg: str):
        if progress_cb:
            progress_cb(msg)

    _prog("📂 Scanning project files...")

    all_docs = []
    for label, path_str in cfg.paths.items():
        project_path = Path(path_str).resolve()
        if not project_path.exists():
            return False, f"Path no longer exists: `{path_str}`\nPlease re-validate your workspace."

        matched: set[Path] = set()
        for pattern in ALL_EXTENSIONS:
            for f in project_path.glob(pattern):
                rel = f.relative_to(project_path)
                if not any(part in EXCLUDE_DIRS for part in rel.parts):
                    matched.add(f)

        for file in sorted(matched):
            ext = file.suffix.lower()
            for enc in ("utf-8", "latin-1"):
                try:
                    loader = TextLoader(str(file), encoding=enc)
                    loaded = loader.load()
                    for doc in loaded:
                        doc.metadata["source"]      = str(file.relative_to(project_path))
                        doc.metadata["extension"]   = ext
                        doc.metadata["filename"]    = file.name
                        doc.metadata["sub_project"] = label
                    all_docs.extend(loaded)
                    break
                except (UnicodeDecodeError, Exception):
                    continue

    if not all_docs:
        return False, (
            "No indexable files found in the provided path(s).\n\n"
            "Supported file types: `.java`, `.ts`, `.tsx`, `.js`, `.jsx`, `.py`, "
            "`.xml`, `.yml`, `.yaml`, `.sql`, `.json`, `.css`, `.md` and more.\n\n"
            "Please check that you provided the correct project root directory."
        )

    _prog(f"✂️ Splitting {len(all_docs)} files into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\npublic class ", "\nprivate class ", "\nprotected class ",
            "\npublic interface ", "\npublic enum ",
            "\n    public ", "\n    private ", "\n    protected ",
            "\nexport default ", "\nexport const ", "\nexport function ",
            "\nexport class ", "\nconst ", "\nfunction ", "\nclass ",
            "\n\n\n", "\n\n", "\n", " ", "",
        ],
    )
    chunks = splitter.split_documents(all_docs)

    _prog(f"🧠 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    _prog(f"💾 Building vector index for {len(chunks)} chunks...")
    db_client = chromadb.PersistentClient(path=DB_PATH)

    # Drop and recreate so re-indexing is always clean
    existing = [c.name for c in db_client.list_collections()]
    if cfg.collection_name in existing:
        db_client.delete_collection(cfg.collection_name)

    BATCH = 200
    db = None
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i: i + BATCH]
        if db is None:
            db = Chroma.from_documents(
                batch, embeddings,
                persist_directory=DB_PATH,
                collection_name=cfg.collection_name,
            )
        else:
            db.add_documents(batch)
        pct = min(i + BATCH, len(chunks))
        _prog(f"💾 Indexed {pct}/{len(chunks)} chunks...")

    mark_indexed(cfg.session_id)
    return True, (
        f"✅ Indexed **{len(chunks)} chunks** from **{len(all_docs)} files** "
        f"across **{len(cfg.paths)} project(s)**.\n\n"
        f"You can now ask questions about your codebase!"
    )


def get_session_db(cfg: WorkspaceConfig):
    """
    Return a Chroma retriever for an already-indexed session.
    Returns None if the session has not been indexed yet.
    """
    if not cfg.indexed:
        return None
    try:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from config import EMBEDDING_MODEL, EMBEDDING_DEVICE, DB_PATH

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        return Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings,
            collection_name=cfg.collection_name,
        )
    except Exception:
        return None


# ── New project scaffolding ───────────────────────────────────────────────────

def scaffold_new_project(path_str: str, project_type: str = "auto") -> tuple[bool, str]:
    """
    Create the directory structure for a new project.

    project_type hints:
      "auto"    — detect from path name (if name contains 'spring', 'react', etc.)
      "generic" — just mkdir

    Returns (success, message).
    """
    try:
        p = Path(path_str)
        p.mkdir(parents=True, exist_ok=True)
        # Create a minimal .poshak file so the agent knows it's a managed project
        marker = p / ".poshak"
        if not marker.exists():
            marker.write_text(
                f"# Poshak-AI project marker\n"
                f"# Created: {__import__('datetime').datetime.utcnow().isoformat()}\n"
            )
        return True, f"✅ Created project directory: `{p.resolve()}`"
    except OSError as e:
        return False, f"❌ Could not create directory `{path_str}`: {e}"


# ── Display helpers ───────────────────────────────────────────────────────────

GUIDANCE_MESSAGE = """
## 👋 Welcome to Poshak-AI

I'm your personal coding assistant. Here's how to get started:

---

### 🗂️ Work on an existing project

Open the **Workspace** panel in the sidebar and paste your project path(s).

**Examples:**
```
D:\\My Projects\\BookCatalogProject
```
```
backend: D:\\Projects\\Api
frontend: D:\\Projects\\Web
```
```
/home/user/myapp
```

Multiple paths? Separate them with a new line or semicolon.  
Want to label them? Use `label: /path/to/project` format.

---

### 🆕 Create a new project

Switch mode to **New Project** and provide the folder where I should create it.  
I'll help you scaffold it from scratch.

---

### 💬 Just chat (no project)

No path needed — just start typing. I can answer general coding questions,  
explain concepts, review code snippets you paste directly, and more.

---

*Tip: Your workspace is saved per chat session. Switch sessions to switch projects.*
"""


def format_workspace_badge(cfg: WorkspaceConfig) -> str:
    """Short badge text for the top bar."""
    if cfg.mode == "none" or not cfg.paths:
        return "💬 General chat"
    names = list(cfg.paths.keys())
    if len(names) == 1:
        return f"📁 {names[0]}"
    return f"📁 {names[0]} +{len(names)-1} more"