# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# ── Optional default project (used only for CLI / index_project.py) ───────────
#
# The UI (ui.py) no longer uses ACTIVE_PROJECT at runtime.
# Workspaces are configured per-session from the sidebar.
# These settings are kept for backward compatibility with
# `python index_project.py` and `python agent.py` (CLI mode).
#
# Leave ACTIVE_PROJECT as "" to start the UI with no default project.
# The user will configure paths from the Workspace panel.

PROJECTS = {
    # "book-catalog-app": r"D:\My Projects\BookCatalogProject",
}

PROJECT_GROUPS = {
    # "book-catalog-full": {
    #     "backend":  r"D:\My Projects\BookCatalogProject",
    #     "frontend": r"D:\My Projects\BookCatalogFrontend",
    # },
}

# Leave empty ("") to start in general-chat mode with no default project.
# Set to a key in PROJECTS or PROJECT_GROUPS to preload a project in CLI mode.
ACTIVE_PROJECT = os.getenv("POSHAK_ACTIVE_PROJECT", "")

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ── LLM settings ──────────────────────────────────────────────────────────────
GEMINI_MODEL        = "gemini-3.1-flash-lite"
GEMMA_26B_MODEL     = "gemma-4-26b-a4b-it"
GEMMA_31B_MODEL     = "gemma-4-31b-it"
GROQ_MODEL_PRIMARY  = "llama-3.3-70b-versatile"
GROQ_MODEL_FALLBACK = "llama-3.1-8b-instant"
PREFERRED_LLM       = "gemini"
TEMPERATURE         = 0.1
MAX_TOKENS          = 4096

# Display name for UI status panel — updated dynamically in agent.py
LLM_MODEL = GEMINI_MODEL if PREFERRED_LLM == "gemini" else GROQ_MODEL_PRIMARY

# ── Embedding model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"

# ── ChromaDB vector store ─────────────────────────────────────────────────────
# Each session gets its own named collection inside this directory.
DB_PATH         = "./db"
COLLECTION_NAME = "codebase"   # legacy default (used by index_project.py CLI)

# ── Persistent chat + workspace history ──────────────────────────────────────
CONVERSATIONS_DB = "./conversations.db"

# ── RAG / retrieval settings ──────────────────────────────────────────────────
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 200
RETRIEVAL_K   = 8

# ── File types to index ───────────────────────────────────────────────────────
JAVA_EXTENSIONS = [
    "**/*.java", "**/*.xml", "**/*.properties",
    "**/*.yml", "**/*.yaml", "**/*.sql",
]
TS_EXTENSIONS = [
    "**/*.ts", "**/*.tsx", "**/*.js", "**/*.jsx",
    "**/*.json", "**/*.css", "**/*.scss", "**/*.html",
]
PYTHON_EXTENSIONS = [
    "**/*.py", "**/*.pyi",
]
COMMON_EXTENSIONS = [
    "**/*.md", "**/*.env.example", "**/*.sh",
    "**/*.dockerfile", "**/Dockerfile", "**/*.toml",
]
ALL_EXTENSIONS = JAVA_EXTENSIONS + TS_EXTENSIONS + PYTHON_EXTENSIONS + COMMON_EXTENSIONS

# ── Folders to skip during indexing ──────────────────────────────────────────
EXCLUDE_DIRS = [
    "target", "build", "dist", "out", ".next",
    "node_modules", ".mvn",
    ".git", ".idea", ".vscode", ".eclipse",
    "__pycache__", ".venv", "venv",
    "coverage", "htmlcov", ".nyc_output",
]

# ── Conversation memory ───────────────────────────────────────────────────────
MEMORY_WINDOW = 15

# ── Logging ───────────────────────────────────────────────────────────────────
VERBOSE = False


# ── Legacy helpers (used by index_project.py CLI) ─────────────────────────────

def get_active_project_paths() -> dict[str, str]:
    """Returns {label: path} for the CLI-configured ACTIVE_PROJECT."""
    if ACTIVE_PROJECT in PROJECT_GROUPS:
        return PROJECT_GROUPS[ACTIVE_PROJECT]
    if ACTIVE_PROJECT in PROJECTS:
        return {"main": PROJECTS[ACTIVE_PROJECT]}
    if not ACTIVE_PROJECT:
        return {}
    raise ValueError(
        f"'{ACTIVE_PROJECT}' not found in PROJECTS or PROJECT_GROUPS.\n"
        f"Available: {list(PROJECTS.keys()) + list(PROJECT_GROUPS.keys())}"
    )


def get_active_project_path() -> str:
    """Returns the first (or only) path."""
    paths = get_active_project_paths()
    if not paths:
        return str(__import__("pathlib").Path.cwd())
    return list(paths.values())[0]


def validate_config() -> None:
    """Runs on startup — checks API keys. Path validation is now per-session."""
    errors = []

    if not GROQ_API_KEY and not GEMINI_API_KEY:
        errors.append("At least one of GROQ_API_KEY or GEMINI_API_KEY must be set in .env")
    if not TAVILY_API_KEY:
        errors.append("TAVILY_API_KEY is missing — web search will be disabled")

    if errors:
        print("\n[!] Config warnings:\n")
        for e in errors:
            print(f"    • {e}")
        print()
        # Only hard-exit if no LLM key at all
        if not GROQ_API_KEY and not GEMINI_API_KEY:
            raise SystemExit(1)