# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# ── Single projects (index one repo at a time) ────────────────────────────────
PROJECTS = {
    "book-catalog-app-1": r"D:\My Projects\CloudflareDeployedProjects\BookCatalogProject",
    # "spring-app-2":     r"C:\Users\Me\projects\spring-app-2",
}

# ── Project GROUPS (index multiple repos as one workspace) ────────────────────
# Each group can contain backend + frontend + shared libs — all indexed together.
# The agent will search across ALL of them in one query.
#
# Example:
#   "book-catalog-full": {
#       "backend":  r"D:\My Projects\...\BookCatalogProject",
#       "frontend": r"D:\My Projects\...\BookCatalogFrontend",
#       "shared":   r"D:\My Projects\...\SharedLibs",
#   }
PROJECT_GROUPS = {
    "book-catalog-full": {
        "backend": r"D:\My Projects\CloudflareDeployedProjects\BookCatalogProject",
        # Uncomment and add more sub-projects as needed:
        # "frontend": r"D:\My Projects\...\BookCatalogFrontend",
        # "shared":   r"D:\My Projects\...\SharedLibs",
    },
}

# ── Active workspace ──────────────────────────────────────────────────────────
# Set to a key from PROJECTS (single) or PROJECT_GROUPS (multi-repo).
# If it's in PROJECT_GROUPS, all repos in that group are indexed together.
ACTIVE_PROJECT = "book-catalog-full"

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ── LLM settings ──────────────────────────────────────────────────────────────
# All Google models use your existing GEMINI_API_KEY — no new key needed.
#
# Model roster and why each exists:
#   gemini-3.1-flash-lite  → fast, free tier, 1M context, best for most queries
#   gemma-4-26b-a4b-it     → Unlimited TPM, MoE (fast as 4B), great for code
#   gemma-4-31b-it         → Unlimited TPM, #3 open model world, best quality
#   llama-3.3-70b          → Groq fallback (fast but aggressive rate limits)
#   llama-3.1-8b           → Groq last resort

# Primary model — tried first on every startup
GEMINI_MODEL = "gemini-3.1-flash-lite"

# Gemma models — also use GEMINI_API_KEY, no extra setup needed
GEMMA_26B_MODEL = "gemma-4-26b-a4b-it"   # MoE: fast + unlimited TPM
GEMMA_31B_MODEL = "gemma-4-31b-it"       # Dense: highest quality + unlimited TPM

# Groq fallback models
GROQ_MODEL_PRIMARY  = "llama-3.3-70b-versatile"
GROQ_MODEL_FALLBACK = "llama-3.1-8b-instant"

# Which LLM to try first: "gemini" or "groq"
# Gemini is recommended — free tier, 1M token context, no aggressive rate limits
PREFERRED_LLM = "gemini"

TEMPERATURE = 0.1
MAX_TOKENS  = 4096

# Display name for the UI status panel — updated dynamically in agent.py
# Expose the active model name for the UI status panel
LLM_MODEL = GEMINI_MODEL if PREFERRED_LLM == "gemini" else GROQ_MODEL_PRIMARY

# ── Embedding model (runs locally, no API needed) ─────────────────────────────
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"

# ── ChromaDB vector store ─────────────────────────────────────────────────────
DB_PATH         = "./db"

# ── Persistent chat history ───────────────────────────────────────────────────
CONVERSATIONS_DB = "./conversations.db"   # SQLite file, auto-created on first run

COLLECTION_NAME = "codebase"

# ── RAG / retrieval settings ──────────────────────────────────────────────────
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 200
RETRIEVAL_K   = 8     # chunks retrieved per query

# ── File types to index ───────────────────────────────────────────────────────
JAVA_EXTENSIONS = [
    "**/*.java", "**/*.xml", "**/*.properties",
    "**/*.yml", "**/*.yaml", "**/*.sql",
]
TS_EXTENSIONS = [
    "**/*.ts", "**/*.tsx", "**/*.js", "**/*.jsx",
    "**/*.json", "**/*.css", "**/*.scss", "**/*.html",
]
COMMON_EXTENSIONS = [
    "**/*.md", "**/*.env.example", "**/*.sh",
    "**/*.dockerfile", "**/Dockerfile",
]
ALL_EXTENSIONS = JAVA_EXTENSIONS + TS_EXTENSIONS + COMMON_EXTENSIONS

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_active_project_paths() -> dict[str, str]:
    """
    Returns a dict of {label: path} for the active workspace.
    Works for both single PROJECTS and multi-repo PROJECT_GROUPS.
    """
    # Check PROJECT_GROUPS first
    if ACTIVE_PROJECT in PROJECT_GROUPS:
        return PROJECT_GROUPS[ACTIVE_PROJECT]
    # Fall back to single project
    if ACTIVE_PROJECT in PROJECTS:
        return {"main": PROJECTS[ACTIVE_PROJECT]}
    raise ValueError(
        f"'{ACTIVE_PROJECT}' not found in PROJECTS or PROJECT_GROUPS in config.py.\n"
        f"Available: {list(PROJECTS.keys()) + list(PROJECT_GROUPS.keys())}"
    )


def get_active_project_path() -> str:
    """Returns the first (or only) path — used by tools that need a single cwd."""
    paths = get_active_project_paths()
    return list(paths.values())[0]


def validate_config() -> None:
    """Runs on startup — catches missing keys and bad paths early."""
    errors = []

    if not GROQ_API_KEY and not GEMINI_API_KEY:
        errors.append("At least one of GROQ_API_KEY or GEMINI_API_KEY must be set in .env")
    if not TAVILY_API_KEY:
        errors.append("TAVILY_API_KEY is missing from your .env file")

    try:
        paths = get_active_project_paths()
        for label, path in paths.items():
            if not os.path.isdir(path):
                errors.append(
                    f"Sub-project '{label}' path does not exist: {path}\n"
                    f"  Fix: update config.py with the correct path"
                )
    except ValueError as e:
        errors.append(str(e))

    if errors:
        print("\n[!] Config errors found:\n")
        for e in errors:
            print(f"    • {e}")
        print()
        raise SystemExit(1)