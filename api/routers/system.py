"""
api/routers/system.py
──────────────────────
System-level endpoints.

GET /api/system/status    — LLM, DB, session count, timestamp
GET /api/system/health    — simple liveness probe
GET /api/system/examples  — example prompts for the UI
GET /api/system/models    — available models with availability check  [NEW v4]
"""

from datetime import datetime
from fastapi import APIRouter

from api.schemas.system import SystemStatusOut, ExamplePromptsOut, ModelInfo, ModelsOut
from config import LLM_MODEL, DB_PATH, GEMINI_API_KEY, GROQ_API_KEY
from agent import get_active_llm_name
from chat_store import list_sessions
from model_selector import AVAILABLE_MODELS

router = APIRouter()

EXAMPLE_PROMPTS = [
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


@router.get("/health")
def health_check():
    """Kubernetes / Docker liveness probe — always returns 200 OK."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@router.get("/status", response_model=SystemStatusOut)
def system_status():
    """Return current system metrics for the status panel."""
    from pathlib import Path
    import chromadb

    llm_name = get_active_llm_name() or LLM_MODEL
    sessions = list_sessions(workspace="", limit=500)
    session_count = len(sessions)

    db_chunks = 0
    try:
        if Path(DB_PATH).exists():
            client = chromadb.PersistentClient(path=DB_PATH)
            cols = client.list_collections()
            for col in cols:
                db_chunks += col.count()
    except Exception:
        pass

    return SystemStatusOut(
        llm_model=llm_name,
        db_chunks=db_chunks,
        session_count=session_count,
        timestamp=datetime.now().strftime("%H:%M:%S"),
    )


@router.get("/examples", response_model=ExamplePromptsOut)
def get_example_prompts():
    """Return the list of example prompts shown in the UI sidebar."""
    return ExamplePromptsOut(prompts=EXAMPLE_PROMPTS)


@router.get("/models", response_model=ModelsOut)
def get_available_models():
    """
    Return all models with their availability status based on configured API keys.

    The frontend uses this to populate the model selector dropdown and grey out
    unavailable models (e.g. Groq models when GROQ_API_KEY is not set).
    """
    has_google = bool(GEMINI_API_KEY)
    has_groq = bool(GROQ_API_KEY)

    models: list[ModelInfo] = []
    for m in AVAILABLE_MODELS:
        provider = m["provider"]
        if provider == "auto":
            available = True
        elif provider == "google":
            available = has_google
        elif provider == "groq":
            available = has_groq
        else:
            available = False

        models.append(ModelInfo(
            id=m["id"],
            label=m["label"],
            description=m["description"],
            tier=m["tier"],
            provider=provider,
            available=available,
        ))

    current_default = get_active_llm_name() or LLM_MODEL
    return ModelsOut(models=models, default=current_default)