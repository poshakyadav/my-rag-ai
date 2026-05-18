"""
api/core/startup.py
────────────────────
Called once at application startup (from lifespan context manager).
Mirrors what ui.py did in its module-level code.
"""

import sys
from pathlib import Path

# Make project root importable regardless of how uvicorn is invoked.
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def run_startup() -> None:
    """Initialise all singletons required before the first request."""
    from config import validate_config
    from chat_store import init_db
    from workspace_manager import init_workspace_db
    from agent import warmup

    print()
    print("-" * 56)
    print("  Poshak-AI API — Initialising...")
    print("-" * 56)

    validate_config()
    init_db()
    init_workspace_db()
    warmup()

    print("-" * 56)
    print("  Poshak-AI API — Ready")
    print("-" * 56)
    print()