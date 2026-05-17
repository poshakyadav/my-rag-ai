# agent.py

"""
agent.py - Poshak-AI
---------------------
Multi-tool AI agent with:
  - Gemini 2.5 Flash (primary, free, 1M context)
  - Groq Llama 3.3 70B (fallback)
  - ChromaDB RAG across multiple projects
  - Tavily web search
  - File read/write tools
  - Shell command runner
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Optional

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from config import (
    GROQ_API_KEY,
    GEMINI_API_KEY,
    TAVILY_API_KEY,
    GEMINI_MODEL,
    GEMMA_26B_MODEL,
    GEMMA_31B_MODEL,  
    GROQ_MODEL_PRIMARY,
    GROQ_MODEL_FALLBACK,
    PREFERRED_LLM,
    TEMPERATURE,
    MAX_TOKENS,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    DB_PATH,
    COLLECTION_NAME,
    RETRIEVAL_K,
    MEMORY_WINDOW,
    VERBOSE,
    ACTIVE_PROJECT,
    PROJECTS,
    PROJECT_GROUPS,
    get_active_project_paths,
    get_active_project_path,
    validate_config,
)


def info(msg: str):  print(f"  [+] {msg}")
def ok(msg: str):    print(f"  [v] {msg}")
def warn(msg: str):  print(f"  [!] {msg}")
def err(msg: str):   print(f"  [x] {msg}")

SEPARATOR = "-" * 56

_embeddings: Optional[HuggingFaceEmbeddings] = None
_db: Optional[Chroma] = None
_agent = None
_memory: Optional[SqliteSaver] = None

# Track which LLM is currently active (for status panel)
_active_llm_name: str = ""

# ── module-level connection (lives for the app's lifetime) ────────────────────
_sqlite_conn: Optional[sqlite3.Connection] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def _get_db() -> Chroma:
    global _db
    if _db is None:
        db_path = Path(DB_PATH)
        if not db_path.exists():
            err(f"Vector DB not found at '{DB_PATH}'")
            err("Run  python index_project.py  first, then try again.")
            raise SystemExit(1)
        _db = Chroma(
            persist_directory=DB_PATH,
            embedding_function=_get_embeddings(),
            collection_name=COLLECTION_NAME,
        )
    return _db


def _build_llm():
    """
    Smart LLM waterfall — tries providers in order, skips failed ones.
    All Google models (Gemini + Gemma) use the same GEMINI_API_KEY.

    Order when PREFERRED_LLM = "gemini":
      1. gemini-3.1-flash-lite   (fast, free, 1M context)
      2. gemma-4-26b-a4b-it      (unlimited TPM, MoE speed)
      3. gemma-4-31b-it          (unlimited TPM, best quality)
      4. groq llama-3.3-70b      (rate-limited fallback)
      5. groq llama-3.1-8b       (last resort)
    """
    global _active_llm_name

    def try_google(model: str, label: str):
        """Works for both Gemini and Gemma models — same API key, same class."""
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set")
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=GEMINI_API_KEY,
            temperature=1.0,           # Google models work best at 1.0
            max_output_tokens=MAX_TOKENS,
            max_retries=2,
        )
        llm.invoke([HumanMessage(content="hi")])
        return llm, model

    def try_groq(model: str):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set")
        llm = ChatGroq(
            model=model,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            api_key=GROQ_API_KEY,
        )
        llm.invoke([HumanMessage(content="hi")])
        return llm, model

    # Build the waterfall based on preference
    if PREFERRED_LLM == "gemini":
        attempts = [
            (f"Gemini 3.1 Flash-Lite",    lambda: try_google(GEMINI_MODEL,    "gemini")),
            (f"Gemma 4 26B (MoE·fast)",   lambda: try_google(GEMMA_26B_MODEL, "gemma26")),
            (f"Gemma 4 31B (Dense·best)", lambda: try_google(GEMMA_31B_MODEL, "gemma31")),
            (f"Groq {GROQ_MODEL_PRIMARY}", lambda: try_groq(GROQ_MODEL_PRIMARY)),
            (f"Groq {GROQ_MODEL_FALLBACK}", lambda: try_groq(GROQ_MODEL_FALLBACK)),
        ]
    else:
        attempts = [
            (f"Groq {GROQ_MODEL_PRIMARY}", lambda: try_groq(GROQ_MODEL_PRIMARY)),
            (f"Gemini 3.1 Flash-Lite",    lambda: try_google(GEMINI_MODEL,    "gemini")),
            (f"Gemma 4 26B (MoE·fast)",   lambda: try_google(GEMMA_26B_MODEL, "gemma26")),
            (f"Gemma 4 31B (Dense·best)", lambda: try_google(GEMMA_31B_MODEL, "gemma31")),
            (f"Groq {GROQ_MODEL_FALLBACK}", lambda: try_groq(GROQ_MODEL_FALLBACK)),
        ]

    for label, fn in attempts:
        try:
            info(f"Trying {label}...")
            llm, name = fn()
            ok(f"LLM ready: {name}")
            _active_llm_name = name
            return llm
        except Exception as e:
            warn(f"{label} unavailable: {e}")

    err("All LLM providers failed. Check API keys in .env")
    raise SystemExit(1)


# ── Project context summary for the system prompt ─────────────────────────────

def _build_project_context() -> str:
    paths = get_active_project_paths()
    if len(paths) == 1:
        label, path = list(paths.items())[0]
        return f"Active project: {ACTIVE_PROJECT}  ({path})"
    lines = [f"Active workspace: {ACTIVE_PROJECT} (multi-repo)"]
    for label, path in paths.items():
        lines.append(f"  • {label}: {path}")
    return "\n".join(lines)


# ── TOOLS ─────────────────────────────────────────────────────────────────────

@tool
def search_codebase(query: str) -> str:
    """
    Search the indexed codebase for relevant source code, configs, and docs.
    This searches across ALL sub-projects in the active workspace simultaneously.
    Use this for ANY question about the project: architecture, classes,
    methods, configs, dependencies, patterns, or existing implementations.
    Always call this before writing new code so you match the existing style.
    """
    try:
        db = _get_db()
        docs = db.similarity_search(query, k=RETRIEVAL_K)
        if not docs:
            return "No relevant code found for this query."
        results = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            sub    = doc.metadata.get("sub_project", "")
            label  = f"{sub}/{source}" if sub else source
            results.append(f"[File: {label}]\n{doc.page_content}")
        return ("\n\n" + "-" * 40 + "\n\n").join(results)
    except Exception as e:
        return f"Codebase search error: {e}"


@tool
def search_web(query: str) -> str:
    """
    Search the live web for current information.
    Use for: latest docs, library versions, best practices, error messages,
    Stack Overflow answers, framework changelogs, or anything not in the codebase.
    """
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query=query, max_results=5)
        results = response.get("results", [])
        if not results:
            return "No web results found."
        output = []
        for r in results:
            output.append(f"Source: {r.get('url', 'unknown')}\n{r.get('content', '')}")
        return "\n\n---\n\n".join(output)
    except Exception as e:
        return f"Web search error: {e}"


@tool
def read_file(filepath: str) -> str:
    """
    Read the full contents of a file from the user's disk.
    Use when you need the exact current content of a specific file
    before editing it, reviewing it, or explaining it in detail.
    Accepts absolute paths or paths relative to any sub-project root.
    """
    try:
        path = Path(filepath)
        if not path.exists():
            # Try each sub-project root
            for label, root in get_active_project_paths().items():
                candidate = Path(root) / filepath
                if candidate.exists():
                    path = candidate
                    break
        if not path.exists():
            return f"File not found: {filepath}"
        if path.stat().st_size > 500_000:
            return f"File too large ({path.stat().st_size} bytes). Use search_codebase instead."
        return f"[{path}]\n\n{path.read_text(encoding='utf-8', errors='replace')}"
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def write_file(filepath: str, content: str) -> str:
    """
    Write content to a file on the user's disk.
    Always read the file first (with read_file) before overwriting it.
    Accepts absolute paths or paths relative to the primary sub-project root.
    """
    try:
        path = Path(filepath)
        if not path.is_absolute():
            path = Path(get_active_project_path()) / filepath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool
def run_command(command: str) -> str:
    """
    Run a shell command and return its output.
    Use for: git commands, build commands (mvn, npm), test runners (pytest, npm test).
    Runs in the primary sub-project directory. Commands time out after 90 seconds.
    """
    BLOCKED = ["rm -rf /", "format c:", "del /f /s /q c:\\", "rmdir /s /q c:\\"]
    if any(b in command.lower() for b in BLOCKED):
        return "Blocked: this command is too destructive to run automatically."
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=90, cwd=get_active_project_path(),
        )
        output = ((result.stdout or "") + (result.stderr or "")).strip()
        if not output:
            return f"Command ran successfully with no output. Exit code: {result.returncode}"
        if len(output) > 4000:
            return output[:2000] + "\n\n[... truncated ...]\n\n" + output[-1000:]
        return output
    except subprocess.TimeoutExpired:
        return "Command timed out after 90 seconds."
    except Exception as e:
        return f"Error running command: {e}"


@tool
def list_project_structure(sub_project: str = "", path: str = "") -> str:
    """
    List the directory structure of the active workspace.
    For multi-repo workspaces, pass sub_project='backend' or 'frontend' etc.
    to focus on a specific repo. Leave empty to list the primary repo.
    Pass path to list a sub-folder (e.g. path='src/main/java').
    """
    try:
        project_paths = get_active_project_paths()
        if sub_project and sub_project in project_paths:
            root = Path(project_paths[sub_project])
        else:
            root = Path(list(project_paths.values())[0])

        target = root / path if path else root
        if not target.exists():
            return f"Path not found: {target}"

        SKIP = {
            ".git", "node_modules", "target", "build", "dist", ".next",
            "__pycache__", ".venv", "venv", ".idea", ".vscode", "coverage",
        }
        lines = [str(target)]

        def walk(p: Path, prefix: str = "", depth: int = 0):
            if depth > 6:
                return
            try:
                entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
            except PermissionError:
                return
            for i, entry in enumerate(entries):
                if entry.name in SKIP:
                    continue
                connector = "L-- " if i == len(entries) - 1 else "+-- "
                lines.append(f"{prefix}{connector}{entry.name}")
                if entry.is_dir():
                    ext = "    " if i == len(entries) - 1 else "|   "
                    walk(entry, prefix + ext, depth + 1)
                if len(lines) > 150:
                    lines.append(f"{prefix}    ... (truncated)")
                    return

        walk(target)
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing structure: {e}"


@tool
def list_workspace_projects(_: str = "") -> str:
    """
    List all sub-projects in the active workspace with their paths.
    Use this first when the user asks about the overall project structure
    or when working in a multi-repo workspace.
    """
    paths = get_active_project_paths()
    lines = [f"Active workspace: {ACTIVE_PROJECT}", ""]
    for label, path in paths.items():
        exists = Path(path).exists()
        status = "✓" if exists else "✗ NOT FOUND"
        lines.append(f"  [{status}] {label}")
        lines.append(f"            {path}")
    return "\n".join(lines)


ALL_TOOLS = [
    search_codebase,
    search_web,
    read_file,
    write_file,
    run_command,
    list_project_structure,
    list_workspace_projects,
]


# ── System prompt ─────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    return f"""You are Poshak-AI, a senior software engineer and personal coding assistant.
You have deep knowledge of Java, Spring Boot, TypeScript, React, and modern software architecture.

{_build_project_context()}

You have access to these tools:
- search_codebase         : search across ALL indexed sub-projects simultaneously
- search_web              : search the live web for docs, best practices, errors
- read_file               : read a specific file from disk
- write_file              : write/create a file on disk
- run_command             : run git, mvn, npm, pytest, and other CLI commands
- list_project_structure  : explore a sub-project's directory layout
- list_workspace_projects : list all sub-projects in the active workspace

STRATEGY:
1. For code questions: call search_codebase first — it searches ALL sub-projects at once
2. For "how to" questions: combine search_codebase + search_web
3. Before writing or editing code: always read_file first
4. For bug fixes: read the file → understand the issue → write the fix
5. For new features: search codebase to match existing style, then write
6. For cross-project questions (e.g. "how does backend API connect to frontend"):
   use search_codebase — it returns results from all repos together

RESPONSE STYLE:
- Be direct and practical: give code, not just advice
- Match the user's existing code style (check with search_codebase)
- For code changes, show the full updated method/class, not just a snippet
- When referencing files, include the sub-project label if multi-repo
- Explain briefly what you did and why

IMPORTANT:
- Never guess at file contents: use read_file to verify
- Never write to files without user confirmation, unless explicitly asked
- For git commits, show the diff first and ask for confirmation
- If a tool call fails, explain why and suggest an alternative
"""


# ── AGENT BUILDER ─────────────────────────────────────────────────────────────

def build_agent(verbose: bool = False):
    global _memory, _sqlite_conn

    llm = _build_llm()

    info("Loading vector database...")
    _get_db()
    ok(f"Vector DB ready: {COLLECTION_NAME}")

    from config import CONVERSATIONS_DB
    from chat_store import init_db
    init_db()

    # Open one persistent connection — SqliteSaver wraps it directly.
    # check_same_thread=False is required: Gradio + LangGraph use multiple threads.
    if _sqlite_conn is None:
        _sqlite_conn = sqlite3.connect(CONVERSATIONS_DB, check_same_thread=False)

    _memory = SqliteSaver(_sqlite_conn)

    agent = create_agent(
        model=llm,
        tools=ALL_TOOLS,
        system_prompt=_build_system_prompt(),
        checkpointer=_memory,
    )

    return agent


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def ask(executor, question: str, session_id: str = "default") -> str:
    try:
        config = {"configurable": {"thread_id": session_id}}
        result = executor.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        )
        messages = result.get("messages", [])
        if messages:
            return messages[-1].content
        return "No response generated."
    except Exception as e:
        error_msg = str(e)
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            return (
                "⚠️ Rate limit hit. The agent will automatically try the other provider "
                "on your next message. If this persists, wait 30-60 seconds."
            )
        if "api_key" in error_msg.lower():
            return "API key error. Check your keys in .env"
        return f"Agent error: {error_msg}"


def clear_memory(session_id: str = "default") -> None:
    """
    For SqliteSaver, we can't easily delete individual thread checkpoints
    from the outside. The cleanest approach: just start a new session_id.
    This function is kept for API compatibility — the UI handles new sessions
    via chat_store.create_session() directly.
    """
    pass  # UI now manages this via chat_store — no-op here


def get_session_messages_for_display(session_id: str) -> list[dict]:
    """
    Reads LangGraph's checkpoint for a session and converts to
    Gradio display format: [{"role": "user/assistant", "content": "..."}]
    Used when loading an old session in the UI.
    """
    global _memory
    if _memory is None:
        return []
    try:
        config = {"configurable": {"thread_id": session_id}}
        checkpoint = _memory.get(config)
        if not checkpoint:
            return []

        messages = checkpoint.get("channel_values", {}).get("messages", [])
        display = []
        for msg in messages:
            # LangGraph messages have a 'type' attribute
            msg_type = getattr(msg, "type", None) or msg.__class__.__name__.lower()
            if "human" in msg_type:
                display.append({"role": "user", "content": msg.content})
            elif "ai" in msg_type and msg.content:
                # Skip empty AI messages (tool-calling intermediate steps)
                display.append({"role": "assistant", "content": msg.content})
        return display
    except Exception:
        return []


def get_executor(verbose: bool = False):
    global _agent
    if _agent is None:
        _agent = build_agent(verbose=verbose)
    return _agent


def get_active_llm_name() -> str:
    return _active_llm_name


# ── CLI ───────────────────────────────────────────────────────────────────────

def run_cli(verbose: bool = False):
    print()
    print(SEPARATOR)
    print("  Poshak-AI - Personal Dev Assistant")
    print(f"  Workspace : {ACTIVE_PROJECT}")
    print(SEPARATOR)

    validate_config()
    executor = build_agent(verbose=verbose)

    print()
    ok("Ready! Ask anything about your code.")
    print("  Type 'exit' or Ctrl+C to quit.")
    print("  Type 'clear' to reset conversation memory.")
    print("  Type 'help' for example questions.")
    print("  Type 'projects' to list sub-projects in this workspace.")
    print()

    while True:
        try:
            question = input("you > ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                print("\n  Goodbye!\n")
                break
            if question.lower() == "clear":
                clear_memory()
                print("  [v] Memory cleared.\n")
                continue
            if question.lower() == "projects":
                from config import get_active_project_paths
                for label, path in get_active_project_paths().items():
                    print(f"  • {label}: {path}")
                print()
                continue
            if question.lower() == "help":
                print("""
  Example questions:
    "List all sub-projects in this workspace"
    "How does the backend API connect to the frontend?"
    "Explain the overall structure of this project"
    "How is authentication implemented?"
    "Write a new endpoint for user registration"
    "What does UserService do?"
    "Run the tests and tell me what is failing"
    "Search the web for Spring Boot 3.3 migration guide"
                """)
                continue
            print()
            print("ai  > ", end="", flush=True)
            response = ask(executor, question)
            print(response)
            print()
        except KeyboardInterrupt:
            print("\n\n  Goodbye!\n")
            break
        except EOFError:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poshak-AI CLI")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_cli(verbose=args.verbose)