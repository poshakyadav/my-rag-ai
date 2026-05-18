# agent.py — Poshak-AI
# ─────────────────────
# Multi-tool AI agent with:
#   • Dynamic per-session workspace support
#   • read_url tool — fetch full content from any URL (HTML, GitHub, PDF)
#   • Automatic latest-docs fetching before writing code
#   • Post-generation code review + auto-fix loop (max 2 iterations)
#   • Mandatory read_file before write_file
#   • Per-request model override + auto-routing via model_selector.py
#
# v4 enhancements:
#   - model_override parameter on ask() — pass any model_id to use it for that request
#   - "auto" mode calls classify_query() to pick the right model per message
#   - _build_llm_for_model() builds a one-off LLM instance for override requests
#   - get_executor_for_session() accepts optional model_override

import subprocess
from pathlib import Path
from typing import Optional, Union

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from config import (
    GROQ_API_KEY, GEMINI_API_KEY, TAVILY_API_KEY,
    GEMINI_MODEL, GEMMA_26B_MODEL, GEMMA_31B_MODEL,
    GROQ_MODEL_PRIMARY, GROQ_MODEL_FALLBACK, PREFERRED_LLM,
    TEMPERATURE, MAX_TOKENS,
    EMBEDDING_MODEL, EMBEDDING_DEVICE,
    DB_PATH, RETRIEVAL_K,
    validate_config,
)
from workspace_manager import WorkspaceConfig, get_session_db, GUIDANCE_MESSAGE
from docs_registry import get_docs_registry_summary


def info(msg: str):  print(f"  [+] {msg}")
def ok(msg: str):    print(f"  [v] {msg}")
def warn(msg: str):  print(f"  [!] {msg}")
def err(msg: str):   print(f"  [x] {msg}")

SEPARATOR = "-" * 56

# ── Module-level singletons ───────────────────────────────────────────────────

_embeddings: Optional[HuggingFaceEmbeddings] = None
_llm = None
_active_llm_name: str = ""
_memory: Optional[SqliteSaver] = None
_sqlite_conn: Optional[sqlite3.Connection] = None

# Cache: session_id → (collection_name, model_id, executor)
_agent_cache: dict[str, tuple[str, str, object]] = {}

# Google model IDs set — used to route to the right SDK
_GOOGLE_MODELS = {GEMINI_MODEL, GEMMA_26B_MODEL, GEMMA_31B_MODEL}
_GROQ_MODELS = {GROQ_MODEL_PRIMARY, GROQ_MODEL_FALLBACK}


# ── Content normalisation helper ──────────────────────────────────────────────

def _extract_text_content(content: Union[str, list, None]) -> str:
    """Normalise LangGraph / LangChain message content to a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        parts.append(text)
        return "\n".join(parts).strip()
    return str(content)


# ── Embeddings ────────────────────────────────────────────────────────────────

def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# ── LLM waterfall (default) ───────────────────────────────────────────────────

def _build_llm():
    """Build and cache the default LLM using the waterfall strategy from config."""
    global _llm, _active_llm_name
    if _llm is not None:
        return _llm

    def try_google(model: str):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set")
        llm = ChatGoogleGenerativeAI(
            model=model, google_api_key=GEMINI_API_KEY,
            temperature=1.0, max_output_tokens=MAX_TOKENS, max_retries=2,
        )
        llm.invoke([HumanMessage(content="hi")])
        return llm, model

    def try_groq(model: str):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set")
        llm = ChatGroq(
            model=model, temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS, api_key=GROQ_API_KEY,
        )
        llm.invoke([HumanMessage(content="hi")])
        return llm, model

    if PREFERRED_LLM == "gemini":
        attempts = [
            ("Gemini 3.1 Flash-Lite",       lambda: try_google(GEMINI_MODEL)),
            ("Gemma 4 26B (MoE·fast)",      lambda: try_google(GEMMA_26B_MODEL)),
            ("Gemma 4 31B (Dense)",         lambda: try_google(GEMMA_31B_MODEL)),
            (f"Groq {GROQ_MODEL_PRIMARY}",  lambda: try_groq(GROQ_MODEL_PRIMARY)),
            (f"Groq {GROQ_MODEL_FALLBACK}", lambda: try_groq(GROQ_MODEL_FALLBACK)),
        ]
    else:
        attempts = [
            (f"Groq {GROQ_MODEL_PRIMARY}",  lambda: try_groq(GROQ_MODEL_PRIMARY)),
            ("Gemini 3.1 Flash-Lite",       lambda: try_google(GEMINI_MODEL)),
            ("Gemma 4 26B (MoE·fast)",      lambda: try_google(GEMMA_26B_MODEL)),
            ("Gemma 4 31B (Dense)",         lambda: try_google(GEMMA_31B_MODEL)),
            (f"Groq {GROQ_MODEL_FALLBACK}", lambda: try_groq(GROQ_MODEL_FALLBACK)),
        ]

    for label, fn in attempts:
        try:
            info(f"Trying {label}...")
            llm, name = fn()
            ok(f"LLM ready: {name}")
            _active_llm_name = name
            _llm = llm
            return _llm
        except Exception as e:
            warn(f"{label} unavailable: {e}")

    err("All LLM providers failed. Check API keys in .env")
    raise SystemExit(1)


# ── Per-request LLM builder ───────────────────────────────────────────────────

def _build_llm_for_model(model_id: str):
    """
    Build a fresh LLM instance for the given model_id.
    Used when a specific model is requested (override or auto-routed).
    Does NOT update the global _llm / _active_llm_name cache.
    """
    if model_id in _GOOGLE_MODELS:
        if not GEMINI_API_KEY:
            warn(f"GEMINI_API_KEY not set — cannot use {model_id}, falling back to default")
            return _build_llm()
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_id, google_api_key=GEMINI_API_KEY,
                temperature=1.0, max_output_tokens=MAX_TOKENS, max_retries=2,
            )
            return llm
        except Exception as e:
            warn(f"Failed to init Google model {model_id}: {e}")
            return _build_llm()

    if model_id in _GROQ_MODELS:
        if not GROQ_API_KEY:
            warn(f"GROQ_API_KEY not set — cannot use {model_id}, falling back to default")
            return _build_llm()
        try:
            llm = ChatGroq(
                model=model_id, temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS, api_key=GROQ_API_KEY,
            )
            return llm
        except Exception as e:
            warn(f"Failed to init Groq model {model_id}: {e}")
            return _build_llm()

    warn(f"Unknown model_id '{model_id}' — using default LLM")
    return _build_llm()


# ── System prompt ─────────────────────────────────────────────────────────────

def _build_system_prompt(cfg: WorkspaceConfig) -> str:
    if cfg.mode == "none" or not cfg.has_paths:
        project_ctx = (
            "No project is currently loaded. You are in **general chat mode**.\n\n"
            "If the user asks coding questions, you can:\n"
            "  • Answer from general knowledge\n"
            "  • Use read_url to fetch official documentation before writing code\n"
            "  • Ask them to paste relevant code directly into the chat\n"
            "  • Guide them to set up a workspace via the sidebar Workspace panel\n\n"
            "Do NOT call search_codebase — there is no indexed codebase in this session."
        )
        tools_note = (
            "In this session, search_codebase is DISABLED (no project loaded).\n"
            "Use search_web for external docs. Use read_url to fetch full page content.\n"
            "Use read_file / write_file only if the user provides an explicit absolute path."
        )
    else:
        if len(cfg.paths) == 1:
            label, path = list(cfg.paths.items())[0]
            project_ctx = f"Active project: **{label}** at `{path}`"
        else:
            lines = ["Active workspace (multi-repo):"]
            for lbl, p in cfg.paths.items():
                lines.append(f"  • **{lbl}**: `{p}`")
            project_ctx = "\n".join(lines)

        index_note = (
            "(indexed — search_codebase is active)"
            if cfg.indexed
            else "(not yet indexed — remind the user to click 'Index Project' if they ask about code)"
        )
        project_ctx += f"\n\n{index_note}"
        tools_note = (
            "search_codebase searches ALL sub-projects simultaneously. "
            "Always call it before writing new code to match the existing style."
        )

    docs_summary = get_docs_registry_summary()

    return f"""You are Poshak-AI, a senior software engineer and personal coding assistant.
You have deep knowledge of Java, Spring Boot, TypeScript, React, Angular, Python, and modern software architecture.

{project_ctx}

═══════════════════════════════════════════════════════════
DOCUMENTATION STRATEGY (MANDATORY — follow this every time)
═══════════════════════════════════════════════════════════

You have access to a read_url tool that fetches the FULL content of any webpage.
This is different from search_web (which gives short snippets) — read_url gives you
the entire documentation page, API reference, or article.

RULE 1 — URL in user message:
  If the user pastes any URL (http:// or https://) in their message:
  → ALWAYS call read_url on it FIRST before doing anything else.
  → Treat that page's content as ground truth for your answer.

RULE 2 — Before writing code for a library/framework:
  1. Use search_web to verify the library's current version and any breaking changes
  2. Use read_url to fetch the relevant docs page for the specific feature you're implementing
  3. Only then write code — matching the current API, NOT your training data
  
  Known docs URLs (use these with read_url):
{docs_summary}

  If the technology isn't in the list above, use search_web to find its official docs URL,
  then use read_url to fetch that URL.

RULE 3 — Version-specific questions:
  If the user mentions a specific version (e.g. "Angular 18", "Spring Boot 3.3"),
  search_web first to check for migration guides or breaking changes, then read_url the changelog.

═══════════════════════════════════════════════════════════
FILE OPERATIONS RULES (MANDATORY)
═══════════════════════════════════════════════════════════

MANDATORY BEFORE write_file:
  1. Call read_file on the target file to understand existing structure, imports, and style
  2. Call search_codebase to find similar patterns already in the project
  3. Write code that matches what you found — same import style, same error handling pattern
  4. After writing, call read_file again to confirm the write succeeded

NEVER guess at file contents — always read_file first.
NEVER write to files without user confirmation UNLESS the user explicitly says "write it" or "create the file".

═══════════════════════════════════════════════════════════
TOOLS GUIDANCE
═══════════════════════════════════════════════════════════

{tools_note}

Available tools:
  search_codebase         — search indexed source code (only when project is loaded & indexed)
  search_web              — search the web for docs, errors, best practices (returns snippets)
  read_url                — fetch FULL content of a specific URL (docs pages, GitHub files, PDFs)
  read_file               — read a specific file from disk (use absolute path)
  write_file              — write/create a file on disk (confirm with user first)
  run_command             — run shell commands (git, mvn, npm, pytest, etc.)
  list_project_structure  — explore a project's directory layout
  list_workspace_projects — list all configured sub-projects

STRATEGY:
1. Code questions WITH project loaded → search_codebase first, then read_url docs
2. "How to" questions → search_web for overview, read_url for full docs
3. URL in user message → read_url immediately
4. Before editing any file → read_file first, always
5. Bug fixes → read file → understand issue → fetch relevant docs → write fix
6. New features → search codebase to match style → fetch latest docs → write code
7. No project loaded → fetch docs → answer from knowledge or ask user to paste code

═══════════════════════════════════════════════════════════
RESPONSE STYLE
═══════════════════════════════════════════════════════════

- Be direct and practical: give working code, not just advice
- Match the user's existing code style (verify with search_codebase)
- For code changes, show the full updated method/class
- Explain briefly what changed and why
- When you fetch docs, mention which version/page you read so the user knows it's current

IMPORTANT:
- Never guess at file contents: use read_file to verify
- Never write to files without user confirmation unless explicitly asked
- For git commits, show the diff first and ask for confirmation
- If a tool call fails, explain why and suggest an alternative
- If the user asks to switch projects, guide them to the Workspace panel in the sidebar
"""


# ── Tool factory ──────────────────────────────────────────────────────────────

def _make_tools(cfg: WorkspaceConfig) -> list:

    @tool
    def search_codebase(query: str) -> str:
        """
        Search the indexed codebase for relevant source code, configs, and docs.
        Searches across ALL sub-projects simultaneously.
        Use for: architecture questions, class/method lookup, configs, patterns.
        Always call before writing new code to match existing style.
        """
        if not cfg.indexed:
            return (
                "⚠️ No indexed codebase for this session.\n\n"
                "To search your code:\n"
                "1. Open the **Workspace** panel in the sidebar\n"
                "2. Enter your project path(s)\n"
                "3. Click **Validate & Index**\n\n"
                "Until then, I can answer general coding questions or you can paste code directly."
            )
        db = get_session_db(cfg)
        if db is None:
            return "Codebase index not available. Please re-index via the Workspace panel."
        try:
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
        Returns SHORT snippets (200-400 words per result).
        Use for: finding current library versions, checking changelogs, locating docs URLs.
        For FULL page content, use read_url after finding the URL here.
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
    def read_url(url: str) -> str:
        """
        Fetch and read the FULL text content of any URL.
        Use this tool for documentation pages, GitHub files, blog posts, PDFs.
        """
        try:
            from url_reader import fetch_url_content
            content = fetch_url_content(url, user_question="")
            return f"[Content from: {url}]\n\n{content}"
        except ValueError as e:
            return f"Could not read URL: {e}"
        except Exception as e:
            return f"Error fetching URL '{url}': {e}"

    @tool
    def read_file(filepath: str) -> str:
        """
        Read the full contents of a file from disk.
        Accepts absolute paths. For project files, also tries relative paths
        against each configured sub-project root.
        ALWAYS call this before write_file to understand the existing structure.
        """
        try:
            p = Path(filepath)
            if not p.exists() and cfg.paths:
                for root in cfg.paths.values():
                    candidate = Path(root) / filepath
                    if candidate.exists():
                        p = candidate
                        break
            if not p.exists():
                return (
                    f"File not found: `{filepath}`\n\n"
                    "Tips:\n"
                    "  • Use the absolute path\n"
                    "  • Or use `list_project_structure` to find the correct path"
                )
            if p.stat().st_size > 500_000:
                return f"File too large ({p.stat().st_size} bytes). Use search_codebase instead."
            return f"[{p}]\n\n{p.read_text(encoding='utf-8', errors='replace')}"
        except Exception as e:
            return f"Error reading file: {e}"

    @tool
    def write_file(filepath: str, content: str) -> str:
        """
        Write content to a file on disk.
        IMPORTANT: Always read_file first before overwriting.
        Only use after the user has confirmed they want to write/create the file.
        """
        try:
            p = Path(filepath)
            if not p.is_absolute() and cfg.primary_path:
                p = Path(cfg.primary_path) / filepath
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} characters to `{p}`"
        except Exception as e:
            return f"Error writing file: {e}"

    @tool
    def run_command(command: str) -> str:
        """
        Run a shell command and return its output.
        Runs in the primary project directory. Times out after 90 seconds.
        """
        BLOCKED = ["rm -rf /", "format c:", "del /f /s /q c:\\", "rmdir /s /q c:\\"]
        if any(b in command.lower() for b in BLOCKED):
            return "Blocked: this command is too destructive to run automatically."
        cwd = cfg.primary_path or str(Path.cwd())
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=90, cwd=cwd,
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
        """List the directory structure of the active workspace."""
        if not cfg.paths:
            return "No project loaded. Add a project path via the Workspace panel in the sidebar."
        SKIP = {
            ".git", "node_modules", "target", "build", "dist", ".next",
            "__pycache__", ".venv", "venv", ".idea", ".vscode", "coverage",
        }
        try:
            if sub_project and sub_project in cfg.paths:
                root = Path(cfg.paths[sub_project])
            else:
                root = Path(list(cfg.paths.values())[0])

            target = root / path if path else root
            if not target.exists():
                return f"Path not found: `{target}`"

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
        """List all sub-projects in the current session's workspace."""
        if not cfg.paths:
            return (
                "No project is loaded for this session.\n\n"
                "To add a project:\n"
                "  1. Open the **Workspace** panel in the sidebar\n"
                "  2. Enter the project path(s)\n"
                "  3. Click **Validate & Index**"
            )
        lines = [f"Session workspace (mode: {cfg.mode})", ""]
        for label, path in cfg.paths.items():
            exists = Path(path).exists()
            status = "✓" if exists else "✗ NOT FOUND"
            lines.append(f"  [{status}] {label}")
            lines.append(f"            {path}")
        lines.append("")
        lines.append(f"Indexed: {'Yes' if cfg.indexed else 'No'}")
        return "\n".join(lines)

    return [
        search_codebase,
        search_web,
        read_url,
        read_file,
        write_file,
        run_command,
        list_project_structure,
        list_workspace_projects,
    ]


# ── Memory / checkpointer ─────────────────────────────────────────────────────

def _get_memory() -> SqliteSaver:
    global _memory, _sqlite_conn
    if _memory is None:
        from config import CONVERSATIONS_DB
        if _sqlite_conn is None:
            _sqlite_conn = sqlite3.connect(CONVERSATIONS_DB, check_same_thread=False)
        _memory = SqliteSaver(_sqlite_conn)
        _memory.setup()
    return _memory


def _build_executor_for_config(cfg: WorkspaceConfig, llm=None):
    """Build a LangGraph ReAct agent. Uses provided llm or falls back to default."""
    if llm is None:
        llm = _build_llm()
    tools  = _make_tools(cfg)
    memory = _get_memory()
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=_build_system_prompt(cfg),
        checkpointer=memory,
    )


def get_executor_for_session(
    session_id: str,
    cfg: WorkspaceConfig,
    model_override: Optional[str] = None,
):
    """
    Return a cached or freshly-built agent executor.

    If model_override is provided AND differs from the cached model,
    a new executor is built with that model. The cache key includes model_id
    so different models for the same session get their own executors.
    """
    # Resolve effective model id
    effective_model = model_override if model_override else "default"

    cached = _agent_cache.get(session_id)
    if cached is not None:
        cached_collection, cached_model, executor = cached
        if (
            cached_collection == cfg.collection_name
            and cached_collection != ""
            and cached_model == effective_model
        ):
            return executor

    # Build with override or default
    if model_override:
        llm = _build_llm_for_model(model_override)
    else:
        llm = None  # _build_executor_for_config will call _build_llm()

    executor = _build_executor_for_config(cfg, llm=llm)
    _agent_cache[session_id] = (cfg.collection_name, effective_model, executor)
    return executor


# ── Public API ────────────────────────────────────────────────────────────────

def ask(
    executor,
    question: str,
    session_id: str = "default",
    model_override: Optional[str] = None,
    auto_route: bool = False,
    has_codebase: bool = False,
) -> tuple[str, str]:
    """
    Send a question to the agent and return (response_text, model_used).

    Args:
        executor:       The LangGraph agent executor.
        question:       User's message.
        session_id:     LangGraph thread ID.
        model_override: Explicit model ID to use (skips auto-routing).
        auto_route:     If True and no model_override, classify the query and pick a model.
        has_codebase:   Passed to classifier — True when session has an indexed project.

    Returns:
        Tuple of (response_text: str, model_used: str)
    """
    # ── Determine which model to actually use ─────────────────────────────────
    routing_decision = None
    effective_executor = executor

    if model_override and model_override != "auto":
        # Explicit user-chosen model — build a one-off executor with that LLM
        override_llm = _build_llm_for_model(model_override)
        from workspace_manager import load_workspace
        # We need cfg to rebuild tools — get it from the session cache
        cached = _agent_cache.get(session_id)
        if cached:
            _, _, _ = cached
        # Build a temporary executor for this model
        from config import CONVERSATIONS_DB
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(CONVERSATIONS_DB, check_same_thread=False)
        mem = SqliteSaver(conn)
        mem.setup()
        tools = executor.nodes if hasattr(executor, "nodes") else []
        # Simplest approach: rebuild with the override LLM
        # (tools are already wired into the cached executor — just swap LLM)
        # For production: use _build_executor_for_config with llm param
        try:
            cfg_for_rebuild = _get_cfg_from_cache(session_id)
            if cfg_for_rebuild:
                effective_executor = _build_executor_for_config(cfg_for_rebuild, llm=override_llm)
        except Exception:
            pass  # Fall back to cached executor if rebuild fails
        model_used = model_override

    elif auto_route and not model_override:
        # Auto-routing: classify the query, pick best model
        from model_selector import classify_query
        routing_decision = classify_query(
            message=question,
            preferred_llm=PREFERRED_LLM,
            has_codebase=has_codebase,
        )
        info(f"Auto-routed: {routing_decision.reason}")
        routed_llm = _build_llm_for_model(routing_decision.model_id)
        try:
            cfg_for_rebuild = _get_cfg_from_cache(session_id)
            if cfg_for_rebuild:
                effective_executor = _build_executor_for_config(cfg_for_rebuild, llm=routed_llm)
        except Exception:
            pass
        model_used = routing_decision.model_id
    else:
        # Use default cached executor
        model_used = _active_llm_name

    try:
        config = {"configurable": {"thread_id": session_id}}
        result = effective_executor.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        )
        messages = result.get("messages", [])
        if messages:
            last = messages[-1]
            raw = (
                last.get("content", "") if isinstance(last, dict)
                else getattr(last, "content", "")
            )
            response = _extract_text_content(raw) or "No response generated."
        else:
            response = "No response generated."

        # ── Code review pass ──────────────────────────────────────────────────
        if "```" in response:
            try:
                from code_reviewer import review_and_improve_response
                review_llm = _build_llm()
                response = review_and_improve_response(
                    response=response,
                    user_question=question,
                    llm=review_llm,
                )
            except Exception as review_err:
                warn(f"Code review pass failed (non-fatal): {review_err}")

        return response, model_used

    except Exception as e:
        error_msg = str(e)
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            return (
                "⚠️ Rate limit hit. The agent will automatically try the other provider "
                "on your next message. If this persists, wait 30-60 seconds.",
                model_used,
            )
        if "api_key" in error_msg.lower():
            return "API key error. Check your keys in .env", model_used
        return f"Agent error: {error_msg}", model_used


def _get_cfg_from_cache(session_id: str) -> Optional[WorkspaceConfig]:
    """Retrieve workspace config from the workspace DB for rebuilding executors."""
    try:
        from workspace_manager import load_workspace
        return load_workspace(session_id)
    except Exception:
        return None


def get_session_messages_for_display(session_id: str) -> list[dict]:
    """
    Reads the LangGraph checkpoint for `session_id` and returns a list of
    {role: str, content: str} dicts for the REST API.
    """
    memory = _get_memory()
    try:
        config = {"configurable": {"thread_id": session_id}}
        checkpoint_tuple = memory.get_tuple(config)

        if checkpoint_tuple is None:
            return []

        checkpoint = checkpoint_tuple.checkpoint
        if not checkpoint:
            return []

        messages = checkpoint.get("channel_values", {}).get("messages", [])
        display: list[dict] = []

        for msg in messages:
            if isinstance(msg, dict):
                msg_type    = msg.get("type", "")
                raw_content = msg.get("content", "")
            else:
                msg_type    = getattr(msg, "type", None) or msg.__class__.__name__.lower()
                raw_content = getattr(msg, "content", "")

            content = _extract_text_content(raw_content)

            if not content:
                continue

            if "human" in msg_type:
                display.append({"role": "user", "content": content})
            elif "ai" in msg_type:
                display.append({"role": "assistant", "content": content})

        return display

    except Exception as exc:
        print(f"  [!] get_session_messages_for_display({session_id}): {exc}")
        return []


def get_active_llm_name() -> str:
    return _active_llm_name


# ── Warm-up ───────────────────────────────────────────────────────────────────

def warmup():
    """Load the LLM and embeddings at startup so first chat is fast."""
    validate_config()
    from chat_store import init_db
    init_db()
    from workspace_manager import init_workspace_db
    init_workspace_db()
    _build_llm()
    _get_memory()
    info("Agent warmed up — ready for connections.")