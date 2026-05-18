# docs_registry.py — Poshak-AI
# ──────────────────────────────
# Maps technology/framework names to their official documentation URLs.
# Used by the agent's system prompt to know which docs page to fetch
# before writing code for a given technology.
#
# Keys are lowercase, partial-match friendly (agent uses keyword search).

DOCS_REGISTRY: dict[str, str] = {
    # ── Frontend ───────────────────────────────────────────────────────────────
    "angular":          "https://angular.dev/overview",
    "react":            "https://react.dev/reference/react",
    "vue":              "https://vuejs.org/guide/introduction",
    "svelte":           "https://svelte.dev/docs/introduction",
    "nextjs":           "https://nextjs.org/docs",
    "nuxt":             "https://nuxt.com/docs/guide/concepts/auto-imports",
    "remix":            "https://remix.run/docs/en/main",
    "vite":             "https://vitejs.dev/guide/",
    "tailwind":         "https://tailwindcss.com/docs/installation",
    "typescript":       "https://www.typescriptlang.org/docs/handbook/2/basic-types.html",
    "rxjs":             "https://rxjs.dev/guide/overview",
    "ngrx":             "https://ngrx.io/guide/store",

    # ── Backend — Python ───────────────────────────────────────────────────────
    "fastapi":          "https://fastapi.tiangolo.com/tutorial/",
    "django":           "https://docs.djangoproject.com/en/stable/",
    "flask":            "https://flask.palletsprojects.com/en/latest/",
    "pydantic":         "https://docs.pydantic.dev/latest/",
    "sqlalchemy":       "https://docs.sqlalchemy.org/en/20/orm/quickstart.html",
    "alembic":          "https://alembic.sqlalchemy.org/en/latest/tutorial.html",
    "celery":           "https://docs.celeryq.dev/en/stable/getting-started/introduction.html",
    "httpx":            "https://www.python-httpx.org/quickstart/",
    "aiohttp":          "https://docs.aiohttp.org/en/stable/",

    # ── LangChain / LangGraph / AI ────────────────────────────────────────────
    "langchain":        "https://python.langchain.com/docs/introduction/",
    "langgraph":        "https://langchain-ai.github.io/langgraph/",
    "langsmith":        "https://docs.smith.langchain.com/",
    "chromadb":         "https://docs.trychroma.com/getting-started",
    "openai":           "https://platform.openai.com/docs/overview",
    "anthropic":        "https://docs.anthropic.com/en/docs/welcome",
    "gemini":           "https://ai.google.dev/gemini-api/docs",
    "groq":             "https://console.groq.com/docs/openai",
    "huggingface":      "https://huggingface.co/docs/transformers/index",
    "ollama":           "https://github.com/ollama/ollama/blob/main/docs/api.md",
    "tavily":           "https://docs.tavily.com/docs/python-sdk/getting-started",

    # ── Backend — Java / Spring ────────────────────────────────────────────────
    "spring boot":      "https://docs.spring.io/spring-boot/docs/current/reference/html/",
    "spring security":  "https://docs.spring.io/spring-security/reference/index.html",
    "spring data":      "https://docs.spring.io/spring-data/jpa/docs/current/reference/html/",
    "hibernate":        "https://docs.jboss.org/hibernate/orm/current/userguide/html_single/Hibernate_User_Guide.html",
    "maven":            "https://maven.apache.org/guides/getting-started/",
    "gradle":           "https://docs.gradle.org/current/userguide/userguide.html",

    # ── Databases ──────────────────────────────────────────────────────────────
    "postgresql":       "https://www.postgresql.org/docs/current/",
    "postgres":         "https://www.postgresql.org/docs/current/",
    "mysql":            "https://dev.mysql.com/doc/refman/8.0/en/",
    "mongodb":          "https://www.mongodb.com/docs/manual/",
    "redis":            "https://redis.io/docs/latest/",
    "sqlite":           "https://www.sqlite.org/docs.html",
    "prisma":           "https://www.prisma.io/docs/getting-started",
    "drizzle":          "https://orm.drizzle.team/docs/overview",

    # ── DevOps / Infra ─────────────────────────────────────────────────────────
    "docker":           "https://docs.docker.com/get-started/",
    "kubernetes":       "https://kubernetes.io/docs/home/",
    "nginx":            "https://nginx.org/en/docs/",
    "terraform":        "https://developer.hashicorp.com/terraform/docs",
    "github actions":   "https://docs.github.com/en/actions/quickstart",

    # ── Testing ───────────────────────────────────────────────────────────────
    "pytest":           "https://docs.pytest.org/en/stable/getting-started.html",
    "jest":             "https://jestjs.io/docs/getting-started",
    "vitest":           "https://vitest.dev/guide/",
    "cypress":          "https://docs.cypress.io/guides/overview/why-cypress",
    "playwright test":  "https://playwright.dev/docs/intro",
}


def find_docs_url(technology: str) -> list[tuple[str, str]]:
    """
    Fuzzy-search the registry for a technology name.
    Returns a list of (tech_name, docs_url) tuples that match.

    Example:
        find_docs_url("angular signals") → [("angular", "https://angular.dev/...")]
    """
    query = technology.lower().strip()
    results: list[tuple[str, str]] = []
    for key, url in DOCS_REGISTRY.items():
        if key in query or query in key or any(word in key for word in query.split()):
            results.append((key, url))
    return results[:3]   # return at most 3 matches


def get_docs_registry_summary() -> str:
    """Return a compact summary for injection into the system prompt."""
    categories = {
        "Frontend": ["angular", "react", "vue", "nextjs", "tailwind", "typescript"],
        "Python Backend": ["fastapi", "django", "pydantic", "sqlalchemy"],
        "AI/LLM": ["langchain", "langgraph", "chromadb", "openai", "gemini"],
        "Java/Spring": ["spring boot", "spring security", "hibernate"],
        "Databases": ["postgresql", "mongodb", "redis", "prisma"],
    }
    lines = []
    for cat, keys in categories.items():
        techs = [k for k in keys if k in DOCS_REGISTRY]
        lines.append(f"  {cat}: {', '.join(techs)}")
    return "\n".join(lines)