# url_reader.py — Poshak-AI
# ─────────────────────────
# Fetches and extracts clean text content from any URL.
# Handles: HTML pages, GitHub raw files, PDFs, JS-rendered pages (playwright fallback).
#
# Used by agent.py's read_url tool.

import re
import hashlib
from pathlib import Path
from typing import Optional

import httpx

# ── Constants ─────────────────────────────────────────────────────────────────

_MAX_CONTENT_CHARS = 24_000   # ~6k tokens — safe for LLM context
_MIN_CONTENT_CHARS = 400      # Below this → try playwright fallback
_CACHE_DIR = Path(".url_cache")
_CACHE_TTL_HOURS = 6          # Re-fetch after 6 hours

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

_TAGS_TO_REMOVE = {
    "script", "style", "nav", "footer", "header", "aside",
    "advertisement", "iframe", "noscript", "form", "meta", "link",
    "button", "input", "select", "textarea",
}

_CONTENT_TAGS = {
    "main", "article", "section", "div", "pre", "code",
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li", "table", "tr", "td", "th",
    "blockquote", "details", "summary",
}


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_url_content(url: str, user_question: str = "") -> str:
    """
    Fetch and return clean text content from a URL.

    Strategy:
      1. GitHub blob URL → rewrite to raw.githubusercontent.com
      2. PDF → extract text pages with pypdf
      3. HTML → httpx + BeautifulSoup extraction
      4. If content < 400 chars → playwright fallback (JS-rendered pages)

    Returns clean text, truncated to ~6k tokens if needed.
    Raises ValueError on permanent failures.
    """
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # ── Cache check ────────────────────────────────────────────────────────────
    cached = _read_cache(url)
    if cached:
        return _maybe_trim(cached, user_question)

    # ── GitHub blob → raw ──────────────────────────────────────────────────────
    url = _rewrite_github_url(url)

    # ── PDF ───────────────────────────────────────────────────────────────────
    if _is_pdf_url(url):
        content = _fetch_pdf(url)
        _write_cache(url, content)
        return _maybe_trim(content, user_question)

    # ── HTML ──────────────────────────────────────────────────────────────────
    content = _fetch_html(url)

    # ── Playwright fallback for JS-rendered pages ──────────────────────────────
    if len(content) < _MIN_CONTENT_CHARS:
        pw_content = _fetch_with_playwright(url)
        if pw_content and len(pw_content) > len(content):
            content = pw_content

    if not content.strip():
        raise ValueError(f"Could not extract meaningful content from {url}")

    _write_cache(url, content)
    return _maybe_trim(content, user_question)


# ── GitHub URL rewriter ────────────────────────────────────────────────────────

def _rewrite_github_url(url: str) -> str:
    """
    github.com/owner/repo/blob/branch/path/file
    → raw.githubusercontent.com/owner/repo/branch/path/file
    """
    m = re.match(
        r"https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)",
        url
    )
    if m:
        owner, repo, branch, path = m.groups()
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    return url


def _is_pdf_url(url: str) -> bool:
    clean = url.split("?")[0].lower()
    return clean.endswith(".pdf")


# ── HTML fetcher ───────────────────────────────────────────────────────────────

def _fetch_html(url: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return _fetch_html_raw(url)

    try:
        with httpx.Client(headers=_HEADERS, follow_redirects=True, timeout=20) as client:
            resp = client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception as e:
        raise ValueError(f"Failed to fetch {url}: {e}")

    soup = BeautifulSoup(html, "html.parser")

    # Remove noise
    for tag in soup.find_all(_TAGS_TO_REMOVE):
        tag.decompose()

    # Try to find main content area
    content_el = (
        soup.find("main") or
        soup.find("article") or
        soup.find(id=re.compile(r"content|main|docs|article", re.I)) or
        soup.find(class_=re.compile(r"content|main|docs|article|post", re.I))
    )
    target = content_el if content_el else soup.body or soup

    lines = []
    for el in target.find_all(True):
        if el.name in _CONTENT_TAGS:
            text = el.get_text(separator=" ", strip=True)
            if text and len(text) > 20:
                lines.append(text)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_lines: list[str] = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)

    return "\n\n".join(unique_lines)


def _fetch_html_raw(url: str) -> str:
    """Fallback: return raw text without BeautifulSoup."""
    try:
        with httpx.Client(headers=_HEADERS, follow_redirects=True, timeout=20) as client:
            resp = client.get(url)
            resp.raise_for_status()
            # Strip tags naively
            text = re.sub(r"<[^>]+>", " ", resp.text)
            text = re.sub(r"\s+", " ", text)
            return text.strip()
    except Exception as e:
        raise ValueError(f"Failed to fetch {url}: {e}")


# ── PDF fetcher ────────────────────────────────────────────────────────────────

def _fetch_pdf(url: str) -> str:
    try:
        import pypdf
    except ImportError:
        raise ValueError("pypdf is not installed. Cannot read PDF URLs.")

    try:
        with httpx.Client(headers=_HEADERS, follow_redirects=True, timeout=30) as client:
            resp = client.get(url)
            resp.raise_for_status()
            pdf_bytes = resp.content
    except Exception as e:
        raise ValueError(f"Failed to download PDF from {url}: {e}")

    import io
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    pages_text: list[str] = []
    for i, page in enumerate(reader.pages[:30]):          # max 30 pages
        text = page.extract_text() or ""
        if text.strip():
            pages_text.append(f"[Page {i + 1}]\n{text.strip()}")

    return "\n\n".join(pages_text)


# ── Playwright fallback ────────────────────────────────────────────────────────

def _fetch_with_playwright(url: str) -> str:
    """
    Use headless Chromium for JS-rendered pages (React docs, Next.js docs, etc.)
    Only called when plain httpx returns < 400 chars.
    """
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=25_000, wait_until="domcontentloaded")
            page.wait_for_timeout(1500)   # give JS time to render
            content = page.inner_text("body")
            browser.close()
            return content.strip()
    except Exception:
        return ""


# ── Content trimming ───────────────────────────────────────────────────────────

def _maybe_trim(content: str, user_question: str = "") -> str:
    """
    Trim content to _MAX_CONTENT_CHARS.
    If a user question is provided, score paragraphs by keyword overlap
    and keep the most relevant sections first, then append remaining.
    """
    if len(content) <= _MAX_CONTENT_CHARS:
        return content

    if not user_question:
        return content[:_MAX_CONTENT_CHARS] + "\n\n[... content truncated — ask for a specific section if you need more]"

    # Keyword-score paragraphs
    keywords = set(re.findall(r"\b\w{4,}\b", user_question.lower()))
    paragraphs = content.split("\n\n")

    def score(p: str) -> int:
        words = set(re.findall(r"\b\w{4,}\b", p.lower()))
        return len(words & keywords)

    scored = sorted(enumerate(paragraphs), key=lambda x: score(x[1]), reverse=True)
    # Reconstruct with high-relevance sections first
    top: list[str] = []
    total = 0
    used: set[int] = set()
    for idx, para in scored:
        if total + len(para) > _MAX_CONTENT_CHARS:
            break
        top.append(para)
        used.add(idx)
        total += len(para) + 2

    # Append remaining in order (truncated)
    rest = [p for i, p in enumerate(paragraphs) if i not in used]
    rest_text = "\n\n".join(rest)
    if rest_text:
        remaining = _MAX_CONTENT_CHARS - total
        if remaining > 200:
            top.append(rest_text[:remaining] + "\n[... further content truncated]")

    return "\n\n".join(top)


# ── Cache ──────────────────────────────────────────────────────────────────────

def _cache_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _read_cache(url: str) -> Optional[str]:
    import time
    _CACHE_DIR.mkdir(exist_ok=True)
    path = _CACHE_DIR / (_cache_key(url) + ".txt")
    if not path.exists():
        return None
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    if age_hours > _CACHE_TTL_HOURS:
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def _write_cache(url: str, content: str) -> None:
    _CACHE_DIR.mkdir(exist_ok=True)
    path = _CACHE_DIR / (_cache_key(url) + ".txt")
    try:
        path.write_text(content, encoding="utf-8")
    except OSError:
        pass