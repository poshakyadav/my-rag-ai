# code_reviewer.py — Poshak-AI
# ─────────────────────────────
# Post-generation code review + auto-fix loop.
# Extracts code blocks from a response, runs an LLM reviewer pass,
# fixes issues, and returns the improved response with a review badge.
#
# Max 2 review iterations to prevent infinite loops.
# Only runs when the response contains fenced code blocks (```).

import re
import json
from dataclasses import dataclass
from typing import Optional

MAX_REVIEW_ITERATIONS = 2


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class CodeBlock:
    language: str
    code: str
    raw: str        # the original fenced block string (for replacement)


# ── Public API ─────────────────────────────────────────────────────────────────

def review_and_improve_response(response: str, user_question: str, llm) -> str:
    """
    If `response` contains code blocks, run an LLM review pass on each block.
    Fixes found issues inline and appends a review badge.

    Args:
        response:      Full response string from the agent.
        user_question: Original user question (for context).
        llm:           The LangChain LLM instance to use for reviewing.

    Returns the improved response string (unchanged if no code found or no issues).
    """
    if "```" not in response:
        return response

    blocks = _extract_code_blocks(response)
    if not blocks:
        return response

    total_issues: list[str] = []
    improved = response

    for block in blocks:
        # Skip tiny snippets < 3 lines — not worth reviewing
        if block.code.count("\n") < 2:
            continue

        fixed_code, issues = _review_code_block(
            code=block.code,
            language=block.language,
            context=user_question,
            llm=llm,
        )

        if issues:
            total_issues.extend(issues)
            new_raw = f"```{block.language}\n{fixed_code}\n```"
            # Replace only the FIRST occurrence to be safe
            improved = improved.replace(block.raw, new_raw, 1)

    if total_issues:
        badge = _format_review_badge(total_issues)
        improved = improved.rstrip() + "\n\n" + badge

    return improved


# ── Code block extraction ──────────────────────────────────────────────────────

def _extract_code_blocks(text: str) -> list[CodeBlock]:
    """Extract all fenced code blocks from a response string."""
    pattern = re.compile(r"```([\w-]*)\n(.*?)```", re.DOTALL)
    blocks: list[CodeBlock] = []
    for m in pattern.finditer(text):
        lang = m.group(1).strip() or "text"
        code = m.group(2).rstrip()
        raw = m.group(0)
        blocks.append(CodeBlock(language=lang, code=code, raw=raw))
    return blocks


# ── LLM review pass ────────────────────────────────────────────────────────────

def _review_code_block(
    code: str,
    language: str,
    context: str,
    llm,
    iteration: int = 1,
) -> tuple[str, list[str]]:
    """
    Ask the LLM to review and fix the code.
    Returns (fixed_code, issues_found_and_fixed).
    Loops up to MAX_REVIEW_ITERATIONS if re-review still finds issues.
    """
    if iteration > MAX_REVIEW_ITERATIONS:
        return code, []

    prompt = _build_review_prompt(code, language, context)

    try:
        from langchain_core.messages import HumanMessage
        result = llm.invoke([HumanMessage(content=prompt)])
        raw_text = result.content if hasattr(result, "content") else str(result)
        # Normalise list-of-blocks response
        if isinstance(raw_text, list):
            raw_text = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in raw_text
            )
    except Exception as e:
        # If review LLM call fails, return original code untouched
        return code, []

    parsed = _parse_review_response(raw_text)
    if not parsed:
        return code, []

    fixed_code = parsed.get("code", code).strip()
    issues = [i for i in parsed.get("issues", []) if i and i.strip()]

    if not issues or not fixed_code or fixed_code == code.strip():
        return code, []

    # Recurse once more to verify the fix didn't introduce new issues
    if iteration < MAX_REVIEW_ITERATIONS:
        re_fixed, re_issues = _review_code_block(
            code=fixed_code,
            language=language,
            context=context,
            llm=llm,
            iteration=iteration + 1,
        )
        # If second pass found more issues, merge them
        if re_issues:
            issues.extend(re_issues)
            fixed_code = re_fixed

    return fixed_code, issues


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _build_review_prompt(code: str, language: str, context: str) -> str:
    return f"""You are a senior code reviewer. Review the following {language} code carefully.

User's request context: {context[:300]}

Code to review:
```{language}
{code}
```

Check for ALL of these issues:
1. Syntax errors or typos
2. Logic bugs or incorrect logic
3. Missing error handling (uncaught exceptions, missing null checks)
4. Missing imports or undefined variables/references
5. Security issues (SQL injection, XSS, hardcoded secrets, path traversal)
6. Obvious performance issues (N+1 queries, unnecessary loops)
7. Style inconsistencies or bad practices for {language}

IMPORTANT RULES:
- If the code is correct and has NO issues, return exactly: {{"code": "", "issues": []}}
- If issues exist, return the COMPLETE fixed code with ALL issues resolved
- Only return JSON, no markdown, no explanation outside the JSON
- The JSON must have exactly these keys: "code" (string) and "issues" (array of strings)
- Each issue string should be short: e.g. "Added missing null check for user param"
- Do NOT invent issues that don't exist — only report real problems

Return ONLY this JSON (no code fences, no preamble):
{{"code": "...", "issues": ["issue 1", "issue 2"]}}"""


# ── Response parser ────────────────────────────────────────────────────────────

def _parse_review_response(raw: str) -> Optional[dict]:
    """Parse the LLM's JSON review response robustly."""
    # Strip markdown fences if LLM wrapped it anyway
    clean = raw.strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean)
    clean = re.sub(r"\s*```$", "", clean)
    clean = clean.strip()

    # Find the outermost JSON object
    start = clean.find("{")
    end = clean.rfind("}")
    if start == -1 or end == -1:
        return None

    try:
        data = json.loads(clean[start:end + 1])
        if isinstance(data, dict) and "issues" in data:
            return data
    except json.JSONDecodeError:
        pass

    return None


# ── Badge formatter ────────────────────────────────────────────────────────────

def _format_review_badge(issues: list[str]) -> str:
    """
    Format the auto-review summary shown below the code response.
    Renders as a blockquote in the Angular markdown renderer.
    """
    count = len(issues)
    noun = "issue" if count == 1 else "issues"
    lines = [f"> 🔍 **Auto-review fixed {count} {noun}:**"]
    for issue in issues[:6]:   # cap at 6 to keep it readable
        lines.append(f">  • {issue}")
    if len(issues) > 6:
        lines.append(f">  • _(and {len(issues) - 6} more)_")
    return "\n".join(lines)