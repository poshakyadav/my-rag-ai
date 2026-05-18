# model_selector.py — Poshak-AI
# ──────────────────────────────
# Lightweight query classifier for intelligent model routing.
#
# Strategy:
#   FAST   → gemini-3.1-flash-lite   (quick Q&A, greetings, simple lookups)
#   HEAVY  → gemma-4-31b-it          (large code generation, architecture, complex reasoning)
#   HEAVY  → llama-3.3-70b-versatile (fallback heavy model via Groq)
#
# The classifier uses a keyword + heuristic scoring approach — no extra LLM call,
# zero latency overhead. Scores above a threshold route to the heavy model.

import re
from dataclasses import dataclass
from typing import Literal

from config import (
    GEMINI_MODEL,
    GEMMA_31B_MODEL,
    GEMMA_26B_MODEL,
    GROQ_MODEL_PRIMARY,
    GROQ_MODEL_FALLBACK,
)

# ── Model tier definitions ────────────────────────────────────────────────────

ModelTier = Literal["fast", "balanced", "heavy"]

# Exposed to the API so the frontend can render a model picker
AVAILABLE_MODELS: list[dict] = [
    {
        "id": "auto",
        "label": "⚡ Auto (Smart routing)",
        "description": "Automatically selects the best model for your query",
        "tier": "auto",
        "provider": "auto",
    },
    {
        "id": GEMINI_MODEL,
        "label": "Gemini 3.1 Flash-Lite",
        "description": "Ultra-fast responses. Best for Q&A, quick lookups, general chat.",
        "tier": "fast",
        "provider": "google",
    },
    {
        "id": GEMMA_26B_MODEL,
        "label": "Gemma 4 26B (MoE)",
        "description": "Fast but capable. Good balance for mid-complexity tasks.",
        "tier": "balanced",
        "provider": "google",
    },
    {
        "id": GEMMA_31B_MODEL,
        "label": "Gemma 4 31B (Dense)",
        "description": "Deep reasoning. Best for large code files, architecture, complex debugging.",
        "tier": "heavy",
        "provider": "google",
    },
    {
        "id": GROQ_MODEL_PRIMARY,
        "label": "Llama 3.3 70B (Groq)",
        "description": "Powerful open-source model via Groq. Great for code reasoning and refactoring.",
        "tier": "heavy",
        "provider": "groq",
    },
    {
        "id": GROQ_MODEL_FALLBACK,
        "label": "Llama 3.1 8B (Groq Fast)",
        "description": "Lightweight Groq model. Very fast for simple tasks.",
        "tier": "fast",
        "provider": "groq",
    },
]

# ── Scoring signals ───────────────────────────────────────────────────────────

# Keywords that strongly suggest a heavy reasoning task
_HEAVY_KEYWORDS: list[tuple[str, int]] = [
    # Code generation / writing
    (r"\b(write|generate|create|implement|build|scaffold)\b.{0,40}\b(class|service|component|module|file|endpoint|api|function|method)\b", 4),
    (r"\b(entire|full|complete|whole)\b.{0,20}\b(file|class|service|component|implementation)\b", 5),
    (r"\bwrite\s+(?:the\s+)?(?:full|complete|entire)\b", 4),
    (r"\brefactor\b", 3),
    (r"\barchitect(?:ure)?\b", 4),
    (r"\bmigrat(?:e|ion)\b", 3),
    (r"\bdesign\s+(?:the\s+)?(?:database|schema|system|api)\b", 4),

    # Complex debugging
    (r"\bdebugg?(?:ing)?\b.{0,40}\b(complex|difficult|hard|tricky)\b", 3),
    (r"\broot\s+cause\b", 3),
    (r"\bperformance\s+(?:issue|problem|bottleneck|optim)\b", 3),
    (r"\bmemory\s+(?:leak|issue|problem)\b", 3),

    # Large content signals
    (r"\b(?:multiple|several|many)\s+(?:files?|classes?|endpoints?|components?)\b", 3),
    (r"\bstep[\s-]by[\s-]step\b.{0,30}\b(?:implement|build|create)\b", 3),
    (r"\b(?:from\s+scratch|ground[\s-]up)\b", 4),

    # Explanation of complex topics
    (r"\bexplain\b.{0,30}\b(?:algorithm|architecture|design\s+pattern|system|concept)\b", 2),
    (r"\bcompare\b.{0,30}\b(?:approach|solution|pattern|architecture)\b", 2),
    (r"\btrade[\s-]?off\b", 2),
    (r"\bbest\s+practice\b", 2),
]

# Keywords that suggest a fast/simple task
_FAST_KEYWORDS: list[tuple[str, int]] = [
    (r"\b(?:what\s+is|what\'s|define|meaning\s+of|tell\s+me\s+about)\b", -2),
    (r"\b(?:hello|hi|hey|thanks|thank\s+you|good\s+(?:morning|evening|day))\b", -5),
    (r"\b(?:quick|quickly|briefly|short|simple)\b", -2),
    (r"\b(?:what\s+version|which\s+version|latest\s+version)\b", -2),
    (r"\b(?:list|show|print)\s+(?:all|the|my)\b", -1),
    (r"\bproject\s+structure\b", -1),
    (r"\bshow\s+(?:me\s+)?(?:the\s+)?(?:directory|folder|files)\b", -2),
]

# Length heuristic thresholds
_LONG_MESSAGE_THRESHOLD = 300    # chars — longer messages often imply complex intent
_VERY_LONG_MESSAGE_THRESHOLD = 600


@dataclass
class RoutingDecision:
    model_id: str
    tier: ModelTier
    reason: str
    score: int


def classify_query(
    message: str,
    preferred_llm: str = "gemini",
    has_codebase: bool = False,
) -> RoutingDecision:
    """
    Analyse the user message and return the best model to use.

    Args:
        message:       The user's raw message text.
        preferred_llm: "gemini" or "groq" — controls which provider's heavy model is used.
        has_codebase:  True if the session has an indexed codebase (heavier tasks more likely).

    Returns a RoutingDecision with the chosen model_id, tier, and explanation.
    """
    text = message.lower().strip()
    score = 0

    # ── Score heavy signals ───────────────────────────────────────────────────
    for pattern, weight in _HEAVY_KEYWORDS:
        if re.search(pattern, text, re.IGNORECASE):
            score += weight

    # ── Score fast signals ────────────────────────────────────────────────────
    for pattern, weight in _FAST_KEYWORDS:
        if re.search(pattern, text, re.IGNORECASE):
            score += weight   # these weights are negative

    # ── Length heuristic ──────────────────────────────────────────────────────
    msg_len = len(message)
    if msg_len > _VERY_LONG_MESSAGE_THRESHOLD:
        score += 3   # very long messages = complex intent
    elif msg_len > _LONG_MESSAGE_THRESHOLD:
        score += 1

    # ── Codebase context boost ────────────────────────────────────────────────
    if has_codebase and score > 0:
        score += 1   # indexed project + write-intent = heavier task likely

    # ── Code block detection ──────────────────────────────────────────────────
    if "```" in message:
        score += 2   # pasting code → likely wants complex analysis/modification

    # ── Route decision ────────────────────────────────────────────────────────
    if score >= 5:
        # Heavy reasoning task
        if preferred_llm == "groq":
            return RoutingDecision(
                model_id=GROQ_MODEL_PRIMARY,
                tier="heavy",
                reason=f"Complex task detected (score={score}) → Llama 3.3 70B",
                score=score,
            )
        return RoutingDecision(
            model_id=GEMMA_31B_MODEL,
            tier="heavy",
            reason=f"Complex task detected (score={score}) → Gemma 4 31B",
            score=score,
        )
    elif score >= 2:
        # Balanced task
        if preferred_llm == "groq":
            return RoutingDecision(
                model_id=GROQ_MODEL_PRIMARY,
                tier="balanced",
                reason=f"Mid-complexity task (score={score}) → Llama 3.3 70B",
                score=score,
            )
        return RoutingDecision(
            model_id=GEMMA_26B_MODEL,
            tier="balanced",
            reason=f"Mid-complexity task (score={score}) → Gemma 4 26B",
            score=score,
        )
    else:
        # Fast task
        return RoutingDecision(
            model_id=GEMINI_MODEL,
            tier="fast",
            reason=f"Simple/fast task (score={score}) → Gemini 3.1 Flash-Lite",
            score=score,
        )


def get_model_label(model_id: str) -> str:
    """Return the display label for a model ID."""
    for m in AVAILABLE_MODELS:
        if m["id"] == model_id:
            return m["label"]
    return model_id