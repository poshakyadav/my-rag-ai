"""
api/core/errors.py
───────────────────
Standardised error responses for all routers.

FIX: Added `internal()` helper (was missing, but referenced in sessions.py fix).
All helpers raise HTTPException — FastAPI routes these through its own exception
handler pipeline which runs INSIDE CORSMiddleware, so CORS headers are always
present on the response regardless of status code.
"""

from fastapi import HTTPException, status


def not_found(detail: str = "Not found") -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


def bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def conflict(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)


def internal(detail: str = "Internal server error") -> HTTPException:
    """
    Returns an HTTPException with status 500.

    IMPORTANT: Always use this instead of raising raw Python exceptions in
    router handlers.  Raw exceptions escape to ServerErrorMiddleware which sits
    OUTSIDE CORSMiddleware — the browser then sees a 500 with no CORS header and
    misreports it as a CORS error.

    HTTPException is caught by FastAPI's exception handler pipeline, which runs
    INSIDE CORSMiddleware, so the response always has CORS headers.
    """
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=detail,
    )