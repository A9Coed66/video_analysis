"""API key authentication middleware and rate limiting for Voice Separator Web.

- APIKeyMiddleware: Starlette middleware checking ``X-API-Key`` for ``/api/*``
- Rate limiting via slowapi: 10 POST/min, 100 GET/min per API key
"""

from __future__ import annotations

from typing import Callable

import structlog
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# API Key Authentication Middleware
# ---------------------------------------------------------------------------


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Reject ``/api/*`` requests that lack a valid ``X-API-Key`` header.

    Parameters
    ----------
    app:
        The ASGI application to wrap.
    valid_keys:
        List of accepted API key strings.
    """

    def __init__(self, app, valid_keys: list[str]) -> None:  # noqa: ANN001
        super().__init__(app)
        self._valid_keys: frozenset[str] = frozenset(valid_keys)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path.startswith("/api/"):
            api_key = request.headers.get("X-API-Key")
            if not api_key or api_key not in self._valid_keys:
                logger.warning(
                    "auth.rejected",
                    path=request.url.path,
                    reason="missing" if not api_key else "invalid",
                )
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "Unauthorized",
                        "detail": "Missing or invalid API key",
                    },
                )
        return await call_next(request)


# ---------------------------------------------------------------------------
# Rate Limiting (slowapi)
# ---------------------------------------------------------------------------


def _key_func(request: Request) -> str:
    """Extract the API key from the request to use as rate-limit identity."""
    return request.headers.get("X-API-Key") or request.client.host


limiter = Limiter(key_func=_key_func)

# Decorators to apply on route functions
post_limit = limiter.limit("10/minute")
get_limit = limiter.limit("100/minute")


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Return 429 with ``Retry-After`` header when rate limit is exceeded."""
    # slowapi provides retry_after on the exception when available
    retry_after_value = getattr(exc, "retry_after", None)
    if retry_after_value is None:
        # Fallback: 60 seconds for per-minute limits
        retry_after_value = 60

    logger.warning(
        "rate_limit.exceeded",
        path=request.url.path,
        detail=str(exc.detail),
    )
    return JSONResponse(
        status_code=429,
        content={"error": "Too Many Requests", "detail": str(exc.detail)},
        headers={"Retry-After": str(retry_after_value)},
    )
