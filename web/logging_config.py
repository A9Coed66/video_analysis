"""Structured JSON logging configuration for the API Server.

Configures *structlog* to emit JSON log lines containing at minimum:
``timestamp``, ``level``, ``logger``, and ``event``.

The request-logging middleware (see :class:`RequestLoggingMiddleware`)
adds per-request fields: ``request_id``, ``method``, ``path``,
``status_code``, ``duration_ms``.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


def configure_logging(log_level: str = "INFO") -> None:
    """Set up structlog with JSON rendering for the API server.

    Call once at application startup (e.g. inside the FastAPI lifespan).
    """
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            # Wrap into stdlib for ProcessorFormatter compatibility
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Single handler that renders JSON for both structlog and stdlib loggers
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root.addHandler(handler)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request with structured fields.

    Emitted fields per requirement 7.5:
    ``timestamp``, ``request_id``, ``method``, ``path``,
    ``status_code``, ``duration_ms``.
    """

    async def dispatch(self, request: Request, call_next) -> Response:  # noqa: ANN001
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        # Attach request_id to the request state so downstream code can use it
        request.state.request_id = request_id

        response: Response = await call_next(request)

        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        logger = structlog.get_logger("web.request")
        logger.info(
            "request_handled",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        return response
