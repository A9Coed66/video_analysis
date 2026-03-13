"""Structured JSON logging configuration for the ML Worker.

Configures *structlog* to emit JSON log lines.  Worker tasks bind
job-specific fields per requirement 7.6: ``job_id``, ``job_type``,
``gpu_device``, ``vram_usage``, ``processing_time``, ``status``.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """Set up structlog with JSON rendering for the Celery worker.

    Call once at worker startup (e.g. via Celery ``worker_init`` signal).
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
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

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


def log_job_event(
    *,
    event: str,
    job_id: str,
    job_type: str,
    gpu_device: int | None = None,
    vram_usage: float | None = None,
    processing_time: float | None = None,
    status: str | None = None,
    **extra,
) -> None:
    """Emit a structured log entry for a worker job.

    Ensures all required fields from requirement 7.6 are present:
    ``job_id``, ``job_type``, ``gpu_device``, ``vram_usage``,
    ``processing_time``, ``status``.
    """
    logger = structlog.get_logger("worker.job")
    logger.info(
        event,
        job_id=job_id,
        job_type=job_type,
        gpu_device=gpu_device,
        vram_usage=vram_usage,
        processing_time=processing_time,
        status=status,
        **extra,
    )
