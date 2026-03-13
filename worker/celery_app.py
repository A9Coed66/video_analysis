"""Celery application configuration with Redis broker.

Two queues:
  - ``separation`` — DPRNN voice-separation tasks
  - ``pipeline``   — full audio-processing pipeline tasks

Recovery settings ensure jobs are not lost on worker crash:
  - ``task_acks_late``              — ACK only after task completes
  - ``task_reject_on_worker_lost``  — requeue if worker dies mid-task
  - ``worker_prefetch_multiplier``  — fetch one task at a time (GPU bound)
"""

from __future__ import annotations

import os

from celery import Celery

# ---------------------------------------------------------------------------
# Broker URL — prefer Settings from web.config; fall back to env / default
# so the module stays importable in test / CI environments where the full
# web stack may not be configured.
# ---------------------------------------------------------------------------

_DEFAULT_REDIS = "redis://redis:6379/0"


def _get_broker_url() -> str:
    """Resolve the Redis broker URL."""
    try:
        from web.config import get_settings

        return get_settings().redis_url
    except Exception:  # noqa: BLE001
        return os.getenv("REDIS_URL", _DEFAULT_REDIS)


broker_url = _get_broker_url()

# ---------------------------------------------------------------------------
# Celery app
# ---------------------------------------------------------------------------

app = Celery("voice_separator", broker=broker_url, include=["worker.tasks"])

app.conf.update(
    # Result backend — same Redis instance
    result_backend=broker_url,
    # --- Reliability / recovery -------------------------------------------
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # One task at a time per worker (GPU-bound workloads)
    worker_prefetch_multiplier=1,
    # --- Routing -----------------------------------------------------------
    task_routes={
        "worker.tasks.run_separation": {"queue": "separation"},
        "worker.tasks.run_pipeline": {"queue": "pipeline"},
    },
    # --- Graceful shutdown -------------------------------------------------
    # Allow up to 150 s for a running task to finish when SIGTERM is received.
    # This matches the Docker stop_grace_period configured in docker-compose.
    worker_shutdown_timeout=150,
    # --- Serialization (safe defaults) ------------------------------------
    accept_content=["json"],
    task_serializer="json",
    result_serializer="json",
)

# ---------------------------------------------------------------------------
# Worker startup signal — validate model checkpoints
# ---------------------------------------------------------------------------

from celery.signals import worker_ready, worker_shutting_down  # noqa: E402


@worker_ready.connect
def on_worker_ready(**kwargs):  # noqa: ARG001
    """Validate model checkpoints and configure structured logging.

    If any required checkpoint is missing the worker logs detailed errors
    and exits with code 1 so Docker can restart it.
    """
    from worker.logging_config import configure_logging
    from worker.checkpoint_validator import validate_and_exit_on_failure
    from web.config import get_settings

    settings = get_settings()

    validate_and_exit_on_failure(
        settings.dprnn_checkpoint_dir,
        settings.pipeline_model_dir,
    )

    # Structured JSON logging — configure after validation passes
    # to avoid interfering with Celery's internal logging during startup.
    configure_logging(log_level=settings.log_level)


# ---------------------------------------------------------------------------
# Worker shutdown signal — log graceful shutdown event
# ---------------------------------------------------------------------------


@worker_shutting_down.connect
def on_worker_shutting_down(sig, how, exitcode, **kwargs):  # noqa: ARG001
    """Log when the worker begins its graceful shutdown sequence.

    Celery will wait up to ``worker_shutdown_timeout`` (300 s) for the
    currently executing task to finish before forcing termination.
    """
    import logging

    logger = logging.getLogger("worker.celery_app")
    logger.info(
        "Worker received shutdown signal (sig=%s, how=%s). "
        "Finishing current task before exit (timeout=%ss).",
        sig,
        how,
        app.conf.worker_shutdown_timeout,
    )
