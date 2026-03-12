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

app = Celery("voice_separator", broker=broker_url)

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
    # --- Serialization (safe defaults) ------------------------------------
    accept_content=["json"],
    task_serializer="json",
    result_serializer="json",
)

# ---------------------------------------------------------------------------
# Worker startup signal — validate model checkpoints
# ---------------------------------------------------------------------------

from celery.signals import worker_ready  # noqa: E402


@worker_ready.connect
def on_worker_ready(**kwargs):  # noqa: ARG001
    """Validate model checkpoints when the Celery worker is ready.

    If any required checkpoint is missing the worker logs detailed errors
    and exits with code 1 so Docker can restart it.
    """
    from worker.checkpoint_validator import validate_and_exit_on_failure
    from web.config import get_settings

    settings = get_settings()
    validate_and_exit_on_failure(
        settings.dprnn_checkpoint_dir,
        settings.pipeline_model_dir,
    )
