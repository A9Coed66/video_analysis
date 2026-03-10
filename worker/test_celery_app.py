"""Unit tests for worker.celery_app configuration."""

from __future__ import annotations

import os
from unittest.mock import patch


def test_app_instance_exists():
    """Celery app is created and named correctly."""
    from worker.celery_app import app

    assert app is not None
    assert app.main == "voice_separator"


def test_broker_url_set():
    """Broker URL is resolved (from settings, env, or default)."""
    from worker.celery_app import app

    assert app.conf.broker_url is not None
    assert "redis" in app.conf.broker_url


def test_result_backend_set():
    """Result backend points to the same Redis instance as the broker."""
    from worker.celery_app import app

    assert app.conf.result_backend is not None
    assert "redis" in app.conf.result_backend


def test_acks_late_enabled():
    """task_acks_late is True for job recovery."""
    from worker.celery_app import app

    assert app.conf.task_acks_late is True


def test_reject_on_worker_lost_enabled():
    """task_reject_on_worker_lost is True for crash recovery."""
    from worker.celery_app import app

    assert app.conf.task_reject_on_worker_lost is True


def test_prefetch_multiplier_is_one():
    """Worker fetches one task at a time (GPU-bound)."""
    from worker.celery_app import app

    assert app.conf.worker_prefetch_multiplier == 1


def test_task_routes_separation_queue():
    """run_separation routes to the 'separation' queue."""
    from worker.celery_app import app

    routes = app.conf.task_routes
    assert routes["worker.tasks.run_separation"]["queue"] == "separation"


def test_task_routes_pipeline_queue():
    """run_pipeline routes to the 'pipeline' queue."""
    from worker.celery_app import app

    routes = app.conf.task_routes
    assert routes["worker.tasks.run_pipeline"]["queue"] == "pipeline"


def test_get_broker_url_env_fallback():
    """_get_broker_url falls back to REDIS_URL env var when Settings fails."""
    custom_url = "redis://custom-host:6380/1"
    with patch.dict(os.environ, {"REDIS_URL": custom_url}):
        from worker.celery_app import _get_broker_url

        # Patch at the source so the local import inside _get_broker_url fails
        with patch("web.config.get_settings", side_effect=RuntimeError):
            url = _get_broker_url()
    assert url == custom_url


def test_get_broker_url_default_fallback():
    """_get_broker_url uses default when both Settings and env are absent."""
    env = os.environ.copy()
    env.pop("REDIS_URL", None)
    with patch.dict(os.environ, env, clear=True):
        from worker.celery_app import _DEFAULT_REDIS, _get_broker_url

        with patch("web.config.get_settings", side_effect=RuntimeError):
            url = _get_broker_url()
    assert url == _DEFAULT_REDIS
