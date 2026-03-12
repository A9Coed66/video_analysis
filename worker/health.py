"""Worker health check.

Verifies Redis connectivity, GPU availability, and VRAM status.
Used by Docker healthchecks and the ``/health`` API endpoint.
"""

from __future__ import annotations

import redis
import structlog

from worker.gpu_manager import GPUManager

logger = structlog.get_logger(__name__)


def check_worker_health(redis_url: str, device_index: int = 0) -> dict:
    """Check overall worker health.

    Parameters
    ----------
    redis_url:
        Redis connection URL (e.g. ``redis://redis:6379/0``).
    device_index:
        CUDA device ordinal to inspect.

    Returns
    -------
    dict
        ``{"healthy": bool, "redis": {...}, "gpu": {...}}``
        where ``healthy`` is ``True`` only when Redis is connected
        **and** the GPU is available **and** VRAM is below threshold.
    """
    redis_status = _check_redis(redis_url)
    gpu_status = _check_gpu(device_index)

    healthy = (
        redis_status["connected"]
        and gpu_status["available"]
        and gpu_status["vram_ok"]
    )

    result = {
        "healthy": healthy,
        "redis": redis_status,
        "gpu": gpu_status,
    }

    logger.info("worker_health_check", **result)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_redis(redis_url: str) -> dict:
    """Ping Redis and return connectivity status."""
    try:
        client = redis.Redis.from_url(redis_url, socket_connect_timeout=5)
        client.ping()
        return {"connected": True}
    except Exception:
        logger.warning("redis_health_check_failed", exc_info=True)
        return {"connected": False}


def _check_gpu(device_index: int) -> dict:
    """Query GPU availability and VRAM via :class:`GPUManager`."""
    try:
        gpu = GPUManager(device_index=device_index)
        info = gpu.get_gpu_info()

        if not info.get("available"):
            return {"available": False, "vram_ok": False}

        vram_ok = gpu.check_vram_available()

        return {
            "available": True,
            "vram_ok": vram_ok,
            **{k: v for k, v in info.items() if k != "available"},
        }
    except Exception:
        logger.warning("gpu_health_check_failed", exc_info=True)
        return {"available": False, "vram_ok": False}
