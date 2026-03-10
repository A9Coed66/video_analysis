"""Unit tests for worker.health.check_worker_health."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from worker.health import check_worker_health, _check_redis, _check_gpu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_torch_mock(*, available: bool = True, name: str = "NVIDIA RTX A6000",
                     free: int = 30_000_000_000, total: int = 48_000_000_000,
                     cuda_version: str = "12.1"):
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = available
    mock_torch.cuda.get_device_name.return_value = name
    mock_torch.cuda.mem_get_info.return_value = (free, total)
    mock_torch.version.cuda = cuda_version
    return mock_torch


# ---------------------------------------------------------------------------
# _check_redis
# ---------------------------------------------------------------------------

class TestCheckRedis:
    def test_connected_when_ping_succeeds(self):
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        with patch("worker.health.redis.Redis.from_url", return_value=mock_client):
            result = _check_redis("redis://localhost:6379/0")
        assert result == {"connected": True}

    def test_not_connected_when_ping_fails(self):
        with patch("worker.health.redis.Redis.from_url", side_effect=ConnectionError("refused")):
            result = _check_redis("redis://localhost:6379/0")
        assert result == {"connected": False}

    def test_not_connected_on_timeout(self):
        mock_client = MagicMock()
        mock_client.ping.side_effect = TimeoutError("timed out")
        with patch("worker.health.redis.Redis.from_url", return_value=mock_client):
            result = _check_redis("redis://localhost:6379/0")
        assert result == {"connected": False}


# ---------------------------------------------------------------------------
# _check_gpu
# ---------------------------------------------------------------------------

class TestCheckGpu:
    def test_available_and_vram_ok(self):
        mock_torch = _make_torch_mock(free=30_000_000_000, total=48_000_000_000)
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = _check_gpu(0)
        assert result["available"] is True
        assert result["vram_ok"] is True
        assert "name" in result

    def test_available_but_vram_over_threshold(self):
        # 95% used
        mock_torch = _make_torch_mock(free=2_400_000_000, total=48_000_000_000)
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = _check_gpu(0)
        assert result["available"] is True
        assert result["vram_ok"] is False

    def test_gpu_not_available(self):
        mock_torch = _make_torch_mock(available=False)
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = _check_gpu(0)
        assert result["available"] is False
        assert result["vram_ok"] is False

    def test_exception_returns_unavailable(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = RuntimeError("driver error")
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = _check_gpu(0)
        assert result["available"] is False
        assert result["vram_ok"] is False


# ---------------------------------------------------------------------------
# check_worker_health (integration of both checks)
# ---------------------------------------------------------------------------

class TestCheckWorkerHealth:
    def test_healthy_when_all_ok(self):
        mock_redis_client = MagicMock()
        mock_redis_client.ping.return_value = True
        mock_torch = _make_torch_mock()

        with patch("worker.health.redis.Redis.from_url", return_value=mock_redis_client), \
             patch.dict("sys.modules", {"torch": mock_torch}):
            result = check_worker_health("redis://localhost:6379/0", device_index=0)

        assert result["healthy"] is True
        assert result["redis"]["connected"] is True
        assert result["gpu"]["available"] is True
        assert result["gpu"]["vram_ok"] is True

    def test_unhealthy_when_redis_down(self):
        mock_torch = _make_torch_mock()
        with patch("worker.health.redis.Redis.from_url", side_effect=ConnectionError), \
             patch.dict("sys.modules", {"torch": mock_torch}):
            result = check_worker_health("redis://localhost:6379/0")

        assert result["healthy"] is False
        assert result["redis"]["connected"] is False
        assert result["gpu"]["available"] is True

    def test_unhealthy_when_gpu_unavailable(self):
        mock_redis_client = MagicMock()
        mock_redis_client.ping.return_value = True
        mock_torch = _make_torch_mock(available=False)

        with patch("worker.health.redis.Redis.from_url", return_value=mock_redis_client), \
             patch.dict("sys.modules", {"torch": mock_torch}):
            result = check_worker_health("redis://localhost:6379/0")

        assert result["healthy"] is False
        assert result["redis"]["connected"] is True
        assert result["gpu"]["available"] is False

    def test_unhealthy_when_vram_over_threshold(self):
        mock_redis_client = MagicMock()
        mock_redis_client.ping.return_value = True
        # 95% used
        mock_torch = _make_torch_mock(free=2_400_000_000, total=48_000_000_000)

        with patch("worker.health.redis.Redis.from_url", return_value=mock_redis_client), \
             patch.dict("sys.modules", {"torch": mock_torch}):
            result = check_worker_health("redis://localhost:6379/0")

        assert result["healthy"] is False
        assert result["gpu"]["vram_ok"] is False

    def test_unhealthy_when_all_down(self):
        mock_torch = _make_torch_mock(available=False)
        with patch("worker.health.redis.Redis.from_url", side_effect=ConnectionError), \
             patch.dict("sys.modules", {"torch": mock_torch}):
            result = check_worker_health("redis://localhost:6379/0")

        assert result["healthy"] is False
        assert result["redis"]["connected"] is False
        assert result["gpu"]["available"] is False
