"""Unit tests for worker.tasks — Celery task definitions.

Tests use mocks to avoid requiring GPU, Redis, or ML model dependencies.
Celery eager mode is used so tasks execute synchronously in-process.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from worker.celery_app import app as celery_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _eager_celery():
    """Run Celery tasks eagerly (synchronous, in-process) for testing."""
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True
    yield
    celery_app.conf.task_always_eager = False
    celery_app.conf.task_eager_propagates = False


@pytest.fixture(autouse=True)
def _reset_job_store_singleton():
    """Reset the module-level _job_store singleton between tests."""
    import worker.tasks as mod

    mod._job_store = None
    yield
    mod._job_store = None


@pytest.fixture()
def mock_job_store():
    """Return a mock RedisJobStore."""
    store = MagicMock()
    store.update_status = MagicMock()
    store.set_result = MagicMock()
    store.set_error = MagicMock()
    return store


@pytest.fixture()
def mock_gpu_available():
    """Return a mock GPUManager that reports VRAM available."""
    gpu = MagicMock()
    gpu.check_vram_available.return_value = True
    gpu.cleanup = MagicMock()
    return gpu


@pytest.fixture()
def mock_gpu_unavailable():
    """Return a mock GPUManager that reports VRAM NOT available."""
    gpu = MagicMock()
    gpu.check_vram_available.return_value = False
    gpu.cleanup = MagicMock()
    return gpu


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


def test_get_gpu_manager_creates_instance():
    """_get_gpu_manager returns a GPUManager with the given device index."""
    from worker.tasks import _get_gpu_manager

    gpu = _get_gpu_manager(1)
    assert gpu.device_index == 1


# ---------------------------------------------------------------------------
# run_separation task
# ---------------------------------------------------------------------------


class TestRunSeparation:
    """Tests for the run_separation Celery task."""

    def test_task_is_registered(self):
        """run_separation is registered in Celery task routes."""
        routes = celery_app.conf.task_routes
        assert "worker.tasks.run_separation" in routes
        assert routes["worker.tasks.run_separation"]["queue"] == "separation"

    def test_task_config(self):
        """run_separation has correct retry and queue config."""
        from worker.tasks import run_separation

        assert run_separation.max_retries == 3
        assert run_separation.default_retry_delay == 30
        assert run_separation.queue == "separation"

    def test_successful_separation(self, mock_job_store, mock_gpu_available, tmp_path):
        """Successful separation copies results and updates job store."""
        input_file = tmp_path / "input.wav"
        input_file.write_bytes(b"fake audio data")

        sep_output_dir = tmp_path / "sep_output"
        sep_output_dir.mkdir()
        spk1 = sep_output_dir / "spk1.wav"
        spk2 = sep_output_dir / "spk2.wav"
        spk1.write_bytes(b"speaker 1")
        spk2.write_bytes(b"speaker 2")

        result_base = str(tmp_path / "results")

        mock_sep_service = MagicMock()
        mock_sep_service.separate.return_value = [str(spk1), str(spk2)]

        with (
            patch("worker.tasks._get_job_store", return_value=mock_job_store),
            patch("worker.tasks._get_gpu_manager", return_value=mock_gpu_available),
            patch("worker.tasks.RESULTS_BASE", result_base),
            patch("web.services.separation.SeparationService", return_value=mock_sep_service),
            patch("web.config.get_settings") as mock_gs,
        ):
            mock_gs.return_value = MagicMock(dprnn_checkpoint_dir="/models/dprnn")

            from worker.tasks import run_separation

            result = run_separation("test-job-123", str(input_file), 0)

        assert result["job_id"] == "test-job-123"
        assert result["status"] == "completed"
        assert len(result["result_files"]) == 2

        # Results were copied to the result directory
        for p in result["result_files"]:
            assert os.path.exists(p)

        mock_job_store.update_status.assert_called()
        mock_job_store.set_result.assert_called_once()
        mock_gpu_available.cleanup.assert_called()

    def test_error_sets_job_error(self, mock_job_store, mock_gpu_available):
        """On exception, job store records the error."""
        mock_sep_service = MagicMock()
        mock_sep_service.separate.side_effect = RuntimeError("Model crashed")

        with (
            patch("worker.tasks._get_job_store", return_value=mock_job_store),
            patch("worker.tasks._get_gpu_manager", return_value=mock_gpu_available),
            patch("web.services.separation.SeparationService", return_value=mock_sep_service),
            patch("web.config.get_settings") as mock_gs,
        ):
            mock_gs.return_value = MagicMock(dprnn_checkpoint_dir="/models/dprnn")

            from worker.tasks import run_separation

            # In eager mode with propagates, MaxRetriesExceededError is raised
            # after exhausting retries
            with pytest.raises(Exception):
                run_separation("test-job", "/fake/input.wav", 0)

        mock_job_store.set_error.assert_called()
        mock_gpu_available.cleanup.assert_called()


# ---------------------------------------------------------------------------
# run_pipeline task
# ---------------------------------------------------------------------------


class TestRunPipeline:
    """Tests for the run_pipeline Celery task."""

    def test_task_config(self):
        """run_pipeline has correct retry and queue config."""
        from worker.tasks import run_pipeline

        assert run_pipeline.max_retries == 3
        assert run_pipeline.default_retry_delay == 30
        assert run_pipeline.queue == "pipeline"

    def test_successful_pipeline(self, mock_job_store, mock_gpu_available, tmp_path):
        """Successful pipeline copies results, saves transcript, updates store."""
        input_file = tmp_path / "input.wav"
        input_file.write_bytes(b"fake audio data")

        pipeline_output_dir = tmp_path / "pipeline_output"
        pipeline_output_dir.mkdir()
        out_file = pipeline_output_dir / "output.wav"
        out_file.write_bytes(b"processed audio")

        result_base = str(tmp_path / "results")

        mock_pipeline_result = MagicMock()
        mock_pipeline_result.output_files = [str(out_file)]
        mock_pipeline_result.transcript = [
            {"speaker": "A", "start": 0.0, "end": 1.0, "text": "Hello"},
        ]

        mock_pipeline_service = MagicMock()
        mock_pipeline_service.process.return_value = mock_pipeline_result

        with (
            patch("worker.tasks._get_job_store", return_value=mock_job_store),
            patch("worker.tasks._get_gpu_manager", return_value=mock_gpu_available),
            patch("worker.tasks.RESULTS_BASE", result_base),
            patch("web.services.pipeline.PipelineService", return_value=mock_pipeline_service),
        ):
            from worker.tasks import run_pipeline

            result = run_pipeline("pipeline-job-456", str(input_file), 1)

        assert result["job_id"] == "pipeline-job-456"
        assert result["status"] == "completed"
        assert result["transcript"] is not None

        # Transcript JSON was saved
        transcript_path = os.path.join(result_base, "pipeline-job-456", "transcript.json")
        assert os.path.exists(transcript_path)

        with open(transcript_path, encoding="utf-8") as f:
            saved_transcript = json.load(f)
        assert saved_transcript[0]["speaker"] == "A"

        mock_job_store.set_result.assert_called_once()

    def test_status_callback_updates_job(self, mock_job_store, mock_gpu_available, tmp_path):
        """Pipeline status_callback updates job store with current step."""
        input_file = tmp_path / "input.wav"
        input_file.write_bytes(b"fake audio data")

        result_base = str(tmp_path / "results")

        def capture_process(file_path, status_callback=None):
            if status_callback:
                status_callback("diarization")
            result = MagicMock()
            result.output_files = []
            result.transcript = []
            return result

        mock_pipeline_service = MagicMock()
        mock_pipeline_service.process.side_effect = capture_process

        with (
            patch("worker.tasks._get_job_store", return_value=mock_job_store),
            patch("worker.tasks._get_gpu_manager", return_value=mock_gpu_available),
            patch("worker.tasks.RESULTS_BASE", result_base),
            patch("web.services.pipeline.PipelineService", return_value=mock_pipeline_service),
        ):
            from worker.tasks import run_pipeline

            run_pipeline("cb-job", str(input_file), 1)

        # Verify "diarization" step was passed to update_status
        update_calls = mock_job_store.update_status.call_args_list
        steps = [
            c.kwargs.get("current_step")
            for c in update_calls
            if c.kwargs.get("current_step")
        ]
        assert "diarization" in steps

    def test_error_sets_job_error(self, mock_job_store, mock_gpu_available):
        """On exception, job store records the error."""
        mock_pipeline_service = MagicMock()
        mock_pipeline_service.process.side_effect = RuntimeError("Pipeline crashed")

        with (
            patch("worker.tasks._get_job_store", return_value=mock_job_store),
            patch("worker.tasks._get_gpu_manager", return_value=mock_gpu_available),
            patch("web.services.pipeline.PipelineService", return_value=mock_pipeline_service),
        ):
            from worker.tasks import run_pipeline

            with pytest.raises(Exception):
                run_pipeline("test-job", "/fake/input.wav", 1)

        mock_job_store.set_error.assert_called()
        mock_gpu_available.cleanup.assert_called()
