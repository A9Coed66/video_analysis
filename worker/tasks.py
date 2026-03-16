"""Celery task definitions for ML inference on GPU.

Tasks
-----
- ``run_separation`` — DPRNN-TasNet voice separation (queue: ``separation``)
- ``run_pipeline``   — Full audio-processing pipeline (queue: ``pipeline``)

Each task:
1. Updates job status via :class:`RedisJobStore`
2. Checks VRAM availability before running inference
3. Copies results to ``/data/results/{job_id}/``
4. Cleans up VRAM after completion
5. Retries up to 3 times with exponential backoff on transient failures
"""

from __future__ import annotations

import json
import os
import shutil
from typing import TYPE_CHECKING

import structlog

from worker.celery_app import app
from worker.gpu_manager import GPUManager
from worker.logging_config import log_job_event

if TYPE_CHECKING:
    from web.job_store import RedisJobStore

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_job_store: RedisJobStore | None = None


def _get_job_store() -> RedisJobStore:
    """Lazily create a singleton :class:`RedisJobStore`."""
    global _job_store  # noqa: PLW0603
    if _job_store is None:
        from web.config import get_settings
        from web.job_store import RedisJobStore

        settings = get_settings()
        _job_store = RedisJobStore(
            redis_url=settings.redis_url,
            ttl_hours=settings.job_ttl_hours,
        )
    return _job_store


def _get_gpu_manager(device_index: int) -> GPUManager:
    """Create a :class:`GPUManager` for the given device."""
    return GPUManager(device_index=device_index)


RESULTS_BASE = "/data/results"


# ---------------------------------------------------------------------------
# Separation task
# ---------------------------------------------------------------------------


@app.task(
    bind=True,
    name="worker.tasks.run_separation",
    queue="separation",
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
)
def run_separation(
    self,
    job_id: str,
    input_path: str,
    gpu_device: int = 0,
) -> dict:
    """DPRNN-TasNet voice separation task.

    Parameters
    ----------
    job_id:
        Unique job identifier stored in Redis.
    input_path:
        Path to the uploaded audio file.
    gpu_device:
        CUDA device ordinal to run inference on.
    """
    job_store = _get_job_store()
    gpu = _get_gpu_manager(gpu_device)
    log = logger.bind(job_id=job_id, task="separation", gpu_device=gpu_device)

    import time as _time
    _start = _time.perf_counter()

    try:
        # 1. Update status → processing
        job_store.update_status(
            job_id, "processing", current_step="separating", gpu_device=gpu_device,
        )
        gpu_info = gpu.get_gpu_info()
        vram_usage = gpu_info.get("vram_used", None)
        log_job_event(
            event="job_started",
            job_id=job_id,
            job_type="separation",
            gpu_device=gpu_device,
            vram_usage=vram_usage,
            processing_time=None,
            status="processing",
        )

        # 2. Check VRAM availability
        if not gpu.check_vram_available():
            log.warning("vram_insufficient, retrying")
            raise self.retry(
                countdown=30 * (2 ** self.request.retries),
                exc=RuntimeError("Insufficient VRAM"),
            )

        # 3. Import and instantiate SeparationService (lazy)
        from web.config import get_settings
        from web.services.separation import SeparationService

        settings = get_settings()
        service = SeparationService(checkpoint_dir=settings.dprnn_checkpoint_dir)

        # 4. Run separation
        result_paths = service.separate(input_path)

        # 5. Copy results to /data/results/{job_id}/
        result_dir = os.path.join(RESULTS_BASE, job_id)
        os.makedirs(result_dir, exist_ok=True)

        copied_paths: list[str] = []
        for src_path in result_paths:
            dst_path = os.path.join(result_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)
            copied_paths.append(dst_path)

        # 6. Store result in job store
        job_store.set_result(job_id, copied_paths)
        processing_time = round((_time.perf_counter() - _start) * 1000, 2)
        gpu_info = gpu.get_gpu_info()
        log_job_event(
            event="job_completed",
            job_id=job_id,
            job_type="separation",
            gpu_device=gpu_device,
            vram_usage=gpu_info.get("vram_used", None),
            processing_time=processing_time,
            status="completed",
        )

        return {"job_id": job_id, "status": "completed", "result_files": copied_paths}

    except self.MaxRetriesExceededError:
        processing_time = round((_time.perf_counter() - _start) * 1000, 2)
        log_job_event(
            event="job_failed",
            job_id=job_id,
            job_type="separation",
            gpu_device=gpu_device,
            vram_usage=None,
            processing_time=processing_time,
            status="failed",
        )
        job_store.set_error(job_id, "Max retries exceeded for separation task")
        raise

    except Exception as exc:
        log.error("separation_failed", error=str(exc), exc_info=True)
        processing_time = round((_time.perf_counter() - _start) * 1000, 2)
        log_job_event(
            event="job_failed",
            job_id=job_id,
            job_type="separation",
            gpu_device=gpu_device,
            vram_usage=None,
            processing_time=processing_time,
            status="failed",
        )
        job_store.set_error(job_id, str(exc))
        gpu.cleanup()

        # Retry with exponential backoff if retries remaining
        try:
            raise self.retry(
                countdown=30 * (2 ** self.request.retries),
                exc=exc,
            )
        except self.MaxRetriesExceededError:
            log.error("separation_max_retries_exceeded")
            raise

    finally:
        gpu.cleanup()


# ---------------------------------------------------------------------------
# Pipeline task
# ---------------------------------------------------------------------------


@app.task(
    bind=True,
    name="worker.tasks.run_pipeline",
    queue="pipeline",
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
)
def run_pipeline(
    self,
    job_id: str,
    input_path: str,
    gpu_device: int = 1,
) -> dict:
    """Full audio-processing pipeline task.

    Parameters
    ----------
    job_id:
        Unique job identifier stored in Redis.
    input_path:
        Path to the uploaded audio file.
    gpu_device:
        CUDA device ordinal to run inference on.
    """
    job_store = _get_job_store()
    gpu = _get_gpu_manager(gpu_device)
    log = logger.bind(job_id=job_id, task="pipeline", gpu_device=gpu_device)

    import time as _time
    _start = _time.perf_counter()

    def _status_callback(step: str) -> None:
        """Update job status with the current pipeline step."""
        job_store.update_status(job_id, "processing", current_step=step)
        log.info("pipeline_step", step=step)

    try:
        # 1. Update status → processing
        job_store.update_status(
            job_id, "processing", current_step="initializing", gpu_device=gpu_device,
        )
        gpu_info = gpu.get_gpu_info()
        vram_usage = gpu_info.get("vram_used", None)
        log_job_event(
            event="job_started",
            job_id=job_id,
            job_type="pipeline",
            gpu_device=gpu_device,
            vram_usage=vram_usage,
            processing_time=None,
            status="processing",
        )

        # 2. Check VRAM availability
        if not gpu.check_vram_available():
            log.warning("vram_insufficient, retrying")
            raise self.retry(
                countdown=30 * (2 ** self.request.retries),
                exc=RuntimeError("Insufficient VRAM"),
            )

        # 3. Import and instantiate PipelineService (lazy)
        from web.services.pipeline import PipelineService

        service = PipelineService()

        # 4. Run pipeline with status callback
        result = service.process(input_path, status_callback=_status_callback)

        # 5. Copy results to /data/results/{job_id}/
        result_dir = os.path.join(RESULTS_BASE, job_id)
        os.makedirs(result_dir, exist_ok=True)

        copied_paths: list[str] = []
        for src_path in result.output_files:
            dst_path = os.path.join(result_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)
            copied_paths.append(dst_path)

        # 6. Save transcript as JSON in result directory
        if result.transcript:
            transcript_path = os.path.join(result_dir, "transcript.json")
            with open(transcript_path, "w", encoding="utf-8") as f:
                json.dump(result.transcript, f, ensure_ascii=False, indent=2)
            copied_paths.append(transcript_path)

        # 7. Store result in job store (with transcript)
        job_store.set_result(job_id, copied_paths, transcript=result.transcript)
        processing_time = round((_time.perf_counter() - _start) * 1000, 2)
        gpu_info = gpu.get_gpu_info()
        log_job_event(
            event="job_completed",
            job_id=job_id,
            job_type="pipeline",
            gpu_device=gpu_device,
            vram_usage=gpu_info.get("vram_used", None),
            processing_time=processing_time,
            status="completed",
        )

        return {
            "job_id": job_id,
            "status": "completed",
            "result_files": copied_paths,
            "transcript": result.transcript,
        }

    except self.MaxRetriesExceededError:
        processing_time = round((_time.perf_counter() - _start) * 1000, 2)
        log_job_event(
            event="job_failed",
            job_id=job_id,
            job_type="pipeline",
            gpu_device=gpu_device,
            vram_usage=None,
            processing_time=processing_time,
            status="failed",
        )
        job_store.set_error(job_id, "Max retries exceeded for pipeline task")
        raise

    except Exception as exc:
        log.error("pipeline_failed", error=str(exc), exc_info=True)
        processing_time = round((_time.perf_counter() - _start) * 1000, 2)
        log_job_event(
            event="job_failed",
            job_id=job_id,
            job_type="pipeline",
            gpu_device=gpu_device,
            vram_usage=None,
            processing_time=processing_time,
            status="failed",
        )
        job_store.set_error(job_id, str(exc))
        gpu.cleanup()

        # Retry with exponential backoff if retries remaining
        try:
            raise self.retry(
                countdown=30 * (2 ** self.request.retries),
                exc=exc,
            )
        except self.MaxRetriesExceededError:
            log.error("pipeline_max_retries_exceeded")
            raise

    finally:
        gpu.cleanup()
