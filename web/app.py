"""FastAPI application for Voice Separator Web — production version.

Key changes from development version:
- ML inference delegated to Celery workers (no BackgroundTasks / lazy-load)
- RedisJobStore replaces in-memory JobManager
- API key authentication middleware
- Rate limiting via slowapi
- File uploads saved to /data/uploads/{uuid}.ext (persistent volume)
- File validation: size + extension checks → HTTP 400
- Pipeline endpoint returns HTTP 503 when HF_TOKEN is missing
"""

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded
from starlette.responses import JSONResponse

from web.config import get_settings
from web.job_store import RedisJobStore
from web.logging_config import RequestLoggingMiddleware, configure_logging
from web.middleware import (
    APIKeyMiddleware,
    get_limit,
    limiter,
    post_limit,
    rate_limit_exceeded_handler,
)
from web.models import ErrorResponse, JobCreatedResponse, JobStatusResponse

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".mp4"}

STATIC_DIR = Path(__file__).resolve().parent / "static"

_MEDIA_TYPES: dict[str, str] = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".mp4": "video/mp4",
}


# ---------------------------------------------------------------------------
# Lazy-initialised singletons (populated on first request via lifespan)
# ---------------------------------------------------------------------------

_job_store: RedisJobStore | None = None
_celery_app = None
_settings = None


def _get_settings():
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings


def _get_job_store() -> RedisJobStore:
    global _job_store
    if _job_store is None:
        s = _get_settings()
        _job_store = RedisJobStore(redis_url=s.redis_url, ttl_hours=s.job_ttl_hours)
    return _job_store


def _get_celery():
    global _celery_app
    if _celery_app is None:
        from worker.celery_app import app as _capp
        _celery_app = _capp
    return _celery_app


def _get_upload_dir() -> Path:
    upload_dir = Path("/data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Application startup/shutdown lifecycle."""
    s = _get_settings()

    # Structured JSON logging — must be configured before any log calls
    configure_logging(log_level=s.log_level)

    yield


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Voice Separator Web",
    description="Web interface for DPRNN-TasNet voice separation and audio pipeline processing.",
    version="1.0.0",
    lifespan=lifespan,
)

# --- Middleware (must be added before app starts, i.e. at module level) ----
_startup_settings = get_settings()

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(APIKeyMiddleware, valid_keys=_startup_settings.api_key_list)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_startup_settings.allowed_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiter state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_file(file: UploadFile) -> JSONResponse | None:
    """Validate file extension. Returns a JSONResponse error or None if valid."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="Unsupported format",
                detail=f"Accepted formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            ).model_dump(),
        )
    return None


async def _read_and_validate_size(file: UploadFile) -> bytes | JSONResponse:
    """Read file contents and validate size. Returns bytes or JSONResponse error."""
    s = _get_settings()
    contents = await file.read()
    if len(contents) > s.max_file_size:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="File too large",
                detail=f"Maximum file size is {s.max_file_size // (1024 * 1024)}MB",
            ).model_dump(),
        )
    return contents


def _save_upload(contents: bytes, original_filename: str) -> str:
    """Save upload to /data/uploads/{uuid}.ext and return the path string."""
    ext = Path(original_filename or "").suffix.lower()
    filename = f"{uuid.uuid4()}{ext}"
    upload_dir = _get_upload_dir()
    dest = upload_dir / filename
    dest.write_bytes(contents)
    return str(dest)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.is_file():
        return FileResponse(index_path, media_type="text/html")
    return {"detail": "index.html not found"}

# ---------------------------------------------------------------------------
# Health check endpoints (no API key required)
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check():
    """Liveness check for Docker healthcheck — checks API + Redis only.

    Workers have their own healthchecks in docker-compose.
    Use /health/ready for a full readiness check including workers.
    """
    s = _get_settings()
    api_status = {"healthy": True}
    redis_status = _check_redis_health(s.redis_url)

    overall_healthy = api_status["healthy"] and redis_status["healthy"]

    body = {
        "healthy": overall_healthy,
        "api": api_status,
        "redis": redis_status,
    }

    status_code = 200 if overall_healthy else 503
    return JSONResponse(content=body, status_code=status_code)

@app.get("/health/ready")
async def readiness_check():
    """Full readiness check including workers — use for monitoring, not Docker healthcheck."""
    s = _get_settings()
    api_status = {"healthy": True}
    redis_status = _check_redis_health(s.redis_url)
    workers_status = _check_workers_health()

    overall_healthy = (
        api_status["healthy"]
        and redis_status["healthy"]
        and workers_status["healthy"]
    )

    body = {
        "healthy": overall_healthy,
        "api": api_status,
        "redis": redis_status,
        "workers": workers_status,
    }

    status_code = 200 if overall_healthy else 503
    return JSONResponse(content=body, status_code=status_code)



@app.get("/health/gpu")
async def health_gpu():
    """Report VRAM usage for each GPU.

    Returns a list of GPU info dicts with id, name, vram_total, vram_used,
    vram_free.  Always returns HTTP 200 (informational endpoint).
    """
    gpus = _get_gpu_info_all()
    return JSONResponse(content={"gpus": gpus}, status_code=200)


def _check_redis_health(redis_url: str) -> dict:
    """Ping Redis and return connectivity status."""
    import redis as redis_lib

    try:
        client = redis_lib.Redis.from_url(redis_url, socket_connect_timeout=5)
        client.ping()
        return {"healthy": True}
    except Exception:
        return {"healthy": False}


def _check_workers_health() -> dict:
    """Inspect Celery workers and return availability status."""
    try:
        celery = _get_celery()
        inspector = celery.control.inspect(timeout=3.0)
        active_workers = inspector.ping()
        if active_workers:
            return {"healthy": True, "count": len(active_workers)}
        return {"healthy": False, "count": 0}
    except Exception:
        return {"healthy": False, "count": 0}


def _get_gpu_info_all() -> list[dict]:
    """Collect GPU info for all available CUDA devices."""
    try:
        import torch

        if not torch.cuda.is_available():
            return []

        gpus = []
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            free_bytes, total_bytes = torch.cuda.mem_get_info(i)
            used_bytes = total_bytes - free_bytes
            gpus.append({
                "id": i,
                "name": name,
                "vram_total": round(total_bytes / (1024 * 1024), 2),
                "vram_used": round(used_bytes / (1024 * 1024), 2),
                "vram_free": round(free_bytes / (1024 * 1024), 2),
            })
        return gpus
    except Exception:
        return []



@app.post(
    "/api/separate",
    status_code=202,
    response_model=JobCreatedResponse,
    responses={400: {"model": ErrorResponse}},
)
@post_limit
async def api_separate(request: Request, file: UploadFile = File(...)):
    # Validate extension
    error = _validate_file(file)
    if error is not None:
        return error

    # Validate size
    contents_or_error = await _read_and_validate_size(file)
    if isinstance(contents_or_error, JSONResponse):
        return contents_or_error

    # Save to persistent storage
    upload_path = _save_upload(contents_or_error, file.filename or "")

    # Create job in Redis
    job_store = _get_job_store()
    job_id = job_store.create_job("separate", upload_path)

    # Enqueue Celery task
    s = _get_settings()
    celery = _get_celery()
    task = celery.send_task(
        "worker.tasks.run_separation",
        kwargs={"job_id": job_id, "input_path": upload_path, "gpu_device": s.gpu_separation},
        queue="separation",
    )

    # Store celery task id
    job_store.update_status(job_id, "pending", celery_task_id=task.id)

    return JobCreatedResponse(job_id=job_id, status="pending")


@app.post(
    "/api/pipeline",
    status_code=202,
    response_model=JobCreatedResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
@post_limit
async def api_pipeline(request: Request, file: UploadFile = File(...)):
    # Check if pipeline is enabled (HF_TOKEN present)
    s = _get_settings()
    if not s.pipeline_enabled:
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error="Service Unavailable",
                detail="Pipeline service disabled — HF_TOKEN not configured",
            ).model_dump(),
        )

    # Validate extension
    error = _validate_file(file)
    if error is not None:
        return error

    # Validate size
    contents_or_error = await _read_and_validate_size(file)
    if isinstance(contents_or_error, JSONResponse):
        return contents_or_error

    # Save to persistent storage
    upload_path = _save_upload(contents_or_error, file.filename or "")

    # Create job in Redis
    job_store = _get_job_store()
    job_id = job_store.create_job("pipeline", upload_path)

    # Enqueue Celery task
    celery = _get_celery()
    task = celery.send_task(
        "worker.tasks.run_pipeline",
        kwargs={"job_id": job_id, "input_path": upload_path, "gpu_device": s.gpu_pipeline},
        queue="pipeline",
    )

    # Store celery task id
    job_store.update_status(job_id, "pending", celery_task_id=task.id)

    return JobCreatedResponse(job_id=job_id, status="pending")


@app.get(
    "/api/status/{job_id}",
    response_model=JobStatusResponse,
    responses={404: {"model": ErrorResponse}},
)
@get_limit
async def api_status(request: Request, job_id: str):
    job_store = _get_job_store()
    job = job_store.get_job(job_id)
    if job is None:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error="Job not found",
                detail=f"No job with id '{job_id}' exists",
            ).model_dump(),
        )

    # Build response matching the existing contract
    result = None
    error = None

    if job.status == "completed":
        from web.models import FileInfo, JobResult, TranscriptEntry

        files = []
        for file_id in job.result_files:
            file_path = job_store.get_result_file(file_id)
            filename = Path(file_path).name if file_path else file_id
            speaker_label = _extract_speaker_label(filename)
            files.append(FileInfo(file_id=file_id, filename=filename, speaker_label=speaker_label))

        transcript_entries = None
        if job.transcript is not None:
            transcript_entries = [TranscriptEntry(**e) for e in job.transcript]

        result = JobResult(files=files, transcript=transcript_entries)

    if job.status == "failed":
        error = job.error_message

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        current_step=job.current_step,
        result=result,
        error=error,
    )


@app.get("/api/download/{file_id}", responses={404: {"model": ErrorResponse}})
@get_limit
async def api_download(request: Request, file_id: str):
    job_store = _get_job_store()
    file_path_str = job_store.get_result_file(file_id)
    if file_path_str is None:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error="File not found",
                detail=f"No result file with id '{file_id}' exists",
            ).model_dump(),
        )

    file_path = Path(file_path_str)
    if not file_path.is_file():
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error="File not found",
                detail=f"No result file with id '{file_id}' exists",
            ).model_dump(),
        )

    media_type = _MEDIA_TYPES.get(file_path.suffix.lower(), "application/octet-stream")
    return FileResponse(path=str(file_path), media_type=media_type, filename=file_path.name)


# ---------------------------------------------------------------------------
# Helpers (speaker label extraction — kept from original)
# ---------------------------------------------------------------------------


def _extract_speaker_label(filename: str) -> str:
    """Derive speaker label from filename (spk1/spk2 patterns)."""
    lower = filename.lower()
    if "spk1" in lower or "speaker_1" in lower or "speaker1" in lower:
        return "Speaker 1"
    if "spk2" in lower or "speaker_2" in lower or "speaker2" in lower:
        return "Speaker 2"
    return Path(filename).stem


# ---------------------------------------------------------------------------
# Static file mounts (last, so API routes take priority)
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/styles", StaticFiles(directory=str(STATIC_DIR / "styles")), name="styles")
app.mount("/js", StaticFiles(directory=str(STATIC_DIR / "js")), name="js")
