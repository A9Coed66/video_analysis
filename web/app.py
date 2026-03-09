"""FastAPI application for Voice Separator Web."""

import logging
import tempfile
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse

from web.jobs import JobManager
from web.models import ErrorResponse, JobCreatedResponse, JobStatusResponse

logger = logging.getLogger(__name__)

MAX_FILE_SIZE: int = 500 * 1024 * 1024  # 500 MB

ALLOWED_SEPARATE_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
ALLOWED_PIPELINE_EXTENSIONS = {".wav", ".mp3", ".mp4", ".flac", ".ogg"}

STATIC_DIR = Path(__file__).resolve().parent / "static"

_MEDIA_TYPES: dict[str, str] = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
}

job_manager = JobManager()

# Lazy-loaded ML services (heavy models, loaded on first use)
_separation_service = None
_pipeline_service = None


def _get_separation_service():
    global _separation_service
    if _separation_service is None:
        from web.services.separation import SeparationService
        _separation_service = SeparationService()
    return _separation_service


def _get_pipeline_service():
    global _pipeline_service
    if _pipeline_service is None:
        from web.services.pipeline import PipelineService
        _pipeline_service = PipelineService()
    return _pipeline_service


# --- Application ---

app = FastAPI(
    title="Voice Separator Web",
    description="Web interface for DPRNN-TasNet voice separation and audio pipeline processing.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Helpers ---

def _validate_upload(file: UploadFile, allowed_extensions: set[str]):
    """Validate file extension and size. Returns (contents, ext) or JSONResponse on error."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in allowed_extensions:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="Unsupported format",
                detail=f"Accepted formats: {', '.join(sorted(allowed_extensions))}",
            ).model_dump(),
        )
    return ext


async def _read_and_validate_size(file: UploadFile):
    """Read file contents and validate size. Returns contents or JSONResponse."""
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="File too large",
                detail=f"Maximum file size is {MAX_FILE_SIZE // (1024 * 1024)}MB",
            ).model_dump(),
        )
    return contents


def _save_temp_file(contents: bytes, ext: str, prefix: str) -> str:
    """Save contents to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext, prefix=prefix)
    tmp.write(contents)
    tmp.close()
    return tmp.name


# --- Routes ---

@app.get("/", include_in_schema=False)
async def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.is_file():
        return FileResponse(index_path, media_type="text/html")
    return {"detail": "index.html not found"}


@app.post("/api/separate", status_code=202, response_model=JobCreatedResponse,
          responses={400: {"model": ErrorResponse}})
async def api_separate(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    ext_or_error = _validate_upload(file, ALLOWED_SEPARATE_EXTENSIONS)
    if isinstance(ext_or_error, JSONResponse):
        return ext_or_error
    ext = ext_or_error

    contents_or_error = await _read_and_validate_size(file)
    if isinstance(contents_or_error, JSONResponse):
        return contents_or_error

    tmp_path = _save_temp_file(contents_or_error, ext, "sep_input_")
    job_id = job_manager.create_job("separate", tmp_path)
    background_tasks.add_task(_run_separation, job_id, tmp_path)
    return JobCreatedResponse(job_id=job_id, status="pending")


@app.post("/api/pipeline", status_code=202, response_model=JobCreatedResponse,
          responses={400: {"model": ErrorResponse}})
async def api_pipeline(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    ext_or_error = _validate_upload(file, ALLOWED_PIPELINE_EXTENSIONS)
    if isinstance(ext_or_error, JSONResponse):
        return ext_or_error
    ext = ext_or_error

    contents_or_error = await _read_and_validate_size(file)
    if isinstance(contents_or_error, JSONResponse):
        return contents_or_error

    tmp_path = _save_temp_file(contents_or_error, ext, "pipe_input_")
    job_id = job_manager.create_job("pipeline", tmp_path)
    background_tasks.add_task(_run_pipeline, job_id, tmp_path)
    return JobCreatedResponse(job_id=job_id, status="pending")


@app.get("/api/status/{job_id}", response_model=JobStatusResponse,
         responses={404: {"model": ErrorResponse}})
async def api_status(job_id: str):
    status = job_manager.get_status(job_id)
    if status is None:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(error="Job not found",
                                  detail=f"No job with id '{job_id}' exists").model_dump(),
        )
    return status


@app.get("/api/download/{file_id}", responses={404: {"model": ErrorResponse}})
async def api_download(file_id: str):
    file_path = job_manager.get_result_file(file_id)
    if file_path is None or not file_path.is_file():
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(error="File not found",
                                  detail=f"No result file with id '{file_id}' exists").model_dump(),
        )
    media_type = _MEDIA_TYPES.get(file_path.suffix.lower(), "application/octet-stream")
    return FileResponse(path=str(file_path), media_type=media_type, filename=file_path.name)


# --- Background tasks ---

async def _run_separation(job_id: str, file_path: str) -> None:
    job_manager.set_status(job_id, "processing", current_step="separating")
    try:
        service = _get_separation_service()
        result_paths = service.separate(file_path)
        job_manager.set_result(job_id, [Path(p) for p in result_paths])
        logger.info("Separation job %s completed", job_id)
    except Exception as exc:
        logger.exception("Separation job %s failed", job_id)
        job_manager.set_error(job_id, str(exc))


async def _run_pipeline(job_id: str, file_path: str) -> None:
    job_manager.set_status(job_id, "processing", current_step="chunking")
    try:
        service = _get_pipeline_service()

        def status_callback(step: str) -> None:
            job_manager.set_status(job_id, "processing", current_step=step)

        result = service.process(file_path, status_callback=status_callback)
        transcript_dicts = [
            {"speaker_label": e.speaker_label, "start_time": e.start_time,
             "end_time": e.end_time, "text": e.text}
            for e in result.transcript
        ]
        job_manager.set_result(job_id, [Path(p) for p in result.output_files],
                               transcript=transcript_dicts)
        logger.info("Pipeline job %s completed", job_id)
    except Exception as exc:
        logger.exception("Pipeline job %s failed", job_id)
        job_manager.set_error(job_id, str(exc))


# --- Static file mounts (last, so API routes take priority) ---

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/styles", StaticFiles(directory=str(STATIC_DIR / "styles")), name="styles")
app.mount("/js", StaticFiles(directory=str(STATIC_DIR / "js")), name="js")
