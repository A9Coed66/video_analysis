"""Data models and response schemas for Voice Separator Web API."""

from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel


@dataclass
class Job:
    """In-memory job representation for async audio processing."""
    job_id: str
    job_type: str          # "separate" | "pipeline"
    status: str            # "pending" | "processing" | "completed" | "failed"
    created_at: datetime
    updated_at: datetime
    input_file: str
    current_step: str | None = None
    result_files: list[str] = field(default_factory=list)
    error_message: str | None = None
    transcript: list[dict] | None = None


class JobCreatedResponse(BaseModel):
    job_id: str
    status: str


class FileInfo(BaseModel):
    file_id: str
    filename: str
    speaker_label: str


class TranscriptEntry(BaseModel):
    speaker_label: str
    start_time: float
    end_time: float
    text: str


class JobResult(BaseModel):
    files: list[FileInfo]
    transcript: list[TranscriptEntry] | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    current_step: str | None = None
    result: JobResult | None = None
    error: str | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str
