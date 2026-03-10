"""Job Manager — in-memory async job store with UUID identifiers."""

import uuid
from datetime import datetime, timezone
from pathlib import Path

from web.models import (
    FileInfo, Job, JobResult, JobStatusResponse, TranscriptEntry,
)


class JobManager:
    """Manages async processing jobs. Lifecycle: pending → processing → completed | failed."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._file_map: dict[str, Path] = {}

    def create_job(self, job_type: str, file_path: str) -> str:
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        self._jobs[job_id] = Job(
            job_id=job_id, job_type=job_type, status="pending",
            created_at=now, updated_at=now, input_file=file_path,
        )
        return job_id

    def get_status(self, job_id: str) -> JobStatusResponse | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None

        result = None
        error = None

        if job.status == "completed":
            files = [
                FileInfo(
                    file_id=fid,
                    filename=(self._file_map[fid].name if fid in self._file_map else fid),
                    speaker_label=self._extract_speaker_label(
                        self._file_map[fid].name if fid in self._file_map else fid
                    ),
                )
                for fid in job.result_files
            ]
            transcript_entries = (
                [TranscriptEntry(**e) for e in job.transcript]
                if job.transcript is not None else None
            )
            result = JobResult(files=files, transcript=transcript_entries)

        if job.status == "failed":
            error = job.error_message

        return JobStatusResponse(
            job_id=job.job_id, status=job.status,
            current_step=job.current_step, result=result, error=error,
        )

    def set_status(self, job_id: str, status: str, current_step: str | None = None) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = status
        job.current_step = current_step
        job.updated_at = datetime.now(timezone.utc)

    def set_result(self, job_id: str, result_files: list[Path],
                   transcript: list[dict] | None = None) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        file_ids: list[str] = []
        for path in result_files:
            file_id = str(uuid.uuid4())
            self._file_map[file_id] = path
            file_ids.append(file_id)
        job.result_files = file_ids
        job.transcript = transcript
        job.status = "completed"
        job.current_step = None
        job.updated_at = datetime.now(timezone.utc)

    def set_error(self, job_id: str, error_message: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = "failed"
        job.error_message = error_message
        job.current_step = None
        job.updated_at = datetime.now(timezone.utc)

    def get_result_file(self, file_id: str) -> Path | None:
        return self._file_map.get(file_id)

    @staticmethod
    def _extract_speaker_label(filename: str) -> str:
        """Derive speaker label from filename (spk1/spk2 patterns)."""
        lower = filename.lower()
        if "spk1" in lower or "speaker_1" in lower or "speaker1" in lower:
            return "Speaker 1"
        if "spk2" in lower or "speaker_2" in lower or "speaker2" in lower:
            return "Speaker 2"
        return Path(filename).stem
