"""Redis-backed job store replacing the in-memory JobManager.

Uses Redis strings for job metadata (JSON-serialized), a sorted set for
time-based indexing, and separate keys for file-id → path mappings.
All keys are set with a configurable TTL so stale data expires automatically.
"""

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone

import redis

from web.models import Job


class RedisJobStore:
    """Persistent job store backed by Redis.

    Redis key schema
    ----------------
    - ``job:{job_id}``        – JSON string with full Job metadata
    - ``job:file:{file_id}``  – plain string mapping file_id → file path
    - ``job:index``           – sorted set (score = created_at unix ts)
    """

    def __init__(self, redis_url: str, ttl_hours: int = 24) -> None:
        self._redis: redis.Redis = redis.Redis.from_url(
            redis_url, decode_responses=True,
        )
        self._ttl_seconds: int = ttl_hours * 3600

    # -- public API -----------------------------------------------------------

    def create_job(self, job_type: str, input_path: str) -> str:
        """Create a new job, persist to Redis, return the job_id."""
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        job = Job(
            job_id=job_id,
            job_type=job_type,
            status="pending",
            created_at=now,
            updated_at=now,
            input_file=input_path,
        )

        key = f"job:{job_id}"
        self._redis.set(key, self._serialize(job), ex=self._ttl_seconds)

        # Track in sorted set for time-based queries / cleanup
        created_ts = datetime.fromisoformat(now).timestamp()
        self._redis.zadd("job:index", {job_id: created_ts})

        return job_id

    def get_job(self, job_id: str) -> Job | None:
        """Fetch and deserialize a job from Redis. Returns *None* if missing."""
        data = self._redis.get(f"job:{job_id}")
        if data is None:
            return None
        return self._deserialize(data)

    def update_status(self, job_id: str, status: str, **kwargs) -> None:
        """Update job status and any extra fields (current_step, etc.)."""
        job = self.get_job(job_id)
        if job is None:
            return

        job.status = status
        job.updated_at = datetime.now(timezone.utc).isoformat()

        for field_name, value in kwargs.items():
            if hasattr(job, field_name):
                setattr(job, field_name, value)

        key = f"job:{job_id}"
        self._redis.set(key, self._serialize(job), ex=self._ttl_seconds)

    def set_result(
        self,
        job_id: str,
        result_files: list[str],
        transcript: list[dict] | None = None,
    ) -> None:
        """Store result file paths, create file-id mappings, mark completed."""
        job = self.get_job(job_id)
        if job is None:
            return

        file_ids: list[str] = []
        for file_path in result_files:
            file_id = str(uuid.uuid4())
            file_key = f"job:file:{file_id}"
            self._redis.set(file_key, file_path, ex=self._ttl_seconds)
            file_ids.append(file_id)

        job.result_files = file_ids
        job.transcript = transcript
        job.status = "completed"
        job.current_step = None
        job.updated_at = datetime.now(timezone.utc).isoformat()

        key = f"job:{job_id}"
        self._redis.set(key, self._serialize(job), ex=self._ttl_seconds)

    def set_error(self, job_id: str, error_message: str) -> None:
        """Mark a job as failed with an error message."""
        job = self.get_job(job_id)
        if job is None:
            return

        job.status = "failed"
        job.error_message = error_message
        job.current_step = None
        job.updated_at = datetime.now(timezone.utc).isoformat()

        key = f"job:{job_id}"
        self._redis.set(key, self._serialize(job), ex=self._ttl_seconds)

    def cleanup_stale_jobs(self, timeout_sec: int = 600) -> int:
        """Detect and mark failed any jobs stuck in 'processing' beyond *timeout_sec*.

        Scans every job tracked in the ``job:index`` sorted set.  If a job's
        status is ``"processing"`` and its ``updated_at`` timestamp is older
        than *timeout_sec* seconds from now, the job is transitioned to
        ``"failed"`` with an appropriate error message.

        Returns the number of jobs that were marked as failed.
        """
        now = datetime.now(timezone.utc)
        stale_count = 0

        # Retrieve all job IDs from the index
        job_ids: list[str] = self._redis.zrange("job:index", 0, -1)

        for job_id in job_ids:
            job = self.get_job(job_id)
            if job is None:
                continue

            if job.status != "processing":
                continue

            updated_at = datetime.fromisoformat(job.updated_at)
            elapsed = (now - updated_at).total_seconds()

            if elapsed > timeout_sec:
                self.set_error(job_id, "Job timed out (stale processing)")
                stale_count += 1

        return stale_count

    def get_result_file(self, file_id: str) -> str | None:
        """Look up the file path for a given file_id."""
        return self._redis.get(f"job:file:{file_id}")

    # -- serialization --------------------------------------------------------

    def _serialize(self, job: Job) -> str:
        """Convert a Job dataclass to a JSON string."""
        return json.dumps(asdict(job))

    def _deserialize(self, data: str) -> Job:
        """Reconstruct a Job dataclass from a JSON string."""
        return Job(**json.loads(data))
