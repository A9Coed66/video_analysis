"""Pipeline Service — wraps pipeline/orchestrator.py for web usage."""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

PIPELINE_STEPS = ("chunking", "diarization", "separation", "transcription", "unification")


@dataclass
class PipelineServiceResult:
    transcript: list = field(default_factory=list)
    output_dir: str = ""
    output_files: list[str] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    processing_log: list[str] = field(default_factory=list)


def _get_pipeline_config_class() -> type:
    from pipeline.models import PipelineConfig
    return PipelineConfig


def _get_orchestrator_class() -> type:
    from pipeline.orchestrator import PipelineOrchestrator
    return PipelineOrchestrator


class PipelineService:
    """Full audio-processing pipeline service."""

    def __init__(self, config: Any | None = None) -> None:
        if config is None:
            PipelineConfig = _get_pipeline_config_class()
            config = PipelineConfig(hf_token=os.environ.get("HF_TOKEN", ""))
        self._config = config
        logger.info("PipelineService initialised (hf_token=%s)", "set" if config.hf_token else "unset")

    def process(self, file_path: str,
                status_callback: Callable[[str], None] | None = None) -> PipelineServiceResult:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")

        output_dir = tempfile.mkdtemp(prefix="pipeline_")

        def _notify(step: str) -> None:
            if status_callback is not None:
                try:
                    status_callback(step)
                except Exception:
                    logger.warning("status_callback raised for step '%s'", step, exc_info=True)

        try:
            PipelineConfig = _get_pipeline_config_class()
            PipelineOrchestrator = _get_orchestrator_class()

            run_config = PipelineConfig(
                max_chunk_duration_sec=self._config.max_chunk_duration_sec,
                similarity_threshold=self._config.similarity_threshold,
                hf_token=self._config.hf_token,
                output_dir=output_dir,
            )
            orchestrator = PipelineOrchestrator(run_config)

            _notify("chunking")
            _notify("diarization")
            result = orchestrator.run(file_path, output_dir)
            _notify("separation")
            _notify("transcription")
            _notify("unification")

            output_files = _collect_output_files(output_dir)
            speakers = sorted(result.unified_speakers.speakers.keys())

            return PipelineServiceResult(
                transcript=result.transcript,
                output_dir=output_dir,
                output_files=output_files,
                speakers=speakers,
                processing_log=result.processing_log,
            )
        except Exception as exc:
            logger.exception("Pipeline failed for %s", file_path)
            raise RuntimeError(f"Pipeline processing failed: {exc}") from exc


def _collect_output_files(output_dir: str) -> list[str]:
    files: list[str] = []
    for root, _dirs, filenames in os.walk(output_dir):
        for fname in filenames:
            files.append(os.path.join(root, fname))
    files.sort()
    return files
