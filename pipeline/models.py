"""Data models for the Audio Diarization Pipeline.

Defines all dataclasses used throughout the pipeline for configuration,
intermediate results, and final output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""

    max_chunk_duration_sec: int = 600
    similarity_threshold: float = 0.7
    hf_token: str = ""
    output_dir: str = "output"


@dataclass
class ChunkInfo:
    """Information about a single audio chunk."""

    chunk_index: int
    file_path: str
    start_sec: float  # thời gian bắt đầu trong file gốc
    end_sec: float  # thời gian kết thúc trong file gốc
    duration_sec: float


@dataclass
class Segment:
    """A contiguous audio segment assigned to a speaker."""

    speaker_label: str  # label cục bộ (chunk) hoặc thống nhất (sau unification)
    start_time: float  # thời gian bắt đầu tuyệt đối trong file gốc
    end_time: float  # thời gian kết thúc tuyệt đối trong file gốc


@dataclass
class DiarizationResult:
    """Result of diarization on a single chunk."""

    chunk_index: int
    segments: list[Segment]
    embeddings: dict[str, np.ndarray]  # speaker_label -> embedding vector


@dataclass
class ResolveResult:
    """Result of overlap resolution on a chunk's segments."""

    cleaned_segments: list[Segment]
    removed_segments: list[Segment]
    next_start_sec: float | None  # None nếu là chunk cuối


@dataclass
class UnifiedSpeakerList:
    """Unified speaker identities across all chunks."""

    speakers: dict[str, np.ndarray]  # unified_label -> mean embedding
    label_mapping: dict[tuple[int, str], str]  # (chunk_index, local_label) -> unified_label


@dataclass
class UnificationResult:
    """Result of speaker unification across chunks."""

    unified_speakers: UnifiedSpeakerList
    updated_segments: list[Segment]  # tất cả segments với label thống nhất


@dataclass
class SeparatedAudio:
    """Audio data separated for a single speaker."""

    speaker_label: str | None  # None trước khi match, unified label sau khi match
    audio_data: np.ndarray
    sample_rate: int  # 8000 cho mossformer2
    file_path: str | None  # path sau khi lưu


@dataclass
class TranscriptEntry:
    """A single entry in the final transcript."""

    speaker_label: str
    start_time: float
    end_time: float
    text: str


@dataclass
class PipelineResult:
    """Final result of the entire pipeline run."""

    transcript: list[TranscriptEntry]
    unified_speakers: UnifiedSpeakerList
    output_dir: str
    processing_log: list[str]
