"""DiarizationEngine: Speaker diarization using pyannote/speaker-diarization-3.1."""

from __future__ import annotations

import logging

import numpy as np
import torch

import pipeline.compat  # noqa: F401
pipeline.compat.apply_patches()

from pyannote.audio import Inference, Model, Pipeline
from pyannote.core import Segment as PyannoteSegment

from pipeline.exceptions import AuthenticationError
from pipeline.models import DiarizationResult, Segment

logger = logging.getLogger(__name__)


class DiarizationEngine:
    """Speaker diarization engine using pyannote/speaker-diarization-3.1."""

    MODEL_NAME = "pyannote/speaker-diarization-3.1"
    EMBEDDING_MODEL_NAME = "pyannote/embedding"

    def __init__(self, hf_token: str) -> None:
        if not hf_token or not hf_token.strip():
            raise AuthenticationError(
                "HuggingFace token is missing or empty. "
                "Please set HF_TOKEN in your .env file."
            )
        self.hf_token = hf_token.strip()
        self._pipeline: Pipeline | None = None
        self._embedding_model: Inference | None = None

    def load_model(self) -> None:
        """Load pyannote diarization and embedding models."""
        try:
            self._pipeline = Pipeline.from_pretrained(self.MODEL_NAME, token=self.hf_token)
        except Exception as e:
            error_msg = str(e).lower()
            if any(k in error_msg for k in ("401", "unauthorized", "forbidden")):
                raise AuthenticationError(
                    "HuggingFace token is invalid or lacks access to pyannote models."
                ) from e
            raise

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._pipeline = self._pipeline.to(device)
        logger.info("Diarization pipeline loaded on %s", device)

        try:
            embedding_model = Model.from_pretrained(self.EMBEDDING_MODEL_NAME, token=self.hf_token)
            self._embedding_model = Inference(embedding_model, window="whole")
            self._embedding_model.to(device)
            logger.info("Embedding model loaded on %s", device)
        except Exception as e:
            error_msg = str(e).lower()
            if any(k in error_msg for k in ("401", "unauthorized", "forbidden")):
                raise AuthenticationError(
                    "HuggingFace token is invalid or lacks access to pyannote embedding model."
                ) from e
            raise

    def diarize(self, chunk_path: str, chunk_index: int = 0) -> DiarizationResult:
        """Run diarization on an audio chunk."""
        if self._pipeline is None:
            raise RuntimeError("Diarization model not loaded. Call load_model() first.")

        logger.info("Diarizing chunk %d: %s", chunk_index, chunk_path)
        diarization_output = self._pipeline(chunk_path)

        annotation = (
            diarization_output.speaker_diarization
            if hasattr(diarization_output, "speaker_diarization")
            else diarization_output
        )

        segments: list[Segment] = []
        speaker_labels: set[str] = set()
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append(Segment(speaker_label=speaker, start_time=turn.start, end_time=turn.end))
            speaker_labels.add(speaker)

        logger.info("Chunk %d: %d segments, %d speakers", chunk_index, len(segments), len(speaker_labels))

        embeddings: dict[str, np.ndarray] = {}
        if self._embedding_model is not None and speaker_labels:
            embeddings = self._extract_embeddings(chunk_path, segments, speaker_labels)

        return DiarizationResult(chunk_index=chunk_index, segments=segments, embeddings=embeddings)

    def _extract_embeddings(
        self, chunk_path: str, segments: list[Segment], speaker_labels: set[str],
    ) -> dict[str, np.ndarray]:
        """Extract mean speaker embeddings by cropping audio to speaker segments."""
        embeddings: dict[str, np.ndarray] = {}

        for speaker in speaker_labels:
            speaker_segments = [s for s in segments if s.speaker_label == speaker]
            speaker_embeddings: list[np.ndarray] = []

            for seg in speaker_segments:
                try:
                    excerpt = PyannoteSegment(seg.start_time, seg.end_time)
                    emb = self._embedding_model.crop(chunk_path, excerpt)
                    if emb is not None:
                        emb_array = np.array(emb).flatten()
                        if emb_array.size > 0:
                            speaker_embeddings.append(emb_array)
                except Exception:
                    logger.debug(
                        "Failed to extract embedding for %s [%.2f, %.2f]",
                        speaker, seg.start_time, seg.end_time,
                    )

            if speaker_embeddings:
                mean_emb = np.mean(speaker_embeddings, axis=0)
                norm = np.linalg.norm(mean_emb)
                if norm > 0:
                    mean_emb = mean_emb / norm
                embeddings[speaker] = mean_emb

        return embeddings

    def unload_model(self) -> None:
        """Release diarization and embedding models from GPU memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        if self._embedding_model is not None:
            del self._embedding_model
            self._embedding_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Diarization models unloaded")
