"""TranscriptionEngine: Speech-to-text using Qwen3-ASR."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

QWEN3_ASR_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
UNRECOGNIZED_MARKER = "[không nhận diện được]"


class TranscriptionEngine:
    """Speech-to-text engine using Qwen3-ASR."""

    def __init__(self, model_id: str = QWEN3_ASR_MODEL_ID) -> None:
        self._model = None
        self._model_id = model_id

    def load_model(self) -> None:
        """Load Qwen3-ASR model."""
        from qwen_asr import Qwen3ASRModel

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Qwen3-ASR: %s on %s", self._model_id, device)
        self._model = Qwen3ASRModel.from_pretrained(
            self._model_id,
            dtype=torch.bfloat16,
            device_map=device,
            max_inference_batch_size=32,
            max_new_tokens=256,
        )
        logger.info("Qwen3-ASR model loaded")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file to text. Returns UNRECOGNIZED_MARKER on failure."""
        if self._model is None:
            raise RuntimeError("Transcription model not loaded. Call load_model() first.")

        logger.info("Transcribing: %s", audio_path)
        try:
            results = self._model.transcribe(audio=audio_path, language=None)
            if results and results[0].text and results[0].text.strip():
                return results[0].text.strip()
            return UNRECOGNIZED_MARKER
        except Exception:
            logger.exception("Transcription failed: %s", audio_path)
            return UNRECOGNIZED_MARKER

    def unload_model(self) -> None:
        """Release Qwen3-ASR model from GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Qwen3-ASR model unloaded")
