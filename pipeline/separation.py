"""SeparationEngine: Speech separation using mossformer2."""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
import torch
import torchaudio

from pipeline.models import SeparatedAudio, UnifiedSpeakerList

logger = logging.getLogger(__name__)

MOSSFORMER2_SAMPLE_RATE = 8000
MOSSFORMER2_MODEL_ID = "damo/speech_mossformer2_separation_temporal_8k"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _parse_pcm_data(pcm_data) -> np.ndarray | None:
    """Parse PCM data from mossformer2 output into float32 array."""
    try:
        if isinstance(pcm_data, bytes):
            return np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        elif isinstance(pcm_data, np.ndarray):
            arr = pcm_data.flatten().astype(np.float32)
            return arr / 32768.0 if np.abs(arr).max() > 2.0 else arr
        else:
            arr = np.array(pcm_data, dtype=np.float32)
            return arr.flatten() if arr.ndim > 1 else arr
    except Exception:
        logger.exception("Failed to parse PCM data of type %s", type(pcm_data))
        return None


def resample_audio(
    waveform: torch.Tensor, orig_sr: int, target_sr: int = MOSSFORMER2_SAMPLE_RATE,
) -> torch.Tensor:
    if orig_sr == target_sr:
        return waveform
    return torchaudio.functional.resample(waveform, orig_sr, target_sr)


class SeparationEngine:
    """Speech separation engine using mossformer2."""

    def __init__(self) -> None:
        self._model = None
        self._embedding_model = None

    def load_model(self) -> None:
        """Load mossformer2 separation model and pyannote embedding model."""
        from modelscope.pipelines import pipeline as ms_pipeline
        from modelscope.utils.constant import Tasks

        logger.info("Loading mossformer2: %s", MOSSFORMER2_MODEL_ID)
        self._model = ms_pipeline(Tasks.speech_separation, model=MOSSFORMER2_MODEL_ID)

        try:
            from pyannote.audio import Inference, Model
            embedding_model = Model.from_pretrained("pyannote/embedding")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._embedding_model = Inference(embedding_model, window="whole")
            self._embedding_model.to(device)
        except Exception:
            logger.warning("Could not load pyannote embedding model for speaker matching.")
            self._embedding_model = None

    def separate(self, chunk_path: str) -> list[SeparatedAudio]:
        """Resample to 8kHz, run separation. Splits long audio (>60s) into sub-segments."""
        if self._model is None:
            raise RuntimeError("Separation model not loaded. Call load_model() first.")

        logger.info("Separating audio: %s", chunk_path)
        try:
            waveform, orig_sr = torchaudio.load(chunk_path)
            waveform_8k = resample_audio(waveform, orig_sr)
            if waveform_8k.shape[0] > 1:
                waveform_8k = waveform_8k.mean(dim=0, keepdim=True)

            duration_sec = waveform_8k.shape[1] / MOSSFORMER2_SAMPLE_RATE
            if duration_sec > 60:
                return self._separate_long(waveform_8k, chunk_path)
            return self._separate_short(waveform_8k, chunk_path)
        except Exception:
            logger.exception("Separation failed for chunk: %s", chunk_path)
            return []

    def _run_model_on_waveform(self, waveform_8k: torch.Tensor) -> list[np.ndarray]:
        """Run mossformer2 on a single 8kHz mono waveform tensor."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        torchaudio.save(tmp_path, waveform_8k, MOSSFORMER2_SAMPLE_RATE)

        try:
            result = self._model(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        streams: list[np.ndarray] = []
        if isinstance(result, dict) and "output_pcm_list" in result:
            for pcm_data in result["output_pcm_list"]:
                arr = _parse_pcm_data(pcm_data)
                if arr is not None and arr.size > 0:
                    streams.append(arr)
        return streams

    def _separate_short(self, waveform_8k: torch.Tensor, chunk_path: str) -> list[SeparatedAudio]:
        streams = self._run_model_on_waveform(waveform_8k)
        return [
            SeparatedAudio(speaker_label=None, audio_data=s, sample_rate=MOSSFORMER2_SAMPLE_RATE, file_path=None)
            for s in streams
        ]

    def _separate_long(self, waveform_8k: torch.Tensor, chunk_path: str) -> list[SeparatedAudio]:
        """Separate long audio by splitting into ~30s sub-segments."""
        seg_samples = 30 * MOSSFORMER2_SAMPLE_RATE
        total_samples = waveform_8k.shape[1]
        num_speakers = None
        speaker_chunks: dict[int, list[np.ndarray]] = {}

        offset = 0
        seg_idx = 0
        while offset < total_samples:
            end = min(offset + seg_samples, total_samples)
            streams = self._run_model_on_waveform(waveform_8k[:, offset:end])

            if not streams:
                offset = end
                seg_idx += 1
                continue

            if num_speakers is None:
                num_speakers = len(streams)

            for spk_idx in range(num_speakers):
                if spk_idx not in speaker_chunks:
                    speaker_chunks[spk_idx] = []
                if spk_idx < len(streams):
                    speaker_chunks[spk_idx].append(streams[spk_idx])
                else:
                    speaker_chunks[spk_idx].append(np.zeros(end - offset, dtype=np.float32))

            offset = end
            seg_idx += 1

        if not speaker_chunks:
            return []

        return [
            SeparatedAudio(
                speaker_label=None,
                audio_data=np.concatenate(speaker_chunks[i]),
                sample_rate=MOSSFORMER2_SAMPLE_RATE,
                file_path=None,
            )
            for i in sorted(speaker_chunks.keys())
        ]

    def match_speakers(
        self, separated: list[SeparatedAudio], unified_speakers: UnifiedSpeakerList,
    ) -> list[SeparatedAudio]:
        """Assign speaker identity to separated audio using cosine similarity."""
        if not separated or not unified_speakers.speakers:
            return separated

        matched: list[SeparatedAudio] = []
        for i, sep_audio in enumerate(separated):
            embedding = self._compute_embedding(sep_audio)

            if embedding is not None:
                best_label, best_sim = None, -1.0
                for label, ref_emb in unified_speakers.speakers.items():
                    sim = _cosine_similarity(embedding, ref_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_label = label
                logger.info("Stream %d matched to %s (similarity=%.3f)", i, best_label, best_sim)
            else:
                labels = sorted(unified_speakers.speakers.keys())
                best_label = labels[i % len(labels)] if labels else None
                logger.warning("Could not compute embedding for stream %d, fallback to %s", i, best_label)

            matched.append(SeparatedAudio(
                speaker_label=best_label,
                audio_data=sep_audio.audio_data,
                sample_rate=sep_audio.sample_rate,
                file_path=sep_audio.file_path,
            ))
        return matched

    def _compute_embedding(self, sep_audio: SeparatedAudio) -> np.ndarray | None:
        if self._embedding_model is None:
            return None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            waveform = torch.from_numpy(sep_audio.audio_data).float().unsqueeze(0)
            torchaudio.save(tmp_path, waveform, sep_audio.sample_rate)
            embedding = self._embedding_model(tmp_path)
            emb_array = np.array(embedding).flatten()
            norm = np.linalg.norm(emb_array)
            if norm > 0:
                emb_array = emb_array / norm
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return emb_array
        except Exception:
            logger.debug("Failed to compute embedding", exc_info=True)
            return None

    def save_separated(
        self, separated: list[SeparatedAudio], output_dir: str, chunk_index: int,
    ) -> list[SeparatedAudio]:
        """Save separated audio files organized by speaker label."""
        saved: list[SeparatedAudio] = []
        for seg_idx, sep_audio in enumerate(separated):
            label = sep_audio.speaker_label or "UNKNOWN"
            speaker_dir = os.path.join(output_dir, label)
            os.makedirs(speaker_dir, exist_ok=True)

            filename = f"{label}_chunk_{chunk_index:03d}_seg_{seg_idx:03d}.wav"
            file_path = os.path.join(speaker_dir, filename)
            waveform = torch.from_numpy(sep_audio.audio_data).float().unsqueeze(0)
            torchaudio.save(file_path, waveform, sep_audio.sample_rate)

            saved.append(SeparatedAudio(
                speaker_label=sep_audio.speaker_label,
                audio_data=sep_audio.audio_data,
                sample_rate=sep_audio.sample_rate,
                file_path=file_path,
            ))
        return saved

    def unload_model(self) -> None:
        """Release separation and embedding models from GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._embedding_model is not None:
            del self._embedding_model
            self._embedding_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Separation models unloaded")
