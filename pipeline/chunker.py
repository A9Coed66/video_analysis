"""AudioChunker: Extract audio from video and chunk into segments."""

from __future__ import annotations

import os
import tempfile

from pydub import AudioSegment

from pipeline.models import ChunkInfo

SUPPORTED_EXTENSIONS = {".wav", ".mp4", ".mkv", ".avi"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi"}


class AudioChunker:
    """Extracts audio from video and chunks WAV files."""

    def __init__(self, max_duration_sec: int = 600):
        self.max_duration_sec = max_duration_sec

    def extract_audio(self, input_path: str) -> str:
        """Extract audio from video (MP4/MKV/AVI) to WAV. Returns path as-is for WAV."""
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        ext = os.path.splitext(input_path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        if ext == ".wav":
            return input_path

        audio = AudioSegment.from_file(input_path, format=ext.lstrip("."))
        input_dir = os.path.dirname(os.path.abspath(input_path))
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        wav_path = os.path.join(input_dir, f"{base_name}_extracted.wav")

        try:
            audio.export(wav_path, format="wav")
        except OSError:
            tmp_dir = tempfile.mkdtemp()
            wav_path = os.path.join(tmp_dir, f"{base_name}_extracted.wav")
            audio.export(wav_path, format="wav")

        return wav_path

    def chunk(self, wav_path: str, start_sec: float = 0.0) -> list[ChunkInfo]:
        """Chunk a WAV file into segments <= max_duration_sec."""
        if not os.path.isfile(wav_path):
            raise FileNotFoundError(f"WAV file not found: {wav_path}")

        audio = AudioSegment.from_wav(wav_path)
        total_duration_ms = len(audio)
        start_ms = int(start_sec * 1000)

        if start_ms >= total_duration_ms:
            raise ValueError(
                f"start_sec ({start_sec}s) is at or beyond audio duration "
                f"({total_duration_ms / 1000:.2f}s)"
            )

        remaining_audio = audio[start_ms:]
        remaining_duration_ms = len(remaining_audio)
        if remaining_duration_ms == 0:
            raise ValueError("Audio has zero duration after start_sec offset")

        max_duration_ms = self.max_duration_sec * 1000
        wav_dir = os.path.dirname(os.path.abspath(wav_path))
        chunks_dir = os.path.join(wav_dir, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)

        chunks: list[ChunkInfo] = []
        chunk_index = 0
        offset_ms = 0

        while offset_ms < remaining_duration_ms:
            chunk_end_ms = min(offset_ms + max_duration_ms, remaining_duration_ms)
            chunk_audio = remaining_audio[offset_ms:chunk_end_ms]

            chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_index:03d}.wav")
            chunk_audio.export(chunk_path, format="wav")

            abs_start_sec = start_sec + offset_ms / 1000.0
            abs_end_sec = start_sec + chunk_end_ms / 1000.0
            duration_sec = (chunk_end_ms - offset_ms) / 1000.0

            chunks.append(ChunkInfo(
                chunk_index=chunk_index,
                file_path=chunk_path,
                start_sec=abs_start_sec,
                end_sec=abs_end_sec,
                duration_sec=duration_sec,
            ))

            chunk_index += 1
            offset_ms = chunk_end_ms

        return chunks
