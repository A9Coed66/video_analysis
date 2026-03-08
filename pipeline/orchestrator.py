"""PipelineOrchestrator: Orchestrates the entire audio diarization pipeline."""

from __future__ import annotations

import logging
import os
import tempfile
import time

import soundfile as sf
import torch
import torchaudio

from pipeline.chunker import AudioChunker
from pipeline.diarization import DiarizationEngine
from pipeline.models import (
    ChunkInfo,
    DiarizationResult,
    PipelineConfig,
    PipelineResult,
    Segment,
    TranscriptEntry,
)
from pipeline.overlap import OverlapResolver
from pipeline.separation import SeparationEngine
from pipeline.serialization import save_speakers_json, save_transcript_json, save_transcript_txt
from pipeline.transcription import TranscriptionEngine
from pipeline.unifier import SpeakerUnifier

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the entire audio diarization pipeline."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._processing_log: list[str] = []

    def _log(self, message: str) -> None:
        logger.info(message)
        self._processing_log.append(message)

    def _transcribe_segments_fallback(
        self,
        transcription_engine: TranscriptionEngine,
        segments: list[Segment],
        chunks: list[ChunkInfo],
    ) -> list[TranscriptEntry]:
        """Transcribe by cropping each segment directly from chunk WAV files."""
        entries: list[TranscriptEntry] = []

        for i, seg in enumerate(segments):
            chunk = self._find_chunk_for_segment(seg, chunks)
            if chunk is None:
                entries.append(self._unrecognized_entry(seg))
                continue

            text = self._crop_and_transcribe(transcription_engine, seg, chunk)
            entries.append(TranscriptEntry(
                speaker_label=seg.speaker_label,
                start_time=seg.start_time,
                end_time=seg.end_time,
                text=text,
            ))

            if (i + 1) % 50 == 0:
                self._log(f"Transcribed {i + 1}/{len(segments)} segments")

        self._log(f"Fallback transcription: {len(entries)} segments processed")
        return entries

    @staticmethod
    def _find_chunk_for_segment(seg: Segment, chunks: list[ChunkInfo]) -> ChunkInfo | None:
        for c in chunks:
            if c.start_sec <= seg.start_time < c.end_sec:
                return c
        return chunks[-1] if chunks else None

    @staticmethod
    def _unrecognized_entry(seg: Segment) -> TranscriptEntry:
        return TranscriptEntry(
            speaker_label=seg.speaker_label,
            start_time=seg.start_time,
            end_time=seg.end_time,
            text="[không nhận diện được]",
        )

    def _crop_and_transcribe(
        self, engine: TranscriptionEngine, seg: Segment, chunk: ChunkInfo,
    ) -> str:
        try:
            info = sf.info(chunk.file_path)
            sr = info.samplerate
            rel_start = seg.start_time - chunk.start_sec
            rel_end = seg.end_time - chunk.start_sec
            start_frame = max(0, int(rel_start * sr))
            end_frame = min(info.frames, int(rel_end * sr))

            if end_frame - start_frame < sr * 0.1:
                return ""

            data, sample_rate = sf.read(chunk.file_path, start=start_frame, stop=end_frame, dtype="float32")
            waveform = torch.from_numpy(data)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            torchaudio.save(tmp_path, waveform, sample_rate)

            try:
                return engine.transcribe(tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        except Exception:
            logger.exception("Failed to crop/transcribe segment at %.1f-%.1fs", seg.start_time, seg.end_time)
            return "[không nhận diện được]"

    def run(self, input_path: str, output_dir: str) -> PipelineResult:
        """Run the entire pipeline: chunk -> diarize -> unify -> separate -> transcribe -> save."""
        self._processing_log = []
        pipeline_start = time.time()
        self._log(f"Pipeline started for: {input_path}")

        os.makedirs(output_dir, exist_ok=True)
        separated_dir = os.path.join(output_dir, "separated")
        os.makedirs(separated_dir, exist_ok=True)

        # Phase 1: Chunking + Diarization + Overlap Resolution
        all_chunks, all_diar_results, all_cleaned_segments = self._run_phase1(input_path)

        # Phase 2: Speaker Unification
        unification_result, all_unified_segments = self._run_phase2(
            all_diar_results, all_cleaned_segments
        )

        # Phase 3: Speech Separation
        self._run_phase3(all_chunks, unification_result, separated_dir)

        # Phase 4: Transcription
        transcript_entries = self._run_phase4(all_unified_segments, all_chunks)

        # Phase 5: Save Outputs
        self._run_phase5(
            transcript_entries, unification_result, all_chunks, input_path, output_dir
        )

        total_time = time.time() - pipeline_start
        self._log(f"Pipeline completed in {total_time:.1f}s")

        return PipelineResult(
            transcript=transcript_entries,
            unified_speakers=unification_result.unified_speakers,
            output_dir=output_dir,
            processing_log=self._processing_log,
        )

    def _run_phase1(self, input_path: str):
        """Phase 1: Chunking + Diarization + Overlap Resolution."""
        phase_start = time.time()
        self._log("Phase 1: Chunking + Diarization + Overlap Resolution")

        chunker = AudioChunker(max_duration_sec=self.config.max_chunk_duration_sec)
        wav_path = chunker.extract_audio(input_path)
        self._log(f"Audio extracted: {wav_path}")

        diarization_engine = DiarizationEngine(hf_token=self.config.hf_token)
        overlap_resolver = OverlapResolver()

        all_chunks: list[ChunkInfo] = []
        all_diar_results: list[DiarizationResult] = []
        all_cleaned_segments: list[list[Segment]] = []

        try:
            diarization_engine.load_model()
            self._log("Diarization model loaded")

            start_sec = 0.0
            chunk_idx = 0
            finished = False

            while not finished:
                chunks = chunker.chunk(wav_path, start_sec=start_sec)
                if not chunks:
                    break

                chunk = ChunkInfo(
                    chunk_index=chunk_idx,
                    file_path=chunks[0].file_path,
                    start_sec=chunks[0].start_sec,
                    end_sec=chunks[0].end_sec,
                    duration_sec=chunks[0].duration_sec,
                )
                all_chunks.append(chunk)
                self._log(f"Chunk {chunk_idx}: {chunk.start_sec:.1f}s - {chunk.end_sec:.1f}s")

                diar_result = diarization_engine.diarize(chunk.file_path, chunk_index=chunk_idx)
                for seg in diar_result.segments:
                    seg.start_time += chunk.start_sec
                    seg.end_time += chunk.start_sec

                self._log(
                    f"Chunk {chunk_idx}: {len(diar_result.segments)} segments, "
                    f"{len(diar_result.embeddings)} speakers"
                )

                is_last = len(chunks) == 1
                resolve_result = overlap_resolver.resolve(diar_result.segments, is_last_chunk=is_last)
                all_diar_results.append(diar_result)
                all_cleaned_segments.append(resolve_result.cleaned_segments)

                if resolve_result.next_start_sec is not None:
                    start_sec = resolve_result.next_start_sec
                    chunk_idx += 1
                else:
                    finished = True
        finally:
            diarization_engine.unload_model()
            self._log("Diarization model unloaded")

        self._log(f"Phase 1 completed in {time.time() - phase_start:.1f}s")
        return all_chunks, all_diar_results, all_cleaned_segments

    def _run_phase2(self, all_diar_results, all_cleaned_segments):
        """Phase 2: Speaker Unification."""
        phase_start = time.time()
        self._log("Phase 2: Speaker Unification")

        unifier = SpeakerUnifier(similarity_threshold=self.config.similarity_threshold)
        unification_result = unifier.unify(all_diar_results)
        unified_mapping = unification_result.unified_speakers.label_mapping

        self._log(f"Unified {len(unification_result.unified_speakers.speakers)} speakers")

        all_unified_segments: list[Segment] = []
        for chunk_idx, cleaned_segs in enumerate(all_cleaned_segments):
            for seg in cleaned_segs:
                label = unified_mapping.get((chunk_idx, seg.speaker_label), seg.speaker_label)
                all_unified_segments.append(Segment(
                    speaker_label=label, start_time=seg.start_time, end_time=seg.end_time,
                ))

        self._log(f"Phase 2 completed in {time.time() - phase_start:.1f}s")
        return unification_result, all_unified_segments

    def _run_phase3(self, all_chunks, unification_result, separated_dir):
        """Phase 3: Speech Separation."""
        phase_start = time.time()
        self._log("Phase 3: Speech Separation")

        separation_engine = SeparationEngine()
        try:
            separation_engine.load_model()
            self._log("Separation model loaded")

            for chunk in all_chunks:
                self._log(f"Separating chunk {chunk.chunk_index}")
                separated = separation_engine.separate(chunk.file_path)
                if separated:
                    matched = separation_engine.match_speakers(
                        separated, unification_result.unified_speakers,
                    )
                    saved = separation_engine.save_separated(matched, separated_dir, chunk.chunk_index)
                    self._log(f"Chunk {chunk.chunk_index}: separated into {len(saved)} streams")
                else:
                    self._log(f"Chunk {chunk.chunk_index}: no streams")
        finally:
            separation_engine.unload_model()
            self._log("Separation model unloaded")

        self._log(f"Phase 3 completed in {time.time() - phase_start:.1f}s")

    def _run_phase4(self, all_unified_segments, all_chunks):
        """Phase 4: Transcription."""
        phase_start = time.time()
        self._log("Phase 4: Transcription")

        transcription_engine = TranscriptionEngine()
        try:
            transcription_engine.load_model()
            self._log("Transcription model loaded")
            self._log(f"Transcribing {len(all_unified_segments)} segments")
            entries = self._transcribe_segments_fallback(
                transcription_engine, all_unified_segments, all_chunks,
            )
        finally:
            transcription_engine.unload_model()
            self._log("Transcription model unloaded")

        self._log(f"Phase 4 completed in {time.time() - phase_start:.1f}s")
        return entries

    def _run_phase5(self, transcript_entries, unification_result, all_chunks, input_path, output_dir):
        """Phase 5: Save Outputs."""
        phase_start = time.time()
        self._log("Phase 5: Saving outputs")

        transcript_entries.sort(key=lambda e: e.start_time)
        total_duration = max((c.end_sec for c in all_chunks), default=0.0)
        speaker_labels = sorted(unification_result.unified_speakers.speakers.keys())

        save_transcript_json(
            entries=transcript_entries,
            source_file=os.path.basename(input_path),
            total_duration=total_duration,
            speakers=speaker_labels,
            path=os.path.join(output_dir, "transcript.json"),
        )
        save_transcript_txt(
            entries=transcript_entries,
            path=os.path.join(output_dir, "transcript.txt"),
        )
        save_speakers_json(
            unified=unification_result.unified_speakers,
            path=os.path.join(output_dir, "speakers.json"),
        )

        log_path = os.path.join(output_dir, "pipeline.log")
        with open(log_path, "w", encoding="utf-8") as f:
            for entry in self._processing_log:
                f.write(entry + "\n")

        self._log(f"Phase 5 completed in {time.time() - phase_start:.1f}s")
