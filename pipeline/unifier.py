"""Speaker unification across diarization chunks."""

from __future__ import annotations

import numpy as np

from pipeline.models import DiarizationResult, Segment, UnificationResult, UnifiedSpeakerList


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SpeakerUnifier:
    """Unifies speaker identities across chunks using cosine similarity."""

    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold

    def unify(self, chunk_results: list[DiarizationResult]) -> UnificationResult:
        """Compare embeddings across chunks and create a unified speaker list."""
        entries: list[tuple[int, str, np.ndarray]] = []
        for result in chunk_results:
            for label, embedding in result.embeddings.items():
                entries.append((result.chunk_index, label, embedding))

        unified_groups: list[list[tuple[int, str, np.ndarray]]] = []
        unified_centroids: list[np.ndarray] = []

        for chunk_index, local_label, embedding in entries:
            best_match_idx, best_similarity = -1, -1.0
            for i, centroid in enumerate(unified_centroids):
                sim = _cosine_similarity(embedding, centroid)
                if sim > best_similarity:
                    best_similarity = sim
                    best_match_idx = i

            if best_match_idx >= 0 and best_similarity > self.similarity_threshold:
                unified_groups[best_match_idx].append((chunk_index, local_label, embedding))
                all_embs = [e for _, _, e in unified_groups[best_match_idx]]
                unified_centroids[best_match_idx] = np.mean(all_embs, axis=0)
            else:
                unified_groups.append([(chunk_index, local_label, embedding)])
                unified_centroids.append(embedding.copy())

        label_mapping: dict[tuple[int, str], str] = {}
        speakers: dict[str, np.ndarray] = {}

        for group_idx, group in enumerate(unified_groups):
            unified_label = f"SPEAKER_{group_idx:02d}"
            speakers[unified_label] = np.mean([e for _, _, e in group], axis=0)
            for chunk_index, local_label, _ in group:
                label_mapping[(chunk_index, local_label)] = unified_label

        unified_speaker_list = UnifiedSpeakerList(speakers=speakers, label_mapping=label_mapping)

        updated_segments: list[Segment] = []
        for result in chunk_results:
            for seg in result.segments:
                unified_label = label_mapping.get(
                    (result.chunk_index, seg.speaker_label), seg.speaker_label
                )
                updated_segments.append(Segment(
                    speaker_label=unified_label,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                ))

        return UnificationResult(
            unified_speakers=unified_speaker_list,
            updated_segments=updated_segments,
        )
