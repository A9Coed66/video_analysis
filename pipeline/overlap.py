"""Overlap resolution between consecutive audio chunks."""

from __future__ import annotations

from pipeline.models import ResolveResult, Segment


class OverlapResolver:
    """Resolves overlap regions between consecutive diarization chunks."""

    def resolve(self, segments: list[Segment], is_last_chunk: bool) -> ResolveResult:
        """Remove the last segment and all overlapping segments."""
        if is_last_chunk or not segments:
            return ResolveResult(
                cleaned_segments=list(segments),
                removed_segments=[],
                next_start_sec=None,
            )

        last_seg = segments[-1]
        removed: list[Segment] = []
        cleaned: list[Segment] = []

        for seg in segments:
            if seg.start_time < last_seg.end_time and seg.end_time > last_seg.start_time:
                removed.append(seg)
            else:
                cleaned.append(seg)

        if last_seg not in removed:
            removed.append(last_seg)
            if last_seg in cleaned:
                cleaned.remove(last_seg)

        next_start_sec = min(seg.start_time for seg in removed)

        return ResolveResult(
            cleaned_segments=cleaned,
            removed_segments=removed,
            next_start_sec=next_start_sec,
        )
