"""Serialization utilities for the Audio Diarization Pipeline."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from pipeline.models import TranscriptEntry, UnifiedSpeakerList


def serialize_unified_speakers(unified: UnifiedSpeakerList) -> dict:
    speakers_data = {}
    for label, embedding in unified.speakers.items():
        speakers_data[label] = {
            "embedding": base64.b64encode(embedding.tobytes()).decode("ascii"),
            "dtype": str(embedding.dtype),
            "shape": list(embedding.shape),
        }
    label_mapping_data = [
        {"chunk_index": ci, "local_label": ll, "unified_label": ul}
        for (ci, ll), ul in unified.label_mapping.items()
    ]
    return {"speakers": speakers_data, "label_mapping": label_mapping_data}


def deserialize_unified_speakers(data: dict) -> UnifiedSpeakerList:
    speakers: dict[str, np.ndarray] = {}
    for label, info in data["speakers"].items():
        raw = base64.b64decode(info["embedding"])
        speakers[label] = np.frombuffer(raw, dtype=np.dtype(info["dtype"])).reshape(tuple(info["shape"])).copy()
    label_mapping: dict[tuple[int, str], str] = {
        (e["chunk_index"], e["local_label"]): e["unified_label"]
        for e in data["label_mapping"]
    }
    return UnifiedSpeakerList(speakers=speakers, label_mapping=label_mapping)


def save_speakers_json(unified: UnifiedSpeakerList, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialize_unified_speakers(unified), f, indent=2, ensure_ascii=False)


def load_speakers_json(path: str) -> UnifiedSpeakerList:
    with open(path, "r", encoding="utf-8") as f:
        return deserialize_unified_speakers(json.load(f))


def save_transcript_json(
    entries: list[TranscriptEntry], source_file: str,
    total_duration: float, speakers: list[str], path: str,
) -> None:
    data = {
        "source_file": source_file,
        "total_duration_sec": total_duration,
        "num_speakers": len(speakers),
        "speakers": speakers,
        "entries": [
            {"speaker": e.speaker_label, "start_time": e.start_time, "end_time": e.end_time, "text": e.text}
            for e in entries
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_transcript_txt(entries: list[TranscriptEntry], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, entry in enumerate(entries):
            start = _format_timestamp(entry.start_time)
            end = _format_timestamp(entry.end_time)
            f.write(f"[{start} - {end}] {entry.speaker_label}\n{entry.text}\n")
            if i < len(entries) - 1:
                f.write("\n")


def _format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:04.1f}"
    return f"{minutes:02d}:{secs:04.1f}"
