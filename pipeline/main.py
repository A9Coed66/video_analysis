"""Entry point for the Audio Diarization Pipeline.

Usage:
    python -m pipeline.main input.wav
    python -m pipeline.main input.mp4 --output-dir results
    python pipeline/main.py input.wav --similarity-threshold 0.8
"""

from __future__ import annotations

import argparse
import os
import sys

import pipeline.compat  # noqa: F401
pipeline.compat.apply_patches()

from dotenv import load_dotenv

from pipeline.models import PipelineConfig
from pipeline.orchestrator import PipelineOrchestrator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audio Diarization Pipeline",
    )
    parser.add_argument("input_path", help="Path to input audio/video file")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument(
        "--max-chunk-duration-sec", type=int, default=None,
        help="Max chunk duration in seconds (default: 600)",
    )
    parser.add_argument(
        "--similarity-threshold", type=float, default=None,
        help="Cosine similarity threshold for speaker matching (default: 0.7)",
    )
    return parser.parse_args(argv)


def load_config(args: argparse.Namespace) -> PipelineConfig:
    """Load PipelineConfig from .env and CLI arguments (CLI > env > defaults)."""
    load_dotenv()
    config = PipelineConfig(hf_token=os.environ.get("HF_TOKEN", ""))
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.max_chunk_duration_sec is not None:
        config.max_chunk_duration_sec = args.max_chunk_duration_sec
    if args.similarity_threshold is not None:
        config.similarity_threshold = args.similarity_threshold
    return config


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args)

    if not config.hf_token:
        print("Error: HF_TOKEN not found. Set it in .env or as env variable.", file=sys.stderr)
        sys.exit(1)

    orchestrator = PipelineOrchestrator(config)
    output_dir = args.output_dir or config.output_dir

    try:
        result = orchestrator.run(args.input_path, output_dir)
        print(f"Pipeline completed. Output: {result.output_dir}")
        print(f"Speakers: {len(result.unified_speakers.speakers)}")
        print(f"Transcript entries: {len(result.transcript)}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
