# Voice Separator Web

Web interface for AI-powered voice separation and audio processing pipeline.

## Overview

Single-page application with a FastAPI backend and vanilla HTML/CSS/JS frontend. Two main features:

1. **Voice Separator** — Split a 2-speaker audio file into individual speaker tracks using DPRNN-TasNet
2. **Pipeline** — Run a full audio processing pipeline: chunking → diarization → separation → transcription → unification

## Tech Stack

- **Backend**: Python 3.11+, FastAPI, Uvicorn
- **Frontend**: Vanilla HTML5, CSS3 (custom properties, dark theme), JavaScript (ES5 IIFE modules)
- **ML Models**: DPRNN-TasNet (voice separation), PyAnnote (diarization), Whisper (transcription)
- **Testing**: pytest, Hypothesis (property-based testing)

## Project Structure

```text
web/
├── app.py                     # FastAPI application, routes, background tasks
├── jobs.py                    # In-memory job manager (pending → processing → completed/failed)
├── models.py                  # Pydantic response models + Job dataclass
├── requirements.txt           # Python dependencies
├── services/
│   ├── separation.py          # DPRNN-TasNet wrapper (voice_separator/06_predict.py)
│   └── pipeline.py            # Pipeline orchestrator wrapper (pipeline/orchestrator.py)
└── static/
    ├── index.html             # Single-page app entry point
    ├── styles/
    │   ├── theme.css          # CSS custom properties, dark theme, reset
    │   ├── layout.css         # Grid system, responsive breakpoints, navbar
    │   ├── components.css     # Upload zone, progress, audio player, buttons, cards
    │   └── accessibility.css  # Reduced motion, focus styles, skip link, sr-only
    └── js/
        ├── api.js             # REST API client with polling logic
        ├── upload.js          # Drag-and-drop upload zone component
        ├── audio-player.js    # Audio player helpers and dynamic player factory
        └── app.js             # Main app: navigation, separator flow, pipeline flow
```

## Installation & Run

```bash
# Install dependencies (from project root)
pip install -r web/requirements.txt

# Start the dev server
uvicorn web.app:app --reload

# Open http://127.0.0.1:8000
```

The ML models (DPRNN-TasNet, PyAnnote, Whisper) are loaded lazily on first request. Ensure:
- GPU drivers are installed for CUDA acceleration
- `voice_separator/checkpoints_premium/` contains trained checkpoints
- `HF_TOKEN` env var is set for PyAnnote diarization model access

## API Endpoints

| Method | Path                    | Description                          |
|--------|-------------------------|--------------------------------------|
| GET    | `/`                     | Serve the SPA                        |
| POST   | `/api/separate`         | Upload audio → 2-speaker separation  |
| POST   | `/api/pipeline`         | Upload audio → full pipeline         |
| GET    | `/api/status/{job_id}`  | Poll job status                      |
| GET    | `/api/download/{file_id}` | Download result file               |

## Running Tests

```bash
# All tests (129 total: unit + property-based + integration)
python -m pytest tests/ -v

# Backend only
python -m pytest tests/backend/ -v

# Frontend property tests only
python -m pytest tests/frontend/ -v
```

## Features

- Dark theme with 8px grid system
- Drag-and-drop file upload with client-side validation
- Async job processing with real-time status polling
- Multi-step pipeline progress indicator
- Audio preview and playback for results
- Responsive layout (mobile / tablet / desktop)
- Keyboard navigation and accessibility (skip links, aria-labels, focus styles)
- Retry on failure with error messages

## Future Improvements

- Persistent job storage (SQLite/Redis) instead of in-memory dict
- WebSocket for real-time progress instead of polling
- Light theme toggle
- Batch file processing
- Waveform visualization
- User authentication and job history
