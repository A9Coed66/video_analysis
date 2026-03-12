# Voice Diarization & Transcription Pipeline

A system that tackles two problems simultaneously:

1. **Speaker-attributed subtitles** — Separate a 2-speaker audio into individual voice streams and generate subtitles with clear speaker labels
2. **English translation** — Translate the transcribed content into English scripts

## Goal

Given a single audio with 2 speakers → split into 2 separate single-speaker audios → transcribe each audio → produce speaker-labeled subtitles with English translation.

**Source language**: Japanese (日本語) → **Output**: English translated subtitles

## Pipeline Architecture

```
Audio/Video Input
       │
       ▼
┌──────────────┐
│  Noise Filter │  ← Extract & clean audio from video
└──────┬───────┘
       ▼
┌──────────────────┐
│ Speaker Diarize  │  ← pyannote/speaker-diarization-3.1
│ + Overlap Resolve│     Identify who speaks when
└──────┬───────────┘
       ▼
┌──────────────────┐
│ Speaker Separator│  ← DPRNN-TasNet (custom trained)
│                  │     Isolate individual voice streams
└──────┬───────────┘
       ▼
┌──────────────────┐
│ Speaker Embedding│  ← Cosine similarity matching
│ + Unification    │     Link speakers across chunks
└──────┬───────────┘
       ▼
┌──────────────────┐
│  Transcription   │  ← Qwen3-ASR-1.7B
│  (per speaker)   │     Speech-to-text for each speaker
└──────┬───────────┘
       ▼
  Attributed Transcript
  (speaker, timestamp, text)
```

## Output

```
output/
├── transcript.json      # Full transcript with speaker labels + timestamps
├── transcript.txt       # Human-readable transcript
├── speakers.json        # Speaker list with embedding vectors
├── pipeline.log         # Processing log
├── chunks/              # Audio chunks
└── separated/           # Isolated audio per speaker
    ├── SPEAKER_00/
    └── SPEAKER_01/
```

## Demo — Speaker Separation Results

Sample results from the current model. Each mix is a 2-speaker audio that gets separated into individual speaker tracks.

```
sample/
├── mix_1.wav                 # Input: 2 speakers mixed
├── mix_1_speaker_1.wav       # Output: Speaker 1 isolated
├── mix_1_speaker_2.wav       # Output: Speaker 2 isolated
├── mix_2.wav                 # Input: 2 speakers mixed
├── mix_2_speaker_1.wav       # Output: Speaker 1 isolated
└── mix_2_speaker_2.wav       # Output: Speaker 2 isolated
```

| Sample | Input (mixed) | Output Speaker 1 | Output Speaker 2 |
|--------|---------------|-------------------|-------------------|
| Mix 1  | [mix_1.wav](sample/mix_1.wav) | [mix_1_speaker_1.wav](sample/mix_1_speaker_1.wav) | [mix_1_speaker_2.wav](sample/mix_1_speaker_2.wav) |
| Mix 2  | [mix_2.wav](sample/mix_2.wav) | [mix_2_speaker_1.wav](sample/mix_2_speaker_1.wav) | [mix_2_speaker_2.wav](sample/mix_2_speaker_2.wav) |

## Project Status

### 1. Speaker Separator Model — ✅ Done (fine-tuning ongoing)
- Custom model trained for 2-speaker separation
- Model is functional, will continue fine-tuning for better accuracy
- Training pipeline: normalize → split speakers → create 2-mix data → train

### 2. Business Version — 🔄 In Progress
- Building the production-ready business application
- End-to-end flow: 2-speaker audio → separate into 2 single-speaker audios → transcribe → generate subtitles

### 3. Production Deployment — ✅ TODO
- Dockerized multi-service architecture (docker-compose)
- Nginx reverse proxy with TLS, FastAPI backend, Celery workers with GPU assignment
- Redis message broker, API key auth, health checks

### 4. Web Interface — ✅ TODO
- Single-page app with dark theme
- Drag-and-drop upload, async job processing, audio preview & download

### 5. Full Pipeline Integration — ✅ TODO
- 5-phase pipeline with per-phase GPU memory management
- Chunking for long audio, cross-chunk speaker unification

### 6. Testing — ✅ TODO
- Unit tests, property-based tests (Hypothesis), integration tests
- 129 tests covering backend, frontend, and integration

## Tech Stack

- **Backend**: FastAPI, Celery, Redis
- **ML**: PyTorch, speaker separation, ASR, vector embedding, denoiser
- **Infra**: Docker, Nginx, NVIDIA CUDA
