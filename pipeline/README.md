# Audio Diarization Pipeline

Pipeline xử lý file audio/video dài để nhận diện người nói (speaker diarization), tách giọng nói (speech separation), và chuyển đổi giọng nói thành văn bản (speech-to-text).

## Bài toán giải quyết

Cho một file audio/video có nhiều người nói, pipeline tự động:
1. Xác định ai nói ở đoạn nào (diarization)
2. Tách riêng giọng từng người (separation)
3. Chuyển giọng nói thành text (transcription)
4. Gán đúng text cho đúng người nói

## Đầu vào

- File audio/video: `.wav`, `.mp4`, `.mkv`, `.avi`
- File `.env` chứa `HF_TOKEN` (token HuggingFace để truy cập model pyannote)

## Đầu ra

```
output/
├── transcript.json      # Transcript dạng JSON (speaker, start, end, text)
├── transcript.txt       # Transcript dạng text dễ đọc
├── speakers.json        # Danh sách speaker + embedding vectors
├── pipeline.log         # Log quá trình xử lý
├── chunks/              # Các đoạn audio đã chia nhỏ
└── separated/           # Audio đã tách theo từng speaker
    ├── SPEAKER_00/
    └── SPEAKER_01/
```

## Kiến trúc Pipeline

Pipeline chạy tuần tự qua 5 phase, mỗi phase load/unload model riêng để tiết kiệm GPU:

```
Phase 1: Chunking + Diarization + Overlap Resolution
  ├── Trích audio từ video (nếu cần)
  ├── Chia audio thành chunk ≤ 10 phút
  ├── Chạy pyannote/speaker-diarization-3.1 trên từng chunk
  └── Xử lý overlap giữa các chunk liên tiếp

Phase 2: Speaker Unification
  └── So sánh embedding cosine similarity để gán label thống nhất

Phase 3: Speech Separation
  ├── Resample audio về 8kHz
  ├── Chạy mossformer2 tách nguồn
  └── Match speaker bằng embedding similarity

Phase 4: Transcription
  └── Chạy Qwen3-ASR trên từng segment

Phase 5: Save Outputs
  └── Lưu transcript.json, transcript.txt, speakers.json, pipeline.log
```

## Cấu trúc code

| File | Chức năng |
|------|-----------|
| `main.py` | Entry point, parse CLI args, load config từ `.env` |
| `orchestrator.py` | Điều phối 5 phase, quản lý GPU memory |
| `chunker.py` | Trích audio từ video, chia chunk |
| `diarization.py` | Speaker diarization (pyannote 3.1) |
| `overlap.py` | Xử lý overlap giữa các chunk |
| `unifier.py` | Thống nhất speaker identity qua các chunk |
| `separation.py` | Tách nguồn giọng nói (mossformer2) |
| `transcription.py` | Speech-to-text (Qwen3-ASR) |
| `serialization.py` | Serialize/deserialize kết quả, lưu JSON/TXT |
| `models.py` | Dataclasses cho config, segment, result |
| `exceptions.py` | Custom exceptions |
| `compat.py` | Patch tương thích pyannote 4.x + torchaudio 2.6+ trên Windows |

## Cách chạy

```bash
# Cơ bản
python -m pipeline input.wav

# Tùy chỉnh
python -m pipeline input.mp4 --output-dir results --max-chunk-duration-sec 300 --similarity-threshold 0.8
```

## Models sử dụng

| Model | Tác vụ | Sample rate |
|-------|--------|-------------|
| pyannote/speaker-diarization-3.1 | Diarization | 16kHz |
| pyannote/embedding | Speaker embedding | 16kHz |
| damo/speech_mossformer2_separation_temporal_8k | Speech separation | 8kHz |
| Qwen/Qwen3-ASR-1.7B | Speech-to-text | Auto |

## Yêu cầu

- Python 3.10+
- GPU (khuyến nghị, pipeline hỗ trợ fallback CPU nhưng rất chậm)
- HuggingFace token có quyền truy cập pyannote models
- Dependencies: xem `requirements.txt`
