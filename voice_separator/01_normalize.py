"""
01_normalize.py — Chuẩn hóa audio nguồn từ data_source/ thành segment 6s WAV 16kHz mono.

Pipeline: quét MP3 → resample 16kHz mono → Silero VAD detect speech →
          mỗi speech segment + random silent padding đầu/cuối → cắt cứng 6s → lưu WAV.

Output: segments/{speaker_id}/{speaker_id}_{index:05d}.wav
"""

import argparse
import logging
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SR = 16000
DEFAULT_SEG_LEN = 6.0  # seconds
DEFAULT_MAX_PAD = 1.0  # max random silence padding (seconds)
SAMPLES_PER_SEGMENT = int(DEFAULT_SR * DEFAULT_SEG_LEN)


# ---------------------------------------------------------------------------
# Silero VAD (loaded once, reused across files)
# ---------------------------------------------------------------------------
_vad_model = None


def get_vad_model():
    """Load Silero VAD model (singleton)."""
    global _vad_model
    if _vad_model is None:
        _vad_model, _utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        logger.info("Silero VAD model loaded")
    return _vad_model


def detect_speech_segments(
    audio: np.ndarray, sr: int = DEFAULT_SR,
) -> list[dict]:
    """Dùng Silero VAD để detect các đoạn có tiếng nói.

    Returns:
        list[{'start': float, 'end': float}] — timestamps tính bằng giây.
    """
    from silero_vad import get_speech_timestamps, read_audio  # noqa: F811

    model = get_vad_model()
    wav_tensor = torch.from_numpy(audio).float()
    # Silero VAD expects mono 16kHz
    timestamps = get_speech_timestamps(
        wav_tensor, model, sampling_rate=sr, return_seconds=True,
    )
    return timestamps


def pad_and_cut(
    speech: np.ndarray,
    sr: int = DEFAULT_SR,
    seg_len: float = DEFAULT_SEG_LEN,
    max_pad: float = DEFAULT_MAX_PAD,
) -> list[np.ndarray]:
    """Tạo audio mới từ speech segment: thêm random silence đầu/cuối, cắt cứng seg_len.

    1. Tạo: [random 0-max_pad s silence] + speech + [random 0-max_pad s silence]
    2. Cắt cứng seg_len giây liên tục, segment cuối pad thêm silence cho đủ.

    Returns:
        list[np.ndarray]: Các segment đúng seg_len giây.
    """
    samples_per_seg = int(sr * seg_len)

    # Random silent padding
    pad_front_samples = int(random.uniform(0, max_pad) * sr)
    pad_back_samples = int(random.uniform(0, max_pad) * sr)

    padded = np.concatenate([
        np.zeros(pad_front_samples, dtype=np.float32),
        speech,
        np.zeros(pad_back_samples, dtype=np.float32),
    ])

    # Cắt cứng seg_len
    segments = []
    offset = 0
    while offset < len(padded):
        chunk = padded[offset : offset + samples_per_seg]
        if len(chunk) < samples_per_seg:
            # Segment cuối: pad silence cho đủ
            chunk = np.pad(chunk, (0, samples_per_seg - len(chunk)))
        segments.append(chunk)
        offset += samples_per_seg

    return segments


def scan_mp3_files(data_source_dir: str) -> dict[str, list[Path]]:
    """Quét đệ quy data_source/, trả về {speaker_id: [mp3_paths]}.

    Mỗi thư mục con trực tiếp của data_source_dir được coi là một speaker_id.
    """
    source_path = Path(data_source_dir)
    if not source_path.exists():
        raise FileNotFoundError(
            f"Thư mục data_source không tồn tại: {data_source_dir}"
        )

    speaker_files: dict[str, list[Path]] = {}
    for speaker_dir in sorted(source_path.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker_id = speaker_dir.name
        mp3_files = sorted(speaker_dir.rglob("*.mp3"))
        if mp3_files:
            speaker_files[speaker_id] = mp3_files

    if not speaker_files:
        raise ValueError(
            f"Không tìm thấy file MP3 trong data_source/ ({data_source_dir})"
        )

    return speaker_files


def load_and_resample(mp3_path: Path, target_sr: int = DEFAULT_SR) -> np.ndarray:
    """Đọc MP3, chuyển sang mono float32, resample về target_sr."""
    audio, _ = librosa.load(str(mp3_path), sr=target_sr, mono=True)
    return audio.astype(np.float32)


def normalize_all(
    data_source_dir: str,
    output_dir: str = "segments",
    target_sr: int = DEFAULT_SR,
    seg_len: float = DEFAULT_SEG_LEN,
    max_pad: float = DEFAULT_MAX_PAD,
) -> dict:
    """Pipeline chính: quét → resample → Silero VAD → pad + cắt 6s → lưu WAV.

    Args:
        data_source_dir: Đường dẫn tới thư mục data_source/.
        output_dir: Thư mục output cho segments.
        target_sr: Sample rate mục tiêu (mặc định 16000).
        seg_len: Độ dài mỗi segment tính bằng giây (mặc định 6.0).
        max_pad: Max random silence padding đầu/cuối mỗi speech segment (mặc định 1.0s).

    Returns:
        dict với thống kê.
    """
    speaker_files = scan_mp3_files(data_source_dir)

    stats = {
        "files_processed": 0,
        "total_segments": 0,
        "speech_regions_found": 0,
        "speakers": {},
    }

    for speaker_id, mp3_paths in speaker_files.items():
        speaker_output = Path(output_dir) / speaker_id
        speaker_output.mkdir(parents=True, exist_ok=True)

        speaker_segments = 0
        speaker_speech_regions = 0
        speaker_files_ok = 0
        seg_index = 0

        for mp3_path in mp3_paths:
            try:
                audio = load_and_resample(mp3_path, target_sr)
            except Exception as e:
                logger.warning(f"Không đọc được file {mp3_path}: {e}")
                continue

            speaker_files_ok += 1

            # Silero VAD: detect speech regions
            speech_timestamps = detect_speech_segments(audio, target_sr)
            speaker_speech_regions += len(speech_timestamps)

            if not speech_timestamps:
                logger.warning(f"Không phát hiện speech trong {mp3_path}")
                continue

            for ts in speech_timestamps:
                start_sample = int(ts["start"] * target_sr)
                end_sample = int(ts["end"] * target_sr)
                speech_chunk = audio[start_sample:end_sample]

                # Tạo audio mới với random silent padding + cắt cứng 6s
                cut_segments = pad_and_cut(speech_chunk, target_sr, seg_len, max_pad)

                for seg in cut_segments:
                    out_path = speaker_output / f"{speaker_id}_{seg_index:05d}.wav"
                    sf.write(str(out_path), seg, target_sr, subtype="FLOAT")
                    seg_index += 1
                    speaker_segments += 1

        stats["files_processed"] += speaker_files_ok
        stats["total_segments"] += speaker_segments
        stats["speech_regions_found"] += speaker_speech_regions
        stats["speakers"][speaker_id] = {
            "files_processed": speaker_files_ok,
            "segments_created": speaker_segments,
            "speech_regions_found": speaker_speech_regions,
        }

        logger.info(
            f"Speaker {speaker_id}: {speaker_files_ok} files → "
            f"{speaker_speech_regions} speech regions → {speaker_segments} segments"
        )

    logger.info("=" * 60)
    logger.info("THỐNG KÊ TỔNG HỢP")
    logger.info(f"  Số file xử lý:              {stats['files_processed']}")
    logger.info(f"  Số speech regions phát hiện: {stats['speech_regions_found']}")
    logger.info(f"  Số segment tạo được:         {stats['total_segments']}")
    logger.info("=" * 60)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chuẩn hóa audio: resample 16kHz mono, Silero VAD detect speech, "
                    "random silent padding, cắt 6s segments."
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="/home/tuanlha/research/video_analysis/no_sound_effect",
        help="Đường dẫn tới thư mục data_source/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="segments",
        help="Thư mục output cho segments (mặc định: segments)",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=DEFAULT_SR,
        help=f"Sample rate mục tiêu (mặc định: {DEFAULT_SR})",
    )
    parser.add_argument(
        "--seg-len",
        type=float,
        default=DEFAULT_SEG_LEN,
        help=f"Độ dài segment tính bằng giây (mặc định: {DEFAULT_SEG_LEN})",
    )
    parser.add_argument(
        "--max-pad",
        type=float,
        default=DEFAULT_MAX_PAD,
        help=f"Max random silence padding đầu/cuối (mặc định: {DEFAULT_MAX_PAD}s)",
    )

    args = parser.parse_args()

    normalize_all(
        data_source_dir=args.data_source,
        output_dir=args.output_dir,
        target_sr=args.sr,
        seg_len=args.seg_len,
        max_pad=args.max_pad,
    )
