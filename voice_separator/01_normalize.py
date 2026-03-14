"""
01_normalize.py — Chuẩn hóa audio nguồn từ data_source/ thành segment 4s WAV 16kHz mono.

Pipeline: quét MP3 → resample 16kHz mono → cắt 4s segments → lọc im lặng → lưu WAV.
Output: segments/{speaker_id}/{speaker_id}_{index:05d}.wav
"""

import argparse
import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SR = 16000
DEFAULT_SEG_LEN = 6.0  # seconds
DEFAULT_SILENCE_THRESHOLD = 1e-4
SAMPLES_PER_SEGMENT = int(DEFAULT_SR * DEFAULT_SEG_LEN)  # 64000


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
    """Đọc MP3, chuyển sang mono float32, resample về target_sr.

    Returns:
        np.ndarray: Audio array mono float32 tại target_sr.

    Raises:
        Exception nếu file không đọc được (caller nên bắt).
    """
    audio, _ = librosa.load(str(mp3_path), sr=target_sr, mono=True)
    return audio.astype(np.float32)


def cut_segments(
    audio: np.ndarray, sr: int = DEFAULT_SR, seg_len: float = DEFAULT_SEG_LEN
) -> list[np.ndarray]:
    """Cắt audio thành các đoạn seg_len giây với 50% overlap, bỏ phần dư < seg_len.

    Overlap 50%: segment i bắt đầu tại i * (samples_per_seg / 2).
    Ví dụ: seg0 = [0 : L], seg1 = [L/2 : 3L/2], seg2 = [L : 2L], ...

    Returns:
        list[np.ndarray]: Danh sách segments, mỗi segment có đúng sr * seg_len samples.
    """
    samples_per_seg = int(sr * seg_len)
    hop = samples_per_seg // 2
    segments = []
    start = 0
    while start + samples_per_seg <= len(audio):
        segments.append(audio[start : start + samples_per_seg])
        start += hop
    return segments


def is_silent(segment: np.ndarray, threshold: float = DEFAULT_SILENCE_THRESHOLD) -> bool:
    """Kiểm tra RMS < threshold.

    Returns:
        True nếu segment im lặng (RMS < threshold).
    """
    rms = np.sqrt(np.mean(segment ** 2))
    return rms < threshold


def normalize_all(
    data_source_dir: str,
    output_dir: str = "segments",
    target_sr: int = DEFAULT_SR,
    seg_len: float = DEFAULT_SEG_LEN,
    silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
) -> dict:
    """Pipeline chính: quét → resample → cắt → lọc im lặng → lưu WAV.

    Args:
        data_source_dir: Đường dẫn tới thư mục data_source/.
        output_dir: Thư mục output cho segments.
        target_sr: Sample rate mục tiêu (mặc định 16000).
        seg_len: Độ dài mỗi segment tính bằng giây (mặc định 4.0).
        silence_threshold: Ngưỡng RMS để phát hiện im lặng (mặc định 1e-4).

    Returns:
        dict với thống kê: files_processed, total_segments, silent_discarded,
        và per_speaker stats.
    """
    # Quét MP3
    speaker_files = scan_mp3_files(data_source_dir)

    stats = {
        "files_processed": 0,
        "total_segments": 0,
        "silent_discarded": 0,
        "speakers": {},
    }

    for speaker_id, mp3_paths in speaker_files.items():
        speaker_output = Path(output_dir) / speaker_id
        speaker_output.mkdir(parents=True, exist_ok=True)

        speaker_segments = 0
        speaker_silent = 0
        speaker_files_ok = 0
        seg_index = 0

        for mp3_path in mp3_paths:
            try:
                audio = load_and_resample(mp3_path, target_sr)
            except Exception as e:
                logger.warning(f"Không đọc được file {mp3_path}: {e}")
                continue

            speaker_files_ok += 1
            segments = cut_segments(audio, target_sr, seg_len)

            for seg in segments:
                if is_silent(seg, silence_threshold):
                    speaker_silent += 1
                    logger.warning(
                        f"Segment im lặng bị loại bỏ: {speaker_id} segment {seg_index}"
                    )
                    continue

                out_path = speaker_output / f"{speaker_id}_{seg_index:05d}.wav"
                sf.write(str(out_path), seg, target_sr, subtype="FLOAT")
                seg_index += 1
                speaker_segments += 1

        stats["files_processed"] += speaker_files_ok
        stats["total_segments"] += speaker_segments
        stats["silent_discarded"] += speaker_silent
        stats["speakers"][speaker_id] = {
            "files_processed": speaker_files_ok,
            "segments_created": speaker_segments,
            "silent_discarded": speaker_silent,
        }

        logger.info(
            f"Speaker {speaker_id}: {speaker_files_ok} files → "
            f"{speaker_segments} segments ({speaker_silent} im lặng bị loại)"
        )

    # In thống kê tổng hợp
    logger.info("=" * 60)
    logger.info("THỐNG KÊ TỔNG HỢP")
    logger.info(f"  Số file xử lý:              {stats['files_processed']}")
    logger.info(f"  Số segment tạo được:         {stats['total_segments']}")
    logger.info(f"  Số segment loại bỏ (im lặng): {stats['silent_discarded']}")
    logger.info("=" * 60)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chuẩn hóa audio nguồn: resample 16kHz mono, cắt 4s segments, lọc im lặng."
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="/home/tuanlha/research/video_analysis/no_sound_effect",
        help="Đường dẫn tới thư mục data_source/ (mặc định: data_source)",
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
        "--silence-threshold",
        type=float,
        default=DEFAULT_SILENCE_THRESHOLD,
        help=f"Ngưỡng RMS im lặng (mặc định: {DEFAULT_SILENCE_THRESHOLD})",
    )

    args = parser.parse_args()

    normalize_all(
        data_source_dir=args.data_source,
        output_dir=args.output_dir,
        target_sr=args.sr,
        seg_len=args.seg_len,
        silence_threshold=args.silence_threshold,
    )
