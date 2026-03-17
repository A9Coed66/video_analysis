"""
01_normalize.py — Chuẩn hóa audio nguồn từ data_source/ thành segment 6s WAV 16kHz mono.

Pipeline: quét MP3 → resample 16kHz mono → Silero VAD detect speech →
          mỗi speech segment + random silent padding đầu/cuối → cắt cứng 6s → lưu WAV.

Output: segments/{speaker_id}/{speaker_id}_{index:05d}.wav
"""

import argparse
import logging
import multiprocessing as mp
import random
from functools import partial
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
DEFAULT_SEG_LEN = 4.0  # seconds
DEFAULT_MAX_PAD = 0.5  # max random silence padding (seconds)
SAMPLES_PER_SEGMENT = int(DEFAULT_SR * DEFAULT_SEG_LEN)


# ---------------------------------------------------------------------------
# Silero VAD (loaded once, reused across files)
# ---------------------------------------------------------------------------
_vad_model = None



def get_vad_model():
    """Load Silero VAD model (singleton).

    Lần đầu (main process) sẽ download từ GitHub.
    Các worker processes dùng source='local' từ cache để tránh race condition.
    """
    global _vad_model
    if _vad_model is None:
        # Kiểm tra cache đã tồn tại chưa
        cache_dir = Path(torch.hub.get_dir()) / "snakers4_silero-vad_master"
        if cache_dir.exists():
            _vad_model, _utils = torch.hub.load(
                repo_or_dir=str(cache_dir),
                model="silero_vad",
                source="local",
                trust_repo=True,
            )
        else:
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


# ---------------------------------------------------------------------------
# Worker function for multiprocessing
# ---------------------------------------------------------------------------

def _process_single_file(
    task: tuple,
    target_sr: int = DEFAULT_SR,
    seg_len: float = DEFAULT_SEG_LEN,
    max_pad: float = DEFAULT_MAX_PAD,
) -> dict:
    """Worker: xử lý 1 file MP3 → trả về dict kết quả (không ghi file).

    Args:
        task: (mp3_path, speaker_id, speaker_output_dir, seg_start_index)

    Returns:
        dict với segments data và stats cho file này.
    """
    mp3_path, speaker_id, speaker_output_dir, seg_start_index = task

    result = {
        "speaker_id": speaker_id,
        "ok": False,
        "segments_written": 0,
        "speech_regions": 0,
        "error": None,
        "skipped": False,
    }

    # Skip nếu đã có output WAV cho file này (resume sau crash)
    speaker_output = Path(speaker_output_dir)
    first_seg_path = speaker_output / f"{speaker_id}_{seg_start_index:05d}.wav"
    if first_seg_path.exists():
        # Đếm số segment đã tạo cho file này
        existing = 0
        while (speaker_output / f"{speaker_id}_{seg_start_index + existing:05d}.wav").exists():
            existing += 1
        result["ok"] = True
        result["segments_written"] = existing
        result["skipped"] = True
        return result

    try:
        audio = load_and_resample(Path(mp3_path), target_sr)
    except Exception as e:
        result["error"] = str(e)
        return result

    speech_timestamps = detect_speech_segments(audio, target_sr)
    result["speech_regions"] = len(speech_timestamps)

    if not speech_timestamps:
        result["ok"] = True
        return result

    seg_index = seg_start_index
    speaker_output = Path(speaker_output_dir)

    for ts in speech_timestamps:
        start_sample = int(ts["start"] * target_sr)
        end_sample = int(ts["end"] * target_sr)
        speech_chunk = audio[start_sample:end_sample]

        cut_segments = pad_and_cut(speech_chunk, target_sr, seg_len, max_pad)

        for seg in cut_segments:
            out_path = speaker_output / f"{speaker_id}_{seg_index:05d}.wav"
            sf.write(str(out_path), seg, target_sr, subtype="FLOAT")
            seg_index += 1

    result["ok"] = True
    result["segments_written"] = seg_index - seg_start_index
    return result


DEFAULT_NUM_WORKERS = 16


def normalize_all(
    data_source_dir: str,
    output_dir: str = "segments",
    target_sr: int = DEFAULT_SR,
    seg_len: float = DEFAULT_SEG_LEN,
    max_pad: float = DEFAULT_MAX_PAD,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> dict:
    """Pipeline chính (parallel): quét → phân phối file cho workers → tổng hợp stats.

    Args:
        data_source_dir: Đường dẫn tới thư mục data_source/.
        output_dir: Thư mục output cho segments.
        target_sr: Sample rate mục tiêu (mặc định 16000).
        seg_len: Độ dài mỗi segment tính bằng giây (mặc định 4.0).
        max_pad: Max random silence padding đầu/cuối mỗi speech segment (mặc định 0.5s).
        num_workers: Số process song song (mặc định 16).

    Returns:
        dict với thống kê.
    """
    speaker_files = scan_mp3_files(data_source_dir)

    # Chuẩn bị output dirs và build task list
    # Mỗi task = (mp3_path_str, speaker_id, speaker_output_dir_str, seg_start_index)
    # Để tránh trùng filename, ta pre-assign seg_start_index cho mỗi file.
    # Vì chưa biết trước mỗi file tạo bao nhiêu segment, ta dùng khoảng cách lớn.
    SEG_INDEX_GAP = 100_000  # mỗi file được cấp tối đa 100k segment indices

    all_tasks: list[tuple] = []
    speaker_file_counts: dict[str, int] = {}

    for speaker_id, mp3_paths in speaker_files.items():
        speaker_output = Path(output_dir) / speaker_id
        speaker_output.mkdir(parents=True, exist_ok=True)
        speaker_file_counts[speaker_id] = len(mp3_paths)

        for file_idx, mp3_path in enumerate(mp3_paths):
            seg_start = file_idx * SEG_INDEX_GAP
            all_tasks.append((
                str(mp3_path),
                speaker_id,
                str(speaker_output),
                seg_start,
            ))

    total_files = len(all_tasks)
    max_cpus = mp.cpu_count() or 1
    actual_workers = min(num_workers, total_files, max_cpus)
    logger.info(
        f"Bắt đầu xử lý {total_files} files với {actual_workers} workers"
    )

    # Pre-download Silero VAD model trong main process trước khi fork workers
    # để tránh race condition khi nhiều worker cùng download master.zip
    logger.info("Pre-loading Silero VAD model trước khi fork workers...")
    get_vad_model()

    # Tạo worker function với các tham số cố định
    worker_fn = partial(
        _process_single_file,
        target_sr=target_sr,
        seg_len=seg_len,
        max_pad=max_pad,
    )

    # Chạy parallel
    with mp.Pool(processes=actual_workers) as pool:
        results = pool.map(worker_fn, all_tasks)

    # Tổng hợp stats
    stats = {
        "files_processed": 0,
        "files_skipped": 0,
        "total_segments": 0,
        "speech_regions_found": 0,
        "speakers": {},
    }

    # Group results by speaker
    speaker_results: dict[str, list[dict]] = {}
    for r in results:
        sid = r["speaker_id"]
        speaker_results.setdefault(sid, []).append(r)

    for speaker_id, file_results in speaker_results.items():
        speaker_files_ok = sum(1 for r in file_results if r["ok"])
        speaker_skipped = sum(1 for r in file_results if r.get("skipped"))
        speaker_segments = sum(r["segments_written"] for r in file_results)
        speaker_speech_regions = sum(r["speech_regions"] for r in file_results)

        # Log errors
        for r in file_results:
            if r["error"]:
                logger.warning(f"Không đọc được file (speaker {speaker_id}): {r['error']}")

        stats["files_processed"] += speaker_files_ok
        stats["files_skipped"] += speaker_skipped
        stats["total_segments"] += speaker_segments
        stats["speech_regions_found"] += speaker_speech_regions
        stats["speakers"][speaker_id] = {
            "files_processed": speaker_files_ok,
            "files_skipped": speaker_skipped,
            "segments_created": speaker_segments,
            "speech_regions_found": speaker_speech_regions,
        }

        logger.info(
            f"Speaker {speaker_id}: {speaker_files_ok} files ({speaker_skipped} skipped) → "
            f"{speaker_speech_regions} speech regions → {speaker_segments} segments"
        )

    logger.info("=" * 60)
    logger.info("THỐNG KÊ TỔNG HỢP")
    logger.info(f"  Số workers:                  {actual_workers}")
    logger.info(f"  Số file xử lý:              {stats['files_processed']}")
    logger.info(f"  Số file đã skip (có sẵn):   {stats['files_skipped']}")
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
        default="/home/tuanlha/Self_project/video_analysis/processed_asmr_local",
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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"Số process song song (mặc định: {DEFAULT_NUM_WORKERS})",
    )

    args = parser.parse_args()

    normalize_all(
        data_source_dir=args.data_source,
        output_dir=args.output_dir,
        target_sr=args.sr,
        seg_len=args.seg_len,
        max_pad=args.max_pad,
        num_workers=args.num_workers,
    )
