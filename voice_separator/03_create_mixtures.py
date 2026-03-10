"""
03_create_mixtures.py — Tạo hỗn hợp 2 người nói (WSJ0-2mix style) với SNR ngẫu nhiên.

Pipeline: liệt kê segments → chọn 2 speakers khác nhau → trộn với SNR → lưu WAV.
Output: data_2mix/{tr,cv,tt}/{mix,s1,s2}/{index:06d}.wav
"""

import argparse
import logging
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 16000
SAMPLES_PER_SEGMENT = 64000  # 4s at 16kHz
DEFAULT_SNR_CHOICES = [0, 5, 10]


def list_segments(split_dir: str) -> dict[str, list[Path]]:
    """Liệt kê segments theo speaker trong một split dir.

    Args:
        split_dir: Đường dẫn tới thư mục split (segments_tr/, segments_cv/, segments_tt/).

    Returns:
        Dict {speaker_id: [wav_paths]} đã sắp xếp.
    """
    split_path = Path(split_dir)
    if not split_path.exists():
        raise FileNotFoundError(f"Thư mục split không tồn tại: {split_dir}")

    speaker_segments: dict[str, list[Path]] = {}
    for speaker_dir in sorted(split_path.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker_id = speaker_dir.name
        wav_files = sorted(speaker_dir.glob("*.wav"))
        if wav_files:
            speaker_segments[speaker_id] = wav_files

    if not speaker_segments:
        logger.warning(f"Không tìm thấy segment nào trong {split_dir}")

    return speaker_segments


def compute_snr_scale(s1: np.ndarray, s2: np.ndarray, target_snr_db: float) -> float:
    """Tính hệ số scale cho s2 để đạt target SNR so với s1.

    Công thức: scale = RMS(s1) / (RMS(s2) * 10^(SNR_db / 20))

    Args:
        s1: Signal 1 (tham chiếu), numpy array float32.
        s2: Signal 2 (cần điều chỉnh), numpy array float32.
        target_snr_db: SNR mục tiêu tính bằng dB.

    Returns:
        Hệ số scale cho s2.
    """
    rms_s1 = np.sqrt(np.mean(s1 ** 2))
    rms_s2 = np.sqrt(np.mean(s2 ** 2))

    if rms_s2 < 1e-10:
        logger.warning("RMS của s2 gần bằng 0, trả về scale = 1.0")
        return 1.0

    scale = rms_s1 / (rms_s2 * (10.0 ** (target_snr_db / 20.0)))
    return float(scale)


def create_mixture(
    s1: np.ndarray, s2: np.ndarray, snr_db: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trộn 2 segment: trả về (mix, s1_original, s2_scaled).

    mix = s1 + s2_scaled, trong đó s2_scaled = s2 * scale_factor.

    Args:
        s1: Segment speaker 1 (tham chiếu).
        s2: Segment speaker 2 (sẽ được scale).
        snr_db: SNR mục tiêu tính bằng dB.

    Returns:
        Tuple (mix, s1, s2_scaled) dạng numpy arrays float32.
    """
    scale = compute_snr_scale(s1, s2, snr_db)
    s2_scaled = s2 * scale
    mix = s1 + s2_scaled
    return mix.astype(np.float32), s1.astype(np.float32), s2_scaled.astype(np.float32)


def _read_wav_safe(wav_path: Path) -> np.ndarray | None:
    """Đọc file WAV an toàn, trả về None nếu file hỏng.

    Args:
        wav_path: Đường dẫn tới file WAV.

    Returns:
        numpy array float32 hoặc None nếu đọc thất bại.
    """
    try:
        audio, _sr = sf.read(str(wav_path), dtype="float32")
        return audio
    except Exception as e:
        logger.warning(f"Không đọc được file WAV {wav_path}: {e}")
        return None


def generate_mixtures(
    split_dir: str,
    output_dir: str,
    num_mixtures: int,
    snr_choices: list[float] | None = None,
    seed: int = 42,
    max_offset_ratio: float = 0.25,
) -> dict:
    """Tạo num_mixtures hỗn hợp, lưu vào output_dir/{mix,s1,s2}/.

    Mỗi hỗn hợp chọn ngẫu nhiên 2 speakers khác nhau, 1 segment mỗi speaker,
    và SNR ngẫu nhiên từ snr_choices.

    Args:
        split_dir: Đường dẫn tới thư mục split chứa segments.
        output_dir: Đường dẫn output (vd: data_2mix/tr).
        num_mixtures: Số hỗn hợp cần tạo.
        snr_choices: Danh sách SNR (dB) để chọn ngẫu nhiên. Mặc định [0, 5, 10].
        seed: Random seed (mặc định 42).

    Returns:
        Dict thống kê: mixtures_created, snr_distribution, elapsed_seconds.
    """
    if snr_choices is None:
        snr_choices = DEFAULT_SNR_CHOICES

    # Skip nếu đã có đủ file
    out_path = Path(output_dir)
    mix_dir = out_path / "mix"
    if mix_dir.exists():
        existing = len(list(mix_dir.glob("*.wav")))
        if existing >= num_mixtures:
            logger.info(
                f"SKIP: {output_dir} đã có {existing} mixtures (>= {num_mixtures})"
            )
            return {"mixtures_created": existing, "snr_distribution": {}, "elapsed_seconds": 0.0, "skipped": True}

    rng = random.Random(seed)

    # Liệt kê segments
    speaker_segments = list_segments(split_dir)
    speaker_ids = list(speaker_segments.keys())

    if len(speaker_ids) < 2:
        logger.warning(
            f"Cần ít nhất 2 speakers để tạo mixture, hiện có: {len(speaker_ids)}"
        )
        return {"mixtures_created": 0, "snr_distribution": {}, "elapsed_seconds": 0.0}

    total_segs = sum(len(v) for v in speaker_segments.values())
    if total_segs < 2:
        logger.warning(f"Không đủ segment để tạo mixture (tổng: {total_segs})")
        return {"mixtures_created": 0, "snr_distribution": {}, "elapsed_seconds": 0.0}

    # Tạo thư mục output
    out_path = Path(output_dir)
    for sub in ("mix", "s1", "s2"):
        (out_path / sub).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    snr_counter: Counter = Counter()
    created = 0
    max_retries = 10  # Số lần thử lại tối đa khi gặp file hỏng

    for i in range(num_mixtures):
        # Chọn 2 speakers khác nhau
        spk1, spk2 = rng.sample(speaker_ids, 2)

        # Chọn segment ngẫu nhiên từ mỗi speaker, retry nếu file hỏng
        s1_audio = None
        for _ in range(max_retries):
            seg1_path = rng.choice(speaker_segments[spk1])
            s1_audio = _read_wav_safe(seg1_path)
            if s1_audio is not None:
                break
        if s1_audio is None:
            logger.warning(f"Mixture {i}: Không đọc được segment từ {spk1}, bỏ qua")
            continue

        s2_audio = None
        for _ in range(max_retries):
            seg2_path = rng.choice(speaker_segments[spk2])
            s2_audio = _read_wav_safe(seg2_path)
            if s2_audio is not None:
                break
        if s2_audio is None:
            logger.warning(f"Mixture {i}: Không đọc được segment từ {spk2}, bỏ qua")
            continue

        # Đảm bảo cùng độ dài (cắt theo min length)
        min_len = min(len(s1_audio), len(s2_audio))
        s1_audio = s1_audio[:min_len]
        s2_audio = s2_audio[:min_len]

        # Chọn SNR ngẫu nhiên
        snr_db = rng.choice(snr_choices)

        # Tạo mixture
        mix, s1_out, s2_out = create_mixture(s1_audio, s2_audio, snr_db)

        # Lưu file WAV
        fname = f"{created:06d}.wav"
        try:
            sf.write(str(out_path / "mix" / fname), mix, SAMPLE_RATE, subtype="FLOAT")
            sf.write(str(out_path / "s1" / fname), s1_out, SAMPLE_RATE, subtype="FLOAT")
            sf.write(str(out_path / "s2" / fname), s2_out, SAMPLE_RATE, subtype="FLOAT")
        except OSError as e:
            logger.error(
                f"Lỗi ghi file (có thể đĩa đầy): {e}\n"
                f"Đã tạo {created}/{num_mixtures} mixtures trước khi lỗi.\n"
                f"Kiểm tra dung lượng đĩa và thử lại."
            )
            break

        snr_counter[snr_db] += 1
        created += 1

        # Log tiến trình mỗi 10%
        if num_mixtures >= 10 and (i + 1) % (num_mixtures // 10) == 0:
            logger.info(f"  Tiến trình: {i + 1}/{num_mixtures} ({100 * (i + 1) // num_mixtures}%)")

    elapsed = time.time() - start_time

    # Thống kê
    snr_dist = {str(k): v for k, v in sorted(snr_counter.items())}
    stats = {
        "mixtures_created": created,
        "snr_distribution": snr_dist,
        "elapsed_seconds": round(elapsed, 2),
    }

    logger.info(f"Hoàn tất: {created}/{num_mixtures} mixtures")
    logger.info(f"Phân bố SNR: {snr_dist}")
    logger.info(f"Thời gian: {elapsed:.2f}s")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tạo hỗn hợp 2 người nói (WSJ0-2mix style) với SNR ngẫu nhiên."
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=20000,
        help="Số hỗn hợp cho tập train (mặc định: 20000)",
    )
    parser.add_argument(
        "--num-val",
        type=int,
        default=1000,
        help="Số hỗn hợp cho tập validation (mặc định: 1000)",
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=1000,
        help="Số hỗn hợp cho tập test (mặc định: 1000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_2mix",
        help="Thư mục output (mặc định: data_2mix)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (mặc định: 42)",
    )

    args = parser.parse_args()

    splits = {
        "tr": ("segments_tr", args.num_train),
        "cv": ("segments_cv", args.num_val),
        "tt": ("segments_tt", args.num_test),
    }

    logger.info("=" * 60)
    logger.info("BẮT ĐẦU TẠO HỖN HỢP 2 NGƯỜI NÓI")
    logger.info("=" * 60)

    all_stats = {}
    for split_name, (split_dir, num_mix) in splits.items():
        logger.info(f"\n--- {split_name.upper()} ({split_dir} → {args.output_dir}/{split_name}) ---")
        out_dir = str(Path(args.output_dir) / split_name)
        stats = generate_mixtures(
            split_dir=split_dir,
            output_dir=out_dir,
            num_mixtures=num_mix,
            seed=args.seed,
        )
        all_stats[split_name] = stats

    logger.info("\n" + "=" * 60)
    logger.info("THỐNG KÊ TỔNG HỢP")
    for split_name, stats in all_stats.items():
        logger.info(
            f"  {split_name.upper()}: {stats['mixtures_created']} mixtures, "
            f"SNR: {stats['snr_distribution']}, "
            f"Thời gian: {stats['elapsed_seconds']}s"
        )
    logger.info("=" * 60)
