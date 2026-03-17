"""
03_create_mixtures.py — Tạo hỗn hợp 2 người nói (WSJ0-2mix style) với SNR ngẫu nhiên.

Pipeline: liệt kê segments → ghép cặp 1-1 (mỗi audio chỉ dùng 1 lần) → trộn với SNR → lưu WAV.
Output: data_2mix/{tr,cv,tt}/{mix,s1,s2}/{index:06d}.wav
"""

import argparse
import logging
import random
import time
from collections import Counter
from pathlib import Path
from multiprocessing import Pool, cpu_count

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
# Giới hạn scale factor để không thay đổi biên độ s2 quá nhiều
SCALE_MIN = 0.5
SCALE_MAX = 2.0


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


def compute_snr_scale(
    s1: np.ndarray,
    s2: np.ndarray,
    target_snr_db: float,
    scale_min: float = SCALE_MIN,
    scale_max: float = SCALE_MAX,
) -> float:
    """Tính hệ số scale cho s2 để đạt target SNR so với s1, có giới hạn clamp.

    Công thức: scale = RMS(s1) / (RMS(s2) * 10^(SNR_db / 20))
    Sau đó clamp vào [scale_min, scale_max] để không thay đổi biên độ s2 quá nhiều.

    Args:
        s1: Signal 1 (tham chiếu), numpy array float32.
        s2: Signal 2 (cần điều chỉnh), numpy array float32.
        target_snr_db: SNR mục tiêu tính bằng dB.
        scale_min: Giới hạn dưới cho scale factor (mặc định 0.5).
        scale_max: Giới hạn trên cho scale factor (mặc định 2.0).

    Returns:
        Hệ số scale cho s2, đã clamp trong [scale_min, scale_max].
    """
    rms_s1 = np.sqrt(np.mean(s1 ** 2))
    rms_s2 = np.sqrt(np.mean(s2 ** 2))

    if rms_s2 < 1e-10:
        logger.warning("RMS của s2 gần bằng 0, trả về scale = 1.0")
        return 1.0

    scale = rms_s1 / (rms_s2 * (10.0 ** (target_snr_db / 20.0)))
    scale = float(np.clip(scale, scale_min, scale_max))
    return scale


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


def build_pairs(speaker_segments: dict[str, list[Path]], seed: int = 42) -> list[tuple[Path, Path]]:
    """Ghép cặp audio 1-1 theo logic tuần tự, mỗi audio chỉ dùng 1 lần.

    Duyệt từ speaker đầu tiên, lấy từng audio ghép với 1 audio của speaker khác
    (còn trong danh sách chờ). Speaker khác được chọn theo xác suất tỉ lệ với
    số audio còn lại — ví dụ B có 10 audio, C có 40 audio thì P(B)=20%, P(C)=80%.

    Args:
        speaker_segments: Dict {speaker_id: [wav_paths]}.
        seed: Random seed để shuffle thứ tự audio trong mỗi speaker.

    Returns:
        List of (path_s1, path_s2) pairs.
    """
    rng = random.Random(seed)

    # Tạo pool: mỗi speaker có 1 list audio đã shuffle
    pool: dict[str, list[Path]] = {}
    for spk_id, paths in speaker_segments.items():
        shuffled = list(paths)
        rng.shuffle(shuffled)
        pool[spk_id] = shuffled

    # Thứ tự duyệt speakers
    speaker_order = sorted(pool.keys())

    pairs: list[tuple[Path, Path]] = []

    for current_spk in speaker_order:
        while pool[current_spk]:
            # Tìm speaker khác còn audio
            other_spks = [s for s in speaker_order if s != current_spk and pool[s]]
            if not other_spks:
                break

            # Chọn speaker khác theo xác suất tỉ lệ số audio còn lại
            weights = [len(pool[s]) for s in other_spks]
            other_spk = rng.choices(other_spks, weights=weights, k=1)[0]

            s1_path = pool[current_spk].pop(0)
            s2_idx = rng.randrange(len(pool[other_spk]))
            s2_path = pool[other_spk].pop(s2_idx)
            pairs.append((s1_path, s2_path))

    return pairs


def _worker_one_mixture(args):
    """Hàm worker chạy trong process con, tạo 1 mixture và lưu 3 file.

    Args:
        args: tuple (index, s1_path, s2_path, snr_db, output_dir)
    """
    (i, s1_path, s2_path, snr_db, output_dir) = args

    s1_audio = _read_wav_safe(Path(s1_path))
    if s1_audio is None:
        return None

    s2_audio = _read_wav_safe(Path(s2_path))
    if s2_audio is None:
        return None

    # Đảm bảo cùng độ dài (cắt theo min length)
    min_len = min(len(s1_audio), len(s2_audio))
    s1_audio = s1_audio[:min_len]
    s2_audio = s2_audio[:min_len]

    # Tạo mixture
    mix, s1_out, s2_out = create_mixture(s1_audio, s2_audio, snr_db)

    # Lưu file WAV
    out_path = Path(output_dir)
    fname = f"{i:06d}.wav"
    try:
        sf.write(str(out_path / "mix" / fname), mix, SAMPLE_RATE, subtype="FLOAT")
        sf.write(str(out_path / "s1" / fname), s1_out, SAMPLE_RATE, subtype="FLOAT")
        sf.write(str(out_path / "s2" / fname), s2_out, SAMPLE_RATE, subtype="FLOAT")
    except OSError as e:
        logger.error(f"Worker {i}: lỗi ghi file: {e}")
        return None

    return snr_db


def generate_mixtures(
    split_dir: str,
    output_dir: str,
    snr_choices: list[float] | None = None,
    seed: int = 42,
    num_workers: int = 24,
) -> dict:
    """Tạo mixtures bằng cách ghép cặp 1-1 (mỗi audio chỉ dùng 1 lần).

    Số lượng mixtures = số cặp ghép được (phụ thuộc vào dữ liệu).
    """
    if snr_choices is None:
        snr_choices = DEFAULT_SNR_CHOICES

    # Tạo thư mục output
    out_path = Path(output_dir)
    for sub in ("mix", "s1", "s2"):
        (out_path / sub).mkdir(parents=True, exist_ok=True)

    # Liệt kê segments
    speaker_segments = list_segments(split_dir)
    speaker_ids = list(speaker_segments.keys())
    if len(speaker_ids) < 2:
        logger.warning(
            f"Cần ít nhất 2 speakers để tạo mixture, hiện có: {len(speaker_ids)}"
        )
        return {
            "mixtures_created": 0,
            "snr_distribution": {},
            "elapsed_seconds": 0.0,
        }

    total_segs = sum(len(v) for v in speaker_segments.values())
    logger.info(f"Tổng segments: {total_segs} từ {len(speaker_ids)} speakers")

    # Ghép cặp 1-1
    pairs = build_pairs(speaker_segments, seed=seed)
    num_mixtures = len(pairs)
    logger.info(f"Số cặp ghép được: {num_mixtures}")

    if num_mixtures == 0:
        return {
            "mixtures_created": 0,
            "snr_distribution": {},
            "elapsed_seconds": 0.0,
        }

    # Skip nếu đã có đủ file
    mix_dir = out_path / "mix"
    if mix_dir.exists():
        existing = len(list(mix_dir.glob("*.wav")))
        if existing >= num_mixtures:
            logger.info(
                f"SKIP: {output_dir} đã có {existing} mixtures (>= {num_mixtures})"
            )
            return {
                "mixtures_created": existing,
                "snr_distribution": {},
                "elapsed_seconds": 0.0,
                "skipped": True,
            }

    start_time = time.time()
    snr_counter: Counter = Counter()
    rng = random.Random(seed)

    # Chuẩn bị tasks: mỗi pair gán 1 SNR ngẫu nhiên
    tasks = []
    for i, (s1_path, s2_path) in enumerate(pairs):
        snr_db = rng.choice(snr_choices)
        tasks.append((i, str(s1_path), str(s2_path), snr_db, output_dir))

    # Sử dụng Pool
    n_proc = min(num_workers, cpu_count())
    logger.info(f"Dùng {n_proc} process để tạo {num_mixtures} mixtures")

    created = 0
    with Pool(processes=n_proc) as pool:
        for idx, snr_db in enumerate(pool.imap_unordered(_worker_one_mixture, tasks)):
            if snr_db is None:
                continue
            snr_counter[snr_db] += 1
            created += 1

            # Log tiến trình mỗi 10%
            if num_mixtures >= 10 and (idx + 1) % (num_mixtures // 10) == 0:
                logger.info(
                    f"  Tiến trình (song song): {idx + 1}/{num_mixtures} "
                    f"({100 * (idx + 1) // num_mixtures}%)"
                )

    elapsed = time.time() - start_time
    snr_dist = {str(k): v for k, v in sorted(snr_counter.items())}
    stats = {
        "mixtures_created": created,
        "snr_distribution": snr_dist,
        "elapsed_seconds": round(elapsed, 2),
    }

    logger.info(f"Hoàn tất (song song): {created}/{num_mixtures} mixtures")
    logger.info(f"Phân bố SNR: {snr_dist}")
    logger.info(f"Thời gian: {elapsed:.2f}s")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tạo hỗn hợp 2 người nói — ghép cặp 1-1, mỗi audio chỉ dùng 1 lần."
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
    parser.add_argument(
        "--workers",
        type=int,
        default=24,
        help="Số process song song (mặc định: 24)",
    )

    args = parser.parse_args()

    splits = ["tr", "cv", "tt"]

    logger.info("=" * 60)
    logger.info("BẮT ĐẦU TẠO HỖN HỢP 2 NGƯỜI NÓI")
    logger.info("=" * 60)

    all_stats = {}
    for split_name in splits:
        split_dir = f"segments_{split_name}"
        out_dir = str(Path(args.output_dir) / split_name)
        logger.info(
            f"\n--- {split_name.upper()} ({split_dir} → {out_dir}) ---"
        )
        stats = generate_mixtures(
            split_dir=split_dir,
            output_dir=out_dir,
            seed=args.seed,
            num_workers=args.workers,
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