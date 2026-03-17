"""
02_split_speakers.py — Chia segments thành train/val/test (80/10/10) random.

Đọc tất cả file .wav từ segments/, shuffle random, rồi chia đều vào
segments_tr/, segments_cv/, segments_tt/.
Mỗi speaker có thể xuất hiện ở nhiều tập.
"""

import logging
import random
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIRS = {
    "train": "segments_tr",
    "val": "segments_cv",
    "test": "segments_tt",
}


def collect_all_segments(segments_dir: str) -> list[tuple[str, Path]]:
    """Thu thập tất cả file .wav từ segments/<speaker_id>/*.wav.

    Returns:
        List of (speaker_id, wav_path) tuples.
    """
    seg_path = Path(segments_dir)
    if not seg_path.exists():
        raise FileNotFoundError(f"Thư mục segments không tồn tại: {segments_dir}")

    all_segments = []
    for speaker_dir in sorted(seg_path.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker_id = speaker_dir.name
        wavs = sorted(speaker_dir.glob("*.wav"))
        for wav in wavs:
            all_segments.append((speaker_id, wav))

    if not all_segments:
        raise FileNotFoundError(f"Không tìm thấy file .wav nào trong {segments_dir}")

    return all_segments


def split_segments_random(
    all_segments: list[tuple[str, Path]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[tuple[str, Path]]]:
    """Shuffle tất cả segments rồi chia theo tỷ lệ.

    Args:
        all_segments: List of (speaker_id, wav_path).
        train_ratio: Tỷ lệ train (mặc định 0.8).
        val_ratio: Tỷ lệ val (mặc định 0.1).
        seed: Random seed.

    Returns:
        Dict {'train': [...], 'val': [...], 'test': [...]}.
    """
    segments = list(all_segments)
    rng = random.Random(seed)
    rng.shuffle(segments)

    n = len(segments)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)

    return {
        "train": segments[:n_train],
        "val": segments[n_train:n_train + n_val],
        "test": segments[n_train + n_val:],
    }


def create_split_dirs(
    split_map: dict[str, list[tuple[str, Path]]],
    segments_dir: str = "segments",
    output_dirs: dict[str, str] | None = None,
) -> None:
    """Chia segment files vào thư mục split tương ứng.

    Chiến lược tối ưu:
      1. Copy val/test ra thư mục riêng (số lượng nhỏ ~20%)
      2. Xóa các file val/test khỏi segments/
      3. Rename segments/ → segments_tr/ (O(1), không copy)

    Cấu trúc output: segments_tr/<speaker_id>/<file.wav>
    """
    if output_dirs is None:
        output_dirs = DEFAULT_OUTPUT_DIRS

    seg_path = Path(segments_dir)

    # --- Bước 1: Copy val & test ra ngoài ---
    for split_name in ("val", "test"):
        segments = split_map[split_name]
        out_dir = Path(output_dirs[split_name])
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        speakers_in_split = set()
        for speaker_id, wav_path in segments:
            speaker_out = out_dir / speaker_id
            speaker_out.mkdir(exist_ok=True)
            shutil.copy2(str(wav_path), str(speaker_out / wav_path.name))
            speakers_in_split.add(speaker_id)

        label = {"val": "VAL", "test": "TEST"}[split_name]
        logger.info(f"--- {label} ({output_dirs[split_name]}) ---")
        logger.info(f"  Segments: {len(segments)}")
        logger.info(f"  Speakers: {len(speakers_in_split)} — {sorted(speakers_in_split)}")

    # --- Bước 2: Xóa các file val/test khỏi segments/ ---
    non_train_files = set()
    for split_name in ("val", "test"):
        for _, wav_path in split_map[split_name]:
            non_train_files.add(wav_path)

    for wav_path in non_train_files:
        wav_path.unlink(missing_ok=True)

    # Dọn thư mục speaker rỗng
    for speaker_dir in seg_path.iterdir():
        if speaker_dir.is_dir() and not any(speaker_dir.iterdir()):
            speaker_dir.rmdir()

    # --- Bước 3: Rename segments/ → segments_tr/ (instant) ---
    train_dir = Path(output_dirs["train"])
    if train_dir.exists():
        shutil.rmtree(train_dir)
    seg_path.rename(train_dir)

    train_speakers = set(s for s, _ in split_map["train"])
    logger.info(f"--- TRAIN ({output_dirs['train']}) ---")
    logger.info(f"  Segments: {len(split_map['train'])}")
    logger.info(f"  Speakers: {len(train_speakers)} — {sorted(train_speakers)}")


if __name__ == "__main__":
    segments_dir = "segments"

    logger.info("Bắt đầu chia segments random...")
    logger.info("=" * 60)

    all_segs = collect_all_segments(segments_dir)
    speakers = sorted(set(s for s, _ in all_segs))
    logger.info(f"Tổng: {len(all_segs)} segments từ {len(speakers)} speakers: {speakers}")

    split_map = split_segments_random(all_segs)
    create_split_dirs(split_map)

    logger.info("=" * 60)
    logger.info("Hoàn tất chia segments.")
